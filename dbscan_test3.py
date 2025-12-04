import torch
import numpy as np
import os

from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import poisson_surface_reconstruction
from gaussian_renderer import generate_neural_gaussians_SDF
from collections import defaultdict
from itertools import combinations
import json
from sklearn.cluster import DBSCAN, Birch, OPTICS
from sklearn.neighbors import KDTree, NearestNeighbors
import scipy.special as sp
from scipy.spatial.distance import jensenshannon, pdist, cosine
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse import csr_matrix
import open3d as o3d




def get_classes():
    classes= []

    with open("info_semantic.json", 'r') as classes_file:
        data = json.load(classes_file)

    for objects in data['classes']:
        classes.append(objects['name'])

    return classes

def setup_gaussian_scene_and_model(dataset : ModelParams, iteration : int, checkpoint: str):
    classes = get_classes()

    with torch.no_grad():
        dataset.eval = True
        gaussianModel = GaussianModel(
            classes,
            dataset.feat_dim, 
            dataset.n_offsets, 
            dataset.voxel_size, 
            dataset.update_depth, 
            dataset.update_init_factor, 
            dataset.update_hierachy_factor, 
            dataset.use_feat_bank
            )
        scene = Scene(dataset, gaussianModel, iteration, shuffle=False)
        if(checkpoint):
            ckpt = torch.load(checkpoint, map_location="cuda")
            if isinstance(ckpt, tuple) and isinstance(ckpt[0], tuple):
                capture = ckpt[0]
            else:
                raise RuntimeError("Unexpected checkpoint format; expected (capture, iteration)")

            # directly grab sem logits by index (based on your printout) ---
            sem_logits = capture[7]

            # make sure anchors count matches the scene’s anchors
            N_ckpt = sem_logits.shape[0]
            N_scene = gaussianModel._anchor.shape[0]
            if N_ckpt != N_scene:
                print(f"[extract_mesh] Anchor count mismatch: ckpt={N_ckpt} vs scene={N_scene}. "
                    f"Build Scene with the SAME iteration as the checkpoint or do a full restore.")
                
            gaussianModel._sem_logits = torch.nn.Parameter(
                sem_logits.to(gaussianModel._anchor.device), requires_grad=False
            )

        gaussianModel.eval() # setup values for trained gaussian model

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        anchors = gaussianModel.get_anchor.detach().cpu().numpy()

    return gaussianModel, scene, anchors, sem_logits

def view_projection(anchors, anchors_id, view): #, gaussianModel, pipeline, background):
    # render_scene = render(view, gaussianModel, pipeline, background, visible_mask=None, learn_SDF=False)
    # visible_anchor = render_scene["visible_anchor"].cpu().detach().numpy()
    camera = view

    
    view_matrix = camera.world_view_transform.cpu().numpy()
    view_matrix = view_matrix.T
    proj_matrix = camera.projection_matrix.cpu().numpy()
    proj_matrix = proj_matrix.T
    full_proj_transform = proj_matrix @ view_matrix

    points_h = np.concatenate([anchors, np.ones((anchors.shape[0], 1))], axis=1)
    clip_coords = (full_proj_transform @ points_h.T).T
    w = clip_coords[:, 3]
    ndc = clip_coords[:, :3] / clip_coords[:, 3:4]
    W, H = camera.image_width, camera.image_height
    u = np.round((ndc[:, 0] + 1) * 0.5 * W).astype(int)
    v = np.round((1 + ndc[:, 1]) * 0.5 * H).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)

    # Keep only points within image bounds and in front of the camera
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (clip_coords[:, 3] > 0)
    

    u_valid = u[valid]
    v_valid = v[valid]
    visible_ids = anchors_id[valid]

    # projection_info = dict(zip(visible_ids, zip(v_valid, u_valid)))
    mask[v_valid, u_valid] = 255

    # projected_anchors = {}
    # projected_anchors["points"] = [v_valid, u_valid]
    # projected_anchors["global_id"] = visible_ids

    # print("valid = ", valid.shape[0])

    # plt.imshow(mask, cmap='gray')
    # plt.title("Visible Anchor Projection")
    # plt.axis('off')
    # plt.show()

    return visible_ids, v_valid, u_valid


def get_original_image(camera):
    org_image_np = camera.original_image.permute(1,2,0).cpu().numpy()
    return (org_image_np*255).clip(0, 255).astype(np.uint8)



def get_views(scene, skip_train, skip_test):
    scene_cameras_train = scene.getTrainCameras() if not skip_train else []
    scene_cameras_test = scene.getTestCameras() if not skip_test else []

    views = scene_cameras_train + scene_cameras_test
    return views

def build_covisibility(visible_ids, covis):
    
    for a, b in combinations(visible_ids, 2):
        i, j = (a, b) if a < b else (b, a)
        covis[(int(i), int(j))] += 1

def build_covisibility_matrix(visible_ids, covis):
    
    for a, b in combinations(visible_ids, 2):
        i, j = (a, b) if a < b else (b, a)
        covis[int(i), int(j)] += 1

import colorsys
def palette_from_labels(labels, s=0.65, v=0.95):
    l = len(labels)
    n = max(1, l)
    table = np.zeros((l, 3), dtype=np.float32)
    for i in range(l):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        table[i] = (r, g, b)
    return table

def contrast_palette(labels, s=0.85, v=0.98, noise_gray=0.00):
    """
    Drop-in replacement: returns an (L,3) RGB table in [0,1],
    with well-separated hues. If -1 is among `labels`, the first
    entry (index 0 after np.unique) is set to gray for noise.
    """
    l = len(labels)
    n = max(1, l)
    table = np.zeros((l, 3), dtype=np.float32)

    phi = 0.6180339887498949  # golden ratio conjugate
    h0 = 0.13                 # start hue (tweak if you want)

    for i in range(l):
        h = (h0 + i * phi) % 1.0
        # tiny value jitter every few colors to boost contrast
        vv = v if (i % 3) else min(1.0, v * 0.92)
        r, g, b = colorsys.hsv_to_rgb(h, s, vv)
        table[i] = (r, g, b)

    # If labels contain -1, np.unique(labels) puts it at index 0.
    # Color that entry gray so noise is subdued.
    # if np.any(np.asarray(labels) == -1):
    #     table[0] = (noise_gray, noise_gray, noise_gray)

    return table

def generate_sam_data_for_anchors(anchor_points, all_views, files_path, anchor_ids = None):
    N = anchor_points.shape[0]
    V = len(all_views)

    if anchor_ids is None:
        anchor_ids = np.arange(anchor_points.shape[0])

    projection_data = np.full((N, V), -1, dtype = np.int32)

    for v_id, view in enumerate(all_views):
        visible_ids, v, u = view_projection(anchor_points, anchor_ids, view)

        if visible_ids.size == 0:
            continue

        image_name = view.image_name
        base = os.path.splitext(image_name)[0]
        npz_path = os.path.join(files_path, f"{base}.npz")

        if os.path.isfile(npz_path):
            npz = np.load(npz_path)
            masks = npz["masks"].astype(bool)

            M, H, W = masks.shape

            local = np.full((H,W), -1, dtype=np.int32)

            for mask_id in range(M):
                local[masks[mask_id]] = mask_id
            
            local_mask_id = local[v,u]
            positive = local_mask_id != -1
            projection_data[visible_ids[positive], v_id] = local_mask_id[positive]

    return projection_data

def weighted_logit_mean(
    anchors_xyz,
    logits,
    k=50,
    self_weight=1.0,
    use_entropy=True,
    alpha=0.5):

    N, K = logits.shape

    idx_sorted = np.argsort(logits, axis=1)
    top1 = logits[np.arange(N), idx_sorted[:, -1]]
    top2 = logits[np.arange(N), idx_sorted[:, -2]]
    margin = top1 - top2
    conf_margin = 1.0 / (1.0 + np.exp(-margin / 2.0))

    if use_entropy:
        z = logits - logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        sum_ez = ez.sum(axis=1, keepdims=True)
        P = ez / np.clip(sum_ez, 1e-12, None)
        # H = -sum p*log p
        H = -(P * np.log(np.clip(P, 1e-12, 1.0))).sum(axis=1)
        Hmax = np.log(K)
        conf_entropy = 1.0 - H / (Hmax + 1e-12)
        
        conf = alpha * conf_entropy + (1.0 - alpha) * conf_margin
    else:
        conf = conf_margin

    nn = NearestNeighbors(n_neighbors=min(k, N)).fit(anchors_xyz)
    dists, nbr_idx = nn.kneighbors(anchors_xyz, return_distance=True)

    nz = dists[dists > 0]
    sigma_s = (np.median(nz) if nz.size else 1.0) + 1e-9

    W = np.exp(-(dists**2) / (2.0 * sigma_s**2))
    W *= conf[nbr_idx]   # neighbor's confidence

    w_self = self_weight * conf
    denom = np.maximum(w_self + W.sum(axis=1), 1e-12)

    z_neighbors = (W[..., None] * logits[nbr_idx]).sum(axis=1)
    z_self = (w_self[:, None] * logits)

    z_out = (z_self + z_neighbors) / denom[:, None]
    return z_out

def row_softmax(Z):
    Z = Z.astype(np.float64, copy=True)
    Z -= Z.max(axis=1, keepdims=True)
    np.exp(Z, out=Z)
    Z /= Z.sum(axis=1, keepdims=True)
    return Z

if __name__ == "__main__":
    # Set up command line argument parser with default parameters
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_train", default=False)
    parser.add_argument("--skip_test", default=False) # from True
    parser.add_argument("--checkpoint_path")
    args = get_combined_args(parser)

    files_path = "./data/replica/scan1/masks_real2/"


    # Initialize system state (RNG) -- what is that ????
    safe_state(args.quiet)

    gaussianModel, scene, anchor_points, sem_logits = setup_gaussian_scene_and_model(
        model.extract(args), 
        args.iteration,
        args.checkpoint_path
        )
    logits = sem_logits.cpu().detach().numpy()
    smoothed_logits = weighted_logit_mean(anchor_points, logits)
    # Pi = gaussians.get_sem_probs()              # [N_anchor, K], softmaxed
    # Pi = sp.softmax(smoothed_logits)

    anchor_id = np.arange(anchor_points.shape[0])

    bg_color = [1,1,1] if model._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    all_views = get_views(scene, skip_train=args.skip_train, skip_test = args.skip_test)
    
    anchor_id = np.arange(anchor_points.shape[0])

    bg_color = [1,1,1] if model._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    projection_data = generate_sam_data_for_anchors(anchor_points, all_views, files_path, anchor_id)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(anchor_points)
    print("Before: ", len(pcd.points))

    voxel_size = 0.04 
     

    newpcd, _, voxel_indices = pcd.voxel_down_sample_and_trace(
    voxel_size, min_bound = pcd.get_min_bound(), max_bound = pcd.get_max_bound(), approximate_class=False)
    print(" After: ", len(newpcd.points))

    anchors_downsampled = np.asarray(newpcd.points)
    counts = np.array([len(points) for points in voxel_indices], dtype=float)
    anchors_downsampled_id = np.arange(anchors_downsampled.shape[0])
    # projection_data_down = generate_sam_data_for_anchors(anchors_downsampled, all_views, files_path, anchors_downsampled_id)

    
    # proj = torch.from_numpy(projection_data_down).long()
    N, V = projection_data.shape
      

    # probs = sp.softmax(logits, axis = 1).astype(np.float32)
    # prob_squares = np.sqrt(probs)

    
    def agreement(a,b):
        both_pos = (a > 0) & (b > 0)                 # mask where both are positive
        denom = int(np.count_nonzero(both_pos))
        if denom == 0:
            return 1.0                                # convention

        num = int(np.count_nonzero((a == b) & both_pos))
        return 1-num / denom
    
    
    def agreement_many(mask_ids_i, mask_ids_B):
        vis_i  = mask_ids_i != -1
        vis_B  = mask_ids_B != -1
        co_vis = vis_B & vis_i
        denom  = np.count_nonzero(co_vis, axis=1)
        same   = (mask_ids_B == mask_ids_i) & co_vis
        num    = np.count_nonzero(same, axis=1)
        out = np.ones(mask_ids_B.shape[0], dtype=np.float32)
        m = denom > 0
        out[m] = 1.0 - (num[m] / denom[m])
        return out
    
    def agreement_single(mask_ids_i: np.ndarray, mask_ids_j: np.ndarray) -> float:   
        vis = (mask_ids_i != -1) & (mask_ids_j != -1)
        denom = int(np.count_nonzero(vis))
        if denom == 0:
            return 1.0
        num = int(np.count_nonzero((mask_ids_i == mask_ids_j) & vis))
        return 1.0 - (num / denom)
    
    
    def euclidean_many(point, neighbours: np.ndarray) -> np.ndarray:
        diff = neighbours - point
        d2 = np.einsum('ij,ij->i', diff, diff, optimize=True)
        return np.sqrt(d2, dtype=np.float32)


    # w_euc = 3 
    # w_emb = 100
    # those two combined with sample_weight=count give actually better results than without it

    # if only emb the 100 was fine

    def build_precomputed_combined(embds, anchors, eps):
        N, V = embds.shape
        
        rows, cols, data = [], [], []
        
        for i in range(N):
            for j in range(i + 1, N):
                d_euc = np.linalg.norm(anchors[i] - anchors[j])  # euclidean
                d_emb = cosine(embds[i], embds[j])  # cosine distance (1-cosine similarity)
                
                    
                # d = w_euc * d_euc + w_emb * d_emb + w_sem * d_sem
                d = d_euc * w_euc + w_emb * d_emb
                if d <= eps:    
                    rows.append(i); cols.append(j); data.append(d)

        A = csr_matrix((np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, N))
        A.setdiag(0.0)
        A = A.maximum(A.T)  # symmetrize
        return A

    def build_precomputed_emb(masks_ids, embds, eps):
        N, V = embds.shape
        
        rows, cols, data = [], [], []
        
        
        for i in range(N):
            for j in range(i + 1, N):
                # d = np.linalg.norm(embds[i] - embds[j])  # euclidean
                
                d = cosine(embds[i], embds[j])
                if d <= eps:    
                    rows.append(i); cols.append(j); data.append(d)

        A = csr_matrix((np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, N))
        A = A.maximum(A.T)  # symmetrize
        return A
    
    def build_precomputed_euc(points, eps):
        N = points.shape[0]
        
    
        rows, cols, data = [], [], []
        i_idx, j_idx = np.triu_indices(N, k=1)
    
        dists = np.linalg.norm(points[i_idx] - points[j_idx], axis=1)

        mask = dists <= eps

        rows.extend(i_idx[mask].tolist())
        cols.extend(j_idx[mask].tolist())
        data.extend(dists[mask].tolist())

        A = csr_matrix((np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, N))
        A.setdiag(0.0)
        A = A.maximum(A.T)  # symmetrize
        return A
    
    def build_precomputed_combined_simple(embds, points, eps):
        N = points.shape[0]
    
        rows, cols, data = [], [], []
        i_idx, j_idx = np.triu_indices(N, k=1)
    
        dists_euc = np.linalg.norm(points[i_idx] - points[j_idx], axis=1)

        D = cosine_distances(embds)      # (N, N) matrix of cosine distances = 1 - cosine similarity
        dists_emb = D[i_idx, j_idx] 

        dists = dists_euc * w_euc + dists_emb * w_emb

        mask = dists <= eps

        rows.extend(i_idx[mask].tolist())
        cols.extend(j_idx[mask].tolist())
        data.extend(dists[mask].tolist())

        A = csr_matrix((np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, N))
        A.setdiag(0.0)
        A = A.maximum(A.T)  # symmetrize
        return A
    
    def build_precomputed(anchor_points, projection_data, embeddings, eps, logits, k_candidates = 512):
        
        mask_ids = projection_data
        N, V = mask_ids.shape

        rows, cols, data = [], [], []

        eps_dist = 0.1 # 0.02
        eps_emb = 0.2
        eps_sem = 0.5


        w_dist = 2 
        w_emb = 0.8
        w_sem = 0.1

        
        nn = NearestNeighbors(
            n_neighbors=min(k_candidates + 1, N),
            metric="euclidean",
            algorithm="ball_tree",
            n_jobs=-1,
        ).fit(anchor_points.astype(np.float32, copy=False))

        dists, inds = nn.kneighbors(anchor_points, return_distance=True)

        # filter out those that are not in eps_dist range (not enough near each other)
        dists_no_self = dists[:, 1:]   # (N, K-1)
        inds_no_self  = inds[:, 1:] 
        within_eps = dists_no_self <= eps_dist

        neighbors_within_eps = [
            inds_row[mask_row] for inds_row, mask_row in zip(inds_no_self, within_eps)
        ]

        dists_within_eps = [
            dists_row[mask_row] for dists_row, mask_row in zip(dists_no_self, within_eps)
        ]

        for i in range(N):
            # neigh = inds[i][1:]  # drop self
            neigh = neighbors_within_eps[i]
            if neigh.size == 0:
                continue
            # diff = anchor_points[i] - anchor_points[neigh]
            # d_euc = np.sqrt(np.einsum('ij,ij->i', diff, diff, optimize=True))
            # keep_euc = d_euc <= 0.4
            # if not np.any(keep_euc):
            #     continue
            # candidates = neigh[keep_euc]

            diff_emb = embeddings[i] - embeddings[neigh]     # (len(neigh), d_emb)
            d_euc_emb = np.linalg.norm(diff_emb, axis=1)
            emb_within_eps = d_euc_emb <= eps_emb
            d_xyz = dists_within_eps[i]
            

            neigh_emb = neigh[emb_within_eps]
            d_xyz_emb = d_xyz[emb_within_eps]
            d_emb_emb = d_euc_emb[emb_within_eps]

            # filter out by embeddings
            # cosine similarity: (v_i · v_j) / (||v_i|| * ||v_j||)
            dot = (smoothed_logits[neigh_emb] * smoothed_logits[i]).sum(axis=1)                 # (M_nz,)
            cos_sim = dot / (np.linalg.norm(smoothed_logits[i]) * np.linalg.norm(smoothed_logits[neigh_emb]))             # (M_nz,)
            d_sem = 1.0 - cos_sim  # cosine distance
            sem_within_eps = d_sem <= eps_sem
            
            neigh_good = neigh_emb[sem_within_eps]
            d_xyz_good = d_xyz_emb[sem_within_eps]
            d_emb_good = d_emb_emb[sem_within_eps]
            d_sem_good = d_sem[sem_within_eps]

            d = d_emb_good * w_emb + d_xyz_good * w_dist + w_sem * d_sem_good
            keep = d < eps
            # keep = d_euc_emb <= eps_emb
            if np.any(keep):
                j = neigh_good[keep]
                d_kept = d[keep]

                rows.extend([i] * len(j))
                cols.extend(j.tolist())
                data.extend(d_kept.tolist())


            # dist_i = dists_within_eps[i]
            # rows.extend([i] * len(neigh))
            # cols.extend(neigh.tolist())
            # data.extend(dist_i.tolist())

        A = csr_matrix((np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, N))
        A = A.maximum(A.T)  # symmetrize
        return A
    
    
    # o3d.visualization.draw_geometries([newpcd])
    
    min_samples = 20
    embds = np.load(f"embeddings_norm_{voxel_size}_{200}_withtrace.npy")

    M, D = embds.shape

    embds_full = np.empty((N, D), dtype=embds.dtype)

    for new_idx, orig_idxs in enumerate(voxel_indices):
        if len(orig_idxs) == 0:
            continue
        embds_full[orig_idxs] = embds[new_idx]


    
    eps = 0.2
    # print(eps)


    
    # A = build_precomputed_emb(
    #     projection_data_down,
    #     embds,
    #     eps
    # )

    # A = build_precomputed_combined_simple(
    #     embds,
    #     anchors_downsampled,
    #     eps
    # )

    # A = build_precomputed_euc(
    #     anchors_downsampled,
    #     eps
    # )

    # A = build_precomputed(
    #     anchor_points,
    #     projection_data,
    #     eps=eps,
    #     k_candidates=512
    # )
    A = build_precomputed(
        anchor_points,
        projection_data,
        embds_full,
        eps,
        smoothed_logits,
        512
    )
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(A)
    # labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(A, sample_weight=counts)  # for later add sample_weight = counts in the sampled voxel as a weight of a poin
  
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"{n_clusters=}, {n_noise=}")

    

    palette = contrast_palette(np.unique(labels))
    anchors_colors = palette[labels]

    # make noise a neutral gray (optional)
    noise = (labels == -1)
    if noise.any():
        anchors_colors[noise] = np.array([0, 0, 0], dtype=np.float64)


    all_views = get_views(scene, skip_train=args.skip_train, skip_test = args.skip_test)

    points, color, opaicity,scaling,rot, normal, _, _, _,_ = generate_neural_gaussians_SDF(all_views[0], gaussianModel, visible_mask=None)
    points = points.cpu().detach().numpy()
    points_normals = torch.nn.functional.normalize(normal).cpu().detach().numpy()
    vertices, triangle, pcd = poisson_surface_reconstruction(points, points_normals, 8) # 9
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangle)
    mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    scale_matrix = np.diag([50, 50, 50])
    pcd.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd.points).T).T)
    normals = np.asarray(pcd.normals)
    scaled_normals =normals * 0.1
    pcd.normals = o3d.utility.Vector3dVector(scaled_normals)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # mesh.compute_vertex_normals()
    
    from scipy.spatial import cKDTree
    kdtree = cKDTree(anchor_points)
    verts = np.asarray(mesh.vertices)
    _, idx = kdtree.query(verts, k=1) 
    v_colors = anchors_colors[idx] 
    mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))

    o3d.io.write_triangle_mesh(f"inst_v2/dist+emb+sem_combined/test1.ply", mesh, write_vertex_colors=True)





















    # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise = int((labels == -1).sum())
    # print(f"{n_clusters=}, {n_noise=}")

    # palette = contrast_palette(np.unique(labels))


    # anchors_colors = palette[labels]

    # # make noise a neutral gray (optional)
    # noise = (labels == -1)
    # if noise.any():
    #     anchors_colors[noise] = np.array([0, 0, 0], dtype=np.float64)

    # # assign to your downsampled cloud (embeddings were for anchors_downsampled)
    # newpcd.colors = o3d.utility.Vector3dVector(anchors_colors)

    
    # o3d.visualization.draw_geometries([newpcd])

    # points, color, opacity,scaling,rot, normal, _, _, _,_ = generate_neural_gaussians_SDF(all_views[0], gaussianModel, visible_mask=None)
    # points = points.cpu().detach().numpy()
    # points_normals = torch.nn.functional.normalize(normal).cpu().detach().numpy()
    # vertices, triangle, pcd2 = poisson_surface_reconstruction(points, points_normals, 8) # 9
    # import open3d as o3d
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh.triangles = o3d.utility.Vector3iVector(triangle)
    # mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    # scale_matrix = np.diag([50, 50, 50])
    # pcd2.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd2.points).T).T)
    # normals = np.asarray(pcd2.normals)
    # scaled_normals =normals * 0.1
    # pcd2.normals = o3d.utility.Vector3dVector(scaled_normals)
    # # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # # mesh.compute_vertex_normals()
    
    # from scipy.spatial import cKDTree
    # kdtree = cKDTree(anchors_downsampled)
    # verts = np.asarray(mesh.vertices)
    # _, idx = kdtree.query(verts, k=1) 

    
    # v_colors = anchors_colors[idx]
    # vc = v_colors.mean(axis=1)
    # print(np.shape(mesh.vertices), np.shape(v_colors))
    # mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))
    
    # o3d.io.write_triangle_mesh(f"inst_v2/test1.ply", mesh, write_vertex_colors=True)



    # NOTES
    # for euclidean only, best where for eps=0.02, kn=2048 and without / r_s