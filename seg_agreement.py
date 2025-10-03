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
from scipy.spatial.distance import jensenshannon, pdist
from scipy.sparse import csr_matrix




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
    if np.any(np.asarray(labels) == -1):
        table[0] = (noise_gray, noise_gray, noise_gray)

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
    anchor_id = np.arange(anchor_points.shape[0])

    bg_color = [1,1,1] if model._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    all_views = get_views(scene, skip_train=args.skip_train, skip_test = args.skip_test)

    gaussianModel, scene, all_anchor_points_cuda, sem_logits = setup_gaussian_scene_and_model(
        model.extract(args), 
        args.iteration,
        args.checkpoint_path
        )
    
    anchor_id = np.arange(anchor_points.shape[0])

    bg_color = [1,1,1] if model._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    projection_data = generate_sam_data_for_anchors(anchor_points, all_views, files_path, anchor_id)
    # ===   

    probs = sp.softmax(logits, axis = 1).astype(np.float32)
    prob_squares = np.sqrt(probs)

    K = probs.shape[1]
    probs = np.clip(probs, 1e-12, 1.0)                          # avoid log/sqrt issues
    U = np.sqrt(probs)                                          # √p
    Z = U / np.sqrt(2.0) 

    A = anchor_points.astype(np.float32)
    # A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)  #uśrednienie do sterowania std zamiast cm, pomogło chyba (wyniki z niżej były na tym)
    # X = np.hstack([anchor_points.astype(np.float32) / w_xyz, Z / w_sem]).astype(np.float32)
    # X = anchor_points.astype(np.float32) / w_xyz
    # X = Z.astype(np.float32) / w_sem


    # min_samples = 20  # try 12–25 for 1-1
    min_samples = 20        
    eps = 0.2 # got killed, last one working was 0.1
    # for xyz only - top min_smaples 40/50, eps = 0.8, w_xyz = 0.04



    # change from Z to Z_norm everywhere
    # idx = np.random.choice(len(Z), size=min(2000, len(Z)), replace=False)
    # E = np.linalg.norm(Z[idx,None]-Z[None,idx], axis=-1)
    # Dq = np.quantile(np.linalg.norm(Z[idx,None]-Z[None,idx], axis=-1), [0.1, 0.5, 0.9])
    # print("Hellinger dist quantiles:", Dq)

    # m = min(2000, len(Z))
    # idx = np.random.choice(len(Z), size=m, replace=False)
    # Zs = Z[idx].astype(np.float32)

    # dists = pdist(Zs, metric='euclidean')  # this is Hellinger since Z = sqrt(p)/√2
    # print("Hellinger pairwise quantiles (0.1, 0.5, 0.9):",
    #     np.quantile(dists, [0.1, 0.5, 0.9]))
    # print("min nonzero distance:", float(dists.min()))

    def hellinger(a,b):
        return np.linalg.norm(a[-K:] - b[-K:])
    
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
    
    def hellinger_many(a_tail: np.ndarray, B_tail: np.ndarray) -> np.ndarray:
        # a_tail, B_tail are the last K dims (already any transform you want)
        diff = B_tail - a_tail  # (M,K)
        return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)
    
    def euclidean_many(point, neighbours: np.ndarray) -> np.ndarray:
        diff = neighbours - point
        d2 = np.einsum('ij,ij->i', diff, diff, optimize=True)
        return np.sqrt(d2, dtype=np.float32)


    def build_precomputed(anchor_points: np.ndarray, projection_data: np.ndarray, eps: float, k_candidates: int = 512):
        
        mask_ids = projection_data
        N, V = mask_ids.shape

        rows, cols, data = [], [], []

        
        nn = NearestNeighbors(
            n_neighbors=min(k_candidates + 1, N),
            metric="euclidean",
            algorithm="ball_tree",
            n_jobs=-1,
        ).fit(anchor_points.astype(np.float32, copy=False))

        dists, inds = nn.kneighbors(anchor_points, return_distance=True)
        rk = dists[:, -1]                                  # k-distance per point
        r_s = float(np.percentile(rk, 80))

        anchor_points = anchor_points / r_s


        for i in range(N):
            neigh = inds[i][1:]  # drop self
            if neigh.size == 0:
                continue
            diff = anchor_points[i] - anchor_points[neigh]
            d_euc = np.sqrt(np.einsum('ij,ij->i', diff, diff, optimize=True))
            keep_euc = d_euc <= eps
            if not np.any(keep_euc):
                continue
            candidates = neigh[keep_euc]

            d = agreement_many(mask_ids[i], mask_ids[candidates])
            keep = np.where(d <= eps)[0]
            if keep.size:
                j = neigh[keep]
                rows.extend([i] * keep.size)
                cols.extend(j.tolist())
                data.extend(d[keep].tolist())

        A = csr_matrix((np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, N))
        A = A.maximum(A.T)  # symmetrize
        return A
    
    def build_precomputed_single(projection_data, eps):
        N = projection_data.shape[0]

        rows, cols, data = [], [], []
        for i in range(N):
            point = projection_data[i]
            for j in range(i + 1, N):
                d = agreement_single(point, projection_data[j])
                if d <= eps:
                    rows.append(i); cols.append(j); data.append(d)

        A = csr_matrix((np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, N))
        A = A.maximum(A.T)  # symmetrize
        return A
    
    # tree = KDTree(anchor_points, leaf_size=40, metric='euclidean')

    # # counts of neighbors within r (includes self)
    # counts_including_self = tree.query_radius(anchor_points, r=1, count_only=True)

    # # exclude self, get stats
    # counts = counts_including_self - 1
    # mean_neighbors = float(np.mean(counts))
    # p50, p90, p99 = np.percentile(counts, [50, 90, 99])

    # print("mean neighbors:", mean_neighbors)
    # print("median / p90 / p99:", p50, p90, p99)
    

    eps = 0.1          
    # k_candidates = 512 

    # A = build_precomputed_single(
    #     projection_data,
    #     eps=eps,
    # )

    A = build_precomputed(
        anchor_points,
        projection_data,
        eps=eps,
        k_candidates=512
    )
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(A)

    # for agreement only - eps=0.1, k_n = 512


    # db = DBSCAN(
    #     eps=eps,
    #     min_samples=min_samples,
    #     metric='euclidean',
    #     algorithm="ball_tree",
    #     # metric=hellinger,
    #     # algorithm="ball_tree",
    #     # leaf_size=100,
    # )
    # labels = db.fit_predict(X)
    
    # ===

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"{n_clusters=}, {n_noise=}")

    palette = contrast_palette(np.unique(labels))
    anchors_colors = palette[labels]
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
    
    o3d.io.write_triangle_mesh("mesh_letsee.ply", mesh, write_vertex_colors=True)



    