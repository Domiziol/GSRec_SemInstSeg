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

def contrast_palette(labels, s=0.85, v=0.98):
    
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


    return table

def contrast_palette2(
    labels,
    s_range=(0.55, 0.95),
    v_range=(0.65, 1.0),
    base_hue=0.13,
    noise_label=-1,
):
    """
    Create a high-contrast RGB palette for the given labels.

    - labels: 1D array-like of label ids (e.g. DBSCAN labels)
    - s_range: (min_s, max_s) saturation range
    - v_range: (min_v, max_v) value/brightness range
    - base_hue: starting hue in [0, 1]
    - noise_label: label id to paint as gray (e.g. -1 for DBSCAN)
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)

    # Separate noise label (if present) so it gets a fixed gray color
    has_noise = noise_label in uniq
    if has_noise:
        class_labels = [l for l in uniq if l != noise_label]
    else:
        class_labels = uniq.tolist()

    n_classes = max(1, len(class_labels))
    phi = 0.6180339887498949  # golden ratio conjugate

    # Pre-allocate output table in *input* label order
    table = np.zeros((len(labels), 3), dtype=np.float32)

    # Make a mapping: label -> (index in class_labels) so colors are stable
    label_to_idx = {lab: i for i, lab in enumerate(class_labels)}

    # Small pattern of (s, v) combinations to boost local contrast
    # neighbors will differ in both hue AND (s, v)
    sv_patterns = [
        (s_range[1], v_range[1]),  # bright & saturated
        (s_range[1], v_range[0]),  # saturated but darker
        (s_range[0], v_range[1]),  # bright but less saturated
        (s_range[0], v_range[0]),  # darker and less saturated
    ]

    # First assign colors for all non-noise labels
    for idx, lab in enumerate(class_labels):
        k = label_to_idx[lab]

        # 1) Hue using golden ratio sequence (good global separation)
        h = (base_hue + k * phi) % 1.0

        # 2) Saturation + value: cycle through sv_patterns for local contrast
        s, v = sv_patterns[k % len(sv_patterns)]

        r, g, b = colorsys.hsv_to_rgb(h, s, v)

        # Set this color for all entries with label == lab
        table[labels == lab] = (r, g, b)

    # Now handle noise label as mid-gray (or tweak as you like)
    if has_noise:
        gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        table[labels == noise_label] = gray

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

def estimate_scales(anchor_points, embeddings, logits,
                    k_candidates=512,
                    sample_size=10000,
                    random_state=0):
    N = anchor_points.shape[0]
    rng = np.random.default_rng(random_state)
    idx_sample = rng.choice(N, size=min(sample_size, N), replace=False)

    # kNN po geometrii
    nn = NearestNeighbors(
        n_neighbors=min(k_candidates + 1, N),
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit(anchor_points.astype(np.float32, copy=False))

    dists, inds = nn.kneighbors(anchor_points[idx_sample], return_distance=True)
    dists_no_self = dists[:, 1:]          # (Ns, K)

    # skala XYZ
    s_xyz = np.percentile(dists_no_self, 90)

    # przygotowanie logitów
    logits_f = logits.astype(np.float32, copy=False)
    norms = np.linalg.norm(logits_f, axis=1)
    norms[norms == 0.0] = 1.0

    emb_vals = []
    sem_vals = []

    for row_idx, i in enumerate(idx_sample):
        neigh = inds[row_idx, 1:]
        if neigh.size == 0:
            continue

        # EMB
        diff_emb = embeddings[i] - embeddings[neigh]
        d_emb = np.linalg.norm(diff_emb, axis=1)
        emb_vals.append(d_emb)

        # SEM (cosine distance)
        v_i = logits_f[i]
        norm_i = norms[i]
        v_j = logits_f[neigh]
        norm_j = norms[neigh]

        dot = np.sum(v_j * v_i, axis=1)
        cos_sim = dot / (norm_i * norm_j)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        d_sem = 1.0 - cos_sim

        sem_vals.append(d_sem)

    emb_vals = np.concatenate(emb_vals)
    sem_vals = np.concatenate(sem_vals)

    s_emb = np.percentile(emb_vals, 85)
    s_sem = np.percentile(sem_vals, 70)

    print("scales:", "xyz", s_xyz, "emb", s_emb, "sem", s_sem)
    # xyz-90, emb-90, sem-90
    # scales: xyz 0.12005415831565483 emb 1.6521441 sem 0.026729107 [07/12 22:17:14]

    # xyz-90, emb-85, sem-70
    # scales: xyz 0.12005415831565483 emb 1.2897918 sem 0.009147525 [07/12 22:48:04]
    return s_xyz, s_emb, s_sem


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

    N, V = anchor_points.shape

    def softmax_max_and_argmax(logits: np.ndarray):
        """Stable softmax max-prob and argmax for each row."""
        x = logits.astype(np.float32, copy=False)
        x = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(x)
        s = ex / np.sum(ex, axis=1, keepdims=True)
        conf = np.max(s, axis=1)
        cls  = np.argmax(s, axis=1).astype(np.int32)
        return conf, cls

    def semantic_distance_onehot_neighbors(
        logits: np.ndarray,
        inds_no_self: np.ndarray,
        conf_thr: float = 0.5,
        same_dist: float = 0.1,
        diff_dist: float = 1.0,
        unknown_unknown_dist: float = 1.0,
        unknown_other_dist: float = 0.6,
    ):
        """
        Returns d_sem of shape (N, K) for each anchor i to its neighbors inds_no_self[i, :].
        Rules:
        - if both unknown (conf < thr): distance = unknown_unknown_dist
        - else if both confident and same class: distance = same_dist
        - else (different class OR one unknown): distance = diff_dist / unknown_other_dist
        """
        N, K = inds_no_self.shape
        conf, cls = softmax_max_and_argmax(logits)
        known = conf >= conf_thr

        d_sem = np.empty((N, K), dtype=np.float32)

        for i in range(N):
            neigh = inds_no_self[i]
            ki = known[i]

            if not ki:
                # i unknown
                kj = known[neigh]
                # both unknown -> 1, mixed -> unknown_other_dist
                d_sem[i] = np.where(kj, unknown_other_dist, unknown_unknown_dist).astype(np.float32)
            else:
                # i known
                kj = known[neigh]
                same = (cls[neigh] == cls[i]) & kj
                # if neighbor unknown -> unknown_other_dist
                # if neighbor known & same -> same_dist
                # else -> diff_dist
                d_sem[i] = np.where(~kj, unknown_other_dist,
                                    np.where(same, same_dist, diff_dist)).astype(np.float32)

        return d_sem

    # setup of weights equal regarding 90th percentile of each distance:
    # w_dist = 0.2
    # w_emb = 0.0185
    # w_sem = 1.5

    # current setup
    w_dist = 0.0    # if 1, eps = 0.2 is quite nice, 90th
    w_emb = 0.0     # if 1, eps = 0.12, 0.15 is quite nice, 90th
    w_sem =1.0     # if 1, eps = 0.05 is quite nice, 90th
    def build_precomputed(anchor_points, projection_data, embeddings, eps, logits, k_candidates = 512):
        
        mask_ids = projection_data
        N, V = mask_ids.shape

        rows, cols, data = [], [], []

        # for experiments 
        #s_xyz, s_emb, s_sem = estimate_scales(anchor_points, embds_full, smoothed_logits, 512, 10000)

        # this is 90th percentile
        # s_xyz=0.12005415831565483 
        # s_emb=1.6521441
        # s_sem=0.026729107


        s_xyz=0.12005415831565483 # 90th
        s_emb=1.2897918     # 85th
        s_sem=0.009147525   # 70th

        nn = NearestNeighbors(
            n_neighbors=min(k_candidates + 1, N),
            metric="euclidean",
            algorithm="ball_tree",
            n_jobs=-1,
        ).fit(anchor_points.astype(np.float32, copy=False))

        dists, inds = nn.kneighbors(anchor_points, return_distance=True)
        dists_no_self = dists[:, 1:]   # (N, K-1)
        inds_no_self  = inds[:, 1:]    # (N, K-1)

        vals = dists[:, 1:].ravel()              # skip self-distances in column 0
        # p90 = np.percentile(vals, 90)
        # print("90th percentile distance:", p90)
        # 90th percentile distance: 0.16545089823789488 [23/11 18:44:52]

        logits = smoothed_logits.astype(np.float32, copy=False)  # (N, C)
        norms = np.linalg.norm(logits, axis=1)
        norms[norms == 0.0] = 1.0

        emb_vals, sem_vals, comb_vals = [], [], []
        d_sem_mat = semantic_distance_onehot_neighbors(
            logits=smoothed_logits,   # or logits argument
            inds_no_self=inds_no_self,
            conf_thr=0.5,
            same_dist=0.1,
            diff_dist=1.0,
            unknown_unknown_dist=1.0,
            unknown_other_dist=0.6,   
        )

        for i in range(N):
            neigh = inds_no_self[i]
            
            if neigh.size == 0:
                continue
            d_xyz = dists_no_self[i]
            
            diff_emb = embeddings[i] - embeddings[neigh]      # (len(neigh), d_emb)
            d_emb = np.linalg.norm(diff_emb, axis=1)

            d_sem = d_sem_mat[i]
                
            # v_i = logits[i]
            # norm_i = norms[i]

            # v_j = logits[neigh]
            # norm_j = norms[neigh]

            # # dot products
            # dot = np.sum(v_j * v_i, axis=1)                   

            # # cosine similarity and distance
            # cos_sim = dot / (norm_i * norm_j)
            # cos_sim = np.clip(cos_sim, -1.0, 1.0)
            # d_sem = 1.0 - cos_sim

            D_xyz = np.clip(d_xyz / s_xyz, 0.0, 1.0)
            D_emb = np.clip(d_emb / s_emb, 0.0, 1.0)
            D_sem = np.clip(d_sem / s_sem, 0.0, 1.0)


            d = D_emb * w_emb + D_xyz * w_dist + w_sem * D_sem

            # emb_vals.append(D_emb)
            # sem_vals.append(d_sem)
            # comb_vals.append(d)

            # rows.extend([i] * len(neigh))
            # cols.extend(neigh.tolist())
            # data.extend(d.tolist())
            
            keep = d < eps
            # keep = d_euc_emb <= eps_emb
            if np.any(keep):
                j = neigh[keep]
                d_kept = d[keep]

                rows.extend([i] * len(j))
                cols.extend(j.tolist())
                data.extend(d_kept.tolist())


        A = csr_matrix((np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, N))
        A = A.maximum(A.T)  # symmetrize

        # for name, vals in [("d_emb", emb_vals), ("d_sem", sem_vals), ("d", comb_vals)]:
        #     v = np.concatenate(vals)
        #     print(name, "90th percentile:", np.percentile(v, 90))

        # d_emb 90th percentile: 1.7861496 [23/11 18:45:06]
        # d_sem 90th percentile: 0.033379257 [23/11 18:45:07]
        # d 90th percentile: 0.20250601678229865 [23/11 18:45:08]

        return A
    
    
    # o3d.visualization.draw_geometries([newpcd])
    
    min_samples = 2
    embds = np.load(f"trained_embeddings/embeddings_norm_{voxel_size}_{200}_withtrace.npy")

    M, D = embds.shape

    embds_full = np.empty((N, D), dtype=embds.dtype)

    for new_idx, orig_idxs in enumerate(voxel_indices):
        if len(orig_idxs) == 0:
            continue
        embds_full[orig_idxs] = embds[new_idx]

    
    
    eps = 0.8
   

    A = build_precomputed(
        anchor_points,
        projection_data,
        embds_full,
        eps,
        smoothed_logits,
        512
    )
    print(eps)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(A)
    # labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(A, sample_weight=counts)  # for later add sample_weight = counts in the sampled voxel as a weight of a poin
    np.save("inst_v2/dist+emb+sem_combined/sem_one_hot/labels.npy", labels)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"{n_clusters=}, {n_noise=}")

    

    palette = contrast_palette2(np.unique(labels))
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


    anchor_points_scaled = anchor_points
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangle)
    mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    # scale_matrix = np.diag([50, 50, 50])
    scale_matrix = np.diag([50, 50, 50])
    pcd.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd.points).T).T)
    normals = np.asarray(pcd.normals)
    scaled_normals =normals * 0.1
    pcd.normals = o3d.utility.Vector3dVector(scaled_normals)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # mesh.compute_vertex_normals()
    
    from scipy.spatial import cKDTree
    kdtree = cKDTree(anchor_points_scaled)
    verts = np.asarray(mesh.vertices)
    _, idx = kdtree.query(verts, k=1) 
    v_colors = anchors_colors[idx] 
    mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))

    o3d.io.write_triangle_mesh(f"inst_v2/dist+emb+sem_combined/sem_one_hot/test1.ply", mesh, write_vertex_colors=True)

