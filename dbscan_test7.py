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
from scipy.special import softmax
from plyfile import PlyData, PlyElement
from pathlib import Path


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
        gray = np.array([1, 1, 1], dtype=np.float32)
        table[labels == noise_label] = gray

    return table


def contrast_palette3(
    labels,
    s_range=(0.55, 0.95),
    v_range=(0.65, 1.0),
    base_hue=0.13,
    noise_label=-1,
):
    """
    Given a 1D array of integer labels (one per point/vertex/anchor),
    return an array of RGB colors of shape (len(labels), 3).

    - noise_label (e.g. -1) is colored as fixed gray [0.6, 0.6, 0.6]
    - every other unique label gets a distinct color.
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)

    # Separate noise label (if present)
    has_noise = noise_label in uniq
    if has_noise:
        class_labels = [l for l in uniq if l != noise_label]
    else:
        class_labels = uniq.tolist()

    phi = 0.6180339887498949  # golden ratio conjugate
    sv_patterns = [
        (s_range[1], v_range[1]),
        (s_range[1], v_range[0]),
        (s_range[0], v_range[1]),
        (s_range[0], v_range[0]),
    ]

    # Build mapping label -> color
    label_to_color = {}

    for k, lab in enumerate(class_labels):
        h = (base_hue + k * phi) % 1.0
        s, v = sv_patterns[k % len(sv_patterns)]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        label_to_color[int(lab)] = np.array([r, g, b], dtype=np.float32)

    if has_noise:
        label_to_color[int(noise_label)] = np.array([0.6, 0.6, 0.6], dtype=np.float32)

    # Now build full color table in original order
    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
    for lab, col in label_to_color.items():
        colors[labels == lab] = col

    return colors


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

def apply_full_transform(points: np.ndarray) -> np.ndarray:
    scale_matrix = np.diag([4, 4, 4])
    scale_factor = 4
    shift_vector = np.array([2.95531, 1.13268, -0.058562])

    transform = np.array([
        [ 1.06030165e+00, -3.20324289e-03,  7.84449640e-04, -1.23199711e-01],
        [ 3.20130877e-03,  1.06029875e+00, 2.60242687e-03, -4.07212017e-02],
        [-7.92305771e-04, -2.60004585e-03,  1.06030329e+00, -6.42458858e-02],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
    ])

    # 1) skala + przesunięcie
    pts = points * scale_factor + shift_vector[None, :]   # (N, 3)
    # 2) przejście do współrzędnych jednorodnych
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)  # (N, 4)
    # 3) zastosowanie macierzy 4x4
    pts_tf = (transform @ pts_h.T).T[:, :3]  # z powrotem (N, 3)
    return pts_tf


def mean_logit_smoothing(
    anchors_xyz,
    logits,
    k=30,
    self_weight=2.0,
):
    

    N, K = logits.shape

    nn = NearestNeighbors(n_neighbors=min(k, N)).fit(anchors_xyz)
    _, nbr_idx = nn.kneighbors(anchors_xyz, return_distance=True)

    z_neighbors = logits[nbr_idx].sum(axis=1)

    z_self = self_weight * logits

    # Normalization
    denom = self_weight + nbr_idx.shape[1]

    z_out = (z_self + z_neighbors) / denom
    return z_out


def row_softmax(Z):
    Z = Z.astype(np.float64, copy=True)
    Z -= Z.max(axis=1, keepdims=True)
    np.exp(Z, out=Z)
    Z /= Z.sum(axis=1, keepdims=True)
    return Z

def js_distance_rows(P_neigh: np.ndarray, p_i: np.ndarray) -> np.ndarray:
    """
    Jensen–Shannon distance między wieloma rozkładami P_neigh (K,C)
    a jednym rozkładem p_i (C,). Zwraca (K,) w [0,1] dla base=2.
    Działa na różnych wersjach SciPy (axis albo fallback pętlą).
    """
    try:
        # SciPy nowsze: obsługuje axis
        return jensenshannon(P_neigh, p_i[None, :], axis=1, base=2)
    except TypeError:
        # SciPy starsze: fallback
        out = np.empty((P_neigh.shape[0],), dtype=np.float32)
        for k in range(P_neigh.shape[0]):
            out[k] = jensenshannon(P_neigh[k], p_i, base=2)
        return out


def estimate_scales(anchor_points, 
                    embeddings, 
                    logits,
                    k_candidates,
                    sample_size,
                    # p_dist, 
                    # p_emb, 
                    # p_sem
                    ):
    N = anchor_points.shape[0]
    rng = np.random.default_rng(0)
    idx_sample = rng.choice(N, size=min(sample_size, N), replace=False)

    # kNN po geometrii
    nn = NearestNeighbors(
        n_neighbors=min(k_candidates + 1, N),
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit(anchor_points.astype(np.float32, copy=False))

    dists, inds = nn.kneighbors(anchor_points[idx_sample], return_distance=True)
    dists_no_self = dists[:, 1:]

    # skala XYZ
    # s_xyz = np.percentile(dists_no_self, p_dist)
    s_xyz = float(np.median(dists_no_self))

    # przygotowanie logitów
    logits_f = logits.astype(np.float32, copy=False)
    norms = np.linalg.norm(logits_f, axis=1)
    norms[norms == 0.0] = 1.0

    emb_vals = []
    sem_vals = []

    P = row_softmax(logits)  # (N, C), float64, sum=1

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

        # SEM: Jensen–Shannon distance na prawdopodobienstwach
        d_sem = js_distance_rows(P[neigh], P[i])   # (K,)
        sem_vals.append(d_sem)

    emb_vals = np.concatenate(emb_vals)
    sem_vals = np.concatenate(sem_vals)

    # s_emb = np.percentile(emb_vals, p_emb)
    # s_sem = np.percentile(sem_vals, p_sem)
    s_emb = float(np.median(emb_vals))
    s_sem = float(np.median(sem_vals))

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

    choose_model = 8        # CHANGE HERE

    files_path = "./data/replica/scan1/2Dclassification_tests/test1/results/"
    emb_path = f"./outputs/final_sam3/d{choose_model}k_l01/"+"embeddings_norm_0.04_200_withtrace.npy"
    # emb_path = "./trained_embeddings/embeddings_norm_0.04_200_withtrace.npy"
    instance_id_save_path = f"./experiments3/model_d{choose_model}k/instance_ids.npy"
    


    safe_state(args.quiet)

    gaussianModel, scene, anchor_points, sem_logits = setup_gaussian_scene_and_model(
        model.extract(args), 
        args.iteration,
        args.checkpoint_path
        )
    logits = sem_logits.cpu().detach().numpy()
    smoothed_logits = weighted_logit_mean(anchor_points, logits)

    anchor_id = np.arange(anchor_points.shape[0])

    bg_color = [1,1,1] if model._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    all_views = get_views(scene, skip_train=args.skip_train, skip_test = args.skip_test)


    # projection_data = generate_sam_data_for_anchors(anchor_points, all_views, files_path, anchor_id)

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
    # setup of weights equal regarding 90th percentile of each distance:
    # w_dist = 0.2
    # w_emb = 0.0185
    # w_sem = 1.5

    # current setup
    a_dist = 0.6
    a_emb = 0.2

    w_dist = 0.0    # if 1, eps = 0.2 is quite nice, 90th
    w_emb = 0.0     # if 1, eps = 0.12, 0.15 is quite nice, 90th
    w_sem = 1.0    # if 1, eps = 0.05 is quite nice, 90th

    eps = 0.7

    # percentiles for scale calculations (lower percentile -> lower eps -> more clusters)
    # p_dist = 85
    # p_emb = 78
    # p_sem = 70

    k_neighbours = 512

    # output_path = f"./experiments3/model_d{choose_model}k/"+f"wdist={w_dist}_wemb={w_emb}_wsem={w_sem}_pdist{p_dist}_pemb{p_emb}_psem{p_sem}_{k_neighbours}"
    # output_path = f"./experiments3/model_d{choose_model}k/"+f"wdist={w_dist}_wemb={w_emb}_wsem={w_sem}_{k_neighbours}"
    # mesh_save_path = output_path + "/both_segmentations.ply"
    # out_path = Path(output_path)
    # out_path.mkdir(parents=True, exist_ok=True)

    def build_precomputed(anchor_points, embeddings, eps, logits, k_candidates = 512):
        
        N, V = anchor_points.shape

        rows, cols, data = [], [], []

        # for experiments 
        s_xyz, s_emb, s_sem = estimate_scales(anchor_points, embds_full, smoothed_logits, 512, 10000)

        # this is 90th percentile
        # s_xyz=0.12005415831565483 
        # s_emb=1.6521441
        # s_sem=0.026729107

        # s_xyz=0.12005415831565483 # 90th
        # s_emb=1.2897918     # 85th
        # s_sem=0.009147525   # 70th

        nn = NearestNeighbors(
            n_neighbors=min(k_candidates + 1, N),
            metric="euclidean",
            algorithm="ball_tree",
            n_jobs=-1,
        ).fit(anchor_points.astype(np.float32, copy=False))

        dists, inds = nn.kneighbors(anchor_points, return_distance=True)
        dists_no_self = dists[:, 1:]   # (N, K-1)
        inds_no_self  = inds[:, 1:]    # (N, K-1)

        # vals = dists[:, 1:].ravel()              # skip self-distances in column 0
        # p90 = np.percentile(vals, 90)
        # print("90th percentile distance:", p90)
        # 90th percentile distance: 0.16545089823789488 [23/11 18:44:52]

        logits = smoothed_logits.astype(np.float32, copy=False)  # (N, C)
        norms = np.linalg.norm(logits, axis=1)
        norms[norms == 0.0] = 1.0

        P = row_softmax(smoothed_logits)
        # pmax = P.max(axis=1)
        # emb_vals, sem_vals, comb_vals = [], [], []

        for i in range(N):
            neigh = inds_no_self[i]
            
            if neigh.size == 0:
                continue
            d_xyz = dists_no_self[i]
            
            diff_emb = embeddings[i] - embeddings[neigh]
            d_emb = np.linalg.norm(diff_emb, axis=1)
                
            # v_i = logits[i]
            # norm_i = norms[i]

            # v_j = logits[neigh]
            # norm_j = norms[neigh]

            # # dot products
            # dot = np.sum(v_j * v_i, axis=1)                   

            # cosine similarity and distance
            # cos_sim = dot / (norm_i * norm_j)
            # cos_sim = np.clip(cos_sim, -1.0, 1.0)
            # d_sem = 1.0 - cos_sim

            d_sem = js_distance_rows(P[neigh], P[i])       # [0,1]

            # d_sem_s = d_sem / s_sem

            # bounded, smooth transform -> [0,1)
            # D_sem = d_sem_s / (1.0 + d_sem_s)

            # # Hard gate: jesli punkt lub sasiad ma plaski rozklad -> semantyka niewiarygodna
            # unreliable = (pmax[i] < tau_sem) | (pmax[neigh] < tau_sem)
            # if np.any(unreliable):
            #     D_sem = D_sem.copy()
            #     D_sem[unreliable] = 0.7

            # D_xyz = np.clip(d_xyz / s_xyz, 0.0, 1.0)
            # D_emb = np.clip(d_emb / s_emb, 0.0, 1.0)
            # D_sem = np.clip(d_sem / s_sem, 0.0, 1.0)

            # d_xyz_s = d_xyz / s_xyz
            # d_emb_s = d_emb / s_emb

            # D_xyz = d_xyz_s / (1.0 + d_xyz_s)
            # D_emb = d_emb_s / (1.0 + d_emb_s)

            D_xyz = (d_xyz / (s_xyz + 1e-12))
            D_xyz = D_xyz / (1.0 + D_xyz)

            D_emb = (d_emb / (s_emb + 1e-12))
            D_emb = D_emb / (1.0 + D_emb)

            D_sem = (d_sem / (s_sem + 1e-12))
            D_sem = D_sem / (1.0 + D_sem)



            d = D_emb * w_emb + D_xyz * w_dist + w_sem * D_sem

            # emb_vals.append(D_emb)
            # sem_vals.append(d_sem)
            # comb_vals.append(d)

            # rows.extend([i] * len(neigh))
            # cols.extend(neigh.tolist())
            # data.extend(d.tolist())
            
            # keep only those under eps - doesn't change results and saves memory
            keep = d < eps
            
            if np.any(keep):
                j = neigh[keep]
                d_kept = d[keep]

                rows.extend([i] * len(j))
                cols.extend(j.tolist())
                data.extend(d_kept.tolist())


        A = csr_matrix((np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, N))
        A = A.maximum(A.T)

        return A
    
    
    # o3d.visualization.draw_geometries([newpcd])
    
    # Load embeddings, assign all anchors in downsampled voxels the same embeddings
    embds = np.load(emb_path)
    M, D = embds.shape
    embds_full = np.empty((N, D), dtype=embds.dtype)
    # embds_full = np.zeros((N, D), dtype=embds.dtype)      # potencjalnie bezpieczniej - sprawdz!

    for new_idx, orig_idxs in enumerate(voxel_indices):
        if len(orig_idxs) == 0:
            continue
        embds_full[orig_idxs] = embds[new_idx]


    # parameters for DBSCAN
    min_samples = 20
    # eps = 0.55 # 0.3 top for xyz only with scale=median and w=1
   
    A = build_precomputed(
        anchor_points,
        embds_full,
        eps,
        smoothed_logits,
        k_neighbours
    )

    # output_path = f"./experiments3/model_d{choose_model}k/"+f"wdist={w_dist}_wemb={w_emb}_wsem={w_sem}_eps={eps}_{k_neighbours}"
    output_path = f"./experiments3/model_d{choose_model}k/wsem_only/"+f"wdist={w_dist}_wemb={w_emb}_wsem={w_sem}_eps={eps}_{k_neighbours}"
    mesh_save_path = output_path + "/both_segmentations.ply"
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)
    
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(A)
    np.save(output_path+"/instance_ids.npy", labels)

    #labels =  np.load(f"./experiments3/model_d{choose_model}k/"+f"wdist={w_dist}_wemb={w_emb}_wsem={w_sem}_eps={eps}_{k_neighbours}/instance_ids.npy")
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"{n_clusters=}, {n_noise=}")


    all_views = get_views(scene, skip_train=args.skip_train, skip_test = args.skip_test)

    points, color, opaicity,scaling,rot, normal, _, _, _,_ = generate_neural_gaussians_SDF(all_views[0], gaussianModel, visible_mask=None)
    points = points.cpu().detach().numpy()
    points_normals = torch.nn.functional.normalize(normal).cpu().detach().numpy()
    vertices, triangle, pcd = poisson_surface_reconstruction(points, points_normals, 8) # 9


    # === SAVE ALREADY TRANSFORMED MESH === (else change anchors_transformed and vertices_transform)
    anchors_transformed = apply_full_transform(anchor_points)
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    verts_transformed = apply_full_transform(vertices)
    mesh.vertices = o3d.utility.Vector3dVector(verts_transformed)
    mesh.triangles = o3d.utility.Vector3iVector(triangle)
    mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    
    scale_matrix = np.diag([50, 50, 50])
    pcd.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd.points).T).T)
    normals = np.asarray(pcd.normals)
    scaled_normals =normals * 0.1
    pcd.normals = o3d.utility.Vector3dVector(scaled_normals)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # mesh.compute_vertex_normals()

    # Pi = softmax(smoothed_logits, axis=1)                            # [N,K]
    # cls_idx = Pi.argmax(axis=1).astype(int)

    # Pi = row_softmax(smoothed_logits)
    # cls_idx = np.full(Pi.shape[0], -1, int)
    # mask_conf = Pi.max(axis=1) >= 0.5   # shape (N,)
    # cls_idx[mask_conf] = Pi[mask_conf].argmax(axis=1)


    # pick either instances or classes
    anchors_colors_inst = contrast_palette2(labels, noise_label=-1)
    # palette = contrast_palette2(np.unique(labels), noise_label = -1)
    # anchors_colors = palette[labels]

    # classes = get_classes()                         # same list used when you constructed GaussianModel
    # palette = contrast_palette2(classes)
    # anchors_colors = palette[np.clip(cls_idx, 0, palette.shape[0]-1)]
    # noise = (cls_idx == -1)
    # anchors_colors[noise] = np.array([0.6, 0.6, 0.6], dtype=np.float64)
    
    from scipy.spatial import cKDTree
    kdtree = cKDTree(anchors_transformed)
    verts = np.asarray(mesh.vertices)
    _, idx = kdtree.query(verts, k=1) 
    v_colors = anchors_colors_inst[idx] 
    # vertex_class_ids = cls_idx[idx]
    # vertex_instance_id = labels[idx]
    mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))

    # o3d.io.write_triangle_mesh(output_path+f"/wdist={w_dist}_wemb={w_emb}_wsem={w_sem}_eps={eps}_{k_neighbours}.ply", mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(output_path+f"/wdist={w_dist}_wemb={w_emb}_wsem={w_sem}_eps={eps}_{k_neighbours}.ply", mesh, write_vertex_colors=True)

    Pi = row_softmax(smoothed_logits)
    # cls_idx = np.full(Pi.shape[0], -2, int)
    # mask_conf = Pi.max(axis=1) >= 0.1  # shape (N,)
    # cls_idx[mask_conf] = Pi[mask_conf].argmax(axis=1)+1
    cls_idx = Pi.argmax(axis=1).astype(np.int32) + 1
    classes = get_classes()                         # same list used when you constructed GaussianModel
    palette = contrast_palette2(classes)
    anchors_colors_class = palette[np.clip(cls_idx, 0, palette.shape[0]-1)]
    noise = (cls_idx == -1)
    anchors_colors_class[noise] = np.array([0.6, 0.6, 0.6], dtype=np.float64)

    anchor_class_conf = Pi[np.arange(Pi.shape[0]), cls_idx - 1].astype(np.float32)

    from scipy.spatial import cKDTree
    kdtree = cKDTree(anchors_transformed)
    verts = np.asarray(mesh.vertices)
    _, idx = kdtree.query(verts, k=1) 
    v_colors = anchors_colors_class[idx] 
    vertex_class_ids = cls_idx[idx]   # important!  because in training I use indices of the list 'classes' whereas in info_semantics ids start from 1
    vertex_instance_id = labels[idx]
    vertex_class_conf = anchor_class_conf[idx]
    vertex_class_probs = Pi[idx].astype(np.float16)
    mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))

    o3d.io.write_triangle_mesh(output_path+f"/semantic.ply", mesh, write_vertex_colors=True)

    np.savez_compressed(
        output_path + "/vertex_class_probs.npz",
        probs=vertex_class_probs,          # (V, K) float16/float32
        pred_class_id=vertex_class_ids,    # (V,) int32 (opcjonalnie)
        pred_object_id=vertex_instance_id, # (V,) int32 (opcjonalnie)
        vertex_index=np.arange(len(vertex_class_ids), dtype=np.int32),  # (V,) (opcjonalnie)
    )

    ## save mesh with class_id and instance_id per vertex
    ply = PlyData.read(output_path+f"/semantic.ply")
    v = ply["vertex"].data

    vertex_class_ids = np.asarray(vertex_class_ids, dtype=np.int32)
    vertex_instance_id = np.asarray(vertex_instance_id, dtype=np.int32)
    vertex_class_conf = np.asarray(vertex_class_conf, dtype=np.float16)

    # extend vertex dtype with a new 'class_id' field
    new_dtype = v.dtype.descr + [("class_id", "i4"), ("instance_id", "i4"), ("class_conf", "f4")]
    new_v = np.empty(v.shape[0], dtype=new_dtype)

    # copy old fields
    for name in v.dtype.names:
        new_v[name] = v[name]

    # set new field
    new_v["class_id"] = vertex_class_ids
    new_v["instance_id"] = vertex_instance_id
    new_v["class_conf"] = vertex_class_conf

    # keep all other elements as they are
    new_vertex_element = PlyElement.describe(new_v, "vertex")
    new_elements = []
    for elt in ply.elements:
        if elt.name == "vertex":
            new_elements.append(new_vertex_element)
        else:
            new_elements.append(elt)

    # replace and save
    ply["vertex"].data = new_v
    new_ply = PlyData(new_elements, text=ply.text)
    new_ply.write(mesh_save_path)
    print("Saved with per-vertex class_id with confidence and instance_id")



    gt_mesh_path = "./data/replica/scan1/mesh_semantic_verts_bothids.ply"
    gt_ply = PlyData.read(gt_mesh_path)
    v = gt_ply["vertex"].data

    gt_xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    pred_xyz = anchors_transformed.astype(np.float32)
    pred_cls = cls_idx.astype(np.int32)
    pred_obj = labels.astype(np.int32)
    pred_conf = anchor_class_conf.astype(np.float16)

    # --- map anchors (pred) -> gt vertices via nearest neighbor ---
    tree = cKDTree(pred_xyz)
    dists, nn_idx = tree.query(gt_xyz, k=1)
    mapped_pred_cls = pred_cls[nn_idx].astype(np.int32)  # (N_gt,)
    mapped_pred_obj = pred_obj[nn_idx].astype(np.int32)
    mapped_pred_conf = pred_conf[nn_idx].astype(np.float16)
    mapped_pred_probs = Pi[nn_idx].astype(np.float16)  # (N_gt, K)

    np.savez_compressed(
        output_path + "/mapped_vertex_class_probs_onto_gt.npz",
        probs=mapped_pred_probs,
        # pred_class_id=mapped_pred_cls.astype(np.int32),
        # pred_object_id=mapped_pred_obj.astype(np.int32),
    )

    # --- add/overwrite vertex field "pred_class_id" ---
    old_names = v.dtype.names
    new_dtype = [(n, v.dtype.fields[n][0]) for n in old_names if n != "pred_class_id" or n != "pred_object_id"] + [("pred_class_id", "i4"), ("pred_object_id", "i4"), ("pred_class_conf", "f4")]
    v2 = np.empty(v.shape, dtype=new_dtype)

    for n in old_names:
        if n == "pred_class_id" or n == "pred_object_id" or n == "pred_class_conf":
            continue
        v2[n] = v[n]
    v2["pred_class_id"] = mapped_pred_cls
    v2["pred_object_id"] = mapped_pred_obj
    v2["pred_class_conf"] = mapped_pred_conf

    vertex_el = PlyElement.describe(v2, "vertex")

    # keep all other elements (e.g., face)
    new_elements = [vertex_el if el.name == "vertex" else el for el in gt_ply.elements]

    out = PlyData(
        new_elements,
        text=gt_ply.text,
        byte_order=gt_ply.byte_order,
        comments=gt_ply.comments,
        obj_info=gt_ply.obj_info,
    )
    out.write(output_path+f"/mapped_semantic_class_id_&_object_id_onto_gt.ply")
    



    # === CHECK IF CLASS_ID IS STORED OK ===
    # ply_check = PlyData.read(out_path)
    # v2 = ply_check["vertex"].data
    # verts2 = np.vstack([v2["x"], v2["y"], v2["z"]]).T
    # cls2 = np.asarray(v2["class_id"], dtype=np.int32)

    # faces2 = np.vstack(ply_check["face"]["vertex_indices"])
    # mesh_check = o3d.geometry.TriangleMesh()
    # mesh_check.vertices = o3d.utility.Vector3dVector(verts2)
    # mesh_check.triangles = o3d.utility.Vector3iVector(faces2.astype(np.int32))

    # classes = get_classes()
    # palette = contrast_palette2(classes)
    # colors2 = palette[np.clip(cls2, 0, palette.shape[0] - 1)]
    # colors2[cls2 == -1] = np.array([0.6, 0.6, 0.6], dtype=np.float64)

    # mesh_check.vertex_colors = o3d.utility.Vector3dVector(colors2.astype(np.float64))
    # o3d.io.write_triangle_mesh(
    #     "inst_v2/dist+emb+sem_combined/experiments/test2_classid_check.ply",
    #     mesh_check,
    #     write_vertex_colors=True,
    # )


    # === CHECK IF INSTANCE_ID IS STORED OK ===
    # ply_check_inst = PlyData.read(out_path)
    # v_inst = ply_check_inst["vertex"].data

    # verts_inst = np.vstack([v_inst["x"], v_inst["y"], v_inst["z"]]).T
    # inst2 = np.asarray(v_inst["instance_id"], dtype=np.int32)
    # faces_inst = np.vstack(ply_check_inst["face"]["vertex_indices"])

    # mesh_inst = o3d.geometry.TriangleMesh()
    # mesh_inst.vertices = o3d.utility.Vector3dVector(verts_inst)
    # mesh_inst.triangles = o3d.utility.Vector3iVector(faces_inst.astype(np.int32))

    # colors_inst = np.zeros((inst2.shape[0], 3), dtype=np.float64)

    # valid_mask = inst2 >= 0
    # uniq_valid = np.unique(inst2[valid_mask])

    # if uniq_valid.size > 0:
    #     palette_inst = contrast_palette2(np.arange(len(uniq_valid)), noise_label=-1)
    #     id2idx = {int(iid): k for k, iid in enumerate(uniq_valid)}

    #     for vid, iid in enumerate(inst2):
    #         if iid < 0:
    #             colors_inst[vid] = np.array([0.6, 0.6, 0.6], dtype=np.float64)  # noise
    #         else:
    #             colors_inst[vid] = palette_inst[id2idx[int(iid)]]
    # else:
    #     colors_inst[:] = np.array([0.6, 0.6, 0.6], dtype=np.float64)

    # # sanity print
    # noise_mask = (inst2 == -1)
    # print("Unique colors for noise verts:", np.unique(colors_inst[noise_mask], axis=0))

    # mesh_inst.vertex_colors = o3d.utility.Vector3dVector(colors_inst)
    # o3d.io.write_triangle_mesh(
    #     "inst_v2/dist+emb+sem_combined/experiments/test2_instanceid_check.ply",
    #     mesh_inst,
    #     write_vertex_colors=True,
    # )

