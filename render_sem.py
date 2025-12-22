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
from gaussian_renderer import prefilter_voxel, render_semantics, render
import json
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
import colorsys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


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



    return visible_ids, v_valid, u_valid


def get_original_image(camera):
    org_image_np = camera.original_image.permute(1,2,0).cpu().numpy()
    return (org_image_np*255).clip(0, 255).astype(np.uint8)



def get_views(scene, skip_train, skip_test):
    scene_cameras_train = scene.getTrainCameras() if not skip_train else []
    scene_cameras_test = scene.getTestCameras() if not skip_test else []

    views = scene_cameras_train + scene_cameras_test
    return views

def contrast_palette2(classes):
    """
    Deterministic per-class palette.
    For class index i: RGB = (i & 255, (i>>8) & 255, (i>>16) & 255).
    Returns a list of (R,G,B) uint8 tuples, len == len(classes).
    """
    n = len(classes)
    pal = []
    for i in range(n):
        r =  i        & 0xFF
        g = (i >> 8)  & 0xFF
        b = (i >> 16) & 0xFF
        pal.append((r, g, b))
    return pal


def contrast_palette(
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

def contrast_palette_for_classes(
    classes,
    s_range=(0.75, 0.95),
    v_range=(0.65, 1.00),
    base_hue=0.13,
):
    """
    Create a high-contrast palette for class names/ids.
    Returns: list of (R, G, B) uint8 tuples, len == len(classes).

    Deterministic: uses golden-ratio hue stepping + alternating S/V patterns.
    """
    K = max(1, len(list(classes)))
    phi = 0.6180339887498949  # golden ratio conjugate

    # cycle saturation/value for extra local contrast
    sv_patterns = [
        (s_range[1], v_range[1]),  # bright & saturated
        (s_range[1], v_range[0]),  # saturated but darker
        (s_range[0], v_range[1]),  # bright but less saturated
        (s_range[0], v_range[0]),  # darker and less saturated
    ]

    palette = []
    for k in range(K):
        h = (base_hue + k * phi) % 1.0
        s, v = sv_patterns[k % len(sv_patterns)]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette

# def _srgb_to_lin(c):  # c in [0,1]
#     a = 0.055
#     return np.where(c <= 0.04045, c/12.92, ((c+a)/(1+a))**2.4)

# def _rgb_to_lab(rgb):  # rgb [...,3] in [0,1]
#     r,g,b = _srgb_to_lin(rgb[...,0]), _srgb_to_lin(rgb[...,1]), _srgb_to_lin(rgb[...,2])
#     X = 0.4124564*r + 0.3575761*g + 0.1804375*b
#     Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
#     Z = 0.0193339*r + 0.1191920*g + 0.9503041*b
#     Xn, Yn, Zn = 0.95047, 1.0, 1.08883
#     x, y, z = X/Xn, Y/Yn, Z/Zn
#     f = lambda t: np.where(t > 216/24389, np.cbrt(t), (24389/27*t + 16)/116)
#     fx, fy, fz = f(x), f(y), f(z)
#     L = 116*fy - 16
#     a = 500*(fx - fy)
#     b = 200*(fy - fz)
#     return np.stack([L,a,b], -1)

# def class_palette1(classes, base_hue=0.11, min_lab=32.0):
#     """
#     Deterministic, high-contrast palette (order preserved).
#     Increase `min_lab` (e.g. 30–36) for stronger separation.
#     """
#     import numpy as np, colorsys
#     # --- sRGB->Lab helpers ---
#     def _srgb_to_lin(c):
#         a = 0.055
#         return np.where(c <= 0.04045, c/12.92, ((c+a)/(1+a))**2.4)
#     def _rgb_to_lab(rgb):
#         r,g,b = _srgb_to_lin(rgb[...,0]), _srgb_to_lin(rgb[...,1]), _srgb_to_lin(rgb[...,2])
#         X = 0.4124564*r + 0.3575761*g + 0.1804375*b
#         Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
#         Z = 0.0193339*r + 0.1191920*g + 0.9503041*b
#         Xn, Yn, Zn = 0.95047, 1.0, 1.08883
#         x, y, z = X/Xn, Y/Yn, Z/Zn
#         f = lambda t: np.where(t > 216/24389, np.cbrt(t), (24389/27*t + 16)/116)
#         fx, fy, fz = f(x), f(y), f(z)
#         L = 116*fy - 16; a = 500*(fx - fy); b = 200*(fy - fz)
#         return np.stack([L,a,b], -1)

#     K   = len(classes)
#     phi = 0.6180339887498949
#     phi2= 0.7548776662466927
#     phi3= 0.5698402909980532
#     sv_patterns = [(1.00,0.95),(0.90,0.80),(0.80,0.98),(0.95,0.70)]

#     rgb_list, lab_list = [], []
#     for k in range(K):
#         h = (base_hue + k*phi) % 1.0
#         s0,v0 = sv_patterns[k % 4]
#         s = np.clip(s0*(0.92 + 0.12*((k*phi2)%1.0)), 0, 1)
#         v = np.clip(v0*(0.92 + 0.12*((k*phi3)%1.0)), 0, 1)

#         tried = 0
#         while True:
#             r,g,b = colorsys.hsv_to_rgb(h,s,v)
#             cand_rgb = np.array([r,g,b], dtype=np.float32)
#             if not lab_list:
#                 break
#             cand_lab = _rgb_to_lab(cand_rgb[None,:])[0]
#             if all(np.linalg.norm(cand_lab - pl) >= min_lab for pl in lab_list):
#                 break
#             # stronger, deterministic hops if too close:
#             tried += 1
#             if tried % 5 == 0:
#                 h = (h + 0.5) % 1.0          # complementary hue jump
#             else:
#                 h = (h + 0.11) % 1.0         # bigger hue step than before
#             if tried % 2 == 0:
#                 s = np.clip(s*0.93 + 0.07, 0, 1)
#                 v = np.clip(v*0.93 + 0.07, 0, 1)
#             if tried > 16:
#                 break

#         rgb_u8 = tuple((cand_rgb*255.0).round().astype(np.uint8).tolist())
#         rgb_list.append(rgb_u8)
#         lab_list.append(_rgb_to_lab(cand_rgb[None,:])[0])
#     return rgb_list

def _srgb_to_linear(c):
    a = 0.055
    return np.where(c <= 0.04045, c/12.92, ((c + a)/(1 + a))**2.4)

def _linear_to_srgb(c):
    a = 0.055
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, 12.92*c, (1+a)*np.power(c, 1/2.4) - a)

def _oklab_to_srgb(L, a, b):
    # https://bottosson.github.io/posts/oklab/
    l_ = L + 0.3963377774*a + 0.2158037573*b
    m_ = L - 0.1055613458*a - 0.0638541728*b
    s_ = L - 0.0894841775*a - 1.2914855480*b
    l, m, s = l_**3, m_**3, s_**3
    r = +4.0767416621*l - 3.3077115913*m + 0.2309699292*s
    g = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
    b = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
    rgb = np.stack([_linear_to_srgb(r), _linear_to_srgb(g), _linear_to_srgb(b)], -1)
    return np.clip(rgb, 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0)


def _oklch_to_oklab(L, C, h):
    # h in radians
    return L, C*np.cos(h), C*np.sin(h)

def _srgb_in_gamut(rgb):
    return (rgb >= 0.0).all(axis=-1) & (rgb <= 1.0).all(axis=-1)

# perceptual distance (Oklab ΔE ~ Euclidean)
def _oklab_deltaE(c1, c2):
    return np.linalg.norm(c1 - c2, axis=-1)



# -------- Palette generator --------






def class_palette(classes, base_angle=0.6180339887498949):
    """
    Deterministic, high-contrast palette with BIG light/dark swings.
    Keeps class order; no randomness.
    """
    K = max(1, len(classes))

    # 4 lightness levels (very light -> dark) to maximize luminance separation
    L_levels = np.array([0.86, 0.72, 0.58, 0.44], dtype=float)
    # matching chroma levels (stronger when darker)
    C_levels = np.array([0.34, 0.40, 0.46, 0.50], dtype=float)

    # small hue offset; golden-angle stepping for even hue spread
    hue0 = 0.10

    out = []
    for i in range(K):
        # alternate lightness aggressively (bit-mix so neighbors differ a lot in L)
        ring = ((i & 1) << 1) | ((i >> 1) & 1)   # pattern: 0,2,1,3, 0,2,1,3, ...
        L = L_levels[ring % 4]
        C = C_levels[ring % 4]

        # hue: golden angle to avoid collisions
        h = (hue0 + i * base_angle) % 1.0
        theta = 2.0 * np.pi * h
        a = C * np.cos(theta)
        b = C * np.sin(theta)

        srgb = _oklab_to_srgb(L, a, b)
        out.append(tuple((srgb * 255.0 + 0.5).astype(np.uint8)))
    return out


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

def calc_3d_metric(rec_meshfile, gt_meshfile):
    """
    3D reconstruction metric.

    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)


    # found the aligned bbox for the mesh
    to_align, _ = trimesh.bounds.oriented_bounds(mesh_gt)
    mesh_gt.vertices = (to_align[:3, :3] @ mesh_gt.vertices.T + to_align[:3, 3:]).T
    mesh_rec.vertices = (to_align[:3, :3] @ mesh_rec.vertices.T + to_align[:3, 3:]).T

    min_points = mesh_gt.vertices.min(axis=0) * 1.005
    max_points = mesh_gt.vertices.max(axis=0) * 1.005

    mask_min = (mesh_rec.vertices - min_points[None]) > 0
    mask_max = (mesh_rec.vertices - max_points[None]) < 0

    mask = np.concatenate((mask_min, mask_max), axis=1).all(axis=1)
    face_mask = mask[mesh_rec.faces].all(axis=1)

    mesh_rec.update_vertices(mask)
    mesh_rec.update_faces(face_mask)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    


from PIL import Image, ImageDraw, ImageFont
def save_labels_with_present_legend(labels_np, classes, out_path="sem_present_legend.png",
                                    right_pad=260, row_h=18, bg_color=(0,0,0)):
    """
    labels_np: [H,W] int, -1 = background, 0..K-1 = class ids
    classes  : list[str] of length K
    """
    H, W = labels_np.shape
    K = len(classes)

    # colorize label image
    palette = class_palette(classes)                 # list of (R,G,B) 0..255
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[labels_np < 0] = bg_color                    # background
    present_ids = np.unique(labels_np[labels_np >= 0])
    for cid in present_ids:
        rgb[labels_np == cid] = palette[int(cid)]

    # make canvas with right legend panel
    canvas = Image.new("RGB", (W + right_pad, H), (255, 255, 255))
    canvas.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    # header
    x0 = W + 10
    y  = 8
    draw.text((x0, y), "Classes in this image:", fill=(0, 0, 0), font=font)
    y += row_h + 4

    # sort legend by area (largest first)
    areas = []
    for cid in present_ids:
        areas.append((int((labels_np == cid).sum()), int(cid)))
    areas.sort(reverse=True)                         # largest first
    ordered_ids = [cid for _, cid in areas]

    # entries
    for cid in ordered_ids:
        color = palette[cid]
        name  = classes[cid]
        draw.rectangle([x0, y + 2, x0 + 12, y + 14], fill=color, outline=color)
        draw.text((x0 + 18, y), f"{name}", fill=(0, 0, 0), font=font)
        y += row_h
        if y > H - row_h:
            break

    canvas.save(out_path)




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


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(anchor_points)
    print("Before: ", len(pcd.points))

    voxel_size = 0.04 

    labels = np.load("inst_v2/dist+emb+sem_combined/weights_only/labels.npy")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"{n_clusters=}, {n_noise=}")

    voxel_visible_mask = prefilter_voxel(all_views[21], gaussianModel, False, background)

    render_pkg = render_semantics(all_views[21], gaussianModel, background, visible_mask=None, logits = logits)
        
    label_image = render_pkg["sem_label_2d"]
    
    labels_np = label_image.detach().cpu().numpy()

    K = 101
    classes = get_classes()

    save_labels_with_present_legend(labels_np, classes, out_path="sem_labels.png")

   