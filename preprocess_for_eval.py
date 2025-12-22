import torch
import numpy as np
import os

from scene import Scene
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import poisson_surface_reconstruction
from gaussian_renderer import generate_neural_gaussians_SDF
import json
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
import colorsys
from plyfile import PlyData, PlyElement
from collections import defaultdict

def get_classes():
    classes= []

    with open("info_semantic.json", 'r') as classes_file:
        data = json.load(classes_file)

    for objects in data['classes']:
        classes.append(objects['name'])

    return classes

def load_objid_to_class():
    # map object_id (instance) to class_id (semantics)
    with open("info_semantic.json", 'r') as f:
        data = json.load(f)

    objid_to_class = {}
    for object in data["objects"]:
        objid_to_class[object["id"]] = object["class_id"]

    return objid_to_class

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


def get_views(scene, skip_train, skip_test):
    scene_cameras_train = scene.getTrainCameras() if not skip_train else []
    scene_cameras_test = scene.getTestCameras() if not skip_test else []

    views = scene_cameras_train + scene_cameras_test
    return views


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

    # == LOAD GAUSSIAN SCENE ==
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
    anchors_transformed = apply_full_transform(anchor_points)

    Pi = row_softmax(smoothed_logits)
    cls_idx = np.full(Pi.shape[0], -2, int)
    mask_conf = Pi.max(axis=1) >= 0.1  # shape (N,)
    cls_idx[mask_conf] = Pi[mask_conf].argmax(axis=1) # !!!
    # == END LOAD GAUSSIAN SCENE

    choose_model = 8

    gt_mesh_path = "./data/replica/scan1/mesh_semantic_verts_bothids.ply"
    gt_info_sem_path = "./data/replica/scan1/info_semantic.json"
    pred_mesh_path = f"./experiments2_fromsam3/model_d{choose_model}k/wdist=0.0_wemb=1.0_wsem=0.0_pdist85_pemb78_psem70_512/both_segmentations.ply"

    # pred = PlyData.read(pred_mesh_path)["vertex"].data
    # gt   = PlyData.read(gt_mesh_path)["vertex"].data
    
    # pred_xyz = np.stack([pred["x"], pred["y"], pred["z"]], axis=1).astype(np.float32)
    # gt_xyz   = np.stack([gt["x"],   gt["y"],   gt["z"]],   axis=1).astype(np.float32)

    # pred_cls = pred["class_id"].astype(np.int32)
    # gt_cls = gt["class_id"].astype(np.int32)

    # # --- map pred -> gt via nearest neighbor ---
    # from scipy.spatial import cKDTree
    # tree = cKDTree(pred_xyz)
    # dists, nn_idx = tree.query(gt_xyz, k=1)  # nn in pred for each gt vertex
    

    # mapped_pred_cls= pred_cls[nn_idx]

    # acc = (mapped_pred_cls == gt_cls).mean()
    # correct = int((mapped_pred_cls == gt_cls).sum())

    # print(
    #     f"Vertex accuracy ({"class_id"}) with NN mapping: "
    #     f"{acc*100:.2f}% ({correct}/{len(gt_cls)})"
    # )
    # print(f"NN distance stats: min={float(dists.min()):.6f}, mean={float(dists.mean()):.6f}, max={float(dists.max()):.6f}")

    # classes = np.unique(gt_cls)

    # acc_per_class = np.array([(mapped_pred_cls[gt_cls == c] == c).mean() for c in classes])
    # print("\n".join([f"class {c:3d}: {a*100:6.2f}% (n={int((gt_cls==c).sum())})" for c, a in zip(classes, acc_per_class)]))
    # classes = np.unique(gt_cls)
    # correct_per_class = np.array([(mapped_pred_cls[gt_cls == c] == c).sum() for c in classes], dtype=np.int32)
    # total_per_class   = np.array([(gt_cls == c).sum() for c in classes], dtype=np.int32)
    # acc_per_class     = correct_per_class / np.maximum(total_per_class, 1)

    # print("\n".join([
    #     f"class {c:3d}: {a*100:6.2f}% ({int(k)}/{int(n)})"
    #     for c, a, k, n in zip(classes, acc_per_class, correct_per_class, total_per_class)
    # ]))



    gt = PlyData.read(gt_mesh_path)["vertex"].data

    gt_xyz = np.stack([gt["x"], gt["y"], gt["z"]], axis=1).astype(np.float32)
    gt_cls = gt["class_id"].astype(np.int32)

    pred_xyz = anchors_transformed.astype(np.float32)
    pred_cls = cls_idx.astype(np.int32)+1

    # --- map anchors (pred) -> gt vertices via nearest neighbor ---
    tree = cKDTree(pred_xyz)
    dists, nn_idx = tree.query(gt_xyz, k=1)

    mapped_pred_cls = pred_cls[nn_idx]

    c = 47
    gt47   = (gt_cls == c)
    pred47 = (mapped_pred_cls == c)

    tp = int((gt47 & pred47).sum())      # GT=47 and Pred=47
    fn = int((gt47 & ~pred47).sum())     # GT=47 but Pred!=47
    fp = int((~gt47 & pred47).sum())     # GT!=47 but Pred=47

    recall = tp / max(tp + fn, 1)        # <- THIS is the per-class accuracy you want
    prec   = tp / max(tp + fp, 1)
    iou    = tp / max(tp + fp + fn, 1)

    print(f"class {c}: TP={tp} FN={fn} FP={fp} | recall={recall*100:.2f}% prec={prec*100:.2f}% IoU={iou*100:.2f}%")
    print(f"class {c}: GT count={int(gt47.sum())}  Pred count={int(pred47.sum())}")

    acc = (mapped_pred_cls == gt_cls).mean()
    correct = int((mapped_pred_cls == gt_cls).sum())

    print(f"Vertex accuracy (class_id) with NN mapping (anchors->gt): {acc*100:.2f}% ({correct}/{len(gt_cls)})")
    print(f"NN distance stats: min={float(dists.min()):.6f}, mean={float(dists.mean()):.6f}, max={float(dists.max()):.6f}")

    # classes = np.unique(gt_cls)
    # acc_per_class = np.array([(mapped_pred_cls[gt_cls == c] == c).mean() for c in classes])
    # print("\n".join([f"class {c:3d}: {a*100:6.2f}% (n={int((gt_cls==c).sum())})"
    #                 for c, a in zip(classes, acc_per_class)]))
    classes = np.unique(gt_cls)
    correct_per_class = np.array([(mapped_pred_cls[gt_cls == c] == c).sum() for c in classes], dtype=np.int32)
    total_per_class   = np.array([(gt_cls == c).sum() for c in classes], dtype=np.int32)
    acc_per_class     = correct_per_class / np.maximum(total_per_class, 1)

    print("\n".join([
        f"class {c:3d}: {a*100:6.2f}% ({int(k)}/{int(n)})"
        for c, a, k, n in zip(classes, acc_per_class, correct_per_class, total_per_class)
    ]))

    c = 47
    gt47   = (gt_cls == c)
    pred47 = (mapped_pred_cls == c)

    tp = gt47 & pred47
    fn = gt47 & (~pred47)
    fp = (~gt47) & pred47

    # colors: TP=white, FN=red, FP=blue, else=black
    r = np.zeros(len(gt_cls), dtype=np.uint8)
    g = np.zeros(len(gt_cls), dtype=np.uint8)
    b = np.zeros(len(gt_cls), dtype=np.uint8)

    r[tp] = g[tp] = b[tp] = 255
    r[fn] = 255
    b[fp] = 255

    # build a vertex array with RGB (keep all existing vertex fields)
    from plyfile import PlyData, PlyElement
    gt_ply = PlyData.read(gt_mesh_path)
    v = gt_ply["vertex"].data
    old_names = v.dtype.names

    new_dtype = list(v.dtype.descr)
    if "red" not in old_names:   new_dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    v2 = np.empty(v.shape, dtype=new_dtype)

    for n in old_names:
        v2[n] = v[n]
    v2["red"], v2["green"], v2["blue"] = r, g, b

    vertex_el = PlyElement.describe(v2, "vertex")

    # replace the existing 'vertex' element while keeping all other elements (e.g., 'face')
    new_elements = []
    for el in gt_ply.elements:
        if el.name == "vertex":
            new_elements.append(vertex_el)
        else:
            new_elements.append(el)

    # rebuild PlyData and write
    out = PlyData(
        new_elements,
        text=gt_ply.text,
        byte_order=gt_ply.byte_order,
        comments=gt_ply.comments,
        obj_info=gt_ply.obj_info,
    )
    out.write("./semantics_comparison/test_mapping_fpfntp.ply")

    # target_c = 47
    # ok47 = (gt_cls == target_c) & (mapped_pred_cls == target_c)  # only vertices correctly predicted as 47

    # # --- build per-vertex colors (white for ok47, else black) ---
    # rgb = np.zeros((len(gt_cls), 3), dtype=np.uint8)
    # rgb[ok47] = (255, 255, 255)

    # # --- write a colored copy of the GT mesh (keep faces, overwrite/add vertex colors) ---
    # ply = PlyData.read(gt_mesh_path)
    # v = ply["vertex"].data

    # names = v.dtype.names
    # has_rgb = all(k in names for k in ("red", "green", "blue"))

    # if has_rgb:
    #     v_new = v.copy()
    # else:
    #     # add color fields if missing
    #     new_dtype = list(v.dtype.descr) + [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    #     v_new = np.empty(v.shape, dtype=new_dtype)
    #     for n in names:
    #         v_new[n] = v[n]

    # v_new["red"] = rgb[:, 0]
    # v_new["green"] = rgb[:, 1]
    # v_new["blue"] = rgb[:, 2]

    # vert_el = PlyElement.describe(v_new, "vertex")
    # elements = [vert_el]
    # if "face" in ply:
    #     elements.append(ply["face"])

    # out_ply = "./semantics_comparison/test_mapping_agreed.ply"
    # PlyData(elements, text=ply.text).write(out_ply)
    # print(f"Saved: {out_ply} (white = correct class {target_c}, black = everything else)")

    # target_class_id = 47
    # out_ply = "./semantics_comparison/test_mapping.ply"

    # ply = PlyData.read(gt_mesh_path)
    # v = ply["vertex"].data
    # names = list(v.dtype.names)

    # # rebuild vertex table: keep everything except possibly existing pred_class_id / colors
    # drop = {"pred_class_id", "red", "green", "blue"}
    # kept = [n for n in names if n not in drop]

    # new_dtype = [(n, v.dtype.fields[n][0]) for n in kept] + [
    #     ("pred_class_id", "i4"),
    #     ("red", "u1"), ("green", "u1"), ("blue", "u1")
    # ]
    # v_new = np.empty(v.shape, dtype=new_dtype)

    # for n in kept:
    #     v_new[n] = v[n]

    # v_new["pred_class_id"] = mapped_pred_cls.astype(np.int32)

    # is_white = (mapped_pred_cls.astype(np.int32) == int(target_class_id))
    # c = np.zeros((len(v_new), 3), dtype=np.uint8)
    # c[is_white] = 255
    # v_new["red"], v_new["green"], v_new["blue"] = c[:, 0], c[:, 1], c[:, 2]

    # verts_el = PlyElement.describe(v_new, "vertex")
    # elements = [verts_el] + ([ply["face"]] if "face" in ply else [])
    # PlyData(elements, text=ply.text).write(out_ply)

    # print(f"Saved: {out_ply} (white=pred_class_id=={target_class_id}, else black)")


    # --- build output vertex dtype: keep all original fields, replace/add red/green/blue ---
    # ply = PlyData.read(gt_mesh_path)

    # v = ply["vertex"].data
    # cls = v["class_id"].astype(np.int32)

    # # --- build output vertex dtype: keep all original fields, replace/add red/green/blue ---
    # drop = {"red", "green", "blue", "r", "g", "b"}
    # old_names = list(v.dtype.names)

    # base_dtype = [(n, v.dtype[n]) for n in old_names if n not in drop]
    # new_dtype = base_dtype + [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    # v_new = np.empty(v.shape, dtype=new_dtype)

    # # copy original fields
    # for n in old_names:
    #     if n in drop:
    #         continue
    #     v_new[n] = v[n]

    # # assign colors: class==TARGET -> white else black
    # mask = (cls == 47)
    # col = np.zeros((len(cls), 3), dtype=np.uint8)
    # col[mask] = 255

    # v_new["red"]   = col[:, 0]
    # v_new["green"] = col[:, 1]
    # v_new["blue"]  = col[:, 2]

    # # replace vertex element, keep everything else (faces etc.) as-is
    # elements_out = [PlyElement.describe(v_new, "vertex")]
    # for e in ply.elements:
    #     if e.name != "vertex":
    #         elements_out.append(e)

    # PlyData(elements_out, text=ply.text).write("semantics_comparison/test_mapping.ply")
    

    # === ADD TO GT MESH THE CLASS_ID PER VERTEX ===
    # with open(gt_info_sem_path, "r") as f:
    #     info = json.load(f)

    # id_to_label = np.asarray(info["id_to_label"], dtype=np.int32)  # length ~95 in your case
    # print(len(id_to_label))
    # uniq_all = np.unique(id_to_label)
    # print("unique class_ids (including -1/-2 if present):", uniq_all)
    # print("count:", len(uniq_all))

    # # If you want only *valid* semantic class ids (ignore negatives)
    # uniq_valid = np.unique(id_to_label[id_to_label >= 0])
    # print("unique valid class_ids (>=0):", uniq_valid)
    # print("count valid:", len(uniq_valid))

    # # --- read mesh ---
    # ply = PlyData.read(gt_mesh_path)
    # v = ply["vertex"].data  # structured array

    # if "object_id" not in v.dtype.names:
    #     raise KeyError(f"'object_id' not found in vertex properties. Found: {v.dtype.names}")

    # obj = v["object_id"].astype(np.int32)

    # # --- map object_id -> class_id safely ---
    # class_id = np.full(obj.shape, -1, dtype=np.int32)  # fallback
    # valid = (obj >= 0) & (obj < len(id_to_label))
    # class_id[valid] = id_to_label[obj[valid]]  # may yield -2/-1 too (we keep them as-is)

    # # --- create new vertex array with extra field class_id ---
    # old_names = v.dtype.names
    # # if class_id already exists, overwrite it (simplest: rebuild anyway)
    # new_dtype = [(n, v.dtype.fields[n][0]) for n in old_names if n != "class_id"] + [("class_id", "i4")]
    # v_new = np.empty(v.shape, dtype=new_dtype)

    # for n in old_names:
    #     if n == "class_id":
    #         continue
    #     v_new[n] = v[n]
    # v_new["class_id"] = class_id

    # # --- write out, keeping faces unchanged ---
    # vertex_el = PlyElement.describe(v_new, "vertex")
    # elements = [vertex_el]
    # if "face" in ply:
    #     elements.append(ply["face"])

    # PlyData(elements, text=ply.text).write("./data/replica/scan1/mesh_semantic_verts_bothids.ply")
    # print("Wrote:", "./data/replica/scan1/mesh_semantic_verts_bothids.ply")
    # print("unique object_id:", len(np.unique(obj)))
    # print("unique class_id:", len(np.unique(class_id)), "min/max:", int(class_id.min()), int(class_id.max()))
    # === END ADD TO GT MESH THE CLASS_ID PER VERTEX ===



    # === CHECK VERTEX OBJECTID ACCURACY (only for equal number of vertices)===
    # pred = PlyData.read(pred_mesh_path)["vertex"].data
    # gt   = PlyData.read(gt_mesh_path)["vertex"].data

    # # sanity checks
    # assert len(pred) == len(gt), f"Vertex count differs: pred={len(pred)} gt={len(gt)}"
    # assert "object_id" in pred.dtype.names, f"pred has no 'object_id' field. Fields: {pred.dtype.names}"
    # assert "object_id" in gt.dtype.names,   f"gt has no 'object_id' field. Fields: {gt.dtype.names}"

    # pred_oid = pred["class_id"].astype(np.int32)
    # gt_oid   = gt["class_id"].astype(np.int32)

    # acc = (pred_oid == gt_oid).mean()
    # print(f"Vertex accuracy (object_id): {acc*100:.2f}% ({(pred_oid==gt_oid).sum()}/{len(gt_oid)})")
    # === END CHECK

    # ==== Process GT mesh and apply object_id from faces to vertices ====
    # ply = PlyData.read(gt_mesh_path)
    # faces = ply["face"].data
    # vtx = ply["vertex"].data

    # print("face properties:", faces.dtype.names)

    # obj = faces["object_id"]          # <-- Replica stores object_id per-face
    # face_vrtx = faces["vertex_indices"]
    # counts = []
    # uniq, cnt = np.unique(obj, return_counts=True)

    # print(f"unique object_ids: {len(uniq)}")
    # print(f"min/max: {int(uniq.min())} {int(uniq.max())}\n")

    # n_verts = len(vtx)
    # votes = [[] for _ in range(n_verts)]

    # for f in faces:
    #     vidx = f["vertex_indices"]   # e.g. [12, 45, 7]
    #     oid  = int(f["object_id"])   # e.g. 23
    #     for v in vidx:
    #         votes[int(v)].append(oid)

    # vertex_object_id = np.full(n_verts, -1, dtype=np.int32)  # -1 = unknown / no faces
    # conflicted = 0

    # from collections import Counter
    # for v in range(n_verts):
    #     if not votes[v]:
    #         continue
    #     c = Counter(votes[v])
    #     # most_common returns [(oid, count), ...]
    #     best_oid, best_count = c.most_common(1)[0]
    #     vertex_object_id[v] = best_oid

    #     if len(c) > 1:
    #         conflicted += 1

    # print(f"Vertices: {n_verts}")
    # print(f"Vertices that saw >1 object_id among incident faces: {conflicted}")

    # old_dtype = vtx.dtype
    # new_dtype = old_dtype.descr + [("object_id", "i4")]

    # verts_out = np.empty(n_verts, dtype=new_dtype)
    # for name in old_dtype.names:
    #     verts_out[name] = vtx[name]
    # verts_out["object_id"] = vertex_object_id

    # vtx_el = PlyElement.describe(verts_out, "vertex")
    # face_el = PlyElement.describe(faces, "face")

    # PlyData([vtx_el, face_el], text=ply.text).write("./data/replica/scan1/mesh_semantic_verts.ply")
    # print("Saved")
    # ==== END Process GT mesh and apply object_id from faces to vertices ====



    # gaussianModel, scene, anchor_points, sem_logits = setup_gaussian_scene_and_model(
    #     model.extract(args), 
    #     args.iteration,
    #     args.checkpoint_path
    #     )
    # logits = sem_logits.cpu().detach().numpy()
    # smoothed_logits = weighted_logit_mean(anchor_points, logits)
    

    # anchor_id = np.arange(anchor_points.shape[0])

    # bg_color = [1,1,1] if model._white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # all_views = get_views(scene, skip_train=args.skip_train, skip_test = args.skip_test)

    # pred_instance_id = np.load("./experiments2_fromsam3/model_d8k/wdist=0.0_wemb=1.0_wsem=0.0_pdist85_pemb78_psem70_512/instance_ids.npy")

    # n_clusters = len(set(pred_instance_id)) - (1 if -1 in pred_instance_id else 0)
    # n_noise = int((pred_instance_id == -1).sum())
    # print(f"{n_clusters=}, {n_noise=}")

    # anchor_points_transformed = apply_full_transform(anchor_points)