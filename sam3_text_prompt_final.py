import os, glob, json
import cv2
import torch
import numpy as np
import colorsys
from ultralytics.models.sam import SAM3SemanticPredictor

import re

def safe_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "", s)
    return s[:80] if len(s) > 80 else s

# ---------------- palette ----------------
def contrast_palette2(
    labels,
    s_range=(0.55, 0.95),
    v_range=(0.65, 1.0),
    base_hue=0.13,
    noise_label=-1,
):
    labels = np.asarray(labels)
    uniq = np.unique(labels)

    has_noise = noise_label in uniq
    class_labels = [l for l in uniq if l != noise_label] if has_noise else uniq.tolist()

    phi = 0.6180339887498949
    table = np.zeros((len(labels), 3), dtype=np.float32)
    label_to_idx = {lab: i for i, lab in enumerate(class_labels)}

    sv_patterns = [
        (s_range[1], v_range[1]),
        (s_range[1], v_range[0]),
        (s_range[0], v_range[1]),
        (s_range[0], v_range[0]),
    ]

    for lab in class_labels:
        k = label_to_idx[lab]
        h = (base_hue + k * phi) % 1.0
        s, v = sv_patterns[k % len(sv_patterns)]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        table[labels == lab] = (r, g, b)

    if has_noise:
        table[labels == noise_label] = np.array([0.6, 0.6, 0.6], dtype=np.float32)

    return table

def get_fixed_colors():
    # older
    # colors = [
    #     (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    #     (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    #     (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    #     (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    #     (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    #     (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    #     (255, 0, 255), (0, 255, 255), (192, 192, 192), (128, 0, 128),
    #     (128, 64, 0), (64, 128, 0), (0, 128, 64), (0, 64, 128),
    #     (64, 0, 128), (128, 0, 64), (200, 0, 0), (0, 200, 0),
    #     (0, 0, 200), (200, 200, 0), (200, 0, 200), (0, 200, 200),
    #     (100, 0, 0), (0, 100, 0), (0, 0, 100), (100, 100, 0),
    #     (100, 0, 100), (0, 100, 100), (150, 50, 50), (50, 150, 50),
    #     (50, 50, 150), (150, 150, 50), (150, 50, 150), (50, 150, 150),
    #     (220, 100, 0), (0, 220, 100), (100, 0, 220), (220, 0, 100),
    #     (100, 220, 0), (0, 100, 220), (160, 80, 0), (0, 160, 80),
    #     (80, 0, 160), (160, 0, 80), (80, 160, 0), (0, 80, 160),
    #     (255, 128, 0), (128, 255, 0), (0, 255, 128), (0, 128, 255),
    #     (128, 0, 255), (255, 0, 128), (200, 100, 50), (50, 200, 100),
    #     (100, 50, 200), (200, 50, 100), (100, 200, 50), (50, 100, 200),
    #     (255, 100, 100), (100, 255, 100), (100, 100, 255),
    #     (255, 255, 100), (255, 100, 255), (100, 255, 255),
    #     (40, 40, 120), (120, 40, 40), (40, 120, 40),
    #     (180, 60, 60), (60, 180, 60), (60, 60, 180),
    #     (180, 180, 60), (180, 60, 180), (60, 180, 180),
    #     (90, 30, 30), (30, 90, 30), (30, 30, 90),
    #     (90, 90, 30), (90, 30, 90)
    # ]
    colors = (
    (242,73,73),(109,127,248),(106, 45, 110),(73,155,242),(242,142,73),
    (155,73,242),(73,242,242),(242,73,203),(203,242,73),(242,160,203),
    (244,108,129),(203,191,242),(191,142,73),(242,236,191),(128,38,38),
    (191,242,203),(142,142,38),(59, 232, 245),(38,38,128),(210,137,91),
    (242,38,38),(38,242,38),(38,38,242),(242,242,38),(242,38,242),
    (38,242,242),(191,191,191),(128,38,128),(202,191,155),(83,128,38),
    (38,128,83),(38,83,128),(83,38,128),(128,38,83),(191,38,38),
    (38,191,38),(38,38,191),(191,191,38),(191,38,191),(38,191,191),
    (102,38,38),(38,102,38),(38,38,102),(102,102,38),(102,38,102),
    (38,102,102),(157,17,16),(83,166,83),(83,83,166),(166,166,83),
    (166,83,166),(83,166,166),(204,121,38),(38,204,121),(121,38,204),
    (204,38,121),(121,204,38),(38,121,204),(84, 46, 12),(38,153,96),
    (96,38,153),(153,38,96),(96,153,38),(38,96,153),(100, 76, 209),
    (162,242,38),(38,242,162),(38,162,242),(162,38,242),(242,38,162),
    (204,142,83),(83,204,142),(142,83,204),(204,83,142),(203,235,193),
    (83,142,204),(242,121,121),(121,242,121),(69,247,172),(242,242,121),
    (242,121,242),(121,242,242),(64,64,140), (140,64,64),(64,140,64),
    (191,96,96),(96,191,96),(96,96,191),(191,191,96),(191,96,191),
    (3,97,104),(77,38,38),(38,77,38),(38,38,77),(77,77,38),
    (77,38,77), (242,74,190), (182,145,212), (199,206,110), (245,134,71),
    (27,253,70)
)
    return colors

# ---------------- IO helpers ----------------
def get_classes_and_ids(json_path="info_semantic.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    names = [obj["name"].replace("_", " ") for obj in data["classes"]]
    name_to_id = {name: i for i, name in enumerate(names)}  # label_id = index in JSON order
    return names, name_to_id

def load_orig_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    masks = z["masks.npy"]  # <-- your case
    if masks.dtype != np.bool_:
        masks = masks > 0   # works for 0/1 and 0/255
    return masks

def make_legend_image(label_to_bgr, box_size=18, pad=10, row_h=26, font_scale=0.6, thickness=1):
    labels = list(label_to_bgr.keys())
    if not labels:
        return np.full((60, 260, 3), 255, dtype=np.uint8)

    max_len = max(len(s) for s in labels)
    w = max(320, pad * 2 + box_size + 12 + int(max_len * 12))
    h = pad * 2 + row_h * len(labels)

    legend = np.full((h, w, 3), 255, dtype=np.uint8)
    y = pad + row_h // 2

    for lab in labels:
        bgr = tuple(int(x) for x in label_to_bgr[lab])
        x0 = pad
        y0 = y - box_size // 2
        cv2.rectangle(legend, (x0, y0), (x0 + box_size, y0 + box_size), bgr, -1)
        cv2.rectangle(legend, (x0, y0), (x0 + box_size, y0 + box_size), (0, 0, 0), 1)
        cv2.putText(
            legend, lab,
            (x0 + box_size + 12, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (0, 0, 0), thickness, cv2.LINE_AA
        )
        y += row_h

    return legend

def save_with_side_legend(out_path, result_bgr, legend_bgr, gap=16, bg=255):
    h1, w1 = result_bgr.shape[:2]
    h2, w2 = legend_bgr.shape[:2]

    if h2 != h1:
        new_w2 = int(round(w2 * (h1 / h2)))
        legend_bgr = cv2.resize(legend_bgr, (new_w2, h1), interpolation=cv2.INTER_NEAREST)
        h2, w2 = legend_bgr.shape[:2]

    canvas = np.full((h1, w1 + gap + w2, 3), bg, dtype=np.uint8)
    canvas[:, :w1] = result_bgr
    canvas[:, w1 + gap : w1 + gap + w2] = legend_bgr
    cv2.imwrite(out_path, canvas)

# ---------------- overlap resolution ----------------
def build_conflict_groups(masks_bool, significant_overlap=300):
    """
    Return list of connected components (list of lists of indices),
    where edges exist if intersection > significant_overlap.
    """
    n = masks_bool.shape[0]
    if n == 0:
        return []

    adj = [[] for _ in range(n)]
    for i in range(n):
        mi = masks_bool[i]
        for j in range(i + 1, n):
            inter = np.logical_and(mi, masks_bool[j]).sum()
            if inter > significant_overlap:
                adj[i].append(j)
                adj[j].append(i)

    seen = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps

def resolve_overlaps_drop_losers(masks_bool, scores, areas, significant_overlap=300):
    """
    Only resolve 'significant' overlap groups by keeping best-scoring mask per group.
    Small overlaps (<= significant_overlap) are allowed.
    Returns indices to keep.
    """
    n = masks_bool.shape[0]
    if n == 0:
        return []

    keep = np.ones(n, dtype=bool)

    comps = build_conflict_groups(masks_bool, significant_overlap=significant_overlap)
    for comp in comps:
        if len(comp) <= 1:
            continue
        best = max(comp, key=lambda i: (scores[i], areas[i]))
        for i in comp:
            if i != best:
                keep[i] = False

    return [i for i in range(n) if keep[i]]

def score_sam3_masks_against_orig(masks_bool, orig_masks_bool):
    """
    score_i = max_j sum(m_i & orig_j)
    """
    n = masks_bool.shape[0]
    scores = np.zeros(n, dtype=np.int32)

    if orig_masks_bool is None or orig_masks_bool.shape[0] == 0:
        return scores

    for i in range(n):
        mi = masks_bool[i]
        best = 0
        for j in range(orig_masks_bool.shape[0]):
            inter = np.logical_and(mi, orig_masks_bool[j]).sum()
            if inter > best:
                best = inter
        scores[i] = best
    return scores

# ---------------- main batch ----------------
classes, name_to_id = get_classes_and_ids("info_semantic.json")

input_path = "./data/replica/scan1/images/"
output_path = "./data/replica/scan1/2Dclassification_tests/test3"
# original_seg_path = "./data/replica/scan1/masks_test3/"
original_seg_path = "./data/replica/scan1/2Dclassification_tests/test3/"

# NEW: results + check folders
results_dir = os.path.join(output_path, "results")
check_dir   = os.path.join(results_dir, "check")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(check_dir, exist_ok=True)

png_files = sorted(glob.glob(os.path.join(input_path, "*.png")))
if not png_files:
    raise FileNotFoundError(f"No .png files found in: {input_path}")

SIGNIFICANT_OVERLAP = 100

overrides = dict(
    conf=0.50,
    task="segment",
    mode="predict",
    model="sam_model/sam3.pt",
    verbose=False,
    half=True,
    imgsz=644,
)

predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.setup_model()

for idx, source in enumerate(png_files, start=1):
    base = os.path.splitext(os.path.basename(source))[0]

    # FINAL outputs now go into results/
    out_png = os.path.join(results_dir, f"{base}_with_legend.png")
    out_npz = os.path.join(results_dir, f"{base}.npz")

    im = cv2.imread(source)
    if im is None:
        print(f"[{idx}/{len(png_files)}] Skipping unreadable: {source}")
        continue
    src_shape = im.shape[:2]

    # load original segmentation masks
    orig_npz = os.path.join(original_seg_path, f"{base}.npz")
    if not os.path.exists(orig_npz):
        print(f"[{idx}/{len(png_files)}] Missing orig npz: {orig_npz} (saving empty)")
        np.savez_compressed(out_npz, **{
            "masks.npy": np.zeros((0, *src_shape), np.uint8),
            "labels.npy": np.zeros((0,), np.int32)
        })
        save_with_side_legend(out_png, im, make_legend_image({}))
        continue
    orig_masks = load_orig_npz(orig_npz)  # (M,H,W) bool

    predictor.set_image(source)

    all_masks = []
    all_label_names = []

    with torch.inference_mode():
        for cls_name in classes:
            masks, boxes = predictor.inference_features(
                predictor.features, src_shape=src_shape, text=[cls_name]
            )
            if masks is None:
                continue
            all_masks.append(masks)
            all_label_names.extend([cls_name] * masks.shape[0])

    if not all_masks:
        np.savez_compressed(out_npz, **{
            "masks.npy": np.zeros((0, *src_shape), np.uint8),
            "labels.npy": np.zeros((0,), np.int32)
        })
        save_with_side_legend(out_png, im, make_legend_image({}))
        print(f"[{idx}/{len(png_files)}] {base}: no SAM3 masks")
        continue

    masks = torch.cat(all_masks, dim=0).cpu().numpy()
    masks_bool = masks if masks.dtype == np.bool_ else (masks > 0.5)

    # --------- NEW: save single-mask images BEFORE overlap resolution ----------
    singles_before_dir = os.path.join(check_dir, f"{base}_single_before")
    os.makedirs(singles_before_dir, exist_ok=True)

    unique_labels_before = []
    seen_before = set()
    for lab in all_label_names:
        if lab not in seen_before:
            seen_before.add(lab)
            unique_labels_before.append(lab)

    label_ids_before = np.arange(len(unique_labels_before), dtype=np.int32)
    rgb01_before = contrast_palette2(label_ids_before)
    rgb255_before = (rgb01_before * 255.0).round().clip(0, 255).astype(np.uint8)
    bgr255_before = rgb255_before[:, ::-1]
    label_to_bgr_before = {lab: bgr255_before[i] for i, lab in enumerate(unique_labels_before)}

    for k in range(masks_bool.shape[0]):
        m = masks_bool[k]
        lab = all_label_names[k]
        color = tuple(int(x) for x in label_to_bgr_before.get(lab, np.array([0, 255, 0], np.uint8)))

        one = im.copy()
        one[m] = color
        legend_one = make_legend_image({lab: np.array(color, dtype=np.uint8)})

        out_one = os.path.join(singles_before_dir, f"{base}_mask{k:03d}_{safe_name(lab)}.png")
        save_with_side_legend(out_one, one, legend_one)
    # -------------------------------------------------------------------------

    # label ids for SAM3 masks
    sam3_label_ids = np.array([name_to_id[n] for n in all_label_names], dtype=np.int32)

    # score each SAM3 mask against original segmentation masks
    scores = score_sam3_masks_against_orig(masks_bool, orig_masks)
    areas = masks_bool.reshape(masks_bool.shape[0], -1).sum(axis=1).astype(np.int32)

    # drop losers to eliminate overlaps
    keep_idx = resolve_overlaps_drop_losers(
        masks_bool=masks_bool,
        scores=scores,
        areas=areas,
        significant_overlap=SIGNIFICANT_OVERLAP
    )

    final_masks_bool = masks_bool[keep_idx]
    final_labels_ids = sam3_label_ids[keep_idx]
    final_label_names = [all_label_names[i] for i in keep_idx]

    # ---- save NPZ (final winners) ----
    final_masks_u8 = final_masks_bool.astype(np.uint8)  # 0/1
    np.savez_compressed(out_npz, **{
        "masks": final_masks_u8,
        "labels": final_labels_ids
    })

    # ---- FINAL visualization (opaque colors + side legend) ----
    unique_labels = []
    seen = set()
    for lab in final_label_names:
        if lab not in seen:
            seen.add(lab)
            unique_labels.append(lab)

    label_ids_vis = np.arange(len(unique_labels), dtype=np.int32)
    rgb01 = contrast_palette2(label_ids_vis)
    rgb255 = (rgb01 * 255.0).round().clip(0, 255).astype(np.uint8)
    bgr255 = rgb255[:, ::-1]

    label_to_bgr = {lab: bgr255[i] for i, lab in enumerate(unique_labels)}
    mask_colors = [tuple(int(x) for x in label_to_bgr[lab]) for lab in final_label_names]

    out_img = im.copy()
    for i in range(final_masks_bool.shape[0]):
        out_img[final_masks_bool[i]] = mask_colors[i]  # alpha=1, no blending

    legend_img = make_legend_image(label_to_bgr)
    save_with_side_legend(out_png, out_img, legend_img)

    print(f"[{idx}/{len(png_files)}] {base}: kept {len(keep_idx)}/{masks_bool.shape[0]} | saved {out_npz} + {out_png}")

    # --------- CHANGED: save AFTER singles into results/check/<base>_single_after ----------
    singles_dir = os.path.join(check_dir, f"{base}_single_after")
    os.makedirs(singles_dir, exist_ok=True)

    for k in range(final_masks_bool.shape[0]):
        m = final_masks_bool[k]
        lab = final_label_names[k]

        if lab in label_to_bgr:
            color = tuple(int(x) for x in label_to_bgr[lab])
        else:
            color = (0, 255, 0)

        one = im.copy()
        one[m] = color  # alpha=1, opaque

        legend_one = make_legend_image({lab: np.array(color, dtype=np.uint8)})

        out_one = os.path.join(singles_dir, f"{base}_mask{k:03d}_{safe_name(lab)}.png")
        save_with_side_legend(out_one, one, legend_one)

    print(f"[{idx}/{len(png_files)}] {base}: saved {final_masks_bool.shape[0]} single-mask AFTER images -> {singles_dir}")
