import os, glob
import cv2, json, torch
import numpy as np
import colorsys
from ultralytics.models.sam import SAM3SemanticPredictor


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
    if has_noise:
        class_labels = [l for l in uniq if l != noise_label]
    else:
        class_labels = uniq.tolist()

    phi = 0.6180339887498949  # golden ratio conjugate
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


def get_classes():
    with open("info_semantic.json", "r") as f:
        data = json.load(f)
    return [obj["name"].replace("_", " ") for obj in data["classes"]]


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


# ------------------- batch processing -------------------

classes = get_classes()

input_path = "./data/replica/scan1/images/"
output_path = "./data/replica/scan1/2Dclassification_tests/test1"
original_seg_path = "./home/domi/repos/3dgs/GSRec_SemInstSeg/data/replica/scan1/masks_test3/"

os.makedirs(output_path, exist_ok=True)

png_files = sorted(glob.glob(os.path.join(input_path, "*.png")))
if not png_files:
    raise FileNotFoundError(f"No .png files found in: {input_path}")

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

for idx, source in enumerate(png_files):
    im = cv2.imread(source)
    if im is None:
        print(f"[{idx+1}/{len(png_files)}] Skipping (failed to read): {source}")
        continue

    src_shape = im.shape[:2]

    predictor.set_image(source)

    all_masks = []
    all_labels = []

    with torch.inference_mode():
        for cls_name in classes:
            # print(cls_name)
            masks, boxes = predictor.inference_features(
                predictor.features, src_shape=src_shape, text=[cls_name]
            )
            # print("finished predicting")
            if masks is None:
                continue
            all_masks.append(masks)
            all_labels.extend([cls_name] * masks.shape[0])

    base = os.path.splitext(os.path.basename(source))[0]
    out_file = os.path.join(output_path, f"{base}_with_legend.png")

    if not all_masks:
        # Save original + empty legend so every input gets an output
        legend_img = make_legend_image({})
        save_with_side_legend(out_file, im, legend_img)
        print(f"[{idx+1}/{len(png_files)}] {base}: no masks -> saved {out_file}")
        continue

    masks = torch.cat(all_masks, dim=0).cpu().numpy()  # (N,H,W) bool/float

    # unique labels in first-seen order
    unique_labels = []
    seen = set()
    for lab in all_labels:
        if lab not in seen:
            seen.add(lab)
            unique_labels.append(lab)

    # palette per unique label (same behavior as your current code)
    label_ids = np.arange(len(unique_labels), dtype=np.int32)
    rgb01 = contrast_palette2(label_ids)
    rgb255 = (rgb01 * 255.0).round().clip(0, 255).astype(np.uint8)
    bgr255 = rgb255[:, ::-1]

    label_to_bgr = {lab: bgr255[i] for i, lab in enumerate(unique_labels)}
    mask_colors = [tuple(int(x) for x in label_to_bgr[lab]) for lab in all_labels]

    # draw OPAQUE (alpha = 1, no blending)
    out_img = im.copy()
    for i in range(masks.shape[0]):
        m = masks[i]
        if m.dtype != np.bool_:
            m = m > 0.5
        out_img[m] = mask_colors[i]

    legend_img = make_legend_image(label_to_bgr)
    save_with_side_legend(out_file, out_img, legend_img)

    print(f"[{idx}/{len(png_files)}] {base}: FINAL saved -> {out_file}")
