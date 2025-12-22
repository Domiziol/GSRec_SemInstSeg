import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import os
import json
import clip
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import colorsys

# image_path = "image.png"
# image = cv2.imread(image_path)
# overlay = image.copy()
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# output_dir = "masks"
device="cuda" if torch.cuda.is_available() else "cpu"


def init_automatic_sam():
    model_type = "vit_h"
    checkpoint_path = "sam_model/sam_vit_h_4b8939.pth"

    # === Load SAM ===
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model = sam,
        points_per_side=8,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.96,
        crop_n_points_downscale_factor=1
        )
    
    return mask_generator

def get_filtered_masks(mask_generator, image):

    masks = mask_generator.generate(image)
    filtered_masks = []

    for i, m in enumerate(masks):
        seg = np.asarray(m["segmentation"], dtype=bool)
        area = int(seg.sum())
        if area > 300: # and m['predicted_iou'] > 0.9:
            m2 = dict(m)  
            m2["segmentation"] = seg  
            filtered_masks.append(m2)
    
    return filtered_masks


def get_mask_img_bbox(mask, image):

    img_np = np.asarray(image)
    
    rows, cols = np.where(mask["segmentation"])
    y_min = rows.min()
    y_max = rows.max()
    x_min = cols.min()
    x_max = cols.max()

    cropped_box = [x_min, y_min, x_max, y_max]

    segmented_img_np = np.zeros_like(img_np)
    segmented_img_np[mask["segmentation"]] = img_np[mask["segmentation"]]
    segmented_image = Image.fromarray(segmented_img_np)

    cropped_box_img = segmented_image.crop(cropped_box)

    # out_path = os.path.join(output_dir, f"mask_{i:03d}.png")
    # segmented_image.save(out_path)
    
    return cropped_box_img

def get_classes():
    classes= []

    with open("info_semantic.json", 'r') as classes_file:
        data = json.load(classes_file)

    for objects in data['classes']:
        classes.append(objects['name'])

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)

    return classes 



# def prepare_clip(classes):
#     class_prompts = [f"a photo of a {c}" for c in classes]
    
#     model, preprocess = clip.load("ViT-B/16", device="cpu")  
#     with torch.inference_mode():
#         tokens = clip.tokenize(class_prompts)               
#         text_feat = model.encode_text(tokens)               
#         text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
#     return model, preprocess, class_prompts, text_feat 

def prepare_clip(classes, device="cpu"):
    templates = [
        "a photo of a {}",
        "a photo of the {}",
        "a close-up photo of a {}",
        "a cropped photo of a {}",
        "a photo of one {}",
        "a photo of {}",
        "a blurry photo of a {}",
        "a low resolution photo of a {}",
    ]

    # keep prompts if you want to inspect/debug
    class_prompts = {c: [t.format(c) for t in templates] for c in classes}

    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    with torch.inference_mode():
        text_feats = []
        for c in classes:
            tokens = clip.tokenize(class_prompts[c]).to(device)
            tf = model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            tf = tf.mean(dim=0, keepdim=True)              # average templates
            tf = tf / tf.norm(dim=-1, keepdim=True)
            text_feats.append(tf)

        text_feat = torch.cat(text_feats, dim=0)           # (C, D)

    return model, preprocess, class_prompts, text_feat


def topk_clip_for_crop(cropped_img_pil, clip_model, clip_preprocess, text_feat, classes, k=3):
    # preprocess once
    with torch.inference_mode():
        img = clip_preprocess(cropped_img_pil).unsqueeze(0)      
        img_feat = clip_model.encode_image(img)                  
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True) 

        # cosine similarities via dot product
        # sims_tens = (img_feat @ text_feat.T)
        # probs = sims_to_probs(sims_tens, clip_model, True, None)    
        sims = (img_feat @ text_feat.T).cpu().numpy().ravel()

    # top-k
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    top_classes = [classes[i] for i in idx]
    top_scores  = sims[idx].tolist()
    return top_classes, top_scores



def crop_bbox_from_mask(mask, image_pil):
    img_np = np.asarray(image_pil)        
    rows, cols = np.nonzero(mask)
    if rows.size == 0 or cols.size == 0:
        return image_pil  # check

    y0, y1 = rows.min(), rows.max()+1
    x0, x1 = cols.min(), cols.max()+1

    crop = img_np[y0:y1, x0:x1].copy()    
    
    local_mask = mask[y0:y1, x0:x1]
    bg = np.zeros_like(crop)
    crop = np.where(local_mask[..., None], crop, bg)

    return Image.fromarray(crop)

def find_mask_for_point(masks, u, v):
    for m in masks:
        x, y, w, h = m["bbox"]
        if not (x <= u < x+w and y <= v < y+h):
            continue
        seg = m["segmentation"]
        if seg[int(v), int(u)]:
            return m
    return None



from pathlib import Path
def preprocess_images_main():
    input_path = "./data/replica/scan1/images/"
    output_path = "./data/replica/scan1/masks_real2"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    predictor = init_automatic_sam()
    classes = get_classes()
    class_to_idx = {n:i for i,n in enumerate(classes)}
    clip_model, clip_preprocess, class_prompts, text_feat = prepare_clip(classes)

    for file in sorted(os.listdir(input_path)):
        if file.lower().endswith((".png", ".jpg")):
            img = Image.open(os.path.join(input_path, file)).convert("RGB")
            img_np = np.array(img)

            masks = get_filtered_masks(predictor, img_np)
            
            H0, W0 = img_np.shape[:2]
            if len(masks) == 0:
                out = Path(output_path) / (Path(file).stem + ".npz")
                np.savez_compressed(
                    out,
                    masks=np.zeros((0, H0, W0), dtype=np.uint8),
                    labels=np.zeros((0,), dtype=np.int16),
                    scores=np.zeros((0,), dtype=np.float32),
                    image_size=np.array([H0, W0], dtype=np.int32),
                    version=np.int32(1),
                )
                print(f"{file}: no masks after filtering, saved empty npz")
                continue

            check_masks = []
            for i, m in enumerate(masks):
                seg = np.asarray(m["segmentation"], dtype = bool)
                a = int(seg.sum())
                if a != 0:
                    check_masks.append(seg)
            masks_bool = np.stack(check_masks, axis=0)        

            labels_idx = []
            scores = [] # cosine similarities
            for m in masks_bool:  
                # crop = crop_bbox_from_mask(mask["segmentation"], img)
                # crop = crop_bbox_from_mask(mask, img)
                # top_classes, top_scores = topk_clip_for_crop(
                #     crop, clip_model, clip_preprocess, text_feat, classes, k=1) # if k=1, the result would be for only one, best class
            
            
                rows, cols = np.nonzero(m)
                y0, y1 = rows.min(), rows.max() + 1
                x0, x1 = cols.min(), cols.max() + 1

                # add small margin
                margin_y = max(2, int(0.10 * (y1 - y0)))
                margin_x = max(2, int(0.10 * (x1 - x0)))

                y0 = max(0, y0 - margin_y)
                y1 = min(H0, y1 + margin_y)
                x0 = max(0, x0 - margin_x)
                x1 = min(W0, x1 + margin_x)

                crop_rgb = np.array(img)[y0:y1, x0:x1].copy()  # keep original background
                crop_pil = Image.fromarray(crop_rgb)

                top_classes, top_scores = topk_clip_for_crop(
                    crop_pil, clip_model, clip_preprocess, text_feat, classes, k=1
                )
                
                idx = class_to_idx[top_classes[0]]
                labels_idx.append(idx)
                scores.append(top_scores[0])

            labels_array = np.asarray(labels_idx, dtype = np.int16)
            scores_array = np.asarray(scores, dtype = np.float32)

            np.savez(f"{output_path}/{file[:-4]}.npz",
                     masks = masks_bool,
                     labels = labels_array,
                     scores = scores_array,
                     version = np.int16(1))



def class_palette(classes, s=0.65, v=0.95):
    n = max(1, len(classes))
    cols = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((int(r*255), int(g*255), int(b*255)))
    return cols  # list of RGB tuples

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
        gray = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        table[labels == noise_label] = gray

    return table

def overlay_masks(img_np, masks_bool, labels_idx, scores, classes):
    H, W = img_np.shape[:2]
    overlay = img_np.copy()

    # build palette for class indices 0..C-1
    C = len(classes)
    palette = (contrast_palette2(np.arange(C)) * 255).astype(np.uint8)  # (C,3)

    for j in range(masks_bool.shape[0]):
        cls_id = int(labels_idx[j])
        if cls_id < 0 or cls_id >= C:
            continue
        color = palette[cls_id]  # (3,)
        m = masks_bool[j]
        if m.sum() == 0:
            continue
        overlay[m] = color

    vis = Image.fromarray(overlay)
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    y = 5
    used = set()
    for j in range(masks_bool.shape[0]):
        cls_id = int(labels_idx[j])
        if cls_id in used or cls_id < 0 or cls_id >= C:
            continue
        used.add(cls_id)

        color = tuple(int(x) for x in palette[cls_id])  # legend uses SAME palette
        cls_name = classes[cls_id]

        draw.rectangle([5, y+2, 17, y+14], fill=color, outline=color)
        draw.text((22, y), cls_name, fill=(255, 255, 255), font=font)
        y += 18
        if y > H - 20:
            break

    return vis

def save_mask_crops(img_pil, masks_bool, labels_idx, scores, classes, crops_dir, limit=None):
    crops_dir.mkdir(parents=True, exist_ok=True)
    H, W = masks_bool.shape[1:]
    
    order = np.arange(masks_bool.shape[0])
    # if limit is not None and masks_bool.shape[0] > limit:
    #     # pick the largest masks by area
    #     areas = masks_bool.reshape(masks_bool.shape[0], -1).sum(axis=1)
    #     order = np.argsort(-areas)[:limit]

    for j in order:
        m = masks_bool[j]
        if m.sum() == 0:
            continue
        rows, cols = np.nonzero(m)
        y0, y1 = rows.min(), rows.max()+1
        x0, x1 = cols.min(), cols.max()+1
        crop_rgb = np.array(img_pil)[y0:y1, x0:x1].copy()
        local = m[y0:y1, x0:x1]
        bg = np.zeros_like(crop_rgb)
        crop_rgb = np.where(local[..., None], crop_rgb, bg)
        crop = Image.fromarray(crop_rgb)
        
        # rows, cols = np.nonzero(m)

        # y0, y1 = rows.min(), rows.max() + 1
        # x0, x1 = cols.min(), cols.max() + 1

        # margin_y = max(2, int(0.10 * (y1 - y0)))
        # margin_x = max(2, int(0.10 * (x1 - x0)))

        # y0 = max(0, y0 - margin_y)
        # y1 = min(H, y1 + margin_y)
        # x0 = max(0, x0 - margin_x)
        # x1 = min(W, x1 + margin_x)

        # crop_rgb = np.array(img_pil)[y0:y1, x0:x1].copy()  
        # crop = Image.fromarray(crop_rgb)

        cls_id = int(labels_idx[j])
        cls_name = classes[cls_id]
        sc = float(scores[j]) if isinstance(scores, (list, np.ndarray)) else float(scores)
        fname = f"mask_{j:03d}_{cls_name}_{sc:.2f}.png"
        crop.save(crops_dir / fname)


def overlay_sam_masks(image, masks, alpha=1.0, out_path="masks_overlay.png"):
    
    if isinstance(masks, list):
        masks_arr = np.stack(masks, axis=0)  # (N, H, W)
    else:
        masks_arr = masks  # assume already (N, H, W)

    overlay = image.copy().astype(np.float32)

    H, W = image.shape[:2]
    N = masks_arr.shape[0]

    # random colors for each mask
    colors = np.random.randint(0, 255, size=(N, 3), dtype=np.uint8)

    for i in range(N):
        mask = masks_arr[i]
        color = colors[i]

        # broadcast color to image shape
        color_img = np.zeros_like(overlay)
        color_img[mask] = color

        # alpha blend where mask is True
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color_img[mask]

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, overlay[:, :, ::-1])


def preprocess_images_main_test():
    input_path = "./data/replica/scan1/images/"
    output_path = "./data/replica/scan1/2Dclassification_tests/test1"
    # sam_masks_output_path = "./data/replica/scan1/masksonly1/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    vis_dir = Path(output_path) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    crops_root = Path(output_path) / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)

    predictor = init_automatic_sam()
    classes = get_classes()
    class_to_idx = {n:i for i,n in enumerate(classes)}
    clip_model, clip_preprocess, class_prompts, text_feat = prepare_clip(classes)

    for file in sorted(os.listdir(input_path)):
        if not file.lower().endswith((".png", ".jpg")):
            continue
            
        img_path = os.path.join(input_path, file)
        stem = Path(file).stem
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)

        
        masks = get_filtered_masks(predictor, img_np)
        H0, W0 = img_np.shape[:2]
        if len(masks) == 0:
            out = Path(output_path) / f"{stem}.npz"
            np.savez_compressed(
                out,
                masks=np.zeros((0, H0, W0), dtype=np.uint8),
                labels=np.zeros((0,), dtype=np.int16),
                scores=np.zeros((0,), dtype=np.float32),
                image_size=np.array([H0, W0], dtype=np.int32),
                version=np.int32(1),
            )
            
            img_pil.save(vis_dir / f"{stem}_overlay.png")
            print(f"{file}: no masks after filtering → saved empty npz + overlay")
            continue

        
        masks_bool = []
        for m in masks:
            seg = np.asarray(m["segmentation"], dtype=bool)
            if seg.sum() > 0:
                masks_bool.append(seg)
        if len(masks_bool) == 0:
            masks_bool = np.zeros((0, H0, W0), dtype=bool)
        else:
            masks_bool = np.stack(masks_bool, axis=0)

        # name = stem + ".jpg"
        # overlay_sam_masks(img_np, masks_bool, alpha=0.5, out_path=os.path.join(sam_masks_output_path,name))

        
        labels_idx = []
        scores = []
        for j in range(masks_bool.shape[0]):
            m = masks_bool[j]
            # crop masked region
            # rows, cols = np.nonzero(m)
            # y0, y1 = rows.min(), rows.max()+1
            # x0, x1 = cols.min(), cols.max()+1
            # crop_rgb = img_np[y0:y1, x0:x1].copy()
            # local = m[y0:y1, x0:x1]
            # crop_rgb = np.where(local[..., None], crop_rgb, 0)
            # crop_pil = Image.fromarray(crop_rgb)
            rows, cols = np.nonzero(m)

            y0, y1 = rows.min(), rows.max() + 1
            x0, x1 = cols.min(), cols.max() + 1

            margin_y = max(2, int(0.10 * (y1 - y0)))
            margin_x = max(2, int(0.10 * (x1 - x0)))

            y0 = max(0, y0 - margin_y)
            y1 = min(H0, y1 + margin_y)
            x0 = max(0, x0 - margin_x)
            x1 = min(W0, x1 + margin_x)

            crop_rgb = np.array(img_pil)[y0:y1, x0:x1].copy()
            crop_pil = Image.fromarray(crop_rgb)

            top_classes, top_scores = topk_clip_for_crop(
                crop_pil, clip_model, clip_preprocess, text_feat, classes, k=1
            )
            if len(top_classes) == 0:
                labels_idx.append(0)
                scores.append(0.0)
            else:
                labels_idx.append(class_to_idx[top_classes[0]])
                scores.append(float(top_scores[0]))

        labels_idx = np.asarray(labels_idx, dtype=np.int16)
        scores_arr = np.asarray(scores, dtype=np.float32)
        masks_u8 = masks_bool.astype(np.uint8)

        
        out = Path(output_path) / f"{stem}.npz"
        np.savez_compressed(
            out,
            masks=masks_u8,
            labels=labels_idx,
            scores=scores_arr,
            image_size=np.array([H0, W0], dtype=np.int32),
            version=np.int32(1),
        )

        
        vis = overlay_masks(img_np, masks_bool, labels_idx, scores_arr, classes)
        vis.save(vis_dir / f"{stem}_overlay.png")

        save_mask_crops(img_pil, masks_bool, labels_idx, scores_arr, classes, crops_root / stem, limit=None)


def preprocess_images_sam_text_prompt_test():
    input_path = "./data/replica/scan1/images/"
    output_path = "./data/replica/scan1/2Dclassification_tests/test1"
    # sam_masks_output_path = "./data/replica/scan1/masksonly1/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    vis_dir = Path(output_path) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    crops_root = Path(output_path) / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)

    predictor = init_automatic_sam()
    classes = get_classes()
    class_to_idx = {n:i for i,n in enumerate(classes)}
    idx_to_class = {i:n for i,n in enumerate(classes)}
    clip_model, clip_preprocess, class_prompts, text_feat = prepare_clip(classes)

    files = sorted(os.listdir(input_path))
    idx = list(range(0,5)) + list(range(150,155)) + list(range(200,205)) + list(range(300,305))+ list(range(380,385))
    for file in (files[i] for i in idx):
        if not file.lower().endswith((".png", ".jpg")):
            continue
            
        img_path = os.path.join(input_path, file)
        stem = Path(file).stem
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)

        
        masks = get_filtered_masks(predictor, img_np)
        H0, W0 = img_np.shape[:2]
        if len(masks) == 0:
            out = Path(output_path) / f"{stem}.npz"
            np.savez_compressed(
                out,
                masks=np.zeros((0, H0, W0), dtype=np.uint8),
                labels=np.zeros((0,), dtype=np.int16),
                scores=np.zeros((0,), dtype=np.float32),
                image_size=np.array([H0, W0], dtype=np.int32),
                version=np.int32(1),
            )
            
            img_pil.save(vis_dir / f"{stem}_overlay.png")
            print(f"{file}: no masks after filtering → saved empty npz + overlay")
            continue

        
        masks_bool = []
        for m in masks:
            seg = np.asarray(m["segmentation"], dtype=bool)
            if seg.sum() > 0:
                masks_bool.append(seg)
        if len(masks_bool) == 0:
            masks_bool = np.zeros((0, H0, W0), dtype=bool)
        else:
            masks_bool = np.stack(masks_bool, axis=0)

        # name = stem + ".jpg"
        # overlay_sam_masks(img_np, masks_bool, alpha=0.5, out_path=os.path.join(sam_masks_output_path,name))

        labels_idx = []
        scores = []
        blurred_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=12))
        for j in range(masks_bool.shape[0]):
            m = masks_bool[j]
            # crop masked region
            # rows, cols = np.nonzero(m)
            # y0, y1 = rows.min(), rows.max()+1
            # x0, x1 = cols.min(), cols.max()+1
            # crop_rgb = img_np[y0:y1, x0:x1].copy()
            # local = m[y0:y1, x0:x1]
            # crop_rgb = np.where(local[..., None], crop_rgb, 0)
            # crop_pil = Image.fromarray(crop_rgb)
            rows, cols = np.nonzero(m)
            if rows.size == 0 or cols.size == 0:
                continue

            y0, y1 = rows.min(), rows.max() + 1
            x0, x1 = cols.min(), cols.max() + 1

            # large margin (pick one style)
            margin_y = max(40, int(0.35 * (y1 - y0)))
            margin_x = max(40, int(0.35 * (x1 - x0)))

            y0 = max(0, y0 - margin_y)
            y1 = min(H0, y1 + margin_y)
            x0 = max(0, x0 - margin_x)
            x1 = min(W0, x1 + margin_x)

            # context crop from ORIGINAL image (no masking / no black bg)
            crop_pil = img_pil.crop((x0, y0, x1, y1))

            crop_dir = crops_root / stem
            crop_dir.mkdir(parents=True, exist_ok=True)

            # crop_pil.save(crops_root / stem / f"mask_{j:03d}_blur.png")

            top_classes, top_scores = topk_clip_for_crop(
                crop_pil, clip_model, clip_preprocess, text_feat, classes, k=1
            )
            if len(top_classes) == 0:
                labels_idx.append(0)
                scores.append(0.0)
            else:
                labels_idx.append(class_to_idx[top_classes[0]])
                scores.append(float(top_scores[0]))

            crop_pil.save(crops_root / stem / f"mask_{j:03d}_crop_{top_classes[0]}.png")

        labels_idx = np.asarray(labels_idx, dtype=np.int16)
        scores_arr = np.asarray(scores, dtype=np.float32)
        masks_u8 = masks_bool.astype(np.uint8)

        
        out = Path(output_path) / f"{stem}.npz"
        np.savez_compressed(
            out,
            masks=masks_u8,
            labels=labels_idx,
            scores=scores_arr,
            image_size=np.array([H0, W0], dtype=np.int32),
            version=np.int32(1),
        )

        
        vis = overlay_masks(img_np, masks_bool, labels_idx, scores_arr, classes)
        vis.save(vis_dir / f"{stem}_overlay.png")

        # save_mask_crops(img_pil, masks_bool, labels_idx, scores_arr, classes, crops_root / stem, limit=None)
        

if __name__ == '__main__':
    # preprocess_images_main()
    # preprocess_images_main_test()
    preprocess_images_sam_text_prompt_test()





