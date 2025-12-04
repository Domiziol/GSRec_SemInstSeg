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
from PIL import Image, ImageDraw, ImageFont

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



def prepare_clip(classes):
    class_prompts = [f"a photo of a {c}" for c in classes]
    model, preprocess = clip.load("ViT-B/32", device="cpu")  
    with torch.inference_mode():
        tokens = clip.tokenize(class_prompts)               
        text_feat = model.encode_text(tokens)               
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
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

def overlay_masks(img_np, masks_bool, labels_idx, scores, classes):
    
    H, W = img_np.shape[:2]
    overlay = img_np.copy()

    palette = class_palette(classes)
    # alpha = 0.45 
    alpha = 1


    for j in range(masks_bool.shape[0]):
        cls_id = int(labels_idx[j])
        color = np.array(palette[cls_id], dtype=np.uint8)
        m = masks_bool[j]  
        if m.sum() == 0:
            continue
        
        overlay[m] = (alpha*color + (1-alpha)*overlay[m]).astype(np.uint8)

    # draw legend
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
        cls_name = classes[cls_id]
        if cls_id in used:
            continue
        used.add(cls_id)
        color = class_palette(classes)[cls_id]
        text = f"{cls_name}"
        
        draw.rectangle([5, y+2, 17, y+14], fill=color, outline=color)
        draw.text((22, y), text, fill=(255,255,255), font=font)
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


def overlay_sam_masks(image, masks, alpha=0.5, out_path="masks_overlay.png"):
    
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
    input_path = "./data/kitchen_static/colmap_my/rgb_images/"
    output_path = "./data/kitchen_static/masks_test3/"
    sam_masks_output_path = "./data/replica/scan1/masksonly1/"
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

if __name__ == '__main__':
    # preprocess_images_main()
    preprocess_images_main_test()





