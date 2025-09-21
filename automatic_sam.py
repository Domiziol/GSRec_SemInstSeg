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
        area = np.sum(m["segmentation"])
        if area > 300: # and m['predicted_iou'] > 0.9:
            filtered_masks.append(m)
    
    return filtered_masks

# def init_clip():
#     classes= []

#     with open("info_semantic.json", 'r') as classes_file:
#         data = json.load(classes_file)

#     for objects in data['classes']:
#         classes.append(objects['name'])

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)

#     return classes, clip_preprocess, clip_model

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

def sims_to_probs(sims_tensor, clip_model, use_logit_scale = True, temperature = None):

    sims = sims_tensor
    if use_logit_scale and hasattr(clip_model, "logit_scale"):
            tau = clip_model.logit_scale.exp()        # scalar tensor
            logits = tau * sims
    elif temperature is not None:
        logits = sims / float(temperature)
    else:
        logits = 100.0 * sims                     # common fallback

    probs = F.softmax(logits, dim=-1)                 # (C,)
    return probs.detach().cpu().numpy()

def prepare_clip(classes):
    class_prompts = [f"a photo of a {c}" for c in classes]
    model, preprocess = clip.load("ViT-B/32", device="cpu")  # or whichever, on CPU
    with torch.inference_mode():
        tokens = clip.tokenize(class_prompts)               # (C,77)
        text_feat = model.encode_text(tokens)               # (C,D)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return model, preprocess, class_prompts, text_feat  # keep text_feat for reuse

def topk_clip_for_crop(cropped_img_pil, clip_model, clip_preprocess, text_feat, classes, k=3):
    # preprocess once
    with torch.inference_mode():
        img = clip_preprocess(cropped_img_pil).unsqueeze(0)      # (1,3,H,W)
        img_feat = clip_model.encode_image(img)                  # (1,D)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True) # normalize

        # cosine similarities via dot product
        sims_tens = (img_feat @ text_feat.T)
        probs = sims_to_probs(sims_tens, clip_model, True, None)    # (C,)
        sims = (img_feat @ text_feat.T).cpu().numpy().ravel()
    # top-k without full sort
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    top_classes = [classes[i] for i in idx]
    top_scores  = sims[idx].tolist()
    return top_classes, top_scores

# def perform_mask_identification(masks, classes, clip_preprocess, clip_model):

#     class_prompts = []
#     for cls in classes: 
#         class_prompts.append(f"a photo of a {cls}") 

#     # === Save colored masks ===
#     for mask in masks:
#         # mask = mask_data["segmentation"].astype(np.uint8) * 255
#         # colored_mask = np.zeros_like(image)
#         # color = np.random.randint(0, 255, size=3)
#         # for c in range(3):
#         #     colored_mask[:, :, c] = mask * (color[c] / 255)

#         # out_path = os.path.join(output_dir, f"mask_{i:03d}.png")
#         # Image.fromarray((colored_mask).astype(np.uint8)).save(out_path)

#         cropped_box_img = get_mask_img_bbox(mask)

#         cosine_distances = []
#         for idx, class_prompt in enumerate(class_prompts):
#             text = clip.tokenize(class_prompt).to(device)
            
#             image = clip_preprocess(cropped_box_img).unsqueeze(0).to(device)
#             image_features = clip_model.encode_image(image)
#             text_features = clip_model.encode_text(text)
#             img_features = image_features.cpu().detach().numpy()
#             txt_features = text_features.cpu().detach().numpy()

#             cosine_dist = cosine_similarity(img_features, txt_features)[0][0]
#             cosine_distances.append(cosine_dist)


#         paired = list(zip(cosine_distances, classes))
#         top3 = heapq.nlargest(3, paired, key=lambda x: x[0])
#         top_similarities, top_classes = zip(*top3)

#         mask["top_classes"] = top_classes

#     return masks

# def save_masks_json(masks, with_seg = False):
#     masks_metadata = []
#     for mask in masks:
#         mask_copy = dict(mask)  # shallow copy
#         if 'segmentation' in mask_copy and with_seg == False:
#             del mask_copy['segmentation']
#         masks_metadata.append(mask_copy)
#     # Save to file
#     with open(os.path.join(output_dir, "masks.json"), "w") as f:
#         json.dump(masks_metadata, f, indent=2)

def get_original_image(camera):
    org_image_np = camera.original_image.permute(1,2,0).cpu().numpy()
    return (org_image_np*255).clip(0, 255).astype(np.uint8)
    
def update_semantics_top3(L, cls_name, score, k = 3):
    # sem = anchor_data.setdefault("semantics", {"top3": []})
    #L = sem["top3"]

    # if class already present, keep the higher score
    for i, item in enumerate(L):
        if item["class"] == cls_name:
            if float(score) <= item["score"]:
                return
            L.pop(i)
            break

    L.append({"class": cls_name, "score": score})
    L.sort(key=lambda d: d["score"], reverse=True)
    if len(L) > k:
        del L[k:] 

def crop_bbox_from_mask(mask, image_pil):
    img_np = np.asarray(image_pil)        # HxWxC
    rows, cols = np.nonzero(mask)
    if rows.size == 0 or cols.size == 0:
        return image_pil  # fallback

    y0, y1 = rows.min(), rows.max()+1
    x0, x1 = cols.min(), cols.max()+1

    crop = img_np[y0:y1, x0:x1].copy()    # small region
    # optional: apply mask inside the crop only
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

from collections import defaultdict
def get_semantic_anchors(anchor3D_info, views):
    predictor = init_automatic_sam()
    # classes, clip_preprocess, clip_model = init_clip()
    # class_prompts = get_class_prompts(classes)

    classes = get_classes()
    clip_model, clip_preprocess, class_prompts, text_feat = prepare_clip(classes)

    
    view_points = defaultdict(list)
    for global_id, anchor_data in anchor3D_info.items():
        projection_info = anchor_data["projection_info"]
        keys = list(projection_info.keys())

        for view_id in keys:
            point2D = projection_info[view_id]
            view_points[int(view_id)].append((global_id, point2D))



    for view_id, items in view_points.items():
        gt_image = get_original_image(views[view_id])
        
        masks = predictor.generate(gt_image)

        for m in masks:
            crop = crop_bbox_from_mask(m["segmentation"], gt_image)
            top_classes, top_scores = topk_clip_for_crop(
                crop, clip_model, clip_preprocess, text_feat, classes, k=3
            )
            m["semantics"] = {"top3": [
                {"class": c, "score": float(s)}
                for c, s in zip(top_classes, top_scores)
            ]}

        for global_id, point2D in items:   # items = [(aid, np.array([[u, v]], ...)), ...]
            u, v = int(point2D[0,0]), int(point2D[0,1])
            m = find_mask_for_point(masks, u, v)
            if m is None:
                continue

            sem = anchor3D_info[global_id].setdefault("semantics", {"top3": []})
            for entry in m["semantics"]["top3"]:
                update_semantics_top3(sem["top3"], entry["class"], entry["score"])
        # crop = crop_bbox_from_mask(masks[0]["segmentation"], gt_image)  # cheap crop
       
        # top_classes, top_similarities = topk_clip_for_crop(
        #     crop, clip_model, clip_preprocess, text_feat, classes, k=3
        # )


        # sem = anchor3D_info[global_id].setdefault("semantics", {"top3": []})
        # for cls_name, score in zip(top_classes, top_similarities):
        #     update_semantics_top3(sem["top3"], cls_name, score)

# if __name__ == '__main__':
#     sam_generator = init_automatic_sam()
#     masks = get_filtered_masks(sam_generator, image)
#     classes, clip_preprocess, clip_model = init_clip()
#     identified_masks = perform_mask_identification(masks, classes, clip_preprocess, clip_model)

#     save_masks_json(identified_masks, with_seg=False)
#     overlay_path = os.path.join(output_dir, "image_with_masks.png")
#     Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)


















# # === Overlay all masks with transparency ===
# for mask_data in filtered_masks:
#     mask = mask_data["segmentation"]
#     color = np.random.randint(0, 255, size=3, dtype=np.uint8)
#     overlay[mask] = overlay[mask] * 0.5 + color * 0.5  # Blend original with color

# # === Optional: draw borders (contours) ===
# for mask_data in filtered_masks:
#     mask = mask_data["segmentation"].astype(np.uint8)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=1)

# === Save overlayed image ===


