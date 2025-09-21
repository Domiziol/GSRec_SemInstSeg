import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import json
import clip
from sklearn.metrics.pairwise import cosine_similarity
import heapq

device="cuda" if torch.cuda.is_available() else "cpu"
# image_path = "image2.png"

# output_dir = "masks"
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# org_image = image


def init_prompted_sam():
    model_type = "vit_h"
    checkpoint_path = "sam_model/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)

    return predictor

def get_mask_per_point(predictor, image, point):
    # predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point,
        point_labels=np.array([1]),     # some kind of background, foreground - check which is better
        multimask_output=False
    )

    # best_mask = masks[np.argmax(scores)]

    return masks[0]


def get_mask_img_bbox(mask, image):

    rows, cols = np.where(mask)
    y_min = rows.min()
    y_max = rows.max()
    x_min = cols.min()
    x_max = cols.max()

    cropped_box = [x_min, y_min, x_max, y_max]

    img_np = np.asarray(image)
    segmented_img_np = np.zeros_like(img_np)
    segmented_img_np[mask] = img_np[mask]
    segmented_image = Image.fromarray(segmented_img_np)

    cropped_box_img = segmented_image.crop(cropped_box)
    
    return cropped_box_img


def init_clip():
    classes= []

    with open("info_semantic.json", 'r') as classes_file:
        data = json.load(classes_file)

    for objects in data['classes']:
        classes.append(objects['name'])

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)

    return classes  #, clip_preprocess, clip_model

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
        sims = (img_feat @ text_feat.T).cpu().numpy().ravel()    # (C,)

    # top-k without full sort
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    top_classes = [classes[i] for i in idx]
    top_scores  = sims[idx].tolist()
    return top_classes, top_scores

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

# def perform_masks_identification(masks, image, classes, clip_preprocess, clip_model):

#     class_prompts = []
#     for cls in classes: 
#         class_prompts.append(f"a photo of a {cls}") 

#     for mask in masks:

#         cosine_distances = identify_mask(mask, image, class_prompts, clip_model, clip_preprocess)
#         paired = list(zip(cosine_distances, classes))
#         top3 = heapq.nlargest(3, paired, key=lambda x: x[0])
#         top_similarities, top_classes = zip(*top3)

#         mask["top_classes"] = top_classes
#         mask["top_scores"] = top_similarities

#     return masks

def get_class_prompts(classes):
    class_prompts = []
    for cls in classes: 
        class_prompts.append(f"a photo of a {cls}")
    
    return class_prompts

def perform_mask_identification(mask, image, classes, class_prompts, clip_preprocess, clip_model):

    cosine_distances = identify_mask(mask, image, class_prompts, clip_preprocess, clip_model)
    paired = list(zip(cosine_distances, classes))
    top3 = heapq.nlargest(3, paired, key=lambda x: x[0])
    top_similarities, top_classes = zip(*top3)

    # mask["top_classes"] = top_classes
    # mask["top_scores"] = top_similarities

    return top_classes, top_similarities


def identify_mask(mask, image, class_prompts, clip_preprocess, clip_model):
    cropped_box_img = get_mask_img_bbox(mask, image)

    cosine_distances = []
    for idx, class_prompt in enumerate(class_prompts):
        text = clip.tokenize(class_prompt).to(device)
        
        image = clip_preprocess(cropped_box_img).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        img_features = image_features.cpu().detach().numpy()
        txt_features = text_features.cpu().detach().numpy()

        cosine_dist = cosine_similarity(img_features, txt_features)[0][0]
        cosine_distances.append(cosine_dist)

    return cosine_distances

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
    

import matplotlib.pyplot as plt
from collections import defaultdict

def get_semantic_anchors(anchor3D_info, views):
    predictor = init_prompted_sam()
    # classes, clip_preprocess, clip_model = init_clip()
    # class_prompts = get_class_prompts(classes)

    classes = init_clip()
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
        predictor.set_image(gt_image)

        for global_id, point2D in items:
            mask = get_mask_per_point(predictor, gt_image, point2D)
            # top_classes, top_similarities = perform_mask_identification(mask, gt_image, classes, class_prompts, clip_preprocess, clip_model)
            
            crop = crop_bbox_from_mask(mask, gt_image)  # cheap crop
            top_classes, top_similarities = topk_clip_for_crop(
                crop, clip_model, clip_preprocess, text_feat, classes, k=3
            )

            sem = anchor3D_info[global_id].setdefault("semantics", {"top3": []})
            for cls_name, score in zip(top_classes, top_similarities):
                update_semantics_top3(sem["top3"], cls_name, score)

    # for global_id, anchor_data in anchor3D_info.items():
    #     projection_info = anchor_data["projection_info"]
    #     keys = list(projection_info.keys())
    #     anchor_data.setdefault("semantics", {"top3" : []})
    #     sem = anchor_data["semantics"]

    #     for view_id in keys:
    #         gt_image = get_original_image(views[view_id])
    #         point2D = projection_info[view_id]
            
    #         mask = get_mask_per_point(predictor, gt_image, point2D)
    #         top_classes, top_similarities = perform_mask_identification(mask, gt_image, classes, class_prompts, clip_preprocess, clip_model)
            

    #         for cls_name, score in zip(top_classes, top_similarities):
    #             update_semantics_top3(sem["top3"], cls_name, score)
            
            
            # print(point2D)

            # plt.imshow(gt_image, cmap='gray')
            # plt.axis('off')
            # plt.show()







### CLIP

# classes= []

# with open("info_semantic.json", 'r') as classes_file:
#     data = json.load(classes_file)

# for objects in data['classes']:
#     classes.append(objects['name'])

# class_prompts = []
# for cls in classes: 
#     class_prompts.append(f"a photo of a {cls}") 

# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


# cosine_distances = []
# for idx, class_prompt in enumerate(class_prompts):
#     text = clip.tokenize(class_prompt).to(device)
    
#     image = clip_preprocess(cropped_box_img).unsqueeze(0).to(device)
#     image_features = clip_model.encode_image(image)
#     text_features = clip_model.encode_text(text)
#     img_features = image_features.cpu().detach().numpy()
#     txt_features = text_features.cpu().detach().numpy()

#     cosine_dist = cosine_similarity(img_features, txt_features)[0][0]
#     cosine_distances.append(cosine_dist)


# paired = list(zip(cosine_distances, classes))
# top3 = heapq.nlargest(3, paired, key=lambda x: x[0])
# top_similarities, top_classes = zip(*top3)

# print(top_similarities)
# print(top_classes)

# plt.imshow(org_image)
# # plt.imshow(best_mask, alpha=0.5)  # overlay mask
# plt.scatter(input_point[:, 0], input_point[:, 1], color='red')
# plt.axis('off')
# plt.show()