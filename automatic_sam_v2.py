import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import os
import json
import clip
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import colorsys
from pathlib import Path

# image_path = "image.png"
# image = cv2.imread(image_path)
# overlay = image.copy()
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# output_dir = "masks"
device="cuda" if torch.cuda.is_available() else "cpu"


def setup_automatic_sam():
    modelName = "vit_h"
    checkpointPath = "sam_model/sam_vit_h_4b8939.pth"

    sam = sam_model_registry[modelName](checkpoint=checkpointPath)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model = sam,
        points_per_side=8,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.96,
        crop_n_points_downscale_factor=1
        )
    return mask_generator

def filter_masks_by_area(mask_generator, image):

    masks = mask_generator.generate(image)
    filtered_masks = []

    for i, m in enumerate(masks):
        seg = np.asarray(m["segmentation"], dtype=bool)
        area = int(seg.sum())
        if area > 300:
            m2 = dict(m)  
            m2["segmentation"] = seg  
            filtered_masks.append(m2)
    
    return filtered_masks

def get_classes():
    classes= []

    with open("info_semantic.json", 'r') as file:
        data = json.load(file)

    for objects in data['classes']:
        classes.append(objects['name'])

    return classes 

def prepare_clip(classes):
    prompts = [f"a photo of a {c}" for c in classes]
    
    modelName, clipDevice = clip.load("ViT-B/16", device="cpu")  
    with torch.inference_mode():
        tokens = clip.tokenize(prompts)               
        textFeatures = modelName.encode_text(tokens)               
        textFeatures = textFeatures / textFeatures.norm(dim=-1, keepdim = True)
    return modelName, clipDevice, textFeatures 


def best_clip_proposals(croppedImg, clipModel, clipDevice, textFeatures, classes, topK=3):
    

    with torch.inference_mode():
        img = clipDevice(croppedImg).unsqueeze(0)      
        imgFeature = clipModel.encode_image(img)                  
        imgFeature = imgFeature / imgFeature.norm(dim=-1) 

        
        cosineSimilarities = (imgFeature @ textFeatures.T).cpu().numpy().ravel()

    # topK classes and cosine sims
    idx = np.argpartition(-cosineSimilarities, topK-1)[:topK]
    sorted = idx[np.argsort(-cosineSimilarities[idx])]
    topClasses = [classes[i] for i in sorted]


    return topClasses


def get_color_palette(classes, sRange=(0.55, 0.95), vRange=(0.65, 1.0), hueBase=0.13):
    
    classes = np.asarray(classes)
    uniq = np.unique(classes)

    labels = uniq.tolist()
    n = len(labels)
    ratio = 0.6180339884798949  # golden ratio
    palette = np.zeros((len(labels), 3), dtype=np.float32)
    labelsIds = {lab: i for i, lab in enumerate(labels)}
    sv_patterns = [
        (sRange[1], vRange[1]),
        (sRange[1], vRange[0]),
        (sRange[0], vRange[1]),
        (sRange[0], vRange[0]),
    ]


    for _, label in enumerate(labels):
        i = labelsIds[label]
        h = (hueBase + i * ratio) % 1.0
        s,v = sv_patterns[i % len(sv_patterns)]
        r,g,b = colorsys.hsv_to_rgb(h,s,v)

        palette[labels==label] = (r,g,b)



    return palette

def color_mask_and_create_legend(img, masks, labelsIds, classes):
    H, W = img.shape[:2]
    overlay = img.copy()

    palette = (get_color_palette(np.arange(len(classes))) * 255).astype(np.uint8)

    for j in range(masks.shape[0]):

        cls = int(labelsIds[j])

        if cls < 0 or cls >= len(classes):
            continue

        color = palette[cls]
        m = masks[j]

        if m.sum() == 0:
            continue

        overlay[m] = color

    colImg = Image.fromarray(overlay)
    draw = ImageDraw.Draw(colImg)
    
    font = ImageFont.load_default()
   

    y = 5
    used = set()
    for j in range(masks.shape[0]):

        cls = int(labelsIds[j])
        if cls in used or cls < 0 or cls >= len(classes):
            continue

        used.add(cls)

        color = tuple(int(x) for x in palette[cls])
        cls_name = classes[cls]

        draw.rectangle([5, y + 2,17, y +14], color, color)
        draw.text((22, y), cls_name, (255,255,255), font)


        y += 18
        if y > H - 20:
            break

    return colImg

def save_masks(img, masks, labelsIds, classes, path):
    path.mkdir(parents=True, exist_ok=True)
    H, W = masks.shape[1:]
    
    order = np.arange(masks.shape[0])

    for i in order:

        m = masks[j]
        if m.sum() == 0:
            continue

        rows, cols = np.nonzero(m)
        y0, y1 = rows.min(), rows.max() + 1
        x0, x1 = cols.min(), cols.max() +1

        crop = np.array(img)[y0:y1, x0:x1].copy()
        local = m[y0:y1, x0:x1]
        bkg = np.zeros_like(crop)

        crop = np.where(local[..., None], crop, bkg)
        cropImg = Image.fromarray(crop)
        

        cls_id = labelsIds[i]
        cls_name = classes[cls_id]
        
        cropImg.save(path / f"mask_{i:03d}_{cls_name}.png")
      

def preprocess_images_with_black_masks_bkg():
    input_path = "./data/replica/scan1/images/"
    output_path = "./data/replica/scan1/2Dclassification_tests/black2"

    Path(output_path).mkdir(parents=True, exist_ok=True)
    vis_dir = Path(output_path) / "overlays"
    vis_dir.mkdir(parents=True, exist_ok=True)
    crops_root = Path(output_path) / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)
    binary_masks = Path(output_path) / "binary_mask_images"
    binary_masks.mkdir(parents=True, exist_ok=True)



    mask_generator = setup_automatic_sam()
    classes = get_classes()
    classesToIds = {n: i for i, n in enumerate(classes)}
    idsToclasses = {i: n for i, n in enumerate(classes)}


    clipModelName, clipDevice, textFeatures = prepare_clip(classes)

    
    for file in sorted(os.listdir(input_path)):
        
        if not file.lower().endswith((".png", ".jpg")):
            continue

        img_path = os.path.join(input_path, file)
        stem = Path(file).stem
        image = Image.open(img_path).convert("RGB")
        imageNP = np.array(image)

        masks = filter_masks_by_area(mask_generator, imageNP)
        H, W = imageNP.shape[:2]

        masks_bool = []
        for m in masks:
            seg = np.asarray(m["segmentation"], dtype=bool)
            if seg.sum() > 0:
                masks_bool.append(seg)
        if len(masks_bool) == 0:
            masks_bool = np.zeros((0, H, W), dtype=bool)
        else:
            masks_bool = np.stack(masks_bool, axis=0)

        
        binary_mask_img_dir = binary_masks / stem
        binary_mask_img_dir.mkdir(parents=True, exist_ok=True)

        for i in range(masks_bool.shape[0]):
            m = masks_bool[i]
            mask_uint8 = (m.astype(np.uint8) * 255)
            Image.fromarray(mask_uint8, mode="L").save(binary_mask_img_dir / f"mask_{i:03d}.png")

        labelsIds = []
        for i in range(masks_bool.shape[0]):
            m = masks_bool[i]
            rows, cols = np.nonzero(m)
            if rows.size == 0 or cols.size == 0:
                continue

            y0, y1 = rows.min(), rows.max() + 1
            x0, x1 = cols.min(), cols.max() + 1

            addPixels = 20     

            margin_y = max(addPixels, int(0.35 * (y1 - y0)))
            margin_x = max(addPixels, int(0.35 * (x1 - x0)))

            y0 = max(0, y0 - margin_y)
            y1 = min(H, y1 + margin_y)
            x0 = max(0, x0 - margin_x)
            x1 = min(W, x1 + margin_x)

            crop = image.crop((x0, y0, x1, y1))

            crop_dir = crops_root / stem
            crop_dir.mkdir(parents=True, exist_ok=True)

            bestClasses= best_clip_proposals(
                crop, clipModelName, clipDevice, textFeatures, classes, 1
            )


            labelsIds.append(classesToIds[bestClasses[0]])
            pred_name = bestClasses[0]

            crop.save(crops_root / stem / f"mask_{i:03d}_crop_{pred_name}.png")


        np.savez_compressed(
            Path(output_path) / f"{stem}.npz",
            masks=masks_bool.astype(np.uint8),
            labels=np.asarray(labelsIds, dtype=np.int16),
            image_size=np.array([H, W], dtype=np.int32),
            version=np.int32(1),
        )

        overlay = color_mask_and_create_legend(imageNP, masks_bool, labelsIds, classes)
        overlay.save(vis_dir / f"{stem}_overlay.png")




if __name__ == '__main__':
    
    preprocess_images_with_black_masks_bkg()





