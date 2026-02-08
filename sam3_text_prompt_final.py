## owner: Dominika Ziolkiewicz

## THESIS 
import os, glob, json
import cv2
import torch
import numpy as np
import colorsys
from ultralytics.models.sam import SAM3SemanticPredictor

import re

def class_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "", s)
    return s[:80] if len(s) > 80 else s

def color_palette(labels, sRange=(0.55, 0.95), vRange=(0.65, 1.0), hBase=0.13, noise = -1):
    labels = np.asarray(labels)
    uniq = np.unique(labels)

    hasNoise = noise in uniq
    class_labels = [l for l in uniq if l != noise] if hasNoise else uniq.tolist()

    phi = 0.6180339887498949
    palette = np.zeros((len(labels), 3), dtype=np.float32)

    labelToId = {lab: i for i, lab in enumerate(class_labels)}


    sv_patterns = [
        (sRange[1], vRange[1]),
        (sRange[1], vRange[0]),
        (sRange[0], vRange[1]),
        (sRange[0], vRange[0]),
    ]

    for label in class_labels:
        k = labelToId[label]
        h = (hBase + k * phi) % 1.0
        s, v = sv_patterns[k % len(sv_patterns)]
        r,g,b = colorsys.hsv_to_rgb(h,s, v)
        palette[labels == label] = (r, g,b)



    if hasNoise:
        palette[labels == noise] = np.array([0.6,0.6,0.6], dtype=np.float32)

    return palette

def get_classes(json_path="info_semantic.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    classes = [obj["name"].replace("_", " ") for obj in data["classes"]]
    classIds = {name: i for i, name in enumerate(classes)}
    return classes, classIds

def load_automatic_sam(path):
    automaticSAMFile = np.load(path, allow_pickle=True)
    masks = automaticSAMFile["masks.npy"]

    if masks.dtype != np.bool_:
        masks = masks > 0

    return masks

def get_legend(label_to_bgr, box_size=18, pad=10, row_h=26, font_scale=0.6, thickness=1):
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


def build_conflict_groups(masks, possible_overlap=100):
   
    n = masks.shape[0]
    if n == 0:
        return []

    group = [[] for _ in range(n)]
    for i in range(n):
        m = masks[i]
        for j in range(i + 1, n):
            intersection = np.logical_and(m, masks[j]).sum()
            if intersection > possible_overlap:
                group[i].append(j)
                group[j].append(i)

    seen = np.zeros(n, dtype=bool)
    final = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        part = []

        while stack:
            u = stack.pop()
            part.append(u)
            for v in group[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        final.append(part)

    return final

def resolve_conflicts(masks, scores, areas, possible_overlap=300):

    N = masks.shape[0]
    if N == 0:
        return []

    keep = np.ones(N, dtype=bool)

    conflictGroups = build_conflict_groups(masks, possible_overlap=possible_overlap)
    for group in conflictGroups:
        if len(group) <= 1:
            continue
        winner = max(group, key=lambda i: (scores[i], areas[i]))
        for i in group:
            if i != winner:
                keep[i] = False

    return [i for i in range(N) if keep[i]]

def calc_mask_score(masks, orig_masks):
    
    N = masks.shape[0]
    scores = np.zeros(N, dtype=np.int32)

    if orig_masks is None or orig_masks.shape[0] == 0:
        return scores

    for i in range(N):
        m = masks[i]
        best = 0
        for j in range(orig_masks.shape[0]):
            intersection = np.logical_and(m, orig_masks[j]).sum()
            if intersection > best:
                best = intersection
        scores[i] = best


    return scores


if __name__ == '__main__':
    classes, classIds = get_classes("info_semantic.json")

    imagesPath = "./data/replica/scan1/images/"
    masksOutPath = "./data/replica/scan1/2Dclassification_tests/test3"
    automaticSAMMasksPath = "./data/replica/scan1/2Dclassification_tests/black/"
    # automaticSAMMasksPath = "./data/replica/scan1/masks_test3/"
    #automaticSAMMasksPath = "./data/replica/scan1/2Dclassification_tests/test3/"

    resultsDir = os.path.join(masksOutPath, "results")
    testsDir= os.path.join(resultsDir, "check")
    os.makedirs(resultsDir, exist_ok=True)
    os.makedirs(testsDir, exist_ok=True)

    imgFiles = sorted(glob.glob(os.path.join(imagesPath, "*.png")))
    possible_overlap = 100



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



    for imageId, imageFile in enumerate(imgFiles, start=1):
        base = os.path.splitext(os.path.basename(imageFile))[0]
        segImgLegend = os.path.join(resultsDir, f"{base}_with_legend.png")
        segImgNPZ = os.path.join(resultsDir, f"{base}.npz")

        img = cv2.imread(imageFile)
        imgShape = img.shape[:2]

        autSamMasksPath = os.path.join(automaticSAMMasksPath, f"{base}.npz")
        
        autSamMasks = load_automatic_sam(autSamMasksPath)

        predictor.set_image(imageFile)

        sam3Masks = []
        sam3Labels = []

        with torch.inference_mode():
            for cls_name in classes:
                masks, boxes = predictor.inference_features(
                    predictor.features, src_shape=imgShape, text=[cls_name]
                )
                if masks is None:
                    continue
                sam3Masks.append(masks)
                sam3Labels.extend([cls_name] * masks.shape[0])


        masks = torch.cat(sam3Masks, dim=0).cpu().numpy()
        masks_bool = masks if masks.dtype == np.bool_ else (masks > 0.5)

        # === save single mask images
        singleMasksDir = os.path.join(testsDir, f"{base}_single_before")
        os.makedirs(singleMasksDir, exist_ok=True)

        uniqueLabelsOnlySAM3 = []
        seen = set()
        for label in sam3Labels:
            if label not in seen:
                seen.add(label)
                uniqueLabelsOnlySAM3.append(label)

        labelIdsOnlySAM3 = np.arange(len(uniqueLabelsOnlySAM3), dtype=np.int32)
        rgbOnlySAM3 = color_palette(labelIdsOnlySAM3)
        rgb255OnlySAM3 = (rgbOnlySAM3 * 255.0).round().clip(0, 255).astype(np.uint8)
        bgr255OnlySAM3 = rgb255OnlySAM3[:, ::-1]
        labelToBGROnlySAM3 = {label: bgr255OnlySAM3[i] for i, label in enumerate(uniqueLabelsOnlySAM3)}

        for k in range(masks_bool.shape[0]):
            m = masks_bool[k]
            label = sam3Labels[k]
            color = tuple(int(x) for x in labelToBGROnlySAM3.get(label, np.array([0, 255, 0], np.uint8)))

            copy = img.copy()
            copy[m] = color
            legendSingleLabel = get_legend({label: np.array(color, dtype=np.uint8)})

            out_one = os.path.join(singleMasksDir, f"{base}_mask{k:03d}_{class_name(label)}.png")
            save_with_side_legend(out_one, copy, legendSingleLabel)
        # ====

        
        labelIdsSAM3 = np.array([classIds[n] for n in sam3Labels], dtype=np.int32)

        # === calc scores
        scores = calc_mask_score(masks_bool, autSamMasks)
        areas = masks_bool.reshape(masks_bool.shape[0], -1).sum(axis=1).astype(np.int32)

        # === resolve conflicts
        winnersIds = resolve_conflicts(masks=masks_bool, scores=scores, areas=areas, significant_overlap=possible_overlap)

        masksFinal = masks_bool[winnersIds]
        labelsIdsFinal = labelIdsSAM3[winnersIds]
        labelsNamesFinal = [sam3Labels[i] for i in winnersIds]

        
        masksFinal_u8 = masksFinal.astype(np.uint8)
        np.savez_compressed(segImgNPZ, **{"masks": masksFinal_u8, "labels": labelsIdsFinal})


        # === singles AFTER
        uniqLabels = []
        seen = set()
        for label in labelsNamesFinal:
            if label not in seen:
                seen.add(label)
                uniqLabels.append(label)

        labelIdsDisplay = np.arange(len(uniqLabels), dtype=np.int32)
        rgb = color_palette(labelIdsDisplay)
        rgb255 = (rgb * 255.0).round().clip(0, 255).astype(np.uint8)
        bgr255 = rgb255[:, ::-1]

        labelToBGR= {lab: bgr255[i] for i, lab in enumerate(uniqLabels)}
        maskColors = [tuple(int(x) for x in labelToBGR[lab]) for lab in labelsNamesFinal]

        img2 = img.copy()
        for i in range(masksFinal.shape[0]):
            img2[masksFinal[i]] = maskColors[i]

        legend = get_legend(labelToBGR)
        save_with_side_legend(segImgLegend, img2, legend)

        print(f"{imageId}/{len(imgFiles)} ({base}), masks left: {len(winnersIds)} from: {masks_bool.shape[0]},,   saved: {segImgNPZ} + {segImgLegend}")

       

        # === save also single masks after conflicts resolve
        singleMasksAfterDir = os.path.join(testsDir, f"{base}_single_after")
        os.makedirs(singleMasksAfterDir, exist_ok=True)

        for i in range(masksFinal.shape[0]):
            m = masksFinal[i]
            label = labelsNamesFinal[i]

            if label in labelToBGR:
                color = tuple(int(x) for x in labelToBGR[label])
            else:
                color = (0, 255, 0)

            img2 = img.copy()
            img2[m] = color

            legendSingle = get_legend({label: np.array(color, dtype=np.uint8)})

            savePath = os.path.join(singleMasksAfterDir, f"{base}_mask{i:03d}_{class_name(label)}.png")
            save_with_side_legend(savePath, img2, legendSingle)
