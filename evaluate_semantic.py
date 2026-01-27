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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def get_class_names():
    classes= []

    with open("./data/replica/scan1/info_semantic.json", 'r') as classes_file:
        data = json.load(classes_file)

    for objects in data['classes']:
        classes.append(objects['name'])

    return classes

def cname(c: int) -> str:
        return class_names[c-1]


import seaborn as sns
def plot_confmat(gt_v, pred_v, show_ids, class_names, add_other=True, normalize=False):
    
    show_ids = [int(x) for x in show_ids]
    show_set = set(show_ids)

    def cname(cid: int) -> str:
        return class_names[cid - 1] if 1 <= cid <= len(class_names) else "UNKNOWN"

    if add_other:
        # map anything not in show_ids -> 0 ("OTHER")
        gt_m   = np.where(np.isin(gt_v, show_ids), gt_v, 0)
        pred_m = np.where(np.isin(pred_v, show_ids), pred_v, 0)
        ids = [0] + show_ids
        ticks = ["OTHER"] + [f"{cid}: {cname(cid)}" for cid in show_ids]
    else:
        # keep only GT in show_ids
        m = np.isin(gt_v, show_ids)
        gt_m, pred_m = gt_v[m], pred_v[m]
      
        pred_m = np.where(np.isin(pred_m, show_ids), pred_m, -999)
        ids = show_ids
        ticks = [f"{cid}: {cname(cid)}" for cid in show_ids]

    # build confusion matrix
    idx = {cid: i for i, cid in enumerate(ids)}
    cm = np.zeros((len(ids), len(ids)), dtype=np.int64)
    for g, p in zip(gt_m, pred_m):
        if g in idx and p in idx:
            cm[idx[g], idx[p]] += 1

    cm_plot = cm.astype(float)
    if normalize:
        cm_plot /= np.maximum(cm_plot.sum(axis=1, keepdims=True), 1.0)

    plt.figure(figsize=(max(6, 0.6 * len(ids)), max(5, 0.6 * len(ids))))
    ax = sns.heatmap(
        cm_plot,
        annot=True,
        fmt=".2f" if normalize else "g",
        xticklabels=ticks,
        yticklabels=ticks,
        square=True,
        cbar=True
    )
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix" + (" (row-normalized)" if normalize else " (counts)"))
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def semantic_metrics_from_confusion(gt_v, pred_v, valid_ids=None, ignore_ids=(-1, -2), num_classes=None):
    
    gt_v = gt_v.astype(np.int64)
    pred_v = pred_v.astype(np.int64)

    
    ign = np.isin(gt_v, np.array(ignore_ids, dtype=np.int64))
    gt_v = gt_v[~ign]
    pred_v = pred_v[~ign]

    # if valid_ids is given: ignore GT outside valid_ids (ScanNet-style "void")
    if valid_ids is not None:
        valid_ids = np.array(list(map(int, valid_ids)), dtype=np.int64)
        m = np.isin(gt_v, valid_ids)
        gt_v = gt_v[m]
        pred_v = pred_v[m]
    else:
        valid_ids = np.unique(gt_v)

    
    use_other = True

    valid_ids = np.array(valid_ids, dtype=np.int64)
    valid_ids_sorted = np.sort(valid_ids)

    if use_other:
        other_id = -999999
        ids = np.concatenate([valid_ids_sorted, np.array([other_id], dtype=np.int64)])
    else:
        ids = valid_ids_sorted

    id2idx = {int(cid): i for i, cid in enumerate(ids)}
    K = len(ids)

    # 4) build confusion matrix
    cm = np.zeros((K, K), dtype=np.int64)

    if use_other:
        pred_m = np.where(np.isin(pred_v, valid_ids_sorted), pred_v, other_id)
    else:
        # drop pairs where pred not in ids
        keep = np.isin(pred_v, valid_ids_sorted)
        gt_v = gt_v[keep]
        pred_m = pred_v[keep]

    # populate
    gt_idx = np.vectorize(id2idx.get)(gt_v)
    pr_idx = np.vectorize(id2idx.get)(pred_m)
    np.add.at(cm, (gt_idx, pr_idx), 1)

    # derive per-class metrics (only for real classes, exclude OTHER column/row)
    real_K = len(valid_ids_sorted)
    cm_real = cm[:real_K, :real_K]
    tp = np.diag(cm_real).astype(np.float64)
    gt_count = cm_real.sum(axis=1).astype(np.float64)
    pred_count = cm_real.sum(axis=0).astype(np.float64)

    # Actually gt_count here excludes OTHER column, so incorporate errors to OTHER:
    if use_other:
        gt_count_full = cm[:real_K, :].sum(axis=1).astype(np.float64)  # includes OTHER column
    else:
        gt_count_full = gt_count

    fn = gt_count_full - tp
    fp = (cm[:real_K, :real_K].sum(axis=0).astype(np.float64) - tp)  # within-set FP
    if use_other:
        # preds mapped to OTHER do not count as FP for any within-set class, but they DO hurt via FN (missed GT)
        pass

    denom_iou = tp + fp + fn
    iou = tp / np.maximum(denom_iou, 1.0)

    acc_class = tp / np.maximum(gt_count_full, 1.0)  # mean class accuracy = recall per class

    # 6) aggregates (only over classes present in GT)
    present = gt_count_full > 0
    mIoU = float(np.mean(iou[present])) if np.any(present) else float("nan")
    mAcc = float(np.mean(acc_class[present])) if np.any(present) else float("nan")

    # overall_acc = float((gt_v == pred_v).mean()) if gt_v.size else float("nan")
    pred_eval = np.where(np.isin(pred_v, valid_ids_sorted), pred_v, other_id)
    overall_acc = float((gt_v == pred_eval).mean()) if gt_v.size else float("nan")


    return {
        "valid_ids": valid_ids_sorted,
        "cm": cm,
        "per_class": {
            "tp": tp.astype(np.int64),
            "fp": fp.astype(np.int64),
            "fn": fn.astype(np.int64),
            "iou": iou,
            "acc": acc_class,
            "gt_count": gt_count_full.astype(np.int64),
        },
        "overall_acc": overall_acc,
        "mIoU": mIoU,
        "mAcc": mAcc,
    }



if __name__ == "__main__":

    choose_model = 8
    gt_mesh_path = "./data/replica/scan1/mesh_semantic_verts_bothids.ply"
    pred_mesh_path = f"./experiments2_fromsam3/model_d{choose_model}k/wdist=0.0_wemb=1.0_wsem=0.0_pdist85_pemb78_psem70_512/mapped_semantic_class_id_&_object_id_onto_gt.ply"
    # pred_mesh_path = f"./outputs/final_sam3/d8k_l01/mapped_semantic_class_id_&_object_id_onto_gt.ply" # raw, not smoothed

    pred = PlyData.read(pred_mesh_path)["vertex"].data
    gt = PlyData.read(gt_mesh_path)["vertex"].data

    pred_cls = pred["pred_class_id"].astype(np.int32)
    gt_cls = gt["class_id"].astype(np.int32)
    print(np.unique(gt_cls))

    VALID_IDS = [3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47, 59, 60, 63, 64, 65, 76, 78, 79 ,80, 91, 92, 93, 95, 97, 98] # exact GT

    # VALID_IDS = [2, 3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47, 59, 60, 61, 63, 64, 65, 75, 76, 78, 79 ,80, 91, 92, 93, 95, 97, 98, 100] # lookup

    # VALID_IDS = [2, 3, 11,   18,  29, 31, 37, 40, 44, 47, 59,61, 63,92, 95,98, 100]
    # VALID_IDS = [12,13,19,20, 60,   64, 65, 75, 76, 78, 79 ,80, 91,  93, 95, 97,]
    class_names = get_class_names()

    #VALID_IDS = [2, 3, 11, 12, 13, 18, 19, 20, 23, 29, 31, 34,  37, 40, 41, 44, 47, 54, 56,  59, 60, 61,  63, 64, 65, 75, 76, 78, 79, 80, 91, 92, 93, 95, 97, 98, 100]


    metrics = semantic_metrics_from_confusion(
        gt_cls, pred_cls,
        valid_ids=VALID_IDS,
        ignore_ids=(-1, -2, 0),   # include 0 if you use 0 as unlabeled/void
    )

    print(f"Overall Acc: {metrics['overall_acc']*100:.2f}%")
    print(f"mAcc (mean class recall): {metrics['mAcc']*100:.2f}%")
    print(f"mIoU: {metrics['mIoU']*100:.2f}%")

    # per-class print
    valid_ids = metrics["valid_ids"]
    pc = metrics["per_class"]
    for i, cid in enumerate(valid_ids):
        name = class_names[cid-1] if 1 <= cid <= len(class_names) else "UNKNOWN"
        print(
            f"class {cid:3d} ({name}): "
            f"IoU={pc['iou'][i]*100:6.2f}%  "
            f"Acc={pc['acc'][i]*100:6.2f}%  "
            f"(TP={pc['tp'][i]} FP={pc['fp'][i]} FN={pc['fn'][i]} GT={pc['gt_count'][i]})"
        )

    ignore_ids = np.array([-1, -2, 0], dtype=np.int32)
    valid_ids = np.array(VALID_IDS, dtype=np.int32)

    
    m = ~np.isin(gt_cls, ignore_ids)
    gt_f = gt_cls[m]
    pr_f = pred_cls[m]

    
    m2 = np.isin(gt_f, valid_ids)
    gt_f = gt_f[m2]
    pr_f = pr_f[m2]

    
    OTHER = 102
    gt_vis = np.where(np.isin(gt_f, valid_ids), gt_f, OTHER)
    pr_vis = np.where(np.isin(pr_f, valid_ids), pr_f, OTHER)

    
    SHOW = VALID_IDS

    show_ids = np.array(SHOW, dtype=np.int32)
    labels = np.concatenate(([OTHER], show_ids))

    display_labels = (["OTHER"] +
        [f"{cid}: {class_names[cid-1]}" if 1 <= cid <= len(class_names) else f"{cid}: UNKNOWN"
        for cid in show_ids]
    )

    cm = confusion_matrix(gt_vis, pr_vis, labels=labels, normalize="true")  # "true" = row-normalized

    fig, ax = plt.subplots(figsize=(0.45 * len(labels) + 4, 0.45 * len(labels) + 3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, xticks_rotation=45, values_format=".2f", cmap="Blues", colorbar=True)
    ax.set_title("Confusion matrix (row-normalized)")
    plt.tight_layout()
    plt.show()

    # valid = (gt_cls >= 0)          # ignores -1 and -2 (and any other negative)
    # gt_v = gt_cls[valid]
    # pred_v = pred_cls[valid]

    
    # classes = np.unique(gt_v)
    
    # class_names = get_class_names()

    # tp = np.array([( (pred_v == c) & (gt_v == c) ).sum() for c in classes], dtype=np.int64)
    # fp = np.array([( (pred_v == c) & (gt_v != c) ).sum() for c in classes], dtype=np.int64)
    # fn = np.array([( (pred_v != c) & (gt_v == c) ).sum() for c in classes], dtype=np.int64)

    # prec_c  = tp / np.maximum(tp + fp, 1)
    # rec_c   = tp / np.maximum(tp + fn, 1)
    # iou_c   = tp / np.maximum(tp + fp + fn, 1)
    # f1_score_c = 2*prec_c*rec_c/np.maximum(prec_c + rec_c, 1e-12)

    # # ----- "general" metrics -----
    # # global over all vertices: this is equivalent to computing TP/FP/FN aggregated across classes.
    # TP = tp.sum()
    # FP = fp.sum()
    # FN = fn.sum()

    # prec_gen = TP / max(TP + FP, 1)
    # rec_gen  = TP / max(TP + FN, 1)
    # iou_gen  = TP / max(TP + FP + FN, 1)
    # f1_score_gen = 2*prec_gen*rec_gen/max(prec_gen + rec_gen, 1e-12)

    # # mean over classes (each class equally weighted)
    # prec_m = float(prec_c.mean())
    # rec_m  = float(rec_c.mean())
    # iou_m  = float(iou_c.mean())
    # f1_score_m = float(f1_score_c.mean())

    # print(f"General over all verts  precision={prec_gen*100:.2f}%  recall={rec_gen*100:.2f}%  IoU={iou_gen*100:.2f}%  F1-score={f1_score_gen*100:.2f}%")
    # print(f"Mean over classes  precision={prec_m*100:.2f}%  recall={rec_m*100:.2f}%  mIoU={iou_m*100:.2f}%  F1-score={f1_score_m*100:.2f}%")

    # print("\nPer-class:")
    # for c, p, r, u, f1, tpi, fpi, fni in zip(classes, prec_c, rec_c, iou_c, f1_score_c, tp, fp, fn):
    #     print(f"class {c:3d} ({cname(int(c))}): P={p*100:6.2f}% R={r*100:6.2f}% IoU={u*100:6.2f}% F1-score={f1*100:.2f}%  (TP={tpi} FP={fpi} FN={fni})")

    # # SHOW_GT = [3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47, 59, 60, 63, 64, 65, 76, 78, 79, 80, 91, 92, 93, 95, 97, 98]
    # SHOW = [1, 2, 3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47, 59, 60, 61, 63, 64, 65, 76, 78, 79, 80, 91, 92, 93, 95, 97, 98, 100]
    # plot_confmat(gt_v, pred_v, SHOW, class_names)
    