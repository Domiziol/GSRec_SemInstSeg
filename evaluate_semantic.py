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
    """
    gt_v, pred_v: 1D int arrays (same length), already filtered to valid GT (e.g. gt>=0)
    show_ids: list of class_ids to display (e.g. [47, 3, 12])
    class_names: list where class_names[cid-1] is name if ids are 1..K
    add_other: groups all classes not in show_ids into 'OTHER'
    normalize: row-normalize (fractions per GT class)
    """
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
        # also clamp preds not in show_ids to first id? nope -> drop by mapping to -999 and ignoring
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




if __name__ == "__main__":

    choose_model = 8
    gt_mesh_path = "./data/replica/scan1/mesh_semantic_verts_bothids.ply"
    pred_mesh_path = f"./experiments2_fromsam3/model_d{choose_model}k/wdist=0.0_wemb=1.0_wsem=0.0_pdist85_pemb78_psem70_512/mapped_semantic_class_id_onto_gt.ply"

    pred = PlyData.read(pred_mesh_path)["vertex"].data
    gt = PlyData.read(gt_mesh_path)["vertex"].data

    pred_cls = pred["pred_class_id"].astype(np.int32)
    gt_cls = gt["class_id"].astype(np.int32)

    valid = (gt_cls >= 0)          # ignores -1 and -2 (and any other negative)
    gt_v = gt_cls[valid]
    pred_v = pred_cls[valid]

    
    classes = np.unique(gt_v)
    
    class_names = get_class_names()

    tp = np.array([( (pred_v == c) & (gt_v == c) ).sum() for c in classes], dtype=np.int64)
    fp = np.array([( (pred_v == c) & (gt_v != c) ).sum() for c in classes], dtype=np.int64)
    fn = np.array([( (pred_v != c) & (gt_v == c) ).sum() for c in classes], dtype=np.int64)

    prec_c  = tp / np.maximum(tp + fp, 1)
    rec_c   = tp / np.maximum(tp + fn, 1)
    iou_c   = tp / np.maximum(tp + fp + fn, 1)
    f1_score_c = 2*prec_c*rec_c/np.maximum(prec_c + rec_c, 1e-12)

    # ----- "general" metrics -----
    # global over all vertices: this is equivalent to computing TP/FP/FN aggregated across classes.
    TP = tp.sum()
    FP = fp.sum()
    FN = fn.sum()

    prec_gen = TP / max(TP + FP, 1)
    rec_gen  = TP / max(TP + FN, 1)
    iou_gen  = TP / max(TP + FP + FN, 1)
    f1_score_gen = 2*prec_gen*rec_gen/max(prec_gen + rec_gen, 1e-12)

    # mean over classes (each class equally weighted)
    prec_m = float(prec_c.mean())
    rec_m  = float(rec_c.mean())
    iou_m  = float(iou_c.mean())
    f1_score_m = float(f1_score_c.mean())

    print(f"General over all verts  precision={prec_gen*100:.2f}%  recall={rec_gen*100:.2f}%  IoU={iou_gen*100:.2f}%  F1-score={f1_score_gen*100:.2f}%")
    print(f"Mean over classes  precision={prec_m*100:.2f}%  recall={rec_m*100:.2f}%  mIoU={iou_m*100:.2f}%  F1-score={f1_score_m*100:.2f}%")

    print("\nPer-class:")
    for c, p, r, u, f1, tpi, fpi, fni in zip(classes, prec_c, rec_c, iou_c, f1_score_c, tp, fp, fn):
        print(f"class {c:3d} ({cname(int(c))}): P={p*100:6.2f}% R={r*100:6.2f}% IoU={u*100:6.2f}% F1-score={f1*100:.2f}%  (TP={tpi} FP={fpi} FN={fni})")

    # SHOW_GT = [3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47, 59, 60, 63, 64, 65, 76, 78, 79, 80, 91, 92, 93, 95, 97, 98]
    SHOW = [1, 2, 3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47, 59, 60, 61, 63, 64, 65, 76, 78, 79, 80, 91, 92, 93, 95, 97, 98, 100]
    plot_confmat(gt_v, pred_v, SHOW, class_names)
    