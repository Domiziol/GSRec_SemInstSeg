import numpy as np
from collections import defaultdict


# ---------------------------------------------------------
# 1. Semantic segmentation metrics (IoU, mIoU)
# ---------------------------------------------------------

def compute_semantic_iou(gt_labels: np.ndarray,
                         pred_labels: np.ndarray,
                         ignore_label: int = -1):
    """
    Compute per-class IoU and mIoU for semantic segmentation.

    Args:
        gt_labels:   (N,) int32 ground-truth semantic labels
        pred_labels: (N,) int32 predicted semantic labels
        ignore_label: label value to ignore in GT (e.g. -1)

    Returns:
        per_class_iou: dict[class_id] -> IoU
        mIoU: float, mean IoU over all classes present in GT (excluding ignore_label)
    """
    assert gt_labels.shape == pred_labels.shape

    # classes present in GT (excluding ignore)
    valid_mask = gt_labels != ignore_label
    gt_valid = gt_labels[valid_mask]
    pred_valid = pred_labels[valid_mask]

    classes = np.unique(gt_valid)
    per_class_iou = {}

    for c in classes:
        gt_c = gt_valid == c
        pred_c = pred_valid == c

        intersection = np.sum(gt_c & pred_c)
        union = np.sum(gt_c | pred_c)

        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union

        per_class_iou[int(c)] = float(iou)

    if len(per_class_iou) == 0:
        mIoU = 0.0
    else:
        mIoU = float(np.mean(list(per_class_iou.values())))

    return per_class_iou, mIoU


# ---------------------------------------------------------
# 2. Instance segmentation metrics (AP, mAP) ScanNet-style
# ---------------------------------------------------------

def _instance_iou(verts_pred: np.ndarray, verts_gt: np.ndarray) -> float:
    """
    IoU between two instance masks given as vertex index arrays.
    """
    if verts_pred.size == 0 and verts_gt.size == 0:
        return 0.0
    inter = np.intersect1d(verts_pred, verts_gt).size
    union = verts_pred.size + verts_gt.size - inter
    if union == 0:
        return 0.0
    return inter / union


def compute_instance_ap(gt_instances,
                        pred_instances,
                        iou_thresholds=(0.25, 0.5)):
    """
    Compute per-class AP at given IoU thresholds, ScanNet-style.

    Args:
        gt_instances:   list of GT instance dicts:
                        { "instance_id", "class_id", "vert_indices" }
        pred_instances: list of predicted instance dicts:
                        { "instance_id", "class_id", "vert_indices", "confidence" }
        iou_thresholds: iterable of IoU thresholds (e.g. [0.25, 0.5])

    Returns:
        results: dict with:
            - "ap_per_class": {thr: {class_id: AP}}
            - "mAP":          {thr: mAP_over_classes}
    """
    # group instances by class
    gt_by_class = defaultdict(list)
    for inst in gt_instances:
        gt_by_class[int(inst["class_id"])].append(inst)

    pred_by_class = defaultdict(list)
    for inst in pred_instances:
        pred_by_class[int(inst["class_id"])].append(inst)

    ap_per_class = {thr: {} for thr in iou_thresholds}
    mAP = {}

    # union of classes that appear in GT or preds
    all_classes = sorted(set(gt_by_class.keys()) | set(pred_by_class.keys()))

    for thr in iou_thresholds:
        aps = []

        for c in all_classes:
            gt_c = gt_by_class.get(c, [])
            pred_c = pred_by_class.get(c, [])

            n_gt = len(gt_c)
            if n_gt == 0:
                # No GT for this class → AP is undefined; skip in mAP
                continue

            # sort predictions by confidence (descending)
            pred_c_sorted = sorted(pred_c, key=lambda x: -float(x["confidence"]))

            # for each GT instance we track if it was matched
            gt_matched = np.zeros(n_gt, dtype=bool)

            tps = []
            fps = []

            for p in pred_c_sorted:
                p_verts = p["vert_indices"]

                best_iou = 0.0
                best_gt_idx = -1

                for j, g in enumerate(gt_c):
                    if gt_matched[j]:
                        continue
                    g_verts = g["vert_indices"]
                    iou = _instance_iou(p_verts, g_verts)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= thr and best_gt_idx >= 0:
                    tps.append(1)
                    fps.append(0)
                    gt_matched[best_gt_idx] = True
                else:
                    tps.append(0)
                    fps.append(1)

            if len(tps) == 0:
                ap = 0.0
            else:
                tps = np.array(tps, dtype=np.float32)
                fps = np.array(fps, dtype=np.float32)

                tp_cum = np.cumsum(tps)
                fp_cum = np.cumsum(fps)

                precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
                recalls = tp_cum / float(n_gt)

                # integrate precision-recall curve (ScanNet-style: area under curve)
                # using numpy.trapz on sorted recall
                # (predictions are already sorted by confidence)
                ap = float(np.trapz(precisions, recalls))

            ap_per_class[thr][c] = ap
            aps.append(ap)

        if len(aps) == 0:
            mAP[thr] = 0.0
        else:
            mAP[thr] = float(np.mean(aps))

    results = {
        "ap_per_class": ap_per_class,
        "mAP": mAP,
    }
    return results


# ---------------------------------------------------------
# 3. Combined evaluation helper
# ---------------------------------------------------------

def evaluate_scannet_style(gt: dict,
                           pred: dict,
                           iou_thresholds=(0.25, 0.5),
                           ignore_label: int = -1):
    """
    Convenience wrapper: given ScanNet-style gt & pred dicts:

      gt = {
        "instance_ids": (N,),
        "label_ids": (N,),
        "instances": [ {...}, ... ],
      }
      pred = {
        "instance_ids": (N,),
        "label_ids": (N,),
        "instances": [ {...}, ... ],
      }

    compute:
      - semantic IoU + mIoU
      - instance AP per class + mAP per IoU threshold
    """
    gt_labels = gt["label_ids"]
    pred_labels = pred["label_ids"]
    gt_instances = gt["instances"]
    pred_instances = pred["instances"]

    per_class_iou, mIoU = compute_semantic_iou(
        gt_labels, pred_labels, ignore_label=ignore_label
    )

    inst_results = compute_instance_ap(
        gt_instances, pred_instances, iou_thresholds=iou_thresholds
    )

    return {
        "semantic": {
            "per_class_iou": per_class_iou,
            "mIoU": mIoU,
        },
        "instance": inst_results,
    }

def debug_semantic_overlap(gt: dict,
                           pred: dict,
                           ignore_label: int = -1,
                           top_k: int = 20):
    """
    Print basic stats to debug semantic alignment:
      - class distribution in GT and Pred
      - overall accuracy
      - per-class IoU
    """
    gt_labels = gt["label_ids"]
    pred_labels = pred["label_ids"]
    assert gt_labels.shape == pred_labels.shape

    valid = gt_labels != ignore_label
    gt_valid = gt_labels[valid]
    pred_valid = pred_labels[valid]

    print(f"[semantic] #vertices total: {gt_labels.shape[0]}")
    print(f"[semantic] #vertices valid: {gt_valid.shape[0]} "
          f"({gt_valid.shape[0] / gt_labels.shape[0]:.6f} of all)")

    overall_acc = (gt_valid == pred_valid).mean() if gt_valid.size > 0 else 0.0
    print(f"[semantic] overall accuracy (valid GT only): {overall_acc:.6f}")

    # class frequencies
    gt_counts = dict(zip(*np.unique(gt_valid, return_counts=True)))
    pred_counts = dict(zip(*np.unique(pred_valid, return_counts=True)))

    print("[semantic] top GT classes by count (class_id: count):")
    for c, cnt in sorted(gt_counts.items(), key=lambda x: -x[1])[:top_k]:
        print(f"  class {c:3d}: {cnt}")

    per_class_iou, mIoU = compute_semantic_iou(gt_labels, pred_labels, ignore_label)
    print(f"[semantic] mIoU: {mIoU:.6f}")
    print("[semantic] per-class IoU (class_id: IoU):")
    for c, iou in sorted(per_class_iou.items(), key=lambda x: -x[1])[:top_k]:
        print(f"  class {c:3d}: {iou:.6f}")

def debug_instance_overlap(gt: dict,
                           pred: dict,
                           iou_thresholds=(0.1, 0.25, 0.5)):
    """
    For each GT instance, find the best-overlapping predicted instance of the same class.
    Print statistics about those IoUs:
      - mean / median best IoU
      - fraction above several thresholds
    """
    gt_instances = gt["instances"]
    pred_instances = pred["instances"]

    print(f"[instance] #GT instances:   {len(gt_instances)}")
    print(f"[instance] #Pred instances: {len(pred_instances)}")

    # group preds by class for speed
    preds_by_class = defaultdict(list)
    for p in pred_instances:
        preds_by_class[int(p["class_id"])].append(p)

    best_ious = []

    for g in gt_instances:
        c = int(g["class_id"])
        g_verts = g["vert_indices"]
        pred_c = preds_by_class.get(c, [])
        if not pred_c:
            best_ious.append(0.0)
            continue

        best = 0.0
        for p in pred_c:
            p_verts = p["vert_indices"]
            inter = np.intersect1d(g_verts, p_verts).size
            union = g_verts.size + p_verts.size - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best:
                best = iou
        best_ious.append(best)

    best_ious = np.array(best_ious, dtype=np.float32)
    print(f"[instance] mean best IoU (per GT inst):   {best_ious.mean():.6f}")
    print(f"[instance] median best IoU (per GT inst): {np.median(best_ious):.6f}")

    for thr in iou_thresholds:
        frac = (best_ious >= thr).mean()
        print(f"[instance] fraction of GT instances with best IoU >= {thr}: {frac:.6f}")


def debug_full(gt: dict,
               pred: dict,
               iou_thresholds=(0.25, 0.5),
               ignore_label: int = -1):
    """
    Run semantic and instance debug checks and then the normal evaluation.
    """
    print("========== SEMANTIC DEBUG ==========")
    debug_semantic_overlap(gt, pred, ignore_label=ignore_label)

    print("\n========== INSTANCE DEBUG ==========")
    debug_instance_overlap(gt, pred, iou_thresholds=(0.1, 0.25, 0.5))

    print("\n========== FINAL METRICS ==========")


    results = evaluate_scannet_style(gt, pred,
                                     iou_thresholds=iou_thresholds,
                                     ignore_label=ignore_label)

    print(f"mIoU:      {results['semantic']['mIoU']:.8f}")
    for thr, v in results["instance"]["mAP"].items():
        print(f"mAP@{thr}: {v:.8f}")

    return results
