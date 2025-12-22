# Evaluates semantic instance task
# Adapted from the CityScapes evaluation: https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/evaluation
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Each .txt prediction file look like:
#    [(pred0) rel. path to pred. mask over verts as .txt] [(pred0) label id] [(pred0) confidence]
#    [(pred1) rel. path to pred. mask over verts as .txt] [(pred1) label id] [(pred1) confidence]
#    [(pred2) rel. path to pred. mask over verts as .txt] [(pred2) label id] [(pred2) confidence]
#    ...
#
# NOTE: The prediction files must live in the root of the given prediction path.
#       Predicted mask .txt files must live in a subfolder.
#       Additionally, filenames must not contain spaces.
# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the *_vh_clean_2.ply (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.
#
# example usage: evaluate_semantic_instance.py --scan_path [path to scan data] --output_file [output file]

# python imports
import math
import os, sys, argparse
import inspect
from copy import deepcopy
from uuid import uuid4
from plyfile import PlyData
import json
import torch
from plyfile import PlyData, PlyElement
try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

from scipy import stats

def save_colored_instance_ply(
    base_ply_path: str,
    out_ply_path: str,
    instance_mask: np.ndarray,
    gt_mask: np.ndarray | None = None,
    dim_color=(120, 120, 120),
    inst_color=(255, 60, 60),
    gt_color=(60, 255, 60),
):
    """
    Saves a colored copy of base_ply_path:
      - dim_color for all vertices
      - inst_color for instance_mask
      - gt_color for gt_mask (optional). If both overlap, instance color wins.
    Works for typical ScanNet-like PLYs with vertex x,y,z.
    """
    ply = PlyData.read(base_ply_path)
    v = ply["vertex"].data

    n = len(v)
    instance_mask = np.asarray(instance_mask, dtype=bool)
    assert instance_mask.shape[0] == n, f"mask length {instance_mask.shape[0]} != n_verts {n}"

    if gt_mask is not None:
        gt_mask = np.asarray(gt_mask, dtype=bool)
        assert gt_mask.shape[0] == n

    # Build new vertex dtype with rgb if missing
    names = v.dtype.names
    has_rgb = all(k in names for k in ("red", "green", "blue"))

    if has_rgb:
        v2 = v.copy()
    else:
        # create extended dtype
        new_dtype = v.dtype.descr + [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        v2 = np.empty(n, dtype=new_dtype)
        for name in names:
            v2[name] = v[name]

    # paint everything dim first
    v2["red"] = np.uint8(dim_color[0])
    v2["green"] = np.uint8(dim_color[1])
    v2["blue"] = np.uint8(dim_color[2])

    # optionally paint GT overlap region
    if gt_mask is not None:
        v2["red"][gt_mask] = np.uint8(gt_color[0])
        v2["green"][gt_mask] = np.uint8(gt_color[1])
        v2["blue"][gt_mask] = np.uint8(gt_color[2])

    # paint predicted instance (wins if overlaps)
    v2["red"][instance_mask] = np.uint8(inst_color[0])
    v2["green"][instance_mask] = np.uint8(inst_color[1])
    v2["blue"][instance_mask] = np.uint8(inst_color[2])

    # Write out: preserve other elements (faces) if present
    new_elems = []
    for e in ply.elements:
        if e.name == "vertex":
            new_elems.append(PlyElement.describe(v2, "vertex"))
        else:
            new_elems.append(e)

    PlyData(new_elems, text=False).write(out_ply_path)
    print(f"[saved] {out_ply_path}")



def setup_replica_labels(info_semantic_path, valid_ids=None):
   
    data = json.load(open(info_semantic_path, "r"))
    names0 = [c["name"].replace("_", " ") for c in data["classes"]]  # len K

    K = len(names0)
    if valid_ids is None:
        valid_ids = list(range(1, K + 1))  # all classes 1..K

    CLASS_LABELS = [names0[i - 1] for i in valid_ids]
    VALID_CLASS_IDS = np.array(valid_ids, dtype=np.int32)

    ID_TO_LABEL = {cid: names0[cid - 1] for cid in valid_ids}
    LABEL_TO_ID = {names0[cid - 1]: cid for cid in valid_ids}

    return CLASS_LABELS, VALID_CLASS_IDS, ID_TO_LABEL, LABEL_TO_ID


#parser = argparse.ArgumentParser()
#parser.add_argument('--gt_path', default='', help='path to directory of gt .txt files')
#parser.add_argument('--output_file', default='', help='output file [default: ./semantic_instance_evaluation.txt]')
#opt = parser.parse_args()

#if opt.output_file == '':
#    opt.output_file = os.path.join(os.getcwd(), 'semantic_instance_evaluation.txt')


# ---------- Label info ---------- #
# CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
# VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
# ID_TO_LABEL = {}
# LABEL_TO_ID = {}
# for i in range(len(VALID_CLASS_IDS)):
#     LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
#     ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt = {}
opt['overlaps']             = np.append(np.arange(0.5,0.95,0.05), 0.25)
# minimum region size for evaluation [verts]
opt['min_region_sizes']     = np.array( [ 10 ] ) # 100 for scannet
# distance thresholds [m]
opt['distance_threshes']    = np.array( [  float('inf') ] )
# distance confidences
opt['distance_confs']       = np.array( [ -float('inf') ] )


def evaluate_matches(matches):
    overlaps = opt['overlaps']
    min_region_sizes = [ opt['min_region_sizes'][0] ]
    dist_threshes = [ opt['distance_threshes'][0] ]
    dist_confs = [ opt['distance_confs'][0] ]

    # results: class x overlap
    ap = np.zeros( (len(dist_threshes) , len(CLASS_LABELS) , len(overlaps)) , float )
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'uuid' in p:
                                pred_visited[p['uuid']] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [ gt for gt in gt_instances if gt['instance_id']>=1000 and gt['vert_count']>=min_region_size and gt['med_dist']<=distance_thresh and gt['dist_conf']>=distance_conf ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true  = np.ones ( len(gt_instances) )
                    cur_score = np.ones ( len(gt_instances) ) * (-float("inf"))
                    cur_match = np.zeros( len(gt_instances) , dtype=bool )
                    # collect matches
                    for (gti,gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['uuid']]:
                                continue
                            overlap = float(pred['intersection']) / (gt['vert_count']+pred['vert_count']-pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max( cur_score[gti] , confidence )
                                    min_score = min( cur_score[gti] , confidence )
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true  = np.append(cur_true,0)
                                    cur_score = np.append(cur_score,min_score)
                                    cur_match = np.append(cur_match,True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['uuid']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true  = cur_true [ cur_match==True ]
                    cur_score = cur_score[ cur_match==True ]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (gt['vert_count']+pred['vert_count']-gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist']>distance_thresh or gt['dist_conf']<distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore)/pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true,0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score,confidence)

                    # append to overall results
                    y_true  = np.append(y_true,cur_true)
                    y_score = np.append(y_score,cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort      = np.argsort(y_score)
                    y_score_sorted      = y_score[score_arg_sort]
                    y_true_sorted       = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds,unique_indices) = np.unique( y_score_sorted , return_index=True )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples      = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    # num_true_examples = y_true_sorted_cumsum[-1]
                    num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    precision         = np.zeros(num_prec_recall)
                    recall            = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append( y_true_sorted_cumsum , 0 )
                    # deal with remaining
                    for idx_res,idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores-1]
                        tp = num_true_examples - cumsum
                        fp = num_examples      - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p  = float(tp)/(tp+fp)
                        r  = float(tp)/(tp+fn)
                        precision[idx_res] = p
                        recall   [idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall   [-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv,[-0.5,0,0.5],'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di,li,oi] = ap_current
    return ap

def compute_averages(aps):
    d_inf = 0
    o50   = np.where(np.isclose(opt['overlaps'],0.5))
    o25   = np.where(np.isclose(opt['overlaps'],0.25))
    oAllBut25  = np.where(np.logical_not(np.isclose(opt['overlaps'],0.25)))
    avg_dict = {}
    #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
    avg_dict["classes"]  = {}
    for (li,label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name]             = {}
        #avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,oAllBut25])
        avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ d_inf,li,o50])
        avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ d_inf,li,o25])
    return avg_dict

def make_pred_info(pred: dict):
    # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
    pred_info = {}
    assert(pred['pred_classes'].shape[0] == pred['pred_scores'].shape[0] == pred['pred_masks'].shape[1])
    for i in range(len(pred['pred_classes'])):
        info = {}
        info["label_id"] = pred['pred_classes'][i]
        info["conf"] = pred['pred_scores'][i]
        info["mask"] = pred['pred_masks'][:,i]
        pred_info[uuid4()] = info # we later need to identify these objects
    return pred_info

def assign_instances_for_scan_from_meshes(
    gt_v, pred_v,
    ignore_mask=None,
    gt_class_field="class_id",
    gt_obj_field="object_id",
    pred_class_field="pred_class_id",
    pred_obj_field="pred_object_id",
    pred_conf_field=None,
    conf_default=1.0,
):
    
    if ignore_mask is None:
        ignore_mask = np.array([], dtype=np.int32)
    else:
        ignore_mask = np.asarray(ignore_mask, dtype=np.int32)

    # --- read fields ---
    gt_sem = gt_v[gt_class_field].astype(np.int32)
    gt_obj = gt_v[gt_obj_field].astype(np.int32)
    pr_sem = pred_v[pred_class_field].astype(np.int32)
    pr_obj = pred_v[pred_obj_field].astype(np.int32)

    # --- apply ignore mask (same semantics as original) ---
    vtx_idx = np.arange(len(gt_sem))
    keep_vtx = ~np.isin(vtx_idx, ignore_mask)

    gt_sem, gt_obj = gt_sem[keep_vtx], gt_obj[keep_vtx]
    pr_sem, pr_obj = pr_sem[keep_vtx], pr_obj[keep_vtx]

    # optional per-vertex confidence
    if pred_conf_field is not None and pred_conf_field in pred_v.dtype.names:
        pr_conf_v = pred_v[pred_conf_field].astype(np.float32)[keep_vtx]
    else:
        pr_conf_v = None

    # --- build ScanNet-style GT instance ids: sem*1000 + inst ---
    gt_ids = np.zeros_like(gt_sem, dtype=np.int32)
    ok_gt = (gt_sem > 0) & (gt_obj >= 0)   # negatives => void (0)
    gt_ids[ok_gt] = gt_sem[ok_gt] * 1000 + gt_obj[ok_gt]

    # --- init GT instances per class (ScanNet style) ---
    gt_instances = {name: [] for name in CLASS_LABELS}
    for inst_id in np.unique(gt_ids):
        if inst_id == 0:
            continue
        sem_id = int(inst_id // 1000)
        if sem_id not in VALID_CLASS_IDS:
            continue
        label_name = ID_TO_LABEL[sem_id]
        gt_instances[label_name].append({
            "instance_id": int(inst_id),
            "label_id": int(sem_id),
            "vert_count": int((gt_ids == inst_id).sum()),
            "med_dist": -1.0,
            "dist_conf": 0.0,
        })

    gt2pred = deepcopy(gt_instances)
    for label_name in gt2pred:
        for gt_inst in gt2pred[label_name]:
            gt_inst["matched_pred"] = []

    pred2gt = {label_name: [] for label_name in CLASS_LABELS}

    # invalid GT semantic mask (used to compute void_intersection)
    bool_void = ~np.isin(gt_ids // 1000, VALID_CLASS_IDS)

    # debug_class_id = 97
    # --- build predicted instances from pred_object_id (one instance per object_id) ---
    num_pred_instances = 0
    for oid in np.unique(pr_obj):
        if oid < 0:
            continue
        pred_mask = (pr_obj == oid)
        vert_count = int(pred_mask.sum())
        # if vert_count < opt["min_region_sizes"][0]:
        #     continue

        # pick semantic label for this predicted instance (mode over its vertices)
        sem_vals = pr_sem[pred_mask]
        sem_vals = sem_vals[sem_vals > 0]
        if sem_vals.size == 0:
            continue
        label_id = int(np.bincount(sem_vals).argmax())  # mode
        # ===
        # if debug_class_id is not None and label_id == debug_class_id:
        #     print(f"\n[PRED inst oid={oid}] label_id={label_id} vert_count={vert_count}")

        #     # show top-5 semantic votes inside this instance
        #     bc = np.bincount(sem_vals)
        #     top = np.argsort(bc)[::-1][:5]
        #     top = [(int(k), int(bc[k])) for k in top if bc[k] > 0]
        #     print("  vote top:", top)

        #     print("  void_intersection:", int(np.count_nonzero(bool_void & pred_mask)))
        #     out_dir = "./instance_debug_meshes"
        #     os.makedirs(out_dir, exist_ok=True)

        #     # IMPORTANT: pred_mask here is on the *filtered* vertex set (keep_vtx applied).
        #     # We need a mask on the *original* vertex indexing to write colors correctly.
        #     full_pred_mask = np.zeros(len(pred_v[pred_class_field]), dtype=bool)  # careful: pred_v is structured array
        #     full_pred_mask[keep_vtx] = pred_mask

        #     # Optionally, also color the best-overlapping GT instance (green)
        #     best_gt_mask_full = None
        #     best_iou = -1.0

        #     best_gt_id = None
        #     best_inter = 0
        #     best_union = 0

        #     for gt in gt2pred[label_name]:
        #         gt_mask_small = (gt_ids == gt["instance_id"])   # this is on filtered vertices
        #         inter = np.count_nonzero(gt_mask_small & pred_mask)
        #         if inter == 0:
        #             continue
        #         union = gt["vert_count"] + vert_count - inter
        #         iou = inter / union
        #         if iou > best_iou:
        #             best_iou = iou
        #             best_gt_id = gt["instance_id"]
        #             best_inter = int(inter)
        #             best_union = int(union)
        #             best_gt_mask_full = np.zeros(len(full_pred_mask), dtype=bool)
        #             best_gt_mask_full[keep_vtx] = gt_mask_small
        #     if best_iou < 0:
        #         print("  best_iou: (no GT intersection)")
        #     else:
        #         print(f"  best_iou: {best_iou:.6f} (GT inst {best_gt_id}, inter={best_inter}, union={best_union})")
        # ===
        if label_id not in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]

        # confidence per instance
        if pr_conf_v is not None:
            conf = float(np.mean(pr_conf_v[pred_mask]))
        else:
            conf = float(conf_default)

        pred_instance = {
            "uuid": str(uuid4()),  # ScanNet eval uses 'uuid' here
            "pred_id": num_pred_instances,
            "label_id": int(label_id),
            "vert_count": vert_count,
            "confidence": conf,
            "void_intersection": int(np.count_nonzero(bool_void & pred_mask)),
            "matched_gt": [],
        }
        num_pred_instances += 1

        # match against GT instances of same class
        for gt_num, gt_inst in enumerate(gt2pred[label_name]):
            gt_mask = (gt_ids == gt_inst["instance_id"])
            inter = int(np.count_nonzero(gt_mask & pred_mask))
            if inter > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = inter
                pred_copy["intersection"] = inter

                pred_instance["matched_gt"].append(gt_copy)
                gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)

        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt

def eval_instance_from_two_meshes(
    gt_ply_path,
    pred_ply_path,
    ignore_mask=None,
    gt_class_field="class_id",
    gt_obj_field="object_id",
    pred_class_field="pred_class_id",
    pred_obj_field="pred_object_id",
    pred_conf_field=None,
    conf_default=1.0,
):
    gt_v = PlyData.read(gt_ply_path)["vertex"].data
    pred_v = PlyData.read(pred_ply_path)["vertex"].data

    gt2pred, pred2gt = assign_instances_for_scan_from_meshes(
        gt_v, pred_v,
        ignore_mask=ignore_mask,
        gt_class_field=gt_class_field,
        gt_obj_field=gt_obj_field,
        pred_class_field=pred_class_field,
        pred_obj_field=pred_obj_field,
        pred_conf_field=pred_conf_field,
        conf_default=conf_default,
    )

    matches = {os.path.abspath(gt_ply_path): {"gt": gt2pred, "pred": pred2gt}}
    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)
    print_results(avgs)
    return avgs


def print_results(avgs):
    sep     = ""
    col1    = ":"
    lineLen = 64

    print("")
    print("#"*lineLen)
    line  = ""
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_25%"    ) + sep
    print(line)
    print("#"*lineLen)

    for (li,label_name) in enumerate(CLASS_LABELS):
        ap_avg  = avgs["classes"][label_name]["ap"]
        ap_50o  = avgs["classes"][label_name]["ap50%"]
        ap_25o  = avgs["classes"][label_name]["ap25%"]
        line  = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg ) + sep
        line += sep + "{:>15.3f}".format(ap_50o ) + sep
        line += sep + "{:>15.3f}".format(ap_25o ) + sep
        print(line)

    all_ap_avg  = avgs["all_ap"]
    all_ap_50o  = avgs["all_ap_50%"]
    all_ap_25o  = avgs["all_ap_25%"]

    print("-"*lineLen)
    line  = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg)  + sep
    line += "{:>15.3f}".format(all_ap_50o)  + sep
    line += "{:>15.3f}".format(all_ap_25o)  + sep
    print(line)
    print("")


def write_result_file(avgs, filename):
    _SPLITTER = ','
    with open(filename, 'w') as f:
        f.write(_SPLITTER.join(['class', 'class id', 'ap', 'ap50', 'ap25']) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(_SPLITTER.join([str(x) for x in [class_name, class_id, ap, ap50, ap25]]) + '\n')


INFO_SEM = "./data/replica/scan1/info_semantic.json"

# choose which class ids you want to evaluate AP on
# valid_ids = [1, 2, 3, ...]  # optional subset
valid_ids = [3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47, 59, 60, 63, 64, 65, 76, 78, 79, 80, 91, 92, 93, 95, 97, 98]  # exactly valid GT
# valid_ids = [12, 59]

CLASS_LABELS, VALID_CLASS_IDS, ID_TO_LABEL, LABEL_TO_ID = setup_replica_labels(INFO_SEM, valid_ids)
gt_ply_path   = "./data/replica/scan1/mesh_semantic_verts_bothids.ply"
pred_ply_path = "./experiments2_fromsam3/model_d8k/wdist=0.0_wemb=1.0_wsem=0.0_pdist85_pemb78_psem70_512/mapped_semantic_class_id_&_object_id_onto_gt.ply"


# ply = PlyData.read(gt_ply_path)
# v = ply["vertex"].data
# print(np.unique(v["class_id"].astype(np.int32)))
# # ensure RGB fields exist
# if not all(k in v.dtype.names for k in ("red","green","blue")):
#     v2 = np.empty(len(v), dtype=v.dtype.descr + [("red","u1"),("green","u1"),("blue","u1")])
#     for k in v.dtype.names: v2[k] = v[k]
# else:
#     v2 = v.copy()
# # gray background
# v2["red"], v2["green"], v2["blue"] = 0, 0, 0
# # red instance
# v2["red"][mask], v2["green"][mask], v2["blue"][mask] = 255, 255, 255

# ply2 = PlyData([PlyElement.describe(v2, "vertex")] + [e for e in ply.elements if e.name != "vertex"], text=False)
# ply2.write("./instance_debug_meshes/test.ply")

avgs = eval_instance_from_two_meshes(
    gt_ply_path, pred_ply_path,
    gt_class_field="class_id",
    gt_obj_field="object_id",
    pred_class_field="pred_class_id",
    pred_obj_field="pred_object_id",
    pred_conf_field=None,
    conf_default=1.0,
)
