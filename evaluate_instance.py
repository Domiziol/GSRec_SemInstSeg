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
import os, sys
from copy import deepcopy
from uuid import uuid4
from plyfile import PlyData

from plyfile import PlyData
try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)




# Scannet part - not used 
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
opt['min_region_sizes']     = np.array( [ 100 ] ) # 100 for scannet
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


def assign_instances_from_meshes(
    gt_v, pred_v,pred_ply_path,
    ignore_mask,
    gt_class_field,
    gt_inst_field,
    pred_class_field,
    pred_inst_field,
):
    
    pred_ply_conf = pred_ply_path
    if ignore_mask is None:
        ignore_mask = np.array([], dtype=np.int32)
    else:
        ignore_mask = np.asarray(ignore_mask, dtype=np.int32)

   
    gt_sem = gt_v[gt_class_field].astype(np.int32)
    gt_inst = gt_v[gt_inst_field].astype(np.int32)
    pred_sem = pred_v[pred_class_field].astype(np.int32)
    pred_inst = pred_v[pred_inst_field].astype(np.int32)


    vtx_idx = np.arange(len(gt_sem))
    keep_vtx = ~np.isin(vtx_idx, ignore_mask)

    gt_sem, gt_inst = gt_sem[keep_vtx], gt_inst[keep_vtx]
    pred_sem, pred_inst = pred_sem[keep_vtx], pred_inst[keep_vtx]

    
    gt_ids = np.zeros_like(gt_sem, dtype=np.int32)
    validGT = (gt_sem > 0) & (gt_inst >= 0)
    gt_ids[validGT] = gt_sem[validGT] * 1000 + gt_inst[validGT]

    gt_instances = {name: [] for name in CLASS_LABELS}
    for inst_id in np.unique(gt_ids):
        if inst_id == 0:
            continue
        sem_id = int(inst_id // 1000)
        if sem_id not in VALID_CLASS_IDS:
            continue
        label = ID_TO_LABEL[sem_id]
        gt_instances[label].append({
            "instance_id": int(inst_id),
            "label_id": int(sem_id),
            "vert_count": int((gt_ids == inst_id).sum()),
            "med_dist": -1.0,
            "dist_conf": 0.0,
        })

    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt_inst in gt2pred[label]:
            gt_inst["matched_pred"] = []

    pred2gt = {label: [] for label in CLASS_LABELS}
    bool_void = ~np.isin(gt_ids // 1000, VALID_CLASS_IDS)
   
    num_pred_instances = 0
    for oid in np.unique(pred_inst):
        if oid < 0:
            continue
        pred_mask = (pred_inst == oid)
        vert_count = int(pred_mask.sum())
        if vert_count < opt["min_region_sizes"][0]: 
            continue

        
        sem_vals = pred_sem[pred_mask]
        sem_vals = sem_vals[sem_vals > 0]
        if sem_vals.size == 0:
            continue
        label_id = int(np.bincount(sem_vals).argmax())  # mode
        
        if label_id not in ID_TO_LABEL:
            continue
        label = ID_TO_LABEL[label_id]


        probs_file = np.load(pred_ply_conf+"/mapped_vertex_class_probs_onto_gt.npz")
        probs = probs_file["probs"]
        
        conf = probs[pred_mask, label_id - 1].mean()

        pred_instance = {
            "uuid": str(uuid4()),
            "pred_id": num_pred_instances,
            "label_id": int(label_id),
            "vert_count": vert_count,
            "confidence": conf,
            "void_intersection": int(np.count_nonzero(bool_void & pred_mask)),
            "matched_gt": [],
        }
        num_pred_instances += 1

        # match against GT instances of same class
        for gt_num, gt_inst in enumerate(gt2pred[label]):
            gt_mask = (gt_ids == gt_inst["instance_id"])
            inter = int(np.count_nonzero(gt_mask & pred_mask))
            if inter > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = inter
                pred_copy["intersection"] = inter

                pred_instance["matched_gt"].append(gt_copy)
                gt2pred[label][gt_num]["matched_pred"].append(pred_copy)

        pred2gt[label].append(pred_instance)

    return gt2pred, pred2gt

def eval_instance_from_two_meshes(
    gt_ply_path,
    pred_ply_path,
    mesh_name,
    ignore_mask=None,
    gt_class_field="class_id",
    gt_inst_field="object_id",
    pred_class_field="pred_class_id",
    pred_inst_field="pred_object_id",
):
    gt_v = PlyData.read(gt_ply_path)["vertex"].data
    pred_v = PlyData.read(pred_ply_path+mesh_name)["vertex"].data

    gt2pred, pred2gt = assign_instances_from_meshes(
        gt_v, pred_v, pred_ply_path,
        ignore_mask=ignore_mask,
        gt_class_field=gt_class_field,
        gt_inst_field=gt_inst_field,
        pred_class_field=pred_class_field,
        pred_inst_field=pred_inst_field
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

CLASS_LABELS = [
    "basket",
    "bed",
    "bench",
    "bin",
    "blanket",
    "blinds",
    "book",
    "bottle",
    "box",
    "bowl",
    "camera",
    "cabinet",
    "candle",
    "chair",
    "clock",
    "cloth",
    "comforter",
    "cushion",
    "desk",
    "desk-organizer",
    "door",
    "indoor-plant",
    "lamp",
    "monitor",
    "nightstand",
    "panel",
    "picture",
    "pillar",
    "pillow",
    "pipe",
    "plant-stand",
    "plate",
    "pot",
    "sculpture",
    "shelf",
    "sofa",
    "stool",
    "switch",
    "table",
    "tablet",
    "tissue-paper",
    "tv-screen",
    "tv-stand",
    "vase",
    "vent",
    "wall-plug",
    "window",
    "rug",
]

VALID_CLASS_IDS = np.asarray(
    [
        3,
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        22,
        23,
        26,
        29,
        34,
        35,
        37,
        44,
        47,
        52,
        54,
        56,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        70,
        71,
        76,
        78,
        79,
        80,
        82,
        83,
        87,
        88,
        91,
        92,
        95,
        97,
        98,
    ]
)
ID_TO_LABEL = {}
LABEL_TO_ID = {}

for pred_id, i in enumerate(range(len(VALID_CLASS_IDS))):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

gt_ply_path   = "./data/replica/scan1/mesh_semantic_verts_bothids.ply"
#pred_ply_path = "./experiments2_fromsam3/model_d8k/wdist=0.0_wemb=1.0_wsem=0.0_pdist85_pemb78_psem70_512/mapped_semantic_class_id_&_object_id_onto_gt.ply"

mesh_name = "/mapped_semantic_class_id_&_object_id_onto_gt.ply"

pred_ply_path = "./experiments3/model_d8k/wdist=0.2_wemb=0.6_wsem=0.2_eps=0.6_512"

avgs = eval_instance_from_two_meshes(
    gt_ply_path, pred_ply_path, mesh_name,
    gt_class_field="class_id",
    gt_inst_field="object_id",
    pred_class_field="pred_class_id",
    pred_inst_field="pred_object_id"
)
