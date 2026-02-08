## owner: Dominika Ziolkiewicz

## THESIS
import numpy as np
import json
from plyfile import PlyData
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_class_names():
    classes= []

    with open("./data/replica/scan1/info_semantic.json", 'r') as classes_file:
        data = json.load(classes_file)

    for objects in data['classes']:
        classes.append(objects['name'])

    return classes

def semantic_metrics(gtClassId, predClassId, validIds, ignore_ids):

    use_other = True


    gtClassId = gtClassId.astype(np.int64)
    predClassId = predClassId.astype(np.int64)
    
    ign = np.isin(gtClassId, np.array(ignore_ids, dtype=np.int64))
    gtClassId = gtClassId[~ign]
    predClassId = predClassId[~ign]



    if validIds is not None:
        validIds = np.array(list(map(int, validIds)), dtype=np.int64)
        m = np.isin(gtClassId, validIds)
        gtClassId = gtClassId[m]
        predClassId = predClassId[m]
    else:
        validIds = np.unique(gtClassId)

    
    
    validIds = np.array(validIds, dtype=np.int64)
    validIdsSorted = np.sort(validIds)

    if use_other:
        otherID = -999999
        ids = np.concatenate([validIdsSorted, np.array([otherID], dtype=np.int64)])
    else:
        ids = validIdsSorted

    ids2 = {int(cid): i for i, cid in enumerate(ids)}
    K = len(ids)
    confMatrix = np.zeros((K, K), dtype=np.int64)



    if use_other:
        predInGt = np.where(np.isin(predClassId, validIdsSorted), predClassId, otherID)
    else:
        keep = np.isin(predClassId, validIdsSorted)
        gtClassId = gtClassId[keep]
        predInGt = predClassId[keep]

    
    gtIds = np.vectorize(ids2.get)(gtClassId)
    prIds = np.vectorize(ids2.get)(predInGt)
    np.add.at(confMatrix, (gtIds, prIds), 1)

    
    validK = len(validIdsSorted)
    confMatValid = confMatrix[:validK, :validK]
    tp = np.diag(confMatValid).astype(np.float64)
    gt_count = confMatValid.sum(axis=1).astype(np.float64)
    

    if use_other:
        gtCountAll = confMatrix[:validK, :].sum(axis=1).astype(np.float64)
    else:
        gtCountAll = gt_count

    fn = gtCountAll - tp
    fp = (confMatrix[:validK, :validK].sum(axis=0).astype(np.float64) - tp)

    iou = tp / np.maximum(tp + fp + fn, 1.0)
    acc_class = tp / np.maximum(gtCountAll, 1.0)

    
    present = gtCountAll > 0
    mIoU = float(np.mean(iou[present])) if np.any(present) else float("nan")
    mAcc = float(np.mean(acc_class[present])) if np.any(present) else float("nan")

    # overall_acc = float((gt_v == pred_v).mean()) if gt_v.size else float("nan")
    predInGtIds = np.where(np.isin(predClassId, validIdsSorted), predClassId, otherID)
    overall_acc = float((gtClassId == predInGtIds).mean()) if gtClassId.size else float("nan")


    return {
        "valid_ids": validIdsSorted,
        "cm": confMatrix,
        "per_class": {
            "tp": tp.astype(np.int64),
            "fp": fp.astype(np.int64),
            "fn": fn.astype(np.int64),
            "iou": iou,
            "acc": acc_class,
            "gt_count": gtCountAll.astype(np.int64),
        },
        "overall_acc": overall_acc,
        "mIoU": mIoU,
        "mAcc": mAcc,
    }



if __name__ == "__main__":

    gt_mesh_path = "./data/replica/scan1/mesh_semantic_verts_bothids.ply"
    pred_mesh_path = f"./experiments2_fromsam3/model_d8k/wdist=0.0_wemb=1.0_wsem=0.0_pdist85_pemb78_psem70_512/mapped_semantic_class_id_&_object_id_onto_gt.ply"
    # pred_mesh_path = f"./outputs/final_sam3/d8k_l01/mapped_semantic_class_id_&_object_id_onto_gt.ply" # raw, not smoothed

    pred = PlyData.read(pred_mesh_path)["vertex"].data
    gt = PlyData.read(gt_mesh_path)["vertex"].data

    predClassId = pred["pred_class_id"].astype(np.int32)
    gtClassId = gt["class_id"].astype(np.int32)
    print(np.unique(gtClassId))

    VALID_IDS = [3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47, 59, 60, 63, 64, 65, 76, 78, 79 ,80, 91, 92, 93, 95, 97, 98] # exact GT

    # VALID_IDS = [2, 3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47, 59, 60, 61, 63, 64, 65, 75, 76, 78, 79 ,80, 91, 92, 93, 95, 97, 98, 100] # lookup

    # VALID_IDS = [2, 3, 11,   18,  29, 31, 37, 40, 44, 47, 59,61, 63,92, 95,98, 100]
    # VALID_IDS = [12,13,19,20, 60,   64, 65, 75, 76, 78, 79 ,80, 91,  93, 95, 97,]
    class_names = get_class_names()

    #VALID_IDS = [2, 3, 11, 12, 13, 18, 19, 20, 23, 29, 31, 34,  37, 40, 41, 44, 47, 54, 56,  59, 60, 61,  63, 64, 65, 75, 76, 78, 79, 80, 91, 92, 93, 95, 97, 98, 100]

    

    metrics = semantic_metrics(
        gtClassId, predClassId,
        validIds=VALID_IDS,
        ignore_ids=(-1, -2, 0),
    )

    print(f"Accuracy: {metrics['overall_acc']*100:.2f}%")
    print(f"mAcc: {metrics['mAcc']*100:.2f}%")
    print(f"mIoU: {metrics['mIoU']*100:.2f}%")

    # per-class
    validIds = metrics["valid_ids"]
    pc = metrics["per_class"]
    for i, cid in enumerate(validIds):
        name = class_names[cid-1] if 1 <= cid <= len(class_names) else "unknown"
        print(
            f"class name {cid:3d} ({name}): "
            f"IoU = {pc['iou'][i]*100:6.2f}%  "
            f"Acc = {pc['acc'][i]*100:6.2f}%  "
            f"(TP = {pc['tp'][i]}, FP = {pc['fp'][i]}, FN = {pc['fn'][i]}, GT = {pc['gt_count'][i]})"
        )

    ignoreIds = np.array([-1, -2, 0], dtype=np.int32)
    validIds = np.array(VALID_IDS, dtype=np.int32)

    
    gtClassValid = ~np.isin(gtClassId, ignoreIds)
    gtClassValidIds = gtClassId[gtClassValid]
    predClassValidIds = predClassId[gtClassValid]

    
    m2 = np.isin(gtClassValidIds, validIds)
    gt_f = gtClassValidIds[m2]
    predClassValidIds = predClassValidIds[m2]

    
    OTHER = 102
    gtDisplay = np.where(np.isin(gtClassValidIds, validIds), gtClassValidIds, OTHER)
    predDisplay = np.where(np.isin(predClassValidIds, validIds), predClassValidIds, OTHER)


    show_ids = np.array(VALID_IDS, dtype=np.int32)
    labels = np.concatenate(([OTHER], show_ids))

    display_labels = (["OTHER"] +[f"{cid}: {class_names[cid-1]}" if 1 <= cid <= len(class_names) else f"{cid}: unknown" for cid in show_ids])

    cm = confusion_matrix(gtDisplay, predDisplay, labels=labels, normalize="true")  #row-normalized

    figure, ax = plt.subplots(figsize=(0.45 * len(labels) + 4, 0.45 * len(labels) + 3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, xticks_rotation=45, values_format=".2f", cmap="Blues", colorbar=True)
    ax.set_title("Confusion matrix (row-normalized)")
    plt.tight_layout()
    plt.show()

    