import json
import numpy as np
from plyfile import PlyData
from collections import defaultdict
from sklearn.neighbors import KDTree
from scipy.special import softmax


# ---------------------------------------------------------
# 1. Load Replica mapping: object_id → class_id
# ---------------------------------------------------------

def load_objid_to_class(info_json_path: str) -> dict:
    """
    Parse info_semantics.json and build:
        object_id (instance id in mesh) -> class_id (semantic id).
    """
    with open(info_json_path, "r") as f:
        info = json.load(f)

    objid_to_class = {}
    for o in info.get("objects", []):
        objid_to_class[int(o["id"])] = int(o["class_id"])
    return objid_to_class


# ---------------------------------------------------------
# 2. Load Replica GT mesh and build ScanNet-style GT data
# ---------------------------------------------------------

def load_replica_gt(gt_mesh_path: str, info_json_path: str):
    """
    Load Replica mesh (with per-vertex object_id) and info_semantics.json

    Returns:
        verts: (N,3) vertex positions
        gt: dict with keys:
            - "instance_ids": (N,) int32 per-vertex instance id (0 = no instance)
            - "label_ids":    (N,) int32 per-vertex semantic class id (-1 = unknown)
            - "instances":    list of dicts:
                {
                  "instance_id": int,
                  "class_id":    int,
                  "vert_indices": np.ndarray[int],
                }
    """
    objid_to_class = load_objid_to_class(info_json_path)

    ply = PlyData.read(gt_mesh_path)

    verts = np.vstack([
        ply["vertex"]["x"],
        ply["vertex"]["y"],
        ply["vertex"]["z"],
    ]).T.astype(np.float32)

    # per-vertex object/instance id from PLY
    gt_object_id = np.array(ply["vertex"]["object_id"], dtype=np.int32)

    # ScanNet-style convention: 0 = "no instance"
    gt_instance_ids = gt_object_id.copy()
    gt_instance_ids[gt_instance_ids < 0] = 0

    # per-vertex semantic label via object_id -> class_id mapping
    gt_label_ids = np.full_like(gt_instance_ids, -1, dtype=np.int32)
    for i, oid in enumerate(gt_object_id):
        if oid < 0:
            continue
        gt_label_ids[i] = objid_to_class.get(int(oid), -1)

    # group vertices by instance_id
    inst_groups = defaultdict(list)
    for vid, inst_id in enumerate(gt_instance_ids):
        if inst_id == 0:
            continue
        inst_groups[int(inst_id)].append(vid)

    # build GT instance list
    gt_instances = []
    for inst_id, verts_idx in inst_groups.items():
        verts_idx = np.array(verts_idx, dtype=np.int32)
        class_id = int(gt_label_ids[verts_idx[0]])  # GT: single class per instance
        gt_instances.append({
            "instance_id": int(inst_id),
            "class_id": class_id,
            "vert_indices": verts_idx,
        })

    gt = {
        "instance_ids": gt_instance_ids,
        "label_ids": gt_label_ids,
        "instances": gt_instances,
    }

    return verts, gt


# ---------------------------------------------------------
# 3. Project anchor predictions onto GT vertices (strict)
# ---------------------------------------------------------

def row_softmax(Z):
    Z = Z.astype(np.float64, copy=True)
    Z -= Z.max(axis=1, keepdims=True)
    np.exp(Z, out=Z)
    Z /= Z.sum(axis=1, keepdims=True)
    return Z

def project_predictions_to_gt_vertices(
    verts: np.ndarray,
    anchor_points_transformed: np.ndarray,
    smoothed_logits: np.ndarray,
    pred_cluster_ids: np.ndarray,
):
    """
    Project anchor predictions (semantic logits + cluster IDs)
    onto GT mesh vertices via nearest neighbor.

    All of the following MUST have the same first dimension:
      - anchor_points_transformed.shape[0]
      - smoothed_logits.shape[0]
      - pred_cluster_ids.shape[0]

    DBSCAN convention assumed:
      - cluster_id == -1 → noise
      - cluster_id >= 0 → valid cluster

    We map this into ScanNet convention:
      - 0 → no instance
      - 1..K → valid instances
    """
    anchor_points_transformed = np.asarray(anchor_points_transformed, dtype=np.float32)
    smoothed_logits = np.asarray(smoothed_logits)
    pred_cluster_ids = np.asarray(pred_cluster_ids, dtype=np.int32)

    n_anchors = anchor_points_transformed.shape[0]
    if smoothed_logits.shape[0] != n_anchors:
        raise ValueError(
            f"smoothed_logits.shape[0] ({smoothed_logits.shape[0]}) "
            f"!= anchor_points_transformed.shape[0] ({n_anchors})"
        )
    if pred_cluster_ids.shape[0] != n_anchors:
        raise ValueError(
            f"pred_cluster_ids.shape[0] ({pred_cluster_ids.shape[0]}) "
            f"!= anchor_points_transformed.shape[0] ({n_anchors})"
        )

    # nearest anchor per vertex
    kdt = KDTree(anchor_points_transformed)
    _, nearest_anchor = kdt.query(verts, k=1)
    nearest_anchor = nearest_anchor[:, 0]  # (N,)

    # semantic prediction per vertex
    probs = row_softmax(smoothed_logits)           # (M,C)
    pred_semantic_ids = np.argmax(probs[nearest_anchor], axis=1).astype(np.int32)

    # instance prediction per vertex (from DBSCAN cluster IDs)
    cluster_ids = pred_cluster_ids.copy()

    # Map DBSCAN ids to ScanNet-style:
    #   -1 (noise) -> 0
    #   0..K-1     -> 1..K
    mapped_cluster_ids = np.zeros_like(cluster_ids, dtype=np.int32)
    valid_mask = cluster_ids >= 0
    mapped_cluster_ids[valid_mask] = cluster_ids[valid_mask] + 1

    pred_instance_ids = mapped_cluster_ids[nearest_anchor]

    return pred_semantic_ids, pred_instance_ids, probs, nearest_anchor


# ---------------------------------------------------------
# 4. Build prediction structures (one instance per cluster)
# ---------------------------------------------------------

def build_pred_structs_from_clusters(
    pred_semantic_ids: np.ndarray,
    pred_instance_ids: np.ndarray,
    probs: np.ndarray,
    nearest_anchor: np.ndarray,
):
    """
    Build prediction data structures such that:

      - Each DBSCAN cluster id (0..K-1) becomes exactly ONE predicted instance
        after mapping to ScanNet-style ids (1..K).
      - class_id for each instance = majority vote of semantic
        predictions over its vertices.
      - confidence is set to 1.0 for now.

    Returns:
        pred: dict with keys:
          - "instance_ids": (N,) per-vertex predicted instance id
          - "label_ids":    (N,) per-vertex predicted semantic id
          - "instances":    list of dicts:
              {
                "instance_id": int,
                "class_id":    int,
                "vert_indices": np.ndarray,
                "confidence":   float,
              }
    """
    inst_groups = defaultdict(list)
    for vid, inst in enumerate(pred_instance_ids):
        if inst == 0:          # 0 means "no instance"
            continue
        inst_groups[int(inst)].append(vid)

    pred_instances = []

    for inst_id, verts_idx in inst_groups.items():
        verts_idx = np.array(verts_idx, dtype=np.int32)

        # majority semantic class for this instance
        inst_classes = pred_semantic_ids[verts_idx]
        classes, counts = np.unique(inst_classes, return_counts=True)
        class_id = int(classes[np.argmax(counts)])

        pred_instances.append({
            "instance_id": int(inst_id),
            "class_id": class_id,
            "vert_indices": verts_idx,
            "confidence": 1.0,  # as requested
        })

    pred = {
        "instance_ids": pred_instance_ids,
        "label_ids": pred_semantic_ids,
        "instances": pred_instances,
    }

    return pred


# ---------------------------------------------------------
# 5. Build both GT and Pred structures together
# ---------------------------------------------------------

def build_replica_scannet_structs(
    gt_mesh_path: str,
    info_json_path: str,
    anchor_points_transformed: np.ndarray,
    smoothed_logits: np.ndarray,
    pred_cluster_ids: np.ndarray,
):
    
    # GT
    verts, gt = load_replica_gt(gt_mesh_path, info_json_path)

    # Project predictions
    pred_semantic_ids, pred_instance_ids, probs, nearest_anchor = \
        project_predictions_to_gt_vertices(
            verts,
            anchor_points_transformed,
            smoothed_logits,
            pred_cluster_ids,
        )

    # Pred structures (one instance per DBSCAN cluster)
    pred = build_pred_structs_from_clusters(
        pred_semantic_ids,
        pred_instance_ids,
        probs,
        nearest_anchor,
    )

    return verts, gt, pred


def load_pred_mesh_with_labels(pred_mesh_path: str):
    """
    Load predicted mesh that has per-vertex:
      - class_id
      - instance_id

    Returns:
        verts:  (N_pred, 3) float32
        cls:    (N_pred,) int32 class_id per vertex
        inst:   (N_pred,) int32 instance_id per vertex (may contain -1 for noise)
    """
    ply = PlyData.read(pred_mesh_path)

    verts = np.vstack([
        ply["vertex"]["x"],
        ply["vertex"]["y"],
        ply["vertex"]["z"],
    ]).T.astype(np.float32)

    vertex_class_ids = np.array(ply["vertex"]["class_id"], dtype=np.int32)
    vertex_instance_ids = np.array(ply["vertex"]["instance_id"], dtype=np.int32)

    return verts, vertex_class_ids, vertex_instance_ids

def project_pred_mesh_to_gt_vertices(gt_verts: np.ndarray,
                                     pred_verts: np.ndarray,
                                     pred_class_ids: np.ndarray,
                                     pred_instance_ids_raw: np.ndarray):
    """
    For each GT vertex, find the nearest predicted vertex and
    copy its class_id and instance_id.

    DBSCAN-style instance ids:
      -1 = noise
       0..K-1 = valid clusters

    We map them to ScanNet-style:
      0 = no instance
      1..M = valid instances (contiguous)
    """
    # KDTree on predicted vertices
    tree = KDTree(pred_verts)
    _, nn = tree.query(gt_verts, k=1)
    nn = nn[:, 0]  # (N_gt,)

    # per-GT-vertex semantic and raw instance id
    pred_semantic_ids = pred_class_ids[nn].astype(np.int32)
    inst_raw = pred_instance_ids_raw[nn].astype(np.int32)

    # map instance ids: -1 -> 0, 0..K-1 -> 1..M (contiguous)
    inst_mapped = np.zeros_like(inst_raw, dtype=np.int32)
    valid_mask = inst_raw >= 0
    valid_ids = np.unique(inst_raw[valid_mask])

    id_map = {old_id: new_id + 1 for new_id, old_id in enumerate(valid_ids)}
    for old_id, new_id in id_map.items():
        inst_mapped[inst_raw == old_id] = new_id

    return pred_semantic_ids, inst_mapped

def build_pred_from_vertex_labels(pred_semantic_ids: np.ndarray,
                                  pred_instance_ids: np.ndarray):
    """
    Build ScanNet-style prediction dict from per-vertex semantic + instance ids.

    Majority voting per instance is applied to ensure one class per instance.
    """
    pred_semantic_ids = pred_semantic_ids.copy()

    inst_groups = defaultdict(list)
    for vid, inst in enumerate(pred_instance_ids):
        if inst == 0:   # 0 = no instance
            continue
        inst_groups[int(inst)].append(vid)

    pred_instances = []

    for inst_id, vert_idx_list in inst_groups.items():
        vert_idx = np.array(vert_idx_list, dtype=np.int32)

        # majority class for this instance
        classes = pred_semantic_ids[vert_idx]
        uniq, counts = np.unique(classes, return_counts=True)
        class_id = int(uniq[np.argmax(counts)])

        # enforce majority class on all vertices of this instance
        pred_semantic_ids[vert_idx] = class_id

        pred_instances.append({
            "instance_id": int(inst_id),
            "class_id": class_id,
            "vert_indices": vert_idx,
            "confidence": 1.0,   # still constant as agreed
        })

    pred = {
        "instance_ids": pred_instance_ids,
        "label_ids": pred_semantic_ids,
        "instances": pred_instances,
    }
    return pred

def build_replica_structs_from_labeled_mesh(
    gt_mesh_path: str,
    info_json_path: str,
    pred_mesh_path: str,
):
    """
    Build ScanNet-style GT + Pred structures using:

      - GT: Replica mesh with vertex-level object_id -> class_id (as before)
      - Pred: reconstructed mesh that already has per-vertex class_id and instance_id

    Pred mesh is projected onto GT vertices via nearest-neighbor.
    """
    # 1) GT side (same as before)
    gt_verts, gt = load_replica_gt(gt_mesh_path, info_json_path)

    # 2) Pred mesh with per-vertex labels
    pred_verts, pred_cls, pred_inst_raw = load_pred_mesh_with_labels(pred_mesh_path)

    # 3) Project predicted labels onto GT vertices
    pred_semantic_ids, pred_instance_ids = project_pred_mesh_to_gt_vertices(
        gt_verts,
        pred_verts,
        pred_cls,
        pred_inst_raw,
    )

    # 4) Build ScanNet-style pred dict
    pred = build_pred_from_vertex_labels(pred_semantic_ids, pred_instance_ids)

    return gt_verts, gt, pred


