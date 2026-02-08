import numpy as np
import json
from plyfile import PlyData, PlyElement
from collections import Counter

def get_classes():
    classes= []

    with open("info_semantic.json", 'r') as classes_file:
        data = json.load(classes_file)

    for objects in data['classes']:
        classes.append(objects['name'])

    return classes

def load_objid_to_class():
    # map object_id (instance) to class_id (semantics)
    with open("info_semantic.json", 'r') as f:
        data = json.load(f)

    objid_to_class = {}
    for object in data["objects"]:
        objid_to_class[object["id"]] = object["class_id"]

    return objid_to_class

   
if __name__ == "__main__":


    gt_mesh_path = "./data/replica/scan1/mesh_semantic_verts_bothids.ply"
    gt_info_sem_path = "./data/replica/scan1/info_semantic.json"


    # === ADD TO GT MESH THE CLASS_ID PER VERTEX ===
    with open(gt_info_sem_path, "r") as f:
        info = json.load(f)

    id_to_label = np.asarray(info["id_to_label"], dtype=np.int32)
    print(len(id_to_label))
    uniq_all = np.unique(id_to_label)
    print("unique class_id:", uniq_all)

    ply = PlyData.read(gt_mesh_path)
    vertex = ply["vertex"].data

    objectId = vertex["object_id"].astype(np.int32)
    classId = np.full(objectId.shape, -1, dtype=np.int32)
    valid = (objectId >= 0) & (objectId < len(id_to_label))
    classId[valid] = id_to_label[objectId[valid]]

    new_dtype = [(n, vertex.dtype.fields[n][0]) for n in vertex.dtype.names if n != "class_id"] + [("class_id", "i4")]
    newVertex = np.empty(vertex.shape, dtype=new_dtype)

    for n in vertex.dtype.names:
        if n == "class_id":
            continue
        newVertex[n] = vertex[n]
    newVertex["class_id"] = classId

    vertex_el = PlyElement.describe(newVertex, "vertex")
    elements = [vertex_el]
    if "face" in ply:
        elements.append(ply["face"])

    PlyData(elements, text=ply.text).write("./data/replica/scan1/mesh_semantic_verts_bothids.ply")
    print("unique object_id:", len(np.unique(objectId)))
    print("unique class_id:", len(np.unique(classId)), "min/max:", int(classId.min()), int(classId.max()))
    # === END ADD TO GT MESH THE CLASS_ID PER VERTEX ===




    # ==== Process GT mesh and apply object_id from faces to vertices ====
    ply = PlyData.read(gt_mesh_path)
    faces = ply["face"].data
    vertices = ply["vertex"].data

    # print("face properties:", faces.dtype.names)

    faceObjectId = faces["object_id"]
    counts = []
    uniqueObjectIds, _ = np.unique(faceObjectId, return_counts=True)

    print(f"unique object_ids: {len(uniqueObjectIds)}")
    print(f"min: {int(uniqueObjectIds.min())}, max: {int(uniqueObjectIds.max())}\n")

    vertexCount = len(vertices)
    votes = [[] for _ in range(vertexCount)]

    for f in faces:
        vertexId = f["vertex_indices"]
        faceObjectId= int(f["object_id"])
        for v in vertexId:
            votes[int(v)].append(faceObjectId)

    vertexObjectId = np.full(vertexCount, -1, dtype=np.int32)
    
    for v in range(vertexCount):
        if not votes[v]:
            continue
        c = Counter(votes[v])
        best_oid, best_count = c.most_common(1)[0]
        vertexObjectId[v] = best_oid

    new_dtype = vertices.dtype.descr + [("object_id", "i4")]

    verts_out = np.empty(vertexCount, dtype=new_dtype)
    for name in vertices.dtype.names:
        verts_out[name] = vertices[name]
    verts_out["object_id"] = vertexObjectId

    vertexElement = PlyElement.describe(verts_out, "vertex")
    faceElement = PlyElement.describe(faces, "face")

    PlyData([vertexElement, faceElement], text=ply.text).write("./data/replica/scan1/mesh_semantic_verts.ply")
    print("Saved")
    # ==== END Process GT mesh and apply object_id from faces to vertices ====
