import numpy as np
from plyfile import PlyData, PlyElement
from collections import defaultdict


def load_replica_mesh(mesh_ply_path):
    """Load Replica mesh_semantic.ply and extract per-face object IDs."""
    ply = PlyData.read(mesh_ply_path)

    # Vertex attributes
    verts = np.vstack([
        ply["vertex"]["x"],
        ply["vertex"]["y"],
        ply["vertex"]["z"]
    ]).T

    # Colors, normals if needed
    normals = None
    if "nx" in ply["vertex"].data.dtype.names:
        normals = np.vstack([
            ply["vertex"]["nx"],
            ply["vertex"]["ny"],
            ply["vertex"]["nz"],
        ]).T

    colors = None
    if "red" in ply["vertex"].data.dtype.names:
        colors = np.vstack([
            ply["vertex"]["red"],
            ply["vertex"]["green"],
            ply["vertex"]["blue"],
        ]).T

    # Faces
    faces = np.array(ply["face"]["vertex_indices"].tolist(), dtype=object)
    face_obj_ids = np.array(ply["face"]["object_id"], dtype=np.int32)

    return verts, normals, colors, faces, face_obj_ids


def compute_vertex_object_ids(num_vertices, faces, face_obj_ids):
    """Assign each vertex an object_id using majority vote from adjacent faces."""
    adjacency = defaultdict(list)

    for face_id, face in enumerate(faces):
        obj = int(face_obj_ids[face_id])
        for v in face:
            adjacency[v].append(obj)

    vertex_obj_ids = np.zeros(num_vertices, dtype=np.int32)

    for v in range(num_vertices):
        objs = adjacency[v]
        if len(objs) == 0:
            vertex_obj_ids[v] = -1  # no face touches this vertex
        else:
            # majority vote
            vertex_obj_ids[v] = max(set(objs), key=objs.count)

    return vertex_obj_ids


def save_vertex_object_mesh(out_path, verts, normals, colors, vertex_obj_ids, faces, face_obj_ids):
    """Save a new PLY file with vertex-level object_id."""

    # Build vertex data structure
    vertex_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4")
    ]

    if normals is not None:
        vertex_dtype += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]

    if colors is not None:
        vertex_dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    vertex_dtype += [("object_id", "i4")]

    vertex_array = np.empty(len(verts), dtype=vertex_dtype)

    vertex_array["x"] = verts[:, 0]
    vertex_array["y"] = verts[:, 1]
    vertex_array["z"] = verts[:, 2]

    if normals is not None:
        vertex_array["nx"] = normals[:, 0]
        vertex_array["ny"] = normals[:, 1]
        vertex_array["nz"] = normals[:, 2]

    if colors is not None:
        vertex_array["red"] = colors[:, 0]
        vertex_array["green"] = colors[:, 1]
        vertex_array["blue"] = colors[:, 2]

    vertex_array["object_id"] = vertex_obj_ids

    # Build face data structure
    face_dtype = [("vertex_indices", "O"), ("object_id", "i4")]

    face_array = np.empty(len(faces), dtype=face_dtype)
    for i in range(len(faces)):
        face_array["vertex_indices"][i] = faces[i]      # list or np.array ok
        face_array["object_id"][i] = face_obj_ids[i]
    

    # Wrap as PlyElements
    el_verts = PlyElement.describe(vertex_array, "vertex")
    el_faces = PlyElement.describe(face_array, "face")

    # Write final PLY
    PlyData([el_verts, el_faces], text=False).write(out_path)

    print(f"Saved new mesh with vertex object_id → {out_path}")



def convert_mesh_with_vertex_object_ids(in_ply, out_ply):
    print("Loading mesh:", in_ply)
    verts, normals, colors, faces, face_obj_ids = load_replica_mesh(in_ply)

    print("Computing per-vertex object_id...")
    vertex_obj_ids = compute_vertex_object_ids(
        num_vertices=len(verts),
        faces=faces,
        face_obj_ids=face_obj_ids
    )

    print("Saving updated mesh:", out_ply)
    save_vertex_object_mesh(
        out_path=out_ply,
        verts=verts,
        normals=normals,
        colors=colors,
        vertex_obj_ids=vertex_obj_ids,
        faces=faces,
        face_obj_ids=face_obj_ids
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ply", required=True)
    parser.add_argument("--out_ply", required=True)
    args = parser.parse_args()

    convert_mesh_with_vertex_object_ids(args.in_ply, args.out_ply)
