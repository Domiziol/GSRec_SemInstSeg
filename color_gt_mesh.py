from pathlib import Path
import numpy as np

from plyfile import PlyData, PlyElement
colors = (
    (242,73,73),(109,127,248),(106, 45, 110),(73,155,242),(242,142,73),
    (155,73,242),(73,242,242),(242,73,203),(203,242,73),(242,160,203),
    (244,108,129),(203,191,242),(191,142,73),(242,236,191),(128,38,38),
    (191,242,203),(142,142,38),(59, 232, 245),(38,38,128),(210,137,91),
    (242,38,38),(38,242,38),(38,38,242),(242,242,38),(242,38,242),
    (38,242,242),(191,191,191),(128,38,128),(202,191,155),(83,128,38),
    (38,128,83),(38,83,128),(83,38,128),(128,38,83),(191,38,38),
    (38,191,38),(38,38,191),(191,191,38),(191,38,191),(38,191,191),
    (102,38,38),(38,102,38),(38,38,102),(102,102,38),(102,38,102),
    (38,102,102),(157,17,16),(83,166,83),(83,83,166),(166,166,83),
    (166,83,166),(83,166,166),(204,121,38),(38,204,121),(121,38,204),
    (204,38,121),(121,204,38),(38,121,204),(84, 46, 12),(38,153,96),
    (96,38,153),(153,38,96),(96,153,38),(38,96,153),(100, 76, 209),
    (162,242,38),(38,242,162),(38,162,242),(162,38,242),(242,38,162),
    (204,142,83),(83,204,142),(142,83,204),(204,83,142),(203,235,193),
    (83,142,204),(242,121,121),(121,242,121),(69,247,172),(242,242,121),
    (242,121,242),(121,242,242),(64,64,140), (140,64,64),(64,140,64),
    (191,96,96),(96,191,96),(96,96,191),(191,191,96),(191,96,191),
    (3,97,104),(77,38,38),(38,77,38),(38,38,77),(77,77,38),
    (77,38,77), (242,74,190), (182,145,212), (199,206,110), (245,134,71),
    (27,253,70)
)

# for instance coloring only
# colors = (
#     (242,73,73),(109,127,248),(106, 45, 110),(73,155,242),(242,142,73),
#     (38,38,77),(73,242,242),(242,73,203),(203,242,73),(242,160,203),
#     (210,137,91),(203,191,242),(128,38,128),(242,236,191),(128,38,38),
#     (191,242,203),(142,142,38),(59, 232, 245),(38,38,128),(244,108,129),
#     (242,38,38),(38,83,128),(38,38,242),(242,242,38),(242,38,242),
#     (38,242,242),(191,191,191),(191,142,73),(202,191,155),(83,128,38),
#     (38,128,83),(38,242,38),(83,38,128),(128,38,83),(191,38,38),
#     (38,191,38),(38,38,191),(191,191,38),(77,38,77),(38,191,191),
#     (102,38,38),(38,102,38),(38,38,102),(102,102,38),(102,38,102),
#     (38,102,102),(157,17,16),(83,166,83),(83,83,166),(166,166,83),
#     (166,83,166),(83,166,166),(204,121,38),(38,204,121),(121,38,204),
#     (204,38,121),(121,204,38),(242,121,121),(84, 46, 12),(38,153,96),
#     (77,77,38),(153,38,96),(96,153,38),(38,96,153),(100, 76, 209),
#     (162,242,38),(38,242,162),(38,162,242),(162,38,242),(242,38,162),
#     (204,142,83),(83,204,142),(142,83,204),(204,83,142),(203,235,193),
#     (83,142,204),(38,121,204),(121,242,121),(69,247,172),(242,242,121),
#     (242,121,242),(121,242,242),(64,64,140), (140,64,64),(64,140,64),
#     (191,96,96),(182,145,212),(96,96,191),(191,191,96),(191,96,191),
#     (3,97,104),(77,38,38), (38,77,38)
# )


# changed 26.01 (semantic)
# 3: 106, 45, 110
# 18: 59, 232, 245
# 59: 84, 46, 12
# 65: 100, 76, 209
assert len(colors) == 101

def color_ply_vertices_by_class_id(
    ply_path: str | Path,
    class_field: str = "class_id",
    indexed_plus_one: bool = True,
    ignore_ids=(-1, -2),
    ignore_color=(0, 0, 0),   # color for ignored vertices
):
    ply_path = Path(ply_path)
    out_path = ply_path.with_name(ply_path.stem + "_colored_semanticpred" + ply_path.suffix)

    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError("PLY has no 'vertex' element.")

    v = ply["vertex"].data  # structured array

    if class_field not in v.dtype.names:
        raise ValueError(f"Vertex property '{class_field}' not found. Available: {v.dtype.names}")

    class_ids = v[class_field].astype(np.int64)

    # ign = np.isin(class_ids, np.array(ignore_ids, dtype=np.int64))

    if indexed_plus_one:
        class_ids = class_ids - 1

    ign = np.isin(class_ids, np.array(ignore_ids, dtype=np.int64))

    palette = np.asarray(colors, dtype=np.uint8)  # (101,3)

    rgb = np.empty((len(v), 3), dtype=np.uint8)
    rgb[ign] = np.array(ignore_color, dtype=np.uint8)

    ok = ~ign
    if ok.any():
        ok_ids = class_ids[ok]
        if ok_ids.min() < 0 or ok_ids.max() >= len(colors):
            raise ValueError(
                f"class_id out of range after shift (excluding ignored): "
                f"min={ok_ids.min()} max={ok_ids.max()} palette_len={len(colors)}"
            )
        rgb[ok] = palette[ok_ids]

    # Build new vertex dtype (preserve everything, ensure RGB exists)
    names = list(v.dtype.names)
    new_descr = list(v.dtype.descr)
    for c in ("red", "green", "blue"):
        if c not in names:
            new_descr.append((c, "u1"))

    v2 = np.empty(v.shape, dtype=new_descr)
    for n in names:
        v2[n] = v[n]

    v2["red"]   = rgb[:, 0]
    v2["green"] = rgb[:, 1]
    v2["blue"]  = rgb[:, 2]

    elements = []
    for el in ply.elements:
        if el.name == "vertex":
            elements.append(PlyElement.describe(v2, "vertex"))
        else:
            elements.append(el)

    PlyData(elements, text=ply.text).write(str(out_path))
    return out_path



if __name__ == "__main__":

    # gt_mesh_path = "./data/replica/scan1/mesh_semantic_verts_bothids.ply"
    gt_mesh_path = "/home/domi/repos/3dgs/GSRec_SemInstSeg/experiments3/model_d8k/wsem_only/wsem=1.0_eps=0.2_512/both_segmentations.ply"
    out = color_ply_vertices_by_class_id(
        gt_mesh_path,
        class_field="class_id",
        indexed_plus_one=True,
        ignore_ids=(-1, -2)
    )
    print("Saved:", out)