## owner: Dominika Ziolkiewicz

## THESIS
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

def ColorPlyByID(plyPath, IDName, shift, ignoreID):

    plyPath = Path(plyPath)
    outputMeshPath = plyPath.with_name(plyPath.stem + "_colored_semanticpred" + plyPath.suffix)

    plyFile = PlyData.read(str(plyPath))
    vertexData = plyFile["vertex"].data

    if IDName not in vertexData.dtype.names:
        print(f"'{IDName}' not found")

    classIds = vertexData[IDName].astype(np.int64)

    if shift:
        classIds = classIds - 1

    

    isIgnore = np.isin(classIds, np.array(ignoreID, dtype=np.int64))
    palette = np.asarray(colors, dtype=np.uint8)  # (101,3)


    newVertexColor = np.empty((len(vertexData), 3), dtype=np.uint8)
    newVertexColor[isIgnore] = np.array((0, 0, 0), dtype=np.uint8)

    isValid = ~isIgnore
    if isValid.any():
        validIds = classIds[isValid]

        newVertexColor[isValid] = palette[validIds]

    
    oldVertexData = list(vertexData.dtype.names)
    newDesc = list(vertexData.dtype.descr)

    # add color descriptor if doesn't exists yet
    for c in ("red", "green", "blue"):
        if c not in oldVertexData:
            newDesc.append((c, "u1"))

    newVertex = np.empty(vertexData.shape, dtype=newDesc)
    for n in oldVertexData:
        newVertex[n] = vertexData[n]

    newVertex["red"] = newVertexColor[:, 0]
    newVertex["green"]= newVertexColor[:, 1]
    newVertex["blue"] = newVertexColor[:, 2]

    plyData = []
    for i in plyFile.elements:
        if i.name == "vertex":
            plyData.append(PlyElement.describe(newVertex, "vertex"))
        else:
            plyData.append(i)

    PlyData(plyData, text=plyFile.text).write(str(outputMeshPath))
    

if __name__ == "__main__":

    # meshPath = "./data/replica/scan1/mesh_semantic_verts_bothids.ply"
    meshPath = "/home/domi/repos/3dgs/GSRec_SemInstSeg/experiments3/model_d8k/wsem_only/wsem=1.0_eps=0.2_512/both_segmentations.ply"

    ColorPlyByID(meshPath, IDName="class_id", shift=True, ignore=(-1, -2))
    