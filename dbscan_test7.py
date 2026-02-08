## owner: Dominika Ziolkiewicz

## THESIS
import torch
import numpy as np
from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import poisson_surface_reconstruction
from gaussian_renderer import generate_neural_gaussians_SDF
import json
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon
from scipy.sparse import csr_matrix
from plyfile import PlyData, PlyElement
from pathlib import Path
import colorsys
import open3d as o3d
from scipy.spatial import cKDTree


def get_classes():
    classes= []

    with open("info_semantic.json", 'r') as classes_file:
        data = json.load(classes_file)

    for objects in data['classes']:
        classes.append(objects['name'])

    return classes

def setup_gaussian_scene_and_model(dataset : ModelParams, iteration : int, checkpoint: str):
    classes = get_classes()

    with torch.no_grad():
        dataset.eval = True
        gaussianModel = GaussianModel(
            classes,
            dataset.feat_dim, 
            dataset.n_offsets, 
            dataset.voxel_size, 
            dataset.update_depth, 
            dataset.update_init_factor, 
            dataset.update_hierachy_factor, 
            dataset.use_feat_bank
            )
        scene = Scene(dataset, gaussianModel, iteration, shuffle=False)
        if(checkpoint):
            ckpt = torch.load(checkpoint, map_location="cuda")
            if isinstance(ckpt, tuple) and isinstance(ckpt[0], tuple):
                capture = ckpt[0]
            
            sem_logits = capture[7]
                
            gaussianModel._sem_logits = torch.nn.Parameter(sem_logits.to(gaussianModel._anchor.device), requires_grad=False)

        gaussianModel.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        anchors = gaussianModel.get_anchor.detach().cpu().numpy()

    return gaussianModel, scene, anchors, sem_logits


def get_views(scene, skip_train, skip_test):
    scene_cameras_train = scene.getTrainCameras() if not skip_train else []
    scene_cameras_test = scene.getTestCameras() if not skip_test else []

    views = scene_cameras_train + scene_cameras_test
    return views

def get_color_palette(labels, sRange=(0.55, 0.95), vRange=(0.65, 1.0), hueBase=0.13, noise=-1):
    
    labels = np.asarray(labels)
    uniq = np.unique(labels)

    has_noise = noise in uniq
    if has_noise:
        classLabels = [l for l in uniq if l != noise]
    else:
        classLabels = uniq.tolist()

    
    phi = 0.6180339887498949  # golden ratio conjugate

    
    palette = np.zeros((len(labels), 3), dtype=np.float32)
    label_to_idx = {lab: i for i, lab in enumerate(classLabels)}
    sv_patterns = [
        (sRange[1], vRange[1]),
        (sRange[1], vRange[0]),
        (sRange[0], vRange[1]),
        (sRange[0], vRange[0]),
    ]

    for _, label in enumerate(classLabels):
        k = label_to_idx[label]

        h = (hueBase + k*phi) % 1.0
        s, v = sv_patterns[k % len(sv_patterns)]
        r,g,b = colorsys.hsv_to_rgb(h,s,v)

       
        palette[labels == label] = (r, g, b)

    if has_noise:
        gray = np.array([1, 1, 1], dtype=np.float32)
        palette[labels == noise] = gray

    return palette


def apply_transform_to_mesh(points):
    scale_factor = 4
    shift_vector = np.array([2.95531, 1.13268, -0.058562])

    transform = np.array([
        [ 1.06030165e+00, -3.20324289e-03,  7.84449640e-04, -1.23199711e-01],
        [ 3.20130877e-03,  1.06029875e+00, 2.60242687e-03, -4.07212017e-02],
        [-7.92305771e-04, -2.60004585e-03,  1.06030329e+00, -6.42458858e-02],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
    ])

    newPts = points * scale_factor + shift_vector[None, :]
    newPts2 = np.concatenate([newPts, np.ones((newPts.shape[0], 1))], axis=1)
    outPts = (transform @ newPts2.T).T[:, :3]
    return outPts


def logits_smoothing(anchors_xyz, logits):

    N, K = logits.shape

    sortedIds = np.argsort(logits, axis=1)
    top1 = logits[np.arange(N), sortedIds[:, -1]]
    top2 = logits[np.arange(N), sortedIds[:, -2]]
    margin = top1 - top2

    margin = 1.0 / (1.0 + np.exp(-margin / 2.0))    # pewność

    
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    expSum = exp.sum(axis=1, keepdims=True)
    prob = exp / np.clip(expSum, 1e-12, None)

    # entropia
    H = - (prob * np.log(np.clip(prob, 1e-12, 1.0))).sum(axis=1)
    smoothingFactor = 1.0 - H / (np.log(K) + 1e-12)
    
    conf = 0.5 * smoothingFactor + 0.5 * margin
    

    nn = NearestNeighbors(n_neighbors=min(50, N)).fit(anchors_xyz)
    d, ids = nn.kneighbors(anchors_xyz, return_distance=True)
    sigma_s = (np.median( d[d > 0]) if  d[d > 0].size else 1.0) + 1e-9

    weight = np.exp(-(d**2) / (2.0 * sigma_s**2))
    weight *= conf[ids]

    weightSelf = 1.0 * conf
    denominator = np.maximum(weightSelf + weight.sum(axis=1), 1e-12)
    neighbours = (weight[..., None] * logits[ids]).sum(axis=1)
    self = (weightSelf[:, None] * logits)

    return (self + neighbours) / denominator[:, None]

def row_softmax(row):

    row = row.astype(np.float64, copy=True)

    row -= row.max(axis=1, keepdims=True)

    np.exp(row, out=row)
    row /= row.sum(axis=1, keepdims=True)

    return row

def get_Jensen_Shannon(neighProb, iProb):
    return jensenshannon(neighProb, iProb[None, :], axis=1, base=2)


def get_scales(anchor_points, embeddings, logits,k_candidates, sample_size):

    N = anchor_points.shape[0]
    rng = np.random.default_rng(0)
    samples = rng.choice(N, size=min(sample_size, N), replace=False)

    # kNN po geometrii
    nn = NearestNeighbors(
        n_neighbors=min(k_candidates + 1, N),
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit(anchor_points.astype(np.float32, copy=False))

    dists, neighIds = nn.kneighbors(anchor_points[samples], return_distance=True)
    dists_no_self = dists[:, 1:]

    scale_xyz = float(np.median(dists_no_self))

    D_emb = []
    D_sem = []

    smoothed_logits = logits_smoothing(anchor_points, logits)
    probs = row_softmax(smoothed_logits)

    for row_idx, i in enumerate(samples):
        neigh = neighIds[row_idx, 1:]
        if neigh.size == 0:
            continue

        # EMB
        diff_emb = embeddings[i] - embeddings[neigh]
        d_emb = np.linalg.norm(diff_emb, axis=1)
        D_emb.append(d_emb)

        # SEM
        d_sem = get_Jensen_Shannon(probs[neigh], probs[i])
        D_sem.append(d_sem)

    D_emb = np.concatenate(D_emb)
    D_sem = np.concatenate(D_sem)

    scale_emb = float(np.median(D_emb))
    scale_sem = float(np.median(D_sem))

    print("scales:", "xyz", scale_xyz, "emb", scale_emb, "sem", scale_sem)
    
    return scale_xyz, scale_emb, scale_sem

def build_precomputed(anchor_points, embeddings, eps, logits, w_dist, w_emb, w_sem, k_candidates = 512):
        
    N= anchor_points.shape[0]

    rows, cols, data = [], [], []

    s_xyz, s_emb, s_sem = get_scales(anchor_points, embeddings, logits, 512, 10000)
    smoothed_logits = logits_smoothing(anchor_points,logits)

    nn = NearestNeighbors(
        n_neighbors=min(k_candidates + 1, N),
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit(anchor_points.astype(np.float32, copy=False))

    distances, ids = nn.kneighbors(anchor_points, return_distance=True)
    dists_wo_self = distances[:, 1:]
    ids_wo_self= ids[:, 1:]

    probs = row_softmax(smoothed_logits)
   
    for i in range(N):
        neigh = ids_wo_self[i]
        
        if neigh.size == 0:
            continue
        d_xyz = dists_wo_self[i]
        d_emb = np.linalg.norm(embeddings[i] - embeddings[neigh], axis=1)
        d_sem = get_Jensen_Shannon(probs[neigh], probs[i])
            
    
        D_xyz = (d_xyz / (s_xyz + 1e-12))
        D_xyz = D_xyz / (1.0 + D_xyz)

        D_emb = (d_emb / (s_emb + 1e-12))
        D_emb = D_emb / (1.0 + D_emb)

        D_sem = (d_sem / (s_sem + 1e-12))
        D_sem = D_sem / (1.0 + D_sem)


        d = D_emb * w_emb + D_xyz * w_dist + w_sem * D_sem

        keep = d < eps
        
        if np.any(keep):
            j = neigh[keep]
            d_kept = d[keep]

            rows.extend([i] * len(j))
            cols.extend(j.tolist())
            data.extend(d_kept.tolist())


    matrix = csr_matrix((np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, N))
    matrix = matrix.maximum(matrix.T)

    return matrix

if __name__ == "__main__":
    # Set up command line argument parser with default parameters
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_train", default=False)
    parser.add_argument("--skip_test", default=False) # from True
    parser.add_argument("--checkpoint_path")
    args = get_combined_args(parser)


    files_path = "./data/replica/scan1/2Dclassification_tests/test1/results/"
    emb_path = f"./outputs/final_sam3/d8k_l01/"+"embeddings_norm_0.04_200_withtrace.npy"
    instance_id_save_path = f"./experiments3/model_d8k/instance_ids.npy"
    
    safe_state(args.quiet)

    gaussianModel, scene, anchor_points, sem_logits = setup_gaussian_scene_and_model(
        model.extract(args), 
        args.iteration,
        args.checkpoint_path
    )

    logits = sem_logits.cpu().detach().numpy()
    anchor_id = np.arange(anchor_points.shape[0])

    bg_color = [1,1,1] if model._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    all_views = get_views(scene, skip_train=args.skip_train, skip_test = args.skip_test)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(anchor_points)
    print("Before: ", len(pcd.points))

    voxel_size = 0.04 

    newpcd, _, voxel_indices = pcd.voxel_down_sample_and_trace(
    voxel_size, min_bound = pcd.get_min_bound(), max_bound = pcd.get_max_bound(), approximate_class=False)
    print("After: ", len(newpcd.points))

    anchors_downsampled = np.asarray(newpcd.points)
    counts = np.array([len(points) for points in voxel_indices], dtype=float)
    anchors_downsampled_id = np.arange(anchors_downsampled.shape[0])

    N, V = anchor_points.shape
    
    w_dist = 0.2
    w_emb = 0.6
    w_sem = 0.2

    min_samples = 20
    eps = 0.6
    k_neighbours = 512

    
    embeds = np.load(emb_path)
    M, D = embeds.shape
    embdsAll = np.empty((N, D), dtype=embeds.dtype)
    for new_idx, orig_idxs in enumerate(voxel_indices):
        if len(orig_idxs) == 0:
            continue
        embdsAll[orig_idxs] = embeds[new_idx]

   
    dataMatrix = build_precomputed(
        anchor_points,
        embdsAll,
        eps,
        logits,
        w_dist,
        w_emb,
        w_sem,
        k_neighbours
    )

    output_path = f"./experiments3/model_d8k/wemb+wsem/"+f"wdist={w_dist}_wemb={w_emb}_wsem={w_sem}_eps={eps}_{k_neighbours}"
    mesh_save_path = output_path + "/both_segmentations.ply"
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)
    
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(dataMatrix)
    np.save(output_path+"/instance_ids.npy", labels)

    #labels =  np.load(f"./experiments3/model_d{choose_model}k/"+f"wdist={w_dist}_wemb={w_emb}_wsem={w_sem}_eps={eps}_{k_neighbours}/instance_ids.npy")
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"{n_clusters=}, {n_noise=}")


    smoothed_logits = logits_smoothing(anchor_points,logits)


    points, color, opaicity,scaling,rot, normal, _, _, _,_ = generate_neural_gaussians_SDF(all_views[0], gaussianModel, visible_mask=None)
    points = points.cpu().detach().numpy()
    points_normals = torch.nn.functional.normalize(normal).cpu().detach().numpy()
    vertices, triangle, pcd = poisson_surface_reconstruction(points, points_normals, 8) # 9


    anchors_transformed = apply_transform_to_mesh(anchor_points)
    
    mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    verts_transformed = apply_transform_to_mesh(vertices)
    mesh.vertices = o3d.utility.Vector3dVector(verts_transformed)
    mesh.triangles = o3d.utility.Vector3iVector(triangle)
    mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    
    scale_matrix = np.diag([50, 50, 50])
    pcd.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd.points).T).T)
    normals = np.asarray(pcd.normals)
    scaled_normals =normals * 0.1
    pcd.normals = o3d.utility.Vector3dVector(scaled_normals)
   
    anchors_colors_inst = get_color_palette(labels, noise=-1)
    
    
    kdtree = cKDTree(anchors_transformed)
    verts = np.asarray(mesh.vertices)
    _, idx = kdtree.query(verts, k=1) 
    v_colors = anchors_colors_inst[idx] 
    mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))

    # o3d.io.write_triangle_mesh(output_path+f"/wdist={w_dist}_wemb={w_emb}_wsem={w_sem}_eps={eps}_{k_neighbours}.ply", mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(output_path+f"/wdist={w_dist}_wemb={w_emb}_wsem={w_sem}_eps={eps}_{k_neighbours}.ply", mesh, write_vertex_colors=True)










    probs = row_softmax(smoothed_logits)
    classIds = probs.argmax(axis=1).astype(np.int32) + 1
    classes = get_classes()
    palette = get_color_palette(classes)
    anchors_colors_class = palette[np.clip(classIds, 0, palette.shape[0]-1)]
    noise = (classIds == -1)
    anchors_colors_class[noise] = np.array([0.6, 0.6, 0.6], dtype=np.float64)

    anchor_class_conf = probs[np.arange(probs.shape[0]), classIds - 1].astype(np.float32)

    
    kdtree = cKDTree(anchors_transformed)
    verts = np.asarray(mesh.vertices)
    _, idx = kdtree.query(verts, k=1) 

    v_colors = anchors_colors_class[idx] 

    vertex_class_id = classIds[idx]
    vertex_instance_id = labels[idx]
    vertex_class_conf = anchor_class_conf[idx]
    vertex_class_probs = probs[idx].astype(np.float16)

    mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))
    o3d.io.write_triangle_mesh(output_path+f"/semantic.ply", mesh, write_vertex_colors=True)


    np.savez_compressed(
        output_path + "/vertex_class_probs.npz",
        probs=vertex_class_probs,
        pred_class_id=vertex_class_id,
        pred_object_id=vertex_instance_id,
        vertex_index=np.arange(len(vertex_class_id), dtype=np.int32)
    )







    # save mesh with class_id and instance_id per vertex
    ply = PlyData.read(output_path+f"/semantic.ply")
    v = ply["vertex"].data

    vertex_class_id = np.asarray(vertex_class_id, dtype=np.int32)
    vertex_instance_id = np.asarray(vertex_instance_id, dtype=np.int32)
    vertex_class_conf = np.asarray(vertex_class_conf, dtype=np.float16)

    new_dtype = v.dtype.descr + [("class_id", "i4"), ("instance_id", "i4"), ("class_conf", "f4")]
    new_v = np.empty(v.shape[0], dtype=new_dtype)

    for name in v.dtype.names:
        new_v[name] = v[name]

    # set new field
    new_v["class_id"] = vertex_class_id
    new_v["instance_id"] = vertex_instance_id
    new_v["class_conf"] = vertex_class_conf

    # keep old
    new_vertex_element = PlyElement.describe(new_v, "vertex")
    new_elements = []
    for elt in ply.elements:
        if elt.name == "vertex":
            new_elements.append(new_vertex_element)
        else:
            new_elements.append(elt)

    ply["vertex"].data = new_v
    new_ply = PlyData(new_elements, text=ply.text)
    new_ply.write(mesh_save_path)





    gt_mesh_path = "./data/replica/scan1/mesh_semantic_verts_bothids.ply"
    gt_ply = PlyData.read(gt_mesh_path)
    v = gt_ply["vertex"].data

    gt_xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    pred_xyz = anchors_transformed.astype(np.float32)
    pred_cls = classIds.astype(np.int32)
    pred_inst = labels.astype(np.int32)
    pred_conf = anchor_class_conf.astype(np.float16)


    tree = cKDTree(pred_xyz)
    dists, nn_idx = tree.query(gt_xyz, k=1)
    mapped_pred_cls = pred_cls[nn_idx].astype(np.int32)
    mapped_pred_obj = pred_inst[nn_idx].astype(np.int32)
    mapped_pred_conf = pred_conf[nn_idx].astype(np.float16)
    mapped_pred_probs = probs[nn_idx].astype(np.float16)

    np.savez_compressed(
        output_path + "/mapped_vertex_class_probs_onto_gt.npz",
        probs=mapped_pred_probs,
    )

    old_names = v.dtype.names
    new_dtype = [(n, v.dtype.fields[n][0]) for n in old_names if n != "pred_class_id" or n != "pred_object_id"] + [("pred_class_id", "i4"), ("pred_object_id", "i4"), ("pred_class_conf", "f4")]
    newVertex = np.empty(v.shape, dtype=new_dtype)

    for n in old_names:
        if n == "pred_class_id" or n == "pred_object_id" or n == "pred_class_conf":
            continue
        newVertex[n] = v[n]
    newVertex["pred_class_id"] = mapped_pred_cls
    newVertex["pred_object_id"] = mapped_pred_obj
    newVertex["pred_class_conf"] = mapped_pred_conf

    vertexElement = PlyElement.describe(newVertex, "vertex")

    newElements = [vertexElement if el.name == "vertex" else el for el in gt_ply.elements]

    newPlyData = PlyData(
        newElements,
        text=gt_ply.text,
        byte_order=gt_ply.byte_order,
        comments=gt_ply.comments,
        obj_info=gt_ply.obj_info,
    )
    newPlyData.write(output_path+f"/mapped_semantic_class_id_&_object_id_onto_gt.ply")
    


