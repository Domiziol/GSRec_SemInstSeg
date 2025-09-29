import torch
import numpy as np

from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import poisson_surface_reconstruction
from gaussian_renderer import generate_neural_gaussians_SDF
from collections import defaultdict
from itertools import combinations
import json
from sklearn.cluster import DBSCAN, Birch, OPTICS
from sklearn.neighbors import KDTree, NearestNeighbors
import scipy.special as sp
from scipy.spatial.distance import jensenshannon, pdist



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
            else:
                raise RuntimeError("Unexpected checkpoint format; expected (capture, iteration)")

            # directly grab sem logits by index (based on your printout) ---
            sem_logits = capture[7]

            # make sure anchors count matches the scene’s anchors
            N_ckpt = sem_logits.shape[0]
            N_scene = gaussianModel._anchor.shape[0]
            if N_ckpt != N_scene:
                print(f"[extract_mesh] Anchor count mismatch: ckpt={N_ckpt} vs scene={N_scene}. "
                    f"Build Scene with the SAME iteration as the checkpoint or do a full restore.")
                
            gaussianModel._sem_logits = torch.nn.Parameter(
                sem_logits.to(gaussianModel._anchor.device), requires_grad=False
            )

        gaussianModel.eval() # setup values for trained gaussian model

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        anchors = gaussianModel.get_anchor.detach().cpu().numpy()

    return gaussianModel, scene, anchors, sem_logits

def view_projection(anchors, anchors_id, view): #, gaussianModel, pipeline, background):
    # render_scene = render(view, gaussianModel, pipeline, background, visible_mask=None, learn_SDF=False)
    # visible_anchor = render_scene["visible_anchor"].cpu().detach().numpy()
    camera = view

    
    view_matrix = camera.world_view_transform.cpu().numpy()
    view_matrix = view_matrix.T
    proj_matrix = camera.projection_matrix.cpu().numpy()
    proj_matrix = proj_matrix.T
    full_proj_transform = proj_matrix @ view_matrix

    points_h = np.concatenate([anchors, np.ones((anchors.shape[0], 1))], axis=1)
    clip_coords = (full_proj_transform @ points_h.T).T
    w = clip_coords[:, 3]
    ndc = clip_coords[:, :3] / clip_coords[:, 3:4]
    W, H = camera.image_width, camera.image_height
    u = np.round((ndc[:, 0] + 1) * 0.5 * W).astype(int)
    v = np.round((1 + ndc[:, 1]) * 0.5 * H).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)

    # Keep only points within image bounds and in front of the camera
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (clip_coords[:, 3] > 0)
    

    u_valid = u[valid]
    v_valid = v[valid]
    visible_ids = anchors_id[valid]

    # projection_info = dict(zip(visible_ids, zip(v_valid, u_valid)))
    mask[v_valid, u_valid] = 255

    # projected_anchors = {}
    # projected_anchors["points"] = [v_valid, u_valid]
    # projected_anchors["global_id"] = visible_ids

    print("valid = ", valid.shape[0])

    # plt.imshow(mask, cmap='gray')
    # plt.title("Visible Anchor Projection")
    # plt.axis('off')
    # plt.show()

    return visible_ids, v_valid, u_valid


def get_original_image(camera):
    org_image_np = camera.original_image.permute(1,2,0).cpu().numpy()
    return (org_image_np*255).clip(0, 255).astype(np.uint8)



def get_views(scene, skip_train, skip_test):
    scene_cameras_train = scene.getTrainCameras() if not skip_train else []
    scene_cameras_test = scene.getTestCameras() if not skip_test else []

    views = scene_cameras_train + scene_cameras_test
    return views

def build_covisibility(visible_ids, covis):
    
    for a, b in combinations(visible_ids, 2):
        i, j = (a, b) if a < b else (b, a)
        covis[(int(i), int(j))] += 1

def build_covisibility_matrix(visible_ids, covis):
    
    for a, b in combinations(visible_ids, 2):
        i, j = (a, b) if a < b else (b, a)
        covis[int(i), int(j)] += 1

import colorsys
def palette_from_labels(labels, s=0.65, v=0.95):
    l = len(labels)
    n = max(1, l)
    table = np.zeros((l, 3), dtype=np.float32)
    for i in range(l):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        table[i] = (r, g, b)
    return table

def contrast_palette(labels, s=0.85, v=0.98, noise_gray=0.30):
    """
    Drop-in replacement: returns an (L,3) RGB table in [0,1],
    with well-separated hues. If -1 is among `labels`, the first
    entry (index 0 after np.unique) is set to gray for noise.
    """
    l = len(labels)
    n = max(1, l)
    table = np.zeros((l, 3), dtype=np.float32)

    phi = 0.6180339887498949  # golden ratio conjugate
    h0 = 0.13                 # start hue (tweak if you want)

    for i in range(l):
        h = (h0 + i * phi) % 1.0
        # tiny value jitter every few colors to boost contrast
        vv = v if (i % 3) else min(1.0, v * 0.92)
        r, g, b = colorsys.hsv_to_rgb(h, s, vv)
        table[i] = (r, g, b)

    # If labels contain -1, np.unique(labels) puts it at index 0.
    # Color that entry gray so noise is subdued.
    if np.any(np.asarray(labels) == -1):
        table[0] = (noise_gray, noise_gray, noise_gray)

    return table

if __name__ == "__main__":
    # Set up command line argument parser with default parameters
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_train", default=False)
    parser.add_argument("--skip_test", default=True)
    parser.add_argument("--checkpoint_path")
    args = get_combined_args(parser)

    # Initialize system state (RNG) -- what is that ????
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

    gaussianModel, scene, all_anchor_points_cuda, sem_logits = setup_gaussian_scene_and_model(
        model.extract(args), 
        args.iteration,
        args.checkpoint_path
        )
    
    anchor_id = np.arange(anchor_points.shape[0])

    bg_color = [1,1,1] if model._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")



    # ===   

    probs = sp.softmax(logits, axis = 1).astype(np.float32)
    maxes = np.max(probs, axis=1)
    prob_squares = np.sqrt(probs)

    K = probs.shape[1]
    probs = np.clip(probs, 1e-12, 1.0)                          # avoid log/sqrt issues
    U = np.sqrt(probs)                                          # √p
    Z = U / np.sqrt(2.0) 

    A = anchor_points.astype(np.float32)
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)  #uśrednienie do sterowania std zamiast cm

    
    w_xyz = 1
    w_sem = 0.1
    # X = np.hstack([anchor_points.astype(np.float32) / w_xyz, Z / w_sem]).astype(np.float32)
    # X = anchor_points.astype(np.float32) / w_xyz
    # X = Z.astype(np.float32) / w_sem

    rng = np.random.default_rng()
    arr = rng.integers(-1, 21, size=(100_000,100), dtype=np.int8)

    from sklearn.preprocessing import StandardScaler
    def make_int8_blobs(n_samples=100_000, n_features=100, n_clusters=5):
        # pick integer centers in [-1, 20]
        centers = rng.integers(-1, 21, size=(n_clusters, n_features))
        # assign each sample to a cluster
        which = rng.integers(0, n_clusters, size=n_samples)
        # gaussian noise around the chosen center (float), then clip & cast
        X = centers[which] + rng.normal(0, 1.0, size=(n_samples, n_features))
        X = np.clip(np.rint(X), -1, 20).astype(np.int8)   # still clustered, but quantized
        return X

    X_int8 = make_int8_blobs()
    # Scale to float for DBSCAN distance calculations
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X_int8)

    # X = arr

    # min_samples = 20  # try 12–25 for 1-1
    min_samples = 20        
    eps = 0.2 # got killed, last one working was 0.1
    
    # for xyz only - top min_smaples 40/50, eps = 0.8, w_xyz = 0.04

    # tree = KDTree(Z, leaf_size=40, metric="euclidean")
    # cnt = np.asarray(tree.query_radius(Z, r=eps, count_only=True))

    Z_norm = Z / w_sem


    # change from Z to Z_norm everywhere
    idx = np.random.choice(len(Z), size=min(2000, len(Z)), replace=False)
    E = np.linalg.norm(Z[idx,None]-Z[None,idx], axis=-1)
    Dq = np.quantile(np.linalg.norm(Z[idx,None]-Z[None,idx], axis=-1), [0.1, 0.5, 0.9])
    print("Hellinger dist quantiles:", Dq)

    m = min(2000, len(Z))
    idx = np.random.choice(len(Z), size=m, replace=False)
    Zs = Z[idx].astype(np.float32)

    dists = pdist(Zs, metric='euclidean')  # this is Hellinger since Z = sqrt(p)/√2
    print("Hellinger pairwise quantiles (0.1, 0.5, 0.9):",
        np.quantile(dists, [0.1, 0.5, 0.9]))
    print("min nonzero distance:", float(dists.min()))

    def hellinger(a,b):
        return np.linalg.norm(a[-K:] - b[-K:])
    
    def agreement(a,b):
        both_pos = (a > 0) & (b > 0)                 # mask where both are positive
        denom = int(np.count_nonzero(both_pos))
        if denom == 0:
            return 1.0                                # convention

        num = int(np.count_nonzero((a == b) & both_pos))
        return 1-num / denom
    

    from scipy.sparse import csr_matrix

    # --- your data generation ---
    # X_int8: (N, D) in int8, values -1..20
    # X: (N, D) float (scaled) for *candidate search only*
    # keep both around
    # X_int8 = make_int8_blobs()
    # X = StandardScaler(with_mean=True, with_std=True).fit_transform(X_int8)

    # --- your custom distance on int8 (vectorized to many) ---
    def agreement_many(a_row: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        a_row: (D,) int8 row
        B    : (M, D) int8 rows
        returns: (M,) float32 distances in [0,1]
        distance = 1 - (# equal & >0) / (# both >0), with denom==0 -> 1.0
        """
        # both positive mask per row
        both_pos = (a_row > 0) & (B > 0)
        denom = np.count_nonzero(both_pos, axis=1)
        # equal on positive positions
        num = np.count_nonzero((B == a_row) & both_pos, axis=1)
        out = np.ones(B.shape[0], dtype=np.float32)
        mask = denom > 0
        out[mask] = 1.0 - (num[mask] / denom[mask])
        return out
    
    def hellinger_many(a_tail: np.ndarray, B_tail: np.ndarray) -> np.ndarray:
        # a_tail, B_tail are the last K dims (already any transform you want)
        diff = B_tail - a_tail  # (M,K)
        return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)

    # --- build sparse eps-graph using k-NN candidates ---
    def build_eps_graph_precomputed(
        X_float: np.ndarray,
        eps: float,
        *,
        metric: str,
        k_candidates: int = 128,
        leaf_size: int = 40,
        # only for agreement:
        X_int8: np.ndarray | None = None,
        # only for hellinger:
        K: int | None = None
    ):
        """
        Build a symmetric CSR with only distances <= eps (good for DBSCAN(metric='precomputed')).

        X_float : (N,Df) float32 — used for fast candidate search (kNN, Euclidean).
                For 'hellinger', it should also contain the last K dims used by the metric.
        eps     : threshold for the chosen metric.
        metric  : 'agreement' or 'hellinger'
        k_candidates : ~64–256 (# of neighbors per point to test)
        X_int8  : (N,D0) int8 — required for 'agreement'
        K       : int — number of tail dims for 'hellinger' (uses X_float[:, -K:])
        """
        N = X_float.shape[0]
        nn = NearestNeighbors(
            n_neighbors=min(k_candidates + 1, N),
            algorithm="ball_tree",
            metric="euclidean",
            leaf_size=leaf_size,
            n_jobs=-1,
        ).fit(X_float)

        _, inds = nn.kneighbors(X_float, return_distance=True)

        rows, cols, data = [], [], []
        if metric == "agreement":
            assert X_int8 is not None, "X_int8 required for 'agreement'"
            for i in range(N):
                neigh = inds[i][1:]  # drop self
                if neigh.size == 0: 
                    continue
                d = agreement_many(X_int8[i], X_int8[neigh])
                keep = np.where(d <= eps)[0]
                if keep.size:
                    j = neigh[keep]
                    rows.extend([i] * keep.size); cols.extend(j.tolist()); data.extend(d[keep].tolist())

        elif metric == "hellinger":
            assert K is not None and K > 0, "Provide K for 'hellinger'"
            tails = X_float[:, -K:].astype(np.float32, copy=False)
            for i in range(N):
                neigh = inds[i][1:]
                if neigh.size == 0:
                    continue
                d = hellinger_many(tails[i], tails[neigh])  # any scaling you want is already in tails
                keep = np.where(d <= eps)[0]
                if keep.size:
                    j = neigh[keep]
                    rows.extend([i] * keep.size); cols.extend(j.tolist()); data.extend(d[keep].tolist())
        else:
            raise ValueError("metric must be 'agreement' or 'hellinger'")

        A = csr_matrix((np.asarray(data, np.float32), (np.asarray(rows), np.asarray(cols))), shape=(N, N))
        A = A.maximum(A.T)  # symmetrize
        return A
    # --- usage ---
    eps = 0.2          # threshold for your agreement distance (tune!)
    k_candidates = 128 # try 64–256; larger = more complete, slower

    # A = build_eps_graph_precomputed(X_float=X, X_int8=X_int8, eps=eps, k_candidates=k_candidates)

    # # Run DBSCAN with the sparse precomputed matrix
    # labels = DBSCAN(eps=eps, min_samples=10, metric="precomputed", n_jobs=-1).fit_predict(A)

    ####

    A = build_eps_graph_precomputed(
        X_float=Z.astype(np.float32),
        eps=eps,
        metric='hellinger',
        k_candidates=128,
        K=Z.shape[1]
    )
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(A)


    # db = DBSCAN(
    #     eps=eps,
    #     min_samples=min_samples,
    #     metric='euclidean',
    #     algorithm="ball_tree",
    #     # metric=hellinger,
    #     # algorithm="ball_tree",
    #     # leaf_size=100,
    # )
    # labels = db.fit_predict(X)

    # db = DBSCAN(
    #     eps=eps,
    #     min_samples=min_samples,
    #     # metric='euclidean',
    #     # algorithm="ball_tree",
    #     metric=agreement,
    #     algorithm="brute",
    #     # leaf_size=100,
    # )
    # labels = db.fit_predict(X)

    

    # nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree', metric=agreement)
    # labels = nbrs.fit(X)

    # db = OPTICS(
    #     eps=eps,
    #     max_eps=eps+0.05,
    #     min_samples=min_samples,
    #     metric=agreement,
    #     cluster_method='dbscan',
    #     algorithm='ball_tree'
    #     # leaf_size=100,
    #              # if your scikit-learn version supports it
    # )
    # labels = db.fit_predict(np.asarray(X, dtype=np.float32))

    
    
    # ===

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"{n_clusters=}, {n_noise=}")

    palette = contrast_palette(np.unique(labels))
    anchors_colors = palette[labels]
    all_views = get_views(scene, skip_train=args.skip_train, skip_test = args.skip_test)

    points, color, opaicity,scaling,rot, normal, _, _, _,_ = generate_neural_gaussians_SDF(all_views[0], gaussianModel, visible_mask=None)
    points = points.cpu().detach().numpy()
    points_normals = torch.nn.functional.normalize(normal).cpu().detach().numpy()
    vertices, triangle, pcd = poisson_surface_reconstruction(points, points_normals, 8) # 9
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangle)
    mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    scale_matrix = np.diag([50, 50, 50])
    pcd.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd.points).T).T)
    normals = np.asarray(pcd.normals)
    scaled_normals =normals * 0.1
    pcd.normals = o3d.utility.Vector3dVector(scaled_normals)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # mesh.compute_vertex_normals()
    
    from scipy.spatial import cKDTree
    kdtree = cKDTree(anchor_points)
    verts = np.asarray(mesh.vertices)
    _, idx = kdtree.query(verts, k=1) 
    v_colors = anchors_colors[idx] 
    mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))
    
    o3d.io.write_triangle_mesh("mesh_2test2.ply", mesh, write_vertex_colors=True)



    # anchor3D_info = {
    # int(global_id): {
    #     "point3D": np.array(point, dtype=np.float32),
    #     "projection_info": {}
    # }
    # for global_id, point in zip(anchor_id, anchor_points)
    # }



    # for view_id, view in enumerate(all_views):
    #     #if view_id == 0:
    #     visible_ids, v_valid, u_valid = view_projection(anchor_points, anchor_id, all_views[view_id])   #, gaussianModel, pipeline, background)

    #     for aid, v, u in zip(visible_ids, v_valid, u_valid):
    #         anchor3D_info[aid]["projection_info"][int(view_id)] = np.array([[v, u]], dtype=np.float32)






        

## SAVE ANCHOR 3D STRUCT ##

# import json

# def to_jsonable(obj):
#     if isinstance(obj, dict):
#         return {k: to_jsonable(v) for k, v in obj.items()}
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, (list, tuple)):
#         return [to_jsonable(v) for v in obj]
#     else:
#         return obj


# with open("anchors3d.json", "w") as f:
#     json.dump(to_jsonable(anchor3D_info), f, indent=2)



    