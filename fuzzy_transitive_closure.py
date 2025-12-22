import torch
import numpy as np
import os

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
from scipy.sparse import csr_matrix
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F




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

    # print("valid = ", valid.shape[0])

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

def contrast_palette(labels, s=0.85, v=0.98, noise_gray=0.00):
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

def generate_sam_data_for_anchors(anchor_points, all_views, files_path, anchor_ids = None):
    N = anchor_points.shape[0]
    V = len(all_views)

    if anchor_ids is None:
        anchor_ids = np.arange(anchor_points.shape[0])

    projection_data = np.full((N, V), -1, dtype = np.int32)

    for v_id, view in enumerate(all_views):
        visible_ids, v, u = view_projection(anchor_points, anchor_ids, view)

        if visible_ids.size == 0:
            continue

        image_name = view.image_name
        base = os.path.splitext(image_name)[0]
        npz_path = os.path.join(files_path, f"{base}.npz")

        if os.path.isfile(npz_path):
            npz = np.load(npz_path)
            masks = npz["masks"].astype(bool)

            M, H, W = masks.shape

            local = np.full((H,W), -1, dtype=np.int32)

            for mask_id in range(M):
                local[masks[mask_id]] = mask_id
            
            local_mask_id = local[v,u]
            positive = local_mask_id != -1
            projection_data[visible_ids[positive], v_id] = local_mask_id[positive]

    return projection_data

# def signed_margin_loss(e, i_idx, j_idx, t, m=1.0, alpha=10.0, eps=1e-8):
#     diff = e[i_idx] - e[j_idx]                 # (P, D)
#     d = (diff.square().sum(dim=-1) + eps).sqrt()  # (P,)
#     logits = -alpha * t * (d - m)
#     return torch.nn.functional.softplus(logits).mean()

# def get_embeddings(indices: torch.Tensor):
#     e = emb_table(indices)
#     return F.normalize(e, p=2, dim=-1)

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
    
    anchor_id = np.arange(anchor_points.shape[0])

    bg_color = [1,1,1] if model._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    # projection_data = generate_sam_data_for_anchors(anchor_points, all_views, files_path, anchor_id)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(anchor_points)
    print("Before: ", len(pcd.points))
    voxel_size = 0.04
    # newpcd = pcd.voxel_down_sample(voxel_size = 0.13)   # 1060 points for vs = 0.13
    # newpcd = pcd.voxel_down_sample(voxel_size = voxel_size)   # 1060 points for vs = 0.13

    newpcd2, _, voxel_indices = pcd.voxel_down_sample_and_trace(
    voxel_size, min_bound = pcd.get_min_bound(), max_bound = pcd.get_max_bound(), approximate_class=False)

    print(" After: ", len(newpcd2.points))

    anchors_downsampled = np.asarray(newpcd2.points)
    steps = 200

    # === Training part === 
    anchors_downsampled_id = np.arange(anchors_downsampled.shape[0])
    projection_data = generate_sam_data_for_anchors(anchors_downsampled, all_views, files_path, anchors_downsampled_id)

    
    proj = torch.from_numpy(projection_data).long()
    N, V = proj.shape

    views_pairs = []
    views_point_matches = []
    for v in range(V):
        labels_v = proj[:, v]
        # convert to: 0 = not present, 1.. = local mask id+1
        view_segment = torch.where(labels_v == -1, torch.zeros_like(labels_v), labels_v + 1)

        view_points = torch.nonzero(view_segment, as_tuple=False).squeeze(1)   # indices visible in this view
        if view_points.numel() < 2:
            continue

        view_pairs = torch.combinations(view_points, r=2)
        pair_labels = view_segment[view_pairs]
        view_point_matches = (pair_labels[:, 0] == pair_labels[:, 1]).float()*2 - 1

        views_pairs.append(view_pairs)
        views_point_matches.append(view_point_matches)

    device = 'cuda'
    D = 5

    embeddings = nn.Embedding(num_embeddings=anchors_downsampled.shape[0], embedding_dim=D, sparse=False)
    optimizer = torch.optim.Adam(embeddings.parameters(), lr=0.1)

    # Optimization
    for step in range(steps):
        losses = []
        for view_pairs, view_point_matches in zip(views_pairs, views_point_matches):
            view_embedding_pairs = nn.functional.normalize(embeddings(view_pairs), dim=2) # and this is what?
            view_embedding_sqdists = (view_embedding_pairs[:,0,:] - view_embedding_pairs[:,1,:]).square().sum(dim=1) # what are those indexes :,0,:?
            loss = torch.dot(view_point_matches, view_embedding_sqdists)
            losses.append(loss)        
            #print(torch.cat((view_pairs, torch.unsqueeze(view_point_matches,1)),1))
        
        # Optimization step
        total_loss = torch.stack(losses).sum()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        #print(total_loss)

    # Results    
    embds = nn.functional.normalize(embeddings.weight, dim=1).detach().cpu().numpy()

    print("total loss", total_loss)
    np.save(f"embeddings_norm_{voxel_size}_{steps}_withtrace_16k.npy", embds)
    # === End of training ===


    # restore (later)
    # embds = np.load(f"embeddings_norm_{voxel_size}_{steps}.npy")
    
    

    # eps = 0.004
    # min_samples = 5
    # db = DBSCAN(
    #     eps=eps,
    #     min_samples=min_samples,
    #     metric='cosine',
    #     algorithm="auto",
    # )
    # labels = db.fit_predict(embds)
    
    # # ===

    # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise = int((labels == -1).sum())
    # print(f"{n_clusters=}, {n_noise=}")

    # palette = contrast_palette(np.unique(labels))

    # N = len(anchor_points)
    # labels_dense = -np.ones(N, dtype=int)
    # for i, idxs in enumerate(voxel_indices):
    #     labels_dense[idxs] = labels[i]

    # anchors_colors = palette[labels]
    # all_views = get_views(scene, skip_train=args.skip_train, skip_test = args.skip_test)

    # # make noise a neutral gray (optional)
    # # noise = (labels == -1)
    # # if noise.any():
    # #     anchors_colors[noise] = np.array([0, 0, 0], dtype=np.float64)

    # # assign to your downsampled cloud (embeddings were for anchors_downsampled)
    # # newpcd.colors = o3d.utility.Vector3dVector(anchors_colors)

    # # view
    # # o3d.visualization.draw_geometries([newpcd])

    # points, color, opaicity,scaling,rot, normal, _, _, _,_ = generate_neural_gaussians_SDF(all_views[0], gaussianModel, visible_mask=None)
    # points = points.cpu().detach().numpy()
    # points_normals = torch.nn.functional.normalize(normal).cpu().detach().numpy()
    # vertices, triangle, pcd2 = poisson_surface_reconstruction(points, points_normals, 8) # 9
    # import open3d as o3d
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh.triangles = o3d.utility.Vector3iVector(triangle)
    # mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    # scale_matrix = np.diag([50, 50, 50])
    # pcd2.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd2.points).T).T)
    # normals = np.asarray(pcd2.normals)
    # scaled_normals =normals * 0.1
    # pcd2.normals = o3d.utility.Vector3dVector(scaled_normals)
    # # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # # mesh.compute_vertex_normals()
    
    # from scipy.spatial import cKDTree
    # kdtree = cKDTree(anchors_downsampled)
    # verts = np.asarray(mesh.vertices)
    # _, idx = kdtree.query(verts, k=1) 

    
    # v_colors = anchors_colors[idx]
    # vc = v_colors.mean(axis=1)
    # print(np.shape(mesh.vertices), np.shape(v_colors))
    # mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))
    
    # o3d.io.write_triangle_mesh(f"testy_embed/test1.ply", mesh, write_vertex_colors=True)
    



