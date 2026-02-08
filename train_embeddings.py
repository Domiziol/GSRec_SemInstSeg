import torch
import numpy as np
import os

from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json
import open3d as o3d
import torch
import torch.nn as nn
import colorsys


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

def single_view_projection(anchors, anchors_id, view):
    camera = view
    
    view_matrix = camera.world_view_transform.cpu().numpy()
    view_matrix = view_matrix.T
    proj_matrix = camera.projection_matrix.cpu().numpy()
    proj_matrix = proj_matrix.T
    full_proj_transform = proj_matrix @ view_matrix

    points = np.concatenate([anchors, np.ones((anchors.shape[0], 1))], axis=1)
    clip_coords = (full_proj_transform @ points.T).T
    w = clip_coords[:, 3]
    ndc = clip_coords[:, :3] / clip_coords[:, 3:4]
    W, H = camera.image_width, camera.image_height
    u = np.round((ndc[:, 0] + 1) * 0.5 * W).astype(int)
    v = np.round((1 + ndc[:, 1]) * 0.5 * H).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (clip_coords[:, 3] > 0)

    u_valid = u[valid]
    v_valid = v[valid]
    visible_ids = anchors_id[valid]

    mask[v_valid, u_valid] = 255

    return visible_ids, v_valid, u_valid


def get_original_image(camera):
    org_image = camera.original_image.permute(1,2,0).cpu().numpy()
    return (org_image*255).clip(0, 255).astype(np.uint8)

def get_views(scene, skip_train, skip_test):
    scene_cameras_train = scene.getTrainCameras() if not skip_train else []
    scene_cameras_test = scene.getTestCameras() if not skip_test else []

    views = scene_cameras_train + scene_cameras_test
    return views


def project_anchors_to_2Dsegments(anchor_points, all_views, files_path, anchorIds = None):
    N = anchor_points.shape[0]
    V = len(all_views)

    if anchorIds is None:
        anchorIds = np.arange(anchor_points.shape[0])

    projection_data = np.full((N, V), -1, dtype = np.int32)

    for view_id, view in enumerate(all_views):
        visibleIds, v, u = single_view_projection(anchor_points, anchorIds, view)

        if visibleIds.size == 0:
            continue

        image_name = view.image_name
        base = os.path.splitext(image_name)[0]
        file = os.path.join(files_path, f"{base}.npz")

        if os.path.isfile(file):
            masks = np.load(file)["masks"].astype(bool)

            M, H, W = masks.shape

            view_data = np.full((H,W), -1, dtype=np.int32)

            for mask_id in range(M):
                view_data[masks[mask_id]] = mask_id
            
            localMaskId = view_data[v,u]
            isInSegment = localMaskId != -1
            projection_data[visibleIds[isInSegment], view_id] = localMaskId[isInSegment]

    return projection_data


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

    safe_state(args.quiet)

    gaussianModel, scene, anchor_points, sem_logits = setup_gaussian_scene_and_model(
        model.extract(args), 
        args.iteration,
        args.checkpoint_path
        )

    all_views = get_views(scene, skip_train=args.skip_train, skip_test = args.skip_test)
    
    anchor_id = np.arange(anchor_points.shape[0])

    bg_color = [1,1,1] if model._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


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

   
    anchors_downsampled_id = np.arange(anchors_downsampled.shape[0])
    projection_data = project_anchors_to_2Dsegments(anchors_downsampled, all_views, files_path, anchors_downsampled_id)


    loss_history = []
    projection = torch.from_numpy(projection_data).long()
    N, V = projection.shape

    views_pairs = []
    views_point_matches = []
    for v in range(V):
        labels_v = projection[:, v]
        # dodaj +1: local mask id+1
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
        
        loss_history.append(total_loss.item())
        print(total_loss)

    # Results    
    embds = nn.functional.normalize(embeddings.weight, dim=1).detach().cpu().numpy()

    print("total loss", total_loss)
    np.save(f"embeddings_norm_{voxel_size}_{steps}_withtrace_16k.npy", embds)
    # === End of training ===

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Total loss")
    plt.title("Training loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # restore (later)
    # embds = np.load(f"embeddings_norm_{voxel_size}_{steps}.npy")
    
    

    



