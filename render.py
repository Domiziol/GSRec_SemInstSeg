#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch

import numpy as np

import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import matplotlib.pyplot as plt
from utils.loss_utils import compute_scale_and_shift
from utils.mesh_utils import poisson_surface_reconstruction

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_renders")
    
    if not os.path.exists(depth_render_path):
        os.makedirs(depth_render_path, exist_ok=True)

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []
    import open3d as o3d
    
    render_pkg = render(views[5], gaussians, pipeline, background, visible_mask=None, learn_SDF=False)

    camera = views[5]
    view_matrix = camera.world_view_transform.cpu().numpy()
    view_matrix = view_matrix.T
    proj_matrix = camera.projection_matrix.cpu().numpy()
    # print("View matrix:\n", view_matrix)
    # print("Projection matrix:\n", proj_matrix)

    # proj_matrix[2, 3] *= -1
    # proj_matrix[2, 2] *= -1
    proj_matrix = proj_matrix.T
    # print("Projection matrix:\n", proj_matrix)


    full_proj_transform = proj_matrix @ view_matrix
    full_proj = camera.full_proj_transform.cpu().numpy()
    # print("Full_proj_to_view:\n", full_proj_transform)
    # print("Full_original:\n", full_proj)


    visible_anchor = render_pkg["visible_anchor"].cpu().detach().numpy()

    # print("visible_anchor min:", visible_anchor.min(axis=0))
    # print("visible_anchor max:", visible_anchor.max(axis=0))



    print("View matrix:\n", view_matrix)
    print("Camera center:\n", camera.camera_center.cpu().numpy())
    print("Projection matrix:\n", proj_matrix)


    p = visible_anchor[0]
    p_h = np.concatenate([p, [1]])
    clip = full_proj_transform @ p_h

    cam_space = view_matrix @ p_h
    print("Camera space:", cam_space)
    print("Z in camera space:", cam_space[2])
    print("Camera pose (view matrix last row):", view_matrix[3])




    print("||clip|| =", np.linalg.norm(clip[:3]))
    print("clip.w =", clip[3])

    ndc = clip[:3] / clip[3]

    print("Single point clip:", clip)
    print("NDC:", ndc)
    cam_space = (view_matrix @ p_h.T).T
    print("Camera-space point:", cam_space)




    points_h = np.concatenate([visible_anchor, np.ones((visible_anchor.shape[0], 1))], axis=1)
    clip_coords = (full_proj_transform @ points_h.T).T
    w = clip_coords[:, 3]
    print("clip w min:", w.min(), "max:", w.max())

    ndc = clip_coords[:, :3] / clip_coords[:, 3:4]
    W, H = camera.image_width, camera.image_height
    u = np.round((ndc[:, 0] + 1) * 0.5 * W).astype(int)
    v = np.round((1 + ndc[:, 1]) * 0.5 * H).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)

    # Keep only points within image bounds and in front of the camera
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (clip_coords[:, 3] > 0)

    u_valid = u[valid]
    v_valid = v[valid]
    mask[v_valid, u_valid] = 255

    print("liczba anchor points na obrazie: \n", len(u_valid))


    plt.imshow(mask, cmap='gray')
    plt.title("Visible Anchor Projection")
    plt.axis('off')
    plt.show()


    camera_center = camera.camera_center.cpu().detach().numpy()
    camera_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    camera_marker.translate(camera_center)
    camera_marker.paint_uniform_color([1.0, 0.0, 0.0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(visible_anchor)
    o3d.visualization.draw_geometries([pcd2, camera_marker])
    points = render_pkg["points"].cpu().detach().numpy()
    points_normals = torch.nn.functional.normalize(render_pkg["normal"], dim=-1).cpu().detach().numpy()
    vertices, triangle, pcd = poisson_surface_reconstruction(points, points_normals, 9)
    # use open3d to save the mesh
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangle)
    mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    
    # save the mls information
    torch.save(
        {'points': render_pkg["points"].cpu().detach(),
         'normals': render_pkg["normal"].cpu().detach(),
         'covs': render_pkg["cov3D"].cpu().detach(),
         'opacity': render_pkg["opacity"].cpu().detach()},
        os.path.join(model_path, name, "ours_{}_mls_info.pt".format(iteration))
    )
    
    
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    scale_matrix = np.diag([50, 50, 50])
    pcd.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd.points).T).T)
    normals = np.asarray(pcd.normals)
    scaled_normals = normals * 0.1
    pcd.normals = o3d.utility.Vector3dVector(scaled_normals)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(model_path, name, "ours_{}".format(iteration), '{0:05d}'.format(0) + ".ply"), mesh)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

     
        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)
        
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        
        # render depth
        render_depth = render_pkg["render_depth"]
        gt_depth = view.depth
        scale, shift = compute_scale_and_shift(render_depth, gt_depth)
        depth = render_depth * scale + shift
        
        depth_concat = torch.cat((depth, gt_depth), dim=0).unsqueeze(1)
        tensor = torchvision.utils.make_grid(depth_concat, padding=0, normalize=False, scale_each=False).cpu().detach().numpy()
        plt.imsave(os.path.join(depth_render_path, '{0:05d}'.format(idx) + "_depth.png"), np.transpose(tensor, (1,2,0))[:,:,0], cmap="viridis")
                
        # add normal rendering
        render_normal = render_pkg["render_normal"]
        gt_normal = view.normal
        normal_concat = torch.stack((render_normal, gt_normal), dim=0)
        normal_concat = (normal_concat + 1)/2.0
        tensor = torchvision.utils.make_grid(normal_concat, padding=0, normalize=False, scale_each=False).cpu().detach().numpy()
        plt.imsave(os.path.join(depth_render_path, '{0:05d}'.format(idx) + "_normal.png"), (tensor.transpose((1,2,0))*255).astype(np.uint8))
              
        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            
   
    
    
    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      

def get_classes():
    classes= []

    with open("info_semantic.json", 'r') as classes_file:
        data = json.load(classes_file)

    for objects in data['classes']:
        classes.append(objects['name'])

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)

    return classes

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    skip_test = False
    skip_train = False
    classes = get_classes()
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(classes, dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank)
        
        # visible_mask = torch.ones(gaussians.get_anchor.shape[0], dtype=torch.bool, device = gaussians.get_anchor.device)
        # all_anchors = gaussians.get_anchor[visible_mask]
        # all_anchors = all_anchors.cpu().detach().numpy()
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()
        anchor = gaussians._anchor.detach().cpu().numpy()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_false")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
