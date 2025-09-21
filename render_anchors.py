import torch
import numpy as np

from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.mesh_utils import poisson_surface_reconstruction
from gaussian_renderer import generate_neural_gaussians_SDF
from gaussian_renderer import GaussianModel
import matplotlib.pyplot as plt
import cv2
import point_prompted_sam
import automatic_sam
import open3d as o3d

def setup_gaussian_scene_and_model(dataset : ModelParams, iteration : int):
    with torch.no_grad():
        dataset.eval = True
        gaussianModel = GaussianModel(
            dataset.feat_dim, 
            dataset.n_offsets, 
            dataset.voxel_size, 
            dataset.update_depth, 
            dataset.update_init_factor, 
            dataset.update_hierachy_factor, 
            dataset.use_feat_bank
            )
        scene = Scene(dataset, gaussianModel, iteration, shuffle=False)
        gaussianModel.eval() # setup values for trained gaussian model

        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        all_anchor_points_cuda = gaussianModel._anchor# or just _anchor

    return gaussianModel, scene, all_anchor_points_cuda

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

import colorsys

def palette_from_classes(cls, s=0.65, v=0.95):
    # sort for deterministic mapping (remove sorted(...) if you want given order)
    
    n = len(cls)
    lookup = {}
    for i, name in enumerate(cls):
        h = i / max(1, n)            # evenly spaced hue
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        lookup[name] = (r, g, b)
    return lookup

def anchor_to_arrays(anchor3D_info, class_to_color):
    points = []
    labels = []
    colors = []
    
    for aid, data in anchor3D_info.items():
        pt = np.asarray(data["point3D"], dtype=np.float32)
        
        sem = data.get("semantics", {}).get("top3", [])
        if sem:   # has semantics
            cls = sem[0]["class"]        # top1 class
            color = class_to_color[cls]  # default grey
        else:      # no semantics at all
            cls = "none"
            color = (0.5,0.5,0.5)
        
        points.append(pt)
        labels.append(cls)
        colors.append(color)
    
    return (
        np.vstack(points),           # shape (N,3)
        np.array(labels),            # shape (N,)
        np.vstack(colors)            # shape (N,3)
    )

if __name__ == "__main__":
    # Set up command line argument parser with default parameters
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_train", default=False)
    parser.add_argument("--skip_test", default=True)
    args = get_combined_args(parser)

    # Initialize system state (RNG) -- what is that ????
    safe_state(args.quiet)

    gaussianModel, scene, all_anchor_points_cuda = setup_gaussian_scene_and_model(
        model.extract(args), 
        args.iteration
        )
    anchor_points = all_anchor_points_cuda.cpu().detach().numpy()
    anchor_id = np.arange(anchor_points.shape[0])

    bg_color = [1,1,1] if model._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    all_views = get_views(scene, skip_train=args.skip_train, skip_test = args.skip_test)

    view_id = 13

    anchor3D_info = {
    int(global_id): {
        "point3D": np.array(point, dtype=np.float32),
        "projection_info": {}
    }
    for global_id, point in zip(anchor_id, anchor_points)
    }

    ### TEST
    # points, color, opaicity,scaling,rot, normal, _, _, _,_ = generate_neural_gaussians_SDF(all_views[0], gaussianModel, visible_mask=None)        
            
    # points = points.cpu().detach().numpy()
    # points_normals = torch.nn.functional.normalize(normal).cpu().detach().numpy()
    # vertices, triangle, pcd = poisson_surface_reconstruction(points, points_normals, 8) # 9

    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh.triangles = o3d.utility.Vector3iVector(triangle)
    # mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    
    # from scipy.spatial import cKDTree
    # kdtree = cKDTree(anchor_points)
    # verts = np.asarray(mesh.vertices)

    # _, idx = kdtree.query(verts, k=1)   # nearest neighbor index
    # mesh.vertex_colors = o3d.utility.Vector3dVector(input_colors[idx])
    ### ENDOFTEST


    for view_id, view in enumerate(all_views):
        #if view_id == 0:
        visible_ids, v_valid, u_valid = view_projection(anchor_points, anchor_id, all_views[view_id])   #, gaussianModel, pipeline, background)

        for aid, v, u in zip(visible_ids, v_valid, u_valid):
            anchor3D_info[aid]["projection_info"][int(view_id)] = np.array([[v, u]], dtype=np.float32)


    automatic_sam.get_semantic_anchors(anchor3D_info, all_views)
    # point_prompted_sam.get_semantic_anchors(anchor3D_info, all_views)
    classes = automatic_sam.get_classes()
    cls = list(sorted(classes))
    print("checkpoint 1")



    class_color_lookup = palette_from_classes(cls)
    points3D, labels, colors = anchor_to_arrays(anchor3D_info, class_color_lookup)

    points, color, opaicity,scaling,rot, normal, _, _, _,_ = generate_neural_gaussians_SDF(all_views[0], gaussianModel, visible_mask=None)        
            
    points = points.cpu().detach().numpy()
    points_normals = torch.nn.functional.normalize(normal).cpu().detach().numpy()
    vertices, triangle, pcd = poisson_surface_reconstruction(points, points_normals, 8) # 9

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangle)
    mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    
    from scipy.spatial import cKDTree
    kdtree = cKDTree(points3D)
    verts = np.asarray(mesh.vertices)

    _, idx = kdtree.query(verts, k=1) 
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[idx])

    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # scale_matrix = np.diag([50, 50, 50])
    # pcd.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd.points).T).T)
    # normals = np.asarray(pcd.normals)
    # scaled_normals =normals * 0.1
    # pcd.normals = o3d.utility.Vector3dVector(scaled_normals)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # mesh.compute_vertex_normals()

    # n = len(mesh.vertices)
    # colors = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float64), (n,1))
    # mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    print("verts:", len(mesh.vertices), "colors:", len(mesh.vertex_colors))
    # Write the mesh to a file, including vertex colors
    o3d.io.write_triangle_mesh("colored_mesh_poisson.ply", mesh, write_vertex_colors=True)

        


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


## TEST ##

    # anchors = anchor_points[np.array(list(view_projection_info.keys()))]
    # anchors_id = anchor_id
    # view = all_views[view_id]
    # camera = view

    

    # view_matrix = camera.world_view_transform.cpu().numpy()
    # view_matrix = view_matrix.T
    # proj_matrix = camera.projection_matrix.cpu().numpy()
    # proj_matrix = proj_matrix.T
    # full_proj_transform = proj_matrix @ view_matrix

    # points_h = np.concatenate([anchors, np.ones((anchors.shape[0], 1))], axis=1)
    # clip_coords = (full_proj_transform @ points_h.T).T
    # w = clip_coords[:, 3]
    # ndc = clip_coords[:, :3] / clip_coords[:, 3:4]
    # W, H = camera.image_width, camera.image_height
    # u = np.round((ndc[:, 0] + 1) * 0.5 * W).astype(int)
    # v = np.round((1 + ndc[:, 1]) * 0.5 * H).astype(int)
    # mask = np.zeros((H, W), dtype=np.uint8)


    # # if np.allclose(u, u_valid) and np.allclose(v, v_valid):
    # #     print("yes")
    # # -> check if element-wise u_valid and u (v_valid and v) are the same (they have the same sequence of values)
    # # therefore whole sequence would be aligned


    # # Keep only points within image bounds and in front of the camera
    # # valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (clip_coords[:, 3] > 0)

    # # u_valid = u[valid]
    # # v_valid = v[valid]
    # # # visible_ids = anchors_id[valid]
    # # mask[v_valid, u_valid] = 255
    # mask[v, u] = 255

    # # print("valid = ", valid.shape[0])

    # plt.imshow(mask, cmap='gray')
    # plt.title("Visible Anchor Projection")
    # plt.axis('off')
    # plt.show()
    