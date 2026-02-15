# THESIS - modified original GSrec script
import os
import torch

import numpy as np

from scene import Scene
import json
from gaussian_renderer import render, prefilter_voxel
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import poisson_surface_reconstruction
from gaussian_renderer import generate_neural_gaussians_SDF

# for marching cube
from skimage import measure
import plotly.graph_objects as go
import trimesh

# for tsdf fusion
import vdbfusion
from utils.graphics_utils import depth2point

from sklearn.neighbors import NearestNeighbors


def get_grid_uniform(resolution, grid_boundary=[-2.0, 2.0]):
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}
    
@torch.no_grad()
def get_surface_trace(path, gs, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0):
    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(gs.get_sdf_value(pnts.cuda()).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            color='#ffffff', opacity=1.0, flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export(path, 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

def get_classes():
    classes= []

    with open("info_semantic.json", 'r') as classes_file:
        data = json.load(classes_file)

    for objects in data['classes']:
        classes.append(objects['name'])

    return classes

def row_softmax(row):

    row = row.astype(np.float64, copy=True)

    row -= row.max(axis=1, keepdims=True)

    np.exp(row, out=row)
    row /= row.sum(axis=1, keepdims=True)

    return row


def mean_logit_smoothing(anchors_xyz,logits,k=30,self_weight=2.0):
   
    N = logits.shape[0]

    nn = NearestNeighbors(n_neighbors=min(k, N)).fit(anchors_xyz)
    _, neighbours = nn.kneighbors(anchors_xyz, return_distance=True)

    neighLogits = logits[neighbours].sum(axis=1)

    selfFactor = self_weight * logits

    # Normalization
    denominator = self_weight + neighbours.shape[1]

    return (selfFactor + neighLogits) / denominator

colors = (
    (242,73,73),(109,127,248),(242,221,73),(73,155,242),(242,142,73),
    (155,73,242),(73,242,242),(242,73,203),(203,242,73),(242,160,203),
    (244,108,129),(203,191,242),(191,142,73),(242,236,191),(128,38,38),
    (191,242,203),(142,142,38),(242,209,191),(38,38,128),(210,137,91),
    (242,38,38),(38,242,38),(38,38,242),(242,242,38),(242,38,242),
    (38,242,242),(191,191,191),(128,38,128),(202,191,155),(83,128,38),
    (38,128,83),(38,83,128),(83,38,128),(128,38,83),(191,38,38),
    (38,191,38),(38,38,191),(191,191,38),(191,38,191),(38,191,191),
    (102,38,38),(38,102,38),(38,38,102),(102,102,38),(102,38,102),
    (38,102,102),(157,17,16),(83,166,83),(83,83,166),(166,166,83),
    (166,83,166),(83,166,166),(204,121,38),(38,204,121),(121,38,204),
    (204,38,121),(121,204,38),(38,121,204),(153,96,38),(38,153,96),
    (96,38,153),(153,38,96),(96,153,38),(38,96,153),(242,162,38),
    (162,242,38),(38,242,162),(38,162,242),(162,38,242),(242,38,162),
    (204,142,83),(83,204,142),(142,83,204),(204,83,142),(203,235,193),
    (83,142,204),(242,121,121),(121,242,121),(69,247,172),(242,242,121),
    (242,121,242),(121,242,242),(64,64,140), (140,64,64),(64,140,64),
    (191,96,96),(96,191,96),(96,96,191),(191,191,96),(191,96,191),
    (3,97,104),(77,38,38),(38,77,38),(38,38,77),(77,77,38),
    (77,38,77), (242,74,190), (182,145,212), (199,206,110), (245,134,71),
    (27,253,70)
)
assert len(colors) == 101

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, mesh_type="mcube"):
    if mesh_type == "poisson":        
        points, color, opaicity,scaling,rot, normal, _, _, _,anchor_from_gaussian = generate_neural_gaussians_SDF(views[0], gaussians, visible_mask=None)
        with torch.no_grad():
            print("anchor:", gaussians.get_anchor.shape)
            if hasattr(gaussians, "sem_logits") and gaussians.sem_logits is not None:
                print("sem_logits shape:", gaussians.sem_logits.shape)
            else:
                print("sem_logits is None")
            
            # 1
            # anchor_xyz = gaussians.get_anchor.detach().cpu().numpy()
            # logits = gaussians.sem_logits.detach().cpu().numpy()
            # smoothed_logits = mean_logit_smoothing(anchor_xyz, logits)
            # probs = row_softmax(smoothed_logits)
            # cls_idx = np.full(probs.shape[0], -1, int)
            # mask_conf = probs.max(axis=1) >= 0.5   # shape (N,)
            # cls_idx[mask_conf] = probs[mask_conf].argmax(axis=1)

            # 2
            # logits = gaussians.sem_logits.detach().cpu().numpy()
            # probs= softmax(logits, axis=1)
            # cls_idx = np.full(probs.shape[0], -1, int)
            # mask_conf = probs.max(axis=1) >= 0.5
            # cls_idx[mask_conf] = probs[mask_conf].argmax(axis=1)

            # 3
            # logits = gaussians.sem_logits.detach().cpu().numpy() 
            # probs = softmax(logits, axis=1)
            # cls_idx = probs.argmax(axis=1).astype(int)
            
            # 4
            logits = gaussians.sem_logits.detach().cpu().numpy()
            anchor_xyz = gaussians.get_anchor.detach().cpu().numpy()
            # smoothed_logits = mean_logit_smoothing(anchor_xyz, logits)
            smoothed_logits = mean_logit_smoothing(anchor_xyz, logits)
            probs = row_softmax(smoothed_logits)
            cls_idx = probs.argmax(axis=1).astype(np.int32)
        
        anchor_xyz = gaussians.get_anchor.detach().cpu().numpy()

        palette = (np.asarray(colors, dtype=np.float32) / 255.0)

        cls = cls_idx.copy()
        cls = np.clip(cls, 0, len(palette) - 1)

        anchor_colors = palette[cls]    

        # anchor_colors = np.zeros((anchor_xyz.shape[0], 3), dtype=np.float32)
        # # cls==60 -> white
        # anchor_colors[cls_idx == 90] = (1.0, 1.0, 1.0)

        # noise = (cls_idx == -1)
        # anchor_colors[noise] = np.array([0.6, 0.6, 0.6], dtype=np.float64)
        
        
        # noise = (cls_idx == -1)
        # if noise.any():
        #     anchor_colors[noise] = np.array([0, 0, 0], dtype=np.float64)

        points = points.cpu().detach().numpy()
        points_normals = torch.nn.functional.normalize(normal).cpu().detach().numpy()
        vertices, triangle, pcd = poisson_surface_reconstruction(points, points_normals, 8) # 9
        # vertices, triangle, pcd = poisson_surface_reconstruction(points, None, 9)
        # use open3d to save the mesh
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangle)
        mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
        
        # pcd = pcd.voxel_down_sample(voxel_size=0.01)
        scale_matrix = np.diag([50, 50, 50])
        pcd.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd.points).T).T)
        normals = np.asarray(pcd.normals)
        scaled_normals =normals * 0.1
        pcd.normals = o3d.utility.Vector3dVector(scaled_normals)
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        # mesh.compute_vertex_normals()

        from scipy.spatial import cKDTree
        kdtree = cKDTree(anchor_xyz)
        verts = np.asarray(mesh.vertices)
        _, idx = kdtree.query(verts, k=1)

        v_colors = anchor_colors[idx] 

        mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))
        
        o3d.io.write_triangle_mesh(os.path.join(model_path, "TESTsemantic_mesh_poisson_mean_d8l0.1_{}".format(iteration)+ ".ply"), mesh, write_vertex_colors=True)
      
    elif mesh_type == "mcube":
        _ = get_surface_trace(
            path = os.path.join(model_path, "extracted_mesh_marching_cube_{}".format(iteration)+".ply"),
            gs=gaussians,
            resolution=512,
            grid_boundary=[-1.0, 1.0],
            level=0.0
        )
    elif mesh_type == "tsdf":
        # use TSDF fusion with rendered mean to reconstruct surface
        # reference project:
        # https://github.com/GAP-LAB-CUHK-SZ/gaustudio
        # https://github.com/surfsplatting/surfsplatting.github.io/blob/main/assets/paper/paper.pdf
        vdb_volume = vdbfusion.VDBVolume(voxel_size=0.005, sdf_trunc=0.08, space_carving=True)
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            torch.cuda.synchronize()
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
            render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
            torch.cuda.synchronize()
            
            render_depth = render_pkg["render_depth"].squeeze()
            
            
            render_pcd_cam, render_pcd_world = depth2point(render_depth, torch.inverse(view.inv_intrinsic).to(render_depth.device), view.world_view_transform.transpose(0, 1).to(render_depth.device))

            vdb_volume.integrate(render_pcd_world.view(-1, 3).double().cpu().numpy(), extrinsic=view.camera_center.double().cpu().numpy())
            
        # get mesh from vdb_volume
        vertices, faces = vdb_volume.extract_triangle_mesh(min_weight = 5)
        geo_mesh = trimesh.Trimesh(vertices, faces)
        geo_mesh.export(os.path.join(model_path, "extracted_mesh_tsdf.ply"), 'ply')
        
    else:
        raise NotImplementedError
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, mesh_type: str, checkpoint: str):
    
    
    classes = get_classes()
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(classes,dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if(checkpoint):
            ckpt = torch.load(checkpoint, map_location="cuda")
            if isinstance(ckpt, tuple) and isinstance(ckpt[0], tuple):
                capture = ckpt[0]
           
                sem_logits = capture[7]
                    
                gaussians.sem_logits = torch.nn.Parameter(sem_logits.to(gaussians._anchor.device), requires_grad=False)
            
                gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # use training view to filter out the voxels    
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, mesh_type)

    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mesh_type", default="poisson", type=str)
    parser.add_argument("--checkpoint_path")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.mesh_type, args.checkpoint_path)
