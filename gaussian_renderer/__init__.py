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
import torch
from einops import repeat
import numpy as np
import matplotlib.pyplot as plt

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from scene.gaussian_model import GaussianModel
from scene.gaussian_model_implicit import GaussianModel
from utils.general_utils import GetMinEigenVector, build_scaling_rotation, build_scaling_inv_rotation, mls_sdf, strip_symmetric

def generate_neural_gaussians_SDF(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    
    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3]

    # get offset's opacity
    # neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    neural_opacity = pc.get_opacity_mlp(feat) # [N, k], discard the view information in opacity

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)
    
    assert mask.sum() > 0, "no visible gaussians"

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    # scale_rot = pc.get_cov_mlp(cat_local_view)
    scale_rot = pc.get_cov_mlp(feat) # discard the view information in the cov mlp
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    assert torch.isnan(scale_rot).sum() == 0, "scale_rot has nan"
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    
    # normal: use learnable parameter
    # grid_normals = pc._normal[visible_mask]
    # normal = grid_normals.view([-1, 3]) # [mask]

    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    
    # calculate the IMLS, reference: "Provably Good Moving Least Squares" by Kolluri
    normal_unmask = GetMinEigenVector(concatenated_all[:, 3:6]*torch.sigmoid(scale_rot[:, :3]), pc.rotation_activation(scale_rot[:,3:7])).view(-1, 3) # [NK, 3]
    L = build_scaling_inv_rotation(concatenated_all[:, 3:6]*torch.sigmoid(scale_rot[:, :3]), pc.rotation_activation(scale_rot[:,3:7])) # [NK, 3, 3]
    actual_cov =  (L @ L.transpose(1, 2))# NK, 3, 3
    cov3D_unmask = strip_symmetric(actual_cov) # NK, 6
    
    
    offsets_unmask = (offsets * concatenated_all[:, :3]).view(-1, 3) # Nk, 3
    
    # use all offset3d
    # offset_3d_all = (offset_3d.unsqueeze(2) - offset_3d.unsqueeze(1)) # N, k, k, 3
    # offset_3d_all = offset_3d_all.view(-1, pc.n_offsets, 3) # NK, K, 3
    

    # results = offset_3d_all.unsqueeze(-2)@cov3D.unsqueeze(1)@offset_3d_all.unsqueeze(-1) # NK, K, 1, 1
    # density = torch.exp(-0.5* neural_opacity.unsqueeze(-1) * results.squeeze(-1)) # Nk, k, 1
    
    # estimated_imls = density * (offset_3d_all @ normal_unmask.unsqueeze(-1)) # Nk, k, 1
    # mask_point = (mask.view(-1, pc.n_offsets, 1) * mask.view(-1, 1, pc.n_offsets)).view(-1, pc.n_offsets, 1)
    # estimated_imls = estimated_imls * mask_point # mask out the negative opacity
    # density_denom = density * mask_point
    # estimated_imls = estimated_imls.sum(1) / (density_denom.sum(1)+1e-6)
    # estimated_imls = estimated_imls.view(-1, pc.n_offsets) * mask.view(-1, pc.n_offsets)
    # estimated_imls = estimated_imls.sum(1) / ((density.view(-1, pc.n_offsets) * mask.view(-1, pc.n_offsets)).sum(1)+1e-6)
    
    
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    
    # post-process cov
    scaling = torch.nan_to_num(scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])) # * (1+torch.sigmoid(repeat_dist))
    
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    
    # # normal: use the minimum egien vector corresponding to scaling to decide the normal vector
    # grid_normal = GetMinEigenVector(pc.rotation_activation(scale_rot[:,3:7]), scale_rot[:,:3])
    normal = GetMinEigenVector(scaling, rot) 
    
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets
    
    if is_training:
        return xyz, color, opacity, scaling, rot, normal, offsets_unmask, normal_unmask, cov3D_unmask, anchor, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot, normal, offsets_unmask, normal_unmask, cov3D_unmask, anchor

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    


    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    
    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3]

    # get offset's opacity
    # neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    neural_opacity = pc.get_opacity_mlp(feat) # [N, k], discard the view information in opacity

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)
    
    assert mask.sum() > 0, "no visible gaussians"

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    # scale_rot = pc.get_cov_mlp(cat_local_view)
    scale_rot = pc.get_cov_mlp(feat) # discard the view information in the cov mlp
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    assert torch.isnan(scale_rot).sum() == 0, "scale_rot has nan"
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    
    # normal: use learnable parameter
    # grid_normals = pc._normal[visible_mask]
    # normal = grid_normals.view([-1, 3]) # [mask]


    
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets= masked.split([6, 3, 3, 7, 3], dim=-1)
    
    
    # post-process cov
    scaling = torch.nan_to_num(scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])) # * (1+torch.sigmoid(repeat_dist))
    
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    
    # # normal: use the minimum egien vector corresponding to scaling to decide the normal vector
    # grid_normal = GetMinEigenVector(pc.rotation_activation(scale_rot[:,3:7]), scale_rot[:,:3])
    normal = GetMinEigenVector(scaling, rot) 
    
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets
    

    if is_training:
        return xyz, color, opacity, scaling, rot, normal, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot, normal

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, learn_SDF=True, class_subset = None):  # visible_mask - all anchors which radius >0 (projected)
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        if learn_SDF:
            xyz, color, opacity, scaling, rot, normal, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
        else:
            xyz, color, opacity, scaling, rot, normal, offsets_unmask, normal_unmask, cov3D_unmask, visbile_anchor, neural_opacity, mask = generate_neural_gaussians_SDF(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        if learn_SDF:
            xyz, color, opacity, scaling, rot, normal = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
        else:
            xyz, color, opacity, scaling, rot, normal, offsets_unmask, normal_unmask, cov3D_unmask, visbile_anchor = generate_neural_gaussians_SDF(viewpoint_camera, pc, visible_mask, is_training=is_training)
    
    assert scaling is not None

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # use the minimum egien vector corresponding to scaling to decide the normal vector
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_depth, render_normal, render_median_depth, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        normal_precomp = pc.normal_activation(normal),
        # opacities = (opacity > 0).float().cuda(),
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    # normalize the render normal
    render_normal = torch.nn.functional.normalize(render_normal, dim=0)
    render_normal = render_normal.contiguous()
    
    # === NEW === # semantic
    p_sem = None
    if is_training and class_subset is not None:
        # [N_anchor_total, K]
        Pi_anchor = pc.get_sem_probs()   # softmax(_sem_logits) along classes
        K = Pi_anchor.shape[1]
        

        # visible_mask: [N_anchor_total] (bool) — passed into this render()
        # mask: [N_visible_anchors * n_offsets] (bool) — returned by generate_* (which offsets are active)
        if visible_mask is not None:
            anchor_idx_visible = torch.nonzero(visible_mask, as_tuple=False).squeeze(1)   # [N_visible_anchors]
        else:
            anchor_idx_visible = torch.arange(pc.get_anchor.shape[0], device=xyz.device)

        N_vis_anchors = anchor_idx_visible.numel()
        
        xyz_sem = xyz.detach()
        screenspace_sem = screenspace_points.detach()
        normal_sem = normal.detach()
        opacity_sem = opacity.detach()
        scaling_sem = scaling.detach()
        rot_sem = rot.detach()

        # For each visible anchor, repeat its index n_offsets times to match flattened offsets:
        owner_idx_full = anchor_idx_visible.repeat_interleave(pc.n_offsets)               # [N_visible_anchors * n_offsets]

        # Select only the active offsets that produced xyz/color/etc.
        gauss_mask = mask.view(-1)                                                       # [N_visible_anchors * n_offsets]
        owner_idx = owner_idx_full[gauss_mask]                                           # [N_rendered_gaussians]
        assert gauss_mask.numel() == N_vis_anchors * pc.n_offsets, \
        "mask length must equal N_vis_anchors * n_offsets (anchor-major flattening)"
        assert owner_idx.shape[0] == xyz.shape[0], \
        f"owner_idx {owner_idx.shape} must match rendered gaussians {xyz.shape[0]}"

        Pi_gauss = Pi_anchor[owner_idx]
        assert Pi_gauss.shape == (xyz.shape[0], K)

       
        def render_scalar_field_as_image(x_scalar_per_gaussian: torch.Tensor) -> torch.Tensor:
            # uses probabilities  as a 'color' input to rasterizer 
            colors_precomp_x = x_scalar_per_gaussian.expand(-1, 3).contiguous()
            # color_x, _, _, _, _ = rasterizer(
            #     means3D = xyz,
            #     means2D = screenspace_points,
            #     shs = None,
            #     colors_precomp = colors_precomp_x,  
            #     normal_precomp = pc.normal_activation(normal),
            #     opacities = opacity,
            #     scales = scaling,
            #     rotations = rot,
            #     cov3D_precomp = None
            # )
            color_x, _, _, _, _ = rasterizer(
                means3D = xyz_sem,
                means2D = screenspace_sem,
                shs = None,
                colors_precomp = colors_precomp_x,  
                normal_precomp = pc.normal_activation(normal_sem),
                opacities = opacity_sem,
                scales = scaling_sem,
                rotations = rot_sem,
                cov3D_precomp = None
            )
            
            if color_x.dim() == 3:
                return color_x[0:1, :, :].unsqueeze(0)    # -> [1,1,H,W]
            else:
                return color_x[:, :1, :, :] 

        # accumulate semantic 'color' s_c for each class
        H, W = rendered_image.shape[-2], rendered_image.shape[-1]
        p_sem = rendered_image.new_zeros((1, K, H, W))  # [1,K,H,W]
        for c in class_subset:
            x = Pi_gauss[:, c:c+1]               # [N_rendered_gaussians, 1]
            s_c = render_scalar_field_as_image(x)  # [1,1,H,W]
            p_sem[:, c:c+1, :, :] = s_c

        ones = torch.ones((Pi_gauss.shape[0], 1), device=Pi_gauss.device, dtype=Pi_gauss.dtype)
        S = render_scalar_field_as_image(ones)     # [1,1,H,W]

        p_sem = p_sem / S.clamp_min(1e-6)  

        labels = p_sem.argmax(dim=1)            # [1, H, W]
        mask36 = (labels == 36).squeeze(0)      # [H, W], bool

        # Save binary mask (white where argmax==36, black elsewhere)
        plt.imsave(
            "argmax_eq_36.png",
            mask36.detach().cpu().numpy().astype(np.float32),  # values 0.0 / 1.0
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    # === END NEW ===

    
    L = build_scaling_inv_rotation(scaling, rot)
    actu_cov3D = (L @ L.transpose(1, 2))
    cov3D = strip_symmetric(actu_cov3D)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "render_depth": rendered_depth,
                "render_median_depth": render_median_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "points": xyz,
                "render_normal": render_normal,
                "normal": normal,
                "cov3D": cov3D,
                "offsets_unmask": None if learn_SDF else offsets_unmask,
                "normal_unmask": None if learn_SDF else normal_unmask,
                "cov3D_unmask": None if learn_SDF else cov3D_unmask,
                "opacity": opacity,
                "visible_anchor": None if learn_SDF else visbile_anchor,
                "semantics": p_sem  # semantic
                }
    else:
        return {"render": rendered_image,
                "render_depth": rendered_depth,
                "render_median_depth": render_median_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "points": xyz,
                "render_normal": render_normal,
                "normal": normal,
                "cov3D": cov3D,
                "opacity": opacity,
                "offsets_unmask": None if learn_SDF else offsets_unmask,
                "normal_unmask": None if learn_SDF else normal_unmask,
                "cov3D_unmask": None if learn_SDF else cov3D_unmask,
                "visible_anchor": None if learn_SDF else visbile_anchor,
                }

def render_semantics(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, logits, scaling_modifier = 1.0, visible_mask=None, learn_SDF=False):  # visible_mask - all anchors which radius >0 (projected)
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        if learn_SDF:
            xyz, color, opacity, scaling, rot, normal, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
        else:
            xyz, color, opacity, scaling, rot, normal, offsets_unmask, normal_unmask, cov3D_unmask, visbile_anchor, neural_opacity, mask = generate_neural_gaussians_SDF(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        if learn_SDF:
            xyz, color, opacity, scaling, rot, normal = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
        else:
            xyz, color, opacity, scaling, rot, normal, offsets_unmask, normal_unmask, cov3D_unmask, visbile_anchor, mask = generate_neural_gaussians_SDF(viewpoint_camera, pc, visible_mask, is_training=is_training)
    
    assert scaling is not None


    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # use the minimum egien vector corresponding to scaling to decide the normal vector
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_depth, render_normal, render_median_depth, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        normal_precomp = pc.normal_activation(normal),
        # opacities = (opacity > 0).float().cuda(),
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    # normalize the render normal
    render_normal = torch.nn.functional.normalize(render_normal, dim=0)
    render_normal = render_normal.contiguous()


    # --- APPROACH 1 - SPLAT
    # === EVAL SEMANTICS (no big K×H×W tensor) ===
    # Per-anchor class probs
    # Pi_anchor = pc.get_sem_probs()                 # [N_anchor, K], softmaxed
    # logits_t = torch.as_tensor(logits, dtype=torch.float32, device='cuda')
    # Pi_anchor = torch.softmax(logits_t, dim=1)
    # K = Pi_anchor.shape[1]

    # # Visible anchors -> rendered gaussian owners (unchanged from your code)
    # if visible_mask is not None:
    #     anchor_idx_visible = torch.nonzero(visible_mask, as_tuple=False).squeeze(1)
    # else:
    #     anchor_idx_visible = torch.arange(pc.get_anchor.shape[0], device=xyz.device)
    # owner_idx_full = anchor_idx_visible.repeat_interleave(pc.n_offsets)
    # gauss_mask = mask.view(-1)
    # owner_idx = owner_idx_full[gauss_mask]

    # Pi_gauss = Pi_anchor[owner_idx]                # [N_rendered, K]

    # # helper: render a scalar per-gaussian field (reusing your detached geo)
    # def render_scalar_field_as_image(x_scalar_per_gaussian: torch.Tensor) -> torch.Tensor:
    #     colors_precomp_x = x_scalar_per_gaussian.expand(-1, 3).contiguous()
    #     color_x, _, _, _, _ = rasterizer(
    #         means3D = xyz.detach(),
    #         means2D = screenspace_points.detach(),
    #         shs = None,
    #         colors_precomp = colors_precomp_x,
    #         normal_precomp = pc.normal_activation(normal.detach()),
    #         opacities = opacity.detach(),
    #         scales = scaling.detach(),
    #         rotations = rot.detach(),
    #         cov3D_precomp = None
    #     )
    #     return color_x[:, :1, :, :] if color_x.dim() != 3 else color_x[0:1, :, :].unsqueeze(0)

    # # Ones pass (sum of weights) — shared denominator
    # ones = torch.ones((Pi_gauss.shape[0], 1), device=Pi_gauss.device, dtype=Pi_gauss.dtype)
    # with torch.no_grad():
    #     S = render_scalar_field_as_image(ones)     # [1,1,H,W]

    # # Stream argmax over classes using s_c (same argmax as s_c / S)
    # H, W = rendered_image.shape[-2], rendered_image.shape[-1]
    # best_score = rendered_image.new_full((1,1,H,W), -1e9)
    # best_class = torch.full((1,1,H,W), -1, dtype=torch.long, device=rendered_image.device)

    
    # with torch.no_grad():
    #     for c in range(K):
    #         s_c = render_scalar_field_as_image(Pi_gauss[:, c:c+1])  # [1,1,H,W]

    #         labels_np = s_c.detach().cpu().numpy()
            
    #         # optional: treat -1 as background class 0 for visualization
    #         vis = labels_np.copy()
    #         # vis[vis < 0] = 0
    #         vis = np.where(vis < 0, 0, vis + 1)

    #         # (optional) pick a discrete colormap
    #         plt.imsave("sem_labels.png", vis, cmap="viridis", vmin=0, vmax=K)
    #         better = s_c > best_score
    #         best_class = torch.where(better, torch.full_like(best_class, c), best_class)
    #         best_score = torch.maximum(best_score, s_c)
    #         del s_c

    # door_id = 36

    # # how many anchors are door overall?
    # door_anchors = (torch.argmax(Pi_anchor, dim=1) == door_id).nonzero().numel()

    # # how many *rendered gaussians* are owned by door anchors?
    # door_rendered = (torch.argmax(Pi_anchor[owner_idx], dim=1) == door_id).nonzero().numel()
    # print(f"door anchors total: {door_anchors}, rendered in this view: {door_rendered}")

    # # Outputs for evaluation
    # sem_label_2d = best_class.squeeze(0).squeeze(0)                       # [H,W] long
    # p_sem_max    = (best_score / S.clamp_min(1e-6)).squeeze(0).squeeze(0) # [H,W] float



        # === APPROACH 2 - NEAREST GAUSS
    # import numpy as np
    # from collections import defaultdict

    # with torch.no_grad():
    #     # per-rendered-gaussian class id
    #     Pi_anchor = pc.get_sem_probs()                      # [N_anchor,K]
    #     y_anchor  = Pi_anchor.argmax(dim=1)                 # [N_anchor]

    #     if visible_mask is not None:
    #         anchor_idx_visible = torch.nonzero(visible_mask, as_tuple=False).squeeze(1)
    #     else:
    #         anchor_idx_visible = torch.arange(pc.get_anchor.shape[0], device=xyz.device)
    #     owner_idx_full = anchor_idx_visible.repeat_interleave(pc.n_offsets)
    #     gauss_mask = mask.view(-1)
    #     owner_idx  = owner_idx_full[gauss_mask]             # [N_rendered]

    #     y_gauss  = y_anchor[owner_idx].to(torch.long)       # [N_rendered]
    #     op_gauss = opacity.detach().view(-1)                # [N_rendered]  (no mask!)
    #     r_gauss  = radii.detach().view(-1)                  # [N_rendered]  (no mask!)


    #     # project to pixels and get camera-space z
    #     view_matrix = viewpoint_camera.world_view_transform.T
    #     proj_matrix = viewpoint_camera.projection_matrix.T
    #     full = proj_matrix @ view_matrix

    #     ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    #     pts_h = torch.cat([xyz, ones], dim=1)               # [N,4]
    #     clip  = (full @ pts_h.T).T                          # [N,4]
    #     w = clip[:, 3:4]
    #     ndc = clip[:, :3] / w

    #     H, W = rendered_image.shape[-2], rendered_image.shape[-1]
    #     u_f = ((ndc[:, 0] + 1) * 0.5 * W)
    #     v_f = ((1 - ndc[:, 1]) * 0.5 * H)

    #     cam = (view_matrix @ pts_h.T).T
    #     z    = cam[:, 2]                                    # smaller = nearer

    #     # keep valid, in-bounds, in front
    #     valid = (w[:,0] > 0) & (u_f >= 0) & (u_f < W) & (v_f >= 0) & (v_f < H)
    #     if not valid.any():
    #         sem_label_2d = torch.full((H, W), -1, dtype=torch.long, device=xyz.device)
    #     else:
    #         u = u_f[valid].detach().cpu().numpy().astype(np.float32)
    #         v = v_f[valid].detach().cpu().numpy().astype(np.float32)
    #         z = z[valid].detach().cpu().numpy()
    #         r = r_gauss[valid].detach().cpu().numpy().astype(np.float32)
    #         cls = y_gauss[valid].detach().cpu().numpy().astype(np.int32)

    #         # --- screen-space binning ---
    #         bin_size = 16  # px
    #         gx = (u // bin_size).astype(np.int32)
    #         gy = (v // bin_size).astype(np.int32)
    #         grid = defaultdict(list)
    #         for i in range(u.shape[0]):
    #             grid[(gx[i], gy[i])].append(i)

    #         out = np.full((H, W), -1, dtype=np.int32)
    #         inf = 1e30

    #         # how many sigmas define coverage (3≈99.7%)
    #         t2 = 3.0 * 3.0

    #         # iterate pixels in tiles for speed
    #         tile = 64
    #         for y0 in range(0, H, tile):
    #             for x0 in range(0, W, tile):
    #                 h = min(tile, H - y0); w_ = min(tile, W - x0)
    #                 yy, xx = np.mgrid[y0:y0+h, x0:x0+w_]
    #                 best_z  = np.full((h, w_), inf, dtype=np.float32)
    #                 best_id = np.full((h, w_), -1, dtype=np.int32)

    #                 # collect candidate indices from 3x3 neighbor bins once per tile
    #                 bx0 = x0 // bin_size
    #                 by0 = y0 // bin_size
    #                 cand = []
    #                 for by in range(by0-1, by0 + (h + bin_size - 1)//bin_size + 2):
    #                     for bx in range(bx0-1, bx0 + (w_ + bin_size - 1)//bin_size + 2):
    #                         cand.extend(grid.get((bx, by), []))
    #                 if not cand:
    #                     out[y0:y0+h, x0:x0+w_] = best_id
    #                     continue
    #                 cand = np.asarray(cand, dtype=np.int32)

    #                 uc = u[cand][:, None, None]    # [C,1,1]
    #                 vc = v[cand][:, None, None]
    #                 rc = r[cand][:, None, None]
    #                 zc = z[cand][:, None, None]
    #                 # coverage test: (dx/rc)^2 + (dy/rc)^2 <= t2
    #                 dx = xx[None, :, :] - uc
    #                 dy = yy[None, :, :] - vc
    #                 # avoid div-by-zero
    #                 rc = np.maximum(rc, 0.75)
    #                 cov = (dx/rc)**2 + (dy/rc)**2 <= t2

    #                 if cov.any():
    #                     # among covered candidates, pick smallest z
    #                     zcand = np.where(cov, zc, inf)
    #                     j = zcand.argmin(axis=0)         # [h,w]
    #                     best_z = zcand[j, np.arange(h)[:,None], np.arange(w_)[None,:]]
    #                     best_id = cls[cand][j]
    #                 out[y0:y0+h, x0:x0+w_] = best_id

    #         sem_label_2d = torch.from_numpy(out).to(xyz.device)
    # === END EVAL SEMANTICS ===

    # === APPROACH 3, one hot
    # --- collapse per-gaussian distribution to a single class ---
    # Pi_gauss: [N_rendered, K] from your code above
    Pi_anchor = pc.get_sem_probs()                 # [N_anchor, K], softmaxed
    logits_t = torch.as_tensor(logits, dtype=torch.float32, device='cuda')
    Pi_anchor = torch.softmax(logits_t, dim=1)
    K = Pi_anchor.shape[1]

    # Visible anchors -> rendered gaussian owners (unchanged from your code)
    if visible_mask is not None:
        anchor_idx_visible = torch.nonzero(visible_mask, as_tuple=False).squeeze(1)
    else:
        anchor_idx_visible = torch.arange(pc.get_anchor.shape[0], device=xyz.device)
    owner_idx_full = anchor_idx_visible.repeat_interleave(pc.n_offsets)
    gauss_mask = mask.view(-1)
    owner_idx = owner_idx_full[gauss_mask]

    Pi_gauss = Pi_anchor[owner_idx]                # [N_rendered, K]
    top_p, top_c = Pi_gauss.max(dim=1)                    # [N_rendered], [N_rendered]

    HARD_ONE_HOT = False  # True: value=1; False: value=top_p (soft one-hot, closer to original)
    weights_1 = (torch.ones_like(top_p) if HARD_ONE_HOT else top_p).unsqueeze(1)  # [N_rendered,1]

    # Classes that actually appear among argmaxes in THIS view (usually << K)
    present_classes = torch.unique(top_c)

    # Prepare outputs
    H, W = rendered_image.shape[-2], rendered_image.shape[-1]
    p_sem = rendered_image.new_zeros((1, K, H, W))        # [1,K,H,W]
    
    def render_scalar_field_as_image(x_scalar_per_gaussian: torch.Tensor) -> torch.Tensor:
        colors_precomp_x = x_scalar_per_gaussian.expand(-1, 3).contiguous()
        color_x, _, _, _, _ = rasterizer(
            means3D = xyz.detach(),
            means2D = screenspace_points.detach(),
            shs = None,
            colors_precomp = colors_precomp_x,
            normal_precomp = pc.normal_activation(normal.detach()),
            opacities = opacity.detach(),
            scales = scaling.detach(),
            rotations = rot.detach(),
            cov3D_precomp = None
        )
        return color_x[:, :1, :, :] if color_x.dim() != 3 else color_x[0:1, :, :].unsqueeze(0)

    # Shared denominator (sum of spatial weights); useful for soft one-hot or for p_sem_max
    with torch.no_grad():
        ones = torch.ones((Pi_gauss.shape[0], 1), device=Pi_gauss.device, dtype=Pi_gauss.dtype)
        S = render_scalar_field_as_image(ones)            # [1,1,H,W]

    # Rasterize only present classes
    for c in present_classes.tolist():
        is_c = (top_c == c).float().unsqueeze(1)          # [N_rendered,1] indicator
        x    = (is_c * weights_1).contiguous()            # one-hot per gaussian (soft or hard)
        s_c  = render_scalar_field_as_image(x)            # [1,1,H,W]
        if not HARD_ONE_HOT:
            s_c = s_c / S.clamp_min(1e-6)                 # normalize to approx prob
        p_sem[:, c:c+1, :, :] = s_c

    # Final 2D outputs (same shapes you used before)
    sem_label_2d = p_sem.argmax(dim=1).squeeze(0)                         # [H,W] long
    p_sem_max    = p_sem.max(dim=1).values.squeeze(0)                     # [H,W] float in [0,1] if not HARD_ONE_HOT


    
    # === END EVAL SEMANTICS ===

    
    L = build_scaling_inv_rotation(scaling, rot)
    actu_cov3D = (L @ L.transpose(1, 2))
    cov3D = strip_symmetric(actu_cov3D)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "render_depth": rendered_depth,
                "render_median_depth": render_median_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "points": xyz,
                "render_normal": render_normal,
                "normal": normal,
                "cov3D": cov3D,
                "offsets_unmask": None if learn_SDF else offsets_unmask,
                "normal_unmask": None if learn_SDF else normal_unmask,
                "cov3D_unmask": None if learn_SDF else cov3D_unmask,
                "opacity": opacity,
                "visible_anchor": None if learn_SDF else visbile_anchor,
                }
    else:
        return {"render": rendered_image,
                "render_depth": rendered_depth,
                "render_median_depth": render_median_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "points": xyz,
                "render_normal": render_normal,
                "normal": normal,
                "cov3D": cov3D,
                "opacity": opacity,
                "offsets_unmask": None if learn_SDF else offsets_unmask,
                "normal_unmask": None if learn_SDF else normal_unmask,
                "cov3D_unmask": None if learn_SDF else cov3D_unmask,
                "visible_anchor": None if learn_SDF else visbile_anchor,
                "sem_label_2d": sem_label_2d
                }

def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False # was pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    # before
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation

    scales = pc.get_scaling
    rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
