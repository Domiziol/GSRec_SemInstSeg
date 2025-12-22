#!/usr/bin/env python3
"""
Render Replica RGB + semantic GT from COLMAP cameras.

- Reads COLMAP cameras.bin / images.bin.
- Converts COLMAP poses -> Habitat camera poses.
- Optionally applies the same similarity transform you use in apply_full_transform().
- For each COLMAP image, renders RGB and semantic image from Habitat Replica.

This is meant as a starting point; tweak the alignment / axis conventions if needed.
"""

import os
import argparse
import collections
import struct

import numpy as np
import imageio.v3 as iio
import habitat_sim
import quaternion  # numpy-quaternion


# -----------------------------------------------------------------------------
# COLMAP LOADER (binary) – adapted from your snippet / 3DGS colmap_loader
# -----------------------------------------------------------------------------

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point3D_idxs"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_intrinsics_binary(path_to_model_file):
    """
    Read COLMAP cameras.bin
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params, dtype=np.float64))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_binary(path_to_model_file):
    """
    Read COLMAP images.bin
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5], dtype=np.float64)
            tvec = np.array(binary_image_properties[5:8], dtype=np.float64)
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


# -----------------------------------------------------------------------------
# Alignment – lift your apply_full_transform() to a 4×4 matrix.
# -----------------------------------------------------------------------------

def get_alignment_matrix(use_alignment: bool) -> np.ndarray:
    """
    Build a 4x4 similarity transform equivalent to your apply_full_transform().

    Original apply_full_transform:

        scale_factor = 4
        shift_vector = [2.95531, 1.13268, -0.058562]
        transform = [... 4x4 ...]

        pts = points * scale_factor + shift_vector
        pts_h = [pts, 1]
        pts_tf = (transform @ pts_h.T).T[:, :3]

    This means: pts_tf = (transform @ S @ [x,y,z,1]) with

        S = [[s,0,0,shift_x],
             [0,s,0,shift_y],
             [0,0,s,shift_z],
             [0,0,0,1]]

    We re-use exactly that here and apply it to camera poses as well.
    If you DON'T want alignment, set use_alignment=False or just return identity.
    """
    if not use_alignment:
        return np.eye(4, dtype=np.float64)

    scale_factor = 1.0
    shift_vector = np.array([2.95531, 1.13268, -0.058562], dtype=np.float64)

    transform = np.array([
        [1.06030165e+00, -3.20324289e-03,  7.84449640e-04, -1.23199711e-01],
        [3.20130877e-03,  1.06029875e+00,  2.60242687e-03, -4.07212017e-02],
        [-7.92305771e-04, -2.60004585e-03,  1.06030329e+00, -6.42458858e-02],
        [0.0,             0.0,             0.0,             1.0]
    ], dtype=np.float64)

    S = np.eye(4, dtype=np.float64)
    S[0, 0] = S[1, 1] = S[2, 2] = scale_factor
    S[0:3, 3] = shift_vector

    T_full = transform @ S
    return T_full


# -----------------------------------------------------------------------------
# COLMAP -> Habitat pose & intrinsics conversion
# -----------------------------------------------------------------------------

def postprocess_rgb(rgb):
    """
    rgb: H x W x 3 uint8 from Habitat
    return: H x W x 3 uint8, aligned to your training images
    """
    # rotate 90° CCW (k=1) then flip left-right
    rgb_rot = np.rot90(rgb, k=1, axes=(0, 1))
    rgb_rot = np.flip(rgb_rot, axis=1)
    return rgb_rot

def postprocess_semantic(sem):
    """
    sem: H x W int32 from Habitat (semantic_sensor)
    return: H x W int32, aligned to your training images
    """
    sem_rot = np.rot90(sem, k=1, axes=(0, 1))
    sem_rot = np.flip(sem_rot, axis=1)
    return sem_rot

def build_pre_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """
    Build a 3x3 rotation matrix R_pre from Euler angles (degrees),
    applied in order: Rx (X axis), then Ry, then Rz:
        R_pre = Rz @ Ry @ Rx
    so that a point x_colmap is mapped to x' = R_pre @ x_colmap
    before the alignment transform.

    Use e.g. (0, 90, 0) to rotate world by +90° around Y.
    """
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    Rx = np.array([
        [1.0,        0.0,         0.0],
        [0.0,  np.cos(rx), -np.sin(rx)],
        [0.0,  np.sin(rx),  np.cos(rx)],
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(ry), 0.0, np.sin(ry)],
        [ 0.0,        1.0, 0.0],
        [-np.sin(ry), 0.0, np.cos(ry)],
    ], dtype=np.float64)

    Rz = np.array([
        [ np.cos(rz), -np.sin(rz), 0.0],
        [ np.sin(rz),  np.cos(rz), 0.0],
        [ 0.0,         0.0,        1.0],
    ], dtype=np.float64)

    R_pre = Rz @ Ry @ Rx
    return R_pre

def apply_global_world_rotation(c2w: np.ndarray, angle_deg: float, axis: str = "z") -> np.ndarray:
    """
    Rotate the whole camera pose around a world axis by angle_deg.
    This rotates both:
      - camera position
      - camera orientation
    """
    if abs(angle_deg) < 1e-6:
        return c2w

    theta = np.deg2rad(angle_deg)

    if axis == "z":
        R_world = np.array([
            [ np.cos(theta), -np.sin(theta), 0.0],
            [ np.sin(theta),  np.cos(theta), 0.0],
            [ 0.0,            0.0,           1.0],
        ], dtype=np.float64)
    elif axis == "y":
        R_world = np.array([
            [ np.cos(theta),  0.0, np.sin(theta)],
            [ 0.0,            1.0, 0.0],
            [-np.sin(theta),  0.0, np.cos(theta)],
        ], dtype=np.float64)
    elif axis == "x":
        R_world = np.array([
            [1.0, 0.0,           0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta),  np.cos(theta)],
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown axis {axis}, use 'x', 'y' or 'z'.")

    R4 = np.eye(4, dtype=np.float64)
    R4[:3, :3] = R_world

    return R4 @ c2w


def colmap_intrinsics_to_fov(camera: Camera):
    """
    From COLMAP Camera (PINHOLE or SIMPLE_PINHOLE) compute fx, fy, fovx, fovy.
    """
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    elif camera.model == "SIMPLE_PINHOLE":
        # SIMPLE_PINHOLE : [f, cx, cy]
        f = camera.params[0]
        fx = fy = f
        cx = camera.params[1]
        cy = camera.params[2]
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}, "
                         "this script assumes PINHOLE/SIMPLE_PINHOLE.")

    W, H = camera.width, camera.height
    fovx = 2.0 * np.arctan(W / (2.0 * fx))
    fovy = 2.0 * np.arctan(H / (2.0 * fy))
    return fx, fy, cx, cy, fovx, fovy


def build_c2w_from_colmap(
    image: Image,
    cameras: dict,
    align_T: np.ndarray,
    roll_deg: float = 0.0,
    world_yaw_deg: float = 0.0,
    world_rot_deg: float = 0.0,
    R_pre = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build camera-to-world transform in Habitat/OpenGL convention.

    - Uses COLMAP world->cam (R_wc, t_wc).
    - Converts to cam->world (R_cw_col, C_world_col).
    - Applies camera-basis conversion COLMAP -> OpenGL/Habitat via B.
    - Applies optional roll in camera frame (roll_deg).
    - Applies optional global yaw in world frame (world_yaw_deg) around Z axis.
    - Applies optional alignment matrix align_T (similarity transform).
    """

    cam = cameras[image.camera_id]

    # --- 1. COLMAP world->cam: X_cam = R_wc * X_world + t_wc ---
    R_wc = qvec2rotmat(image.qvec)             # 3x3
    t_wc = image.tvec.reshape(3, 1)            # 3x1

    # camera->world in COLMAP world
    R_cw = R_wc.T                              # 3x3
    C_world = -R_cw @ t_wc                     # 3x1

    # --- 2. Optional pre-rotation of COLMAP world ---
    # x' = R_pre @ x_world
    if R_pre is not None:
        R_cw = R_pre @ R_cw
        C_world = R_pre @ C_world

    # --- 3. Build C2W in "pre-rotated COLMAP" world ---
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_cw
    c2w[:3, 3] = C_world[:, 0]

    # --- 4. Optional alignment to Replica world ---
    if align_T is not None:
        c2w = align_T @ c2w

    # --- 5. Orthonormalize rotation (remove numerical scaling) ---
    R_final = c2w[:3, :3]
    U, _, Vt = np.linalg.svd(R_final)
    R_final = U @ Vt
    c2w[:3, :3] = R_final

    position = c2w[:3, 3].copy()
    return position, R_final

    ## == VERSION WITH CAMERAS ROTATION
    # # 1) world->camera rotation in COLMAP convention
    # R_wc = qvec2rotmat(image.qvec)              # 3x3
    # t_wc = image.tvec.reshape(3, 1)             # 3x1

    # # 2) camera->world in COLMAP coordinates
    # R_cw_col = R_wc.T                           # 3x3
    # C_world_col = -R_cw_col @ t_wc              # 3x1

    # # 3) camera basis conversion: COLMAP -> OpenGL/Habitat
    # # COLMAP: x right, y down, z forward
    # # Habitat: x right, y up, z BACK (-z forward)
    # # This is the standard conversion:
    # B = np.diag([1.0, -1.0, -1.0])
    # R_cw_hab = R_cw_col @ B

    # # 4) optional roll around camera Z axis (to fix sideways images)
    # if abs(roll_deg) > 1e-6:
    #     theta = np.deg2rad(roll_deg)
    #     R_roll = np.array([
    #         [ np.cos(theta), -np.sin(theta), 0.0],
    #         [ np.sin(theta),  np.cos(theta), 0.0],
    #         [ 0.0,            0.0,           1.0],
    #     ], dtype=np.float64)
    #     # roll in camera frame -> multiply on the RIGHT
    #     R_cw_hab = R_cw_hab @ R_roll

    # # 5) optional global yaw around *world* Z axis
    # if abs(world_yaw_deg) > 1e-6:
    #     psi = np.deg2rad(world_yaw_deg)
    #     R_yaw = np.array([
    #         [ np.cos(psi), -np.sin(psi), 0.0],
    #         [ np.sin(psi),  np.cos(psi), 0.0],
    #         [ 0.0,          0.0,         1.0],
    #     ], dtype=np.float64)

    #     # yaw acts in world frame -> multiply on the LEFT
    #     R_cw_hab = R_yaw @ R_cw_hab
    #     C_world_col = R_yaw @ C_world_col

    # # 6) build homogeneous C2W
    # c2w = np.eye(4, dtype=np.float64)
    # c2w[:3, :3] = R_cw_hab
    # c2w[:3, 3] = C_world_col[:, 0]

    # # 7) optional alignment (similar to apply_full_transform)
    # if align_T is not None:
    #     c2w = align_T @ c2w

    # # 8) optional global world rotation (e.g. rotate all cameras 90° around Z)
    # if abs(world_rot_deg) > 1e-6:
    #     c2w = apply_global_world_rotation(c2w, world_rot_deg, axis="z")

    # # 9) clean numerical drift in rotation
    # R_final = c2w[:3, :3]
    # U, _, Vt = np.linalg.svd(R_final)
    # R_final = U @ Vt
    # c2w[:3, :3] = R_final

    # position = c2w[:3, 3].copy()
    # return position, R_final



def rotation_matrix_to_quaternion(R: np.ndarray) -> quaternion.quaternion:
    """
    Convert 3x3 rotation matrix to numpy.quaternion.
    """
    return quaternion.from_rotation_matrix(R)


# -----------------------------------------------------------------------------
# Habitat-Sim setup & rendering
# -----------------------------------------------------------------------------

def make_sim(scene_path: str, width: int, height: int, hfov_deg: float) -> habitat_sim.Simulator:
    """
    Create a Habitat simulator with RGB + semantic sensors mounted at the agent origin.
    """
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [height, width]  # [H, W]
    rgb_sensor_spec.position = [0.0, 0.0, 0.0]
    rgb_sensor_spec.hfov = hfov_deg

    sem_sensor_spec = habitat_sim.CameraSensorSpec()
    sem_sensor_spec.uuid = "semantic_sensor"
    sem_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem_sensor_spec.resolution = [height, width]
    sem_sensor_spec.position = [0.0, 0.0, 0.0]
    sem_sensor_spec.hfov = hfov_deg

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec, sem_sensor_spec]

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


def render_from_colmap(
    sim: habitat_sim.Simulator,
    images: dict,
    cameras: dict,
    align_T: np.ndarray,
    out_rgb_dir: str,
    out_sem_dir: str,
    roll_deg: float = 0.0,
    world_yaw_deg: float = 0.0,
    world_rot_deg: float = 0.0,
    R_pre = None,
):


    """
    For each COLMAP image, set the Habitat agent pose and render RGB + semantics.
    """
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_sem_dir, exist_ok=True)

    agent = sim.get_agent(0)

    for img_id, img in images.items():
        # Pose
        # position, R_c2w = build_c2w_from_colmap(
        #     img, cameras, align_T,
        #     roll_deg=roll_deg,
        #     world_rot_deg=world_rot_deg,  # <--- pass it down
        # )
        # q = rotation_matrix_to_quaternion(R_c2w)

        position, R_c2w = build_c2w_from_colmap(img, cameras, align_T, R_pre=R_pre)
        q = rotation_matrix_to_quaternion(R_c2w)

        state = agent.get_state()
        state.position = position.tolist()
        state.rotation = q
        agent.set_state(state)

        # Render
        obs = sim.get_sensor_observations()

        rgb = obs["color_sensor"]  # typically HxWx4 uint8 (RGBA)
        sem = obs["semantic_sensor"]  # HxW int32

        # strip alpha if present
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        rgb_path = os.path.join(out_rgb_dir, img.name)
        sem_path_npy = os.path.join(out_sem_dir, os.path.splitext(img.name)[0] + ".npy")
        sem_path_png = os.path.join(out_sem_dir, os.path.splitext(img.name)[0] + ".png")

        

        # Save RGB
        iio.imwrite(rgb_path, rgb)

        # Save semantics both as raw ids (.npy) and a colored visualization (.png)
        np.save(sem_path_npy, sem.astype(np.int32))

        # Simple color map for visualization (you can replace with Replica's palette)
        max_label = int(sem.max()) if sem.size > 0 else 0
        colors = np.random.RandomState(0).randint(0, 255, size=(max_label + 1, 3), dtype=np.uint8)
        sem_rgb = colors[sem.clip(0, max_label)]
        iio.imwrite(sem_path_png, sem_rgb)

        print(f"Rendered {img.name} -> {rgb_path}, {sem_path_npy}, {sem_path_png}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render Replica RGB + semantic GT from COLMAP cameras.")
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to Replica scene mesh (e.g. office_0/habitat/mesh_semantic.ply)")
    parser.add_argument("--colmap_model", type=str, required=True,
                        help="Path to COLMAP model directory containing cameras.bin and images.bin")
    parser.add_argument("--out_rgb", type=str, default="rgb_out",
                        help="Output directory for rendered RGB images")
    parser.add_argument("--out_sem", type=str, default="sem_out",
                        help="Output directory for rendered semantic GT")
    parser.add_argument("--use_alignment", action="store_true",
                        help="Apply similarity transform equivalent to apply_full_transform()")

    args = parser.parse_args()

    cam_bin = os.path.join(args.colmap_model, "cameras.bin")
    img_bin = os.path.join(args.colmap_model, "images.bin")

    if not os.path.exists(cam_bin) or not os.path.exists(img_bin):
        raise FileNotFoundError(f"Could not find cameras.bin/images.bin in {args.colmap_model}")

    print(f"Loading COLMAP intrinsics from {cam_bin}")
    cameras = read_intrinsics_binary(cam_bin)

    print(f"Loading COLMAP extrinsics from {img_bin}")
    images = read_extrinsics_binary(img_bin)

    if len(images) == 0:
        raise RuntimeError("No registered images found in images.bin")

    # Use the first image's camera to define sensor resolution + FOV
    first_img = next(iter(images.values()))
    first_cam = cameras[first_img.camera_id]
    fx, fy, cx, cy, fovx, fovy = colmap_intrinsics_to_fov(first_cam)

    print(f"Using camera {first_cam.id} model={first_cam.model}, "
          f"size={first_cam.width}x{first_cam.height}, fx={fx:.2f}, fy={fy:.2f}")
    print(f"FOVx={np.degrees(fovx):.2f} deg, FOVy={np.degrees(fovy):.2f} deg")

    align_T = get_alignment_matrix(args.use_alignment)
    if args.use_alignment:
        print("Applying alignment transform (similar to apply_full_transform()) to camera poses.")
    else:
        print("No alignment transform applied (identity).")

    print(f"Creating Habitat simulator for scene: {args.scene}")
    sim = make_sim(args.scene, first_cam.width, first_cam.height, np.degrees(fovx))

    R_pre = build_pre_rotation_matrix(0, 0, 0)

    print("Rendering from COLMAP cameras...")
    # render_from_colmap(sim, images, cameras, align_T,
    #                    args.out_rgb, args.out_sem)

    render_from_colmap(
        sim, images, cameras, align_T,
        args.out_rgb, args.out_sem,
        R_pre=R_pre,
    )

    sim.close()
    print("Done.")


if __name__ == "__main__":
    main()
