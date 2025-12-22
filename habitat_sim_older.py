#!/usr/bin/env python3
"""
Render Replica RGB + semantic images from COLMAP cameras.

This minimal version:
 - reads COLMAP cameras.bin / images.bin,
 - converts poses to Habitat coordinate frame (COLMAP -> OpenGL),
 - renders RGB and semantic views.

No alignment, no scaling, no apply_full_transform.
"""

import os
import argparse
import collections
import struct
import numpy as np
import imageio.v3 as iio
import habitat_sim
import quaternion  # pip install numpy-quaternion
import magnum as mn


# -----------------------------------------------------------------------------
# COLMAP binary reader (minimal)
# -----------------------------------------------------------------------------

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
}
CAMERA_MODEL_IDS = dict([(m.model_id, m) for m in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    return struct.unpack(endian + fmt, fid.read(num_bytes))


def qvec2rotmat(qvec):
    """Convert COLMAP quaternion [w,x,y,z] to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,       2*y*z + 2*w*x,     1 - 2*x**2 - 2*y**2]
    ])


def read_intrinsics_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            cam_props = read_next_bytes(fid, 24, "iiQQ")
            cam_id, model_id, width, height = cam_props
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, 8*num_params, "d"*num_params)
            cameras[cam_id] = Camera(cam_id, CAMERA_MODEL_IDS[model_id].model_name,
                                     width, height, np.array(params))
    return cameras


def read_extrinsics_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_imgs = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_imgs):
            props = read_next_bytes(fid, 64, "idddddddi")
            img_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            cam_id = props[8]
            name = ""
            c = read_next_bytes(fid, 1, "c")[0]
            while c != b"\x00":
                name += c.decode("utf-8")
                c = read_next_bytes(fid, 1, "c")[0]
            num_pts2D = read_next_bytes(fid, 8, "Q")[0]
            _ = fid.read(24 * num_pts2D)  # skip x,y,point_id triplets
            images[img_id] = BaseImage(img_id, qvec, tvec, cam_id, name, None, None)
    return images


# -----------------------------------------------------------------------------
# COLMAP → Habitat pose conversion
# -----------------------------------------------------------------------------

def build_c2w_from_colmap(image, cameras):
    """
    Convert COLMAP camera pose (world→cam) to Habitat (cam→world).
    Habitat uses OpenGL convention: +X right, +Y up, -Z forward.
    COLMAP uses: +X right, +Y down, +Z forward.
    """
    R_wc = qvec2rotmat(image.qvec)  # world→cam
    t_wc = image.tvec.reshape(3, 1)

    R_cw = R_wc.T
    C_world = -R_cw @ t_wc  # camera center in world

    # COLMAP -> OpenGL basis: flip Y and Z
    B = np.diag([1, -1, -1])
    R_cw_hab = R_cw @ B

    c2w = np.eye(4)
    c2w[:3, :3] = R_cw_hab
    c2w[:3, 3] = C_world[:, 0]
    return c2w


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to numpy.quaternion."""
    return quaternion.from_rotation_matrix(R)


def colmap_intrinsics_to_fov(camera):
    """Compute FOV from COLMAP intrinsics."""
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    elif camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        fx = fy = f
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")
    W, H = camera.width, camera.height
    fovx = 2 * np.arctan(W / (2 * fx))
    fovy = 2 * np.arctan(H / (2 * fy))
    return np.degrees(fovx), np.degrees(fovy)


# -----------------------------------------------------------------------------
# Habitat rendering
# -----------------------------------------------------------------------------

def make_sim(scene_path: str, width: int, height: int, hfov_deg: float) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [height, width]  # this is ok as a list
    rgb_sensor_spec.position = mn.Vector3(0.0, 0.0, 0.0)
    rgb_sensor_spec.hfov = hfov_deg

    sem_sensor_spec = habitat_sim.CameraSensorSpec()
    sem_sensor_spec.uuid = "semantic_sensor"
    sem_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem_sensor_spec.resolution = [height, width]
    sem_sensor_spec.position = mn.Vector3(0.0, 0.0, 0.0)
    sem_sensor_spec.hfov = hfov_deg

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec, sem_sensor_spec]

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


def render_from_colmap(sim, images, cameras, out_rgb, out_sem):
    os.makedirs(out_rgb, exist_ok=True)
    os.makedirs(out_sem, exist_ok=True)

    agent = sim.get_agent(0)

    for img in images.values():
        cam = cameras[img.camera_id]
        c2w = build_c2w_from_colmap(img, cameras)
        pos = c2w[:3, 3]
        R = c2w[:3, :3]

        state = agent.get_state()
        state.position = pos.tolist()
        state.rotation = rotation_matrix_to_quaternion(R)
        agent.set_state(state)

        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][..., :3]
        sem = obs["semantic_sensor"].astype(np.int32)

        iio.imwrite(os.path.join(out_rgb, img.name), rgb)
        np.save(os.path.join(out_sem, img.name.replace(".png", ".npy")), sem)
        print(f"Rendered {img.name}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=str)
    ap.add_argument("--colmap_model", required=True, type=str)
    ap.add_argument("--out_rgb", default="out_rgb", type=str)
    ap.add_argument("--out_sem", default="out_sem", type=str)
    args = ap.parse_args()

    cam_bin = os.path.join(args.colmap_model, "cameras.bin")
    img_bin = os.path.join(args.colmap_model, "images.bin")

    cams = read_intrinsics_binary(cam_bin)
    imgs = read_extrinsics_binary(img_bin)

    first_cam = cams[next(iter(imgs.values())).camera_id]
    fovx, fovy = colmap_intrinsics_to_fov(first_cam)

    print(f"Loaded {len(cams)} cameras, {len(imgs)} images")
    print(f"Camera: {first_cam.model}, size={first_cam.width}x{first_cam.height}, FOVx={fovx:.2f}, FOVy={fovy:.2f}")

    sim = make_sim(args.scene, first_cam.width, first_cam.height, fovx)
    render_from_colmap(sim, imgs, cams, args.out_rgb, args.out_sem)
    sim.close()


if __name__ == "__main__":
    main()
