"""
SAPIEN-based robot-only renderer for RoboTwin inference.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml

ROBOT_NAME_MAP = {
    "aloha-agilex": "aloha-agilex",
    "arx-x5": "ARX-X5",
    "franka": "franka-panda",
    "piper": "piper",
    "ur5": "ur5-wsg",
}

DEFAULT_CAMERA_CONFIG = {
    "L515":       {"fovy": 45, "w": 320, "h": 180},
    "Large_L515": {"fovy": 45, "w": 640, "h": 360},
    "D435":       {"fovy": 37, "w": 320, "h": 240},
    "Large_D435": {"fovy": 37, "w": 640, "h": 480},
}


def parse_variant(variant_str: str) -> Tuple[str, str]:
    """Extract robot name and variant type from e.g. 'aloha-agilex_clean_50'."""
    for robot_prefix in sorted(ROBOT_NAME_MAP.keys(), key=len, reverse=True):
        if variant_str.startswith(robot_prefix + "_"):
            rest = variant_str[len(robot_prefix) + 1:]
            return robot_prefix, rest
    raise ValueError(f"Cannot parse robot name from variant: {variant_str}")


def load_embodiment_config(
    embodiment_dir: str, robot_prefix: str
) -> Tuple[Dict, str]:
    """Load the embodiment config.yml for the given robot."""
    embodiment_dir_name = ROBOT_NAME_MAP[robot_prefix]
    robot_dir = os.path.join(embodiment_dir, "embodiments", embodiment_dir_name)
    config_path = os.path.join(robot_dir, "config.yml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Embodiment config not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config, robot_dir


class RobotOnlyScene:
    """Minimal SAPIEN scene containing only the robot.

    Uses direct qpos setting (no physics simulation) for speed.
    """

    def __init__(self, embodiment_config: Dict, robot_dir: str,
                 camera_config_dict: Optional[Dict] = None):
        self.embodiment_config = embodiment_config
        self.robot_dir = robot_dir
        self.camera_config_dict = camera_config_dict or DEFAULT_CAMERA_CONFIG

        self.engine = None
        self.renderer = None
        self.scene = None
        self.robot = None

        self.active_joints = []
        self.joint_name_to_idx: Dict[str, int] = {}
        self.left_arm_indices: List[int] = []
        self.right_arm_indices: List[int] = []
        self.left_gripper_info: List[Tuple] = []
        self.right_gripper_info: List[Tuple] = []
        self.cameras: Dict = {}
        self.camera_names_ordered: List[str] = []

    def setup(self):
        import sapien.core as sapien
        import sapien.render
        from sapien.render import set_global_config

        self._sapien = sapien
        self._create_engine_and_scene()
        self._load_robot()
        self._build_joint_index()
        self._load_cameras()
        self._set_homestate()
        self.scene.step()
        self.scene.update_render()

    def _create_engine_and_scene(self):
        sapien = self._sapien
        import sapien.render
        from sapien.render import set_global_config

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.engine = sapien.Engine()
            set_global_config(max_num_materials=50000, max_num_textures=50000)
            self.renderer = sapien.SapienRenderer()
            self.engine.set_renderer(self.renderer)

        sapien.render.set_camera_shader_dir("default")

        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1 / 250)

        self.scene.set_ambient_light([0.8, 0.8, 0.8])
        self.scene.add_directional_light(
            [0, 0.5, -1], [0.5, 0.5, 0.5], shadow=False)
        self.scene.add_point_light(
            [1, 0, 1.8], [0.5, 0.5, 0.5], shadow=False)
        self.scene.add_point_light(
            [-1, 0, 1.8], [0.5, 0.5, 0.5], shadow=False)

    def _load_robot(self):
        sapien = self._sapien
        cfg = self.embodiment_config
        urdf_path = os.path.join(self.robot_dir, cfg["urdf_path"])

        robot_pose_raw = cfg.get(
            "robot_pose", [[0, -0.65, 0, 0.707, 0, 0, 0.707]])[0]
        pose = sapien.Pose(robot_pose_raw[:3], robot_pose_raw[-4:])

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(urdf_path)
        self.robot.set_root_pose(pose)

        for link in self.robot.get_links():
            link.set_mass(1)

    def _build_joint_index(self):
        cfg = self.embodiment_config
        self.active_joints = self.robot.get_active_joints()
        self.joint_name_to_idx = {
            j.get_name(): i for i, j in enumerate(self.active_joints)
        }

        self.left_arm_indices = [
            self.joint_name_to_idx[n] for n in cfg["arm_joints_name"][0]]
        self.right_arm_indices = [
            self.joint_name_to_idx[n] for n in cfg["arm_joints_name"][1]]

        self.gripper_scale = cfg.get("gripper_scale", [-0.01, 0.045])

        def parse_gripper(gripper_cfg):
            base_idx = self.joint_name_to_idx[gripper_cfg["base"]]
            entries = [(base_idx, 1.0, 0.0)]
            for m in gripper_cfg.get("mimic", []):
                entries.append((self.joint_name_to_idx[m[0]], m[1], m[2]))
            return entries

        self.left_gripper_info = parse_gripper(cfg["gripper_name"][0])
        self.right_gripper_info = parse_gripper(cfg["gripper_name"][1])

        self.left_camera_link = (
            self.robot.find_link_by_name("left_camera")
            or self.robot.find_link_by_name("camera")
            or self.robot.get_links()[0]
        )
        self.right_camera_link = (
            self.robot.find_link_by_name("right_camera")
            or self.robot.find_link_by_name("camera")
            or self.robot.get_links()[0]
        )

    def _set_homestate(self):
        cfg = self.embodiment_config
        left_home = cfg.get("homestate", [[0] * 6, [0] * 6])[0]
        right_home = cfg.get("homestate", [[0] * 6, [0] * 6])[1]

        qpos = np.zeros(len(self.active_joints), dtype=np.float32)
        for i, idx in enumerate(self.left_arm_indices):
            qpos[idx] = left_home[i]
        for i, idx in enumerate(self.right_arm_indices):
            qpos[idx] = right_home[i]

        scale = self.gripper_scale
        open_val = scale[0] + 1.0 * (scale[1] - scale[0])
        for idx, mult, offset in self.left_gripper_info:
            qpos[idx] = open_val * mult + offset
        for idx, mult, offset in self.right_gripper_info:
            qpos[idx] = open_val * mult + offset

        self.robot.set_qpos(qpos)

    def _load_cameras(self):
        sapien = self._sapien
        cfg = self.embodiment_config
        near, far = 0.1, 100

        for cam_info in cfg.get("static_camera_list", []):
            cam_type = cam_info.get("type", "D435")
            cam_cfg = self.camera_config_dict.get(
                cam_type, {"w": 320, "h": 240, "fovy": 37})

            cam_pos = np.array(cam_info["position"])
            cam_forward = np.array(
                cam_info.get("forward", (-cam_pos).tolist()))
            cam_forward = cam_forward / np.linalg.norm(cam_forward)
            cam_left = np.array(cam_info.get(
                "left", [-cam_forward[1], cam_forward[0], 0]))
            cam_left = cam_left / np.linalg.norm(cam_left)
            up = np.cross(cam_forward, cam_left)

            mat44 = np.eye(4)
            mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
            mat44[:3, 3] = cam_pos

            camera = self.scene.add_camera(
                name=cam_info["name"],
                width=cam_cfg["w"], height=cam_cfg["h"],
                fovy=np.deg2rad(cam_cfg["fovy"]),
                near=near, far=far,
            )
            camera.entity.set_pose(sapien.Pose(mat44))
            self.cameras[cam_info["name"]] = camera
            self.camera_names_ordered.append(cam_info["name"])

        wrist_cfg = self.camera_config_dict.get(
            "D435", {"w": 320, "h": 240, "fovy": 37})

        self.left_wrist_camera = self.scene.add_camera(
            name="left_camera", width=wrist_cfg["w"], height=wrist_cfg["h"],
            fovy=np.deg2rad(wrist_cfg["fovy"]), near=near, far=far,
        )
        self.right_wrist_camera = self.scene.add_camera(
            name="right_camera", width=wrist_cfg["w"], height=wrist_cfg["h"],
            fovy=np.deg2rad(wrist_cfg["fovy"]), near=near, far=far,
        )
        self.cameras["left_camera"] = self.left_wrist_camera
        self.cameras["right_camera"] = self.right_wrist_camera
        for name in ("left_camera", "right_camera"):
            if name not in self.camera_names_ordered:
                self.camera_names_ordered.append(name)

    def _gripper_val_to_qpos(self, val):
        val = np.clip(val, 0, 1)
        scale = self.gripper_scale
        return scale[0] + val * (scale[1] - scale[0])

    def set_pose_and_render(
        self, left_arm, right_arm, left_gripper, right_gripper,
        camera_name: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Set robot qpos and render.  Returns ``{camera_name: rgb_uint8}``."""
        qpos = self.robot.get_qpos().copy()

        for i, idx in enumerate(self.left_arm_indices):
            qpos[idx] = left_arm[i]
        for i, idx in enumerate(self.right_arm_indices):
            qpos[idx] = right_arm[i]

        left_g = self._gripper_val_to_qpos(left_gripper)
        for idx, mult, offset in self.left_gripper_info:
            qpos[idx] = left_g * mult + offset
        right_g = self._gripper_val_to_qpos(right_gripper)
        for idx, mult, offset in self.right_gripper_info:
            qpos[idx] = right_g * mult + offset

        self.robot.set_qpos(qpos)
        self.left_wrist_camera.entity.set_pose(
            self.left_camera_link.get_entity_pose())
        self.right_wrist_camera.entity.set_pose(
            self.right_camera_link.get_entity_pose())
        self.scene.update_render()

        render_cameras = (
            {camera_name: self.cameras[camera_name]}
            if camera_name and camera_name in self.cameras
            else self.cameras
        )

        rgb_dict = {}
        for name, camera in render_cameras.items():
            camera.take_picture()
            rgba = camera.get_picture("Color")
            rgb = (rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3]
            rgb_dict[name] = rgb
        return rgb_dict

    def close(self):
        if self.scene is not None:
            del self.scene
            self.scene = None
        if self.renderer is not None:
            del self.renderer
            self.renderer = None
        if self.engine is not None:
            del self.engine
            self.engine = None


class RobotOnlyRenderer:
    """High-level wrapper: lazy-inits a SAPIEN scene and renders episodes.

    Parameters
    ----------
    embodiment_dir : str
        Directory containing ``embodiments/``.
    variant : str
        Robot variant string, e.g. ``"aloha-agilex_clean_50"``.
    camera_config : dict, optional
        Override default camera config.
    render_resolution : tuple of (width, height), optional
        If set, override all camera widths/heights to render at this
        resolution while preserving the original fovy.
    """

    def __init__(
        self,
        embodiment_dir: str,
        variant: str = "aloha-agilex_clean_50",
        camera_config: Optional[Dict] = None,
        render_resolution: Optional[Tuple[int, int]] = None,
    ):
        self.embodiment_dir = embodiment_dir
        self.variant = variant
        self.render_resolution = render_resolution

        if render_resolution is not None and camera_config is None:
            rw, rh = render_resolution
            camera_config = {
                cam_type: {**cfg, "w": rw, "h": rh}
                for cam_type, cfg in DEFAULT_CAMERA_CONFIG.items()
            }

        self.camera_config = camera_config
        self._scene: Optional[RobotOnlyScene] = None

    def _ensure_scene(self):
        if self._scene is not None:
            return
        robot_prefix, _ = parse_variant(self.variant)
        embodiment_config, robot_dir = load_embodiment_config(
            self.embodiment_dir, robot_prefix)
        self._scene = RobotOnlyScene(
            embodiment_config, robot_dir, self.camera_config)
        self._scene.setup()
        res_str = f"{self.render_resolution[0]}x{self.render_resolution[1]}" if self.render_resolution else "native"
        print(f"[RobotOnlyRenderer] SAPIEN scene ready "
              f"(variant={self.variant}, render_resolution={res_str})")

    def render_episode(
        self,
        action_hdf5_path: str,
        indices: List[int],
        camera: str = "head_camera",
    ) -> List[np.ndarray]:
        """Render robot-only RGB at specific time indices.

        Parameters
        ----------
        action_hdf5_path : str
            Path to HDF5 with ``joint_action/{left_arm,right_arm,...}``.
        indices : list of int
            Frame indices to render (e.g. from ``_uniform_resample_indices``).
        camera : str
            Camera name to return frames for.

        Returns
        -------
        list of np.ndarray
            RGB uint8 frames at native camera resolution.
        """
        self._ensure_scene()
        scene = self._scene

        with h5py.File(action_hdf5_path, "r") as f:
            left_arm = f["joint_action/left_arm"][:]
            right_arm = f["joint_action/right_arm"][:]
            left_gripper = f["joint_action/left_gripper"][:]
            right_gripper = f["joint_action/right_gripper"][:]

        frames = []
        for t_idx in indices:
            rgb_dict = scene.set_pose_and_render(
                left_arm[t_idx], right_arm[t_idx],
                left_gripper[t_idx], right_gripper[t_idx],
                camera_name=camera,
            )
            frames.append(rgb_dict[camera])
        return frames

    def get_episode_length(self, action_hdf5_path: str) -> int:
        """Return the number of timesteps in an action HDF5 file."""
        with h5py.File(action_hdf5_path, "r") as f:
            return f["joint_action/left_arm"].shape[0]

    def close(self):
        if self._scene is not None:
            self._scene.close()
            self._scene = None
