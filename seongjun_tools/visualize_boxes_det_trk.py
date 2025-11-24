import argparse
import json
import math
import os
from typing import Any, Dict, Generic, Iterable, List, Optional, Sequence, Tuple

# 일부 배포 환경에서는 typing_extensions 모듈에 Generic 심볼이 빠져 있어
# open3d → dash 임포트 과정에서 AttributeError가 발생한다.
# dash 는 typing_extensions.Generic 을 참조하므로, 누락된 경우 typing.Generic 로 보강한다.
import typing_extensions as _typing_ext
_typing_ext.Generic = Generic  # type: ignore[attr-defined]

import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


DETECTION_COLOR = (0.1, 0.45, 0.95)
TRACKING_COLOR = (0.95, 0.2, 0.2)
TRACK_ID_COLOR = (0.0, 0.0, 0.0)
GT_COLOR = (0.0, 0.0, 0.0)
LIDAR_COLOR = 0.65
TOPDOWN_FOV_DEG = 5.0

DIGIT_SEGMENT_COORDS = {
    "A": ((0.0, 2.0), (1.0, 2.0)),
    "B": ((1.0, 2.0), (1.0, 1.0)),
    "C": ((1.0, 1.0), (1.0, 0.0)),
    "D": ((0.0, 0.0), (1.0, 0.0)),
    "E": ((0.0, 0.0), (0.0, 1.0)),
    "F": ((0.0, 1.0), (0.0, 2.0)),
    "G": ((0.0, 1.0), (1.0, 1.0)),
}

DIGIT_SEGMENTS = {
    "0": ("A", "B", "C", "D", "E", "F"),
    "1": ("B", "C"),
    "2": ("A", "B", "G", "E", "D"),
    "3": ("A", "B", "G", "C", "D"),
    "4": ("F", "G", "B", "C"),
    "5": ("A", "F", "G", "C", "D"),
    "6": ("A", "F", "G", "E", "C", "D"),
    "7": ("A", "B", "C"),
    "8": ("A", "B", "C", "D", "E", "F", "G"),
    "9": ("A", "B", "C", "D", "F", "G"),
}

DIGIT_WIDTH = 1.0
DIGIT_SPACING = 0.35
TRACK_LABEL_HEIGHT_RATIO = 0.55
TRACK_LABEL_MIN_HEIGHT = 0.8
TRACK_ID_LABEL_MAX_CHARS = 6
TRACK_ID_LABEL_SCALE_RATIO = 0.65
TRACK_ID_LABEL_OFFSET_RATIO = 2.0


def color_from_tracking_id(tracking_id: Any) -> Tuple[float, float, float]:
    integer_hash = abs(hash(str(tracking_id)))
    hue = (integer_hash % 360) / 360.0
    saturation = 0.75
    value = 0.95
    i = int(hue * 6)
    f = hue * 6 - i
    p = value * (1 - saturation)
    q = value * (1 - f * saturation)
    t = value * (1 - (1 - f) * saturation)
    i = i % 6
    if i == 0:
        r, g, b = value, t, p
    elif i == 1:
        r, g, b = q, value, p
    elif i == 2:
        r, g, b = p, value, t
    elif i == 3:
        r, g, b = p, q, value
    elif i == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q
    return (float(r), float(g), float(b))


def compute_track_label_scale(box: o3d.geometry.OrientedBoundingBox) -> float:
    """
    숫자 라벨의 실제 높이를 박스 크기에 비례하게 조절하기 위한 스케일을 계산한다.
    """
    extents = np.asarray(box.extent, dtype=np.float64)
    planar_extent = float(np.min(extents[:2])) if extents.size >= 2 else float(extents[0])
    planar_extent = max(planar_extent, 1e-3)
    target_height = max(planar_extent * TRACK_LABEL_HEIGHT_RATIO, TRACK_LABEL_MIN_HEIGHT)
    return target_height * 0.5  # create_track_index_label 은 2 * scale 높이를 생성한다.


def create_track_index_label(
    text: str,
    center: np.ndarray,
    planar_scale: float,
    color: Tuple[float, float, float],
) -> Optional[o3d.geometry.LineSet]:
    digits = "".join(ch for ch in str(text) if ch in DIGIT_SEGMENTS)
    if not digits or planar_scale <= 0.0:
        return None

    points: List[np.ndarray] = []
    lines: List[List[int]] = []
    point_offset = 0

    for idx, digit in enumerate(digits):
        segment_keys = DIGIT_SEGMENTS.get(digit, ())
        if not segment_keys:
            continue
        x_offset = idx * (DIGIT_WIDTH + DIGIT_SPACING)
        for segment_key in segment_keys:
            start_xy, end_xy = DIGIT_SEGMENT_COORDS[segment_key]
            start = np.array([start_xy[0] + x_offset, start_xy[1], 0.0], dtype=np.float64)
            end = np.array([end_xy[0] + x_offset, end_xy[1], 0.0], dtype=np.float64)
            points.append(start)
            points.append(end)
            lines.append([point_offset, point_offset + 1])
            point_offset += 2

    if not points or not lines:
        return None

    points_np = np.vstack(points)
    points_np *= float(planar_scale)

    min_xy = points_np[:, :2].min(axis=0)
    max_xy = points_np[:, :2].max(axis=0)
    text_center_xy = (min_xy + max_xy) * 0.5

    points_np[:, 0] = points_np[:, 0] - text_center_xy[0] + float(center[0])
    points_np[:, 1] = points_np[:, 1] - text_center_xy[1] + float(center[1])
    points_np[:, 2] = float(center[2])

    label = o3d.geometry.LineSet()
    label.points = o3d.utility.Vector3dVector(points_np)
    label.lines = o3d.utility.Vector2iVector(lines)
    colors = np.tile(np.array(color, dtype=np.float64), (len(lines), 1))
    label.colors = o3d.utility.Vector3dVector(colors)
    return label


def find_scene_by_name(nusc: NuScenes, scene_name: str) -> Dict[str, Any]:
    for scene_record in nusc.scene:
        if scene_record.get("name") == scene_name:
            return scene_record
    raise ValueError(f"Scene with name '{scene_name}' not found.")


def collect_ordered_sample_tokens(nusc: NuScenes, scene_record: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    current_token: Optional[str] = scene_record["first_sample_token"]
    while current_token:
        tokens.append(current_token)
        sample = nusc.get("sample", current_token)
        current_token = sample["next"]
    return tokens


def normalize_quaternion(w: float, x: float, y: float, z: float) -> Tuple[float, float, float, float]:
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return w / norm, x / norm, y / norm, z / norm


def rotation_matrix_from_wxyz_quaternion(w: float, x: float, y: float, z: float) -> np.ndarray:
    w, x, y, z = normalize_quaternion(w, x, y, z)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_results_map(json_path: str) -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    with open(json_path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "results" in payload:
        data = payload["results"]
    else:
        data = payload
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected format in {json_path}. Expected dict keyed by sample_token.")
    results_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for key, detections in data.items():
        if detections is None:
            continue
        if not isinstance(detections, list):
            raise ValueError(f"Expected list for sample '{key}', got {type(detections).__name__}")
        results_by_sample[str(key)] = detections
    return results_by_sample


def choose_quaternion(rotation: Sequence[float], order: str) -> Tuple[float, float, float, float]:
    if len(rotation) != 4:
        raise ValueError(f"Rotation must have 4 values, got {len(rotation)}")
    if order == "wxyz":
        return float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])
    if order == "xyzw":
        return float(rotation[3]), float(rotation[0]), float(rotation[1]), float(rotation[2])
    if order == "auto":
        cand_wxyz = (float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]))
        cand_xyzw = (float(rotation[3]), float(rotation[0]), float(rotation[1]), float(rotation[2]))
        tilt_wxyz = cand_wxyz[1] * cand_wxyz[1] + cand_wxyz[2] * cand_wxyz[2]
        tilt_xyzw = cand_xyzw[1] * cand_xyzw[1] + cand_xyzw[2] * cand_xyzw[2]
        return cand_wxyz if tilt_wxyz <= tilt_xyzw else cand_xyzw
    raise ValueError(f"Unsupported quaternion order '{order}'")


def extents_from_size(size: Sequence[float], order: str) -> np.ndarray:
    if len(size) != 3:
        raise ValueError(f"Size must have 3 values, got {len(size)}")
    if order == "wlh":
        width, length, height = float(size[0]), float(size[1]), float(size[2])
    elif order == "lwh":
        length, width, height = float(size[0]), float(size[1]), float(size[2])
    else:
        raise ValueError(f"Unsupported size order '{order}'")
    return np.array([length, width, height], dtype=np.float64)


def create_box_geometry(
    record: Dict[str, Any],
    color: Tuple[float, float, float],
    anchor_center: np.ndarray,
    anchor_rotation: Optional[np.ndarray],
    size_order: str,
    quat_order: str,
) -> Optional[o3d.geometry.OrientedBoundingBox]:
    translation = record.get("translation")
    size = record.get("size")
    rotation = record.get("rotation")
    if rotation is None:
        # rotation = record.get("rotation_det")
        rotation = record.get("rotation_kalman_cv")
    if translation is None or size is None or rotation is None:
        return None
    center = np.array(translation, dtype=np.float64) - anchor_center
    if anchor_rotation is not None:
        center = anchor_rotation.T @ center
    extents = extents_from_size(size, size_order)
    w, x, y, z = choose_quaternion(rotation, quat_order)
    rot_matrix = rotation_matrix_from_wxyz_quaternion(w, x, y, z)
    if anchor_rotation is not None:
        rot_matrix = anchor_rotation.T @ rot_matrix
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=rot_matrix, extent=extents)
    obb.color = color
    return obb


def create_front_sphere(
    box: o3d.geometry.OrientedBoundingBox,
    color: Tuple[float, float, float],
    sphere_radius_ratio: float = 0.15,
) -> o3d.geometry.TriangleMesh:
    """
    박스의 앞 방향에 구체를 생성합니다.
    
    Args:
        box: OrientedBoundingBox 객체
        color: 구체의 색상 (RGB, 0-1 범위)
        sphere_radius_ratio: 구체 반지름을 박스의 최소 크기에 대한 비율 (기본값: 0.15)
    
    Returns:
        구체 메시 객체
    """
    center = np.asarray(box.center, dtype=np.float64)
    rot_matrix = np.asarray(box.R, dtype=np.float64)
    extents = np.asarray(box.extent, dtype=np.float64)
    
    # extents는 [length, width, height] 순서
    # 회전 행렬의 첫 번째 열이 length 방향 (앞뒤 방향)
    forward_direction = rot_matrix[:, 0]
    length = extents[0]
    
    # 박스 중심에서 앞 방향으로 length/2 만큼 이동한 위치
    front_position = center + forward_direction * (length / 2.0)
    
    # 구체 반지름: 박스의 최소 크기의 일정 비율
    min_extent = float(np.min(extents))
    sphere_radius = min_extent * sphere_radius_ratio
    
    # 구체 생성
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere.translate(front_position)
    sphere.paint_uniform_color(color)
    
    return sphere


def determine_anchor_pose(
    nusc: NuScenes,
    sample_token: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"].get("LIDAR_TOP")
    if lidar_token:
        sd_rec = nusc.get("sample_data", lidar_token)
        ego_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        center = np.array(ego_rec["translation"], dtype=np.float64)
        rotation = Quaternion(ego_rec["rotation"]).rotation_matrix
        return center, rotation
    else:
        raise ValueError(f"Lidar token not found for sample {sample_token}")


def collect_ground_truth_boxes(nusc: NuScenes, sample_token: str) -> List[Dict[str, Any]]:
    sample = nusc.get("sample", sample_token)
    boxes: List[Dict[str, Any]] = []
    for ann_token in sample.get("anns", []):
        ann = nusc.get("sample_annotation", ann_token)
        boxes.append(
            {
                "translation": ann["translation"],
                "size": ann["size"],
                "rotation": ann["rotation"],
            }
        )
    return boxes


def get_lidar_top_points_in_global(nusc: NuScenes, sample_token: str) -> Optional[np.ndarray]:
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"].get("LIDAR_TOP")
    if lidar_token is None:
        return None
    sd_rec = nusc.get("sample_data", lidar_token)
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    ego_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])

    file_path = os.path.join(nusc.dataroot, sd_rec["filename"])
    pc = LidarPointCloud.from_file(file_path)

    sensor_to_ego = np.eye(4, dtype=np.float64)
    sensor_to_ego[:3, :3] = Quaternion(cs_rec["rotation"]).rotation_matrix
    sensor_to_ego[:3, 3] = np.array(cs_rec["translation"], dtype=np.float64)

    ego_to_global = np.eye(4, dtype=np.float64)
    ego_to_global[:3, :3] = Quaternion(ego_rec["rotation"]).rotation_matrix
    ego_to_global[:3, 3] = np.array(ego_rec["translation"], dtype=np.float64)

    transform = ego_to_global @ sensor_to_ego
    pc.transform(transform)
    return pc.points.T[:, :3]


def collect_bounds_points(geometries: Iterable[o3d.geometry.Geometry]) -> Optional[np.ndarray]:
    points: List[np.ndarray] = []
    for geom in geometries:
        if isinstance(geom, o3d.geometry.OrientedBoundingBox):
            pts = np.asarray(geom.get_box_points())
        elif isinstance(geom, o3d.geometry.PointCloud):
            pts = np.asarray(geom.points)
        elif isinstance(geom, o3d.geometry.LineSet):
            pts = np.asarray(geom.points)
        elif isinstance(geom, o3d.geometry.TriangleMesh):
            pts = np.asarray(geom.vertices)
        else:
            pts = np.empty((0, 3), dtype=np.float64)
        if pts.size > 0:
            points.append(pts)
    if not points:
        return None
    return np.vstack(points)


def compute_camera_target(points: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
    if points is None or len(points) == 0:
        return np.zeros(3, dtype=np.float64), 15.0
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    center = (min_corner + max_corner) * 0.5
    extent = max_corner - min_corner
    radius = max(float(np.linalg.norm(extent)), 1.0)
    return center, radius


def set_camera_view(
    view_control: Optional[o3d.visualization.ViewControl],
    center: np.ndarray,
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
    distance_multiplier: float,
    view_mode: str,
) -> None:
    if view_control is None:
        return

    view_mode = view_mode.lower()
    if view_mode == "topdown":
        view_control.set_front([0.0, 0.0, 1.0])
        view_control.set_up([0.0, 1.0, 0.0])
        view_control.set_lookat(center.tolist())
        current_fov = float(view_control.get_field_of_view())
        target_fov = float(TOPDOWN_FOV_DEG)
        delta = target_fov - current_fov
        if abs(delta) > 1e-3:
            view_control.change_field_of_view(step=delta)
        view_control.set_zoom(0.15)
        return

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    direction = np.array(
        [
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ],
        dtype=np.float64,
    )
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    distance = max(radius * distance_multiplier, 1.0)
    eye = center + direction * distance

    look_at_fn = getattr(view_control, "look_at", None)
    if callable(look_at_fn):
        look_at_fn(center.tolist(), eye.tolist(), up.tolist())
        return

    _apply_view_control_fallback(
        view_control=view_control,
        center=center,
        eye=eye,
        up=up,
        radius=radius,
        distance=distance,
    )


def _compute_look_at_extrinsic(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = center - eye
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        forward_norm = 1.0
    forward /= forward_norm

    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up_norm = 1.0
    up = up / up_norm

    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        # forward 와 up 이 평행할 때를 대비해 직교 벡터를 재구성한다.
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up)
        right_norm = float(np.linalg.norm(right))
    right /= max(right_norm, 1e-6)

    true_up = np.cross(right, forward)

    rotation = np.stack([right, true_up, -forward], axis=0)
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye
    return extrinsic


def _apply_view_control_fallback(
    view_control: o3d.visualization.ViewControl,
    center: np.ndarray,
    eye: np.ndarray,
    up: np.ndarray,
    radius: float,
    distance: float,
) -> None:
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = _compute_look_at_extrinsic(eye, center, up)
    view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def build_sample_geometries(
    detections: List[Dict[str, Any]],
    trackings: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    anchor_center: np.ndarray,
    anchor_rotation: Optional[np.ndarray],
    size_order: str,
    quat_order: str,
    add_axes: bool,
    axes_size: float,
    track_order_registry: Optional[Dict[str, int]] = None,
) -> List[o3d.geometry.Geometry]:
    geometries: List[o3d.geometry.Geometry] = []
    for record in detections:
        box = create_box_geometry(record, DETECTION_COLOR, anchor_center, anchor_rotation, size_order, quat_order)
        if box is not None:
            geometries.append(box)
            # 앞 방향에 구체 추가
            sphere = create_front_sphere(box, DETECTION_COLOR)
            geometries.append(sphere)
        else:
            raise ValueError(f"Box not found for record: {record}")
    for record in trackings:
        tracking_id = record.get("tracking_id")
        if tracking_id is None:
            raise ValueError(f"Tracking ID not found for record: {record}")
        tracking_key = str(tracking_id)
        track_order_index: Optional[int] = None
        if track_order_registry is not None:
            track_order_index = track_order_registry.get(tracking_key, 0) + 1
            track_order_registry[tracking_key] = track_order_index
        record["translation"][2] += 2.0
        box = create_box_geometry(record, TRACKING_COLOR, anchor_center, anchor_rotation, size_order, quat_order)
        if box is not None:
            geometries.append(box)
            # 앞 방향에 구체 추가
            sphere = create_front_sphere(box, TRACKING_COLOR)
            geometries.append(sphere)
            box_center = np.asarray(box.center, dtype=np.float64)
            label_scale = compute_track_label_scale(box)
            if track_order_index is not None:
                order_label = create_track_index_label(
                    text=str(track_order_index),
                    center=box_center,
                    planar_scale=label_scale,
                    color=TRACKING_COLOR,
                )
                if order_label is not None:
                    geometries.append(order_label)

            track_id_text = str(tracking_id)
            if TRACK_ID_LABEL_MAX_CHARS > 0:
                track_id_text = track_id_text[:TRACK_ID_LABEL_MAX_CHARS]
            track_id_center = np.array(box_center, copy=True)
            track_id_center[1] -= label_scale * TRACK_ID_LABEL_OFFSET_RATIO
            track_id_scale = label_scale * TRACK_ID_LABEL_SCALE_RATIO
            track_id_label = create_track_index_label(
                text=track_id_text,
                center=track_id_center,
                planar_scale=track_id_scale,
                color=TRACK_ID_COLOR,
            )
            if track_id_label is None:
                fallback_text = str(abs(hash(tracking_id)) % 10000)
                track_id_label = create_track_index_label(
                    text=fallback_text,
                    center=track_id_center,
                    planar_scale=track_id_scale,
                    color=TRACK_ID_COLOR,
                )
            if track_id_label is not None:
                geometries.append(track_id_label)
        else:
            raise ValueError(f"Box not found for record: {record}")
    for record in ground_truths:
        record["translation"][2] += 1.0
        box = create_box_geometry(record, GT_COLOR, anchor_center, anchor_rotation, "wlh", "wxyz")
        if box is not None:
            geometries.append(box)
            # 앞 방향에 구체 추가
            sphere = create_front_sphere(box, GT_COLOR)
            geometries.append(sphere)
        else:
            raise ValueError(f"Box not found for record: {record}")
    if add_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axes_size))
        geometries.append(axes)
    return geometries


def add_lidar_geometry(
    geometries: List[o3d.geometry.Geometry],
    points_global: Optional[np.ndarray],
    anchor_center: np.ndarray,
    anchor_rotation: Optional[np.ndarray],
    voxel_size: float,
) -> None:
    if points_global is None or points_global.size == 0:
        return
    centered = points_global - anchor_center.reshape(1, 3)
    if anchor_rotation is not None:
        points_local = (anchor_rotation.T @ centered.T).T
    else:
        points_local = centered
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_local)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    if len(pcd.points) == 0:
        return
    colors = np.full((len(pcd.points), 3), LIDAR_COLOR, dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render nuScenes detection & tracking boxes for each frame into images."
    )
    parser.add_argument("--detection_json", type=str, required=True, help="nuScenes-format detection JSON path.")
    parser.add_argument("--tracking_json", type=str, required=True, help="nuScenes-format tracking JSON path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store rendered images.")
    parser.add_argument("--scene_name", type=str, required=True, help="Scene name to visualize.")

# val = \
#     ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
#      'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
#      'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
#      'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
#      'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
#      'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
#      'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
#      'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
#      'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
#      'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
#      'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
#      'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
#      'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
#      'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
#      'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
#      'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
#      'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
#      'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
#      'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']

    parser.add_argument("--nusc_root", type=str, default="/3dmot_ws/MCTrack/data/nuscenes/datasets", help="nuScenes dataset root.")
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="nuScenes version (default: v1.0-trainval).")
    parser.add_argument("--start_index", type=int, default=0, help="Start sample index (inclusive).")
    parser.add_argument("--end_index", type=int, default=-1, help="End sample index (inclusive, -1 for last).")
    parser.add_argument("--quat_order", type=str, default="auto", choices=["auto", "wxyz", "xyzw"], help="Quaternion order.")
    parser.add_argument("--size_order", type=str, default="wlh", choices=["wlh", "lwh"], help="Size ordering in JSON.")
    parser.add_argument("--window_width", type=int, default=1280, help="Visualizer window width.")
    parser.add_argument("--window_height", type=int, default=720, help="Visualizer window height.")
    parser.add_argument("--line_width", type=float, default=3.0, help="Line width for boxes.")
    parser.add_argument("--point_size", type=float, default=1.5, help="Point size for lidar points.")
    parser.add_argument("--show_lidar", type=bool, default=True, help="Overlay LIDAR_TOP points.")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="Voxel size when downsampling lidar points.")
    parser.add_argument("--azimuth_deg", type=float, default=-120.0, help="Camera azimuth in degrees.")
    parser.add_argument("--elevation_deg", type=float, default=35.0, help="Camera elevation in degrees.")
    parser.add_argument(
        "--distance_multiplier",
        type=float,
        default=2.2,
        help="Camera distance multiplier applied to scene radius.",
    )
    parser.add_argument(
        "--view_mode",
        type=str,
        default="topdown",
        choices=["topdown", "isometric"],
        help="Camera viewpoint. topdown은 버즈아이, isometric은 입력한 방위각/고도각 사용.",
    )
    parser.add_argument("--axes_size", type=float, default=3.0, help="Coordinate axes size.")
    parser.add_argument("--show_axes", action="store_true", help="Show coordinate axes at origin.")
    parser.add_argument(
        "--background_color",
        type=float,
        nargs=3,
        default=(0.95, 0.95, 0.95),
        metavar=("R", "G", "B"),
        help="Background color (0-1 range).",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip rendering if image already exists.")
    parser.add_argument("--image_format", type=str, default="png", help="Output image extension (e.g., png, jpg).")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    print("Data Loading [1/6] Loading detection and tracking results...")
    detection_results = load_results_map(args.detection_json)
    tracking_results = load_results_map(args.tracking_json)

    ensure_output_dir(args.output_dir)
    save_dir = os.path.join(args.output_dir, args.scene_name)
    ensure_output_dir(save_dir)

    print("Data Loading [2/6] Loading nuScenes dataset...")
    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)

    print("Data Loading [3/6] Finding scene record...")
    scene_record = find_scene_by_name(nusc, args.scene_name)

    print("Data Loading [4/6] Collecting sample tokens...")
    sample_tokens = collect_ordered_sample_tokens(nusc, scene_record)
    if not sample_tokens:
        raise RuntimeError(f"No samples found for scene '{args.scene_name}'.")

    start_index = max(0, int(args.start_index))
    end_index = int(args.end_index)
    if end_index < 0 or end_index >= len(sample_tokens):
        end_index = len(sample_tokens) - 1
    if start_index > end_index:
        raise ValueError("start_index must be <= end_index.")

    print("Data Loading [5/6] Creating visualizer window...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"nuScenes Visualization - {args.scene_name}",
        width=int(args.window_width),
        height=int(args.window_height),
        visible=False,
    )

    print("Data Loading [6/6] Setting render options...")
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.line_width = float(args.line_width)
        render_option.point_size = float(args.point_size)
        render_option.background_color = np.array(args.background_color, dtype=np.float32)

    print("Rendering Start")
    total_frames = end_index - start_index + 1
    track_order_registry: Dict[str, int] = {}
    for offset, sample_index in enumerate(range(start_index, end_index + 1)):
        token = sample_tokens[sample_index]
        dets = detection_results.get(token, [])
        trks = tracking_results.get(token, [])
        anchor_center, anchor_rotation = determine_anchor_pose(nusc, token)
        gt_boxes = collect_ground_truth_boxes(nusc, token)

        geometries = build_sample_geometries(
            detections=dets,
            trackings=trks,
            ground_truths=gt_boxes,
            anchor_center=anchor_center,
            anchor_rotation=anchor_rotation,
            size_order=args.size_order,
            quat_order=args.quat_order,
            add_axes=args.show_axes,
            axes_size=args.axes_size,
            track_order_registry=track_order_registry,
        )

        if args.show_lidar:
            lidar_points = get_lidar_top_points_in_global(nusc, token)
            add_lidar_geometry(
                geometries,
                lidar_points,
                anchor_center,
                anchor_rotation,
                voxel_size=float(args.voxel_size),
            )

        if not geometries:
            raise ValueError(f"No geometries found for sample {token}")

        # Ensure there is at least an empty axes frame to avoid blank captures.
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.axes_size))
        geometries.append(axes)

        vis.clear_geometries()
        for geom in geometries:
            vis.add_geometry(geom, reset_bounding_box=True)

        bounds_points = collect_bounds_points(geometries)
        _, radius = compute_camera_target(bounds_points)
        center = np.zeros(3, dtype=np.float64)
        view_control = vis.get_view_control()
        if view_control is not None:
            set_camera_view(
                view_control=view_control,
                center=center,
                radius=radius,
                azimuth_deg=args.azimuth_deg,
                elevation_deg=args.elevation_deg,
                distance_multiplier=args.distance_multiplier,
                view_mode=args.view_mode,
            )

        vis.poll_events()
        vis.update_renderer()

        filename = f"{sample_index:04d}_{token}.{args.image_format}"
        output_path = os.path.join(save_dir, filename)
        if args.skip_existing and os.path.isfile(output_path):
            print(f"[SKIP] {filename} already exists.")
        else:
            vis.capture_screen_image(output_path, do_render=True)
            print(f"[{offset + 1:04d}/{total_frames:04d}] Saved {output_path}")

    vis.destroy_window()


if __name__ == "__main__":
    main()

