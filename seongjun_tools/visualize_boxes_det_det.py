import argparse
import os
from typing import Any, Dict, Generic, List, Optional

import typing_extensions as _typing_ext
_typing_ext.Generic = Generic  # type: ignore[attr-defined]

import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes

# 공통 함수들을 visualize_boxes_det_trk에서 import
from visualize_boxes_det_trk import (
    TOPDOWN_FOV_DEG,
    add_lidar_geometry,
    collect_bounds_points,
    collect_ground_truth_boxes,
    collect_ordered_sample_tokens,
    compute_camera_target,
    create_box_geometry,
    create_front_sphere,
    determine_anchor_pose,
    ensure_output_dir,
    find_scene_by_name,
    get_lidar_top_points_in_global,
    load_results_map,
    set_camera_view,
)

DETECTION1_COLOR = (0.1, 0.45, 0.95)
DETECTION2_COLOR = (0.95, 0.2, 0.2)
GT_COLOR = (0.0, 0.0, 0.0)

def build_sample_geometries(
    detections_primary: List[Dict[str, Any]],
    detections_secondary: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    anchor_center: np.ndarray,
    anchor_rotation: Optional[np.ndarray],
    size_order: str,
    quat_order: str,
    add_axes: bool,
    axes_size: float,
) -> List[o3d.geometry.Geometry]:
    geometries: List[o3d.geometry.Geometry] = []
    for record in detections_primary:
        box = create_box_geometry(
            record,
            DETECTION1_COLOR,
            anchor_center,
            anchor_rotation,
            size_order,
            quat_order,
        )
        if box is not None:
            geometries.append(box)
            # 앞 방향에 구체 추가
            sphere = create_front_sphere(box, DETECTION1_COLOR)
            geometries.append(sphere)
        else:
            raise ValueError(f"Box not found for record: {record}")
    for record in detections_secondary:
        box = create_box_geometry(
            record,
            DETECTION2_COLOR,
            anchor_center,
            anchor_rotation,
            size_order,
            quat_order,
        )
        if box is not None:
            geometries.append(box)
            # 앞 방향에 구체 추가
            sphere = create_front_sphere(box, DETECTION2_COLOR)
            geometries.append(sphere)
        else:
            raise ValueError(f"Box not found for record: {record}")
    for record in ground_truths:
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="두 개의 nuScenes detection 결과를 동시에 렌더링하여 각 프레임 이미지를 생성합니다."
    )
    parser.add_argument("--detection_json1", type=str, required=True, help="Primary nuScenes-format detection JSON path.")
    parser.add_argument("--detection_json2", type=str, required=True, help="Secondary nuScenes-format detection JSON path.")
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

    print("Data Loading [1/6] Loading detection results...")
    detection_results_primary = load_results_map(args.detection_json1)
    detection_results_secondary = load_results_map(args.detection_json2)

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
    for offset, sample_index in enumerate(range(start_index, end_index + 1)):
        token = sample_tokens[sample_index]
        det_primary = detection_results_primary.get(token, [])
        det_secondary = detection_results_secondary.get(token, [])
        anchor_center, anchor_rotation = determine_anchor_pose(nusc, token)
        gt_boxes = collect_ground_truth_boxes(nusc, token)

        geometries = build_sample_geometries(
            detections_primary=det_primary,
            detections_secondary=det_secondary,
            ground_truths=gt_boxes,
            anchor_center=anchor_center,
            anchor_rotation=anchor_rotation,
            size_order=args.size_order,
            quat_order=args.quat_order,
            add_axes=args.show_axes,
            axes_size=args.axes_size,
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

