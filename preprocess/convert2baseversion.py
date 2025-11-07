# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse

from convert_kitti import kitti_main
from convert_nuscenes import nuscenes_main
# from convert_waymo import waymo_main

# kitti_cfg = {
#     "raw_data_path": "data/kitti/datasets",
#     "dets_path": "data/kitti/detectors/",
#     "save_path": "data/base_version/kitti/",
#     "detector": "virconv",  # virconv / casa / ... /
#     "split": "test",  # val / test
# }

# nuscenes_cfg = {
#     "raw_data_path": "data/nuscenes/datasets/",
#     "dets_path": "data/nuscenes/detectors/",
#     "save_path": "data/base_version/nuscenes/",
#     "detector": "centerpoint",  #  centerpoint(val) / largekernel(test) / ....
#     "split": "val",  # val / test
# }

# waymo_cfg = {
#     "raw_data_path": "data/waymo/datasets/",
#     "dets_path": "data/waymo/detectors/",
#     "save_path": "data/base_version/waymo/",
#     "detector": "ctrl",
#     "split": "val",  # val / test
# }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="kitti", help="kitti/nuscenes/waymo"
    )
    # NuScenes 전용 인자들 (기본값은 상단 nuscenes_cfg 사용)
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/nuscenes/datasets",
        help="NuScenes raw data path",
    )
    parser.add_argument(
        "--dets_file_path",
        type=str,
        required=True,
        help="NuScenes detection file path",
    )
    parser.add_argument(
        "--save_folder_path",
        type=str,
        required=True,
        help="NuScenes converted base version save folder path",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["val", "test","train"],
        help="NuScenes split (val/test/train)",
    )
    args = parser.parse_args()

    # if args.dataset == "kitti":
    #     kitti_main(
    #         args.raw_data_path,
    #         args.dets_path,
    #         args.detector,
    #         args.save_path,
    #         args.split,
    #     )
    # elif args.dataset == "nuscenes":
    nuscenes_main(
        args.raw_data_path,
        args.dets_file_path,
        args.save_folder_path,
        args.split,
    )
    # elif args.dataset == "waymo":
    #     waymo_main(
    #         waymo_cfg["raw_data_path"],
    #         waymo_cfg["dets_path"],
    #         waymo_cfg["detector"],
    #         waymo_cfg["save_path"],
    #         waymo_cfg["split"],
    #     )
