import json
import os
import copy
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp

from nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.config import config_factory as track_configs
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingConfig, TrackingMetrics, TrackingMetricDataList, TrackingMetricData
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP


_GLOBAL_tracks_gt = None
_GLOBAL_tracks_pred_variants = None
_GLOBAL_cfg = None

# Class-level parallel globals (used inside variant worker)
_CLS_tracks_gt = None
_CLS_tracks_pred = None
_CLS_cfg = None


def _compute_overall_metrics_for_variant_mp(variant_name: str,
                                            tracks_gt,
                                            tracks_pred,
                                            cfg,
                                            disable_class_tqdm: bool = True):
    metrics = TrackingMetrics(cfg)
    metric_data_list = TrackingMetricDataList()

    class_names = cfg.class_names if hasattr(cfg, 'class_names') else cfg.tracking_names

    if disable_class_tqdm:
        # 병렬: 클래스 단위 포크 풀 (워커에서는 진행률 숨김)
        global _CLS_tracks_gt, _CLS_tracks_pred, _CLS_cfg
        _CLS_tracks_gt = tracks_gt
        _CLS_tracks_pred = tracks_pred
        _CLS_cfg = cfg
        with mp.get_context('fork').Pool(processes=min(len(class_names), os.cpu_count() or 1)) as pool:
            for cname, md in pool.imap_unordered(_compute_class_md_entry, class_names):
                metric_data_list.set(cname, md)
    else:
        # 직렬: 진행률 표시
        for class_name in tqdm(class_names, disable=False, desc=f"{variant_name} classes"):
            curr_ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                         cfg.dist_th_tp, cfg.min_recall,
                                         num_thresholds=TrackingMetricData.nelem,
                                         metric_worst=cfg.metric_worst,
                                         verbose=False,
                                         output_dir=None,
                                         render_classes=None)
            curr_md = curr_ev.accumulate()
            metric_data_list.set(class_name, curr_md)

    for class_name in class_names:
        md = metric_data_list[class_name]
        if np.all(np.isnan(md.mota)):
            best_thresh_idx = None
        else:
            best_thresh_idx = np.nanargmax(md.mota)

        if best_thresh_idx is not None:
            for metric_name in MOT_METRIC_MAP.values():
                if metric_name == '':
                    continue
                value = md.get_metric(metric_name)[best_thresh_idx]
                metrics.add_label_metric(metric_name, class_name, value)

        for metric_name in AVG_METRIC_MAP.keys():
            values = np.array(md.get_metric(AVG_METRIC_MAP[metric_name]))
            assert len(values) == TrackingMetricData.nelem
            if np.all(np.isnan(values)):
                value = np.nan
            else:
                np.all(values[np.logical_not(np.isnan(values))] >= 0)
                values[np.isnan(values)] = cfg.metric_worst[metric_name]
                value = float(np.nanmean(values))
            metrics.add_label_metric(metric_name, class_name, value)

    return variant_name, metrics.serialize()


def _compute_class_md_entry(class_name: str):
    """Compute TrackingMetricData for one class (used in class-level Pool)."""
    tracks_gt = _CLS_tracks_gt
    tracks_pred = _CLS_tracks_pred
    cfg = _CLS_cfg
    curr_ev = TrackingEvaluation(tracks_gt, tracks_pred, class_name, cfg.dist_fcn_callable,
                                 cfg.dist_th_tp, cfg.min_recall,
                                 num_thresholds=TrackingMetricData.nelem,
                                 metric_worst=cfg.metric_worst,
                                 verbose=False,
                                 output_dir=None,
                                 render_classes=None)
    curr_md = curr_ev.accumulate()
    return class_name, curr_md


def _compute_overall_metrics_for_variant_entry(args):
    """Multiprocessing entry point for overall variant metrics (avoids lambda)."""
    return _compute_overall_metrics_for_variant_mp(*args[:-1], disable_class_tqdm=args[-1])


def _compute_overall_metrics_for_variant_by_name(variant_name: str):
    """Fork-safe worker that only receives a picklable name and reads globals."""
    tracks_gt = _GLOBAL_tracks_gt
    tracks_pred = _GLOBAL_tracks_pred_variants[variant_name]
    cfg = _GLOBAL_cfg
    _, metrics_summary = _compute_overall_metrics_for_variant_mp(
        variant_name, tracks_gt, tracks_pred, cfg, disable_class_tqdm=True)
    return variant_name, metrics_summary


def _evaluate_scene_mp(args):
    """Multiprocessing entry point: evaluate a single scene across variants."""
    scene_id, scene_tracks_gt, scene_tracks_pred, cfg_local = args
    scene_result = {}
    # GT track ID 수집
    gt_track_ids = set()
    for frame_boxes in scene_tracks_gt.values():
        for box in frame_boxes:
            gt_track_ids.add((box.tracking_id, box.tracking_name))
    # 각 GT track에 대해 평가
    for track_id, class_name in gt_track_ids:
        track_key = f"{scene_id}_{track_id}_{class_name}"
        # GT track 수집
        track_gt = []
        for frame_boxes in scene_tracks_gt.values():
            for box in frame_boxes:
                if box.tracking_id == track_id and box.tracking_name == class_name:
                    track_gt.append(box)
        if len(track_gt) == 0:
            continue
        track_metrics = {
            'scene_id': scene_id,
            'track_id': track_id,
            'class_name': class_name,
            'track_length_gt': len(track_gt),
            'variants': {}
        }
        for variant_name in ['det', 'kalman_cv', 'diff', 'curve']:
            if variant_name not in scene_tracks_pred:
                continue
            # Prediction track 찾기
            track_pred = []
            pred_track_ids = set()
            for timestamp in scene_tracks_pred[variant_name].keys():
                for box in scene_tracks_pred[variant_name][timestamp]:
                    if box.tracking_name == class_name:
                        pred_track_ids.add(box.tracking_id)
            best_match_id = None
            best_overlap = 0
            for pred_track_id in pred_track_ids:
                pred_track = []
                for timestamp in scene_tracks_pred[variant_name].keys():
                    for box in scene_tracks_pred[variant_name][timestamp]:
                        if box.tracking_id == pred_track_id and box.tracking_name == class_name:
                            pred_track.append(box)
                gt_samples = {box.sample_token for box in track_gt}
                pred_samples = {box.sample_token for box in pred_track}
                overlap = len(gt_samples & pred_samples)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match_id = pred_track_id
            if best_match_id is not None:
                for timestamp in scene_tracks_pred[variant_name].keys():
                    for box in scene_tracks_pred[variant_name][timestamp]:
                        if box.tracking_id == best_match_id and box.tracking_name == class_name:
                            track_pred.append(box)
            dist_th_tp = cfg_local.dist_th_tp if hasattr(cfg_local, 'dist_th_tp') else 2.0
            variant_metrics = compute_track_metrics(
                track_gt, track_pred, variant_name, dist_th_tp=dist_th_tp
            )
            track_metrics['variants'][variant_name] = variant_metrics
        scene_result[track_key] = track_metrics
    return scene_id, scene_result


def quaternion_to_yaw(q):
    """쿼터니언을 yaw 각도로 변환"""
    qw, qx, qy, qz = q
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def quaternion_distance(q1, q2):
    """두 쿼터니언 간의 각도 차이 계산 (라디안)"""
    q1 = np.array(q1)
    q2 = np.array(q2)
    # 쿼터니언 내적
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    # 각도 차이 (라디안)
    angle = 2 * np.arccos(dot)
    return angle


def velocity_distance(v1, v2):
    """두 속도 벡터 간의 유클리드 거리"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.linalg.norm(v1 - v2)


def process_data(data_variant, rotation_field, velocity_field):
    """데이터 변형: 특정 rotation과 velocity 필드를 표준 필드로 변환"""
    for sample_token in data_variant['results'].keys():
        detections = data_variant['results'][sample_token]
        for detection in detections:
            if rotation_field in detection:
                detection['rotation'] = detection[rotation_field]
            if velocity_field in detection:
                detection['velocity'] = detection[velocity_field]
            
            # 다른 rotation/velocity 필드 제거
            keys_to_remove = [
                'rotation_det', 'rotation_kalman_cv',
                'velocity_det', 'velocity_kalman_cv',
                'velocity_diff', 'velocity_curve'
            ]
            for key in keys_to_remove:
                if key in detection:
                    del detection[key]


def load_prediction_variants(result_path: str, max_boxes_per_sample: int, box_cls, cfg: TrackingConfig = None, verbose: bool = False):
    """
    결과 파일을 로드하고 여러 variant로 변환
    
    Args:
        result_path: 결과 파일 경로
        max_boxes_per_sample: 샘플당 최대 박스 수
        box_cls: 박스 클래스
        cfg: TrackingConfig (TRACKING_NAMES를 설정하기 위해 필요)
        verbose: 상세 출력 여부
    """
    # Config가 제공되면 TRACKING_NAMES를 먼저 설정
    if cfg is not None:
        # TrackingConfig가 초기화되면 자동으로 TRACKING_NAMES가 설정됨
        pass
    
    with open(result_path) as f:
        data = json.load(f)
    
    variants = {}
    variant_configs = [
        ('rotation_det', 'velocity_det', 'det'),
        ('rotation_kalman_cv', 'velocity_kalman_cv', 'kalman_cv'),
        ('rotation_det', 'velocity_diff', 'diff'),
        ('rotation_det', 'velocity_curve', 'curve'),
    ]
    
    for rot_field, vel_field, variant_name in tqdm(variant_configs, disable=False, desc="Deserialize variants"):
        data_variant = copy.deepcopy(data)
        process_data(data_variant, rot_field, vel_field)
        all_results = EvalBoxes.deserialize(data_variant['results'], box_cls)
        variants[variant_name] = all_results
    
    meta = data['meta']
    return variants, meta


def compute_track_metrics(track_gt: List[TrackingBox], track_pred: List[TrackingBox], 
                         variant_name: str = 'det', dist_th_tp: float = 2.0) -> Dict[str, Any]:
    """
    단일 track에 대한 메트릭 계산
    
    Args:
        track_gt: GT track 박스 리스트
        track_pred: Prediction track 박스 리스트
        variant_name: variant 이름 (det, kalman_cv, diff, curve)
        dist_th_tp: True positive를 결정하는 거리 threshold (미터)
    """
    metrics = {
        'track_length_gt': len(track_gt),
        'track_length_pred': len(track_pred),
        'num_matches': 0,
        'num_false_positives': 0,
        'num_false_negatives': 0,
        'translation_errors': [],
        'rotation_errors': [],
        'velocity_errors': [],
        'size_errors': [],
        'matched_samples': [],
    }
    
    # GT와 prediction을 sample_token으로 매칭
    gt_dict = {box.sample_token: box for box in track_gt}
    pred_dict = {box.sample_token: box for box in track_pred}
    
    all_samples = set(gt_dict.keys()) | set(pred_dict.keys())
    
    for sample_token in all_samples:
        gt_box = gt_dict.get(sample_token)
        pred_box = pred_dict.get(sample_token)
        
        if gt_box is not None and pred_box is not None:
            # Center distance 계산 (2D)
            center_dist = np.linalg.norm(
                np.array(gt_box.translation[:2]) - np.array(pred_box.translation[:2])
            )
            
            # Distance threshold 내에 있으면 매칭으로 간주
            if center_dist <= dist_th_tp:
                # 매칭됨
                metrics['num_matches'] += 1
                metrics['matched_samples'].append(sample_token)
                
                # Translation error (3D)
                trans_error = np.linalg.norm(
                    np.array(gt_box.translation) - np.array(pred_box.translation)
                )
                metrics['translation_errors'].append(float(trans_error))
                
                # Rotation error
                if hasattr(gt_box, 'rotation') and hasattr(pred_box, 'rotation'):
                    rot_error = quaternion_distance(gt_box.rotation, pred_box.rotation)
                    metrics['rotation_errors'].append(float(rot_error))
                
                # Velocity error
                if hasattr(gt_box, 'velocity') and hasattr(pred_box, 'velocity'):
                    vel_error = velocity_distance(gt_box.velocity, pred_box.velocity)
                    metrics['velocity_errors'].append(float(vel_error))
                
                # Size error
                size_error = np.linalg.norm(
                    np.array(gt_box.size) - np.array(pred_box.size)
                )
                metrics['size_errors'].append(float(size_error))
            else:
                # 거리가 너무 멀어서 매칭되지 않음
                metrics['num_false_positives'] += 1
                metrics['num_false_negatives'] += 1
                
        elif gt_box is None and pred_box is not None:
            # False positive
            metrics['num_false_positives'] += 1
        elif gt_box is not None and pred_box is None:
            # False negative
            metrics['num_false_negatives'] += 1
    
    # 평균 오차 계산
    metrics['mean_translation_error'] = float(np.mean(metrics['translation_errors'])) if metrics['translation_errors'] else np.nan
    metrics['std_translation_error'] = float(np.std(metrics['translation_errors'])) if len(metrics['translation_errors']) > 1 else np.nan
    metrics['mean_rotation_error'] = float(np.mean(metrics['rotation_errors'])) if metrics['rotation_errors'] else np.nan
    metrics['std_rotation_error'] = float(np.std(metrics['rotation_errors'])) if len(metrics['rotation_errors']) > 1 else np.nan
    metrics['mean_velocity_error'] = float(np.mean(metrics['velocity_errors'])) if metrics['velocity_errors'] else np.nan
    metrics['std_velocity_error'] = float(np.std(metrics['velocity_errors'])) if len(metrics['velocity_errors']) > 1 else np.nan
    metrics['mean_size_error'] = float(np.mean(metrics['size_errors'])) if metrics['size_errors'] else np.nan
    metrics['std_size_error'] = float(np.std(metrics['size_errors'])) if len(metrics['size_errors']) > 1 else np.nan
    
    # MOTA 계산
    total_gt = metrics['track_length_gt']
    if total_gt > 0:
        metrics['precision'] = metrics['num_matches'] / (metrics['num_matches'] + metrics['num_false_positives']) if (metrics['num_matches'] + metrics['num_false_positives']) > 0 else 0.0
        metrics['recall'] = metrics['num_matches'] / total_gt
        metrics['mota'] = 1.0 - (metrics['num_false_positives'] + metrics['num_false_negatives']) / total_gt
    else:
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['mota'] = 0.0
    
    return metrics


def evaluate_trackwise(
    result_path: str,
    nusc_dataroot: str,
    eval_set: str = "val",
    nusc_version: str = "v1.0-trainval",
    output_path: str = None,
    verbose: bool = True
):
    """
    각 track별로 성능을 평가하고 결과를 JSON 파일로 저장
    
    Args:
        result_path: 결과 JSON 파일 경로
        nusc_dataroot: nuScenes 데이터셋 루트 경로
        eval_set: 평가할 데이터셋 split (val, train, test)
        nusc_version: nuScenes 버전
        output_path: 출력 JSON 파일 경로 (None이면 result_path와 같은 디렉토리에 저장)
        verbose: 상세 출력 여부
    """
    if verbose:
        print("Trackwise 평가 시작...")
    
    # 출력 경로 설정
    if output_path is None:
        output_dir = os.path.dirname(result_path)
        output_path = os.path.join(output_dir, "trackwise_evaluation.json")
    
    # Config 로드 (이것이 TRACKING_NAMES를 설정함)
    cfg = track_configs("tracking_nips_2019")
    
    # NuScenes 객체 초기화
    nusc = NuScenes(version=nusc_version, verbose=False, dataroot=nusc_dataroot)
    
    # Prediction 로드 (여러 variant)
    if verbose:
        print("결과 파일 로드 중...")
    pred_variants, meta = load_prediction_variants(
        result_path, cfg.max_boxes_per_sample, TrackingBox, cfg=cfg, verbose=verbose
    )
    
    # GT 로드
    if verbose:
        print("Ground truth 로드 중...")
    gt_boxes = load_gt(nusc, eval_set, TrackingBox, verbose=False)
    
    # Center distance 추가
    gt_boxes = add_center_dist(nusc, gt_boxes)
    for variant_name in tqdm(list(pred_variants.keys()), disable=False, desc="Add center dist (pred)"):
        pred_variants[variant_name] = add_center_dist(nusc, pred_variants[variant_name])
    
    # 필터링
    if verbose:
        print("박스 필터링 중...")
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=False)
    for variant_name in tqdm(list(pred_variants.keys()), disable=False, desc="Filter boxes (pred)"):
        pred_variants[variant_name] = filter_eval_boxes(
            nusc, pred_variants[variant_name], cfg.class_range, verbose=False
        )
    
    # Tracks 생성
    if verbose:
        print("트랙 생성 중...")
    tracks_gt = create_tracks(gt_boxes, nusc, eval_set, gt=True)
    tracks_pred_variants = {}
    for variant_name in tqdm(list(pred_variants.keys()), disable=False, desc="Create tracks (pred)"):
        tracks_pred_variants[variant_name] = create_tracks(
            pred_variants[variant_name], nusc, eval_set, gt=False
        )
    
    # nuScenes 스타일의 전체/클래스별 메트릭 계산 (variant별)
    def compute_overall_metrics_for_variant(variant_name: str):
        tracks_pred = tracks_pred_variants[variant_name]
        # 직렬 실행 시에는 클래스 진행률 바를 표시
        _, metrics_summary = _compute_overall_metrics_for_variant_mp(
            variant_name, tracks_gt, tracks_pred, cfg, disable_class_tqdm=False)
        return metrics_summary

    # Track별 평가
    if verbose:
        print("각 트랙별 평가 수행 중...")
    
    trackwise_results = {}

    

    scene_ids = list(tracks_gt.keys())
    # 병렬 처리: 씬 단위로 분할
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    tasks = []
    for scene_id in scene_ids:
        scene_tracks_gt = tracks_gt[scene_id]
        scene_tracks_pred = {variant: tracks_pred_variants[variant][scene_id]
                             for variant in tracks_pred_variants.keys()}
        tasks.append((scene_id, scene_tracks_gt, scene_tracks_pred, cfg))

    with mp.get_context('fork').Pool(processes=min(len(scene_ids), os.cpu_count() or 1)) as pool:
        for sid, sres in tqdm(pool.imap_unordered(_evaluate_scene_mp, tasks), total=len(tasks), disable=False, desc="Scenes"):
            trackwise_results.update(sres)
    
    # variant별 전체 메트릭 계산
    if verbose:
        print("nuScenes 메트릭(전체/클래스별) 계산 중...")
    overall_metrics = {}
    variant_order = [v for v in ['det', 'kalman_cv', 'diff', 'curve'] if v in tracks_pred_variants]
    # 프로세스 병렬화: 변이 단위를 병렬 실행 (이름만 전달하여 피클 이슈 회피)
    if len(variant_order) > 1:
        # 중첩 병렬화 방지: BLAS/OMP 스레드 수를 1로 제한(이미 설정되어 있지 않으면)
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

        # 전역 참조 설정(포크에서는 복사됨)
        global _GLOBAL_tracks_gt, _GLOBAL_tracks_pred_variants, _GLOBAL_cfg
        _GLOBAL_tracks_gt = tracks_gt
        _GLOBAL_tracks_pred_variants = tracks_pred_variants
        _GLOBAL_cfg = cfg

        with mp.get_context('fork').Pool(processes=min(len(variant_order), os.cpu_count() or 1)) as pool:
            for variant_name, metrics_summary in tqdm(
                    pool.imap_unordered(_compute_overall_metrics_for_variant_by_name, variant_order),
                    total=len(variant_order), disable=False, desc="Overall metrics"):
                overall_metrics[variant_name] = metrics_summary
    else:
        for variant_name in tqdm(variant_order, disable=False, desc="Overall metrics"):
            overall_metrics[variant_name] = compute_overall_metrics_for_variant(variant_name)

    # 결과 저장
    if verbose:
        print(f"결과 저장 중: {output_path}")
    
    output_data = {
        'meta': meta,
        'trackwise_results': trackwise_results,
        'overall_metrics': overall_metrics,
        'summary': {
            'total_tracks': len(trackwise_results),
            'variants_evaluated': ['det', 'kalman_cv', 'diff', 'curve']
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"평가 완료! 결과가 {output_path}에 저장되었습니다.")
        print(f"총 {len(trackwise_results)}개의 트랙이 평가되었습니다.")
    
    return output_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trackwise 평가 수행')
    parser.add_argument('result_path', type=str, help='결과 JSON 파일 경로')
    parser.add_argument('--nusc_dataroot', type=str, required=True, help='nuScenes 데이터셋 루트 경로')
    parser.add_argument('--eval_set', type=str, default='val', help='평가할 데이터셋 split')
    parser.add_argument('--nusc_version', type=str, default='v1.0-trainval', help='nuScenes 버전')
    parser.add_argument('--output_path', type=str, default=None, help='출력 JSON 파일 경로')
    parser.add_argument('--verbose', type=int, default=1, help='상세 출력 여부')
    
    args = parser.parse_args()
    
    evaluate_trackwise(
        result_path=args.result_path,
        nusc_dataroot=args.nusc_dataroot,
        eval_set=args.eval_set,
        nusc_version=args.nusc_version,
        output_path=args.output_path,
        verbose=bool(args.verbose)
    )
