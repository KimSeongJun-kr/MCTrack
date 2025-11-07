import json
import os
import copy
import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.config import config_factory as track_configs
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingConfig, TrackingMetrics, TrackingMetricDataList, TrackingMetricData
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP


# 전역 변수 (multiprocessing에서 사용)
_GLOBAL_tracks_gt = None
_GLOBAL_tracks_pred_variants = None
_GLOBAL_cfg = None

# Variant 설정
VARIANT_NAMES = ['det', 'kalman_cv', 'diff', 'curve']
VARIANT_CONFIGS = [
    ('rotation_det', 'velocity_det', 'det'),
    ('rotation_kalman_cv', 'velocity_kalman_cv', 'kalman_cv'),
    ('rotation_det', 'velocity_diff', 'diff'),
    ('rotation_det', 'velocity_curve', 'curve'),
]


def _accumulate_class_metric(tracks_gt, tracks_pred, class_name: str, cfg) -> Tuple[str, Any]:
    """단일 클래스에 대한 TrackingMetricData 계산"""
    ev = TrackingEvaluation(
        tracks_gt, tracks_pred, class_name,
        cfg.dist_fcn_callable, cfg.dist_th_tp, cfg.min_recall,
        num_thresholds=TrackingMetricData.nelem,
        metric_worst=cfg.metric_worst,
        verbose=False, output_dir=None, render_classes=None
    )
    md = ev.accumulate()
    return class_name, md


def _accumulate_class_metric_wrapper(args):
    """클래스 메트릭 계산 래퍼 (ThreadPoolExecutor용)"""
    tracks_gt, tracks_pred, class_name, cfg = args
    return _accumulate_class_metric(tracks_gt, tracks_pred, class_name, cfg)


def _compute_class_metrics_parallel(tracks_gt, tracks_pred, class_names, cfg,
                                    variant_name: str, show_progress: bool = False):
    """클래스별 메트릭을 병렬로 계산"""
    metric_data_list = TrackingMetricDataList()
    max_workers = min(len(class_names), (os.cpu_count() or 1))
    
    if max_workers <= 1:
        # 직렬 처리
        iterator = tqdm(class_names, disable=not show_progress, desc=f"{variant_name} classes") if show_progress else class_names
        for cname in iterator:
            cname, md = _accumulate_class_metric(tracks_gt, tracks_pred, cname, cfg)
            metric_data_list.set(cname, md)
    else:
        # 스레드 병렬 처리
        tasks = [(tracks_gt, tracks_pred, cname, cfg) for cname in class_names]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_accumulate_class_metric_wrapper, task): task[2] for task in tasks}
            
            if show_progress:
                with tqdm(total=len(class_names), disable=False, desc=f"{variant_name} classes") as pbar:
                    for fut in as_completed(futures):
                        cname, md = fut.result()
                        metric_data_list.set(cname, md)
                        pbar.update(1)
            else:
                for fut in as_completed(futures):
                    cname, md = fut.result()
                    metric_data_list.set(cname, md)
    
    return metric_data_list


def _compute_rotation_velocity_metrics(tracks_gt, tracks_pred, class_names, cfg):
    """rotation과 velocity 오차 메트릭 계산"""
    from nuscenes.eval.common.utils import center_distance
    
    rot_errors = {cn: [] for cn in class_names}
    vel_errors = {cn: [] for cn in class_names}
    rot_error_counts = {cn: 0 for cn in class_names}
    vel_error_counts = {cn: 0 for cn in class_names}
    
    # 모든 scene을 순회
    for scene_id in tracks_gt.keys():
        if scene_id not in tracks_pred:
            continue
        
        scene_tracks_gt = tracks_gt[scene_id]
        scene_tracks_pred = tracks_pred[scene_id]
        
        # 모든 timestamp에서 매칭
        for timestamp in scene_tracks_gt.keys():
            if timestamp not in scene_tracks_pred:
                continue
            
            frame_gt = scene_tracks_gt[timestamp]
            frame_pred = scene_tracks_pred[timestamp]
            
            # 각 클래스별로 처리
            for class_name in class_names:
                # 현재 클래스의 GT와 prediction 필터링
                gt_boxes = [b for b in frame_gt if b.tracking_name == class_name]
                pred_boxes = [b for b in frame_pred if b.tracking_name == class_name]
                
                if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                    continue
                
                # 가장 가까운 매칭 찾기 (greedy matching)
                taken_gt = set()
                taken_pred = set()
                
                # 거리 순으로 정렬
                matches = []
                for i, gt_box in enumerate(gt_boxes):
                    for j, pred_box in enumerate(pred_boxes):
                        dist = center_distance(gt_box, pred_box)
                        if dist <= cfg.dist_th_tp:
                            matches.append((dist, i, j))
                
                matches.sort(key=lambda x: x[0])
                
                # 매칭 수행
                for dist, i, j in matches:
                    if i in taken_gt or j in taken_pred:
                        continue
                    
                    gt_box = gt_boxes[i]
                    pred_box = pred_boxes[j]
                    
                    # rotation 오차 계산
                    if hasattr(gt_box, 'rotation') and hasattr(pred_box, 'rotation'):
                        if gt_box.rotation is not None and pred_box.rotation is not None:
                            try:
                                rot_err = quaternion_distance(gt_box.rotation, pred_box.rotation)
                                rot_errors[class_name].append(rot_err)
                                rot_error_counts[class_name] += 1
                            except:
                                pass
                    
                    # velocity 오차 계산
                    if hasattr(gt_box, 'velocity') and hasattr(pred_box, 'velocity'):
                        if gt_box.velocity is not None and pred_box.velocity is not None:
                            try:
                                vel_err = velocity_distance(gt_box.velocity, pred_box.velocity)
                                vel_errors[class_name].append(vel_err)
                                vel_error_counts[class_name] += 1
                            except:
                                pass
                    
                    taken_gt.add(i)
                    taken_pred.add(j)
    
    # 평균 및 표준편차 계산
    rot_metrics = {}
    vel_metrics = {}
    
    for class_name in class_names:
        if len(rot_errors[class_name]) > 0:
            rot_metrics[class_name] = {
                'mean': float(np.mean(rot_errors[class_name])),
                'std': float(np.std(rot_errors[class_name])),
                'median': float(np.median(rot_errors[class_name])),
                'count': rot_error_counts[class_name]
            }
        else:
            rot_metrics[class_name] = {
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'count': 0
            }
        
        if len(vel_errors[class_name]) > 0:
            vel_metrics[class_name] = {
                'mean': float(np.mean(vel_errors[class_name])),
                'std': float(np.std(vel_errors[class_name])),
                'median': float(np.median(vel_errors[class_name])),
                'count': vel_error_counts[class_name]
            }
        else:
            vel_metrics[class_name] = {
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'count': 0
            }
    
    return rot_metrics, vel_metrics


def _compute_overall_metrics_for_variant_mp(variant_name: str,
                                            tracks_gt,
                                            tracks_pred,
                                            cfg,
                                            disable_class_tqdm: bool = True):
    """변이별 전체 메트릭 계산 (multiprocessing용)"""
    metrics = TrackingMetrics(cfg)
    class_names = cfg.class_names if hasattr(cfg, 'class_names') else cfg.tracking_names
    
    # 클래스별 메트릭 계산
    metric_data_list = _compute_class_metrics_parallel(
        tracks_gt, tracks_pred, class_names, cfg,
        variant_name, show_progress=not disable_class_tqdm
    )

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
                values[np.isnan(values)] = cfg.metric_worst[metric_name]
                value = float(np.nanmean(values))
            metrics.add_label_metric(metric_name, class_name, value)
    
    # Rotation 및 Velocity 메트릭 계산
    rot_metrics, vel_metrics = _compute_rotation_velocity_metrics(
        tracks_gt, tracks_pred, class_names, cfg
    )
    
    # 메트릭 시리얼라이즈
    metrics_dict = metrics.serialize()
    
    # rotation 및 velocity 메트릭 추가
    metrics_dict['rotation_errors'] = {}
    metrics_dict['velocity_errors'] = {}
    
    # 클래스별 rotation/velocity 메트릭
    for class_name in class_names:
        metrics_dict['rotation_errors'][class_name] = rot_metrics[class_name]
        metrics_dict['velocity_errors'][class_name] = vel_metrics[class_name]
    
    # 전체 평균 계산 (클래스 평균)
    rot_means = [rot_metrics[cn]['mean'] for cn in class_names if not np.isnan(rot_metrics[cn]['mean'])]
    vel_means = [vel_metrics[cn]['mean'] for cn in class_names if not np.isnan(vel_metrics[cn]['mean'])]
    
    metrics_dict['rotation_errors']['all'] = {
        'mean': float(np.mean(rot_means)) if len(rot_means) > 0 else np.nan,
        'std': float(np.std(rot_means)) if len(rot_means) > 0 else np.nan,
        'median': float(np.median(rot_means)) if len(rot_means) > 0 else np.nan,
        'count': sum(rot_metrics[cn]['count'] for cn in class_names)
    }
    
    metrics_dict['velocity_errors']['all'] = {
        'mean': float(np.mean(vel_means)) if len(vel_means) > 0 else np.nan,
        'std': float(np.std(vel_means)) if len(vel_means) > 0 else np.nan,
        'median': float(np.median(vel_means)) if len(vel_means) > 0 else np.nan,
        'count': sum(vel_metrics[cn]['count'] for cn in class_names)
    }

    return variant_name, metrics_dict


def _compute_overall_metrics_for_variant_by_name(variant_name: str):
    """Fork-safe worker that only receives a picklable name and reads globals."""
    tracks_gt = _GLOBAL_tracks_gt
    tracks_pred = _GLOBAL_tracks_pred_variants[variant_name]
    cfg = _GLOBAL_cfg
    _, metrics_summary = _compute_overall_metrics_for_variant_mp(
        variant_name, tracks_gt, tracks_pred, cfg, disable_class_tqdm=True)
    return variant_name, metrics_summary


def _extract_gt_track_ids(scene_tracks_gt):
    """씬에서 GT track ID와 클래스명 쌍을 추출"""
    gt_track_ids = set()
    for frame_boxes in scene_tracks_gt.values():
        for box in frame_boxes:
            gt_track_ids.add((box.tracking_id, box.tracking_name))
    return gt_track_ids


def _collect_track_boxes(scene_tracks, track_id, class_name):
    """특정 track_id와 class_name에 해당하는 박스들을 수집"""
    track_boxes = []
    for frame_boxes in scene_tracks.values():
        for box in frame_boxes:
            if box.tracking_id == track_id and box.tracking_name == class_name:
                track_boxes.append(box)
    return track_boxes


def _find_best_matching_pred_track(scene_tracks_pred, variant_name, class_name, track_gt):
    """GT track과 가장 겹치는 prediction track 찾기"""
    # 해당 클래스의 prediction track ID들 수집
    pred_track_ids = set()
    for frame_boxes in scene_tracks_pred[variant_name].values():
        for box in frame_boxes:
            if box.tracking_name == class_name:
                pred_track_ids.add(box.tracking_id)
    
    # 각 prediction track과의 겹침 계산
    best_match_id = None
    best_overlap = 0
    gt_samples = {box.sample_token for box in track_gt}
    
    for pred_track_id in pred_track_ids:
        pred_track = _collect_track_boxes(scene_tracks_pred[variant_name], pred_track_id, class_name)
        pred_samples = {box.sample_token for box in pred_track}
        overlap = len(gt_samples & pred_samples)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_match_id = pred_track_id
    
    # 최적 매칭 track의 박스들 반환
    if best_match_id is not None:
        return _collect_track_boxes(scene_tracks_pred[variant_name], best_match_id, class_name)
    return []


def _evaluate_scene_mp(args):
    """씬 단위 평가 (multiprocessing entry point)"""
    scene_id, scene_tracks_gt, scene_tracks_pred, cfg_local = args
    scene_result = {}
    dist_th_tp = cfg_local.dist_th_tp if hasattr(cfg_local, 'dist_th_tp') else 2.0
    
    # GT track ID 수집
    gt_track_ids = _extract_gt_track_ids(scene_tracks_gt)
    
    # 각 GT track에 대해 평가
    for track_id, class_name in gt_track_ids:
        track_key = f"{scene_id}_{track_id}_{class_name}"
        
        # GT track 수집
        track_gt = _collect_track_boxes(scene_tracks_gt, track_id, class_name)
        if len(track_gt) == 0:
            continue
        
        track_metrics = {
            'scene_id': scene_id,
            'track_id': track_id,
            'class_name': class_name,
            'track_length_gt': len(track_gt),
            'variants': {}
        }
        
        # 각 variant에 대해 평가
        for variant_name in VARIANT_NAMES:
            if variant_name not in scene_tracks_pred:
                continue
            
            # 최적 매칭 prediction track 찾기
            track_pred = _find_best_matching_pred_track(
                scene_tracks_pred, variant_name, class_name, track_gt
            )
            
            # 메트릭 계산
            variant_metrics = compute_track_metrics(
                track_gt, track_pred, variant_name, dist_th_tp=dist_th_tp
            )
            track_metrics['variants'][variant_name] = variant_metrics
        
        scene_result[track_key] = track_metrics
    
    return scene_id, scene_result


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


def _verify_original_json_differences(data, verbose=False, max_samples=10):
    """원본 JSON에서 variant별 필드 차이 확인 (디버깅용)"""
    if not verbose:
        return
    
    sample_tokens = list(data['results'].keys())
    if not sample_tokens:
        return
    
    print(f"\n[디버깅] 원본 JSON 파일 Variant 차이 확인:")
    print(f"  전체 샘플 수: {len(sample_tokens)}, 확인할 샘플 수: {min(max_samples, len(sample_tokens))}")
    
    # 전체 통계
    total_detections = 0
    rot_identical_count = 0
    rot_different_count = 0
    vel_identical_count = 0
    vel_different_count = 0
    rot_diffs = []
    vel_diffs = []
    samples_with_diffs = []
    
    # 여러 sample_token 확인 (차이가 있는 샘플 찾기)
    samples_checked = 0
    for sample_token in sample_tokens[:max_samples]:
        detections = data['results'][sample_token]
        if not detections:
            continue
        
        samples_checked += 1
        sample_rot_diff_count = 0
        sample_vel_diff_count = 0
        
        for det in detections:
            total_detections += 1
            
            rot_det = det.get('rotation_det')
            rot_kalman = det.get('rotation_kalman_cv')
            vel_det = det.get('velocity_det')
            vel_kalman = det.get('velocity_kalman_cv')
            
            if rot_det and rot_kalman:
                if rot_det == rot_kalman:
                    rot_identical_count += 1
                else:
                    rot_different_count += 1
                    sample_rot_diff_count += 1
                    rot_diff = quaternion_distance(rot_det, rot_kalman)
                    rot_diffs.append(rot_diff)
            
            if vel_det and vel_kalman:
                if vel_det == vel_kalman:
                    vel_identical_count += 1
                else:
                    vel_different_count += 1
                    sample_vel_diff_count += 1
                    vel_diff = velocity_distance(vel_det, vel_kalman)
                    vel_diffs.append(vel_diff)
        
        # 차이가 있는 샘플 기록
        if sample_rot_diff_count > 0 or sample_vel_diff_count > 0:
            samples_with_diffs.append({
                'sample_token': sample_token,
                'detection_count': len(detections),
                'rot_diff_count': sample_rot_diff_count,
                'vel_diff_count': sample_vel_diff_count
            })
    
    # 전체 통계 출력
    print(f"  확인한 샘플 수: {samples_checked}")
    print(f"  총 detection 수: {total_detections}")
    print()
    print(f"  Rotation (det vs kalman_cv):")
    print(f"    동일: {rot_identical_count}/{total_detections} ({rot_identical_count/total_detections*100:.1f}%)")
    print(f"    다름: {rot_different_count}/{total_detections} ({rot_different_count/total_detections*100:.1f}%)")
    if rot_diffs:
        print(f"    평균 차이: {np.mean(rot_diffs):.6f} 라디안")
        print(f"    최대 차이: {np.max(rot_diffs):.6f} 라디안")
    
    print(f"  Velocity (det vs kalman_cv):")
    print(f"    동일: {vel_identical_count}/{total_detections} ({vel_identical_count/total_detections*100:.1f}%)")
    print(f"    다름: {vel_different_count}/{total_detections} ({vel_different_count/total_detections*100:.1f}%)")
    if vel_diffs:
        print(f"    평균 차이: {np.mean(vel_diffs):.6f} m/s")
        print(f"    최대 차이: {np.max(vel_diffs):.6f} m/s")
    
    # 차이가 있는 샘플 출력
    if samples_with_diffs:
        print(f"\n  차이가 있는 샘플: {len(samples_with_diffs)}/{samples_checked}개")
        # 첫 번째 차이가 있는 샘플에서 예시 출력
        first_diff_sample = samples_with_diffs[0]['sample_token']
        detections = data['results'][first_diff_sample]
        print(f"  예시 샘플 (sample_token: {first_diff_sample[:30]}...):")
        
        for det in detections:
            rot_det = det.get('rotation_det')
            rot_kalman = det.get('rotation_kalman_cv')
            vel_det = det.get('velocity_det')
            vel_kalman = det.get('velocity_kalman_cv')
            
            rot_diff = quaternion_distance(rot_det, rot_kalman) if (rot_det and rot_kalman) else 0
            vel_diff = velocity_distance(vel_det, vel_kalman) if (vel_det and vel_kalman) else 0
            
            if rot_diff > 1e-6 or vel_diff > 1e-6:
                print(f"    tracking_id: {det.get('tracking_id', 'unknown')}, class: {det.get('tracking_name', 'unknown')}")
                if rot_diff > 1e-6:
                    print(f"      rotation 차이: {rot_diff:.6f} 라디안")
                if vel_diff > 1e-6:
                    print(f"      velocity 차이: {vel_diff:.6f} m/s")
                break
    else:
        print(f"\n  ⚠️  확인한 모든 샘플에서 rotation과 velocity가 동일합니다!")
        print(f"  → 더 많은 샘플을 확인하거나, 실제로 파일에 차이가 없을 수 있습니다.")
    print()


def load_prediction_variants(result_path: str, max_boxes_per_sample: int, box_cls, cfg: TrackingConfig = None, verbose: bool = False, step_info=None):
    """
    결과 파일을 로드하고 여러 variant로 변환
    
    Args:
        result_path: 결과 파일 경로
        max_boxes_per_sample: 샘플당 최대 박스 수 (사용되지 않으나 호출 호환성을 위해 유지)
        box_cls: 박스 클래스
        cfg: TrackingConfig (TRACKING_NAMES를 설정하기 위해 필요)
        verbose: 상세 출력 여부 (사용되지 않으나 호출 호환성을 위해 유지)
        step_info: (step, total_steps) 튜플, 단계 번호 표시용
    """
    with open(result_path) as f:
        data = json.load(f)
    
    # 원본 JSON 차이 확인 (디버깅) - 여러 샘플 확인
    _verify_original_json_differences(data, verbose, max_samples=50)
    
    variants = {}
    if step_info:
        step, total_steps = step_info
        desc = f"[{step}/{total_steps}] Deserialize variants"
    else:
        desc = "Deserialize variants"
    
    # process_data 직후 변환된 데이터 확인용 (디버깅) - 여러 샘플 확인
    processed_samples = {}
    max_check_samples = 20 if verbose else 0
    
    for rot_field, vel_field, variant_name in tqdm(VARIANT_CONFIGS, disable=False, desc=desc):
        data_variant = copy.deepcopy(data)
        process_data(data_variant, rot_field, vel_field)
        
        # process_data 직후 변환된 데이터 확인 (디버깅) - 여러 샘플 저장
        if verbose and len(data_variant['results']) > 0:
            sample_tokens = list(data_variant['results'].keys())[:max_check_samples]
            for sample_token in sample_tokens:
                if sample_token not in processed_samples:
                    processed_samples[sample_token] = {}
                processed_samples[sample_token][variant_name] = data_variant['results'][sample_token]
        
        all_results = EvalBoxes.deserialize(data_variant['results'], box_cls)
        variants[variant_name] = all_results
    
    # process_data 직후 variant별 차이 확인 (디버깅) - 차이가 있는 샘플 찾아서 확인
    if verbose and processed_samples:
        print(f"\n[디버깅] process_data 직후 변환된 데이터 Variant 차이 확인:")
        
        # 차이가 있는 샘플 찾기
        diff_samples = []
        for sample_token, variant_data in processed_samples.items():
            if 'det' not in variant_data or 'kalman_cv' not in variant_data:
                continue
            
            dets_det = variant_data['det']
            dets_kalman = variant_data['kalman_cv']
            
            det_dict = {d.get('tracking_id', ''): d for d in dets_det}
            kalman_dict = {d.get('tracking_id', ''): d for d in dets_kalman}
            common_ids = set(det_dict.keys()) & set(kalman_dict.keys())
            
            rot_diff_count = 0
            vel_diff_count = 0
            
            for tid in common_ids:
                d_det = det_dict[tid]
                d_kalman = kalman_dict[tid]
                
                rot_det = d_det.get('rotation')
                rot_kalman = d_kalman.get('rotation')
                vel_det = d_det.get('velocity')
                vel_kalman = d_kalman.get('velocity')
                
                if rot_det and rot_kalman and rot_det != rot_kalman:
                    rot_diff_count += 1
                if vel_det and vel_kalman and vel_det != vel_kalman:
                    vel_diff_count += 1
            
            if rot_diff_count > 0 or vel_diff_count > 0:
                diff_samples.append((sample_token, variant_data))
        
        # 차이가 있는 샘플 출력 (없으면 첫 번째 샘플 출력)
        samples_to_check = diff_samples[:3] if diff_samples else list(processed_samples.items())[:1]
        
        for sample_token, variant_data in samples_to_check:
            print(f"  Sample_token: {sample_token[:30]}...")
            
            if 'det' in variant_data and 'kalman_cv' in variant_data:
                dets_det = variant_data['det']
                dets_kalman = variant_data['kalman_cv']
                
                det_dict = {d.get('tracking_id', ''): d for d in dets_det}
                kalman_dict = {d.get('tracking_id', ''): d for d in dets_kalman}
                common_ids = set(det_dict.keys()) & set(kalman_dict.keys())
                
                rot_diff_count = 0
                vel_diff_count = 0
                rot_diffs = []
                vel_diffs = []
                
                for tid in common_ids:
                    d_det = det_dict[tid]
                    d_kalman = kalman_dict[tid]
                    
                    rot_det = d_det.get('rotation')
                    rot_kalman = d_kalman.get('rotation')
                    vel_det = d_det.get('velocity')
                    vel_kalman = d_kalman.get('velocity')
                    
                    if rot_det and rot_kalman:
                        if rot_det != rot_kalman:
                            rot_diff_count += 1
                            rot_diffs.append(quaternion_distance(rot_det, rot_kalman))
                    
                    if vel_det and vel_kalman:
                        if vel_det != vel_kalman:
                            vel_diff_count += 1
                            vel_diffs.append(velocity_distance(vel_det, vel_kalman))
                
                print(f"    det vs kalman_cv (매칭: {len(common_ids)}개):")
                print(f"      rotation 다름: {rot_diff_count}/{len(common_ids)} ({rot_diff_count/len(common_ids)*100:.1f}%)")
                if rot_diffs:
                    print(f"      rotation 평균 차이: {np.mean(rot_diffs):.6f} 라디안")
                print(f"      velocity 다름: {vel_diff_count}/{len(common_ids)} ({vel_diff_count/len(common_ids)*100:.1f}%)")
                if vel_diffs:
                    print(f"      velocity 평균 차이: {np.mean(vel_diffs):.6f} m/s")
                
                # 차이가 있는 예시 출력
                if rot_diff_count == 0 and vel_diff_count == 0:
                    print(f"      ⚠️  이 샘플에서는 차이가 없습니다. 다른 샘플을 확인해야 합니다.")
        
        if diff_samples:
            print(f"  총 {len(diff_samples)}개 샘플에서 차이 발견됨 (위에 3개만 표시)")
        print()
    
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


def _set_thread_limits():
    """중첩 병렬화 방지를 위해 BLAS/OMP 스레드 수를 1로 제한"""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


def _load_and_preprocess_data(result_path, nusc_dataroot, eval_set, nusc_version, cfg, verbose, step_info):
    """데이터 로드 및 전처리"""
    step, total_steps = step_info
    
    # Prediction 로드 (여러 variant)
    if verbose:
        print(f"[{step}/{total_steps}] 결과 파일 로드 중...")
    pred_variants, meta = load_prediction_variants(
        result_path, cfg.max_boxes_per_sample, TrackingBox, cfg=cfg, verbose=verbose, step_info=(step, total_steps)
    )
    
    # GT 로드
    step += 1
    if verbose:
        print(f"[{step}/{total_steps}] Ground truth 로드 중...")
    nusc = NuScenes(version=nusc_version, verbose=False, dataroot=nusc_dataroot)
    gt_boxes = load_gt(nusc, eval_set, TrackingBox, verbose=False)
    
    # Center distance 추가
    step += 1
    if verbose:
        print(f"[{step}/{total_steps}] Center distance 추가 중...")
    gt_boxes = add_center_dist(nusc, gt_boxes)
    for variant_name in tqdm(list(pred_variants.keys()), disable=False, desc=f"[{step}/{total_steps}] Add center dist (pred)"):
        pred_variants[variant_name] = add_center_dist(nusc, pred_variants[variant_name])
    
    # 필터링
    step += 1
    if verbose:
        print(f"[{step}/{total_steps}] 박스 필터링 중...")
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=False)
    for variant_name in tqdm(list(pred_variants.keys()), disable=False, desc=f"[{step}/{total_steps}] Filter boxes (pred)"):
        pred_variants[variant_name] = filter_eval_boxes(
            nusc, pred_variants[variant_name], cfg.class_range, verbose=False
        )
    
    # EvalBoxes 단계에서 variant 차이 확인 (디버깅) - 여러 샘플 확인
    _verify_variant_differences_in_boxes(pred_variants, verbose, max_samples=20)
    
    # Tracks 생성
    step += 1
    if verbose:
        print(f"[{step}/{total_steps}] 트랙 생성 중...")
    tracks_gt = create_tracks(gt_boxes, nusc, eval_set, gt=True)
    tracks_pred_variants = {}
    for variant_name in tqdm(list(pred_variants.keys()), disable=False, desc=f"[{step}/{total_steps}] Create tracks (pred)"):
        tracks_pred_variants[variant_name] = create_tracks(
            pred_variants[variant_name], nusc, eval_set, gt=False
        )
    
    # EvalBoxes와 Tracks에서 같은 샘플 비교 (디버깅)
    if verbose:
        _verify_boxes_vs_tracks(pred_variants, tracks_pred_variants, nusc, max_samples=5)
    
    return tracks_gt, tracks_pred_variants, meta


def _filter_tracks_by_scenes(tracks_gt, tracks_pred_variants, max_scenes=None):
    """씬 제한을 적용하여 tracks 필터링 (디버깅용)"""
    if max_scenes is None:
        return tracks_gt, tracks_pred_variants
    
    scene_ids = sorted(list(tracks_gt.keys()))[:max_scenes]
    
    # GT tracks 필터링
    tracks_gt_filtered = {scene_id: tracks_gt[scene_id] for scene_id in scene_ids}
    
    # Pred tracks 필터링
    tracks_pred_variants_filtered = {}
    for variant_name, variant_tracks in tracks_pred_variants.items():
        tracks_pred_variants_filtered[variant_name] = {
            scene_id: variant_tracks[scene_id] for scene_id in scene_ids
        }
    
    return tracks_gt_filtered, tracks_pred_variants_filtered


def _evaluate_scenes_parallel(tracks_gt, tracks_pred_variants, cfg, step_info):
    """씬 단위 평가를 병렬로 수행"""
    step, total_steps = step_info
    _set_thread_limits()
    
    scene_ids = list(tracks_gt.keys())
    tasks = []
    for scene_id in scene_ids:
        scene_tracks_gt = tracks_gt[scene_id]
        scene_tracks_pred = {
            variant: tracks_pred_variants[variant][scene_id]
            for variant in tracks_pred_variants.keys()
        }
        tasks.append((scene_id, scene_tracks_gt, scene_tracks_pred, cfg))
    
    trackwise_results = {}
    with mp.get_context('fork').Pool(processes=min(len(scene_ids), os.cpu_count() or 1)) as pool:
        for sid, sres in tqdm(
                pool.imap_unordered(_evaluate_scene_mp, tasks),
                total=len(tasks), disable=False, desc=f"[{step}/{total_steps}] Scenes"
        ):
            trackwise_results.update(sres)
    
    return trackwise_results


def _verify_variant_differences_in_boxes(pred_variants, verbose=False, max_samples=10):
    """EvalBoxes 단계에서 variant별 데이터 차이 확인 (디버깅용)"""
    if not verbose:
        return
    
    variant_order = list(pred_variants.keys())
    if len(variant_order) < 2:
        return
    
    sample_tokens = pred_variants[variant_order[0]].sample_tokens
    if not sample_tokens:
        return
    
    print(f"\n[디버깅] EvalBoxes 단계 Variant 차이 확인:")
    print(f"  확인할 샘플 수: {min(max_samples, len(sample_tokens))}/{len(sample_tokens)}")
    
    # 여러 샘플에서 통계 수집
    total_trans_diff = 0
    total_rot_diff = 0
    total_vel_diff = 0
    total_count = 0
    samples_checked = 0
    
    for sample_token in sample_tokens[:max_samples]:
        samples_checked += 1
        sample_trans_diff = 0
        sample_rot_diff = 0
        sample_vel_diff = 0
        sample_count = 0
        
        for i, var1 in enumerate(variant_order):
            for var2 in variant_order[i+1:]:
                boxes1 = pred_variants[var1][sample_token]
                boxes2 = pred_variants[var2][sample_token]
                
                if len(boxes1) == 0 or len(boxes2) == 0:
                    continue
                
                # tracking_id와 tracking_name으로 딕셔너리 생성하여 정확한 매칭
                boxes1_dict = {(b.tracking_id, b.tracking_name): b for b in boxes1}
                boxes2_dict = {(b.tracking_id, b.tracking_name): b for b in boxes2}
                
                common_keys = set(boxes1_dict.keys()) & set(boxes2_dict.keys())
                
                if len(common_keys) == 0:
                    continue
                
                for key in common_keys:
                    b1 = boxes1_dict[key]
                    b2 = boxes2_dict[key]
                    
                    sample_trans_diff += np.linalg.norm(np.array(b1.translation) - np.array(b2.translation))
                    if hasattr(b1, 'rotation') and hasattr(b2, 'rotation'):
                        sample_rot_diff += quaternion_distance(b1.rotation, b2.rotation)
                    if hasattr(b1, 'velocity') and hasattr(b2, 'velocity'):
                        sample_vel_diff += velocity_distance(b1.velocity, b2.velocity)
                    sample_count += 1
                    total_count += 1
        
        if sample_count > 0:
            total_trans_diff += sample_trans_diff
            total_rot_diff += sample_rot_diff
            total_vel_diff += sample_vel_diff
    
    # det vs kalman_cv만 출력
    if 'det' in variant_order and 'kalman_cv' in variant_order:
        var1, var2 = 'det', 'kalman_cv'
        trans_diff = 0
        rot_diff = 0
        vel_diff = 0
        count = 0
        
        for sample_token in sample_tokens[:max_samples]:
            boxes1 = pred_variants[var1][sample_token]
            boxes2 = pred_variants[var2][sample_token]
            
            boxes1_dict = {(b.tracking_id, b.tracking_name): b for b in boxes1}
            boxes2_dict = {(b.tracking_id, b.tracking_name): b for b in boxes2}
            common_keys = set(boxes1_dict.keys()) & set(boxes2_dict.keys())
            
            for key in common_keys:
                b1 = boxes1_dict[key]
                b2 = boxes2_dict[key]
                
                trans_diff += np.linalg.norm(np.array(b1.translation) - np.array(b2.translation))
                if hasattr(b1, 'rotation') and hasattr(b2, 'rotation'):
                    rot_diff += quaternion_distance(b1.rotation, b2.rotation)
                if hasattr(b1, 'velocity') and hasattr(b2, 'velocity'):
                    vel_diff += velocity_distance(b1.velocity, b2.velocity)
                count += 1
        
        if count > 0:
            print(f"  {var1} vs {var2} (총 {count}개 박스, {samples_checked}개 샘플):")
            print(f"    translation_diff={trans_diff/count:.6f}, "
                  f"rotation_diff={rot_diff/count:.6f}, velocity_diff={vel_diff/count:.6f}")
            if rot_diff == 0 and vel_diff == 0:
                print(f"    ⚠️  모든 박스에서 rotation과 velocity가 동일합니다!")
        else:
            print(f"  {var1} vs {var2}: 매칭되는 박스 없음")
    print()


def _verify_boxes_vs_tracks(pred_variants, tracks_pred_variants, nusc, max_samples=5):
    """EvalBoxes와 Tracks에서 같은 샘플 비교하여 create_tracks가 제대로 복사했는지 확인"""
    if 'det' not in pred_variants or 'kalman_cv' not in pred_variants:
        return
    
    print(f"\n[디버깅] EvalBoxes vs Tracks 동일성 확인:")
    
    sample_tokens = pred_variants['det'].sample_tokens[:max_samples]
    
    for sample_token in sample_tokens:
        # EvalBoxes에서 가져오기
        boxes_det = pred_variants['det'][sample_token]
        boxes_kalman = pred_variants['kalman_cv'][sample_token]
        
        if len(boxes_det) == 0 or len(boxes_kalman) == 0:
            continue
        
        # Tracks에서 가져오기
        sample_record = nusc.get('sample', sample_token)
        scene_token = sample_record['scene_token']
        timestamp = sample_record['timestamp']
        
        tracks_det = tracks_pred_variants['det'][scene_token][timestamp]
        tracks_kalman = tracks_pred_variants['kalman_cv'][scene_token][timestamp]
        
        # tracking_id와 tracking_name으로 매칭
        boxes_det_dict = {(b.tracking_id, b.tracking_name): b for b in boxes_det}
        boxes_kalman_dict = {(b.tracking_id, b.tracking_name): b for b in boxes_kalman}
        tracks_det_dict = {(b.tracking_id, b.tracking_name): b for b in tracks_det}
        tracks_kalman_dict = {(b.tracking_id, b.tracking_name): b for b in tracks_kalman}
        
        common_keys = set(boxes_det_dict.keys()) & set(boxes_kalman_dict.keys()) & \
                      set(tracks_det_dict.keys()) & set(tracks_kalman_dict.keys())
        
        if len(common_keys) == 0:
            continue
        
        # 같은 tracking_id에 대해 EvalBoxes와 Tracks의 값이 같은지 확인
        rot_diff_in_boxes = 0
        vel_diff_in_boxes = 0
        rot_diff_in_tracks = 0
        vel_diff_in_tracks = 0
        count = 0
        
        for key in common_keys:
            b_det_box = boxes_det_dict[key]
            b_kalman_box = boxes_kalman_dict[key]
            b_det_track = tracks_det_dict[key]
            b_kalman_track = tracks_kalman_dict[key]
            
            # EvalBoxes에서의 차이
            if hasattr(b_det_box, 'rotation') and hasattr(b_kalman_box, 'rotation'):
                rot_diff_in_boxes += quaternion_distance(b_det_box.rotation, b_kalman_box.rotation)
            if hasattr(b_det_box, 'velocity') and hasattr(b_kalman_box, 'velocity'):
                vel_diff_in_boxes += velocity_distance(b_det_box.velocity, b_kalman_box.velocity)
            
            # Tracks에서의 차이
            if hasattr(b_det_track, 'rotation') and hasattr(b_kalman_track, 'rotation'):
                rot_diff_in_tracks += quaternion_distance(b_det_track.rotation, b_kalman_track.rotation)
            if hasattr(b_det_track, 'velocity') and hasattr(b_kalman_track, 'velocity'):
                vel_diff_in_tracks += velocity_distance(b_det_track.velocity, b_kalman_track.velocity)
            
            count += 1
        
        if count > 0:
            print(f"  Sample_token: {sample_token[:30]}... ({count}개 박스):")
            print(f"    EvalBoxes: rotation_diff={rot_diff_in_boxes/count:.6f}, velocity_diff={vel_diff_in_boxes/count:.6f}")
            print(f"    Tracks:    rotation_diff={rot_diff_in_tracks/count:.6f}, velocity_diff={vel_diff_in_tracks/count:.6f}")
            if abs(rot_diff_in_boxes - rot_diff_in_tracks) > 1e-6 or abs(vel_diff_in_boxes - vel_diff_in_tracks) > 1e-6:
                print(f"    ⚠️  EvalBoxes와 Tracks에서 차이가 다릅니다!")
    
    print()


def _verify_variant_differences(tracks_pred_variants, verbose=False, max_timestamps=20):
    """tracks 단계에서 variant별 데이터 차이 확인 (디버깅용)"""
    if not verbose:
        return
    
    variant_order = list(tracks_pred_variants.keys())
    if len(variant_order) < 2:
        return
    
    print(f"\n[디버깅] Tracks 단계 Variant 차이 확인:")
    
    # 여러 scene과 timestamp에서 샘플 수집
    all_scenes = list(tracks_pred_variants[variant_order[0]].keys())
    timestamps_checked = 0
    
    # det vs kalman_cv만 확인
    if 'det' not in variant_order or 'kalman_cv' not in variant_order:
        return
    
    var1, var2 = 'det', 'kalman_cv'
    total_trans_diff = 0
    total_rot_diff = 0
    total_vel_diff = 0
    total_count = 0
    
    # 각 scene에서 여러 timestamp 확인
    for scene_token in all_scenes[:5]:  # 최대 5개 scene
        timestamps = sorted(tracks_pred_variants[var1][scene_token].keys())
        
        for timestamp in timestamps[:max_timestamps//5 + 1]:  # scene당 여러 timestamp
            if timestamps_checked >= max_timestamps:
                break
            
            boxes1 = tracks_pred_variants[var1][scene_token][timestamp]
            boxes2 = tracks_pred_variants[var2][scene_token][timestamp]
            
            if len(boxes1) == 0 or len(boxes2) == 0:
                continue
            
            # tracking_id와 tracking_name으로 딕셔너리 생성하여 정확한 매칭
            boxes1_dict = {(b.tracking_id, b.tracking_name): b for b in boxes1}
            boxes2_dict = {(b.tracking_id, b.tracking_name): b for b in boxes2}
            
            common_keys = set(boxes1_dict.keys()) & set(boxes2_dict.keys())
            
            if len(common_keys) == 0:
                continue
            
            for key in common_keys:
                b1 = boxes1_dict[key]
                b2 = boxes2_dict[key]
                
                total_trans_diff += np.linalg.norm(np.array(b1.translation) - np.array(b2.translation))
                if hasattr(b1, 'rotation') and hasattr(b2, 'rotation'):
                    total_rot_diff += quaternion_distance(b1.rotation, b2.rotation)
                if hasattr(b1, 'velocity') and hasattr(b2, 'velocity'):
                    total_vel_diff += velocity_distance(b1.velocity, b2.velocity)
                total_count += 1
            
            timestamps_checked += 1
            if timestamps_checked >= max_timestamps:
                break
        
        if timestamps_checked >= max_timestamps:
            break
    
    if total_count > 0:
        print(f"  {var1} vs {var2} (총 {total_count}개 박스, {timestamps_checked}개 timestamp):")
        print(f"    translation_diff={total_trans_diff/total_count:.6f}, "
              f"rotation_diff={total_rot_diff/total_count:.6f}, velocity_diff={total_vel_diff/total_count:.6f}")
        if total_rot_diff == 0 and total_vel_diff == 0:
            print(f"    ⚠️  모든 박스에서 rotation과 velocity가 동일합니다!")
        
        # 차이가 있는 timestamp 찾기
        if total_rot_diff == 0 and total_vel_diff == 0:
            print(f"  → 차이가 없는 이유를 확인하기 위해 더 많은 timestamp를 확인합니다...")
            # 첫 번째 차이가 있는 timestamp 찾기
            found_diff = False
            for scene_token in all_scenes:
                if found_diff:
                    break
                timestamps = sorted(tracks_pred_variants[var1][scene_token].keys())
                for timestamp in timestamps:
                    boxes1 = tracks_pred_variants[var1][scene_token][timestamp]
                    boxes2 = tracks_pred_variants[var2][scene_token][timestamp]
                    boxes1_dict = {(b.tracking_id, b.tracking_name): b for b in boxes1}
                    boxes2_dict = {(b.tracking_id, b.tracking_name): b for b in boxes2}
                    common_keys = set(boxes1_dict.keys()) & set(boxes2_dict.keys())
                    
                    for key in common_keys:
                        b1 = boxes1_dict[key]
                        b2 = boxes2_dict[key]
                        if hasattr(b1, 'rotation') and hasattr(b2, 'rotation'):
                            rot_diff = quaternion_distance(b1.rotation, b2.rotation)
                            if rot_diff > 1e-6:
                                print(f"    차이가 있는 timestamp 발견: Scene={scene_token[:30]}..., Timestamp={timestamp}")
                                print(f"      tracking_id={key[0]}, rotation_diff={rot_diff:.6f}")
                                found_diff = True
                                break
                        if hasattr(b1, 'velocity') and hasattr(b2, 'velocity'):
                            vel_diff = velocity_distance(b1.velocity, b2.velocity)
                            if vel_diff > 1e-6:
                                print(f"    차이가 있는 timestamp 발견: Scene={scene_token[:30]}..., Timestamp={timestamp}")
                                print(f"      tracking_id={key[0]}, velocity_diff={vel_diff:.6f}")
                                found_diff = True
                                break
                    if found_diff:
                        break
    else:
        print(f"  {var1} vs {var2}: 매칭되는 박스 없음")
    print()


def _compute_overall_metrics_for_variant_serial(variant_name: str, tracks_gt, tracks_pred_variants, cfg):
    """직렬 실행용 variant 메트릭 계산"""
    tracks_pred = tracks_pred_variants[variant_name]
    _, metrics_summary = _compute_overall_metrics_for_variant_mp(
        variant_name, tracks_gt, tracks_pred, cfg, disable_class_tqdm=False
    )
    return variant_name, metrics_summary


def _compute_overall_metrics_parallel(tracks_gt, tracks_pred_variants, cfg, verbose, step_info):
    """변이별 전체 메트릭을 병렬로 계산"""
    step, total_steps = step_info
    variant_order = [v for v in VARIANT_NAMES if v in tracks_pred_variants]
    overall_metrics = {}
    
    # Variant 차이 확인 (디버깅) - 여러 timestamp 확인
    _verify_variant_differences(tracks_pred_variants, verbose, max_timestamps=50)
    
    if len(variant_order) > 1:
        # 병렬 실행
        _set_thread_limits()
        
        # 전역 참조 설정 (fork에서는 복사됨)
        global _GLOBAL_tracks_gt, _GLOBAL_tracks_pred_variants, _GLOBAL_cfg
        _GLOBAL_tracks_gt = tracks_gt
        _GLOBAL_tracks_pred_variants = tracks_pred_variants
        _GLOBAL_cfg = cfg
        
        with mp.get_context('fork').Pool(processes=min(len(variant_order), os.cpu_count() or 1)) as pool:
            for variant_name, metrics_summary in tqdm(
                    pool.imap_unordered(_compute_overall_metrics_for_variant_by_name, variant_order),
                    total=len(variant_order), disable=False, desc=f"[{step}/{total_steps}] Overall metrics"
            ):
                overall_metrics[variant_name] = metrics_summary
    else:
        # 직렬 실행
        for variant_name in tqdm(variant_order, disable=False, desc=f"[{step}/{total_steps}] Overall metrics"):
            _, metrics_summary = _compute_overall_metrics_for_variant_serial(
                variant_name, tracks_gt, tracks_pred_variants, cfg
            )
            overall_metrics[variant_name] = metrics_summary
    
    return overall_metrics


def evaluate_trackwise(
    result_path: str,
    nusc_dataroot: str,
    eval_set: str = "val",
    nusc_version: str = "v1.0-trainval",
    output_path: str = None,
    verbose: bool = True,
    max_scenes: int = None
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
        max_scenes: 최대 평가할 씬 수 (None이면 전체 평가, 디버깅용)
    """
    # 전체 단계 수 정의
    TOTAL_STEPS = 8
    current_step = 1
    
    if verbose:
        print("Trackwise 평가 시작...")
        if max_scenes is not None:
            print(f"[디버깅 모드] 최대 {max_scenes}개 씬만 평가합니다.")
    
    # 출력 경로 설정
    if output_path is None:
        output_dir = os.path.dirname(result_path)
        output_path = os.path.join(output_dir, "trackwise_evaluation.json")
    
    # Config 로드
    cfg = track_configs("tracking_nips_2019")
    
    # 데이터 로드 및 전처리 (단계 1-5)
    tracks_gt, tracks_pred_variants, meta = _load_and_preprocess_data(
        result_path, nusc_dataroot, eval_set, nusc_version, cfg, verbose, (current_step, TOTAL_STEPS)
    )
    
    # 씬 제한 적용 (디버깅용)
    if max_scenes is not None:
        tracks_gt, tracks_pred_variants = _filter_tracks_by_scenes(tracks_gt, tracks_pred_variants, max_scenes)
        if verbose:
            print(f"[디버깅] {len(tracks_gt)}개 씬으로 제한되었습니다.")
    
    current_step = 6  # 전처리 완료 후 다음 단계
    
    # 씬별 평가 (병렬) - 단계 6
    if verbose:
        print(f"[{current_step}/{TOTAL_STEPS}] 각 트랙별 평가 수행 중...")
    trackwise_results = _evaluate_scenes_parallel(tracks_gt, tracks_pred_variants, cfg, (current_step, TOTAL_STEPS))
    current_step += 1
    
    # 전체 메트릭 계산 (병렬) - 단계 7
    if verbose:
        print(f"[{current_step}/{TOTAL_STEPS}] nuScenes 메트릭(전체/클래스별) 계산 중...")
    overall_metrics = _compute_overall_metrics_parallel(tracks_gt, tracks_pred_variants, cfg, verbose, (current_step, TOTAL_STEPS))
    current_step += 1
    
    # 결과 저장 - 단계 8
    if verbose:
        print(f"[{current_step}/{TOTAL_STEPS}] 결과 저장 중: {output_path}")
    
    output_data = {
        'meta': meta,
        'trackwise_results': trackwise_results,
        'overall_metrics': overall_metrics,
        'summary': {
            'total_tracks': len(trackwise_results),
            'variants_evaluated': VARIANT_NAMES
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
    parser.add_argument('--result_path', type=str, required=True, help='결과 JSON 파일 경로')
    parser.add_argument('--nusc_dataroot', type=str, default='data/nuscenes/datasets', help='nuScenes 데이터셋 루트 경로')
    parser.add_argument('--eval_set', type=str, default='val', help='평가할 데이터셋 split')
    parser.add_argument('--nusc_version', type=str, default='v1.0-trainval', help='nuScenes 버전')
    parser.add_argument('--output_path', type=str, default=None, help='출력 JSON 파일 경로')
    parser.add_argument('--verbose', type=int, default=1, help='상세 출력 여부')
    parser.add_argument('--max_scenes', type=int, default=None, help='최대 평가할 씬 수 (디버깅용, None이면 전체)')
    
    args = parser.parse_args()
    
    evaluate_trackwise(
        result_path=args.result_path,
        nusc_dataroot=args.nusc_dataroot,
        eval_set=args.eval_set,
        nusc_version=args.nusc_version,
        output_path=args.output_path,
        verbose=bool(args.verbose),
        max_scenes=args.max_scenes
    )
