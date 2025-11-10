import json
import numpy as np
import argparse
from collections import defaultdict


def quaternion_distance(q1, q2):
    """두 쿼터니언 간의 각도 차이 계산 (라디안)"""
    q1 = np.array(q1)
    q2 = np.array(q2)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2 * np.arccos(dot)
    return angle


def velocity_distance(v1, v2):
    """두 속도 벡터 간의 유클리드 거리"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.linalg.norm(v1 - v2)


def check_tracking_result(result_path: str, max_samples: int = 100, verbose: bool = True):
    """
    트래킹 결과 파일에서 det와 kalman_cv의 rotation, velocity 차이 확인
    
    Args:
        result_path: 트래킹 결과 JSON 파일 경로
        max_samples: 확인할 최대 샘플 수 (None이면 전체)
        verbose: 상세 출력 여부
    """
    print(f"트래킹 결과 파일 확인: {result_path}")
    
    with open(result_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    sample_tokens = list(results.keys())
    
    if max_samples is not None:
        sample_tokens = sample_tokens[:max_samples]
    
    print(f"총 {len(results)}개 샘플 중 {len(sample_tokens)}개 확인\n")
    
    # 통계 수집
    rotation_identical_count = 0
    rotation_different_count = 0
    velocity_identical_count = 0
    velocity_different_count = 0
    
    rotation_diffs = []
    velocity_diffs = []
    
    samples_with_diffs = []
    
    for sample_token in sample_tokens:
        detections = results[sample_token]
        
        for detection in detections:
            # 필수 필드 존재 확인
            has_rotation_det = 'rotation_det' in detection
            has_rotation_kalman = 'rotation_kalman_cv' in detection
            has_velocity_det = 'velocity_det' in detection
            has_velocity_kalman = 'velocity_kalman_cv' in detection
            
            if not (has_rotation_det and has_rotation_kalman):
                if verbose:
                    print(f"[경고] sample_token={sample_token[:20]}... rotation 필드 누락")
                continue
            
            if not (has_velocity_det and has_velocity_kalman):
                if verbose:
                    print(f"[경고] sample_token={sample_token[:20]}... velocity 필드 누락")
                continue
            
            # Rotation 비교
            rot_det = detection['rotation_det']
            rot_kalman = detection['rotation_kalman_cv']
            
            # 정확히 동일한지 확인
            rot_exact_match = rot_det == rot_kalman
            # 쿼터니언 거리 계산
            rot_diff = quaternion_distance(rot_det, rot_kalman)
            
            if rot_exact_match:
                rotation_identical_count += 1
            else:
                rotation_different_count += 1
                rotation_diffs.append(rot_diff)
            
            # Velocity 비교
            vel_det = detection['velocity_det']
            vel_kalman = detection['velocity_kalman_cv']
            
            # 정확히 동일한지 확인
            vel_exact_match = vel_det == vel_kalman
            # 유클리드 거리 계산
            vel_diff = velocity_distance(vel_det, vel_kalman)
            
            if vel_exact_match:
                velocity_identical_count += 1
            else:
                velocity_different_count += 1
                velocity_diffs.append(vel_diff)
            
            # 차이가 있는 샘플 기록
            if not rot_exact_match or not vel_exact_match or rot_diff > 1e-6 or vel_diff > 1e-6:
                samples_with_diffs.append({
                    'sample_token': sample_token,
                    'tracking_id': detection.get('tracking_id', 'unknown'),
                    'tracking_name': detection.get('tracking_name', 'unknown'),
                    'rotation_exact_match': rot_exact_match,
                    'rotation_diff': rot_diff,
                    'velocity_exact_match': vel_exact_match,
                    'velocity_diff': vel_diff,
                    'rotation_det': rot_det,
                    'rotation_kalman_cv': rot_kalman,
                    'velocity_det': vel_det,
                    'velocity_kalman_cv': vel_kalman
                })
    
    # 결과 출력
    total_detections = rotation_identical_count + rotation_different_count
    
    print("=" * 80)
    print("검사 결과 요약")
    print("=" * 80)
    print(f"총 검사한 detection 수: {total_detections}")
    print()
    
    print("Rotation 비교:")
    print(f"  동일한 경우: {rotation_identical_count} ({rotation_identical_count/total_detections*100:.2f}%)")
    print(f"  다른 경우: {rotation_different_count} ({rotation_different_count/total_detections*100:.2f}%)")
    if rotation_diffs:
        print(f"  평균 차이: {np.mean(rotation_diffs):.6f} 라디안")
        print(f"  최대 차이: {np.max(rotation_diffs):.6f} 라디안")
        print(f"  최소 차이: {np.min(rotation_diffs):.6f} 라디안")
    print()
    
    print("Velocity 비교:")
    print(f"  동일한 경우: {velocity_identical_count} ({velocity_identical_count/total_detections*100:.2f}%)")
    print(f"  다른 경우: {velocity_different_count} ({velocity_different_count/total_detections*100:.2f}%)")
    if velocity_diffs:
        print(f"  평균 차이: {np.mean(velocity_diffs):.6f} m/s")
        print(f"  최대 차이: {np.max(velocity_diffs):.6f} m/s")
        print(f"  최소 차이: {np.min(velocity_diffs):.6f} m/s")
    print()
    
    # 차이가 있는 샘플 상세 출력
    if samples_with_diffs:
        print("=" * 80)
        print(f"차이가 발견된 샘플 (최대 10개 출력)")
        print("=" * 80)
        
        for i, diff_info in enumerate(samples_with_diffs[:10]):
            print(f"\n[{i+1}] sample_token: {diff_info['sample_token'][:30]}...")
            print(f"    tracking_id: {diff_info['tracking_id']}, class: {diff_info['tracking_name']}")
            
            if not diff_info['rotation_exact_match'] or diff_info['rotation_diff'] > 1e-6:
                print(f"    Rotation 차이: {diff_info['rotation_diff']:.6f} 라디안")
                print(f"      det:        {diff_info['rotation_det']}")
                print(f"      kalman_cv:  {diff_info['rotation_kalman_cv']}")
            
            if not diff_info['velocity_exact_match'] or diff_info['velocity_diff'] > 1e-6:
                print(f"    Velocity 차이: {diff_info['velocity_diff']:.6f} m/s")
                print(f"      det:        {diff_info['velocity_det']}")
                print(f"      kalman_cv:  {diff_info['velocity_kalman_cv']}")
        
        if len(samples_with_diffs) > 10:
            print(f"\n... 외 {len(samples_with_diffs) - 10}개 샘플에서 차이 발견")
    else:
        print("=" * 80)
        print("모든 샘플에서 rotation과 velocity가 동일합니다.")
        print("=" * 80)
    
    # 문제 분석
    print()
    print("=" * 80)
    print("문제 분석")
    print("=" * 80)
    
    if rotation_identical_count == total_detections:
        print("⚠️  [문제] 모든 detection에서 rotation_det과 rotation_kalman_cv가 동일합니다!")
        print("   → 이는 데이터 생성 시 kalman_cv rotation이 제대로 저장되지 않았을 수 있습니다.")
    else:
        print("✓ Rotation: det와 kalman_cv 사이에 차이가 있습니다 (정상)")
    
    if velocity_identical_count == total_detections:
        print("⚠️  [문제] 모든 detection에서 velocity_det과 velocity_kalman_cv가 동일합니다!")
        print("   → 이는 데이터 생성 시 kalman_cv velocity가 제대로 저장되지 않았을 수 있습니다.")
    else:
        print("✓ Velocity: det와 kalman_cv 사이에 차이가 있습니다 (정상)")
    
    # 통계: 차이가 있는 샘플 비율
    if samples_with_diffs:
        unique_samples = len(set(s['sample_token'] for s in samples_with_diffs))
        print(f"\n차이가 있는 고유 샘플 수: {unique_samples} / {len(sample_tokens)} ({unique_samples/len(sample_tokens)*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='트래킹 결과 파일에서 det vs kalman_cv 차이 확인')
    parser.add_argument('result_path', type=str, help='트래킹 결과 JSON 파일 경로')
    parser.add_argument('--max_samples', type=int, default=100, help='확인할 최대 샘플 수 (기본: 100, 전체: -1)')
    parser.add_argument('--verbose', type=int, default=1, help='상세 출력 여부 (1: 켜기, 0: 끄기)')
    
    args = parser.parse_args()
    
    max_samples = None if args.max_samples == -1 else args.max_samples
    
    check_tracking_result(
        result_path=args.result_path,
        max_samples=max_samples,
        verbose=bool(args.verbose)
    )

