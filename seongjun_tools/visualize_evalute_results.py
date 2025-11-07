import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_trackwise_dataframe(result_json_path: str, variant: str) -> pd.DataFrame:
    """
    evaluate_trackwise.py 또는 evaluate_trackwise_parallel.py가 생성한
    trackwise_evaluation.json 파일을 읽어서 데이터프레임으로 변환
    
    Args:
        result_json_path: trackwise_evaluation.json 파일 경로
        variant: 시각화할 variant 이름 (det, kalman_cv, diff, curve)
    
    Returns:
        pandas.DataFrame: track별 메트릭 데이터
    """
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    trackwise = data.get("trackwise_results", {})
    
    if not trackwise:
        raise ValueError(f"trackwise_results가 비어있습니다. 파일 형식을 확인해주세요: {result_json_path}")
    
    for track_key, t in trackwise.items():
        class_name = t.get("class_name")
        track_length_gt = t.get("track_length_gt")
        variants = t.get("variants", {})
        v = variants.get(variant, {})

        # mota 처리: None이거나 없으면 np.nan으로 변환
        mota = v.get("mota")
        if mota is None:
            mota = np.nan
        elif isinstance(mota, (int, float)):
            if np.isnan(float(mota)):
                mota = np.nan
            else:
                mota = float(mota)
        else:
            # 문자열 등 예상치 못한 타입
            try:
                mota = float(mota) if mota is not None else np.nan
            except (ValueError, TypeError):
                mota = np.nan
        
        # track_length_pred 처리: None이거나 없으면 np.nan으로 변환
        track_length_pred = v.get("track_length_pred")
        if track_length_pred is None:
            track_length_pred = np.nan
        elif isinstance(track_length_pred, (int, float)):
            if np.isnan(float(track_length_pred)):
                track_length_pred = np.nan
            else:
                track_length_pred = int(track_length_pred)
        else:
            # 문자열 등 예상치 못한 타입
            try:
                track_length_pred = int(track_length_pred) if track_length_pred is not None else np.nan
            except (ValueError, TypeError):
                track_length_pred = np.nan

        records.append(
            {
                "track_key": track_key,
                "class_name": class_name,
                "track_length_gt": track_length_gt if track_length_gt is not None else np.nan,
                "track_length_pred": track_length_pred,
                "mota": mota,
            }
        )

    df = pd.DataFrame.from_records(records)
    
    if len(df) == 0:
        raise ValueError(f"variant '{variant}'에 대한 데이터가 없습니다. 사용 가능한 variant를 확인해주세요.")
    
    return df


def load_overall_mota(result_json_path: str, variant: str) -> float:
    """
    overall_metrics에서 variant별 전체 MOTA 값을 가져옴
    
    Args:
        result_json_path: trackwise_evaluation.json 파일 경로
        variant: variant 이름 (det, kalman_cv, diff, curve)
    
    Returns:
        float: 전체 MOTA 값 (없으면 None)
    """
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    overall_metrics = data.get("overall_metrics", {})
    variant_metrics = overall_metrics.get(variant, {})
    mota = variant_metrics.get("mota")
    
    if mota is not None:
        try:
            return float(mota)
        except (ValueError, TypeError):
            return None
    return None


def load_all_variants_metrics(result_json_path: str) -> pd.DataFrame:
    """
    모든 variant의 overall metrics를 로드하여 데이터프레임으로 변환
    
    Args:
        result_json_path: trackwise_evaluation.json 파일 경로
    
    Returns:
        pandas.DataFrame: variant별 메트릭 데이터
    """
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    overall_metrics = data.get("overall_metrics", {})
    variants = ['det', 'kalman_cv', 'diff', 'curve']
    
    records = []
    for variant in variants:
        if variant not in overall_metrics:
            continue
        
        variant_data = overall_metrics[variant]
        record = {'variant': variant}
        
        # 주요 메트릭 추출
        for metric in ['mota', 'amota', 'amotp', 'recall', 'motar']:
            value = variant_data.get(metric)
            if value is not None:
                try:
                    record[metric] = float(value)
                except (ValueError, TypeError):
                    record[metric] = np.nan
            else:
                record[metric] = np.nan
        
        # Rotation 및 Velocity errors 추출
        rotation_errors = variant_data.get('rotation_errors', {})
        velocity_errors = variant_data.get('velocity_errors', {})
        
        # 'all' 키가 있으면 전체 평균 사용
        if 'all' in rotation_errors:
            rot_all = rotation_errors['all']
            record['rotation_error_mean'] = rot_all.get('mean', np.nan)
            record['rotation_error_std'] = rot_all.get('std', np.nan)
            record['rotation_error_median'] = rot_all.get('median', np.nan)
        else:
            record['rotation_error_mean'] = np.nan
            record['rotation_error_std'] = np.nan
            record['rotation_error_median'] = np.nan
        
        if 'all' in velocity_errors:
            vel_all = velocity_errors['all']
            record['velocity_error_mean'] = vel_all.get('mean', np.nan)
            record['velocity_error_std'] = vel_all.get('std', np.nan)
            record['velocity_error_median'] = vel_all.get('median', np.nan)
        else:
            record['velocity_error_mean'] = np.nan
            record['velocity_error_std'] = np.nan
            record['velocity_error_median'] = np.nan
        
        records.append(record)
    
    df = pd.DataFrame.from_records(records)
    return df


def ensure_outdir(outdir: str):
    if outdir is None or len(outdir) == 0:
        return None
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save_or_show(fig: plt.Figure, outdir: str, filename: str, show: bool):
    if outdir:
        path = os.path.join(outdir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_mota_hist(df: pd.DataFrame, outdir: str, show: bool, variant: str, mota_ylim: tuple = None, bins: int = 100, overall_mota: float = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["mota"].dropna(), bins=bins, kde=False, ax=ax)
    ax.set_title(f"MOTA Histogram ({variant})")
    ax.set_xlabel("MOTA")
    ax.set_ylabel("Count")
    if mota_ylim is not None:
        ax.set_xlim(mota_ylim[0], mota_ylim[1])
    
    # Overall MOTA 수직선 표시
    if overall_mota is not None:
        ax.axvline(x=overall_mota, color='r', linestyle='-', linewidth=2, zorder=10, label=f'Overall MOTA: {overall_mota:.4f}')
        ax.legend()
    
    save_or_show(fig, outdir, f"mota_hist_{variant}.png", show)


def plot_mota_by_gt_len(df: pd.DataFrame, outdir: str, show: bool, variant: str, alpha: float = 0.7, mota_ylim: tuple = None, overall_mota: float = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df, x="track_length_gt", y="mota", s=12, ax=ax, linewidth=0, alpha=alpha
    )
    ax.set_title(f"MOTA vs GT Track Length ({variant})")
    ax.set_xlabel("GT Track Length (frames)")
    ax.set_ylabel("MOTA")
    if mota_ylim is not None:
        ax.set_ylim(mota_ylim[0], mota_ylim[1])
    
    # Overall MOTA 수평선 표시
    if overall_mota is not None:
        xlim = ax.get_xlim()
        ax.axhline(y=overall_mota, color='r', linestyle='-', linewidth=2, zorder=10, label=f'Overall MOTA: {overall_mota:.4f}')
        ax.set_xlim(xlim)
        ax.legend()
    
    save_or_show(fig, outdir, f"mota_by_gt_length_{variant}.png", show)


def plot_mota_by_class_scatter(df: pd.DataFrame, outdir: str, show: bool, variant: str, alpha: float = 0.7, mota_ylim: tuple = None, overall_mota: float = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    classes = sorted(df["class_name"].dropna().unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    x_vals = df["class_name"].map(class_to_idx)
    ax.scatter(x_vals, df["mota"], s=10, alpha=alpha)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_title(f"MOTA by Class (scatter) ({variant})")
    ax.set_xlabel("Class")
    ax.set_ylabel("MOTA")
    if mota_ylim is not None:
        ax.set_ylim(mota_ylim[0], mota_ylim[1])
    
    # Overall MOTA 수평선 표시
    if overall_mota is not None:
        xlim = ax.get_xlim()
        ax.axhline(y=overall_mota, color='r', linestyle='-', linewidth=2, zorder=10, label=f'Overall MOTA: {overall_mota:.4f}')
        ax.set_xlim(xlim)
        ax.legend()
    
    fig.tight_layout()
    save_or_show(fig, outdir, f"mota_by_class_scatter_{variant}.png", show)


def plot_mota_by_class_box(df: pd.DataFrame, outdir: str, show: bool, variant: str, mota_ylim: tuple = None, overall_mota: float = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="class_name", y="mota", ax=ax, fliersize=1)
    ax.set_title(f"MOTA by Class (box) ({variant})")
    ax.set_xlabel("Class")
    ax.set_ylabel("MOTA")
    if mota_ylim is not None:
        ax.set_ylim(mota_ylim[0], mota_ylim[1])
    
    # Overall MOTA 수평선 표시
    if overall_mota is not None:
        xlim = ax.get_xlim()
        ax.axhline(y=overall_mota, color='r', linestyle='-', linewidth=2, zorder=10, label=f'Overall MOTA: {overall_mota:.4f}')
        ax.set_xlim(xlim)
        ax.legend()
    
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    save_or_show(fig, outdir, f"mota_by_class_box_{variant}.png", show)


def plot_pred_len_by_class_scatter(df: pd.DataFrame, outdir: str, show: bool, variant: str, alpha: float = 0.7):
    fig, ax = plt.subplots(figsize=(9, 5))
    classes = sorted(df["class_name"].dropna().unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    x_vals = df["class_name"].map(class_to_idx)
    ax.scatter(x_vals, df["track_length_pred"], s=10, alpha=alpha)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_title(f"Pred Track Length by Class (scatter) ({variant})")
    ax.set_xlabel("Class")
    ax.set_ylabel("Pred Track Length (frames)")
    fig.tight_layout()
    save_or_show(fig, outdir, f"predlen_by_class_scatter_{variant}.png", show)


def plot_pred_len_by_gt_len_scatter(df: pd.DataFrame, outdir: str, show: bool, variant: str, alpha: float = 0.7):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df, x="track_length_gt", y="track_length_pred", s=12, ax=ax, linewidth=0, alpha=alpha
    )
    ax.set_title(f"Pred Track Length vs GT Track Length ({variant})")
    ax.set_xlabel("GT Track Length (frames)")
    ax.set_ylabel("Pred Track Length (frames)")
    save_or_show(fig, outdir, f"predlen_by_gtlen_{variant}.png", show)


def plot_variant_comparison(result_json_path: str, outdir: str, show: bool):
    """
    variant별 overall 성능 비교 플롯
    
    Args:
        result_json_path: trackwise_evaluation.json 파일 경로
        outdir: 출력 디렉토리
        show: 화면 표시 여부
    """
    df = load_all_variants_metrics(result_json_path)
    
    if len(df) == 0:
        print("Warning: No variant metrics found for comparison")
        return
    
    # Variant 순서 설정
    variant_order = ['det', 'kalman_cv', 'diff', 'curve']
    df['variant'] = pd.Categorical(df['variant'], categories=variant_order, ordered=True)
    df = df.sort_values('variant')
    
    # Rotation Error 비교 (det와 kalman_cv만)
    df_rot = df[df['variant'].isin(['det', 'kalman_cv'])].copy()
    
    print(f"[Variant Comparison] Loaded {len(df)} variants, {len(df_rot)} for rotation comparison")
    
    # 데이터 확인 및 출력
    has_rot_data = len(df_rot) > 0 and 'rotation_error_mean' in df_rot.columns and not df_rot['rotation_error_mean'].isna().all()
    has_vel_data = 'velocity_error_mean' in df.columns and not df['velocity_error_mean'].isna().all()
    
    if not has_rot_data and not has_vel_data:
        print("[Variant Comparison] Warning: No rotation or velocity error data found in overall_metrics.")
        print("  This may happen if the evaluation was run with an older version of evaluate_trackwise_parallel.py")
        print("  that did not compute rotation/velocity errors.")
        print("  To generate these plots, please re-run evaluation with the latest version.")
        return
    
    if not has_rot_data:
        print("[Variant Comparison] Warning: No rotation error data found. Skipping rotation error plots.")
    if not has_vel_data:
        print("[Variant Comparison] Warning: No velocity error data found. Skipping velocity error plots.")
    
    # Rotation 및 Velocity Error 함께 비교 (Line plot)
    # Rotation은 det와 kalman_cv만, Velocity는 모든 variant
    if (len(df_rot) > 0 and 'rotation_error_mean' in df_rot.columns and not df_rot['rotation_error_mean'].isna().all() and
        'velocity_error_mean' in df.columns and not df['velocity_error_mean'].isna().all()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Rotation Error (det와 kalman_cv만)
        x_pos_rot = np.arange(len(df_rot))
        ax1.plot(x_pos_rot, df_rot['rotation_error_mean'].values, marker='o', linewidth=2, markersize=8, label='Mean')
        if 'rotation_error_median' in df_rot.columns:
            ax1.plot(x_pos_rot, df_rot['rotation_error_median'].values, marker='s', linewidth=2, markersize=8, label='Median', linestyle='--')
        ax1.set_xticks(x_pos_rot)
        ax1.set_xticklabels(df_rot['variant'].values)
        ax1.set_xlabel('Variant')
        ax1.set_ylabel('Rotation Error (radians)')
        ax1.set_title('Rotation Error Comparison (det vs kalman_cv)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # 값 표시
        for j, val in enumerate(df_rot['rotation_error_mean'].values):
            if not np.isnan(val):
                ax1.text(j, val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Velocity Error (모든 variant)
        x_pos_vel = np.arange(len(df))
        ax2.plot(x_pos_vel, df['velocity_error_mean'].values, marker='o', linewidth=2, markersize=8, label='Mean')
        if 'velocity_error_median' in df.columns:
            ax2.plot(x_pos_vel, df['velocity_error_median'].values, marker='s', linewidth=2, markersize=8, label='Median', linestyle='--')
        ax2.set_xticks(x_pos_vel)
        ax2.set_xticklabels(df['variant'].values)
        ax2.set_xlabel('Variant')
        ax2.set_ylabel('Velocity Error (m/s)')
        ax2.set_title('Velocity Error Comparison')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        # 값 표시
        for j, val in enumerate(df['velocity_error_mean'].values):
            if not np.isnan(val):
                ax2.text(j, val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_or_show(fig, outdir, "variant_comparison_rotation_velocity_errors.png", show)
        print("[Variant Comparison] Created rotation_velocity_errors combined plot")
    else:
        if len(df_rot) == 0:
            print(f"[Variant Comparison] Skipping combined plot: No det/kalman_cv variants found")
        elif 'rotation_error_mean' not in df_rot.columns or df_rot['rotation_error_mean'].isna().all():
            print(f"[Variant Comparison] Skipping combined plot: rotation_error_mean data not available")
        elif 'velocity_error_mean' not in df.columns or df['velocity_error_mean'].isna().all():
            print(f"[Variant Comparison] Skipping combined plot: velocity_error_mean data not available")


def main():
    parser = argparse.ArgumentParser(description="Trackwise 평가 결과 시각화")
    parser.add_argument(
        "result_json",
        nargs="?",
        default="/3dmot_ws/MCTrack/prsys_results/eval_code_test/trackwise_evaluation.json",
        type=str,
        help="evaluate_trackwise.py 또는 evaluate_trackwise_parallel.py가 생성한 trackwise_evaluation.json 경로",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="kalman_cv",
        choices=["det", "kalman_cv", "diff", "curve"],
        help="시각화에 사용할 variant",
    )
    parser.add_argument(
        "--class_filter",
        type=str,
        nargs="*",
        default=None,
        help="특정 클래스만 포함 (공백 구분 다중 지정)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="그림 저장 디렉토리 (미지정 시 화면 표시만)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="화면 표시 생략 (파일 저장만)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="scatter plot 점들의 투명도 (0.0 ~ 1.0, 기본값: 0.4)",
    )
    parser.add_argument(
        "--mota_ylim",
        type=float,
        nargs=2,
        default=[-1.1, 1.1],
        metavar=("MIN", "MAX"),
        help="MOTA 표시 범위 (min max, 기본값: -5.0 1.1). 예: --mota_ylim -5.0 1.1",
    )
    parser.add_argument(
        "--hist_bins",
        type=int,
        default=500,
        help="히스토그램 분해능 (bins 수, 기본값: 100)",
    )

    args = parser.parse_args()

    sns.set(style="whitegrid")

    outdir = ensure_outdir(args.outdir)
    
    # 데이터 로드
    df = load_trackwise_dataframe(args.result_json, args.variant)
    
    if args.class_filter:
        df = df[df["class_name"].isin(args.class_filter)].reset_index(drop=True)

    # Overall MOTA 로드
    overall_mota = load_overall_mota(args.result_json, args.variant)
    if overall_mota is not None:
        print(f"Overall MOTA ({args.variant}): {overall_mota:.4f}")
    else:
        print(f"Warning: Overall MOTA not found for variant '{args.variant}'")

    # 결측치 정리 (플롯에 방해되는 NaN 제거는 각 플롯에서 처리되지만, 전반적으로 정리)
    # 여기서는 그대로 두고, 각 플롯 함수에서 dropna 처리

    show = not args.no_show and (outdir is None)
    
    # MOTA y축 범위 설정
    mota_ylim = tuple(args.mota_ylim)
    
    plot_mota_hist(df, outdir, show, args.variant, mota_ylim=mota_ylim, bins=args.hist_bins, overall_mota=overall_mota)
    plot_mota_by_gt_len(df, outdir, show, args.variant, alpha=args.alpha, mota_ylim=mota_ylim, overall_mota=overall_mota)
    plot_mota_by_class_scatter(df, outdir, show, args.variant, alpha=args.alpha, mota_ylim=mota_ylim, overall_mota=overall_mota)
    plot_mota_by_class_box(df, outdir, show, args.variant, mota_ylim=mota_ylim, overall_mota=overall_mota)
    plot_pred_len_by_class_scatter(df, outdir, show, args.variant, alpha=args.alpha)
    plot_pred_len_by_gt_len_scatter(df, outdir, show, args.variant, alpha=args.alpha)
    
    # Variant별 비교 플롯
    plot_variant_comparison(args.result_json, outdir, show)


if __name__ == "__main__":
    main()



