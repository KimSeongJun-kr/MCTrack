import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_trackwise_dataframe(result_json_path: str, variant: str) -> pd.DataFrame:
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    trackwise = data.get("trackwise_results", {})
    for track_key, t in trackwise.items():
        class_name = t.get("class_name")
        track_length_gt = t.get("track_length_gt")
        variants = t.get("variants", {})
        v = variants.get(variant, {})

        mota = v.get("mota", np.nan)
        track_length_pred = v.get("track_length_pred", np.nan)

        records.append(
            {
                "track_key": track_key,
                "class_name": class_name,
                "track_length_gt": track_length_gt,
                "track_length_pred": track_length_pred,
                "mota": mota,
            }
        )

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


def plot_mota_hist(df: pd.DataFrame, outdir: str, show: bool, variant: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["mota"].dropna(), bins=30, kde=False, ax=ax)
    ax.set_title(f"MOTA Histogram ({variant})")
    ax.set_xlabel("MOTA")
    ax.set_ylabel("Count")
    save_or_show(fig, outdir, f"mota_hist_{variant}.png", show)


def plot_mota_by_gt_len(df: pd.DataFrame, outdir: str, show: bool, variant: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df, x="track_length_gt", y="mota", s=12, ax=ax, linewidth=0
    )
    ax.set_title(f"MOTA vs GT Track Length ({variant})")
    ax.set_xlabel("GT Track Length (frames)")
    ax.set_ylabel("MOTA")
    save_or_show(fig, outdir, f"mota_by_gt_length_{variant}.png", show)


def plot_mota_by_class_scatter(df: pd.DataFrame, outdir: str, show: bool, variant: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    # jitter를 위해 x를 카테고리 index로 매핑 후 소량 노이즈 추가
    classes = sorted(df["class_name"].dropna().unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    x_vals = df["class_name"].map(class_to_idx)
    jitter = (np.random.rand(len(x_vals)) - 0.5) * 0.2
    ax.scatter(x_vals + jitter, df["mota"], s=10, alpha=0.7)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_title(f"MOTA by Class (scatter) ({variant})")
    ax.set_xlabel("Class")
    ax.set_ylabel("MOTA")
    fig.tight_layout()
    save_or_show(fig, outdir, f"mota_by_class_scatter_{variant}.png", show)


def plot_mota_by_class_box(df: pd.DataFrame, outdir: str, show: bool, variant: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="class_name", y="mota", ax=ax, fliersize=1)
    ax.set_title(f"MOTA by Class (box) ({variant})")
    ax.set_xlabel("Class")
    ax.set_ylabel("MOTA")
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    save_or_show(fig, outdir, f"mota_by_class_box_{variant}.png", show)


def plot_pred_len_by_class_scatter(df: pd.DataFrame, outdir: str, show: bool, variant: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    classes = sorted(df["class_name"].dropna().unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    x_vals = df["class_name"].map(class_to_idx)
    jitter = (np.random.rand(len(x_vals)) - 0.5) * 0.2
    ax.scatter(x_vals + jitter, df["track_length_pred"], s=10, alpha=0.7)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_title(f"Pred Track Length by Class (scatter) ({variant})")
    ax.set_xlabel("Class")
    ax.set_ylabel("Pred Track Length (frames)")
    fig.tight_layout()
    save_or_show(fig, outdir, f"predlen_by_class_scatter_{variant}.png", show)


def plot_pred_len_by_gt_len_scatter(df: pd.DataFrame, outdir: str, show: bool, variant: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df, x="track_length_gt", y="track_length_pred", s=12, ax=ax, linewidth=0
    )
    ax.set_title(f"Pred Track Length vs GT Track Length ({variant})")
    ax.set_xlabel("GT Track Length (frames)")
    ax.set_ylabel("Pred Track Length (frames)")
    save_or_show(fig, outdir, f"predlen_by_gtlen_{variant}.png", show)


def main():
    parser = argparse.ArgumentParser(description="Trackwise 평가 결과 시각화")
    parser.add_argument(
        "result_json",
        type=str,
        help="evaluate_trackwise.py가 생성한 trackwise_evaluation.json 경로",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="det",
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

    args = parser.parse_args()

    sns.set(style="whitegrid")

    outdir = ensure_outdir(args.outdir)
    df = load_trackwise_dataframe(args.result_json, args.variant)

    if args.class_filter:
        df = df[df["class_name"].isin(args.class_filter)].reset_index(drop=True)

    # 결측치 정리 (플롯에 방해되는 NaN 제거는 각 플롯에서 처리되지만, 전반적으로 정리)
    # 여기서는 그대로 두고, 각 플롯 함수에서 dropna 처리

    show = not args.no_show and (outdir is None)

    plot_mota_hist(df, outdir, show, args.variant)
    plot_mota_by_gt_len(df, outdir, show, args.variant)
    plot_mota_by_class_scatter(df, outdir, show, args.variant)
    plot_mota_by_class_box(df, outdir, show, args.variant)
    plot_pred_len_by_class_scatter(df, outdir, show, args.variant)
    plot_pred_len_by_gt_len_scatter(df, outdir, show, args.variant)


if __name__ == "__main__":
    main()



