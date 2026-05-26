"""
analysis/model_comparison.py — 모델 성능 비교 평가

과제 요구사항:
- 동일 태스크에 대해 2개 이상의 모델 비교
- 최소 100개 이상 한국어 텍스트로 평가
- 정확도, F1, 추론 속도, 모델 크기 중 최소 2개 지표 산출
- dataset.map() 활용 (단순 for 루프 금지)
"""

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # GUI 없는 환경에서도 동작
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate
from transformers import pipeline

# ── 경로 설정 ──────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_CHART = DATA_DIR / "comparison_results.png"

DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = "GPU" if DEVICE == 0 else "CPU"

# ── 비교 대상 모델 ─────────────────────────────────────────────
SENTIMENT_MODELS = {
    "KoELECTRA": "monologg/koelectra-base-finetuned-sentiment",
    "KR-FinBert": "snunlp/KR-FinBert-SC",
}

# NSMC 레이블 매핑 (0=부정, 1=긍정)
NSMC_LABEL_MAP = {0: "부정", 1: "긍정"}

# 모델 출력 레이블 → NSMC 레이블 통일 매핑
RAW_LABEL_MAP = {
    "POSITIVE": 1, "positive": 1, "pos": 1, "LABEL_1": 1, "1": 1,
    "NEGATIVE": 0, "negative": 0, "neg": 0, "LABEL_0": 0, "0": 0,
}

# ── 파이프라인 캐시 ─────────────────────────────────────────────
_pipelines: dict = {}


def _get_sentiment_pipeline(model_key: str):
    if model_key not in _pipelines:
        model_id = SENTIMENT_MODELS[model_key]
        print(f"[comparison] {model_key} 로딩 중 ({DEVICE_NAME})...")
        _pipelines[model_key] = pipeline(
            "text-classification",
            model=model_id,
            device=DEVICE,
            truncation=True,
            max_length=512,
        )
        print(f"[comparison] {model_key} 로드 완료.")
    return _pipelines[model_key]


def get_model_size_mb(model_id: str) -> float:
    """
    모델의 파라미터 수를 MB 단위로 환산하여 반환한다.
    float32 기준: 파라미터 수 × 4바이트 / 1,048,576

    Args:
        model_id: Hugging Face 모델 ID.

    Returns:
        모델 크기 (MB). 로드 실패 시 -1.
    """
    try:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(model_id)
        n_params = sum(p.numel() for p in m.parameters())
        return round(n_params * 4 / 1_048_576, 1)
    except Exception:
        return -1.0


def measure_inference_speed(pipe, texts: list[str]) -> dict:
    """
    파이프라인의 추론 속도를 측정한다.

    Args:
        pipe: Hugging Face pipeline 객체.
        texts: 테스트 텍스트 리스트.

    Returns:
        {"mean_ms": float, "min_ms": float, "max_ms": float, "total_ms": float}
    """
    times = []
    for text in texts:
        start = time.perf_counter()
        pipe(text, truncation=True, max_length=512)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": round(float(np.mean(times)), 2),
        "min_ms": round(float(np.min(times)), 2),
        "max_ms": round(float(np.max(times)), 2),
        "total_ms": round(float(np.sum(times)), 2),
    }


def _load_nsmc_balanced(n_samples: int = 100) -> tuple[list[str], list[int]]:
    """
    NSMC 테스트 셋에서 긍정/부정 균형 샘플을 로드한다.

    Returns:
        (texts, labels): 텍스트 리스트, 정수 레이블 리스트 (0=부정, 1=긍정)
    """
    print("[comparison] NSMC 데이터셋 로딩 중...")
    ds = load_dataset("sepidmnorozy/Korean_sentiment", split="test")

    half = n_samples // 2
    # 긍정/부정 각 half개 추출
    pos_samples = ds.filter(lambda x: x["label"] == 1).select(range(half))
    neg_samples = ds.filter(lambda x: x["label"] == 0).select(range(half))

    # 합치기
    texts, labels = [], []
    for item in pos_samples:
        texts.append(item["text"])
        labels.append(1)
    for item in neg_samples:
        texts.append(item["text"])
        labels.append(0)

    print(f"[comparison] {len(texts)}개 샘플 로드 완료 (긍정 {half}개, 부정 {half}개).")
    return texts, labels


def _run_inference_with_map(pipe, texts: list[str], batch_size: int = 16) -> list[int]:
    """
    dataset.map()을 활용하여 일괄 추론을 수행한다.
    단순 for 루프 대신 Hugging Face Datasets의 map() 사용.

    Args:
        pipe: 감성 분석 파이프라인.
        texts: 입력 텍스트 리스트.
        batch_size: map() 배치 크기.

    Returns:
        예측 레이블 정수 리스트 (0 또는 1).
    """
    from datasets import Dataset

    ds = Dataset.from_dict({"text": texts})

    def predict_batch(batch):
        """배치 단위 추론 함수. dataset.map()에 전달된다."""
        results = pipe(
            batch["text"],
            batch_size=batch_size,
            truncation=True,
            max_length=512,
        )
        # 레이블 정수 변환
        preds = [RAW_LABEL_MAP.get(r["label"], -1) for r in results]
        return {"prediction": preds}

    # dataset.map()으로 일괄 추론 (for 루프 없이)
    ds = ds.map(predict_batch, batched=True, batch_size=batch_size)
    return ds["prediction"]


def evaluate_sentiment_models(n_samples: int = 100) -> dict:
    """
    두 감성 분석 모델을 NSMC 데이터셋으로 비교 평가한다.

    Args:
        n_samples: 평가에 사용할 총 샘플 수 (긍정/부정 균등 분할).

    Returns:
        {
            "model_a": {"name", "accuracy", "f1", "inference_time_ms", "model_size_mb"},
            "model_b": { ... },
            "comparison_table": list[dict]
        }
    """
    texts, true_labels = _load_nsmc_balanced(n_samples)

    comparison_table = []
    model_results = {}

    for key, model_id in SENTIMENT_MODELS.items():
        print(f"\n[comparison] {key} 평가 시작...")
        pipe = _get_sentiment_pipeline(key)

        # dataset.map() 기반 배치 추론
        start_total = time.perf_counter()
        predictions = _run_inference_with_map(pipe, texts, batch_size=16)
        total_time_ms = (time.perf_counter() - start_total) * 1000

        # 추론 속도 측정 (샘플 10개로)
        speed = measure_inference_speed(pipe, texts[:10])

        # 평가 지표 계산
        valid_pairs = [(p, t) for p, t in zip(predictions, true_labels) if p != -1]
        preds = [p for p, _ in valid_pairs]
        trues = [t for _, t in valid_pairs]

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average="binary")

        # 모델 크기 (첫 실행 후 캐시 활용 가능)
        size_mb = get_model_size_mb(model_id)

        result = {
            "name": key,
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "inference_time_ms": speed["mean_ms"],
            "model_size_mb": size_mb,
            "total_inference_ms": round(total_time_ms, 1),
            "n_evaluated": len(valid_pairs),
        }
        model_results[key] = result
        comparison_table.append(result)
        print(f"  → 정확도: {acc:.4f}, F1: {f1:.4f}, 속도: {speed['mean_ms']}ms/샘플")

    keys = list(model_results.keys())
    return {
        "model_a": model_results[keys[0]],
        "model_b": model_results[keys[1]],
        "comparison_table": comparison_table,
    }


def evaluate_qa_models(n_samples: int = 100) -> dict:
    """
    두 QA 모델을 KorQuAD 데이터셋으로 비교 평가한다 (EM, F1).

    Args:
        n_samples: 평가 샘플 수.

    Returns:
        {
            "model_a": {"name", "em", "f1", "inference_time_ms"},
            "model_b": { ... }
        }
    """
    from transformers import pipeline as hf_pipeline

    print("[comparison] KorQuAD 데이터셋 로딩 중...")
    try:
        ds = load_dataset("squad_kor_v1", split="validation", trust_remote_code=True)
    except Exception:
        print("[comparison] KorQuAD 로드 실패. squad_kor_v1 대신 klue/mrc 시도...")
        ds = load_dataset("klue", "mrc", split="validation", trust_remote_code=True)

    ds = ds.select(range(min(n_samples, len(ds))))

    qa_models = {
        "KoELECTRA": "monologg/koelectra-base-v3-finetuned-korquad",
        "KLUE-RoBERTa": "klue/roberta-base",
    }

    def normalize_answer(s: str) -> str:
        return " ".join(s.lower().split())

    def compute_f1(pred: str, gold: str) -> float:
        pred_tokens = normalize_answer(pred).split()
        gold_tokens = normalize_answer(gold).split()
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            return 0.0
        p = len(common) / len(pred_tokens)
        r = len(common) / len(gold_tokens)
        return 2 * p * r / (p + r)

    results = {}
    for key, model_id in qa_models.items():
        print(f"[comparison] {key} QA 평가 중...")
        qa_pipe = hf_pipeline("question-answering", model=model_id, device=DEVICE)

        em_scores, f1_scores_list, times = [], [], []

        def run_qa_batch(batch):
            preds = []
            for q, ctx in zip(batch["question"], batch["context"]):
                t0 = time.perf_counter()
                out = qa_pipe({"question": q, "context": ctx})
                times.append((time.perf_counter() - t0) * 1000)
                preds.append(out["answer"])
            return {"prediction": preds}

        ds_pred = ds.map(run_qa_batch, batched=True, batch_size=8)

        for item in ds_pred:
            pred = item["prediction"]
            gold = item["answers"]["text"][0] if item["answers"]["text"] else ""
            em_scores.append(int(normalize_answer(pred) == normalize_answer(gold)))
            f1_scores_list.append(compute_f1(pred, gold))

        results[key] = {
            "name": key,
            "em": round(float(np.mean(em_scores)), 4),
            "f1": round(float(np.mean(f1_scores_list)), 4),
            "inference_time_ms": round(float(np.mean(times)), 2),
        }
        print(f"  → EM: {results[key]['em']:.4f}, F1: {results[key]['f1']:.4f}")

    keys = list(results.keys())
    return {"model_a": results[keys[0]], "model_b": results[keys[1]]}


def plot_comparison_chart(comparison: dict, save_path: str = str(OUTPUT_CHART)):
    """
    모델 비교 결과를 막대 차트로 시각화하여 PNG로 저장한다.

    Args:
        comparison: evaluate_sentiment_models() 반환값.
        save_path: 저장 경로.
    """
    # 한글 폰트 설정 (서버 환경에서는 NanumGothic 또는 DejaVu 사용)
    try:
        font_path = fm.findfont("NanumGothic")
        if font_path:
            plt.rcParams["font.family"] = "NanumGothic"
    except Exception:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    table = comparison["comparison_table"]
    model_names = [r["name"] for r in table]
    metrics = {
        "Accuracy": [r["accuracy"] for r in table],
        "F1 Score": [r["f1"] for r in table],
    }
    speed_vals = [r["inference_time_ms"] for r in table]
    size_vals = [r["model_size_mb"] if r["model_size_mb"] > 0 else 0 for r in table]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Sentiment Model Comparison (NSMC)", fontsize=14, fontweight="bold")

    x = np.arange(len(model_names))
    width = 0.35
    colors = ["#4C72B0", "#DD8452"]

    # 1. 정확도 & F1 비교
    ax = axes[0]
    for i, (metric, values) in enumerate(metrics.items()):
        ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.85)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Accuracy & F1 Score")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)

    # 2. 추론 속도 비교
    ax = axes[1]
    bars = ax.bar(model_names, speed_vals, color=colors, alpha=0.85)
    ax.set_xlabel("Model")
    ax.set_ylabel("ms / sample")
    ax.set_title("Inference Speed (ms/sample)")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, speed_vals):
        ax.annotate(f"{val:.1f}ms", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)

    # 3. 모델 크기 비교
    ax = axes[2]
    bars = ax.bar(model_names, size_vals, color=colors, alpha=0.85)
    ax.set_xlabel("Model")
    ax.set_ylabel("Size (MB)")
    ax.set_title("Model Size (MB)")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, size_vals):
        label = f"{val:.0f}MB" if val > 0 else "N/A"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 1)),
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[comparison] 비교 차트 저장: {save_path}")


def print_comparison_table(comparison: dict):
    """비교 결과를 콘솔 표로 출력한다."""
    table = comparison["comparison_table"]
    headers = ["모델", "정확도", "F1", "속도(ms)", "크기(MB)", "평가 샘플"]
    rows = [
        [
            r["name"],
            f"{r['accuracy']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['inference_time_ms']:.1f}",
            f"{r['model_size_mb']:.0f}" if r['model_size_mb'] > 0 else "N/A",
            r["n_evaluated"],
        ]
        for r in table
    ]
    print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))


# ── 직접 실행 ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("감성 분석 모델 비교 평가 시작 (NSMC 100샘플)")
    print("=" * 65)

    comparison = evaluate_sentiment_models(n_samples=100)
    print_comparison_table(comparison)
    plot_comparison_chart(comparison)

    print("\n" + "=" * 65)
    print("최종 선택 모델 및 근거")
    print("=" * 65)
    a, b = comparison["model_a"], comparison["model_b"]
    winner = a if a["accuracy"] >= b["accuracy"] else b
    print(f"\n선택 모델: {winner['name']}")
    print(f"  → 정확도 {winner['accuracy']:.4f}, F1 {winner['f1']:.4f}")
    print(f"  → 추론 속도 {winner['inference_time_ms']:.1f}ms/샘플")
    print(f"  → 모델 크기 {winner['model_size_mb']:.0f}MB")
