"""
tasks/sentiment.py — 한국어 영화 리뷰 감성 분석 (Pipeline 방식)

[왜 Pipeline을 선택했는가]
감성 분석은 텍스트 → 레이블+확률의 단순하고 표준화된 흐름이다.
pipeline("text-classification")은 이 흐름을 단 한 줄로 캡슐화하며,
토크나이징·추론·소프트맥스 후처리까지 자동으로 처리해준다.
반면 AutoClass 방식은 logits → softmax → argmax를 직접 구현해야 한다.
이 태스크에서는 빠른 프로토타이핑과 안정적인 배치 처리가 우선이므로
Pipeline이 적합하다. 생성 파라미터 제어가 필요한 요약과 달리,
분류 태스크는 파이프라인 추상화 수준으로 충분하다.
"""

import time
import warnings
from typing import Optional

import torch
from transformers import pipeline

# ── 디바이스 설정 ──────────────────────────────────────────────
DEVICE = 0 if torch.cuda.is_available() else -1  # GPU 있으면 0, 없으면 CPU(-1)
DEVICE_NAME = "GPU" if DEVICE == 0 else "CPU"

# ── 지원 모델 정의 ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "default": {
        "model_id": "snunlp/KR-FinBert-SC",
        "display_name": "KR-FinBert-SC",
        # Apache 2.0 라이선스 (Hub에서 확인 필요)
    },
    "koelectra": {
        "model_id": "monologg/koelectra-base-finetuned-sentiment",
        "display_name": "KoELECTRA-Sentiment",
        # Apache 2.0 라이선스
    },
}

# ── 레이블 한국어 변환 매핑 ────────────────────────────────────
LABEL_MAP = {
    "POSITIVE": "긍정",
    "NEGATIVE": "부정",
    "positive": "긍정",
    "negative": "부정",
    "pos": "긍정",
    "neg": "부정",
    "1": "긍정",
    "0": "부정",
    "LABEL_1": "긍정",
    "LABEL_0": "부정",
}

# ── 모델 캐시 (모듈 임포트 시 1회만 로드) ──────────────────────
_pipelines: dict = {}


def _get_pipeline(model_key: str = "default"):
    """모델 파이프라인을 캐싱하여 반환. 최초 호출 시만 다운로드·로드."""
    global _pipelines
    if model_key not in _pipelines:
        cfg = MODEL_CONFIGS.get(model_key)
        if cfg is None:
            raise ValueError(f"지원하지 않는 모델 키: {model_key}. 사용 가능: {list(MODEL_CONFIGS)}")
        print(f"[sentiment] {cfg['display_name']} 로딩 중 ({DEVICE_NAME})...")
        _pipelines[model_key] = pipeline(
            "text-classification",
            model=cfg["model_id"],
            device=DEVICE,
            truncation=True,        # 512 토큰 초과 시 자동 잘라냄
            max_length=512,
        )
        print(f"[sentiment] {cfg['display_name']} 로드 완료.")
    return _pipelines[model_key]


def _normalize_label(raw_label: str) -> str:
    """모델별로 다른 레이블 문자열을 통일된 한국어로 변환."""
    return LABEL_MAP.get(raw_label, raw_label)


# ── 공개 API ───────────────────────────────────────────────────

def analyze_sentiment(text: str, model_name: str = "default") -> dict:
    """
    단일 한국어 텍스트의 감성을 분석한다.

    Args:
        text: 분석할 영화 리뷰 텍스트.
        model_name: 사용할 모델 키. "default" 또는 "koelectra".

    Returns:
        {
            "label": "긍정" | "부정",
            "score": float,          # 해당 레이블의 확률 (0~1)
            "model": str,            # 사용된 모델 이름
            "raw_label": str         # 원본 레이블 (디버깅용)
        }

    Raises:
        ValueError: text가 빈 문자열인 경우.
    """
    if not text or not text.strip():
        raise ValueError("입력 텍스트가 비어 있습니다.")

    # 512 토큰 초과 경고 (pipeline 내부에서 truncation 처리됨)
    if len(text) > 1000:
        warnings.warn(
            f"입력 텍스트가 깁니다 ({len(text)}자). 512 토큰 초과 부분은 잘립니다.",
            UserWarning,
        )

    pipe = _get_pipeline(model_name)
    result = pipe(text)[0]

    raw_label = result["label"]
    return {
        "label": _normalize_label(raw_label),
        "score": round(result["score"], 4),
        "model": MODEL_CONFIGS[model_name]["display_name"],
        "raw_label": raw_label,
    }


def batch_analyze(texts: list[str], model_name: str = "default") -> list[dict]:
    """
    여러 텍스트를 한꺼번에 분석한다. datasets.map() 연동용.

    Args:
        texts: 분석할 텍스트 리스트.
        model_name: 사용할 모델 키.

    Returns:
        각 텍스트에 대한 analyze_sentiment() 결과 리스트.

    Example (datasets.map 사용):
        >>> from datasets import load_dataset
        >>> ds = load_dataset("nsmc", split="test")
        >>> ds = ds.map(
        ...     lambda batch: {"prediction": batch_analyze(batch["document"])},
        ...     batched=True,
        ...     batch_size=32,
        ... )
    """
    if not texts:
        return []

    pipe = _get_pipeline(model_name)
    # pipeline은 리스트 입력을 받아 배치 처리 가능
    raw_results = pipe(texts, batch_size=32, truncation=True, max_length=512)

    return [
        {
            "label": _normalize_label(r["label"]),
            "score": round(r["score"], 4),
            "model": MODEL_CONFIGS[model_name]["display_name"],
            "raw_label": r["label"],
        }
        for r in raw_results
    ]


def compare_models(text: str) -> dict:
    """
    두 모델(default, koelectra)로 같은 텍스트를 분석하고 결과를 비교한다.

    Args:
        text: 비교 분석할 텍스트.

    Returns:
        {
            "text": str,
            "model_a": { label, score, model, inference_time_ms },
            "model_b": { label, score, model, inference_time_ms },
            "agreement": bool   # 두 모델의 레이블 일치 여부
        }
    """
    if not text or not text.strip():
        raise ValueError("입력 텍스트가 비어 있습니다.")

    results = {}
    for key in ("default", "koelectra"):
        pipe = _get_pipeline(key)
        start = time.perf_counter()
        raw = pipe(text, truncation=True, max_length=512)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        results[key] = {
            "label": _normalize_label(raw["label"]),
            "score": round(raw["score"], 4),
            "model": MODEL_CONFIGS[key]["display_name"],
            "inference_time_ms": round(elapsed_ms, 2),
        }

    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "model_a": results["default"],
        "model_b": results["koelectra"],
        "agreement": results["default"]["label"] == results["koelectra"]["label"],
    }


# ── 직접 실행 테스트 ───────────────────────────────────────────
if __name__ == "__main__":
    test_texts = [
        "이 영화는 정말 감동적이었어요. 배우들의 연기가 너무 훌륭했습니다.",
        "스토리가 너무 진부하고 지루했습니다. 시간 낭비였어요.",
        "그냥 평범한 영화였어요. 특별히 좋지도 나쁘지도 않았습니다.",
    ]

    print("=" * 60)
    print("단일 텍스트 감성 분석 테스트")
    print("=" * 60)
    for t in test_texts:
        result = analyze_sentiment(t)
        print(f"\n텍스트: {t[:40]}...")
        print(f"  → 레이블: {result['label']}  확률: {result['score']:.4f}  모델: {result['model']}")

    print("\n" + "=" * 60)
    print("모델 비교 테스트")
    print("=" * 60)
    cmp = compare_models(test_texts[0])
    print(f"모델 A: {cmp['model_a']['model']}  →  {cmp['model_a']['label']} ({cmp['model_a']['score']:.4f})  {cmp['model_a']['inference_time_ms']}ms")
    print(f"모델 B: {cmp['model_b']['model']}  →  {cmp['model_b']['label']} ({cmp['model_b']['score']:.4f})  {cmp['model_b']['inference_time_ms']}ms")
    print(f"레이블 일치: {cmp['agreement']}")

    print("\n" + "=" * 60)
    print("배치 분석 테스트")
    print("=" * 60)
    batch_results = batch_analyze(test_texts)
    for t, r in zip(test_texts, batch_results):
        print(f"  {t[:30]}... → {r['label']} ({r['score']:.4f})")
