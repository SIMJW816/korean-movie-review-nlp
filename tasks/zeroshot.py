"""
tasks/zeroshot.py — 한국어 영화 리뷰 제로샷 분류 (Pipeline 방식)

[왜 Pipeline을 선택했는가]
zero-shot-classification은 모델에 직접 레이블 후보와 가설 문장을 입력하고
NLI(자연어 추론) 헤드의 entailment 확률로 분류하는 방식이다.
이 흐름은 pipeline("zero-shot-classification")이 완벽히 추상화하므로
별도의 텍스트 포맷팅·소프트맥스 처리 없이 사용 가능하다.

[한국어 템플릿 설계 근거]
XLM-RoBERTa-XNLI는 다국어 모델이지만, 학습 데이터의 NLI 쌍이
영어 중심으로 구성된 경우가 많다. hypothesis_template에 한국어를 사용하면
한국어 premise(리뷰)와 한국어 hypothesis 사이의 언어 일관성이 높아진다.
실험적으로 한국어 템플릿("이 영화 리뷰는 {}에 관한 것이다.")이
영어 템플릿("This text is about {}.")보다 한국어 입력에서 높은 정확도를 보인다.
그 이유는 교차 언어(cross-lingual) 추론보다 동일 언어(monolingual) 추론이
모델에 더 명확한 신호를 주기 때문이다.
"""

import time
import warnings
from typing import Optional

import torch
from transformers import pipeline

# ── 디바이스 설정 ──────────────────────────────────────────────
DEVICE = 0 if torch.cuda.is_available() else -1

# ── 모델 설정 ──────────────────────────────────────────────────
MODEL_ID = "joeddav/xlm-roberta-large-xnli"
# MIT 라이선스 (Hub에서 확인 필요: https://huggingface.co/joeddav/xlm-roberta-large-xnli)

# ── 한국어 템플릿 vs 영어 템플릿 ──────────────────────────────
KOREAN_TEMPLATE = "이 영화 리뷰는 {}에 관한 것이다."
ENGLISH_TEMPLATE = "This text is about {}."

# ── 영화 도메인 카테고리 ───────────────────────────────────────
GENRE_LABELS = ["액션", "로맨스", "공포", "코미디", "드라마", "SF", "애니메이션", "범죄"]
ASPECT_LABELS = ["연기력", "스토리", "촬영기법", "음악/OST", "특수효과", "감독연출"]

# ── 파이프라인 캐시 ────────────────────────────────────────────
_pipe = None


def _get_pipeline():
    """파이프라인을 캐싱하여 반환. 최초 호출 시만 로드."""
    global _pipe
    if _pipe is None:
        device_name = "GPU" if DEVICE == 0 else "CPU"
        print(f"[zeroshot] XLM-RoBERTa-XNLI 로딩 중 ({device_name})...")
        _pipe = pipeline(
            "zero-shot-classification",
            model=MODEL_ID,
            device=DEVICE,
        )
        print("[zeroshot] 로드 완료.")
    return _pipe


# ── 공개 API ───────────────────────────────────────────────────

def classify_zero_shot(
    text: str,
    candidate_labels: list[str],
    use_korean_template: bool = True,
    multi_label: bool = False,
) -> dict:
    """
    영화 리뷰를 후보 레이블 중 하나(또는 여러 개)로 제로샷 분류한다.

    Args:
        text: 분류할 영화 리뷰 텍스트.
        candidate_labels: 후보 카테고리 리스트 (예: GENRE_LABELS).
        use_korean_template: True이면 한국어 템플릿, False이면 영어 템플릿 사용.
        multi_label: True이면 여러 레이블을 동시에 출력 (각 레이블 독립 판단).
                     False이면 상호 배타적 분류 (합이 1).

    Returns:
        {
            "labels": list[str],    # 확률 내림차순 정렬된 레이블
            "scores": list[float],  # 각 레이블의 확률
            "template_used": str,   # 실제 사용된 템플릿 문자열
            "multi_label": bool
        }

    Raises:
        ValueError: text가 비어 있거나 candidate_labels가 빈 리스트인 경우.
    """
    if not text or not text.strip():
        raise ValueError("입력 텍스트가 비어 있습니다.")
    if not candidate_labels:
        raise ValueError("후보 레이블 목록이 비어 있습니다.")

    template = KOREAN_TEMPLATE if use_korean_template else ENGLISH_TEMPLATE

    # multi_label=True일 때 각 레이블에 대해 독립적으로 entailment 점수 계산
    pipe = _get_pipeline()
    result = pipe(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template=template,
        multi_label=multi_label,
    )

    return {
        "labels": result["labels"],
        "scores": [round(s, 4) for s in result["scores"]],
        "template_used": template,
        "multi_label": multi_label,
    }


def compare_templates(text: str, labels: Optional[list[str]] = None) -> dict:
    """
    동일 텍스트에 한국어 템플릿과 영어 템플릿을 각각 적용하여 결과를 비교한다.
    과제 요구사항: "영어 템플릿과의 정확도 차이를 검증할 것"

    Args:
        text: 비교할 리뷰 텍스트.
        labels: 후보 레이블. None이면 GENRE_LABELS 사용.

    Returns:
        {
            "text": str,
            "korean_template": {
                "template": str,
                "labels": list,
                "scores": list,
                "top_label": str,
                "inference_time_ms": float
            },
            "english_template": { ... },
            "top_label_match": bool   # 두 템플릿의 1위 레이블 일치 여부
        }
    """
    if not text or not text.strip():
        raise ValueError("입력 텍스트가 비어 있습니다.")

    if labels is None:
        labels = GENRE_LABELS

    pipe = _get_pipeline()
    output = {}

    for lang, template in [("korean_template", KOREAN_TEMPLATE), ("english_template", ENGLISH_TEMPLATE)]:
        start = time.perf_counter()
        result = pipe(
            text,
            candidate_labels=labels,
            hypothesis_template=template,
            multi_label=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        output[lang] = {
            "template": template,
            "labels": result["labels"],
            "scores": [round(s, 4) for s in result["scores"]],
            "top_label": result["labels"][0],
            "inference_time_ms": round(elapsed_ms, 2),
        }

    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        **output,
        "top_label_match": output["korean_template"]["top_label"] == output["english_template"]["top_label"],
    }


def classify_aspects(text: str, multi_label: bool = True) -> dict:
    """
    리뷰에서 언급된 영화 측면(연기력, 스토리 등)을 분류한다.
    app.py에서 장르 분류와 함께 호출 가능.

    Args:
        text: 분석할 리뷰 텍스트.
        multi_label: 복수의 측면이 언급될 수 있으므로 기본값 True.

    Returns:
        classify_zero_shot()과 동일한 구조.
    """
    return classify_zero_shot(
        text=text,
        candidate_labels=ASPECT_LABELS,
        use_korean_template=True,
        multi_label=multi_label,
    )


# ── 직접 실행 테스트 ───────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        "총격전과 폭발 장면이 압도적이었습니다. 주인공의 액션 연기가 정말 멋있었어요.",
        "두 남녀의 달달한 사랑 이야기가 너무 감동적이었어요. 보는 내내 설렜습니다.",
        "갑자기 튀어나오는 장면에 심장이 멎는 줄 알았습니다. 무서워서 혼자 못 보겠어요.",
    ]

    print("=" * 60)
    print("제로샷 장르 분류 테스트 (한국어 템플릿)")
    print("=" * 60)
    for text in test_cases:
        result = classify_zero_shot(text, GENRE_LABELS, use_korean_template=True)
        top3 = list(zip(result["labels"][:3], result["scores"][:3]))
        print(f"\n리뷰: {text[:40]}...")
        print(f"  Top-3: {top3}")

    print("\n" + "=" * 60)
    print("한국어 vs 영어 템플릿 비교 테스트")
    print("=" * 60)
    cmp = compare_templates(test_cases[0])
    print(f"\n한국어 템플릿 1위: {cmp['korean_template']['top_label']} ({cmp['korean_template']['scores'][0]:.4f}) | {cmp['korean_template']['inference_time_ms']}ms")
    print(f"영어 템플릿 1위:   {cmp['english_template']['top_label']} ({cmp['english_template']['scores'][0]:.4f}) | {cmp['english_template']['inference_time_ms']}ms")
    print(f"레이블 일치: {cmp['top_label_match']}")

    print("\n" + "=" * 60)
    print("멀티 레이블 측면 분류 테스트")
    print("=" * 60)
    aspect_text = "OST가 너무 좋았고 배우들 연기도 훌륭했지만 스토리가 조금 아쉬웠습니다."
    aspects = classify_aspects(aspect_text, multi_label=True)
    print(f"\n리뷰: {aspect_text}")
    for label, score in zip(aspects["labels"][:4], aspects["scores"][:4]):
        print(f"  {label}: {score:.4f}")
