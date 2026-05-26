"""
tasks/summarization.py — 한국어 텍스트 요약 (AutoClass 방식)

[왜 AutoClass를 선택했는가]
요약 태스크는 생성(generation) 단계에서 다양한 파라미터를 조절해야 한다.
pipeline("summarization")은 내부적으로 generate()를 호출하지만
min_length, num_beams, length_penalty 등의 세밀한 제어가 번거롭다.
AutoTokenizer + AutoModelForSeq2SeqLM 조합을 직접 사용하면:
  1. 토크나이징 → 추론 → 디코딩의 각 단계를 명시적으로 확인·수정 가능
  2. model.generate()의 모든 파라미터를 자유롭게 실험 가능
  3. 배치 처리 시 패딩·attention_mask를 직접 제어하여 효율 최적화 가능
  4. 수업에서 강조한 "전이학습 흐름 이해"를 코드로 직접 체험
이는 수업의 AutoClass 이해 목표와도 부합한다.
"""

import warnings
from typing import Any, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── 디바이스 설정 ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = "GPU" if torch.cuda.is_available() else "CPU"

# ── 모델 설정 ──────────────────────────────────────────────────
MODEL_ID = "gogamza/kobart-summarization"
# MIT 라이선스 (Hub에서 확인 필요: https://huggingface.co/gogamza/kobart-summarization)
MIN_INPUT_LENGTH = 100  # 이보다 짧은 입력은 경고 발생

# ── 모델·토크나이저 캐시 (모듈 로드 시 1회만 초기화) ───────────
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSeq2SeqLM] = None


def _load_model():
    """토크나이저와 모델을 처음 한 번만 로드한다."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"[summarization] KoBART 로딩 중 ({DEVICE_NAME})...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        _model = _model.to(DEVICE)
        _model.eval()  # 추론 모드 (Dropout 비활성화)
        print("[summarization] 로드 완료.")
    return _tokenizer, _model


# ── 공개 API ───────────────────────────────────────────────────

def summarize(
    text: str,
    min_length: int = 30,
    max_length: int = 100,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
) -> dict:
    """
    한국어 텍스트를 추상적 요약(abstractive summarization)한다.

    Args:
        text: 요약할 원문 텍스트 (영화 리뷰 또는 기사).
        min_length: 요약문의 최소 토큰 수. 너무 짧은 요약 방지.
        max_length: 요약문의 최대 토큰 수. 길이 상한선.
        num_beams: 빔 서치 너비. 클수록 품질↑ 속도↓.
                   1 = greedy, 4 이상 권장.
        no_repeat_ngram_size: 이 크기의 n-gram이 반복되지 않도록 제한.
                              3으로 설정 시 3-gram 중복 방지, 반복 문장 억제.
        length_penalty: 생성 길이에 대한 패널티.
                        1.0 = 중립, >1.0 = 긴 요약 선호, <1.0 = 짧은 요약 선호.

    Returns:
        {
            "summary": str,
            "input_length": int,     # 원문 글자 수
            "output_length": int,    # 요약문 글자 수
            "input_tokens": int,     # 원문 토큰 수
            "output_tokens": int,    # 요약문 토큰 수
            "params_used": dict      # 사용된 생성 파라미터
        }

    Raises:
        ValueError: text가 비어 있는 경우.
    """
    if not text or not text.strip():
        raise ValueError("입력 텍스트가 비어 있습니다.")

    if len(text) < MIN_INPUT_LENGTH:
        warnings.warn(
            f"입력 텍스트가 짧습니다 ({len(text)}자). "
            f"{MIN_INPUT_LENGTH}자 이상 입력 시 더 좋은 요약 품질을 기대할 수 있습니다.",
            UserWarning,
        )

    tokenizer, model = _load_model()

    # ── Step 1: 토크나이징 (입력 텍스트 → input_ids) ──────────
    # KoBART의 최대 입력 길이는 1024 토큰
    inputs = tokenizer(
        text,
        return_tensors="pt",     # PyTorch 텐서 반환
        max_length=1024,
        truncation=True,         # 1024 토큰 초과 시 잘라냄
        padding=False,
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    n_input_tokens = input_ids.shape[1]

    # ── Step 2: 모델 추론 (generate() 호출) ───────────────────
    # torch.no_grad(): 그래디언트 계산 비활성화 → 메모리·속도 최적화
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_length=min_length,           # 최소 길이 강제
            max_length=max_length,           # 최대 길이 상한
            num_beams=num_beams,             # 빔 서치 너비
            no_repeat_ngram_size=no_repeat_ngram_size,  # 반복 억제
            length_penalty=length_penalty,   # 길이 패널티
            early_stopping=True,             # EOS 토큰 생성 시 조기 종료
        )

    # ── Step 3: 디코딩 (output_ids → 한국어 텍스트) ───────────
    # output_ids[0]: 배치 크기 1이므로 첫 번째 결과만 사용
    summary_raw = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # ── Step 4: 후처리 (특수 토큰 제거 및 텍스트 정제) ─────────
    # KoBART 특수 토큰: <s>, </s>, <pad>
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # SentencePiece ▁ 기호 및 언더스코어 잔재 제거
    summary = summary.replace('▁', ' ')
    summary = summary.replace('__', ' ')
    # 다중 공백 정리
    import re
    summary = re.sub(r' +', ' ', summary)
    summary = summary.strip()

    return {
        "summary": summary,
        "input_length": len(text),
        "output_length": len(summary),
        "input_tokens": n_input_tokens,
        "output_tokens": int(output_ids.shape[1]),
        "params_used": {
            "min_length": min_length,
            "max_length": max_length,
            "num_beams": num_beams,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "length_penalty": length_penalty,
        },
    }


def summarize_batch(texts: list[str], **kwargs) -> list[dict]:
    """
    여러 텍스트를 순차적으로 요약한다. datasets.map() 연동용.

    Args:
        texts: 요약할 텍스트 리스트.
        **kwargs: summarize()에 전달할 생성 파라미터.

    Returns:
        각 텍스트에 대한 summarize() 결과 리스트.

    Example (datasets.map 사용):
        >>> ds = ds.map(
        ...     lambda batch: {
        ...         "summary": [r["summary"] for r in summarize_batch(batch["document"])]
        ...     },
        ...     batched=True,
        ...     batch_size=8,
        ... )
    """
    if not texts:
        return []

    results = []
    for text in texts:
        try:
            results.append(summarize(text, **kwargs))
        except Exception as e:
            # 배치 중 개별 오류는 건너뜀
            results.append({"summary": f"[오류: {str(e)}]", "error": True})
    return results


# ── 직접 실행 테스트 ───────────────────────────────────────────
if __name__ == "__main__":
    test_reviews = [
        """
        이 영화는 정말 오랜만에 본 수작이었습니다. 배우들의 연기력은 말할 것도 없고,
        감독의 연출력이 특히 돋보였습니다. 특히 클라이맥스 장면에서의 긴장감은 
        관객을 완전히 몰입하게 만들었습니다. 음악도 영화의 분위기와 완벽하게 
        어우러져서 감동을 배가시켰습니다. 스토리도 탄탄하고 예측 불가능한 반전이
        있어서 끝까지 눈을 뗄 수 없었습니다. 강력히 추천하는 영화입니다.
        """,
        """
        솔직히 기대했던 것보다 많이 실망스러웠습니다. 예고편에서 보여줬던 장면들이
        거의 전부였고, 나머지 2시간은 지루한 전개로 가득했습니다. 배우들의 연기는
        훌륭했지만 대본이 너무 약했습니다. 특히 결말이 너무 급하게 처리되어서
        많은 복선들이 회수되지 않았습니다. 다음 작품에서는 더 좋은 시나리오 작가를
        기용해주길 바랍니다. CGI는 훌륭했지만 전체적인 완성도는 아쉬웠습니다.
        """,
    ]

    print("=" * 60)
    print("KoBART 요약 테스트 (AutoClass 방식)")
    print("=" * 60)

    for i, review in enumerate(test_reviews, 1):
        review = review.strip()
        print(f"\n[리뷰 {i}]")
        print(f"원문 ({len(review)}자): {review[:80]}...")

        result = summarize(review, min_length=20, max_length=80, num_beams=4)
        print(f"요약 ({len(result['summary'])}자): {result['summary']}")
        print(f"토큰: 입력 {result['input_tokens']} → 출력 {result['output_tokens']}")

    print("\n" + "=" * 60)
    print("파라미터 변화에 따른 요약 품질 비교")
    print("=" * 60)
    review = test_reviews[0].strip()
    for beams in [1, 2, 4]:
        r = summarize(review, num_beams=beams, max_length=80)
        print(f"\nnum_beams={beams}: {r['summary']}")
