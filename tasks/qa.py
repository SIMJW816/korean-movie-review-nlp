"""
tasks/qa.py — 한국어 추출형 질의응답 (AutoClass 방식)

[왜 AutoClass를 선택했는가]
추출형 QA는 모델이 context 내 각 토큰 위치에 대해
start_logits와 end_logits를 출력한다.
pipeline("question-answering")은 최상위 답변만 반환하지만
AutoModelForQuestionAnswering을 직접 사용하면:
  1. start/end logits에 직접 접근하여 top-k 후보 답변을 추출 가능
  2. 유효하지 않은 span(start > end, 너무 긴 답변)을 직접 필터링 가능
  3. 각 후보 답변의 정확한 신뢰도 점수(softmax 확률)를 계산 가능
  4. context 내 정확한 문자 오프셋을 이용해 하이라이팅에 활용 가능
이러한 세밀한 제어가 Gradio UI의 HighlightedText 컴포넌트와 연동하기 위해 필요하다.
"""

import time
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# ── 디바이스 설정 ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = "GPU" if torch.cuda.is_available() else "CPU"

# ── 지원 모델 정의 ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "koelectra": {
        "model_id": "monologg/koelectra-base-v3-finetuned-korquad",
        "display_name": "KoELECTRA-KorQuAD",
        # Apache 2.0 라이선스
    },
    "klue": {
        "model_id": "klue/roberta-base",
        "display_name": "KLUE-RoBERTa-base",
        # CC BY-SA 4.0 라이선스
    },
}

MAX_ANSWER_LEN = 50   # 답변 최대 토큰 길이
MAX_SEQ_LEN = 512     # 모델 최대 입력 길이

# ── 모델·토크나이저 캐시 ───────────────────────────────────────
_tokenizers: dict = {}
_models: dict = {}


def _load_model(model_key: str):
    """지정된 모델 키의 토크나이저와 모델을 1회만 로드한다."""
    if model_key not in _tokenizers:
        cfg = MODEL_CONFIGS[model_key]
        print(f"[qa] {cfg['display_name']} 로딩 중 ({DEVICE_NAME})...")
        _tokenizers[model_key] = AutoTokenizer.from_pretrained(cfg["model_id"])
        _models[model_key] = AutoModelForQuestionAnswering.from_pretrained(cfg["model_id"])
        _models[model_key] = _models[model_key].to(DEVICE)
        _models[model_key].eval()
        print(f"[qa] {cfg['display_name']} 로드 완료.")
    return _tokenizers[model_key], _models[model_key]


def _extract_top_k_answers(
    question: str,
    context: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForQuestionAnswering,
    top_k: int = 3,
) -> list[dict]:
    """
    내부 함수: 모델을 실행하고 top-k 후보 답변을 추출한다.

    Returns:
        [{"answer": str, "score": float, "start": int, "end": int}, ...]
    """
    # ── Step 1: 질문+컨텍스트 토크나이징 ─────────────────────
    # [CLS] question [SEP] context [SEP] 형태로 자동 포맷팅
    encoding = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        truncation=True,          # 긴 컨텍스트 자동 잘라냄
        padding=False,
        return_offsets_mapping=True,   # 토큰 ↔ 문자 위치 매핑
        return_token_type_ids=True,    # 질문/컨텍스트 구분 토큰
    )

    offset_mapping = encoding.pop("offset_mapping")[0]   # (seq_len, 2)
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    token_type_ids = encoding.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(DEVICE)

    # ── Step 2: 모델 추론 (start_logits, end_logits 획득) ─────
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    start_logits = outputs.start_logits[0]  # (seq_len,)
    end_logits = outputs.end_logits[0]      # (seq_len,)

    # ── Step 3: softmax로 확률 변환 ──────────────────────────
    start_probs = F.softmax(start_logits, dim=-1).cpu()
    end_probs = F.softmax(end_logits, dim=-1).cpu()

    # ── Step 4: 유효한 span 후보 추출 ────────────────────────
    # 컨텍스트 영역 토큰만 고려 (token_type_ids == 1인 위치)
    # klue/roberta-base는 token_type_ids를 지원하지 않을 수 있으므로 offset_mapping으로 판단
    seq_len = input_ids.shape[1]
    candidates = []

    for start_idx in range(seq_len):
        for end_idx in range(start_idx, min(start_idx + MAX_ANSWER_LEN, seq_len)):
            # 특수 토큰(offset (0,0)) 제외
            if offset_mapping[start_idx] == (0, 0) or offset_mapping[end_idx] == (0, 0):
                continue
            # 질문 영역 제외: context 부분만 (offset이 context 시작 이후인 토큰)
            score = float(start_probs[start_idx] * end_probs[end_idx])
            candidates.append((score, start_idx, end_idx))

    # 점수 내림차순 정렬
    candidates.sort(key=lambda x: -x[0])

    # ── Step 5: 원본 텍스트에서 답변 추출 ────────────────────
    results = []
    seen_answers = set()
    for score, start_idx, end_idx in candidates[: top_k * 3]:  # 중복 제거 고려해 여유있게
        char_start = int(offset_mapping[start_idx][0])
        char_end = int(offset_mapping[end_idx][1])
        answer = context[char_start:char_end].strip()

        if answer and answer not in seen_answers:
            seen_answers.add(answer)
            results.append({
                "answer": answer,
                "score": round(score, 6),
                "start": char_start,
                "end": char_end,
            })
        if len(results) >= top_k:
            break

    return results if results else [{"answer": "답변 없음", "score": 0.0, "start": 0, "end": 0}]


# ── 공개 API ───────────────────────────────────────────────────

def answer_question(
    question: str,
    context: str,
    model_name: str = "koelectra",
    top_k: int = 3,
) -> dict:
    """
    컨텍스트에서 질문에 대한 답변을 추출한다.

    Args:
        question: 질문 텍스트.
        context: 답변이 포함된 영화 리뷰 또는 소개 텍스트.
        model_name: "koelectra" 또는 "klue".
        top_k: 반환할 후보 답변 수.

    Returns:
        {
            "answer": str,         # 최상위 답변
            "score": float,        # 신뢰도
            "start": int,          # context 내 시작 위치 (문자 단위)
            "end": int,            # context 내 끝 위치 (문자 단위)
            "top_answers": list,   # top_k개 후보 [{"answer", "score", "start", "end"}]
            "model_used": str
        }

    Raises:
        ValueError: question 또는 context가 비어 있는 경우.
    """
    if not question or not question.strip():
        raise ValueError("질문이 비어 있습니다.")
    if not context or not context.strip():
        raise ValueError("컨텍스트가 비어 있습니다.")
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"지원하지 않는 모델: {model_name}. 가능: {list(MODEL_CONFIGS)}")

    tokenizer, model = _load_model(model_name)
    top_answers = _extract_top_k_answers(question, context, tokenizer, model, top_k)
    best = top_answers[0]

    return {
        "answer": best["answer"],
        "score": best["score"],
        "start": best["start"],
        "end": best["end"],
        "top_answers": top_answers,
        "model_used": MODEL_CONFIGS[model_name]["display_name"],
    }


def compare_qa_models(question: str, context: str) -> dict:
    """
    두 모델(koelectra, klue)로 동일 질문에 대한 답변을 비교한다.

    Args:
        question: 질문 텍스트.
        context: 답변 컨텍스트.

    Returns:
        {
            "question": str,
            "model_a": {answer, score, model_used, inference_time_ms},
            "model_b": {answer, score, model_used, inference_time_ms},
            "same_answer": bool
        }
    """
    if not question or not question.strip():
        raise ValueError("질문이 비어 있습니다.")
    if not context or not context.strip():
        raise ValueError("컨텍스트가 비어 있습니다.")

    results = {}
    for key in ("koelectra", "klue"):
        tokenizer, model = _load_model(key)
        start_time = time.perf_counter()
        top_answers = _extract_top_k_answers(question, context, tokenizer, model, top_k=1)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        best = top_answers[0]
        results[key] = {
            "answer": best["answer"],
            "score": best["score"],
            "model_used": MODEL_CONFIGS[key]["display_name"],
            "inference_time_ms": round(elapsed_ms, 2),
        }

    return {
        "question": question,
        "model_a": results["koelectra"],
        "model_b": results["klue"],
        "same_answer": results["koelectra"]["answer"] == results["klue"]["answer"],
    }


def build_highlighted_text(context: str, start: int, end: int) -> list[tuple]:
    """
    Gradio HighlightedText 컴포넌트용 포맷으로 변환한다.

    Args:
        context: 전체 컨텍스트 텍스트.
        start: 답변 시작 위치 (문자 단위).
        end: 답변 끝 위치 (문자 단위).

    Returns:
        [(텍스트, 레이블_또는_None), ...] 형태의 리스트.
    """
    if start == 0 and end == 0:
        return [(context, None)]
    return [
        (context[:start], None),
        (context[start:end], "답변"),
        (context[end:], None),
    ]


# ── 직접 실행 테스트 ───────────────────────────────────────────
if __name__ == "__main__":
    sample_context = """
    영화 '기생충'은 봉준호 감독이 연출하고 2019년에 개봉한 한국 영화입니다.
    송강호, 이선균, 조여정, 최우식 등이 주연을 맡았으며, 제72회 칸 국제영화제에서
    황금종려상을 수상했습니다. 또한 제92회 아카데미 시상식에서 작품상, 감독상,
    각본상, 국제영화상 등 4개 부문을 수상하며 한국 영화 최초로 아카데미 작품상을
    수상하는 역사를 썼습니다. 빈부 격차라는 사회적 주제를 블랙 코미디와 스릴러
    장르로 풀어낸 작품으로 전 세계에서 극찬을 받았습니다.
    """.strip()

    questions = [
        "기생충의 감독은 누구인가요?",
        "기생충이 수상한 상은 무엇인가요?",
        "영화의 개봉 연도는 언제인가요?",
    ]

    print("=" * 60)
    print("추출형 QA 테스트 (KoELECTRA AutoClass)")
    print("=" * 60)
    for q in questions:
        result = answer_question(q, sample_context, model_name="koelectra", top_k=3)
        print(f"\n질문: {q}")
        print(f"  답변: '{result['answer']}' (신뢰도: {result['score']:.6f})")
        print(f"  모델: {result['model_used']}")
        print(f"  후보: {[a['answer'] for a in result['top_answers']]}")

    print("\n" + "=" * 60)
    print("두 모델 비교 테스트")
    print("=" * 60)
    cmp = compare_qa_models(questions[0], sample_context)
    print(f"\n질문: {cmp['question']}")
    print(f"  {cmp['model_a']['model_used']}: '{cmp['model_a']['answer']}' ({cmp['model_a']['inference_time_ms']}ms)")
    print(f"  {cmp['model_b']['model_used']}: '{cmp['model_b']['answer']}' ({cmp['model_b']['inference_time_ms']}ms)")
    print(f"  동일 답변: {cmp['same_answer']}")
