"""
analysis/tokenizer_analysis.py — 한국어 토크나이저 특수성 분석

과제 요구사항: "교착어 한국어의 특수성을 직접 코드로 시연"
- 영어 모델(BERT) vs 한국어 모델 토크나이저 비교
- 형태소 단위 분리 패턴 관찰
- OOV/신조어/외래어 처리 방식 확인
- WordPiece / SentencePiece / BPE 방식 차이 비교
"""

from typing import Optional
from transformers import AutoTokenizer

# ── 분석 대상 토크나이저 ────────────────────────────────────────
TOKENIZER_CONFIGS = {
    "KoELECTRA": {
        "model_id": "monologg/koelectra-base-v3-discriminator",
        "method": "WordPiece",
        "lang": "한국어",
        "subword_prefix": "##",   # WordPiece의 서브워드 접두사
    },
    "KoBART": {
        "model_id": "gogamza/kobart-summarization",
        "method": "SentencePiece (BPE)",
        "lang": "한국어",
        "subword_prefix": "▁",    # SentencePiece의 공백 표시
    },
    "KLUE-RoBERTa": {
        "model_id": "klue/roberta-base",
        "method": "BPE (Byte-Pair Encoding)",
        "lang": "한국어",
        "subword_prefix": "▁",
    },
    "BERT(영어)": {
        "model_id": "bert-base-uncased",
        "method": "WordPiece",
        "lang": "영어",
        "subword_prefix": "##",
    },
}

# ── 분석용 예시 문장 ────────────────────────────────────────────
SAMPLE_SENTENCES = [
    "이 영화는 정말 재미있었습니다.",                              # 기본 문장
    "배우들의 연기가 너무 훌륭했어요.",                           # 조사·어미 변화
    "스토리가 조금 아쉬웠지만 영상미는 최고였다.",               # 복합 문장
    "갓생 사는 주인공 캐릭터가 MZ세대 공감 폭발이었음 ㅋㅋ",    # 신조어·준말체
    "CGI 퀄리티가 헐리우드급이라 깜짝 놀랐습니다.",              # 영문+한국어 혼합
    "OST가 너무 좋아서 영화 보고 바로 스포티파이에서 찾았어요.", # 외래어
    "먹었습니다, 먹겠습니다, 먹었을 것입니다",                   # 동사 활용형 비교
]

# ── 토크나이저 캐시 ─────────────────────────────────────────────
_tokenizers: dict = {}


def _get_tokenizer(name: str) -> AutoTokenizer:
    """토크나이저를 캐싱하여 반환."""
    if name not in _tokenizers:
        cfg = TOKENIZER_CONFIGS[name]
        print(f"[tokenizer_analysis] {name} ({cfg['method']}) 로딩 중...")
        _tokenizers[name] = AutoTokenizer.from_pretrained(cfg["model_id"])
        print(f"[tokenizer_analysis] {name} 로드 완료.")
    return _tokenizers[name]


# ── 공개 API ────────────────────────────────────────────────────

def tokenize_and_display(text: str, tokenizer_name: str) -> dict:
    """
    주어진 텍스트를 특정 토크나이저로 토큰화하고 결과를 반환한다.

    Args:
        text: 토큰화할 텍스트.
        tokenizer_name: TOKENIZER_CONFIGS의 키 중 하나.

    Returns:
        {
            "tokens": list[str],
            "token_ids": list[int],
            "num_tokens": int,
            "tokenizer": str,
            "method": str,
            "subword_prefix": str
        }
    """
    if tokenizer_name not in TOKENIZER_CONFIGS:
        raise ValueError(f"지원하지 않는 토크나이저: {tokenizer_name}. 가능: {list(TOKENIZER_CONFIGS)}")

    tok = _get_tokenizer(tokenizer_name)
    encoding = tok(text, add_special_tokens=False)
    tokens = tok.convert_ids_to_tokens(encoding["input_ids"])
    cfg = TOKENIZER_CONFIGS[tokenizer_name]

    return {
        "tokens": tokens,
        "token_ids": encoding["input_ids"],
        "num_tokens": len(tokens),
        "tokenizer": tokenizer_name,
        "method": cfg["method"],
        "subword_prefix": cfg["subword_prefix"],
        "lang": cfg["lang"],
    }


def compare_tokenizers(text: str) -> dict:
    """
    4개 토크나이저 모두로 동일 텍스트를 토큰화하고 결과를 비교한다.

    Args:
        text: 비교 분석할 텍스트.

    Returns:
        {
            "text": str,
            "results": { 토크나이저명: tokenize_and_display() 결과 },
            "summary": { 토크나이저명: 토큰 수 }
        }
    """
    results = {}
    for name in TOKENIZER_CONFIGS:
        results[name] = tokenize_and_display(text, name)

    return {
        "text": text,
        "results": results,
        "summary": {name: results[name]["num_tokens"] for name in results},
    }


def analyze_korean_morphology(texts: Optional[list[str]] = None) -> list[dict]:
    """
    한국어 문장들의 형태소 분리 패턴을 KoELECTRA 토크나이저로 분석한다.
    교착어 특성(조사·어미·활용형)이 어떻게 서브워드로 분리되는지 관찰.

    Args:
        texts: 분석할 문장 리스트. None이면 SAMPLE_SENTENCES 사용.

    Returns:
        각 문장에 대한 분석 결과 리스트.
        [
            {
                "sentence": str,
                "tokens_koelectra": list[str],
                "num_tokens": int,
                "subword_analysis": str,    # 서브워드 분리 패턴 설명
                "oov_tokens": list[str],    # [UNK] 처리된 토큰
            },
            ...
        ]
    """
    if texts is None:
        texts = SAMPLE_SENTENCES

    tok = _get_tokenizer("KoELECTRA")
    results = []

    for sentence in texts:
        encoding = tok(sentence, add_special_tokens=False)
        tokens = tok.convert_ids_to_tokens(encoding["input_ids"])

        # ## 접두사가 붙은 서브워드 토큰들 (WordPiece 분리 결과)
        subwords = [t for t in tokens if t.startswith("##")]
        # [UNK] 처리된 OOV 토큰
        unk_count = tokens.count("[UNK]")

        # 서브워드 비율로 분리 복잡도 측정
        subword_ratio = len(subwords) / len(tokens) if tokens else 0

        results.append({
            "sentence": sentence,
            "tokens_koelectra": tokens,
            "num_tokens": len(tokens),
            "subword_count": len(subwords),
            "subword_ratio": round(subword_ratio, 3),
            "subword_analysis": (
                f"서브워드 {len(subwords)}개 / 전체 {len(tokens)}개 "
                f"({subword_ratio:.1%}) — "
                + ("높은 형태소 분해" if subword_ratio > 0.4 else "낮은 형태소 분해")
            ),
            "oov_count": unk_count,
            "oov_tokens": [sentence.split()[i] for i in range(min(unk_count, len(sentence.split())))
                           if i < len(tokens) and tokens[i] == "[UNK]"],
        })

    return results


def korean_vs_english_comparison(sentences: Optional[list[str]] = None) -> list[dict]:
    """
    동일 한국어 문장을 KoELECTRA와 BERT(영어) 토크나이저로 토큰화하여 비교한다.
    핵심 관찰: 한국어를 영어 토크나이저로 처리하면 토큰이 폭발적으로 늘어난다.

    Args:
        sentences: 비교할 문장 리스트. None이면 SAMPLE_SENTENCES[:5] 사용.

    Returns:
        [
            {
                "sentence": str,
                "korean_tokens": list[str],
                "english_tokens": list[str],
                "korean_count": int,
                "english_count": int,
                "token_ratio": float,   # 영어/한국어 토큰 수 비율 (>1이면 영어가 더 많음)
                "efficiency": str       # 해석 메시지
            },
            ...
        ]
    """
    if sentences is None:
        sentences = SAMPLE_SENTENCES[:5]

    ko_tok = _get_tokenizer("KoELECTRA")
    en_tok = _get_tokenizer("BERT(영어)")
    results = []

    for sentence in sentences:
        ko_enc = ko_tok(sentence, add_special_tokens=False)
        en_enc = en_tok(sentence, add_special_tokens=False)

        ko_tokens = ko_tok.convert_ids_to_tokens(ko_enc["input_ids"])
        en_tokens = en_tok.convert_ids_to_tokens(en_enc["input_ids"])

        ratio = len(en_tokens) / len(ko_tokens) if ko_tokens else 0

        results.append({
            "sentence": sentence,
            "korean_tokens": ko_tokens,
            "english_tokens": en_tokens,
            "korean_count": len(ko_tokens),
            "english_count": len(en_tokens),
            "token_ratio": round(ratio, 2),
            "efficiency": (
                f"영어 모델이 {ratio:.1f}배 더 많은 토큰 사용 "
                f"→ 한국어 전용 토크나이저가 훨씬 효율적"
                if ratio > 1.5
                else f"토큰 수 유사 (비율: {ratio:.2f})"
            ),
        })

    return results


def print_analysis_report():
    """
    전체 토크나이저 분석 결과를 콘솔에 보기 좋게 출력한다.
    보고서 및 데모 영상 시연용.
    """
    sep = "=" * 65

    print(f"\n{sep}")
    print("  한국어 토크나이저 분석 리포트")
    print(f"{sep}\n")

    # 1. 단일 문장 4종 비교
    sample = "이 영화는 정말 재미있었습니다."
    print(f"[1] '{sample}' — 4종 토크나이저 비교\n")
    cmp = compare_tokenizers(sample)
    for name, result in cmp["results"].items():
        print(f"  [{name}] ({result['method']}, {result['lang']})")
        print(f"    토큰 수: {result['num_tokens']}")
        print(f"    토큰: {result['tokens']}\n")

    # 2. 형태소 분석 (7개 문장)
    print(f"{sep}")
    print("[2] 한국어 형태소 분리 패턴 (KoELECTRA WordPiece)\n")
    morphology = analyze_korean_morphology()
    for item in morphology:
        print(f"  문장: {item['sentence']}")
        print(f"  토큰: {item['tokens_koelectra']}")
        print(f"  분석: {item['subword_analysis']}")
        if item["oov_count"] > 0:
            print(f"  ⚠ OOV 토큰 수: {item['oov_count']}")
        print()

    # 3. 한국어 vs 영어 토크나이저 비교
    print(f"{sep}")
    print("[3] 한국어 vs 영어(BERT) 토크나이저 효율 비교\n")
    comparison = korean_vs_english_comparison()
    for item in comparison:
        print(f"  문장: {item['sentence']}")
        print(f"  한국어 모델: {item['korean_count']}개 토큰 {item['korean_tokens']}")
        print(f"  영어  모델: {item['english_count']}개 토큰 {item['english_tokens']}")
        print(f"  → {item['efficiency']}\n")

    # 4. 신조어·외래어 처리
    print(f"{sep}")
    print("[4] 신조어·외래어 OOV 처리 분석\n")
    oov_sentences = [
        "갓생 사는 MZ세대 요즘 트렌드",
        "스포티파이 알고리즘이 갓벽하다",
        "이번 영화 CGI 레전드 ㅋㅋㅋ",
    ]
    for sentence in oov_sentences:
        for name in ("KoELECTRA", "BERT(영어)"):
            result = tokenize_and_display(sentence, name)
            unk = result["tokens"].count("[UNK]")
            print(f"  [{name}] '{sentence}'")
            print(f"    토큰: {result['tokens']}")
            print(f"    [UNK] 개수: {unk}\n")


# ── 직접 실행 ───────────────────────────────────────────────────
if __name__ == "__main__":
    print_analysis_report()
