"""
app.py — 한국어 영화 리뷰 분석 도구 (Gradio 웹 인터페이스)

Hugging Face Spaces 배포용 진입점.
tasks/ 및 analysis/ 모듈을 통합한 단일 Gradio 앱.

실행:
    python app.py

의존성:
    pip install -r requirements.txt
"""

import json
import os
import sys
from pathlib import Path

import gradio as gr

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── 모든 태스크 모듈 임포트 (임포트 시 모델 자동 로드) ───────
print("=" * 60)
print("모델 로딩 중... (첫 실행 시 다운로드 필요, 수 분 소요)")
print("=" * 60)

from tasks.sentiment import analyze_sentiment, compare_models as sentiment_compare, LABEL_MAP
from tasks.zeroshot import (
    classify_zero_shot,
    compare_templates,
    classify_aspects,
    GENRE_LABELS,
    ASPECT_LABELS,
    KOREAN_TEMPLATE,
    ENGLISH_TEMPLATE,
)
from tasks.summarization import summarize
from tasks.qa import answer_question, build_highlighted_text
from analysis.tokenizer_analysis import (
    tokenize_and_display,
    compare_tokenizers,
    TOKENIZER_CONFIGS,
)

print("\n모든 모델 로드 완료! Gradio 앱 시작 중...\n")

# ── 비교 결과 이미지 경로 ──────────────────────────────────────
CHART_PATH = ROOT / "data" / "comparison_results.png"

# ─────────────────────────────────────────────────────────────
# 탭 1: 감성 분석
# ─────────────────────────────────────────────────────────────
def run_sentiment(text: str, model_choice: str) -> tuple:
    """Gradio 콜백: 감성 분석 실행."""
    import time
    if not text.strip():
        return {}, "입력 텍스트가 비어 있습니다."
    model_key = "default" if model_choice == "KR-FinBert" else "koelectra"
    try:
        t0 = time.perf_counter()
        result = analyze_sentiment(text, model_key)
        elapsed = (time.perf_counter() - t0) * 1000
        label_data = {result["label"]: result["score"]}
        time_str = f"{elapsed:.1f}ms | 모델: {result['model']}"
        return label_data, time_str
    except Exception as e:
        return {}, f"오류: {str(e)}"


def run_sentiment_compare(text: str) -> str:
    """두 모델 비교 결과 반환."""
    if not text.strip():
        return "입력 텍스트가 비어 있습니다."
    cmp = sentiment_compare(text)
    return json.dumps({
        "모델 A": f"{cmp['model_a']['model']} → {cmp['model_a']['label']} ({cmp['model_a']['score']:.4f}, {cmp['model_a']['inference_time_ms']}ms)",
        "모델 B": f"{cmp['model_b']['model']} → {cmp['model_b']['label']} ({cmp['model_b']['score']:.4f}, {cmp['model_b']['inference_time_ms']}ms)",
        "레이블 일치": cmp["agreement"],
    }, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# 탭 2: 제로샷 분류
# ─────────────────────────────────────────────────────────────
def run_zeroshot(
    text: str,
    selected_genres: list,
    template_lang: str,
    multi_label: bool,
) -> tuple:
    """Gradio 콜백: 제로샷 분류 실행."""
    if not text.strip():
        return {}, ""
    labels = selected_genres if selected_genres else GENRE_LABELS
    use_korean = template_lang == "한국어 템플릿"
    result = classify_zero_shot(text, labels, use_korean_template=use_korean, multi_label=multi_label)
    label_data = {l: s for l, s in zip(result["labels"], result["scores"])}
    template_info = f"사용된 템플릿: {result['template_used']}"
    return label_data, template_info


def run_template_compare(text: str) -> str:
    """한국어 vs 영어 템플릿 비교."""
    if not text.strip():
        return "입력 텍스트가 비어 있습니다."
    cmp = compare_templates(text, GENRE_LABELS)
    return json.dumps({
        "한국어 템플릿": {
            "template": cmp["korean_template"]["template"],
            "top_label": cmp["korean_template"]["top_label"],
            "top_score": cmp["korean_template"]["scores"][0],
            "inference_ms": cmp["korean_template"]["inference_time_ms"],
        },
        "영어 템플릿": {
            "template": cmp["english_template"]["template"],
            "top_label": cmp["english_template"]["top_label"],
            "top_score": cmp["english_template"]["scores"][0],
            "inference_ms": cmp["english_template"]["inference_time_ms"],
        },
        "레이블 일치": cmp["top_label_match"],
    }, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# 탭 3: 텍스트 요약
# ─────────────────────────────────────────────────────────────
def run_summarize(text: str, min_len: int, max_len: int, num_beams: int) -> tuple:
    """Gradio 콜백: 텍스트 요약 실행."""
    if not text.strip():
        return "입력 텍스트가 비어 있습니다.", {}
    result = summarize(text, min_length=int(min_len), max_length=int(max_len), num_beams=int(num_beams))
    meta = {
        "입력 길이": f"{result['input_length']}자 / {result['input_tokens']}토큰",
        "출력 길이": f"{result['output_length']}자 / {result['output_tokens']}토큰",
        "사용된 파라미터": result["params_used"],
    }
    return result["summary"], meta


# ─────────────────────────────────────────────────────────────
# 탭 4: QA
# ─────────────────────────────────────────────────────────────
def run_qa(context: str, question: str, model_choice: str, top_k: int) -> tuple:
    """Gradio 콜백: QA 실행 후 HighlightedText + JSON 반환."""
    if not context.strip() or not question.strip():
        return [(context, None)], {}
    model_key = "koelectra" if model_choice == "KoELECTRA" else "klue"
    result = answer_question(question, context, model_name=model_key, top_k=int(top_k))
    highlighted = build_highlighted_text(context, result["start"], result["end"])
    answers_json = {
        "최상위 답변": result["answer"],
        "신뢰도": result["score"],
        "모델": result["model_used"],
        "후보 답변들": [
            {"답변": a["answer"], "점수": a["score"]} for a in result["top_answers"]
        ],
    }
    return highlighted, answers_json


# ─────────────────────────────────────────────────────────────
# 탭 5: 태스크 체이닝 (확장 기능)
# ─────────────────────────────────────────────────────────────
def run_chaining(text: str) -> tuple:
    """
    확장 기능: 요약 → 제로샷 분류 → 감성 분석 순서로 태스크를 체이닝.
    각 단계의 출력이 다음 단계의 입력이 된다.
    """
    if not text.strip():
        return "입력 텍스트가 비어 있습니다.", {}, {}

    # Step 1: 긴 리뷰/기사를 요약
    summary_result = summarize(text, min_length=20, max_length=100, num_beams=4)
    summary = summary_result["summary"]

    # Step 2: 요약본에 대해 제로샷 장르 분류
    genre_result = classify_zero_shot(summary, GENRE_LABELS, use_korean_template=True)
    genre_data = {l: s for l, s in zip(genre_result["labels"][:5], genre_result["scores"][:5])}

    # Step 3: 요약본에 대해 감성 분석
    sentiment_result = analyze_sentiment(summary, "default")
    sentiment_data = {sentiment_result["label"]: sentiment_result["score"]}

    chain_log = f"[체이닝 완료]\n원문 {len(text)}자 → 요약 {len(summary)}자 → 장르: {genre_result['labels'][0]} → 감성: {sentiment_result['label']}"
    print(chain_log)

    return summary, genre_data, sentiment_data


# ─────────────────────────────────────────────────────────────
# 탭 6: 토크나이저 분석
# ─────────────────────────────────────────────────────────────
def run_tokenize(text: str, tokenizer_name: str) -> tuple:
    """단일 토크나이저 분석."""
    if not text.strip():
        return [], {}
    result = tokenize_and_display(text, tokenizer_name)

    # HighlightedText 형식: 각 토큰에 교대로 색상 레이블 부여
    highlighted = [(tok, "A" if i % 2 == 0 else "B") for i, tok in enumerate(result["tokens"])]
    meta = {
        "토크나이저": result["tokenizer"],
        "방식": result["method"],
        "언어": result["lang"],
        "토큰 수": result["num_tokens"],
        "토큰 목록": result["tokens"],
        "토큰 ID": result["token_ids"],
    }
    return highlighted, meta


def run_all_tokenizers(text: str) -> str:
    """4개 토크나이저 동시 비교."""
    if not text.strip():
        return "입력 텍스트가 비어 있습니다."
    result = compare_tokenizers(text)
    output = {"원문": text, "비교 결과": {}}
    for name, r in result["results"].items():
        output["비교 결과"][name] = {
            "방식": r["method"],
            "토큰 수": r["num_tokens"],
            "토큰": r["tokens"],
        }
    output["토큰 수 요약"] = result["summary"]
    return json.dumps(output, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# 탭 7: 모델 비교 결과
# ─────────────────────────────────────────────────────────────
COMPARISON_MARKDOWN = """
## 감성 분석 모델 비교 결과 (NSMC 100샘플)

| 모델 | 정확도 | F1 Score | 추론 속도 (ms/샘플) | 모델 크기 |
|------|--------|----------|---------------------|-----------|
| KoELECTRA-Sentiment | 실행 후 표시 | 실행 후 표시 | - | ~440MB |
| KR-FinBert-SC | 실행 후 표시 | 실행 후 표시 | - | ~440MB |

> **비교 재실행** 버튼을 눌러 실시간 평가를 수행하세요. (수 분 소요)

### 모델 선택 근거
- **KoELECTRA**: 한국어 전용 사전학습, NSMC에 파인튜닝, 빠른 추론
- **KR-FinBert**: 금융 도메인 특화이나 일반 감성 분류에도 활용 가능
- 두 모델 모두 Apache 2.0 라이선스 (상업적 사용 가능)
"""


def run_model_comparison() -> tuple:
    """실시간 모델 비교 평가 실행."""
    from analysis.model_comparison import evaluate_sentiment_models, plot_comparison_chart

    comparison = evaluate_sentiment_models(n_samples=100)
    plot_comparison_chart(comparison)

    a, b = comparison["model_a"], comparison["model_b"]
    md = f"""
## 감성 분석 모델 비교 결과 (NSMC {a['n_evaluated']}샘플)

| 모델 | 정확도 | F1 Score | 추론 속도 | 모델 크기 |
|------|--------|----------|-----------|-----------|
| {a['name']} | **{a['accuracy']:.4f}** | **{a['f1']:.4f}** | {a['inference_time_ms']}ms | {a['model_size_mb']:.0f}MB |
| {b['name']} | {b['accuracy']:.4f} | {b['f1']:.4f} | {b['inference_time_ms']}ms | {b['model_size_mb']:.0f}MB |

**선택 모델**: {'KoELECTRA' if a['accuracy'] >= b['accuracy'] else b['name']} (정확도 기준)
"""
    chart = str(CHART_PATH) if CHART_PATH.exists() else None
    return md, chart


# ─────────────────────────────────────────────────────────────
# Gradio 앱 빌드
# ─────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🎬 한국어 영화 리뷰 분석 도구",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("""
    # 🎬 한국어 영화 리뷰 분석 도구
    **Hugging Face Transformers + Gradio** 기반 멀티태스크 NLP 애플리케이션
    감성 분석 · 장르 분류 · 텍스트 요약 · 질의응답을 하나의 인터페이스에서
    """)

    with gr.Tabs():

        # ── 탭 1: 감성 분석 ──────────────────────────────────
        with gr.Tab("😊 감성 분석"):
            gr.Markdown("### 영화 리뷰의 긍정/부정 감성을 분석합니다 (Pipeline 방식)")
            with gr.Row():
                with gr.Column():
                    sent_text = gr.Textbox(label="영화 리뷰 입력", lines=4,
                                           placeholder="영화 리뷰를 입력하세요...")
                    sent_model = gr.Dropdown(
                        choices=["KR-FinBert", "KoELECTRA"],
                        value="KR-FinBert",
                        label="모델 선택",
                    )
                    with gr.Row():
                        sent_btn = gr.Button("분석 실행", variant="primary")
                        sent_cmp_btn = gr.Button("두 모델 비교")
                with gr.Column():
                    sent_label = gr.Label(label="감성 분석 결과")
                    sent_time = gr.Textbox(label="추론 정보", interactive=False)
                    sent_cmp_out = gr.Textbox(label="모델 비교 결과", lines=5, interactive=False)

            gr.Examples(
                examples=[
                    ["이 영화는 정말 감동적이었어요. 배우들의 연기가 훌륭하고 스토리도 탄탄했습니다. 강력 추천!"],
                    ["2시간 내내 지루했습니다. 스토리도 뻔하고 결말도 실망스러웠어요. 돈이 아깝네요."],
                    ["배우들 연기는 좋았지만 CGI가 조금 아쉬웠어요. 그래도 전반적으로 볼만했습니다."],
                ],
                inputs=[sent_text],
                label="예시 리뷰 클릭해보기",
            )

            sent_btn.click(run_sentiment, inputs=[sent_text, sent_model],
                           outputs=[sent_label, sent_time])
            sent_cmp_btn.click(run_sentiment_compare, inputs=[sent_text],
                               outputs=[sent_cmp_out])

        # ── 탭 2: 제로샷 분류 ────────────────────────────────
        with gr.Tab("🏷️ 장르 분류"):
            gr.Markdown("### 리뷰를 영화 장르/주제로 제로샷 분류합니다 (Pipeline 방식)")
            with gr.Row():
                with gr.Column():
                    zs_text = gr.Textbox(label="영화 리뷰 입력", lines=4,
                                         placeholder="분류할 리뷰를 입력하세요...")
                    zs_genres = gr.CheckboxGroup(
                        choices=GENRE_LABELS,
                        value=GENRE_LABELS[:4],
                        label="후보 장르 선택 (선택 안 하면 전체 사용)",
                    )
                    zs_template = gr.Radio(
                        choices=["한국어 템플릿", "영어 템플릿"],
                        value="한국어 템플릿",
                        label="템플릿 언어 선택",
                    )
                    zs_multi = gr.Checkbox(label="멀티 레이블 허용 (복수 장르)", value=False)
                    with gr.Row():
                        zs_btn = gr.Button("분류 실행", variant="primary")
                        zs_cmp_btn = gr.Button("한국어 vs 영어 템플릿 비교")
                with gr.Column():
                    zs_label = gr.Label(label="분류 결과 (확률)")
                    zs_template_info = gr.Textbox(label="사용된 템플릿", interactive=False)
                    zs_cmp_out = gr.Textbox(label="템플릿 비교 결과", lines=5, interactive=False)

            gr.Examples(
                examples=[
                    ["총격전과 폭발 장면이 압도적이었습니다. 주인공의 액션 연기가 정말 멋있었어요."],
                    ["두 남녀의 달달한 사랑 이야기에 보는 내내 설렜습니다. 엔딩에 눈물이 났어요."],
                    ["갑자기 튀어나오는 장면들에 심장이 쫄깃했어요. 혼자 보기 무서운 공포 영화입니다."],
                ],
                inputs=[zs_text],
                label="예시 리뷰 클릭해보기",
            )

            zs_btn.click(run_zeroshot, inputs=[zs_text, zs_genres, zs_template, zs_multi],
                         outputs=[zs_label, zs_template_info])
            zs_cmp_btn.click(run_template_compare, inputs=[zs_text], outputs=[zs_cmp_out])

        # ── 탭 3: 텍스트 요약 ────────────────────────────────
        with gr.Tab("📝 텍스트 요약"):
            gr.Markdown("### KoBART로 긴 리뷰를 요약합니다 (AutoClass 방식 — 생성 파라미터 직접 제어)")
            with gr.Row():
                with gr.Column():
                    sum_text = gr.Textbox(label="요약할 리뷰/기사 (100자 이상 권장)", lines=8,
                                          placeholder="긴 영화 리뷰나 기사를 입력하세요...")
                    sum_min = gr.Slider(minimum=10, maximum=50, value=30, step=5, label="최소 길이 (토큰)")
                    sum_max = gr.Slider(minimum=50, maximum=200, value=100, step=10, label="최대 길이 (토큰)")
                    sum_beams = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Beam 수 (클수록 품질↑ 속도↓)")
                    sum_btn = gr.Button("요약 실행", variant="primary")
                with gr.Column():
                    sum_out = gr.Textbox(label="요약 결과", lines=4)
                    sum_meta = gr.Textbox(label="파라미터 및 토큰 정보", lines=5, interactive=False)

            gr.Examples(
                examples=[
                    ["이 영화는 정말 오랜만에 본 수작이었습니다. 감독의 연출력이 특히 돋보였고 배우들의 연기력은 말할 것도 없이 훌륭했습니다. 특히 클라이맥스 장면에서의 긴장감은 관객을 완전히 몰입하게 만들었고, 음악도 영화의 분위기와 완벽하게 어우러졌습니다. 스토리도 탄탄하고 예측 불가능한 반전이 있어서 끝까지 눈을 뗄 수 없었습니다. 단 하나의 단점을 꼽자면 러닝타임이 조금 길다는 것인데, 그것마저도 내용이 충실하게 채워져 있어서 지루하지 않았습니다. 강력히 추천드리는 영화입니다."],
                    ["솔직히 기대했던 것보다 많이 실망스러웠습니다. 예고편에서 보여줬던 화려한 장면들이 거의 전부였고, 나머지 2시간은 지루한 전개로 가득했습니다. 배우들의 연기는 나쁘지 않았지만 대본이 너무 약했습니다. 특히 결말이 너무 급하게 처리되어서 전반부에 깔아뒀던 수많은 복선들이 제대로 회수되지 않았습니다. CGI는 화려했지만 그게 이 영화의 유일한 장점이었습니다. 다음 작품에서는 더 좋은 시나리오 작가를 기용해주길 바랍니다."],
                    ["봉준호 감독의 신작은 역시 기대를 저버리지 않았습니다. 철저한 세계관 구축과 사회적 메시지, 그리고 독창적인 연출이 조화를 이루어 완성도 높은 작품이 탄생했습니다. 주연 배우들의 앙상블 연기도 인상적이었고, 특히 중반부 반전 장면은 영화사에 길이 남을 명장면으로 기억될 것 같습니다. 다만 일부 관객에게는 메시지가 지나치게 직접적으로 느껴질 수 있다는 점은 아쉬움으로 남습니다."],
                ],
                inputs=[sum_text],
                label="예시 리뷰 클릭해보기",
            )

            sum_btn.click(run_summarize, inputs=[sum_text, sum_min, sum_max, sum_beams],
                          outputs=[sum_out, sum_meta])

        # ── 탭 4: QA ─────────────────────────────────────────
        with gr.Tab("❓ 질의응답"):
            gr.Markdown("### 영화 소개/리뷰에서 질문에 대한 답변을 추출합니다 (AutoClass 방식)")
            with gr.Row():
                with gr.Column():
                    qa_context = gr.Textbox(
                        label="컨텍스트 (영화 소개 또는 리뷰)", lines=6,
                        placeholder="답변이 포함된 텍스트를 입력하세요...",
                        value="영화 '기생충'은 봉준호 감독이 연출하고 2019년에 개봉한 한국 영화입니다. 송강호, 이선균, 조여정, 최우식 등이 주연을 맡았으며, 제72회 칸 국제영화제에서 황금종려상을 수상했습니다. 또한 제92회 아카데미 시상식에서 작품상을 포함해 4개 부문을 수상했습니다."
                    )
                    qa_question = gr.Textbox(label="질문", lines=2,
                                             placeholder="컨텍스트에 대해 질문하세요...",
                                             value="기생충의 감독은 누구인가요?")
                    qa_model = gr.Dropdown(choices=["KoELECTRA", "KLUE-RoBERTa"],
                                           value="KoELECTRA", label="모델 선택")
                    qa_topk = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="후보 답변 수")
                    qa_btn = gr.Button("질문 실행", variant="primary")
                with gr.Column():
                    qa_highlighted = gr.HighlightedText(
                        label="컨텍스트에서 답변 위치",
                        color_map={"답변": "green"},
                    )
                    qa_json = gr.Textbox(label="후보 답변 목록", lines=5, interactive=False)

            gr.Examples(
                examples=[
                    [
                        "영화 '기생충'은 봉준호 감독이 연출하고 2019년에 개봉한 한국 영화입니다. 송강호, 이선균, 조여정, 최우식 등이 주연을 맡았으며, 제72회 칸 국제영화제에서 황금종려상을 수상했습니다.",
                        "기생충이 수상한 상은 무엇인가요?",
                    ],
                    [
                        "크리스토퍼 놀란 감독의 '인터스텔라'는 2014년에 개봉한 SF 영화입니다. 매튜 맥커너히, 앤 해서웨이가 주연을 맡았으며 우주와 시간의 개념을 탐구하는 작품입니다. 제작비는 약 1억 6500만 달러였습니다.",
                        "인터스텔라의 감독은 누구인가요?",
                    ],
                    [
                        "어벤져스: 엔드게임은 2019년 개봉한 마블 스튜디오의 슈퍼히어로 영화입니다. 로버트 다우니 주니어, 크리스 에반스, 스칼렛 요한슨 등이 출연했으며 전 세계 흥행 수익 약 27억 9800만 달러를 기록했습니다.",
                        "어벤져스 엔드게임의 흥행 수익은 얼마인가요?",
                    ],
                ],
                inputs=[qa_context, qa_question],
                label="예시 클릭해보기",
            )

            qa_btn.click(run_qa, inputs=[qa_context, qa_question, qa_model, qa_topk],
                         outputs=[qa_highlighted, qa_json])

        # ── 탭 5: 태스크 체이닝 ─────────────────────────────
        with gr.Tab("⛓️ 태스크 체이닝"):
            gr.Markdown("""
            ### 확장 기능: 태스크 파이프라인 체이닝
            긴 리뷰를 입력하면 세 단계가 순서대로 자동 실행됩니다.

            **Step 1** → 요약 (KoBART)
            **Step 2** → 요약본 장르 분류 (XLM-RoBERTa)
            **Step 3** → 요약본 감성 분석 (KR-FinBert)
            """)
            with gr.Row():
                with gr.Column():
                    chain_text = gr.Textbox(
                        label="긴 영화 리뷰 또는 기사 (150자 이상 권장)", lines=10,
                        placeholder="긴 리뷰나 기사를 입력하면 요약 → 장르 분류 → 감성 분석 순으로 자동 처리됩니다..."
                    )
                    chain_btn = gr.Button("⛓️ 체이닝 실행", variant="primary", size="lg")
                with gr.Column():
                    chain_summary = gr.Textbox(label="Step 1. 요약 결과", lines=3)
                    chain_genre = gr.Label(label="Step 2. 장르 분류 (요약본 기준)")
                    chain_sentiment = gr.Label(label="Step 3. 감성 분석 (요약본 기준)")

            gr.Examples(
                examples=[
                    ["이 영화는 정말 오랜만에 본 수작이었습니다. 감독의 연출력이 특히 돋보였고 배우들의 연기력은 말할 것도 없이 훌륭했습니다. 특히 클라이맥스 장면에서의 긴장감은 관객을 완전히 몰입하게 만들었고, 음악도 영화의 분위기와 완벽하게 어우러졌습니다. 스토리도 탄탄하고 예측 불가능한 반전이 있어서 끝까지 눈을 뗄 수 없었습니다. 총격전과 폭발씬이 압도적이었고 주인공의 액션 연기가 정말 멋있었습니다. 강력히 추천드리는 영화입니다."],
                ],
                inputs=[chain_text],
                label="예시 리뷰로 체이닝 체험",
            )
            chain_btn.click(run_chaining, inputs=[chain_text],
                            outputs=[chain_summary, chain_genre, chain_sentiment])

        # ── 탭 6: 토크나이저 분석 ────────────────────────────
        with gr.Tab("🔤 토크나이저 분석"):
            gr.Markdown("""
            ### 한국어 토크나이징 특수성 분석
            교착어인 한국어가 어떻게 서브워드로 분리되는지 비교합니다.
            - **WordPiece** (KoELECTRA, BERT): `##` 접두사로 서브워드 표시
            - **SentencePiece/BPE** (KoBART, KLUE-RoBERTa): `▁`로 공백 위치 표시
            """)
            with gr.Row():
                with gr.Column():
                    tok_text = gr.Textbox(label="분석할 한국어 문장", lines=2,
                                          value="이 영화는 정말 재미있었습니다.")
                    tok_name = gr.Dropdown(
                        choices=list(TOKENIZER_CONFIGS.keys()),
                        value="KoELECTRA",
                        label="토크나이저 선택",
                    )
                    with gr.Row():
                        tok_btn = gr.Button("단일 분석", variant="primary")
                        tok_all_btn = gr.Button("4종 동시 비교")
                with gr.Column():
                    tok_highlighted = gr.HighlightedText(
                        label="토큰 분리 결과 (색상은 짝수/홀수 토큰 구분)",
                        color_map={"A": "blue", "B": "orange"},
                    )
                    tok_meta = gr.Textbox(label="상세 분석 정보", lines=5, interactive=False)
                    tok_all_out = gr.Textbox(label="4종 동시 비교 결과", lines=5, interactive=False)

            gr.Examples(
                examples=[
                    ["이 영화는 정말 재미있었습니다."],
                    ["갓생 사는 MZ세대 공감 폭발이었음 ㅋㅋ"],
                    ["먹었습니다, 먹겠습니다, 먹었을 것입니다"],
                    ["CGI 퀄리티가 헐리우드급이라 깜짝 놀랐습니다."],
                    ["OST가 너무 좋아서 스포티파이에서 바로 찾았어요."],
                ],
                inputs=[tok_text],
                label="예시 문장 클릭해보기",
            )

            tok_btn.click(run_tokenize, inputs=[tok_text, tok_name],
                          outputs=[tok_highlighted, tok_meta])
            tok_all_btn.click(run_all_tokenizers, inputs=[tok_text], outputs=[tok_all_out])

        # ── 탭 7: 모델 비교 결과 ─────────────────────────────
        with gr.Tab("📊 모델 비교"):
            gr.Markdown("""
            ### 감성 분석 모델 비교 평가
            NSMC 100샘플을 사용한 정량 비교. **비교 재실행** 버튼으로 실시간 평가 가능 (약 5~10분 소요).
            """)
            comp_md = gr.Markdown(COMPARISON_MARKDOWN)
            comp_chart = gr.Image(
                label="비교 차트",
                value=str(CHART_PATH) if CHART_PATH.exists() else None,
            )
            comp_btn = gr.Button("🔄 비교 재실행 (시간 소요)", variant="secondary")
            comp_btn.click(run_model_comparison, outputs=[comp_md, comp_chart])

    # ── 하단 면책 고지 ────────────────────────────────────────
    gr.Markdown("""
    ---
    ### ⚠️ 면책 고지
    이 애플리케이션은 **교육 목적**으로 제작된 실습 프로젝트입니다.
    모델의 예측 결과는 참고용으로만 활용하세요.

    ### 📋 사용 모델 라이선스
    | 모델 | Hub URL | 라이선스 |
    |------|---------|---------|
    | snunlp/KR-FinBert-SC | [🔗](https://huggingface.co/snunlp/KR-FinBert-SC) | Apache 2.0 |
    | monologg/koelectra-base-finetuned-sentiment | [🔗](https://huggingface.co/monologg/koelectra-base-finetuned-sentiment) | Apache 2.0 |
    | joeddav/xlm-roberta-large-xnli | [🔗](https://huggingface.co/joeddav/xlm-roberta-large-xnli) | MIT |
    | gogamza/kobart-summarization | [🔗](https://huggingface.co/gogamza/kobart-summarization) | MIT |
    | monologg/koelectra-base-v3-finetuned-korquad | [🔗](https://huggingface.co/monologg/koelectra-base-v3-finetuned-korquad) | Apache 2.0 |
    | klue/roberta-base | [🔗](https://huggingface.co/klue/roberta-base) | CC BY-SA 4.0 |

    > ⚠️ 위 라이선스는 작성 시점 기준이며 Hub에서 직접 확인 바랍니다.
    """)

# ── 앱 실행 ───────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=True, server_port=7861)
