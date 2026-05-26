# 🎬 한국어 영화 리뷰 분석 도구

> Hugging Face Transformers + Gradio 기반 멀티태스크 NLP 애플리케이션  
> KoELECTRA · XLM-RoBERTa · KoBART · KLUE-RoBERTa를 통합한 한국어 AI 도구상자
> hugging face url: https://huggingface.co/spaces/simjaewook/korean_movie_review_nlp<img width="468" height="69" alt="image" src="https://github.com/user-attachments/assets/c501bcd1-46f9-4e85-8fac-4a9896755dad" />


---

## 📌 프로젝트 소개

이 프로젝트는 **영화 리뷰 도메인**에서 4가지 NLP 태스크를 하나의 Gradio 인터페이스로 통합한 애플리케이션입니다.  
수업에서 배운 전이학습(Transfer Learning) 패러다임을 직접 적용하여,  
한국어 사전학습 모델을 영화 리뷰 분석 도구로 활용합니다.

### 통합 태스크

| 태스크 | 구현 방식 | 사용 모델 |
|--------|----------|---------|
| 😊 감성 분석 (Sentiment Analysis) | Pipeline | KR-FinBert-SC, KoELECTRA-Sentiment |
| 🏷️ 장르 분류 (Zero-Shot Classification) | Pipeline | XLM-RoBERTa-XNLI |
| 📝 텍스트 요약 (Summarization) | **AutoClass** | KoBART |
| ❓ 추출형 QA (Extractive QA) | **AutoClass** | KoELECTRA-KorQuAD, KLUE-RoBERTa |

### 확장 기능

- ⛓️ **태스크 체이닝**: 요약 → 장르 분류 → 감성 분석 자동 연결
- 🌊 **스트리밍 모드**: NSMC 대용량 데이터셋 streaming=True 처리 (`analysis/streaming_demo.py`)
- 📊 **모델 비교 평가**: NSMC 100샘플 기준 정확도·F1·추론속도·모델크기 비교
- 🔤 **토크나이저 분석**: 4종 토크나이저 한국어 특수성 비교 시연

---

## 🤖 사용한 모델 목록

> ⚠️ 아래 라이선스는 작성 시점(2024년) 기준이며, **실제 사용 전 Hub에서 직접 확인**하세요.

| 모델 | Hub URL | 라이선스 | 용도 |
|------|---------|---------|------|
| snunlp/KR-FinBert-SC | [🔗 Hub](https://huggingface.co/snunlp/KR-FinBert-SC) | Apache 2.0 | 감성 분석 (기본 모델) |
| monologg/koelectra-base-finetuned-sentiment | [🔗 Hub](https://huggingface.co/monologg/koelectra-base-finetuned-sentiment) | Apache 2.0 | 감성 분석 (비교 모델) |
| joeddav/xlm-roberta-large-xnli | [🔗 Hub](https://huggingface.co/joeddav/xlm-roberta-large-xnli) | MIT | 제로샷 분류 |
| gogamza/kobart-summarization | [🔗 Hub](https://huggingface.co/gogamza/kobart-summarization) | MIT | 텍스트 요약 |
| monologg/koelectra-base-v3-finetuned-korquad | [🔗 Hub](https://huggingface.co/monologg/koelectra-base-v3-finetuned-korquad) | Apache 2.0 | 추출형 QA (기본 모델) |
| klue/roberta-base | [🔗 Hub](https://huggingface.co/klue/roberta-base) | CC BY-SA 4.0 | QA 비교 모델, 토크나이저 분석 |
| monologg/koelectra-base-v3-discriminator | [🔗 Hub](https://huggingface.co/monologg/koelectra-base-v3-discriminator) | Apache 2.0 | 토크나이저 분석 |
| bert-base-uncased | [🔗 Hub](https://huggingface.co/google-bert/bert-base-uncased) | Apache 2.0 | 영어 토크나이저 비교용 |

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
# Python 3.9+ 권장
git clone https://github.com/YOUR_USERNAME/korean-movie-nlp.git
cd korean-movie-nlp

pip install -r requirements.txt

# Ubuntu/Colab에서 matplotlib 한글 폰트 설치 (선택)
sudo apt-get install -y fonts-nanum
```

### 2. 앱 실행

```bash
python app.py
```

브라우저에서 `http://localhost:7860` 접속

> ⚠️ 첫 실행 시 모델 다운로드로 수 분이 소요됩니다.  
> 모델은 `~/.cache/huggingface/`에 저장되며, 이후 실행 시 재다운로드 불필요.

### 3. 개별 모듈 테스트

```bash
# 각 태스크 단독 실행
python tasks/sentiment.py
python tasks/zeroshot.py
python tasks/summarization.py
python tasks/qa.py

# 분석 모듈
python analysis/tokenizer_analysis.py
python analysis/model_comparison.py     # NSMC 100샘플 평가 (수 분 소요)
python analysis/streaming_demo.py       # 스트리밍 vs 일반 로드 비교
```

---

## 📂 프로젝트 구조

```
korean-movie-nlp/
├── README.md
├── requirements.txt
├── app.py                          # Gradio 앱 진입점 (Spaces 배포용)
│
├── tasks/
│   ├── sentiment.py                # 감성 분석 (Pipeline 방식)
│   ├── zeroshot.py                 # 제로샷 분류 (Pipeline 방식)
│   ├── summarization.py            # 텍스트 요약 (AutoClass 방식)
│   └── qa.py                       # 추출형 QA (AutoClass 방식)
│
├── analysis/
│   ├── tokenizer_analysis.py       # 한국어 토크나이저 분석
│   ├── model_comparison.py         # 모델 성능 비교 평가
│   └── streaming_demo.py           # 스트리밍 모드 vs 일반 로드 비교
│
└── data/
    └── comparison_results.png      # 모델 비교 차트 (자동 생성)
```

---

## 🔑 Pipeline vs AutoClass 선택 근거

| 태스크 | 선택 | 이유 |
|--------|------|------|
| 감성 분석 | Pipeline | 텍스트→레이블+확률의 표준화된 흐름. 빠른 프로토타이핑 우선. |
| 제로샷 분류 | Pipeline | NLI 헤드의 entailment 확률 계산이 파이프라인에 완전히 추상화됨. |
| 텍스트 요약 | **AutoClass** | `generate()` 파라미터(num_beams, length_penalty 등) 직접 제어 필요. |
| 추출형 QA | **AutoClass** | start/end logits에 직접 접근하여 top-k 후보 추출 및 오프셋 매핑 필요. |

---

## 🌏 Hugging Face Spaces 배포

```bash
# Spaces 배포 시
# 1. README.md 상단에 아래 메타데이터 추가:
#    ---
#    title: Korean Movie Review Analyzer
#    sdk: gradio
#    sdk_version: 4.29.0
#    app_file: app.py
#    ---
# 2. requirements.txt 포함하여 Push
# 3. Spaces URL: https://huggingface.co/spaces/YOUR_USERNAME/korean-movie-nlp
```

---

## 🤖 AI 도구 활용 고지

이 프로젝트 개발 과정에서 **Claude (Anthropic)** 를 다음 목적으로 활용했습니다:
- 코드 구조 설계 및 초안 작성
- 디버깅 및 오류 수정 방향 탐색
- 문서화(README, docstring) 초안 작성

AI가 생성한 코드는 직접 실행·검증하고 수정하였으며,  
핵심 설계 결정(모델 선택, Pipeline/AutoClass 선택, 한국어 템플릿 설계 등)은 직접 판단했습니다.

---

## ⚠️ 면책 고지

- 이 애플리케이션은 **교육 목적의 실습 프로젝트**입니다.
- 모델 예측 결과는 참고용이며 최종 판단은 사용자가 직접 해야 합니다.
- 사용된 모델들은 학습 데이터의 특성에 따라 편향이 있을 수 있습니다.
- 의료·법률·금융 등 민감 분야에는 사용하지 마세요.
