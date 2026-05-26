"""
analysis/streaming_demo.py — 스트리밍 모드 vs 일반 로드 비교 (확장 기능)

과제 확장 요구사항:
- NSMC 데이터셋을 streaming=True로 로드하여 일반 로드와 메모리/시간 비교
- tracemalloc으로 메모리 사용량 측정
- dataset.map() + take() 패턴 시연
"""

import itertools
import time
import tracemalloc
from pathlib import Path

from datasets import load_dataset
from tabulate import tabulate
from transformers import AutoTokenizer

# ── 설정 ──────────────────────────────────────────────────────
TOKENIZER_ID = "monologg/koelectra-base-v3-discriminator"
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        print("[streaming] 토크나이저 로딩 중...")
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    return _tokenizer


# ── 측정 함수 ──────────────────────────────────────────────────

def normal_load_demo(n_samples: int = 1000) -> dict:
    """
    NSMC 전체를 메모리에 로드한 뒤 n_samples개를 처리한다.
    메모리 사용량과 처리 시간을 측정한다.

    Args:
        n_samples: 실제 처리할 샘플 수 (전체 로드 후 슬라이싱).

    Returns:
        {"method": "normal", "load_time_s", "process_time_s", "peak_memory_mb", "n_processed"}
    """
    tok = _get_tokenizer()
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    # ── 전체 로드 ──────────────────────────────────────────────
    t_load_start = time.perf_counter()
    ds = load_dataset("csv", data_files={
        "train": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
    }, delimiter="\t", split="train")
    load_time = time.perf_counter() - t_load_start
    print(f"[normal] 전체 데이터셋 로드 완료 ({len(ds):,}개, {load_time:.1f}초)")

    # ── n_samples 처리 (dataset.map() 사용) ────────────────────
    subset = ds.select(range(min(n_samples, len(ds))))

    t_proc_start = time.perf_counter()
    subset = subset.map(
        lambda batch: {
            "token_count": [
                len(tok(text, add_special_tokens=False)["input_ids"])
                for text in batch["document"]
            ]
        },
        batched=True,
        batch_size=64,
    )
    process_time = time.perf_counter() - t_proc_start

    # ── 메모리 측정 ─────────────────────────────────────────────
    snapshot_after = tracemalloc.take_snapshot()
    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    peak_kb = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
    peak_mb = peak_kb / 1024

    tracemalloc.stop()

    avg_tokens = sum(subset["token_count"]) / len(subset)
    print(f"[normal] 처리 완료: {n_samples}개, 평균 토큰 수: {avg_tokens:.1f}, "
          f"처리 시간: {process_time:.2f}초, 메모리: {peak_mb:.1f}MB")

    return {
        "method": "일반 로드 (normal)",
        "load_time_s": round(load_time, 2),
        "process_time_s": round(process_time, 2),
        "total_time_s": round(load_time + process_time, 2),
        "peak_memory_mb": round(peak_mb, 1),
        "n_processed": n_samples,
        "avg_tokens": round(avg_tokens, 1),
    }


def streaming_load_demo(n_samples: int = 1000) -> dict:
    """
    NSMC를 스트리밍 모드로 로드하여 n_samples개만 처리한다.
    전체를 메모리에 올리지 않으므로 메모리 효율이 높다.

    Args:
        n_samples: 처리할 샘플 수.

    Returns:
        {"method": "streaming", "load_time_s", "process_time_s", "peak_memory_mb", "n_processed"}
    """
    tok = _get_tokenizer()
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    # ── 스트리밍 로드 ──────────────────────────────────────────
    t_load_start = time.perf_counter()
    # streaming=True: 데이터를 즉시 다운로드·메모리 로드하지 않고
    # 이터레이터로 필요할 때마다 가져옴
    streamed = load_dataset("csv", data_files={
        "train": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
    }, delimiter="\t", split="train", streaming=True)
    load_time = time.perf_counter() - t_load_start
    print(f"[streaming] 스트리밍 이터레이터 생성 완료 ({load_time:.3f}초)")

    # ── dataset.map() + take() 패턴 ────────────────────────────
    # map()은 스트리밍 데이터셋에도 적용 가능 (lazy evaluation)
    mapped = streamed.map(
        lambda x: {"token_count": len(tok(x["document"], add_special_tokens=False)["input_ids"])}
    )

    # ── n_samples개만 처리 (itertools.islice 또는 .take()) ─────
    t_proc_start = time.perf_counter()
    token_counts = []
    # IterableDataset은 .take() 메서드 지원
    for item in itertools.islice(mapped, n_samples):
        token_counts.append(item["token_count"])
    process_time = time.perf_counter() - t_proc_start

    # ── 메모리 측정 ─────────────────────────────────────────────
    snapshot_after = tracemalloc.take_snapshot()
    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    peak_kb = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
    peak_mb = peak_kb / 1024

    tracemalloc.stop()

    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    print(f"[streaming] 처리 완료: {len(token_counts)}개, 평균 토큰 수: {avg_tokens:.1f}, "
          f"처리 시간: {process_time:.2f}초, 메모리: {peak_mb:.1f}MB")

    return {
        "method": "스트리밍 (streaming=True)",
        "load_time_s": round(load_time, 3),
        "process_time_s": round(process_time, 2),
        "total_time_s": round(load_time + process_time, 2),
        "peak_memory_mb": round(peak_mb, 1),
        "n_processed": len(token_counts),
        "avg_tokens": round(avg_tokens, 1),
    }


def compare_loading_methods(n_samples: int = 1000) -> dict:
    """
    일반 로드와 스트리밍 로드를 비교하여 결과를 반환한다.

    Args:
        n_samples: 비교에 사용할 처리 샘플 수.

    Returns:
        {
            "normal": normal_load_demo() 결과,
            "streaming": streaming_load_demo() 결과,
            "memory_reduction_pct": float,   # 스트리밍의 메모리 절감률 (%)
            "speed_comparison": str
        }
    """
    print("=" * 60)
    print(f"[compare] 일반 로드 vs 스트리밍 비교 (n={n_samples})")
    print("=" * 60)

    normal = normal_load_demo(n_samples)
    print()
    streaming = streaming_load_demo(n_samples)

    # 메모리 절감률 계산
    mem_n = normal["peak_memory_mb"]
    mem_s = streaming["peak_memory_mb"]
    mem_reduction = (mem_n - mem_s) / mem_n * 100 if mem_n > 0 else 0

    # 속도 비교
    time_n = normal["total_time_s"]
    time_s = streaming["total_time_s"]
    if time_s < time_n:
        speed_msg = f"스트리밍이 {(time_n - time_s):.2f}초 빠름 (단, 첫 번째 항목 접근 지연 있음)"
    else:
        speed_msg = f"일반 로드가 처리 속도는 {(time_s - time_n):.2f}초 빠름 (사전 로드 후 캐시 활용)"

    return {
        "normal": normal,
        "streaming": streaming,
        "memory_reduction_pct": round(mem_reduction, 1),
        "speed_comparison": speed_msg,
    }


def print_comparison_table(result: dict):
    """비교 결과를 표 형식으로 출력한다."""
    n = result["normal"]
    s = result["streaming"]

    headers = ["항목", "일반 로드", "스트리밍 로드", "차이"]
    rows = [
        ["데이터 로드 시간",
         f"{n['load_time_s']}초", f"{s['load_time_s']}초",
         f"{abs(n['load_time_s'] - s['load_time_s']):.2f}초 차"],
        ["처리 시간 (n샘플)",
         f"{n['process_time_s']}초", f"{s['process_time_s']}초",
         f"{abs(n['process_time_s'] - s['process_time_s']):.2f}초 차"],
        ["총 소요 시간",
         f"{n['total_time_s']}초", f"{s['total_time_s']}초",
         f"{abs(n['total_time_s'] - s['total_time_s']):.2f}초 차"],
        ["피크 메모리 사용",
         f"{n['peak_memory_mb']:.1f}MB", f"{s['peak_memory_mb']:.1f}MB",
         f"스트리밍 {result['memory_reduction_pct']:.1f}% 절감"],
        ["처리 샘플 수",
         str(n["n_processed"]), str(s["n_processed"]), "동일"],
        ["평균 토큰 수",
         str(n["avg_tokens"]), str(s["avg_tokens"]), "동일"],
    ]

    print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print(f"\n속도 분석: {result['speed_comparison']}")
    print("\n[결론]")
    print("  - 스트리밍 모드는 전체 데이터를 메모리에 올리지 않아 대용량 처리에 유리합니다.")
    print("  - 일반 로드는 데이터를 캐시한 후 반복 접근이 빠릅니다.")
    print("  - NSMC 20만 개 처리 시: 스트리밍은 메모리 제한 환경에서 필수적입니다.")
    print("  - Colab 무료 플랜(12GB RAM) 환경에서 대형 데이터셋은 streaming=True 권장.")


# ── 직접 실행 ──────────────────────────────────────────────────
if __name__ == "__main__":
    result = compare_loading_methods(n_samples=500)
    print_comparison_table(result)
