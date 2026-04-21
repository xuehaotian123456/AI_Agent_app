"""
基于 eval_dataset.json 的 API 评估脚本

评估维度：
1) 接口可用性与成功率
2) 召回命中率（引用 source 是否命中标注的 relevant_ids/source）
3) "资料不足" 比例
4) 响应时延（平均/P95）
5) 失败样本导出
"""

import argparse
import json
import os
import statistics
import time
import uuid
from typing import Any, Dict, List, Tuple

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG API with dataset")
    parser.add_argument(
        "--dataset",
        default="data/eval_dataset.json",
        help="评测数据集路径（默认: data/eval_dataset.json）",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://localhost:8010"),
        help="API 地址（默认读取 BASE_URL，否则 http://localhost:8010）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="单请求超时时间（秒）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评测前 N 条（0 表示全部）",
    )
    parser.add_argument(
        "--output",
        default="eval_api_report.json",
        help="评测报告输出路径（默认: eval_api_report.json）",
    )
    parser.add_argument(
        "--failures-output",
        default="eval_api_failures.json",
        help="失败样本输出路径（默认: eval_api_failures.json）",
    )
    return parser.parse_args()


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("数据集格式错误：根节点应为 list")
    return data


def normalize_text(s: str) -> str:
    return (s or "").replace("\\", "/").strip().lower()


def source_hit(
    citations: List[Dict[str, Any]],
    expected_source: str,
    relevant_ids: List[str],
) -> Tuple[bool, List[str]]:
    citation_sources = [normalize_text(c.get("source", "")) for c in citations]
    expected_source_n = normalize_text(expected_source)
    relevant_ids_n = [normalize_text(x) for x in relevant_ids]

    matched_items: List[str] = []
    for src in citation_sources:
        if expected_source_n and expected_source_n in src:
            matched_items.append(expected_source)
            continue
        for rid in relevant_ids_n:
            if rid and rid in src:
                matched_items.append(rid)
                break

    return (len(matched_items) > 0, matched_items)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]

    total = len(dataset)
    if total == 0:
        raise ValueError("数据集为空，无法评估")

    print("=" * 72)
    print("API 评测开始")
    print("=" * 72)
    print(f"BASE_URL       : {args.base_url}")
    print(f"DATASET        : {args.dataset}")
    print(f"SAMPLES        : {total}")
    print(f"TIMEOUT(sec)   : {args.timeout}")
    print()

    metrics: Dict[str, Any] = {
        "total": total,
        "http_200": 0,
        "http_non_200": 0,
        "exceptions": 0,
        "source_hit_count": 0,
        "source_hit_rate": 0.0,
        "insufficient_count": 0,
        "insufficient_rate": 0.0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "avg_retry_count": 0.0,
        "corrected_count": 0,
        "corrected_rate": 0.0,
        "model_used_distribution": {},
        "retrieval_method_distribution": {},
    }

    latencies: List[float] = []
    retry_counts: List[int] = []
    failures: List[Dict[str, Any]] = []

    for idx, item in enumerate(dataset, 1):
        query = item.get("query", "").strip()
        expected_source = item.get("source", "")
        relevant_ids = item.get("relevant_ids", [])
        sample_id = f"eval_{idx}_{uuid.uuid4().hex[:8]}"

        payload = {
            "query": query,
            "user_id": "dataset_eval_user",
            "session_id": sample_id,
            "use_history": False,
            "enable_correction": True,
        }

        start = time.time()
        try:
            resp = requests.post(
                f"{args.base_url}/api/v1/chat",
                json=payload,
                timeout=args.timeout,
            )
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)

            if resp.status_code != 200:
                metrics["http_non_200"] += 1
                failures.append(
                    {
                        "idx": idx,
                        "query": query,
                        "reason": "http_non_200",
                        "status_code": resp.status_code,
                        "response_text": resp.text[:1000],
                    }
                )
                print(f"[{idx}/{total}] FAIL HTTP {resp.status_code} | {query}")
                continue

            metrics["http_200"] += 1
            data = resp.json()

            answer = (data.get("answer") or "").strip()
            citations = data.get("citations") or []
            retry_count = int(data.get("retry_count") or 0)
            corrected = bool(data.get("corrected"))
            model_used = (data.get("model_used") or "unknown").strip()
            retrieval_method = ((data.get("retrieval_metadata") or {}).get("method") or "unknown").strip()

            retry_counts.append(retry_count)
            if corrected:
                metrics["corrected_count"] += 1

            metrics["model_used_distribution"][model_used] = metrics["model_used_distribution"].get(model_used, 0) + 1
            metrics["retrieval_method_distribution"][retrieval_method] = (
                metrics["retrieval_method_distribution"].get(retrieval_method, 0) + 1
            )

            # 资料不足统计
            if "资料不足" in answer:
                metrics["insufficient_count"] += 1

            # source hit 统计
            hit, matched_items = source_hit(citations, expected_source, relevant_ids)
            if hit:
                metrics["source_hit_count"] += 1
            else:
                failures.append(
                    {
                        "idx": idx,
                        "query": query,
                        "reason": "source_not_hit",
                        "expected_source": expected_source,
                        "relevant_ids": relevant_ids,
                        "citations": citations,
                        "answer_preview": answer[:300],
                        "retrieval_metadata": data.get("retrieval_metadata"),
                    }
                )

            print(
                f"[{idx}/{total}] OK | hit={hit} | retry={retry_count} | "
                f"method={retrieval_method} | model={model_used} | {query}"
            )

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)
            metrics["exceptions"] += 1
            failures.append(
                {
                    "idx": idx,
                    "query": query,
                    "reason": "exception",
                    "error": str(e),
                }
            )
            print(f"[{idx}/{total}] EXCEPTION | {query} | {e}")

    # 汇总
    metrics["http_non_200"] = total - metrics["http_200"] - metrics["exceptions"]
    metrics["source_hit_rate"] = round(metrics["source_hit_count"] / total, 4)
    metrics["insufficient_rate"] = round(metrics["insufficient_count"] / total, 4)
    metrics["corrected_rate"] = round(metrics["corrected_count"] / total, 4)
    metrics["avg_latency_ms"] = round(statistics.mean(latencies), 2) if latencies else 0.0
    metrics["p95_latency_ms"] = round(percentile(latencies, 0.95), 2) if latencies else 0.0
    metrics["avg_retry_count"] = round(statistics.mean(retry_counts), 3) if retry_counts else 0.0

    report = {
        "summary": metrics,
        "base_url": args.base_url,
        "dataset": args.dataset,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "failure_count": len(failures),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(args.failures_output, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 72)
    print("API 评测完成")
    print("=" * 72)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告文件: {args.output}")
    print(f"失败样本: {args.failures_output}")


if __name__ == "__main__":
    main()
