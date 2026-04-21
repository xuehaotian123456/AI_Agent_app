"""
结构化输出效果评估脚本
"""
from rag.rag_service import RagSummarizeService
import time
import json


def evaluate_structured_output():
    rag = RagSummarizeService(enable_structured_output=True)

    test_queries = [
        "小户型适合哪些扫地机器人？",
        "扫地机器人如何避障？",
        "扫拖一体机的水箱容量多大合适？",
        "机器人报错E03是什么意思？",
    ]

    metrics = {
        "total": len(test_queries),
        "success_count": 0,
        "fail_count": 0,
        "avg_citations": 0,
        "avg_confidence": 0,
        "avg_time": 0,
        "parse_errors": 0,
    }

    print("=" * 60)
    print("结构化输出效果评估")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] 查询：{query}")

        start_time = time.time()
        try:
            result = rag.rag_summarize_structured(query, thread_id=f"eval_{i}")
            elapsed = time.time() - start_time

            structured = result.get('structured_result')

            if structured:
                metrics["success_count"] += 1
                citations_count = len(structured.get('citations', []))
                confidence = structured.get('confidence', 0)

                metrics["avg_citations"] += citations_count
                metrics["avg_confidence"] += confidence if confidence else 0

                print(f"  ✅ 成功 | 引用数：{citations_count} | 置信度：{confidence:.2f} | 耗时：{elapsed:.2f}s")
                print(f"  回答预览：{result['answer'][:80]}...")
            else:
                metrics["fail_count"] += 1
                metrics["parse_errors"] += 1
                print(f"  ⚠️  降级为纯文本 | 耗时：{elapsed:.2f}s")

        except Exception as e:
            metrics["fail_count"] += 1
            print(f"  ❌ 异常：{e}")

        metrics["avg_time"] += time.time() - start_time

    # 计算平均值
    if metrics["success_count"] > 0:
        metrics["avg_citations"] /= metrics["success_count"]
        metrics["avg_confidence"] /= metrics["success_count"]
    metrics["avg_time"] /= metrics["total"]

    # 输出总结
    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)
    print(
        f"成功率：{metrics['success_count']}/{metrics['total']} ({metrics['success_count'] / metrics['total'] * 100:.1f}%)")
    print(
        f"解析失败率：{metrics['parse_errors']}/{metrics['total']} ({metrics['parse_errors'] / metrics['total'] * 100:.1f}%)")
    print(f"平均引用数：{metrics['avg_citations']:.2f}")
    print(f"平均置信度：{metrics['avg_confidence']:.2f}")
    print(f"平均耗时：{metrics['avg_time']:.2f}s")

    # 评级
    success_rate = metrics['success_count'] / metrics['total']
    if success_rate >= 0.9:
        print("\n🎉 评级：优秀（结构化输出稳定可靠）")
    elif success_rate >= 0.7:
        print("\n👍 评级：良好（大部分情况正常工作）")
    elif success_rate >= 0.5:
        print("\n⚠️  评级：一般（需要优化 Prompt 或模型）")
    else:
        print("\n❌ 评级：较差（建议检查配置或降级使用）")

    return metrics


if __name__ == '__main__':
    evaluate_structured_output()