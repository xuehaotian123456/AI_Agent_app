#!/usr/bin/env python3
"""
Multi-Agent 知识问答系统 — 四智能体协作演示

演示内容：
1. 基础知识问答（标准四Agent流程）
2. 工具调用展示（查询设备规格 + 混合检索）
3. 故障诊断场景（触发troubleshoot_issue工具）
4. 多轮对话（带历史上下文的查询改写）

运行方式：
    cd langgraph-local
    python scripts/demo_multi_agent.py

注意：需要配置 DASHSCOPE_API_KEY 环境变量
"""
import sys
import time
import os

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_separator(title: str):
    """打印分隔线"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_agent_stage(emoji: str, name: str, content: str, indent: int = 0):
    """格式化输出Agent阶段信息"""
    prefix = "  " * indent
    print(f"{prefix}{emoji} [{name}] {content}")


def demo_basic_qa():
    """演示1：基础知识问答"""
    print_separator("演示1: 基础知识问答 — 小户型扫地机器人选购")

    from agent.multi_agent_orchestrator import MultiAgentOrchestrator

    print("🔧 初始化 Multi-Agent 编排器...")
    try:
        orch = MultiAgentOrchestrator()
    except Exception as e:
        print(f"⚠️ 初始化失败（可能缺少API Key）: {e}")
        print("  演示将以模拟模式继续...")
        _demo_simulated()
        return

    query = "小户型适合哪种扫地机器人？需要静音的"
    print(f"\n📝 用户查询: {query}\n")

    start = time.time()
    result = orch.chat(query, session_id="demo_basic")

    # Planner
    plan = result.plan or {}
    print_agent_stage("🧠", "Planner", f"意图: {plan.get('intent', '?')} | 子任务数: {len(plan.get('sub_tasks', []))}")
    for st in plan.get("sub_tasks", []):
        print_agent_stage("", "", f"  · {st.get('task_id', '?')}: {st.get('description', '')[:60]}", indent=1)
        if st.get("required_tools"):
            print_agent_stage("", "", f"    工具: {', '.join(st['required_tools'])}", indent=1)

    # Retriever
    rounds = result.retrieval_rounds or []
    for r in rounds:
        print_agent_stage("🔍", "Retriever", f"第{r.get('round_number', '?')}轮: {r.get('docs_count', 0)}篇文档, KG={r.get('kg_used', False)}")

    # Tool Calls
    tool_calls = result.tool_calls or []
    for tc in tool_calls:
        status = "✓" if tc.get("success") else "✗"
        print_agent_stage("🔧", "Tool", f"{tc.get('tool', '?')} {status} — {tc.get('result_summary', '')}")

    # Reflector
    reflection = result.reflection or {}
    print_agent_stage("🤔", "Reflector", f"判定: {reflection.get('verdict', '?')} | 置信度: {reflection.get('confidence', '?')}")

    # Summarizer
    print_agent_stage("📝", "Summarizer", f"答案长度: {len(result.answer)}字 | 引用: {len(result.citations)}个")
    print(f"\n{'─'*40}")
    print(f"📄 最终答案:")
    print(f"{result.answer[:500]}")
    if len(result.citations) > 0:
        print(f"\n📚 引用来源:")
        for i, cit in enumerate(result.citations[:3], 1):
            print(f"  {i}. {cit.get('source', '?')}: {cit.get('content', '')[:80]}...")

    elapsed = time.time() - start
    print(f"\n⏱️ 总耗时: {elapsed:.2f}s | 重试: {result.retry_count}次 | 置信度: {result.confidence}")


def demo_troubleshoot():
    """演示2：故障诊断 + 工具调用"""
    print_separator("演示2: 故障诊断 — 充电慢问题")

    from agent.multi_agent_orchestrator import MultiAgentOrchestrator

    try:
        orch = MultiAgentOrchestrator()
    except Exception as e:
        print(f"⚠️ 初始化失败: {e}")
        return

    query = "我的石头P10最近充电很慢，充一晚上都充不满，怎么办？"
    print(f"\n📝 用户查询: {query}\n")

    start = time.time()
    result = orch.chat(query, session_id="demo_trouble")

    plan = result.plan or {}
    print_agent_stage("🧠", "Planner", f"意图: {plan.get('intent', '?')}")

    tool_calls = result.tool_calls or []
    print_agent_stage("🔧", "工具调用", f"共 {len(tool_calls)} 次")
    for tc in tool_calls:
        status = "✓" if tc.get("success") else "✗"
        print_agent_stage("", "", f"  {tc.get('tool', '?')} {status} ({tc.get('elapsed_ms', 0)}ms)", indent=1)

    reflection = result.reflection or {}
    print_agent_stage("🤔", "Reflector", f"判定: {reflection.get('verdict', '?')} | 置信度: {reflection.get('confidence', '?')}")

    print(f"\n{'─'*40}")
    print(f"📄 最终答案:")
    print(f"{result.answer[:500]}")

    elapsed = time.time() - start
    print(f"\n⏱️ 总耗时: {elapsed:.2f}s | 重试: {result.retry_count}")


def demo_multi_turn():
    """演示3：多轮对话"""
    print_separator("演示3: 多轮对话 — 上下文保持")

    from agent.multi_agent_orchestrator import MultiAgentOrchestrator

    try:
        orch = MultiAgentOrchestrator()
    except Exception as e:
        print(f"⚠️ 初始化失败: {e}")
        return

    session_id = "demo_multi_turn"

    # 第一轮
    q1 = "石头P10有什么特色功能？"
    print(f"\n📝 第1轮: {q1}")
    r1 = orch.chat(q1, session_id=session_id)
    print(f"   回答: {r1.answer[:150]}...")

    # 第二轮（依赖上下文）
    q2 = "它的吸力够不够大户型用？"
    print(f"\n📝 第2轮: {q2}")
    # 传递上轮消息
    messages = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1.answer[:300]},
    ]
    r2 = orch.chat(q2, session_id=session_id, messages=messages)
    print(f"   回答: {r2.answer[:150]}...")

    print(f"\n⏱️ 多轮对话完成（2轮）")


def _demo_simulated():
    """模拟演示（无API Key时）"""
    print("\n  📋 模拟演示模式")
    print("  " + "=" * 50)
    print("""
  🧠 [Planner] 分析用户意图 + 工具选择...
     → 意图: knowledge (知识问答)
     → Plan: 拆解为2个子任务:
       t1: 检索小户型扫地机器人选购要点 (工具: hybrid_search)
       t2: 检索静音型号推荐 (工具: hybrid_search, get_device_specs)

  🔧 [Tool] hybrid_search("小户型 静音 扫地机器人 推荐") ✓ (245ms)
  🔧 [Tool] get_device_specs("石头P10") ✓ (12ms)

  🔍 [Retriever] 第1轮检索: 5篇文档, KG=False

  🤔 [Reflector] 置信度自检...
     → 置信度: 0.88 ✓ pass

  📝 [Summarizer] 生成最终答案...
     → 完成 (响应时间: 1.2s, 工具调用 2次, 模型调用 3次)

  📄 回答: 对于小户型且需要静音的扫地机器人，推荐石头P10...
  """)


def main():
    print("""
╔══════════════════════════════════════════════════════╗
║   Multi-Agent 知识问答系统 — 四智能体协作演示          ║
║   Planner → Retriever → Reflector → Summarizer       ║
╚══════════════════════════════════════════════════════╝
""")

    # 检查环境
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("⚠️ 未检测到 DASHSCOPE_API_KEY 环境变量")
        print("  将使用模拟模式展示系统架构\n")

    try:
        demo_basic_qa()
    except Exception as e:
        print(f"\n⚠️ 演示1失败: {e}")

    try:
        demo_troubleshoot()
    except Exception as e:
        print(f"\n⚠️ 演示2失败: {e}")

    try:
        demo_multi_turn()
    except Exception as e:
        print(f"\n⚠️ 演示3失败: {e}")

    print(f"\n{'='*60}")
    print("  ✅ 演示完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
