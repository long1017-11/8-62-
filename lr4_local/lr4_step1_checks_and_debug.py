import os
import json
import hashlib
from unittest.mock import MagicMock
import agentdojo
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatToolResultMessage,
    text_content_block_from_string as text,
)
from agent_harness import CustomHarnessAgent

os.makedirs("lr4_artifacts/traces_debug", exist_ok=True)
os.makedirs("lr4_artifacts/traces_main", exist_ok=True)

suite = get_suites("v1.2.2")["workspace"]
all_task_ids = [f"user_task_{i}" for i in range(40)]
debug_ids = [f"user_task_{i}" for i in range(0, 40, 4)]  # 10 个编号为 4 的倍数的调试任务
eval_ids = [tid for tid in all_task_ids if tid not in debug_ids]  # 30 个最终评测任务

# 1. 保存实验数据清单（第 3.1 节要求） / Манифест данных (п. 3.1)
manifest = {
    "agentdojo_version": getattr(agentdojo, "__version__", "0.1.35"),
    "benchmark_version": "v1.2.2",
    "suite": "workspace",
    "total_tasks": 40,
    "debug_ids": debug_ids,
    "eval_ids": eval_ids,
    "eval_ids_sha256": hashlib.sha256(json.dumps(eval_ids).encode()).hexdigest(),
}
with open("lr4_artifacts/manifest_lr4.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("--- Проверка 3.4 без модели / 运行前 4 项框架单元测试（不调用大模型） ---")

# 单元测试 1：不存在的工具与缺少必填参数返回错误文本而非崩溃 / Тест 1: обработка ошибок аргументов
dummy_agent = CustomHarnessAgent({"mode": "react"})
env0 = suite.load_and_inject_default_environment({})
rt0 = FunctionsRuntime(suite.tools)
_, out_unk, err_unk = dummy_agent._execute_tool(rt0, env0, "nonexistent_tool", {})
assert err_unk is not None and "unknown tool" in out_unk
_, out_val, err_val = dummy_agent._execute_tool(rt0, env0, "search_emails", {})
assert err_val is not None and len(out_val) > 0

# 单元测试 2：死循环脚本智能体在第 15 次工具调用时触发 max_tools 并以文本结束 / Тест 2: срабатывание лимита 15 вызовов
loop_agent = CustomHarnessAgent({"mode": "react"})
fake_tc = MagicMock()
fake_tc.id = "call_loop"
fake_tc.function.name = "get_unread_emails"
fake_tc.function.arguments = "{}"
fake_msg = MagicMock()
fake_msg.content = None
fake_msg.tool_calls = [fake_tc]
loop_agent._call_llm = lambda *a, **kw: (fake_msg, 100, 10, 0.01)
_, _, _, log_loop, _ = loop_agent.query("loop test", rt0, env0)
assert loop_agent.last_trace["tool_calls"] == 15
assert loop_agent.last_trace["stop_reason"] == "max_tools"
assert log_loop[-1]["role"] == "assistant" and log_loop[-1]["tool_calls"] is None

# 单元测试 3：Ground Truth 脚本智能体在 40 个任务上至少通过 36 个 / Тест 3: сценарный агент ground_truth (>=36/40)
class OracleGroundTruthAgent(BasePipelineElement):
    def __init__(self, task):
        self.name = "oracle-gt"
        self.task = task

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args=None):
        extra_args = extra_args or {}
        gt_calls = self.task.ground_truth(env)
        log = []
        outputs = []
        for idx, fc in enumerate(gt_calls):
            call = FunctionCall(function=fc.function, args=fc.args, id=f"gt_{idx}")
            res, err = runtime.run_function(env, fc.function, fc.args)
            out_str = tool_result_to_str(res) if err is None else str(err)
            outputs.append(out_str)
            log.append(ChatAssistantMessage(role="assistant", content=None, tool_calls=[call]))
            log.append(
                ChatToolResultMessage(
                    role="tool", content=[text(out_str)], tool_call_id=call.id, tool_call=call, error=err
                )
            )
        final_txt = "\n".join(outputs) if outputs else "Done."
        log.append(ChatAssistantMessage(role="assistant", content=[text(final_txt)], tool_calls=None))
        return query, runtime, env, log, extra_args

gt_passed = 0
for tid in all_task_ids:
    t_obj = suite.get_user_task_by_id(tid)
    u, _ = suite.run_task_with_pipeline(OracleGroundTruthAgent(t_obj), t_obj, None, {})
    if u:
        gt_passed += 1
print(f"Проверка ground_truth: пройдено {gt_passed} из 40 задач (требуется >= 36)")
assert gt_passed >= 36

# 单元测试 4：Plan-then-execute 拒绝非当前步骤计划的工具且不修改环境 / Тест 4: отклонение чужого инструмента в плане
plan_guard_agent = CustomHarnessAgent({"mode": "plan_then_execute"})
p_msg = MagicMock()
p_msg.content = json.dumps({"steps": [{"tool": "get_unread_emails", "purpose": "read"}]})
wrong_tc = MagicMock()
wrong_tc.id = "bad_call"
wrong_tc.function.name = "send_email"
wrong_tc.function.arguments = '{"recipients":["a@b.com"],"subject":"x","body":"y"}'
s_msg = MagicMock()
s_msg.content = None
s_msg.tool_calls = [wrong_tc]
f_msg = MagicMock()
f_msg.content = "Done."
f_msg.tool_calls = None
call_seq = iter([(p_msg, 50, 20, 0.01), (s_msg, 50, 10, 0.01), (f_msg, 50, 10, 0.01)])
plan_guard_agent._call_llm = lambda *a, **kw: next(call_seq)
_, _, _, _, _ = plan_guard_agent.query("test plan guard", rt0, env0)
assert plan_guard_agent.last_trace["tool_calls"] == 0

print("✅ Все 4 проверки обвязки 3.4 успешно пройдены! / 全部 4 项框架单元测试均已通过！\n")

# 2. 在 10 个调试任务（0, 4, ..., 36）上试运行双模式 / Отладочный прогон на 10 задачах (п. 4.1)
print("=== Отладочный запуск на 10 задачах (user_task_0, 4, ..., 36) ===")
debug_summary = []
for tid in debug_ids:
    t_obj = suite.get_user_task_by_id(tid)
    row = {"task_id": tid}
    for mode in ("react", "plan_then_execute"):
        ag = CustomHarnessAgent({"mode": mode})
        util, _ = suite.run_task_with_pipeline(ag, t_obj, None, {})
        row[mode] = bool(util)
        with open(f"lr4_artifacts/traces_debug/{tid}_{mode}.json", "w", encoding="utf-8") as f:
            json.dump({"task_id": tid, "utility": bool(util), "trace": ag.last_trace}, f, indent=2, ensure_ascii=False)
    debug_summary.append(row)
    print(f"{tid}: ReAct={row['react']} | Plan-then-execute={row['plan_then_execute']}")

with open("lr4_artifacts/debug_summary.json", "w", encoding="utf-8") as f:
    json.dump(debug_summary, f, indent=2)
print(
    f"\nИтог отладки (10 задач): ReAct = {sum(r['react'] for r in debug_summary)}/10, "
    f"Plan = {sum(r['plan_then_execute'] for r in debug_summary)}/10"
)