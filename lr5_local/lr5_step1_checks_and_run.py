import os
import json
from functools import partial
from unittest.mock import MagicMock
from tqdm import tqdm
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, FunctionCall
from agentdojo.types import (
    ChatAssistantMessage,
    ChatToolResultMessage,
    text_content_block_from_string as text,
)
from fault_runtime_lr5 import FaultyRuntime, WORKSPACE_WRITE_TOOLS
from agent_lr5 import ReActEvalAgentLR5

os.makedirs("lr5_artifacts/traces", exist_ok=True)
RESULTS_FILE = "lr5_artifacts/results.jsonl"

suites = get_suites("v1.2.2")
ws_suite = suites["workspace"]
bk_suite = suites["banking"]

all_ws_ids = [f"user_task_{i}" for i in range(40)]
debug_ws_ids = [f"user_task_{i}" for i in range(0, 40, 4)]
eval_ws_ids = [tid for tid in all_ws_ids if tid not in debug_ws_ids]  # 30 个 workspace 正式任务
eval_bk_ids = [f"user_task_{i}" for i in range(16)]  # 16 个 banking 迁移任务

print("--- Проверка 3.3 без модели / 运行前 3 项故障策略单元测试（不调用大模型） ---")

# 测试 1：fault_p=1.0 时类型 A 故障（只读工具），策略 P2 在第 2 次尝试（第 1 次自动重试）成功获取结果
env_t1 = ws_suite.load_and_inject_default_environment({})
rt_t1 = FaultyRuntime(ws_suite.tools, task_id="test_a", fault_p=1.0, write_tools=WORKSPACE_WRITE_TOOLS)
# 模拟第 1 次返回 A 故障，第 2 次恢复正常
orig_run = super(FaultyRuntime, rt_t1).run_function
call_cnt = {"n": 0}
def mock_run_a(env, function, kwargs, raise_on_error=False):
    call_cnt["n"] += 1
    if call_cnt["n"] == 1:
        rt_t1.log.append((1, function, "A", False))
        return "", "TemporaryError: service unavailable, try again later"
    rt_t1.log.append((2, function, None, True))
    return orig_run(env, function, kwargs, raise_on_error=raise_on_error)
rt_t1.run_function = mock_run_a

ag_p2 = ReActEvalAgentLR5({"retry_policy": "P2"})
tr_dummy = {"tool_calls": 0, "tool_errors": 0, "extra_retry_calls": 0, "stop_reason": "answered"}
_, out_a, err_a, retries_a = ag_p2._execute_with_retry_policy(rt_t1, env_t1, "get_current_day", {}, tr_dummy)
assert err_a is None and out_a == "2024-05-15" and retries_a == 1 and len(rt_t1.log) == 2

# 测试 2：类型 B 故障发生在 send_email 时，P1 盲目重试发送 2 封邮件，而 P2 仅发送 1 封邮件
for pol, expected_sent in [("P1", 3), ("P2", 1)]:
    env_b = ws_suite.load_and_inject_default_environment({})
    before_cnt = len(env_b.inbox.sent)
    rt_b = FaultyRuntime(ws_suite.tools, task_id="test_b", fault_p=1.0, write_tools=WORKSPACE_WRITE_TOOLS)
    ag_b = ReActEvalAgentLR5({"retry_policy": pol})
    tr_b = {"tool_calls": 0, "tool_errors": 0, "extra_retry_calls": 0, "stop_reason": "answered"}
    ag_b._execute_with_retry_policy(
        rt_b, env_b, "send_email", {"recipients": ["test@example.com"], "subject": "Hi", "body": "Test"}, tr_b
    )
    after_cnt = len(env_b.inbox.sent)
    assert (after_cnt - before_cnt) == expected_sent, f"Policy {pol} sent {after_cnt - before_cnt}, expected {expected_sent}"

# 测试 3：fault_p=0.0 时 FaultyRuntime 不改变行为，ground_truth 通过 36/40 题
class OracleGT(BasePipelineElement):
    def __init__(self, task):
        self.name = "oracle"
        self.task = task
    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args=None):
        extra_args = extra_args or {}
        gt = self.task.ground_truth(env)
        log, outs = [], []
        for i, fc in enumerate(gt):
            call = FunctionCall(function=fc.function, args=fc.args, id=f"gt_{i}")
            r, e = runtime.run_function(env, fc.function, fc.args)
            s = tool_result_to_str(r) if e is None else str(e)
            outs.append(s)
            log.append(ChatAssistantMessage(role="assistant", content=None, tool_calls=[call]))
            log.append(ChatToolResultMessage(role="tool", content=[text(s)], tool_call_id=call.id, tool_call=call, error=e))
        log.append(ChatAssistantMessage(role="assistant", content=[text("\n".join(outs) or "Done.")], tool_calls=None))
        return query, runtime, env, log, extra_args

gt_ok = 0
for tid in all_ws_ids:
    tobj = ws_suite.get_user_task_by_id(tid)
    rt_cls = partial(FaultyRuntime, task_id=tid, fault_p=0.0, write_tools=WORKSPACE_WRITE_TOOLS)
    u, _ = ws_suite.run_task_with_pipeline(OracleGT(tobj), tobj, None, {}, runtime_class=rt_cls)
    if u:
        gt_ok += 1
assert gt_ok >= 36
print(f"✅ Все 3 проверки 3.3 пройдены! (ground_truth с fault_p=0: {gt_ok}/40)\n")

# 记录运行前预测（第 4 节要求） / Прогноз до запусков (п. 4)
forecast_lr5 = """ПРОГНОЗ ДО ЗАПУСКОВ (ЛР5) / 实验五运行前预测:
1. Стабильность (Эксперимент 1): при temperature=0.0 расхождения между двумя прогонами будут минимальными (0–2 задачи из 30 из-за неассоциативности вычислений на GPU), тогда как при temperature=0.7 нестабильными (0 < c < 5) окажутся около 35–50% задач. Метрика pass^k будет монотонно падать от k=1 (~53%) к k=5 (~25–30%), особенно на многошаговых задачах, где ошибка на любом шаге обнуляет весь эпизод.
   （稳定性预测：T=0 时两次运行差异极小，约 0–2 题；T=0.7 时约 35–50% 的任务表现为不稳定，pass^k 将从 k=1 的约 53% 显著下降至 k=5 的约 25–30%，多步任务下降尤为明显。）
2. Сбои инструментов (Эксперимент 2, fault_p=0.2): политика P2 покажет наилучший баланс — она безопасно восстановится после сбоев типа A на чтении (опередив P0), но не допустит дублирования писем и событий при сбоях типа B, в отличие от слепых повторов P1, которые приведут к повторному выполнению действий.
   （故障预测：P2 策略综合表现最佳——既能安全重试只读工具的 A 类瞬时故障从而优于 P0，又能避免 B 类写操作响应丢失时 P1 盲目重试导致的重复发信/重复建日程副作用。）
3. Перенос на banking (Эксперимент 3): без изменения промптов успешность на 16 незнакомых задачах banking просядет на 10–20 п.п. по сравнению с workspace (до ~35–45%) из-за новых инструментов платежей и транзакций, однако 95% доверительные интервалы Уилсона будут заметно перекрываться из-за малого размера выборок (n=16 и n=30).
   （跨域迁移预测：不改任何提示词直接迁移到 banking 16 个新任务时，成功率预计下降 10–20 个百分点至约 35–45%，但由于样本量较小，两者的 95% 威尔逊置信区间将存在明显重叠。）
"""
with open("lr5_artifacts/pre_run_forecast_lr5.txt", "w", encoding="utf-8") as f:
    f.write(forecast_lr5)

# 加载已完成主键，实现严格断点续跑（第 3.1 节） / Загрузка уже выполненных ключей из results.jsonl
completed_keys = set()
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                completed_keys.add(rec["episode_key"])
print(f"Уже выполнено в results.jsonl: {len(completed_keys)} эпизодов.")

# 构建全部 316 个实验任务队列
schedule = []

# Эксперимент 1.1: 30 задач workspace × 2 прогона при T=0.0 (60 эпизодов)
for run_i in range(2):
    for tid in eval_ws_ids:
        schedule.append(("workspace", tid, "exp1_t0", run_i, 1000 + run_i, 0.0, 1.0, "P0", 0.0))

# Эксперимент 1.2: 30 задач workspace × 5 прогонов с сэмплированием T=0.7, top_p=0.8 (150 эпизодов)
for run_i in range(5):
    for tid in eval_ws_ids:
        schedule.append(("workspace", tid, "exp1_sample", run_i, 2000 + run_i, 0.7, 0.8, "P0", 0.0))

# Эксперимент 2: 30 задач workspace × 3 политики P0, P1, P2 при T=0.0, fault_p=0.2 (90 эпизодов)
for pol in ("P0", "P1", "P2"):
    for tid in eval_ws_ids:
        schedule.append(("workspace", tid, f"exp2_fault_{pol}", 0, 3000, 0.0, 1.0, pol, 0.2))

# Эксперимент 3: 16 задач banking × 1 прогон при T=0.0 (16 эпизодов)
for tid in eval_bk_ids:
    schedule.append(("banking", tid, "exp3_banking", 0, 4000, 0.0, 1.0, "P0", 0.0))

pending = [
    item for item in schedule
    if f"{item[0]}|{item[1]}|{item[2]}|{item[3]}|{item[4]}" not in completed_keys
]
print(f"Всего запланировано: {len(schedule)}, осталось выполнить: {len(pending)} эпизодов.\n")

for suite_name, tid, cfg_name, run_i, seed, temp, top_p, pol, fault_p in tqdm(pending):
    ep_key = f"{suite_name}|{tid}|{cfg_name}|{run_i}|{seed}"
    curr_suite = ws_suite if suite_name == "workspace" else bk_suite
    t_obj = curr_suite.get_user_task_by_id(tid)

    agent = ReActEvalAgentLR5({
        "retry_policy": pol,
        "temperature": temp,
        "top_p": top_p,
        "seed": seed,
        "write_tools": WORKSPACE_WRITE_TOOLS if suite_name == "workspace" else [],
    })

    if fault_p > 0.0:
        rt_cls = partial(FaultyRuntime, task_id=tid, fault_p=fault_p, write_tools=WORKSPACE_WRITE_TOOLS)
        util, _ = curr_suite.run_task_with_pipeline(agent, t_obj, None, {}, runtime_class=rt_cls)
    else:
        util, _ = curr_suite.run_task_with_pipeline(agent, t_obj, None, {})

    tr = agent.last_trace
    # 统计实际执行的写操作及重复执行的完全相同写操作（第 3.1 与 4.2 节要求）
    write_calls_seq = []
    for st in tr["steps"]:
        for ex in st["tool_executions"]:
            if ex["tool"] in WORKSPACE_WRITE_TOOLS and ex["error"] is None:
                write_calls_seq.append((ex["tool"], json.dumps(ex["args"], sort_keys=True)))
    # 结合 FaultyRuntime 日志统计 B 类故障造成的底层重复执行
    rt_log = tr.get("fault_runtime_log", [])
    injected_a = sum(1 for x in rt_log if x[2] == "A")
    injected_b = sum(1 for x in rt_log if x[2] == "B")

    # 统计底层实际成功执行的写操作次数（含 B 类成功）
    executed_writes = [
        x[1] for x in rt_log if x[1] in WORKSPACE_WRITE_TOOLS and x[3] is True
    ] if rt_log else [w[0] for w in write_calls_seq]

    # 统计同一任务内完全相同的写调用被执行 >= 2 次的次数
    seen_w = set()
    dup_actions = 0
    for st in tr["steps"]:
        for ex in st["tool_executions"]:
            if ex["tool"] in WORKSPACE_WRITE_TOOLS:
                sig = (ex["tool"], json.dumps(ex["args"], sort_keys=True))
                if sig in seen_w:
                    dup_actions += 1
                seen_w.add(sig)

    trace_path = f"lr5_artifacts/traces/{suite_name}_{tid}_{cfg_name}_r{run_i}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({"episode_key": ep_key, "utility": bool(util), "trace": tr}, f, indent=2, ensure_ascii=False)

    rec = {
        "episode_key": ep_key,
        "suite": suite_name,
        "task_id": tid,
        "config": cfg_name,
        "run_idx": run_i,
        "seed": seed,
        "utility": bool(util),
        "stop_reason": tr["stop_reason"],
        "model_calls": tr["model_calls"],
        "tool_calls": tr["tool_calls"],
        "extra_retry_calls": tr["extra_retry_calls"],
        "injected_faults_total": injected_a + injected_b,
        "injected_faults_A": injected_a,
        "injected_faults_B": injected_b,
        "executed_write_calls": executed_writes,
        "duplicate_write_actions": dup_actions,
        "prompt_tokens": tr["total_prompt_tokens"],
        "completion_tokens": tr["total_completion_tokens"],
        "total_tokens": tr["total_prompt_tokens"] + tr["total_completion_tokens"],
        "time_s": tr["episode_time_s"],
        "trace_path": trace_path,
    }
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("\n✅ Все запланированные эпизоды ЛР5 завершены и сохранены в lr5_artifacts/results.jsonl!")