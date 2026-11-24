import json
import time
import requests
from openai import OpenAI
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, FunctionCall
from agentdojo.types import (
    ChatAssistantMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    text_content_block_from_string as text,
)
from prompts_lr4 import (
    REACT_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    PLAN_STEP_SYSTEM_PROMPT,
    PLAN_FINAL_SYSTEM_PROMPT,
)

SERVER_ROOT = "http://127.0.0.1:8000"
BASE_URL = f"{SERVER_ROOT}/v1"
MODEL_NAME = "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# 严格对齐第 2.3 节资源限制 / Лимиты на эпизод (п. 2.3)
MAX_MODEL_CALLS = 15
MAX_TOOL_CALLS = 15
MAX_OUTPUT_TOKENS = 1024
MAX_PLAN_STEPS = 8


def to_openai_tool(f) -> dict:
    """将 AgentDojo 工具函数转换为 OpenAI tools JSON Schema / Перевод функции стенда в формат tools."""
    return {
        "type": "function",
        "function": {
            "name": f.name,
            "description": f.description,
            "parameters": f.parameters.model_json_schema(),
        },
    }


def make_plan_schema(tool_names: list) -> dict:
    """附录要求：使用 enum 约束计划中的工具名，防止模型编造不存在的工具 / Схема плана с enum имён инструментов."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "execution_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["steps"],
                "properties": {
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_PLAN_STEPS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["tool", "purpose"],
                            "properties": {
                                "tool": {"type": "string", "enum": tool_names},
                                "purpose": {"type": "string"},
                            },
                        },
                    }
                },
            },
        },
    }


class CustomHarnessAgent(BasePipelineElement):
    """
    自建工具调用智能体框架（实现 ReAct 与 Plan-then-execute 双模式、完整轨迹记录与硬性上限控制）
    Собственная обвязка агента (режимы ReAct и plan-then-execute, лимиты и подробный трейс).
    """

    def __init__(self, config: dict):
        self.config = config
        self.mode = config["mode"]  # "react" 或 "plan_then_execute"
        self.name = f"local-{config.get('model', MODEL_NAME)}-{self.mode}"
        self.client = OpenAI(base_url=BASE_URL, api_key="none")
        self.session = requests.Session()
        self.context_limit = config.get("context_limit", 16384)
        self.last_trace = None  # 保存最近一次 Episode 的完整执行轨迹 / Трейс последнего эпизода

    def _estimate_prompt_tokens(self, messages: list, tools: list = None) -> int:
        """调用 /tokenize 估算当前输入长度，防止静默截断 / Проверка длины входа перед вызовом модели."""
        payload_text = json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False)
        try:
            resp = self.session.post(f"{SERVER_ROOT}/tokenize", json={"content": payload_text}, timeout=15)
            if resp.status_code == 200:
                return len(resp.json().get("tokens", []))
        except Exception:
            pass
        return len(payload_text) // 3

    def _call_llm(self, messages: list, tools: list = None, tool_choice=None, response_format=None):
        """封装一次大模型 API 调用并记录耗时与 token 消耗 / Один вызов LLM с замером времени и токенов."""
        t0 = time.perf_counter()
        kwargs = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format

        resp = self.client.chat.completions.create(**kwargs)
        dt = time.perf_counter() - t0
        msg = resp.choices[0].message
        p_toks = resp.usage.prompt_tokens if resp.usage else 0
        c_toks = resp.usage.completion_tokens if resp.usage else 0
        return msg, p_toks, c_toks, round(dt, 4)

    def _execute_tool(self, runtime, env, name: str, raw_args):
        """安全解析参数并调用 runtime.run_function；错误以文本形式返回而不中断 Episode / Исполнитель инструментов."""
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except Exception as e:
                return {}, f"JSONDecodeError in tool arguments: {e}", f"JSONDecodeError in tool arguments: {e}"
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        if name not in runtime.functions:
            err = f"ToolError: unknown tool '{name}'"
            return args, err, err

        result, error = runtime.run_function(env, name, args)
        output = tool_result_to_str(result) if error is None else str(error)
        return args, output, (str(error) if error is not None else None)

    def query(self, query: str, runtime, env=EmptyEnv(), messages=(), extra_args=None):
        extra_args = extra_args or {}
        t_ep_start = time.perf_counter()
        tools_map = {name: to_openai_tool(fn) for name, fn in runtime.functions.items()}
        all_tools = list(tools_map.values())

        trace = {
            "mode": self.mode,
            "query": query,
            "stop_reason": "answered",
            "model_calls": 0,
            "tool_calls": 0,
            "tool_errors": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "steps": [],
            "plan": None,
            "final_answer": "",
        }

        if self.mode == "react":
            log, answer = self._run_react(query, runtime, env, all_tools, trace)
        elif self.mode == "plan_then_execute":
            log, answer = self._run_plan_then_execute(query, runtime, env, tools_map, all_tools, trace)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        trace["final_answer"] = answer
        trace["episode_time_s"] = round(time.perf_counter() - t_ep_start, 4)
        self.last_trace = trace
        return query, runtime, env, log, extra_args

    def _run_react(self, query: str, runtime, env, all_tools: list, trace: dict):
        chat = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        log = [
            ChatSystemMessage(role="system", content=[text(REACT_SYSTEM_PROMPT)]),
            ChatUserMessage(role="user", content=[text(query)]),
        ]
        final_answer = None

        while True:
            # 优先检查工具调用上限，防止与模型调用上限同时达到时被 max_calls 抢先触发
            if trace["tool_calls"] >= MAX_TOOL_CALLS:
                trace["stop_reason"] = "max_tools"
                final_answer = "Stopped: reached maximum tool calls limit (15)."
                break
            if trace["model_calls"] >= MAX_MODEL_CALLS:
                trace["stop_reason"] = "max_calls"
                final_answer = "Stopped: reached maximum model calls limit (15)."
                break

            est_toks = self._estimate_prompt_tokens(chat, all_tools)
            if est_toks >= self.context_limit - MAX_OUTPUT_TOKENS:
                trace["stop_reason"] = "context_limit"
                final_answer = "Stopped: input context limit reached."
                break

            msg, p_toks, c_toks, dt = self._call_llm(chat, tools=all_tools)
            trace["model_calls"] += 1
            trace["total_prompt_tokens"] += p_toks
            trace["total_completion_tokens"] += c_toks

            step_rec = {
                "step": trace["model_calls"],
                "phase": "react",
                "num_messages": len(chat),
                "prompt_tokens": p_toks,
                "completion_tokens": c_toks,
                "time_s": dt,
                "assistant_text": msg.content,
                "tool_executions": [],
            }

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                final_answer = (msg.content or "").strip() or "Task completed."
                trace["steps"].append(step_rec)
                break

            # 记录助手发出的 tool_calls / Сохраняем вызовы инструментов ассистента
            assistant_chat_msg = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
            chat.append(assistant_chat_msg)

            dojo_calls = []
            tool_res_msgs = []
            for tc in tool_calls:
                if trace["tool_calls"] >= MAX_TOOL_CALLS:
                    trace["stop_reason"] = "max_tools"
                    break
                trace["tool_calls"] += 1
                fn_name = tc.function.name
                args, output, err = self._execute_tool(runtime, env, fn_name, tc.function.arguments)
                if err is not None:
                    trace["tool_errors"] += 1

                fc = FunctionCall(function=fn_name, args=args, id=tc.id)
                dojo_calls.append(fc)
                chat.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                tool_res_msgs.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text(output)],
                        tool_call_id=tc.id,
                        tool_call=fc,
                        error=err,
                    )
                )
                step_rec["tool_executions"].append(
                    {
                        "tool": fn_name,
                        "args": args,
                        "output_preview": output[:500],
                        "output_len": len(output),
                        "error": err,
                    }
                )

            log.append(ChatAssistantMessage(role="assistant", content=None, tool_calls=dojo_calls))
            log.extend(tool_res_msgs)
            trace["steps"].append(step_rec)

            if trace["stop_reason"] == "max_tools":
                final_answer = "Stopped: reached maximum tool calls limit (15)."
                break

        # 第 2.3.6 节：Episode 必须以文本回复结束 / Эпизод всегда завершается текстовым ответом
        log.append(ChatAssistantMessage(role="assistant", content=[text(final_answer)], tool_calls=None))
        return log, final_answer

    def _run_plan_then_execute(self, query: str, runtime, env, tools_map: dict, all_tools: list, trace: dict):
        log = [
            ChatSystemMessage(role="system", content=[text(PLAN_SYSTEM_PROMPT)]),
            ChatUserMessage(role="user", content=[text(query)]),
        ]

        # 阶段 1：规划（仅传入工具描述，不执行任何工具，输出 1..8 步 JSON 计划） / Этап 1: План
        tool_desc_summary = "\n".join(
            [f"- {name}: {spec['function']['description']}" for name, spec in tools_map.items()]
        )
        plan_messages = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": f"User request: {query}\n\nAvailable tools:\n{tool_desc_summary}"},
        ]
        plan_schema = make_plan_schema(list(tools_map.keys()))
        msg_plan, p_toks, c_toks, dt = self._call_llm(plan_messages, response_format=plan_schema)
        trace["model_calls"] += 1
        trace["total_prompt_tokens"] += p_toks
        trace["total_completion_tokens"] += c_toks

        try:
            plan_obj = json.loads(msg_plan.content or "{}")
            steps = plan_obj.get("steps", [])
            if not (1 <= len(steps) <= MAX_PLAN_STEPS) or any(s.get("tool") not in tools_map for s in steps):
                raise ValueError("Invalid plan structure or unknown tool")
        except Exception as e:
            trace["stop_reason"] = "plan_invalid"
            trace["steps"].append(
                {
                    "step": 1,
                    "phase": "plan",
                    "prompt_tokens": p_toks,
                    "completion_tokens": c_toks,
                    "time_s": dt,
                    "error": str(e),
                }
            )
            ans = "Не удалось составить план (plan_invalid)."
            log.append(ChatAssistantMessage(role="assistant", content=[text(ans)], tool_calls=None))
            return log, ans

        trace["plan"] = steps
        trace["steps"].append(
            {
                "step": 1,
                "phase": "plan",
                "prompt_tokens": p_toks,
                "completion_tokens": c_toks,
                "time_s": dt,
                "plan": steps,
            }
        )

        # 阶段 2：按计划逐步执行（每步仅暴露当前计划指定的 1 个工具） / Этап 2: Выполнение по шагам плана
        exec_history_text = []
        for idx, step_spec in enumerate(steps, start=1):
            if trace["tool_calls"] >= MAX_TOOL_CALLS:
                trace["stop_reason"] = "max_tools"
                break
            if trace["model_calls"] >= MAX_MODEL_CALLS:
                trace["stop_reason"] = "max_calls"
                break

            target_tool = step_spec["tool"]
            purpose = step_spec["purpose"]
            step_sys = PLAN_STEP_SYSTEM_PROMPT.format(
                step_idx=idx, total_steps=len(steps), purpose=purpose, tool_name=target_tool
            )
            obs_block = "\n\n".join(exec_history_text) if exec_history_text else "(No previous tool results yet)"
            step_msgs = [
                {"role": "system", "content": step_sys},
                {"role": "user", "content": f"User request: {query}\n\nPrevious step results:\n{obs_block}"},
            ]
            single_tool = [tools_map[target_tool]]

            est_toks = self._estimate_prompt_tokens(step_msgs, single_tool)
            if est_toks >= self.context_limit - MAX_OUTPUT_TOKENS:
                trace["stop_reason"] = "context_limit"
                break

            msg_step, p_toks, c_toks, dt = self._call_llm(
                step_msgs,
                tools=single_tool,
                tool_choice={"type": "function", "function": {"name": target_tool}},
            )
            trace["model_calls"] += 1
            trace["total_prompt_tokens"] += p_toks
            trace["total_completion_tokens"] += c_toks

            step_rec = {
                "step": trace["model_calls"],
                "phase": f"exec_step_{idx}",
                "planned_tool": target_tool,
                "prompt_tokens": p_toks,
                "completion_tokens": c_toks,
                "time_s": dt,
                "tool_executions": [],
            }

            tcs = msg_step.tool_calls or []
            if not tcs or tcs[0].function.name != target_tool:
                # 模型未调用指定工具：该步记为失败且不修改环境，继续下一步 / Шаг считается неудачным
                err_msg = f"Step {idx} skipped: model did not call planned tool '{target_tool}'."
                exec_history_text.append(f"Step {idx} ({target_tool}): {err_msg}")
                step_rec["tool_executions"].append({"tool": target_tool, "error": err_msg})
                trace["steps"].append(step_rec)
                continue

            tc = tcs[0]
            trace["tool_calls"] += 1
            args, output, err = self._execute_tool(runtime, env, target_tool, tc.function.arguments)
            if err is not None:
                trace["tool_errors"] += 1

            fc = FunctionCall(function=target_tool, args=args, id=tc.id)
            log.append(ChatAssistantMessage(role="assistant", content=None, tool_calls=[fc]))
            log.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text(output)],
                    tool_call_id=tc.id,
                    tool_call=fc,
                    error=err,
                )
            )
            exec_history_text.append(f"Step {idx} ({target_tool}, args={json.dumps(args)}):\n{output}")
            step_rec["tool_executions"].append(
                {
                    "tool": target_tool,
                    "args": args,
                    "output_preview": output[:500],
                    "output_len": len(output),
                    "error": err,
                }
            )
            trace["steps"].append(step_rec)

        # 阶段 3：最终汇总回答（不传任何工具） / Этап 3: Финальный текстовый ответ без инструментов
        if trace["stop_reason"] == "answered" and trace["model_calls"] < MAX_MODEL_CALLS:
            obs_block = "\n\n".join(exec_history_text) if exec_history_text else "(No results)"
            final_msgs = [
                {"role": "system", "content": PLAN_FINAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"User request: {query}\n\nExecuted steps results:\n{obs_block}"},
            ]
            est_toks = self._estimate_prompt_tokens(final_msgs)
            if est_toks < self.context_limit - MAX_OUTPUT_TOKENS:
                msg_fin, p_toks, c_toks, dt = self._call_llm(final_msgs)
                trace["model_calls"] += 1
                trace["total_prompt_tokens"] += p_toks
                trace["total_completion_tokens"] += c_toks
                final_answer = (msg_fin.content or "").strip() or "Task completed."
                trace["steps"].append(
                    {
                        "step": trace["model_calls"],
                        "phase": "final_answer",
                        "prompt_tokens": p_toks,
                        "completion_tokens": c_toks,
                        "time_s": dt,
                    }
                )
            else:
                trace["stop_reason"] = "context_limit"
                final_answer = "Stopped: context limit reached before final summary."
        else:
            final_answer = f"Stopped ({trace['stop_reason']}): partial execution finished."

        log.append(ChatAssistantMessage(role="assistant", content=[text(final_answer)], tool_calls=None))
        return log, final_answer