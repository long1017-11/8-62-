import hashlib
import random
from agentdojo.functions_runtime import FunctionsRuntime

# 根据 AgentDojo workspace 工具描述确定的“修改环境状态（写操作）”工具列表（п. 2.3）
# Список инструментов набора workspace, меняющих состояние среды (п. 2.3)
WORKSPACE_WRITE_TOOLS = [
    # Почта (Email)
    "send_email",
    "delete_email",
    "mark_email_as_read",
    "mark_email_as_unread",
    # Календарь (Calendar)
    "create_calendar_event",
    "cancel_calendar_event",
    "reschedule_calendar_event",
    "add_calendar_event_participants",
    # Контакты (Contacts)
    "add_contact",
    # Файлы (Cloud Drive)
    "create_file",
    "delete_file",
    "append_to_file",
    "share_file",
]


class FaultyRuntime(FunctionsRuntime):
    """
    工具故障注入包装器（严格实现指导书第 2.2 节与附录逻辑）：
    - 类型 A（执行前失败）：所有工具均可能触发，环境未改变，返回 TemporaryError，安全可重试。
    - 类型 B（执行后丢失响应）：仅写操作工具（write_tools）触发，先实际修改环境，再返回 TimeoutError，盲目重试会造成重复执行！
    Обёртка внедрения сбоев инструментов (п. 2.2 и Приложение).
    """

    def __init__(self, functions, task_id: str, fault_p: float, write_tools: list):
        super().__init__(functions)
        self.task_id = task_id
        self.fault_p = fault_p
        self.write_tools = set(write_tools)
        self.n_calls = 0
        self.log = []  # 记录每次底层调用：(调用序号, 工具名, 故障类型A/B/None, 是否实际修改/成功执行)

    def _rng(self):
        # 确定性种子：仅依赖 (task_id, n_calls)，保证在轨迹分叉前各策略在完全相同的调用位置遇到相同的故障
        seed = int(hashlib.sha256(f"{self.task_id}:{self.n_calls}".encode()).hexdigest()[:8], 16)
        return random.Random(seed)

    def run_function(self, env, function: str, kwargs: dict, raise_on_error: bool = False):
        self.n_calls += 1
        if self.fault_p > 0.0 and self._rng().random() < self.fault_p:
            if function in self.write_tools:
                # 类型 B：环境已被修改，但响应丢失 / Тип B: действие выполнено, ответ потерян
                super().run_function(env, function, kwargs, raise_on_error=raise_on_error)
                self.log.append((self.n_calls, function, "B", True))
                return "", "TimeoutError: no response from server"
            else:
                # 类型 A：未执行，环境未变 / Тип A: ошибка до выполнения, среда не изменилась
                self.log.append((self.n_calls, function, "A", False))
                return "", "TemporaryError: service unavailable, try again later"

        result, error = super().run_function(env, function, kwargs, raise_on_error=raise_on_error)
        self.log.append((self.n_calls, function, None, error is None))
        return result, error