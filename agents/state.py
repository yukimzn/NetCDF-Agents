# state.py
from typing import Annotated, Sequence, List, Dict, Any, Optional,TypedDict
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """多智能体系统的全局状态"""
    
    # ---------- 会话与标识 ----------
    session_id: str                         # 会话唯一标识，支持并行多对话
    
    # ---------- 消息与交互 ----------
    messages: Annotated[Sequence[BaseMessage], operator.add]   # 对话历史，自动累加
    
    # ---------- 文件与元数据 ----------
    file_paths: List[str]                   # 当前会话关联的 netCDF 文件列表（支持多文件）
    metadata: Dict[str, Any]                # 解析后的元数据，键为文件路径，值为 analyzer 返回的字典
    current_cache_key: Optional[str]        # 工具链当中存储临时文件的路径
    
    # ---------- 工具链与执行 ----------
    current_tool_chain: Dict[str, Any]      # 当前待执行的工具链（JSON 格式）
    execution_result: Any                   # 工具链或代码执行的最终结果
    
    # ---------- 代码生成与纠错 ----------
    current_code: str                       # 当前生成的代码（或出错的代码）
    error_info: str                         # 最后一次的错误信息（原始字符串）
    error_history: List[Dict[str, Any]]     # 结构化的历史错误列表，每条包含 error_type, line, cause, fix
    iteration_count: int                    # 当前纠错循环的迭代次数（最大 5）
    
    # ---------- 路由与输出 ----------
    next_step: str                          # 路由控制字段（parse / process / fix / end）
    final_answer: str                       # 最终返回给用户的答案或报告
    selected_dataset_keys: List[str]   # 经 LLM 筛选后，实际需要处理的数据集键列表
    original_request:str
    
    