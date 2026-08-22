import re
import json
import uuid
import os
from typing import Literal, Dict, Any, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from utils.JsonProcess import safe_serialize
from state import AgentState  # 您的状态定义文件


class MainAgent:
    """
    基于 LangGraph 的主协调智能体。

    流程：
    1. 加载文件 → 提取元数据 → LLM 选择目标数据集
    2. 调用 Process Agent 子图执行任务
    3. 生成最终报告
    """

    def __init__(
        self,
        dataset_manager,      # DatasetManager 实例
        process_agent,        # Process Agent 子图（已编译的 LangGraph 图）
        data_analyzer,        # 元数据解析器实例
        llm=None,             # 语言模型实例（用于文件选择和报告生成）
        checkpointer=None     # 状态持久化器（默认使用内存）
    ):
        print("main agent初始化")
        self.manager = dataset_manager
        self.process_agent = process_agent
        self.analyzer = data_analyzer
        self.llm = llm
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = self._build_graph()
        

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------
    def _build_graph(self):
        """构建 LangGraph 工作流"""
        workflow = StateGraph(AgentState)
        print("\n构建graph\n")
        # 添加节点
        workflow.add_node("load_and_analyze", self.load_and_analyze_files)
        workflow.add_node("process_agent", self.process_agent_node)
        workflow.add_node("generate_report", self.generate_report)
        # 设置入口
        workflow.set_entry_point("load_and_analyze")

        # 条件路由：根据选择结果决定是否进入处理
        workflow.add_conditional_edges(
            "load_and_analyze",
            self.should_continue,
            {
                "process": "process_agent",
                "error": END,
                "end": END  
            }
        )
        # 处理完成后生成报告并结束
        workflow.add_edge("process_agent", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # 节点函数
    # ------------------------------------------------------------------
    def _build_clean_metadata(self, metadata: Dict) -> Dict:
        """
        深度清理 metadata，移除所有 numpy 类型
        """
        import numpy as np
        
        def clean_value(v):
            if v is None:
                return None
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                if np.isnan(v):
                    return None
                if np.isinf(v):
                    return None
                return float(v)
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (np.ndarray,)):
                if v.size > 100:
                    return f"array_shape_{v.shape}"
                return v.tolist()
            if isinstance(v, dict):
                return {k: clean_value(val) for k, val in v.items()}
            if isinstance(v, list):
                return [clean_value(item) for item in v]
            return v
        
        return clean_value(metadata)
    
    def load_and_analyze_files(self, state: AgentState) -> Dict[str, Any]:
        """
        节点1：加载文件 → 提取元数据 → LLM 选择目标数据集

        所需外部接口：
        - self.manager.add(dataset, key=None, file_path=None) -> str
        - self.manager.list_keys() -> List[str]
        - self.manager.get_file_path(key) -> str
        - self.analyzer.extract_metadata(file_path) -> Dict
        - self.llm.invoke(prompt) -> response (可选，用于文件选择)
        """

        user_input = ""
        if state.get("messages"):
            last_msg = state["messages"][-1]
            user_input = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

        all_keys = self.manager.list_keys()
        raw_metadata = self.manager.get_all_light_metadata()
        metadata = safe_serialize(raw_metadata)

        # ---- 3. 拦截文件加载指令 ----
        lower_input = user_input.strip().lower()
        if lower_input.startswith(('load', '加载', '添加', '读取')):
            file_paths = self._extract_file_paths(user_input)
            if file_paths:
                existing_selected = state.get("selected_dataset_keys", [])
                new_keys = [k for k in all_keys if k not in existing_selected]
                updated_selected = existing_selected + new_keys
                return safe_serialize({
                    "file_paths": state.get("file_paths", []) + file_paths,
                    "metadata": metadata,
                    "selected_dataset_keys": updated_selected,
                    "current_cache_key": updated_selected[0] if updated_selected else None,
                    "final_answer": f"已成功加载文件：{', '.join(file_paths)}。",
                    "next_step": "end"
                })
            else:
                return safe_serialize({
                    "final_answer": "未检测到有效的文件路径。",
                    "next_step": "end"
                })

        if self.llm and metadata:
            selected_keys = self._select_relevant_files(user_input, metadata)
        else:
            selected_keys = list(metadata.keys())

        all_file_paths = [
            self.manager.get_file_path(k) for k in all_keys 
            if self.manager.get_file_path(k)
        ]

        result = {
            "file_paths": all_file_paths,
            "metadata": metadata,
            "current_cache_key": selected_keys[0] if selected_keys else None,
            "selected_dataset_keys": selected_keys,
            "next_step": "process" if selected_keys else "error"
        }

        return safe_serialize(result)

    def _select_relevant_files(
        self,
        user_input: str,
        available_metadata: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        使用 LLM 根据用户输入和每个文件的元数据摘要，选出需要处理的数据集键。
        若 LLM 无法判断或返回无效结果，则返回最近添加的至多 10 个数据集键。

        所需外部接口：
        - self.llm.invoke(prompt) -> 具有 content 属性的响应对象
        """
        if not available_metadata:
            return []

        # 构建精简的元数据摘要供 LLM 分析
        summary_lines = []
        for key, meta in available_metadata.items():
            raw_vars = meta.get("variables", [])[:20]
            # 将变量列表转换为纯字符串列表
            vars_list = []
            for var in raw_vars:
                if isinstance(var, dict):
                    # 如果有 'name' 字段，取 name；否则转成字符串
                    vars_list.append(var.get('name', str(var)))
                else:
                    vars_list.append(str(var))
            
            dims = meta.get("dimensions", {})
            attrs = meta.get("global_attrs", {})
            title = attrs.get("title", "无标题")
            summary = f"- 键: {key}\n  变量: {', '.join(vars_list)}\n  维度: {dims}\n  标题: {title}"
            summary_lines.append(summary)
        summaries_text = "\n".join(summary_lines)
        prompt = f"""
用户请求: "{user_input}"

当前已加载的数据集元数据摘要:
{summaries_text}

请根据用户请求，判断当前dataset manager当中哪些数据集与任务相关。返回一个 JSON 数组，包含相关数据集的键名。
例如: ["ds_abc123", "ds_def456"]
如果用户请求没有明确指向任何数据集，或者无法判断，请返回空数组 []。
只输出 JSON 数组，不要包含任何其他文字。
"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            # 提取 JSON 数组
            print(f"\n返回的相关数据集：{content}\n")
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                selected = json.loads(match.group())
                if isinstance(selected, list):
                    # 过滤掉不存在的键
                    valid_keys = [k for k in selected if k in available_metadata]
                    if valid_keys:
                        return valid_keys
        except Exception as e:
            print(f"[MainAgent] LLM 文件选择失败: {e}")

        # 回退逻辑：返回最近添加的至多10个数据集键
        all_keys = list(available_metadata.keys())
        return all_keys[-10:] if len(all_keys) > 10 else all_keys

    def should_continue(self, state: AgentState) -> Literal["process", "error"]:
        """条件边：判断是否有选中的数据集"""
        if state.get("next_step") == "end":
            return "end" 
        selected = state.get("selected_dataset_keys", [])
        if selected and state.get("next_step") != "error":
            return "process"
        return "error"

    def process_agent_node(self, state: AgentState) -> Dict[str, Any]:
        """
        节点2：调用 Process Agent 子图执行实际任务。

        所需外部接口：
        - self.process_agent 必须是一个可调用的 LangGraph 子图（CompiledGraph），
          接收当前 state，返回部分更新的 state 字典。
        """
        try:
            print("\nprocess_agent_node\n")
            # 将当前 state 传递给 Process Agent 子图
            messages = state.get("messages", [])
            original_request = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    original_request = msg.content
                    break
            
            # 2. 保存原始任务，供 generate_report 使用
            state["original_task"] = original_request
            
            # 3. 用 LLM 分解步骤
            if self.llm and original_request:
                file_paths = state.get("file_paths", [])  # ← 从 state 获取
                file_paths_str = "\n".join([f"- {fp}" for fp in file_paths]) if file_paths else "未提供文件路径"
                decompose_prompt = f"""你是一个NetCDF数据处理领域的专家，负责将下面用户任务分解为具体的、可执行的单步步骤，
                并以'step1：...，step2：...'这样的格式输出，每个步骤应简洁明确。

    用户任务：
{original_request}

    可用的文件路径：
{file_paths_str}

    要求：
    1. 每个步骤只包含一个明确的操作（如加载数据、提取变量、计算统计等）
    2. 步骤之间要有逻辑先后顺序
    3. 如果任务简单，可以只有1-2步
    4. 直接输出步骤列表，不要添加任何额外说明
    5，不要用占位符或占位路径，直接参考当前文件路径
    """
                response = self.llm.invoke(decompose_prompt)
                steps_text = response.content.strip()
                print(f"原任务为：{original_request}，\n任务分解结果：\n{steps_text}\n")
                rethink_prompt=f"""用户任务为：\n{original_request}\n\n
                 可用的文件路径：\n{file_paths_str}
可以参考初步分解后的步骤为：{steps_text}\n\n"""
                # 4. 将分解后的步骤作为新的消息追加到消息列表
                state["messages"].append(HumanMessage(content=rethink_prompt))
            
            result_state = self.process_agent.invoke(state)
            print("\nprocess agent 调用完毕\n")
            result= {
                "execution_result": result_state.get("execution_result"),
                #"current_tool_chain": result_state.get("current_tool_chain", {}),
                "final_answer": result_state.get("final_answer", ""),
                "current_cache_key": result_state.get("current_cache_key"),
                "next_step": "report"
            }
            print("\n========调用完成到report=========\n")
            print(result_state["final_answer"])
            return safe_serialize(result)
        except Exception as e:
            result= {
                "error_info": f"Process Agent 执行失败: {str(e)}",
                "next_step": "error"
            }
            return safe_serialize(result)

    def generate_report(self, state: AgentState) -> Dict[str, Any]:
        """
        节点3：基于执行结果生成最终报告。

        所需外部接口：
        - self.llm (可选) 用于生成自然语言报告
        """
        execution_result = state.get("execution_result")
        final_answer = state.get("final_answer", "")
        user_task = state["messages"][-1].content if state["messages"] else ""

        # 使用 LLM 生成报告（如果有）
        if self.llm:
            prompt = f"""
            用户任务：{user_task}
            处理结果：{final_answer}
            详细执行数据：{execution_result}
            整合一下全部内容，根据用户任务，生成回答。不要生成多余信息，用自然语言把结果和执行过程整合表述一下。
            """
            response = self.llm.invoke(prompt)
            report = response.content
        else:
            report = f"任务完成。\n结果摘要：{final_answer}\n详细结果：{execution_result}"

        result= {
            "final_answer": report,
            "messages": [AIMessage(content=report)],
            "next_step": "end"
        }
        return safe_serialize(result)

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------
   
   
    def _extract_file_paths(self, text: str) -> List[str]:
        """
        从文本中提取所有 .nc 文件路径（支持绝对、相对路径）。
        如果 data manager 中尚未加载该文件，则自动加载并添加；
        如果已存在，则直接返回该文件的路径。
        """
        print("\n _extract_file_paths\n")
        pattern = r"""["']?(\S+?\.nc)["']?"""
        raw_matches = re.findall(pattern, text)

        valid_paths = []
        seen = set()

        for raw in raw_matches:
            norm = os.path.normpath(raw)

            if not norm or norm in seen:
                continue
            seen.add(norm)

            if not os.path.exists(norm):
                print(f"[警告] 文件不存在，跳过: {norm}")
                continue

            already_loaded = False
            for key in self.manager.list_keys():
                existing_path = self.manager.get_file_path(key)
                if existing_path and os.path.normpath(existing_path) == norm:
                    print("manager中已加载文件")
                    already_loaded = True
                    break

            if not already_loaded:
                try:
                    import xarray as xr
                    ds = xr.open_dataset(norm, decode_times=True)
                    new_key = self.manager.add(ds, file_path=norm)
                    full_meta = self.analyzer.extract_metadata(norm)
                    light_meta = {
                        "variables": [v["name"] for v in full_meta.get("variables", [])],
                        "dimensions": dict(full_meta.get("dimensions", {})),
                        "title": full_meta.get("global_attributes", {}).get("title", "")
                    }
                    self.manager.cache_light_metadata(new_key, light_meta)
                    print(f"[Manager] 已添加文件: {norm}")
                except Exception as e:
                    print(f"[错误] 无法加载文件 {norm}: {e}")
                    continue

            valid_paths.append(norm)

        return valid_paths

    def _cleanup_temp_datasets(self, keep_keys: List[str] = None):
        """
        删除 DatasetManager 中没有关联物理文件的数据集（临时中间结果）。
        
        Args:
            keep_keys: 额外需要保留的数据集键列表（例如用户明确要求保留的结果）
        """
        if keep_keys is None:
            keep_keys = []
        to_remove = []
        for key in self.manager.list_keys():
            # 如果有关联的原始文件路径，则保留
            if self.manager.get_file_path(key) is not None:
                continue
            # 如果该键在 explicitly keep 列表中，也保留
            if key in keep_keys:
                continue
            # 标记为待删除
            to_remove.append(key)
        for key in to_remove:
            self.manager.remove(key)
            if self.process_agent and self.process_agent.current_dataset_key == key:
                self.process_agent.current_dataset_key = None
        if to_remove:
            print(f"[Cleanup] 已删除 {len(to_remove)} 个临时数据集: {to_remove}")
        # ------------------------------------------------------------------
    # 公共调用接口
    # ------------------------------------------------------------------
    def _build_initial_state(self, user_input: str, session_id: Optional[str] = None) -> AgentState:
        """
        构建初始状态字典。
        
        Args:
            user_input: 用户输入的自然语言请求
            session_id: 会话标识，用于状态持久化（可选）
        
        Returns:
            初始化的 AgentState 字典
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        user_prompt=f"""
当前用户任务为：{user_input}，
当前session_id为：{session_id}，
需要思考接下来

"""
        
        initial_state: AgentState = {
            "session_id": session_id,
            "messages": [HumanMessage(content=user_input)],
            "file_paths": [],
            "metadata": {},
            "current_cache_key": None,
            "current_tool_chain": {},
            "execution_result": None,
            "current_code": "",
            "error_info": "",
            "error_history": [],
            "iteration_count": 0,
            "next_step": "",
            "final_answer": "",
            "selected_dataset_keys": []
        }
        
        return safe_serialize(initial_state)
      
    def run(self, user_input: str, session_id: str) -> str:
            print(f"\n开始对话：{user_input}, 会话ID: {session_id}\n")

            config = {"configurable": {"thread_id": session_id}}

            initial_state = self._build_initial_state(user_input, session_id)
            initial_state["final_answer"] = ""
            initial_state["execution_result"] = None
            initial_state["next_step"] = ""
            initial_state["error_info"] = ""
            initial_state["current_code"] = ""
            initial_state["current_tool_chain"] = {}
            initial_state["iteration_count"] = 0
            final_state = self.graph.invoke(initial_state, config)
    
            
            keep = final_state.get("selected_dataset_keys", [])
            if final_state.get("current_cache_key"):
                keep.append(final_state["current_cache_key"])
            self._cleanup_temp_datasets(keep_keys=keep)
            
            
            return final_state.get("final_answer", "处理完成，但未生成报告。")
    
 