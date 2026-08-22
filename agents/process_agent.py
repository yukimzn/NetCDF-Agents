
import re
import json
import os
from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

import warnings
import os

# 屏蔽所有警告（包括 Pydantic）
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# 导入 DatasetManager 和工具函数
from utils.DatasetManager import DatasetManager
from utils.JsonProcess import safe_serialize,to_json
from tools.tool import (
    set_dataset_manager,
    load_dataset,
    save_dataset,
    list_variables,
    list_dimensions,
    get_variable_info,
    get_global_attributes,
    extract_variable,
    rename_variable,
    delete_variable,
    extract_level,
    reduce_dimension,
    calculate_statistics,
    spatial_statistics,
    time_statistics,
    spatial_subset,
    mask_by_region,
    time_subset,
    time_slice,
    regrid,
    regrid_unstructured_to_structured,
    merge_files,
    split_by_time,
    subset_by_index,
    compare_datasets,
    release_dataset,
    get_dataset_summary,
    filter_by_condition,apply_math_operation

)
from agents.coding import CodeGenerationAgent, CodeCorrectionAgent

class ProcessAgent:
    """ReAct 模式 NetCDF 数据处理智能体，融合【工具优先、代码兜底、自纠错闭环】"""
    
    def __init__(self, model: str = "qwen3-coder:480b-cloud", verbose: bool = True, manager: Optional[DatasetManager] = None):
        self.llm = ChatOllama(model=model, temperature=0)
        self.dsllm=ChatOllama(model="deepseek-coder-v2:16b",temperature=0)
        self.largellm=ChatOllama(model="qwen3-coder:480b-cloud",temperature=0)
        self.verbose = verbose

        # 允许外部注入共享的 DatasetManager
        if manager is None:
            self.manager = DatasetManager()
        else:
            self.manager = manager
        set_dataset_manager(self.manager)
        
        self.code_correction_agent = CodeCorrectionAgent(llm=self.largellm)
        self.code_generation_agent = CodeGenerationAgent(llm=self.largellm)
        # 建立纠错闭环
        self.code_generation_agent.set_correction_agent(self.code_correction_agent)

        
        # 工具映射表（所有工具均使用 dataset_key 参数）
        self.tool_map = {
            "load_dataset": load_dataset,
            "save_dataset": save_dataset,
            "list_variables": list_variables,
            "list_dimensions": list_dimensions,
            "get_variable_info": get_variable_info,
            "get_global_attributes": get_global_attributes,
            "extract_variable": extract_variable,
            "rename_variable": rename_variable,
            "delete_variable": delete_variable,
            "extract_level": extract_level,
            "reduce_dimension": reduce_dimension,
            "calculate_statistics": calculate_statistics,
            "spatial_statistics": spatial_statistics,
            "time_statistics": time_statistics,
            "spatial_subset": spatial_subset,
            "mask_by_region": mask_by_region,
            "time_subset": time_subset,
            "time_slice": time_slice,
            "regrid": regrid,
            "regrid_unstructured_to_structured": regrid_unstructured_to_structured,
            "merge_files": merge_files,
            "split_by_time": split_by_time,
            "subset_by_index": subset_by_index,
            "compare_datasets": compare_datasets,
            "release_dataset":release_dataset,
            "get_dataset_summary":get_dataset_summary,
            "filter_by_condition":filter_by_condition,
            "apply_math_operation":apply_math_operation,
        }
        
        # 会话状态
        self.current_dataset_key: Optional[str] = None  # 当前活动数据集的键
        self.file_to_key_map: Dict[str, str] = {}       # 文件路径 -> 数据集键的映射
        self.conversation_history: List[Dict] = []
        self.max_steps = 10
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_checkpoint_path(self, session_id: str) -> str:
        """获取检查点文件路径"""
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.checkpoint_dir, f"process_agent_{safe_id}.json")
    
    def _save_checkpoint(self, session_id: str, data: Dict[str, Any],save):
        """保存检查点"""
        if not save:
            return 
        try:
            # 序列化 chat_messages（LangChain 消息对象需转换）
            if "chat_messages" in data:
                data["chat_messages"] = [
                    {"type": type(msg).__name__, "content": msg.content}
                    for msg in data["chat_messages"]
                ]
            
            # 添加时间戳
            from datetime import datetime
            data["timestamp"] = datetime.now().isoformat()
            
            filepath = self._get_checkpoint_path(session_id)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[检查点已保存] {filepath}")
        except Exception as e:
            print(f"[检查点保存失败] {e}")
    
    def _load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        try:
            filepath = self._get_checkpoint_path(session_id)
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 恢复 chat_messages
            if "chat_messages" in data:
                restored = []
                for msg in data["chat_messages"]:
                    
                    if msg["type"] == "HumanMessage":
                        restored.append(HumanMessage(content=msg["content"]))
                    elif msg["type"] == "AIMessage":
                        restored.append(AIMessage(content=msg["content"]))
                    elif msg["type"] == "SystemMessage":
                        restored.append(SystemMessage(content=msg["content"]))
                        
                    
                data["chat_messages"] = restored
            
            print(f"[检查点已加载] {filepath}")
            return data
        except Exception as e:
            print(f"[检查点加载失败] {e}")
            return None

    def _assess_task_feasibility(self, user_task: str) -> bool:
        """评估现有工具链是否能解决任务"""
        tools_desc = self._build_tools_description(self.tool_map)
        eval_prompt = f"""你是一个 NetCDF 数据处理系统的路由专家。
请评估用户任务是否可以通过工具链完成：

【可用工具列表】：
{tools_desc}

【用户任务】：
{user_task}

【判断规则】：
1. 如果用户请求可以使用上面的工具（组合）完成（如加载、提取变量、计算统计量、裁剪等），回复 "YES"。
2. 如果涉及复杂公式、自定义绘图、特定插值等缺失功能，回复 "NO"。
3. 如果包含“可视化”、“图表”、“计算复杂气象指数”或“自定义高级算法”，强制回复 "NO"。

请仅回答 "YES" 或 "NO"。
"""
        response = self.llm.invoke([SystemMessage(content=eval_prompt)])
        result = response.content.strip().upper()
        print(f"\n[混合路由预审] 是否可用现有工具: {result}")
        return "YES" in result
    
    def _build_tools_description(self, tools_subset: Dict[str, Any]) -> str:
        """根据选中的工具子集生成包含参数说明的描述（用于 prompt）"""
        lines = []
        for name, tool in tools_subset.items():
            # 获取工具描述
            desc = getattr(tool, 'description', '') or getattr(tool, '__doc__', '')
            if desc:
                short_desc = desc.split('\n')[0].strip()
            else:
                short_desc = name

            # 提取参数信息（如果有 args_schema）
            args_info = ""
            if hasattr(tool, 'args_schema') and tool.args_schema is not None:
                schema = tool.args_schema
                if hasattr(schema, '__fields__'):  # Pydantic v1/v2
                    fields = schema.__fields__
                    params = []
                    for field_name, field in fields.items():
                        field_type = field.annotation.__name__ if hasattr(field.annotation, '__name__') else str(field.annotation)
                        required = "必需" if field.is_required() else "可选"
                        params.append(f"{field_name}: {field_type}({required})")
                    if params:
                        args_info = f" 参数: {', '.join(params)}"
                elif hasattr(schema, 'properties'):  # 备用
                    props = schema.get('properties', {})
                    params = [f"{p}: {info.get('type', 'any')}" for p, info in props.items()]
                    if params:
                        args_info = f" 参数: {', '.join(params)}"

            # 合并成一行
            lines.append(f"- {name}: {short_desc} {args_info}".strip())
        return "\n".join(lines)
    
    def get_current_key(self) -> Optional[str]:
        return self.current_dataset_key
    
    def _build_system_prompt(self, user_input: str, state: Optional[Dict] = None) -> str:
        
        # ================= 明确调用的位置 2 =================
        # 获取轻量元数据字典
        light_meta = self.manager.get_all_light_metadata()
        # ===================================================
        
        dataset_lines = []
        for key, meta in light_meta.items():
            vars_list = meta.get("variables", [])
            dims = meta.get("dimensions", {})
            dataset_lines.append(f"- {key}: 变量={vars_list}, 维度={list(dims.keys())}")
            
        datasets_info = "\n".join(dataset_lines) if dataset_lines else "无"

        # 若 state 提供了预选数据集键，额外强调
        if state:
            selected = state.get("selected_dataset_keys", [])
            if selected:
                datasets_info += "\n\n**任务目标数据集（优先使用）**: " + ", ".join(selected)

        current_info = f"当前活动数据集: {self.current_dataset_key}" if self.current_dataset_key else "无活动数据集"

        tools_desc = self._build_tools_description(self.tool_map)

        return f"""你是一个 NetCDF 数据处理助手，采用 ReAct 模式逐步完成任务。

**当前已加载的数据集**：
{datasets_info}

{current_info}

**可用工具列表**：
{tools_desc}

规则：
1. 首次操作通常调用 `load_dataset`，参数 `file_path` 从可用文件路径中选择。它会返回一个包含 `cache_key` 的 JSON，请记住这个键。
2. 后续操作的数据集参数名为 `dataset_key`，使用上一步返回的键或管理器中的已有键。
3. 修改操作会返回 `new_cache_key`，表示生成了新数据集，后续操作请使用这个新键。
4. 查询操作直接返回结果，不会产生新数据集。
5. 如果需要同时处理多个数据集，可以使用 `compare_datasets` 等工具，传入 `dataset_keys` 列表。
6. 严禁编造变量名或坐标名，应先调用 `list_variables` 或 `list_dimensions` 查看可用名称。
7. 涉及输入变量作为参数时，参数名为 "var_name"。
8. 工具名称只能从上述列表中选择。
9. 当需要了解数据集的具体变量、维度、属性时，请调用 `get_dataset_summary` 并传入目标 dataset_key。
10. 当不再需要某个中间数据集时，可调用 `release_dataset` 释放内存。
11. 加载文件后，如果用户只要求查看变量列表/维度/属性，直接从工具返回的结果中提取并输出 final_answer，不要逐个查看单个变量。
12. 若 load_dataset 的返回摘要已包含用户所需信息，直接使用即可，无需重复查询。
13. 路径错误时，优先使用用户提供的原始路径（如 ..\\data\\xxx.nc），不要反复尝试不同的转义方式。

14. 【跨数据集运算指南】
当需要计算两个数据集的差值、和、积等时，按以下三步操作：
   步骤A：使用 merge_files 工具将两个数据集合并为一个，参数 merge_dim="time"
   步骤B：使用 get_dataset_summary 查看合并后的变量名（通常会自动添加后缀）
   步骤C：使用 apply_math_operation 对合并后的两个变量进行计算
   
   示例：计算 ds_day1 和 ds_day15 中变量 t 的差值
   → merge_files(file_paths=["...day1.nc", "...day15.nc"], merge_dim="time")  // 返回 ds_merged
   → get_dataset_summary(dataset_key="ds_merged")  // 确认变量名变为 t_0, t_1
   → apply_math_operation(dataset_key="ds_merged", var_name="t_0", operation="subtract", operand="t_1")  // 差值

15. 如果 merge_files 合并后变量名重复，请用 rename_variable 重命名后再计算。

16. 对于空间统计，使用 spatial_statistics 工具，它已经内置了经纬度裁剪功能。

 **输出格式**：
每一步输出一个严格的 JSON 对象，包含以下字段：
{{
    "thought": "分析当前状态，解释下一步行动的原因",
    "action": "要调用的工具名称",
    "action_input": {{"参数名": "参数值", ...}}
}}
当任务完成时，输出：
{{
    "thought": "任务已完成，总结如下",
    "final_answer": "最终回答内容，包含关键结果"
}}
"""

    def _parse_llm_response(self, response_text: str) -> Optional[Dict]:
        """从 LLM 响应中解析 JSON，处理可能的 Markdown 代码块"""
        # 尝试提取 JSON 块
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1) if json_match.lastindex else json_match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """执行工具并返回解析后的结果字典"""
        tool_func = self.tool_map.get(tool_name)
        if not tool_func:
            return {"status": "error", "message": f"工具 '{tool_name}' 不存在"}
        
        try:
            # 所有工具现在都只接受参数，通过 invoke 调用
            result_str = tool_func.invoke(params)
            result = json.loads(result_str)
            return result
        except Exception as e:
            return {"status": "error", "message": f"工具执行异常: {str(e)}"}

    def _update_state_from_result(self, result: Dict):
        """根据工具返回结果更新当前数据集键和文件映射"""
        if result.get("status") != "success":
            return
        
        # 处理 load_dataset 或 merge_files 返回的 cache_key
        if "cache_key" in result:
            
            self.current_dataset_key = result["cache_key"]
            # 如果是 load_dataset，尝试建立文件路径到键的映射（从工具调用参数中获取）
            # 这里简单记录，实际可在工具中增加 file_path 返回，但当前不强制
        
        # 处理修改操作返回的 new_cache_key
        elif "new_cache_key" in result:
            self.current_dataset_key = result["new_cache_key"]
        
        # 如果有 file_path 信息（自定义扩展），可更新映射
        if "file_path" in result and "cache_key" in result:
            self.file_to_key_map[result["file_path"]] = result["cache_key"]

    def _fallback_to_coding(self, task: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """关键功能：降级转接至 Coding Agent，支持代码自纠错"""
        print("\n[降级兜底触发] 工具无法直接完成，正在切换至高级代码生成(Coding Agent)模式...")
        
        # ================= 修改：提取所有文件路径和元数据 =================
        file_paths = state.get("file_paths", [])
        
        # 如果 state 中没有路径，回退到当前活动数据集
        if not file_paths and self.current_dataset_key:
            fp = self.manager.get_file_path(self.current_dataset_key)
            if fp:
                file_paths = [fp]
        
        # 获取所有已加载数据集的完整元数据（而非仅当前活动数据集）
        all_metadata = self.manager.get_all_light_metadata()
        
        # 过滤：只保留 file_paths 中存在的文件对应的元数据
        # 建立路径到键的映射
        path_to_key = {}
        for key in self.manager.list_keys():
            path = self.manager.get_file_path(key)
            if path:
                path_to_key[os.path.normpath(path)] = key
        
        # 收集相关数据集的元数据
        relevant_metadata = {}
        for fp in file_paths:
            norm_fp = os.path.normpath(fp)
            if norm_fp in path_to_key:
                key = path_to_key[norm_fp]
                meta = all_metadata.get(key, {})
                if meta:
                    relevant_metadata[key] = meta
        
        # 如果没有匹配到任何元数据，回退到当前数据集
        if not relevant_metadata:
            ds_key = self.current_dataset_key
            relevant_metadata = self.manager.get_light_metadata(ds_key)
        # =================================================================
        
        # 调用代码生成代理（传递完整文件路径列表和完整元数据）
        coding_result = self.code_generation_agent.execute_with_retry(
            task=task,
            metadata=relevant_metadata,  # ← 传递所有相关文件的元数据
            file_paths=file_paths,       # ← 传递所有文件路径
            max_attempts=5
        )
        return coding_result

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """智能体混合处理流程核心方法"""
        if self.current_dataset_key and not self.manager.contains(self.current_dataset_key):
            print(f"[Warning] current_dataset_key {self.current_dataset_key} 已失效，重置为 None")
            self.current_dataset_key = None
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("state 中必须包含至少一条用户消息")
        
        # 提取最后一条用户消息
        last_user_msg = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'role') and msg.role == 'user'):
                last_user_msg = msg.content
                break

        print("\n" + "=" * 60)
        print(f"用户请求: {last_user_msg}")
        print("=" * 60)

        # 1. 预加载文件
        for fp in state.get("file_paths", []):
            if not self.manager.find_key_by_file_path(fp): 
                self.load_file(fp)

        # 2. 设置当前缓存键
        current_key = state.get("current_cache_key")
        if current_key and self.manager.contains(current_key):
            self.current_dataset_key = current_key

        # 3. 断点恢复或正常初始化
        session_id = state.get("session_id", "default")
        checkpoint = self._load_checkpoint(session_id)

        if checkpoint and not state.get("force_restart"):
            self.current_dataset_key = checkpoint.get("current_dataset_key")
            self.file_to_key_map = checkpoint.get("file_to_key_map", {})

            system_prompt = self._build_system_prompt(last_user_msg, state)
            chat_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"**用户任务**: {last_user_msg}\n\n请开始第一步思考与行动。")
            ]
        else:
            system_prompt = self._build_system_prompt(last_user_msg, state)
            chat_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"**用户任务**: {last_user_msg}\n\n请开始第一步思考与行动。")
            ]
        step_count = 0    

        # 4. 策略前置判断
        can_use_tools = self._assess_task_feasibility(last_user_msg)

        if not can_use_tools:
            coding_result = self._fallback_to_coding(last_user_msg, state)
            if coding_result["success"]:
                final_answer = f"✅ 已通过代码生成完成任务。\n\n**执行结果**:\n{coding_result['result']}\n\n**生成并纠正后的代码**:\n```python\n{coding_result['code']}\n```"
            else:
                final_answer = f"❌ 代码生成失败。\n\n**错误日志**:\n{coding_result.get('error', '未知错误')}"
            
            state["final_answer"] = final_answer
            state["current_cache_key"] = self.current_dataset_key
            return state

        # 5. 进入工具链 ReAct 模式
        print("\n[策略维持] 现有工具可以完成任务，进入 ReAct 工具调用模式...\n")
        final_answer = ""
        tool_execution_failed = False
        fail = 0

        while step_count < self.max_steps:
            step_count += 1
            print(f"\n--- Step {step_count} ---")

            response = self.llm.invoke(chat_messages)
            response_text = response.content
            parsed = self._parse_llm_response(response_text)

            if parsed is None:
                chat_messages.append(AIMessage(content=response_text))
                chat_messages.append(HumanMessage(content="无法解析 LLM 响应为 JSON。"))
                self._truncate_history(chat_messages)
                continue

            if "final_answer" in parsed:
                final_answer = parsed["final_answer"]
                print(f"\n任务通过工具完成: {final_answer}")
                chat_messages.append(AIMessage(content=final_answer))
                
                self._save_checkpoint(session_id, {
                    "step_count": step_count,
                    "chat_messages": chat_messages,
                    "last_task_result": final_answer
                }, False)
                break

            if "action" not in parsed or "action_input" not in parsed:
                chat_messages.append(AIMessage(content=response_text))
                chat_messages.append(HumanMessage(content="缺少 'action' 或 'action_input' 字段。"))
                # 【修改2】continue 前执行截断
                self._truncate_history(chat_messages)
                continue

            tool_name = parsed["action"]
            tool_params = parsed["action_input"]
            print(f"思考: {parsed.get('thought', '')}")
            print(f"调用工具: {tool_name}, 参数: {tool_params}")

            result = self._execute_tool(tool_name, tool_params)
            print(f"工具结果: {json.dumps(result, ensure_ascii=False)}")
            
            if result.get("status") == "error":
                fail += 1
                if fail > 2:
                    print(f"\n⚠️ 工具执行出错: {result.get('message')}。正在触发降级机制...")
                    tool_execution_failed = True
                    self._save_checkpoint(session_id, {
                        "step_count": step_count,
                        "tool_execution_failed": True,
                        "last_error": result.get("message"),
                        "chat_messages": chat_messages
                    }, False)
                    break
            else:
                fail = 0

            self._update_state_from_result(result)
            chat_messages.append(AIMessage(content=response_text))
            chat_messages.append(HumanMessage(content=f"工具执行结果: {json.dumps(result, ensure_ascii=False)}"))
            chat_messages[0] = SystemMessage(content=self._build_system_prompt(last_user_msg, state))
            
            # 【修改3】压缩工具结果（保留原有压缩逻辑）
            if 'result' in locals() and result:
                compressed = {
                    "status": result.get("status"),
                    "key": result.get("cache_key") or result.get("new_cache_key"),
                    "msg": result.get("message", "执行完毕")
                }
                if "result" in result and isinstance(result["result"], (int, float)):
                    compressed["value"] = result["result"]
                if chat_messages and isinstance(chat_messages[-1], HumanMessage):
                    chat_messages[-1] = HumanMessage(
                        content=f"工具返回: {json.dumps(compressed, ensure_ascii=False)}"
                    )
            
            self._save_checkpoint(session_id, {
                "step_count": step_count,
                "current_dataset_key": self.current_dataset_key,
                "file_to_key_map": self.file_to_key_map,
                "chat_messages": chat_messages
            }, False)
            
            self._truncate_history(chat_messages)

        # 6. 双重兜底（工具报错 or 步数耗尽）
        if tool_execution_failed or (step_count >= self.max_steps and not final_answer):
            print("\n[兜底触发] 工具运行异常或步数超限，立即执行 Coding Agent 进行代码自动生成与纠错兜底。")
            coding_result = self._fallback_to_coding(last_user_msg, state)
            
            if coding_result["success"]:
                final_answer = f"✅ 已通过降级代码生成兜底完成复杂任务。\n\n**执行结果**:\n{coding_result['result']}\n\n**使用的代码**:\n```python\n{coding_result['code']}\n```"
            else:
                final_answer = f"❌ 代码降级兜底失败。\n\n**最终错误**:\n{coding_result.get('error', '未知错误')}"

        state["final_answer"] = final_answer
        state["current_cache_key"] = self.current_dataset_key
        state["messages"] = chat_messages
        return state
    MAX_HISTORY_ROUNDS = 4

    def _truncate_history(self, chat_messages: List):
        if len(chat_messages) > 2 + self.MAX_HISTORY_ROUNDS * 2:
            keep = [chat_messages[0], chat_messages[1]]
            keep.extend(chat_messages[-(self.MAX_HISTORY_ROUNDS * 2):])
            chat_messages[:] = keep
            if self.verbose:
                print(f"[上下文精简] 截断至 {len(chat_messages)} 条消息")

    def _format_result_for_history(self, result: Dict) -> str:
        compressed = {
            "status": result.get("status"),
            "key": result.get("cache_key") or result.get("new_cache_key"),
            "msg": result.get("message", "执行完毕")
        }
        if "result" in result and isinstance(result["result"], (int, float)):
            compressed["value"] = result["result"]
        return f"工具返回: {json.dumps(compressed, ensure_ascii=False)}"

    def _handle_code_fallback(self, task: str, state: Dict) -> Dict:
        """统一的代码降级处理"""
        coding_result = self._fallback_to_coding(task, state)
        if coding_result.get("success"):
            final_answer = f"✅ 已通过代码生成完成任务。\n\n**执行结果**:\n{coding_result['result']}\n\n**使用的代码**:\n```python\n{coding_result['code']}\n```"
        else:
            final_answer = f"❌ 代码生成失败。\n\n**错误日志**:\n{coding_result.get('error', '未知错误')}"
        
        state["final_answer"] = final_answer
        state["current_cache_key"] = self.current_dataset_key
        return state

    def load_file(self, file_path: str) -> str:
        """便捷方法：加载文件并返回状态信息"""
        if not os.path.exists(file_path):
            return f"错误: 文件 '{file_path}' 不存在"
        
        result_str = load_dataset.invoke({"file_path": file_path})
        result = json.loads(result_str)
        if result.get("status") == "success":
            self.current_dataset_key = result["cache_key"]
            self.file_to_key_map[file_path] = result["cache_key"]
            summary = result.get("summary", {})
            return f"加载成功。键: {self.current_dataset_key}\n摘要: {json.dumps(summary, ensure_ascii=False)}"
        else:
            return f"加载失败: {result.get('message')}"

    def show_status(self) -> str:
        """显示当前管理器状态"""
        keys = self.manager.list_keys()
        if not keys:
            return "管理器中没有数据集。"
        lines = ["当前管理器中的数据集:"]
        for key in keys:
            ds = self.manager.get(key)
            if ds:
                vars_list = list(ds.data_vars.keys())
                lines.append(f"  {key}: {len(vars_list)} 个变量, 维度 {dict(ds.sizes)}")
        if self.current_dataset_key:
            lines.append(f"\n当前活动数据集键: {self.current_dataset_key}")
        return "\n".join(lines)


def main():
    print("=" * 60)
    print("NetCDF ReAct 智能体 (基于 DatasetManager)")
    print("=" * 60)
    
    # 修正这里的类名为 ProcessAgent，而不是原来的 NetCDFReActAgent
    agent = ProcessAgent(model="qwen3:8b", verbose=True)
    
    # 可选：预加载文件
    file_path = input("\n请输入初始 NetCDF 文件路径（可跳过）: ").strip()
    if file_path:
        print(agent.load_file(file_path))
    
    print("\n命令说明：")
    print("- 直接输入处理需求，智能体会自动逐步执行")
    print("- 输入 'load <文件路径>' 手动加载文件")
    print("- 输入 'status' 查看当前管理器状态")
    print("- 输入 'quit' 或 'exit' 退出")
    print("-" * 60)
    
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("退出智能体")
            break
        if not user_input:
            continue
        
        # 特殊命令处理
        if user_input.lower().startswith("load "):
            file_path = user_input[5:].strip()
            print(agent.load_file(file_path))
            continue
        
        if user_input.lower() == "status":
            print(agent.show_status())
            continue
        
        # 正常 ReAct 处理
        try:
            # 修改这里以适应 state 的传入
            state = {
                "messages": [HumanMessage(content=user_input)],
                "file_paths": []
            }
            final_state = agent.invoke(state)
            print("\n" + "=" * 60)
            print(f"最终答案: {final_state['final_answer']}")
            print("=" * 60)
        except Exception as e:
            print(f"处理出错: {str(e)}")


if __name__ == "__main__":
    main()