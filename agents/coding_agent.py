"""
coding.py
代码生成智能体 + 结构化错误解析 + 迭代纠错
支持 DatasetManager 内存数据访问，符合 Process Agent 接口
"""

import re
import json
import traceback
import builtins
from typing import Dict, Any, List, Optional, Union

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

# 假设 DatasetManager 和工具已正确配置
from tools.tool import get_manager, set_dataset_manager


# ----------------------------------------------------------------------
# 辅助函数：错误解析、代码提取
# ----------------------------------------------------------------------

def parse_error(error_text: str) -> Dict[str, Any]:
    """
    解析错误信息为结构化字典（不包含修复建议）
    :param error_text: 原始错误字符串
    :return: 结构化错误信息
    """
    print(f"\n错误信息解析：{error_text}\n")
    structured = {
        "error_type": "unknown",
        "line": None,
        "full_error": error_text,          # 完整错误堆栈
        "traceback": None                  # 最后几行堆栈摘要
    }
    
    # 提取错误类型（取第一行的冒号前内容）
    lines = error_text.strip().splitlines()
    if lines:
        first_line = lines[0]
        if ":" in first_line:
            possible_type = first_line.split(":", 1)[0].strip()
            if possible_type and not possible_type.startswith("Traceback"):
                structured["error_type"] = possible_type
    
    # 提取行号（常见格式 "line 123", "File \"...\", line 123"）
    line_match = re.search(r"line\s+(\d+)", error_text, re.IGNORECASE)
    if line_match:
        structured["line"] = int(line_match.group(1))
    
    # 提取最后几行作为堆栈摘要（便于LLM快速定位）
    structured["traceback"] = "\n".join(lines[-5:]) if len(lines) > 5 else error_text
    
    return structured

def extract_python_code(response: str) -> Optional[str]:
    """从 LLM 响应中提取 Python 代码"""
    if not response:
        return None
    code_patterns = [
        r'```python\s*\n(.*?)```',
        r'```python\s*(.*?)```',
        r'```\s*\n(.*?)```',
        r'```\s*(.*?)```',
    ]
    for pattern in code_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    # 没有代码块，尝试从第二行开始提取
    lines = response.splitlines()
    if len(lines) > 1:
        return '\n'.join(lines[1:]).strip()
    return None


# ----------------------------------------------------------------------
# 进程内代码执行器（集成 DatasetManager）
# ----------------------------------------------------------------------

class InProcessCodeExecutor:
    """在受限命名空间中执行 Python 代码，包裹固定前后置代码块，增强容错与结果提取"""

    @staticmethod
    def execute(code: str, allowed_modules: Optional[Dict] = None,
                file_paths: Optional[Dict[str, str]] = None,   # 新增：键 → 文件路径
                current_key: Optional[str] = None  ) -> Dict[str, Any]:
        """
        执行代码并返回结果
        """
        import numpy as np
        import xarray as xr
        import json as json_module

        # ---- 前置代码块：导入常用模块、编码修复、设置 ----
        preamble = """
# === 前置代码：导入常用模块 ===
import sys
import os
import io
import json
import numpy as np
import xarray as xr
import traceback

# === 编码修复（Windows 环境） ===
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# === 设置文件路径变量（由外部填入） ===
_key_file_paths = {}
_current_key = ""
"""

        # ---- 后置代码块：结果捕获、序列化、输出标记 ----
        postamble = """
# === 后置代码：增强的结果捕获 ===
_result = None
_result_error = None

try:
    # 尝试从局部或全局命名空间获取 result
    if 'result' in dir():
        _result = result
    elif 'result' in globals():
        _result = globals()['result']
except Exception as e:
    _result_error = f"获取 result 时出错: {e}"

# 序列化并输出结果，使用特定标记包裹
print("\\n===RESULT_START===输出结果")
try:
    if _result_error:
        print(f"ERROR: {_result_error}")
    elif _result is None:
        print("None")
    elif isinstance(_result, (dict, list)):
        # 对复杂对象进行转换
        def _convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert(item) for item in obj]
            else:
                return obj
        try:
            converted = _convert(_result)
            print(json.dumps(converted, ensure_ascii=False, indent=2))
        except:
            print(str(_result))
    else:
        print(str(_result))
except Exception as e:
    print(f"结果序列化失败: {e}")
print("===RESULT_END===")

# 刷新输出
sys.stdout.flush()
sys.stderr.flush()
"""

        # 拼接完整代码
        full_code = preamble + "\n" + code + "\n" + postamble

        safe_globals = {
            '__builtins__': {
                'print': builtins.print,
                'len': builtins.len,
                'range': builtins.range,
                'enumerate': builtins.enumerate,
                'zip': builtins.zip,
                'int': builtins.int,
                'float': builtins.float,
                'str': builtins.str,
                'list': builtins.list,
                'dict': builtins.dict,
                'tuple': builtins.tuple,
                'set': builtins.set,
                'bool': builtins.bool,
                'abs': builtins.abs,
                'min': builtins.min,
                'max': builtins.max,
                'sum': builtins.sum,
                'round': builtins.round,
                'sorted': builtins.sorted,
                'isinstance': builtins.isinstance,
                'Exception': builtins.Exception,
                '__import__': builtins.__import__,  # 允许导入模块
            },
            '__name__': '__main__',
            'np': np,
            'xr': xr,
            'json': json_module,
            'get_manager': get_manager,
        }
        safe_globals['_key_file_paths'] = file_paths or {}
        safe_globals['_current_key'] = current_key or ""

        # 合并额外模块
        if allowed_modules:
            safe_globals.update(allowed_modules)

        # ---- 执行 ----
        local_ns = {}
        success = False
        error_msg = None

        try:
            # 重定向标准输出以便捕获 print 内容
            import io as io_module
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io_module.StringIO()
            sys.stderr = io_module.StringIO()

            try:
                exec(full_code, safe_globals, local_ns)
                success = True
            except Exception as e:
                success = False
                error_msg = traceback.format_exc()
            finally:
                # 获取捕获的输出
                captured_stdout = sys.stdout.getvalue()
                captured_stderr = sys.stderr.getvalue()
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        except Exception as e:
            success = False
            error_msg = f"执行环境异常: {traceback.format_exc()}"
            captured_stdout = ""
            captured_stderr = str(e)
        finally:
            # 确保标准输出被恢复
            try:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            except:
                pass

        # ---- 从捕获的输出中提取结果 ----
        exec_result = None
        if captured_stdout:
            match = re.search(r'===RESULT_START===\s*\n(.*?)\n===RESULT_END===', captured_stdout, re.DOTALL)
            if match:
                result_text = match.group(1).strip()
                if result_text == "None" or result_text.startswith("ERROR:"):
                    exec_result = None
                else:
                    try:
                        exec_result = json_module.loads(result_text)
                    except:
                        exec_result = result_text

        return {
            "status": "success" if success else "error", 
            "success": success,
            "stdout": captured_stdout,
            "stderr": captured_stderr if success else (error_msg or captured_stderr),
            "execution_result": exec_result,
            "returncode": 0 if success else 1
        }

# ----------------------------------------------------------------------
# 代码生成智能体
# ----------------------------------------------------------------------

class CodeGenerationAgent:
    """生成 Python 代码，支持错误上下文和历史"""

    def __init__(self, llm: ChatOllama, system_prompt: str = None):
        self.llm = llm
        self.system_prompt = system_prompt or "你是一个专业的 NetCDF 数据分析师，使用 Python 处理数据。"
        self.correction_agent = None

    def set_correction_agent(self, correction_agent):
        self.correction_agent = correction_agent

    def generate_code(
        self,
        task: str,
        target_keys: List[str],
        metadata: Dict[str, Any],           # {key: {variables: [...], dimensions: {...}}}
        current_key: Optional[str] = None,
        parameters: Optional[Dict] = None,
        error_info: str = None,
        error_history: List[Dict] = None,
        fix_suggestions: str = None
    ) -> str:
        """
        生成代码，支持纠错上下文。
        """
        # 构建元数据描述（支持多文件）
        metadata_desc_lines = ["## 可用数据集（通过键获取）"]
        for key in target_keys:
            meta = metadata.get(key, {})
            vars_list = meta.get('variables', [])
            var_display = ', '.join(vars_list[:10])
            if len(vars_list) > 10:
                var_display += f" ... (共{len(vars_list)}个)"
            metadata_desc_lines.append(
                f"- 键: '{key}', 变量: {var_display}"
            )
        if current_key:
            metadata_desc_lines.append(f"\n当前活动数据集键: '{current_key}'")

        metadata_constraint = "\n".join(metadata_desc_lines)

        # 历史错误提示
        history_prompt = ""
        if error_history:
            history_prompt = "\n## 历史错误（请避免）\n"
            for i, err in enumerate(error_history[-3:], 1):
                history_prompt += f"{i}. {err.get('error_type')}: {err.get('full_error', '')[:200]}\n"

        # 纠错指令
        fix_instruction = ""
        if error_info and fix_suggestions:
            structured = parse_error(error_info)
            fix_instruction = f"""
## 上次执行错误
- 类型: {structured['error_type']}
- 行号: {structured['line']}
- 堆栈摘要: {structured['traceback'][:300]}

## 修改建议
{fix_suggestions}

请严格根据建议修正，只修改必要部分，输出完整代码。
"""
        elif error_info:
            structured = parse_error(error_info)
            fix_instruction = f"""
## 上次执行错误
- 类型: {structured['error_type']}
- 行号: {structured['line']}
- 堆栈摘要: {structured['traceback'][:300]}

请分析错误原因并修正，输出完整代码。
"""

        # 参数提示
        param_prompt = ""
        if parameters:
            param_prompt = f"\n## 额外参数\n{json.dumps(parameters, ensure_ascii=False)}"

        prompt = f"""{self.system_prompt}

{metadata_constraint}
{param_prompt}
{history_prompt}
{fix_instruction}

## 用户任务
{task}

## 代码要求
1. **必须通过以下方式获取数据集**：
   ```python
   from tools.tool2 import get_manager
   manager = get_manager()
   ds = manager.get("具体的键名")   # 使用上面列出的键名
   不要使用 xr.open_dataset() 直接读文件。

最终结果赋值给变量 result（可以是 Dataset、数值、字典等）。

包含必要的 import 语句和错误处理。

直接输出可执行的 Python 代码，用 python 包裹。
"""
        response = self.llm.invoke([SystemMessage(content=prompt)])
        content = response.content if hasattr(response, 'content') else str(response)
        code = extract_python_code(content)
        return code or content # 降级返回原始内容
    
class CodeCorrectionAgent:
    """分析错误，提供修改建议（不生成代码）"""
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def analyze_error(
        self,
        code: str,
        error_info: str,
        target_keys: List[str],
        metadata: Dict[str, Any],
        error_history: List[Dict],
        parameters: Optional[Dict] = None
        ) -> Dict[str, Any]:
        structured = parse_error(error_info)
        meta_desc = []
        for key in target_keys:
            meta = metadata.get(key, {})
            meta_desc.append(f"{key}: 变量={meta.get('variables', [])[:10]}")
        meta_text = "\n".join(meta_desc)

        history_text = ""
        if error_history:
            history_text = "\n## 历史错误\n"
            for i, err in enumerate(error_history[-3:], 1):
                history_text += f"{i}. {err.get('error_type')}: {err.get('full_error', '')[:150]}\n"
        
        prompt = f"""你是代码错误分析专家，只分析错误原因并提供修改建议，不要生成代码。

原始代码
python
{code}
错误信息
类型: {structured['error_type']}

行号: {structured['line']}

完整错误: {structured['full_error'][:500]}

数据集元数据
{meta_text}
{history_text}

输出格式
错误原因分析：
[1-3句话说明根本原因]

修改建议：
具体修改点（如“第X行：将变量名改为...”、“添加 .values 属性”等）

预期效果：
[修改后应达到的效果]
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, 'content') else str(response)
        analysis = ""
        suggestions = ""


        match = re.search(r'### 错误原因分析：\s\n(.?)(?=###|$)', content, re.DOTALL)
        if match:
            analysis = match.group(1).strip()
        match = re.search(r'### 修改建议：\s\n(.?)(?=###|$)', content, re.DOTALL)
        if match:
            suggestions = match.group(1).strip()

        return {
        "error_analysis": analysis or "无法分析错误原因",
        "fix_suggestions": suggestions or "请检查变量名和数据类型",
        "structured_error": structured
        }
    
class CodingAgent:
    """
    代码生成与执行智能体，符合 Process Agent 调用接口。
    """

    def __init__(
    self,
    llm: ChatOllama,
    max_attempts: int = 5,
    verbose: bool = True
    ):
        self.llm = llm
        self.max_attempts = max_attempts
        self.verbose = verbose
        self.generator = CodeGenerationAgent(llm)
        self.corrector = CodeCorrectionAgent(llm)
        self.generator.set_correction_agent(self.corrector)
        self.executor = InProcessCodeExecutor()

    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        主入口，符合 Process Agent 接口。
        """
        task = request.get("task", "")
        target_keys = request.get("target_keys", [])
        current_key = request.get("current_cache_key")
        metadata = request.get("metadata", {})
        parameters = request.get("parameters")
        file_paths_map = request.get("file_paths", {})

        if not target_keys:
            return {
                "status": "error",
                "message": "未指定目标数据集键 (target_keys)",
                "attempts": 0
            }

        error_history = []
        final_code = None

        for attempt in range(self.max_attempts):
            if self.verbose:
                print(f"\n[CodingAgent] 第 {attempt + 1} 次尝试")

            # 生成代码
            if attempt == 0:
                code = self.generator.generate_code(
                    task=task,
                    target_keys=target_keys,
                    metadata=metadata,
                    current_key=current_key,
                    parameters=parameters
                )
            else:
                last_err = error_history[-1]
                code = self.generator.generate_code(
                    task=task,
                    target_keys=target_keys,
                    metadata=metadata,
                    current_key=current_key,
                    parameters=parameters,
                    error_info=last_err.get('full_error', ''),
                    error_history=error_history,
                    fix_suggestions=last_err.get('fix_suggestions', '')
                )
            final_code = code
            if self.verbose:
                print(f"[CodingAgent] 生成代码:\n{code[:500]}...")

            # 执行代码
            exec_result = self.executor.execute(code,
                        file_paths=file_paths_map,current_key=request.get("current_cache_key", ""))

            if exec_result["success"]:
                result = exec_result["result"]
                import xarray as xr
                if isinstance(result, xr.Dataset):
                    manager = get_manager()
                    new_key = manager.add(result)
                    return {
                        "status": "success",
                        "new_cache_key": new_key,
                        "result_data": None,
                        "attempts": attempt + 1,
                        "code": final_code
                    }
                else:
                    return {
                        "status": "success",
                        "new_cache_key": None,
                        "result_data": result,
                        "attempts": attempt + 1,
                        "code": final_code
                    }

            # 执行失败，进行错误分析
            error_msg = exec_result["error"]
            if self.verbose:
                print(f"[CodingAgent] 执行失败: {error_msg[:200]}...")

            # ---- 保护性调用纠错分析 ----
            fix_suggestions = ""
            error_analysis = ""
            try:
                analysis = self.corrector.analyze_error(
                    code=code,
                    error_info=error_msg,
                    target_keys=target_keys,
                    metadata=metadata,
                    error_history=error_history,
                    parameters=parameters
                )
                fix_suggestions = analysis.get("fix_suggestions", "")
                error_analysis = analysis.get("error_analysis", "")
            except Exception as e:
                # 纠错分析本身出错时，降级处理，仍继续循环
                fix_suggestions = f"纠错分析内部异常: {str(e)}"
                error_analysis = ""
                if self.verbose:
                    print(f"[CodingAgent] 纠错分析异常: {e}")

            # 构建结构化错误记录
            structured_err = {
                "error_type": "execution_error",
                "line": None,
                "full_error": error_msg,
                "traceback": error_msg,   # 保留完整错误信息
                "fix_suggestions": fix_suggestions,
                "error_analysis": error_analysis,
            }
            error_history.append(structured_err)

        # 超过最大尝试次数
        return {
            "status": "error",
            "message": (
                f"超过最大尝试次数 ({self.max_attempts})，"
                f"最后错误: {error_history[-1].get('full_error', '未知')[:200] if error_history else '无'}"
            ),
            "attempts": self.max_attempts,
            "error_history": error_history,
            "code": final_code
        }

