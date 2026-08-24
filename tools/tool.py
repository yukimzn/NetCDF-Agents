# tool.py
import json
import numpy as np
import xarray as xr
from typing import List, Optional, Union, Literal
from langchain.tools import tool

# 导入全局管理器实例
from utils.DatasetManager import DatasetManager
from utils.TimeUnitsFixer import TimeUnitFixer
from utils.JsonProcess import safe_serialize


# 全局管理器单例（在 data_processing_agent 中初始化）
_MANAGER: Optional[DatasetManager] = None

def set_dataset_manager(manager: DatasetManager):
    """设置全局管理器（由 Agent 在开始时调用）"""
    global _MANAGER
    _MANAGER = manager

def get_manager() -> DatasetManager:
    """获取全局管理器，若未设置则抛出异常"""
    if _MANAGER is None:
        raise RuntimeError("DatasetManager 未初始化")
    return _MANAGER
import os

@tool
def load_dataset(file_path: str, auto_fix_time: bool = False) -> str:
    """
    从文件加载 NetCDF 数据集，存入管理器并返回数据集键名。
    """
    try:
        ds = None
        fix_report = {}
        load_errors = []
        
        # 尝试标准加载
        try:
            ds = xr.open_dataset(file_path, decode_times=False)
        except Exception as e:
            error_msg = str(e)
            if "unable to decode time units" in error_msg or "Failed to decode variable" in error_msg:
                print(f"⚠️ 遇到非标准时间格式，正在使用 decode_times=False 重新加载: {file_path}")
                ds = xr.open_dataset(file_path, decode_times=False)
            else:
                load_errors.append(f"标准加载失败: {str(e)}")
            try:
                ds = xr.open_dataset(file_path, decode_times=False)
                if auto_fix_time:
                    ds, fix_report = TimeUnitFixer.create_fixed_dataset(ds)
                    if fix_report.get('fixed_coordinates'):
                        try:
                            ds = xr.decode_cf(ds)
                            fix_report['decode_success'] = True
                        except Exception as decode_error:
                            fix_report['decode_success'] = False
                            fix_report['decode_error'] = str(decode_error)
            except Exception as e2:
                load_errors.append(f"关闭时间解码仍失败: {str(e2)}")
                raise e2
        
        if ds is None:
            raise Exception("所有加载方式均失败")
        
        # 修复：将fix_report转换为JSON字符串前先清理numpy类型
        def clean_for_json(obj):
            """递归清理对象中的numpy类型"""
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.datetime64):
                return str(obj)
            else:
                return obj
        
        # 清理fix_report
        clean_fix_report = clean_for_json(fix_report)
        ds.attrs['_load_fix_report'] = json.dumps(clean_fix_report, ensure_ascii=False)
        
        manager = get_manager()
        key = manager.add(ds, None,file_path)
        summary = manager.get_summary(key)
        
        # 清理summary
        clean_summary = clean_for_json(summary)
        
        return json.dumps({
            "status": "success", 
            "cache_key": key, 
            "summary": clean_summary
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "status": "error", 
            "message": f"加载失败: {str(e)}"
        }, ensure_ascii=False)

@tool
def save_dataset(dataset_key: str, output_path: str, compress: bool = False) -> str:
    """将管理器中的数据集保存为 NetCDF 文件（自动修复常见兼容性问题）
        输入参数名为：dataset_key: str, output_path: str
    """
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})

    try:
        # 创建副本，避免污染原始数据集
        ds_to_save = ds.copy()

        # ===== 兼容性修复：处理可能导致 Invalid argument 的变量 =====
        for var_name in ds_to_save.variables:
            var = ds_to_save[var_name]
            # 1. 将 int64 时间 / 坐标变量转换为 float64（经典格式不支持 int64）
            if var.dtype == 'int64':
                ds_to_save[var_name] = var.astype('float64')
                # 清除可能与类型冲突的属性
                for bad_attr in ['_FillValue', 'missing_value', 'add_offset', 'scale_factor']:
                    if bad_attr in ds_to_save[var_name].attrs:
                        del ds_to_save[var_name].attrs[bad_attr]

            # 2. 清除可能导致问题的非标准 calendar 编码（如 proleptic_gregorian）
            if 'calendar' in var.attrs:
                try:
                    import cftime
                    del ds_to_save[var_name].attrs['calendar']
                except Exception:
                    pass  # 如果删除失败，忽略

        # 3. 全局清除可能导致格式冲突的编码（保留用户指定的压缩编码）
        for var_name in ds_to_save.variables:
            # 重置编码，避免前序处理遗留的编码冲突
            ds_to_save[var_name].encoding = {}

        # ===== 设置压缩编码 =====
        encoding = {}
        if compress:
            for var in ds_to_save.data_vars:
                encoding[var] = {"zlib": True, "complevel": 4}
            # 坐标变量也可以压缩（可选，这里不强制）
            # for coord in ds_to_save.coords:
            #     encoding[coord] = {"zlib": True, "complevel": 4}
        else:
            encoding = None

        # 执行保存
        ds_to_save.to_netcdf(output_path, encoding=encoding)

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        return json.dumps({
            "status": "success",
            "output_path": output_path,
            "size_mb": round(size_mb, 2)
        })

    except Exception as e:
        return json.dumps({"status": "error", "message": f"保存失败: {str(e)}"})

# ============================================
# 2. 元数据查询工具
# ============================================
@tool
def list_variables(dataset_key: str) -> str:
    """列出指定数据集的变量名"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    return json.dumps({"status": "success", "variables": list(ds.variables.keys())})


@tool
def list_dimensions(dataset_key: str) -> str:
    """列出指定数据集的维度及大小"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    return json.dumps({"status": "success", "dimensions": {d: s for d, s in ds.sizes.items()}})

def _get_fingerprint(ds: xr.Dataset) -> dict:
    """内部辅助函数：获取数据指纹"""
    fingerprint = {
        "variables": list(ds.data_vars.keys()),
        "dims": dict(ds.sizes),
        "time_range": None,
        "spatial_range": {}
    }
    # 尝试自动寻找时间轴
    for v in ds.coords:
        if "time" in v.lower() and np.issubdtype(ds[v].dtype, np.datetime64):
            fingerprint["time_range"] = [str(ds[v].min().values), str(ds[v].max().values)]
        if "lat" in v.lower():
            fingerprint["spatial_range"]["lat"] = [float(ds[v].min()), float(ds[v].max())]
        if "lon" in v.lower():
            fingerprint["spatial_range"]["lon"] = [float(ds[v].min()), float(ds[v].max())]
    return fingerprint

def sanitize_dict(d: dict) -> dict:
    """
    递归清理字典，将所有无法 JSON 序列化的 NumPy 类型转换为 Python 标准类型。
    """
    new_dict = {}
    for k, v in d.items():
        # 处理嵌套字典
        if isinstance(v, dict):
            new_dict[k] = sanitize_dict(v)
        # 处理 NumPy 浮点数
        elif isinstance(v, (np.float32, np.float64)):
            new_dict[k] = float(v) if not np.isnan(v) else None
        # 处理 NumPy 整数
        elif isinstance(v, (np.int32, np.int64)):
            new_dict[k] = int(v)
        # 处理 列表 或 NumPy 数组
        elif isinstance(v, (list, np.ndarray)):
            new_dict[k] = [
                float(x) if isinstance(x, (np.float32, np.float64)) else x 
                for x in v
            ]
        # 处理 时间戳 或 其他对象
        elif isinstance(v, (np.datetime64, np.generic)):
            new_dict[k] = str(v)
        else:
            new_dict[k] = v
    return new_dict

@tool
def get_variable_info(dataset_key: str, var_name: Union[str, List[str]]) -> str:
    """
    获取一个或多个变量的详细信息：维度、形状、类型、属性和基本统计值（支持数值和时间类型）。
    
    参数:
        dataset_key: 数据集键
        var_name: 变量名（字符串）或变量名列表（如 ["SSS", "SST"]）
    """
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    
    # 统一转换为列表处理
    is_single = isinstance(var_name, str)
    var_names = [var_name] if is_single else var_name
    
    result_info = {}
    missing_vars = []
    
    for name in var_names:
        if name not in ds.variables:
            missing_vars.append(name)
            continue
        
        var = ds[name]
        
        # 1. 基础信息封装
        info = {
            "name": name,
            "dims": list(var.sizes),
            "shape": list(var.shape),
            "dtype": str(var.dtype),
            "attrs": sanitize_dict(dict(var.attrs)) # 使用安全序列化处理属性
        }
        
        # 2. 判断变量类型：数值 或 时间
        is_numeric = np.issubdtype(var.dtype, np.number)
        is_datetime = np.issubdtype(var.dtype, np.datetime64)
        
        if is_numeric or is_datetime:
            try:
                # 获取极值，处理时间类型时转为字符串
                v_min = var.min(skipna=True).values
                v_max = var.max(skipna=True).values
                
                if is_datetime:
                    info["min"] = str(v_min)
                    info["max"] = str(v_max)
                else:
                    info["min"] = float(v_min) if not np.isnan(v_min) else None
                    info["max"] = float(v_max) if not np.isnan(v_max) else None
                    # 数值类型额外增加平均值
                    v_mean = var.mean(skipna=True).values
                    info["mean"] = float(v_mean) if not np.isnan(v_mean) else None
            except Exception as e:
                # 记录错误但不中断，确保基础元数据能返回
                info["summary_error"] = f"统计信息获取失败: {str(e)}"
        
        result_info[name] = info
    
    # 3. 错误处理：如果有变量不存在
    if missing_vars:
        return json.dumps({
            "status": "error",
            "message": f"变量不存在: {missing_vars}",
            "available_vars": list(ds.variables.keys())
        })
    
    # 4. 根据输入类型返回不同结构
    if is_single:
        return json.dumps({"status": "success", "info": result_info[var_name]})
    else:
        return json.dumps({"status": "success", "info": result_info})
    

@tool
def get_global_attributes(dataset_key: str) -> str:
    """获取数据集的全局属性"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    safe_attrs=sanitize_dict(dict(ds.attrs))
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    return json.dumps({"status": "success", "global_attrs": safe_attrs})


# ============================================
# 3. 变量操作工具
# ============================================

from typing import Union, List
@tool
def extract_variable(dataset_key: str, var_name: Union[str, List[str]]) -> str:
    """
    提取单个或多个变量，生成新数据集并返回新键名
    
    参数:
        dataset_key: 源数据集键
        var_name: 变量名（字符串）或变量名列表（如 ["SSS", "SST"]）
    
    返回:
        JSON 字符串，格式:
        成功: {"status": "success", "new_cache_key": "新数据集键", "extracted_vars": ["var1", "var2"]}
        失败: {"status": "error", "message": "错误信息"}
    """
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    
    # 统一转换为列表处理
    is_single = isinstance(var_name, str)
    var_names = [var_name] if is_single else var_name
    
    # 检查缺失的变量
    missing_vars = [v for v in var_names if v not in ds.data_vars]
    if missing_vars:
        return json.dumps({
            "status": "error",
            "message": f"变量不存在: {missing_vars}",
            "available_vars": list(ds.data_vars.keys())
        })
    
    # 提取变量（支持单个或多个）
    # ds[var_names] 当 var_names 是列表时返回包含多个变量的子集
    new_ds = ds[var_names]
    new_key = manager.add(new_ds)
    
    return json.dumps({
        "status": "success",
        "new_cache_key": new_key,
        "extracted_vars": var_names,  # 返回实际提取的变量列表
        "is_single": is_single        # 可选字段，便于调试
    })


@tool
def rename_variable(dataset_key: str, old_name: str, new_name: str) -> str:
    """重命名数据集中的变量，返回新键名"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    if old_name not in ds.data_vars:
        return json.dumps({"status": "error", "message": f"变量 '{old_name}' 不存在"})
    new_ds = ds.rename_vars({old_name: new_name})
    new_key = manager.add(new_ds)
    return json.dumps({"status": "success", "new_cache_key": new_key})


@tool
def delete_variable(dataset_key: str, var_name: Union[str, List[str]]) -> str:
    """
    删除数据集中的一个或多个变量，返回新键名
    
    参数:
        dataset_key: 源数据集键
        var_name: 变量名（字符串）或变量名列表（如 ["SSS", "SST"]）
    
    返回:
        JSON 字符串，格式:
        成功: {"status": "success", "new_cache_key": "新数据集键", "deleted_vars": ["var1", "var2"]}
        失败: {"status": "error", "message": "错误信息"}
    """
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    
    # 统一转换为列表处理
    is_single = isinstance(var_name, str)
    var_names = [var_name] if is_single else var_name
    
    # 检查缺失的变量
    missing_vars = [v for v in var_names if v not in ds.data_vars]
    if missing_vars:
        return json.dumps({
            "status": "error",
            "message": f"变量不存在: {missing_vars}",
            "available_vars": list(ds.data_vars.keys())
        })
    
    # 删除变量（支持单个或多个）
    # drop_vars 方法接受字符串或字符串列表
    new_ds = ds.drop_vars(var_names)
    new_key = manager.add(new_ds)
    
    return json.dumps({
        "status": "success",
        "new_cache_key": new_key,
        "deleted_vars": var_names,  # 返回实际删除的变量列表
        "is_single": is_single      # 可选字段，便于调试
    })

# ============================================
# 4. 维度操作工具
# ============================================
@tool
def extract_level(dataset_key: str, dim_name: str, index: int) -> str:
    """提取指定维度的单个索引层，返回新键名"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    if dim_name not in ds.sizes:
        return json.dumps({"status": "error", "message": f"维度 '{dim_name}' 不存在"})
    
    # 索引越界检查
    dim_size = ds.sizes[dim_name]
    if index < 0 or index >= dim_size:
        return json.dumps({
            "status": "error", 
            "message": f"索引 {index} 超出维度 '{dim_name}' 的范围 [0, {dim_size - 1}]",
            "dim_name": dim_name,
            "dim_size": dim_size,
            "valid_range": [0, dim_size - 1]
        })
    
    new_ds = ds.isel({dim_name: index})
    new_key = manager.add(new_ds)
    return json.dumps({
        "status": "success", 
        "new_cache_key": new_key,
        "extracted_index": index,
        "dim_name": dim_name
    })


@tool
def reduce_dimension(
    dataset_key: str,
    dim_name: str,
    stride: int,
    start: Optional[int] = None,
    stop: Optional[int] = None
) -> str:
    """对指定维度进行间隔采样（下采样），返回新键名"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    if dim_name not in ds.dizes:
        return json.dumps({"status": "error", "message": f"维度 '{dim_name}' 不存在"})
    size = ds.sizes[dim_name]
    start = start or 0
    stop = stop or size
    indices = list(range(start, stop, stride))
    new_ds = ds.isel({dim_name: indices})
    new_key = manager.add(new_ds)
    return json.dumps({"status": "success", "new_cache_key": new_key})


# ============================================
# 5. 统计计算工具
# ============================================

@tool
def calculate_statistics(
    dataset_key: str,
    var_name: Union[str, List[str]],
    operations: Union[Literal["mean", "max", "min", "std", "sum", "median"], 
                     List[Literal["mean", "max", "min", "std", "sum", "median"]]],
    dim: Optional[Union[str, List[str]]] = None
) -> str:
    """
    计算一个或多个变量的统计值，支持单个或多个统计操作
    
    Args:
        dataset_key: 数据集键名
        var_name: 变量名（字符串）或变量名列表（如 ["SSS", "SST"]）
        operations: 单个统计操作或统计操作列表
        dim: 计算维度（可选），可为维度名字符串或列表
    
    Returns:
        JSON格式的结果字符串

    使用案例：
# 单变量单操作
calculate_statistics("ds_123", "SSS", "mean", dim="N_prof")
# 返回: {"status": "success", "var_name": "SSS", "operations": ["mean"], "results": 34.5}

# 单变量多操作
calculate_statistics("ds_123", "SSS", ["mean", "std", "min", "max"])
# 返回: {"status": "success", "var_name": "SSS", "results": {"mean": 34.5, "std": 0.2, ...}}

# 多变量多操作
calculate_statistics("ds_123", ["SSS", "SST", "MLD"], ["mean", "std"])
# 返回: {"status": "success", "variables": ["SSS", "SST", "MLD"], "results": {"SSS": {...}, "SST": {...}, "MLD": {...}}}
    """
    print("\n启动工具：calculate_statistics\n")
    manager = get_manager()
    ds = manager.get(dataset_key)
    
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    
    # 统一转换为列表处理
    is_single_var = isinstance(var_name, str)
    var_names = [var_name] if is_single_var else var_name
    
    # 统一转换 operations 为列表
    if isinstance(operations, str):
        operations = [operations]
    
    # 支持的操作映射
    ops_map = {
        "mean": lambda x, d: x.mean(dim=d, skipna=True),
        "max": lambda x, d: x.max(dim=d, skipna=True),
        "min": lambda x, d: x.min(dim=d, skipna=True),
        "std": lambda x, d: x.std(dim=d, skipna=True),
        "sum": lambda x, d: x.sum(dim=d, skipna=True),
        "median": lambda x, d: x.median(dim=d, skipna=True) if hasattr(x, 'median') else None
    }
    
    # 验证所有操作是否支持
    invalid_ops = [op for op in operations if op not in ops_map]
    if invalid_ops:
        return json.dumps({
            "status": "error", 
            "message": f"不支持的操作: {invalid_ops}，可用: {list(ops_map.keys())}"
        })
    
    # 检查缺失的变量
    missing_vars = [v for v in var_names if v not in ds.data_vars]
    if missing_vars:
        return json.dumps({
            "status": "error",
            "message": f"变量不存在: {missing_vars}",
            "available_vars": list(ds.data_vars.keys())
        })
    
    # 存储所有变量的统计结果
    all_results = {}
    
    for name in var_names:
        var = ds[name]
        
        # 检查数值类型
        if not np.issubdtype(var.dtype, np.number):
            all_results[name] = {"error": f"变量 '{name}' 不是数值类型"}
            continue
        
        results = {}
        try:
            for op in operations:
                calc_func = ops_map[op]
                result = calc_func(var, dim)
                
                if result is None:
                    results[op] = {"error": f"不支持 '{op}' 操作"}
                    continue
                
                # 根据结果大小决定存储方式
                if result.size == 1:
                    # 处理标量值
                    value = float(result.values) if hasattr(result, 'values') else float(result)
                    results[op] = value
                else:
                    # 多维结果返回形状和采样
                    results[op] = {
                        "shape": list(result.shape),
                        "dims": list(result.dims),
                        "sample": result.values.flatten()[:10].tolist() if result.size > 0 else [],
                        "note": "结果较大，仅返回形状和前十采样值"
                    }
        except Exception as e:
            results = {"error": f"计算失败: {str(e)}"}
        
        all_results[name] = results
    
    # 构建返回结果
    if is_single_var:
        # 单变量时扁平化返回，保持向后兼容
        single_result = all_results[var_name]
        # 如果所有操作都失败，返回错误
        if "error" in single_result and len(operations) == 1:
            return json.dumps({
                "status": "error",
                "var_name": var_name,
                "message": single_result["error"]
            })
        return json.dumps({
            "status": "success",
            "var_name": var_name,
            "operations": operations,
            "dim": dim,
            "results": single_result
        }, ensure_ascii=False)
    else:
        # 多变量时返回嵌套结构
        return json.dumps({
            "status": "success",
            "variables": var_names,
            "operations": operations,
            "dim": dim,
            "results": all_results
        }, ensure_ascii=False)

@tool
def spatial_statistics(
    dataset_key: str,
    var_name: str,
    operation: Literal["mean", "sum"],
    lat_range: Optional[str] = None,
    lon_range: Optional[str] = None
) -> str:
    """空间统计（区域平均或总和），可指定经纬度范围。返回统计值 JSON"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    if var_name not in ds.data_vars:
        return json.dumps({"status": "error", "message": f"变量 '{var_name}' 不存在"})
    var = ds[var_name]
    lat_coord = _find_coord(ds, ['lat'])
    lon_coord = _find_coord(ds, ['lon'])
    if lat_coord is None or lon_coord is None:
        return json.dumps({"status": "error", "message": "无法识别经纬度坐标"})
    if lat_range:
        lat_min, lat_max = map(float, lat_range.split(','))
        var = var.sel({lat_coord: slice(lat_min, lat_max)})
    if lon_range:
        lon_min, lon_max = map(float, lon_range.split(','))
        var = var.sel({lon_coord: slice(lon_min, lon_max)})
    if operation == "mean":
        weights = np.cos(np.deg2rad(var[lat_coord]))
        weights.name = "weights"
        weighted_mean = var.weighted(weights).mean(dim=[lat_coord, lon_coord])
        result = float(weighted_mean.values)
    elif operation == "sum":
        result = float(var.sum(dim=[lat_coord, lon_coord]).values)
    else:
        return json.dumps({"status": "error", "message": "operation 仅支持 mean 或 sum"})
    return json.dumps({"status": "success", "operation": operation, "result": result})


@tool
def time_statistics(
    dataset_key: str,
    var_name: str,
    operation: Literal["monmean", "seasmean", "yearmean"]
) -> str:
    """时间维度的聚合统计（月平均、季节平均、年平均），返回新键名"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    if var_name not in ds.data_vars:
        return json.dumps({"status": "error", "message": f"变量 '{var_name}' 不存在"})
    time_coord = _find_coord(ds, ['time', 't'])
    if time_coord is None:
        return json.dumps({"status": "error", "message": "未找到时间坐标"})
    freq_map = {
        "monmean": "1M",
        "seasmean": "QS-DEC",
        "yearmean": "1Y"
    }
    resampled = ds.resample({time_coord: freq_map[operation]}).mean()
    new_key = manager.add(resampled)
    return json.dumps({"status": "success", "new_cache_key": new_key})


# ============================================
# 6. 空间裁剪工具
# ============================================
@tool
def spatial_subset(
    dataset_key: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> str:
    """按经纬度范围裁剪数据，自动处理坐标顺序问题"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    
    lat_coord = _find_coord(ds, ['lat'])
    lon_coord = _find_coord(ds, ['lon'])
    if lat_coord is None or lon_coord is None:
        return json.dumps({"status": "error", "message": f"无法识别经纬度坐标"})
    
    # 获取实际坐标值
    lat_values = ds[lat_coord].values
    lon_values = ds[lon_coord].values
    
    # 添加容差：根据分辨率自动计算
    lat_res = abs(np.diff(lat_values).mean()) if len(lat_values) > 1 else 0.1
    lon_res = abs(np.diff(lon_values).mean()) if len(lon_values) > 1 else 0.1
    
    # 扩展边界以包含边界值
    lat_min_adj = min(lat_min, lat_max) - lat_res / 2
    lat_max_adj = max(lat_min, lat_max) + lat_res / 2
    lon_min_adj = min(lon_min, lon_max) - lon_res / 2
    lon_max_adj = max(lon_min, lon_max) + lon_res / 2
    
    # 使用条件索引（更可靠）
    lat_mask = (ds[lat_coord] >= lat_min_adj) & (ds[lat_coord] <= lat_max_adj)
    lon_mask = (ds[lon_coord] >= lon_min_adj) & (ds[lon_coord] <= lon_max_adj)
    subset = ds.where(lat_mask & lon_mask, drop=True)
    
    # 验证裁剪结果
    if subset.sizes.get(lat_coord, 0) == 0 or subset.sizes.get(lon_coord, 0) == 0:
        return json.dumps({
            "status": "error",
            "message": f"裁剪后数据集为空。原始纬度范围: [{lat_values.min():.2f}, {lat_values.max():.2f}], "
                      f"请求纬度范围: [{lat_min}, {lat_max}]; "
                      f"原始经度范围: [{lon_values.min():.2f}, {lon_values.max():.2f}], "
                      f"请求经度范围: [{lon_min}, {lon_max}]"
        })
    
    new_key = manager.add(subset)
    return json.dumps({
        "status": "success",
        "new_cache_key": new_key,
        "subset_shape": {lat_coord: subset.sizes[lat_coord], lon_coord: subset.sizes[lon_coord]}
    })

def _find_coord(ds: xr.Dataset, patterns: List[str], prefer_standard_name: bool = True) -> Optional[str]:
    """
    在数据集的坐标中查找匹配指定模式的坐标名称。
    
    Args:
        ds: xarray Dataset 对象
        patterns: 要匹配的模式列表（如 ['lat'] 或 ['lon', 'longitude']）
        prefer_standard_name: 是否优先返回包含 standard_name 属性的坐标
    
    Returns:
        第一个匹配的坐标名称，若未找到则返回 None
    """
    # 优先查找具有标准名称的坐标
    if prefer_standard_name:
        standard_names = {'lat': 'latitude', 'lon': 'longitude'}
        for coord_name in ds.coords:
            attrs = ds[coord_name].attrs
            if 'standard_name' in attrs:
                for pattern in patterns:
                    if attrs['standard_name'] == standard_names.get(pattern.lower(), pattern.lower()):
                        return coord_name
    
    # 按名称模式匹配
    for coord_name in ds.coords:
        coord_lower = coord_name.lower()
        for pattern in patterns:
            if pattern.lower() in coord_lower:
                return coord_name
    return None


@tool
def mask_by_region(dataset_key: str, shapefile_path: str) -> str:
    """使用 shapefile 对数据进行掩膜（仅保留多边形内区域），返回新键名"""
    try:
        import geopandas as gpd
        import regionmask
    except ImportError:
        return json.dumps({"status": "error", "message": "缺少依赖 geopandas 或 regionmask"})
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    try:
        gdf = gpd.read_file(shapefile_path)
        poly = gdf.geometry[0]
        lat_coord = _find_coord(ds, ['lat'])
        lon_coord = _find_coord(ds, ['lon'])
        if lat_coord is None or lon_coord is None:
            return json.dumps({"status": "error", "message": "无法识别经纬度坐标"})
        mask = regionmask.RegionMask(poly, lon_name=lon_coord, lat_name=lat_coord)
        masked = mask.mask(ds)
        new_key = manager.add(masked)
        return json.dumps({"status": "success", "new_cache_key": new_key})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"掩膜失败: {str(e)}"})


# ============================================
# 7. 时间裁剪工具
# ============================================
@tool
def time_subset(dataset_key: str, start_time: str, end_time: str) -> str:
    """按时间范围裁剪数据，返回新键名"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    time_coord = _find_coord(ds, ['time', 't'])
    if time_coord is None:
        return json.dumps({"status": "error", "message": f"未找到时间坐标，现有坐标: {list(ds.coords.keys())}"})
    subset = ds.sel({time_coord: slice(start_time, end_time)})
    new_key = manager.add(subset)
    return json.dumps({"status": "success", "new_cache_key": new_key})


@tool
def time_slice(dataset_key: str, start_index: int, end_index: int, stride: int = 1) -> str:
    """按时间索引切片（类似 Python 切片），返回新键名"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    time_coord = _find_coord(ds, ['time', 't'])
    if time_coord is None:
        return json.dumps({"status": "error", "message": "未找到时间坐标"})
    indices = list(range(start_index, end_index, stride))
    subset = ds.isel({time_coord: indices})
    new_key = manager.add(subset)
    return json.dumps({"status": "success", "new_cache_key": new_key})


# ============================================
# 8. 重网格化工具
# ============================================
@tool
def regrid(
    dataset_key: str,
    method: Literal["bilinear", "conservative"] = "bilinear",
    lat_res: Optional[float] = None,
    lon_res: Optional[float] = None,
    target_dataset_key: Optional[str] = None
) -> str:
    """将数据插值到目标网格，返回新键名"""
    try:
        import xesmf as xe
    except ImportError:
        return json.dumps({"status": "error", "message": "请安装 xesmf"})
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    try:
        if target_dataset_key:
            target_ds = manager.get(target_dataset_key)
            if target_ds is None:
                return json.dumps({"status": "error", "message": f"目标数据集 '{target_dataset_key}' 不存在"})
        elif lat_res and lon_res:
            lat = np.arange(-90, 90 + lat_res, lat_res)
            lon = np.arange(-180, 180 + lon_res, lon_res)
            target_ds = xr.Dataset({"lat": lat, "lon": lon})
        else:
            return json.dumps({"status": "error", "message": "必须提供 target_dataset_key 或 (lat_res, lon_res)"})
        regridder = xe.Regridder(ds, target_ds, method, periodic=True)
        regridded = regridder(ds)
        new_key = manager.add(regridded)
        return json.dumps({"status": "success", "new_cache_key": new_key})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"重网格失败: {str(e)}"})


@tool
def regrid_unstructured_to_structured(
    dataset_key: str,
    lat_res: float,
    lon_res: float,
    method: Literal["bilinear", "nearest"] = "bilinear"
) -> str:
    """将非结构网格插值到规则经纬度网格，返回新键名"""
    try:
        import xesmf as xe
    except ImportError:
        return json.dumps({"status": "error", "message": "请安装 xesmf"})
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    try:
        lat = np.arange(-90, 90 + lat_res, lat_res)
        lon = np.arange(-180, 180 + lon_res, lon_res)
        target_grid = xr.Dataset({"lat": lat, "lon": lon})
        regridder = xe.Regridder(ds, target_grid, method, periodic=True, unmapped_to_nan=True)
        regridded = regridder(ds)
        new_key = manager.add(regridded)
        return json.dumps({"status": "success", "new_cache_key": new_key})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"重网格失败: {str(e)}"})


# ============================================
# 9. 文件合并/拆分工具
# ============================================
@tool
def merge_files(file_paths: List[str], merge_dim: Literal["time", "var"] = "time") -> str:
    """将多个 NetCDF 文件合并为一个数据集，返回新键名"""
    manager = get_manager()
    try:
        datasets = [xr.open_dataset(p) for p in file_paths]
        if merge_dim == "time":
            merged = xr.concat(datasets, dim="time")
        elif merge_dim == "var":
            merged = xr.merge(datasets)
        else:
            return json.dumps({"status": "error", "message": "merge_dim 仅支持 'time' 或 'var'"})
        for ds in datasets:
            ds.close()
        key = manager.add(merged)
        return json.dumps({"status": "success", "cache_key": key})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"合并失败: {str(e)}"})


@tool
def split_by_time(
    dataset_key: str,
    freq: Literal["day", "month", "year"],
    output_dir: Optional[str] = None
) -> str:
    """按时间频率拆分数据集。若提供 output_dir 则保存文件并返回路径列表；否则返回多个新键名列表"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    time_coord = _find_coord(ds, ['time', 't'])
    if time_coord is None:
        return json.dumps({"status": "error", "message": "未找到时间坐标"})
    freq_map = {"day": "1D", "month": "1M", "year": "1Y"}
    if freq not in freq_map:
        return json.dumps({"status": "error", "message": "freq 仅支持 day, month, year"})
    resampled_groups = ds.resample({time_coord: freq_map[freq]})
    keys = []
    for label, sub_ds in resampled_groups:
        if output_dir:
            import os
            path = os.path.join(output_dir, f"{str(label)}.nc")
            sub_ds.to_netcdf(path)
            keys.append(path)
        else:
            sub_key = manager.add(sub_ds)
            keys.append(sub_key)
    return json.dumps({"status": "success", "splits": keys, "count": len(keys), "saved_to_files": output_dir is not None})


@tool
def subset_by_index(
    dataset_key: str,
    dim_name: str,
    start: int,
    end: int,
    stride: int = 1
) -> str:
    """按维度索引提取子集（类似 Python 切片），返回新键名"""
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    if dim_name not in ds.sizes:
        return json.dumps({"status": "error", "message": f"维度 '{dim_name}' 不存在"})
    indices = list(range(start, end, stride))
    new_ds = ds.isel({dim_name: indices})
    new_key = manager.add(new_ds)
    return json.dumps({"status": "success", "new_cache_key": new_key})


# ============================================
# 10. 多数据集对比工具（示例）
# ============================================
@tool
def compare_datasets(dataset_keys: List[str], var_name: str) -> str:
    """对比多个数据集中同一变量的均值差异，返回对比结果 JSON"""
    manager = get_manager()
    means = {}
    for key in dataset_keys:
        ds = manager.get(key)
        if ds is None:
            return json.dumps({"status": "error", "message": f"数据集 '{key}' 不存在"})
        if var_name not in ds.data_vars:
            return json.dumps({"status": "error", "message": f"变量 '{var_name}' 在数据集 '{key}' 中不存在"})
        var = ds[var_name]
        if np.issubdtype(var.dtype, np.number):
            means[key] = float(var.mean().values)
        else:
            means[key] = None
    return json.dumps({"status": "success", "means": means})

@tool
def release_dataset(dataset_key: str) -> str:
    """
    从内存中释放（删除）指定的数据集缓存，释放内存资源。
    
    Args:
        dataset_key: 要释放的数据集键名
    
    Returns:
        操作状态 JSON
    """
    manager = get_manager()
    if dataset_key not in manager.list_keys():
        return json.dumps({
            "status": "error",
            "message": f"数据集 '{dataset_key}' 不存在"
        })
    try:
        manager.remove(dataset_key)
        return json.dumps({
            "status": "success",
            "message": f"数据集 '{dataset_key}' 已释放"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"释放失败: {str(e)}"
        })
    
@tool
def get_dataset_summary(dataset_key: Optional[str] = None) -> str:
    """
    获取数据集摘要信息。若不指定 dataset_key，则返回所有已加载数据集的键及简要统计；
    若指定，则返回该数据集的详细摘要（变量列表、维度、坐标等）。
    
    Args:
        dataset_key: 数据集键名（可选）
    
    Returns:
        JSON 格式的摘要信息
    """
    manager = get_manager()
    if dataset_key is None:
        # 返回所有键的概览
        keys = manager.list_keys()
        overview = {}
        for key in keys:
            ds = manager.get(key)
            overview[key] = {
                "n_variables": len(ds.data_vars),
                "n_dims": len(ds.dims),
                "dims": {d: s for d, s in ds.sizes.items()},
                "coord_names": list(ds.coords.keys())
            }
        return json.dumps({"status": "success", "datasets": overview})
    else:
        # 返回指定数据集的详细摘要
        try:
            summary = manager.get_summary(dataset_key)
            return json.dumps({"status": "success", "summary": summary}, ensure_ascii=False)
        except KeyError:
            return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})

from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

@tool
def unstructured_to_structured_grid(
    dataset_key: str,
    var_name: str,
    target_lat_min: float,
    target_lat_max: float,
    target_lon_min: float,
    target_lon_max: float,
    target_lat_res: float = 0.25,
    target_lon_res: float = 0.25,
    fill_value: float = np.nan,
    extrapolate: bool = False,
    extrapolate_margin_deg: float = 0.1
) -> str:
    """
    将非结构网格数据（散点数据）插值到规则经纬度结构网格
    
    适用于：Argo浮标数据、卫星L2级散点数据、非结构模型网格等
    
    Args:
        dataset_key: 源数据集键名（包含经纬度坐标和变量值）
        var_name: 需要插值的变量名
        target_lat_min: 目标网格最小纬度（度）
        target_lat_max: 目标网格最大纬度（度）
        target_lon_min: 目标网格最小经度（度）
        target_lon_max: 目标网格最大经度（度）
        target_lat_res: 目标网格纬度分辨率（度），默认0.25度
        target_lon_res: 目标网格经度分辨率（度），默认0.25度
        fill_value: 外推填充值，默认NaN（不填充）
        extrapolate: 是否允许外推（基于Delaunay三角剖分的线性外推），默认False
        extrapolate_margin_deg: 外推边界扩展范围（度），默认0.1度
    
    Returns:
        JSON格式字符串，包含：
        成功时: {"status": "success", "new_cache_key": "xxx", "grid_shape": [nlat, nlon], 
                 "lat_range": [min, max], "lon_range": [min, max], "valid_points": N}
        失败时: {"status": "error", "message": "错误信息"}
    """
    print("\n启动工具：unstructured_to_structured_grid")
    
    from tools.tool2 import get_manager
    manager = get_manager()
    ds = manager.get(dataset_key)
    
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    
    if var_name not in ds.data_vars:
        return json.dumps({
            "status": "error", 
            "message": f"变量 '{var_name}' 不存在",
            "available_vars": list(ds.data_vars.keys())
        })
    
    # 获取经纬度坐标
    lat_coord = _find_coord(ds, ["lat", "latitude", "nav_lat"])
    lon_coord = _find_coord(ds, ["lon", "longitude", "nav_lon"])
    if lat_coord is None or lon_coord is None:
        return json.dumps({
            "status": "error",
            "message": "无法识别经纬度坐标变量，请确保数据包含 lat/lon 或 latitude/longitude 变量"
        })
    
    # 提取坐标和数值
    try:
        # 获取经纬度值
        lat_values = ds[lat_coord].values.flatten()
        lon_values = ds[lon_coord].values.flatten()
        var_values = ds[var_name].values.flatten()
        
        # 移除 NaN 和无效值
        valid_mask = ~(np.isnan(lat_values) | np.isnan(lon_values) | np.isnan(var_values))
        
        # 检查数据量
        if valid_mask.sum() < 4:
            return json.dumps({
                "status": "error",
                "message": f"有效数据点不足（{valid_mask.sum()}个），至少需要4个点才能进行三角剖分"
            })
        
        lat_valid = lat_values[valid_mask]
        lon_valid = lon_values[valid_mask]
        var_valid = var_values[valid_mask]
        
        # 处理经度范围归一化（转换到 -180~180）
        lon_valid_normalized = lon_valid.copy()
        lon_valid_normalized[lon_valid_normalized > 180] -= 360
        
        target_lon_min_norm = target_lon_min
        target_lon_max_norm = target_lon_max
        if target_lon_min_norm > 180:
            target_lon_min_norm -= 360
            target_lon_max_norm -= 360
        
        print(f"  有效数据点: {len(lat_valid)}个")
        print(f"  经度范围: [{lon_valid_normalized.min():.2f}, {lon_valid_normalized.max():.2f}]")
        
        # 创建插值器
        points = np.column_stack([lon_valid_normalized, lat_valid])
        
        # 检查是否有重复点
        _, unique_indices = np.unique(points, axis=0, return_index=True)
        if len(unique_indices) < 4:
            return json.dumps({
                "status": "error",
                "message": f"去重后有效数据点不足（{len(unique_indices)}个）"
            })
        
        if len(unique_indices) < len(points):
            print(f"  警告: 发现 {len(points) - len(unique_indices)} 个重复点，已去重")
            points = points[unique_indices]
            var_valid = var_valid[unique_indices]
        
        # 创建插值器
        try:
            interpolator = LinearNDInterpolator(points, var_valid, fill_value=np.nan if not extrapolate else None)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"创建插值器失败: {str(e)}"
            })
        
        # 构建目标结构网格
        nlat = max(1, int(abs(target_lat_max - target_lat_min) / target_lat_res) + 1)
        nlon = max(1, int(abs(target_lon_max - target_lon_min) / target_lon_res) + 1)
        
        lat_grid = np.linspace(target_lat_min, target_lat_max, nlat)
        lon_grid = np.linspace(target_lon_min, target_lon_max, nlon)
        
        # 网格化经度范围（如果目标网格在 0-360 范围，保持原样）
        lon_grid_for_interp = lon_grid.copy()
        if target_lon_min_norm < -180 or target_lon_max_norm > 180:
            # 跨180度经线情况，特殊处理
            lon_grid_for_interp[lon_grid_for_interp > 180] -= 360
        
        # 创建网格点并执行插值
        lon_mesh, lat_mesh = np.meshgrid(lon_grid_for_interp, lat_grid)
        grid_points = np.column_stack([lon_mesh.flatten(), lat_mesh.flatten()])
        
        try:
            result_flat = interpolator(grid_points)
            result_grid = result_flat.reshape(nlat, nlon)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"插值计算失败: {str(e)}"
            })
        
        # 处理外推区域
        if not extrapolate:
            # 创建凸包掩膜，标记有效区域
            try:
                tri = Delaunay(points)
                hull_mask = tri.find_simplex(grid_points) >= 0
                result_grid[~hull_mask.reshape(nlat, nlon)] = fill_value
            except Exception as e:
                print(f"  凸包掩膜创建失败: {e}，将使用插值器自带的 fill_value")
        else:
            # 使用线性外推（LinearNDInterpolator 在外推时填 None/NaN 时不会外推）
            # 需要手动扩展边界点进行外推
            if extrapolate_margin_deg > 0:
                # 在原始点集周围添加缓冲区点（简单外推策略）
                margin_lat_min = lat_valid.min() - extrapolate_margin_deg
                margin_lat_max = lat_valid.max() + extrapolate_margin_deg
                margin_lon_min = lon_valid_normalized.min() - extrapolate_margin_deg
                margin_lon_max = lon_valid_normalized.max() + extrapolate_margin_deg
                
                # 对超出原始范围且未被插值的点进行最近邻填充
                # 这里简化：不做额外处理，直接使用 fill_value
                pass
        
        # 检查结果是否全为 NaN
        valid_result_count = np.sum(~np.isnan(result_grid))
        if valid_result_count == 0:
            return json.dumps({
                "status": "error",
                "message": "插值结果全部无效。请检查目标经纬度范围是否与原始数据范围重叠",
                "original_lat_range": [float(lat_valid.min()), float(lat_valid.max())],
                "original_lon_range": [float(lon_valid_normalized.min()), float(lon_valid_normalized.max())],
                "target_lat_range": [target_lat_min, target_lat_max],
                "target_lon_range": [target_lon_min, target_lon_max]
            })
        
        # 创建结构化数据集
        import xarray as xr
        
        result_da = xr.DataArray(
            result_grid,
            dims=["lat", "lon"],
            coords={
                "lat": lat_grid,
                "lon": lon_grid
            },
            name=var_name,
            attrs={
                "long_name": f"Gridded {var_name} from unstructured data",
                "gridding_method": "LinearNDInterpolator",
                "original_dataset_key": dataset_key,
                "original_points_count": len(lat_valid),
                "valid_grid_points": int(valid_result_count),
                "fill_value": str(fill_value),
                "extrapolate": extrapolate
            }
        )
        
        # 创建新数据集
        new_ds = xr.Dataset({var_name: result_da})
        new_key = manager.add(new_ds)
        
        # 计算结果统计
        if not np.isnan(fill_value):
            valid_range_mask = ~np.isnan(result_grid)
            result_mean = np.nanmean(result_grid)
            result_std = np.nanstd(result_grid)
        else:
            result_mean = np.nanmean(result_grid)
            result_std = np.nanstd(result_grid)
        
        return json.dumps({
            "status": "success",
            "new_cache_key": new_key,
            "var_name": var_name,
            "grid_shape": [nlat, nlon],
            "lat_range": [float(lat_grid.min()), float(lat_grid.max())],
            "lon_range": [float(lon_grid.min()), float(lon_grid.max())],
            "valid_points": int(valid_result_count),
            "total_points": nlat * nlon,
            "coverage_percentage": round(100 * valid_result_count / (nlat * nlon), 2),
            "result_mean": float(result_mean) if not np.isnan(result_mean) else None,
            "result_std": float(result_std) if not np.isnan(result_std) else None,
            "original_points_used": len(lat_valid)
        }, ensure_ascii=False)
        
    except Exception as e:
        import traceback
        return json.dumps({
            "status": "error",
            "message": f"处理失败: {str(e)}",
            "traceback": traceback.format_exc()[:500] if traceback else None
        })

import json
import numpy as np
from typing import Union, Dict, Any
@tool
def apply_math_operation(
    dataset_key: str,
    var_name: Union[str, List[str]],
    operation: Literal[
        "add", "subtract", "multiply", "divide", "power", 
        "negate", "abs", "copy", "linear",
        "kelvin_to_celsius", "celsius_to_kelvin", 
        "celsius_to_fahrenheit", "fahrenheit_to_celsius", 
        "kelvin_to_fahrenheit"
    ],
    operand: Optional[Union[float, int, str, List[Union[float, int]]]] = None,
    new_var_name: Optional[Union[str, List[str]]] = None
) -> str:
    """
    对数据集中的一个或多个变量应用数学运算，生成新变量并返回新数据集键名。
    
    支持的运算类型 (operation):
    - 'add':        var + operand
    - 'subtract':   var - operand
    - 'multiply':   var * operand
    - 'divide':     var / operand
    - 'power':      var ** operand
    - 'negate':     -var (无需 operand)
    - 'abs':        |var| (无需 operand)
    - 'copy':       复制变量 (无需 operand, 仅重命名)
    - 'linear':     线性变换: var * a + b, operand 为 [a, b] 或 "a,b"
    - 'kelvin_to_celsius':       K → °C (var - 273.15)
    - 'celsius_to_kelvin':       °C → K (var + 273.15)
    - 'celsius_to_fahrenheit':   °C → °F (var * 9/5 + 32)
    - 'fahrenheit_to_celsius':   °F → °C ((var - 32) * 5/9)
    - 'kelvin_to_fahrenheit':    K → °F ((var - 273.15) * 9/5 + 32)
    
    参数:
    - dataset_key: str, 数据集键名
    - var_name: str 或 List[str], 单个变量名或多个变量名列表
    - operation: str, 运算类型（所有变量使用相同的运算）
    - operand: float/int/list/str, 运算数。若为列表且 var_name 为列表：
               长度需与 var_name 一致（每个变量使用不同的操作数）
               长度=1 则所有变量共用同一操作数
    - new_var_name: str 或 List[str], 新变量名。若为列表需与 var_name 长度一致。
                    默认: 原变量名 + "_" + operation缩写
    
    返回: JSON 字符串
    {
        "status": "success",
        "new_cache_key": "...",
        "results": [
            {"source_var": "...", "new_var_name": "...", "operation": "...", "statistics": {...}},
            ...
        ]
    }
    """
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})
    
    # ========== 统一转为列表处理 ==========
    if isinstance(var_name, str):
        var_name_list = [var_name]
    else:
        var_name_list = var_name
    
    # 检查所有变量是否存在
    unavailable = []
    available_vars = list(ds.data_vars.keys())
    for v in var_name_list:
        if v not in ds.variables:
            unavailable.append(v)
    if unavailable:
        return json.dumps({
            "status": "error",
            "message": f"以下变量不存在: {unavailable}",
            "available_vars": available_vars
        })
    
    # ========== 处理 operand ==========
    # 根据运算类型决定是否需要 operand
    no_operand_ops = {"negate", "abs", "copy", 
                      "kelvin_to_celsius", "celsius_to_kelvin",
                      "celsius_to_fahrenheit", "fahrenheit_to_celsius", 
                      "kelvin_to_fahrenheit"}
    
    if operation not in no_operand_ops and operand is None:
        return json.dumps({"status": "error", "message": f"'{operation}' 运算需要提供 operand"})
    
    # 构建每个变量的 operand 列表
    if operation in no_operand_ops:
        operand_list = [None] * len(var_name_list)
    elif isinstance(operand, (list, tuple)):
        if len(operand) == 1:
            operand_list = [operand[0]] * len(var_name_list)
        elif len(operand) == len(var_name_list):
            operand_list = list(operand)
        else:
            return json.dumps({
                "status": "error",
                "message": f"operand 列表长度({len(operand)})与 var_name 列表长度({len(var_name_list)})不匹配"
            })
    else:
        # 单个值，所有变量共用
        operand_list = [operand] * len(var_name_list)
    
    # ========== 处理 new_var_name ==========
    if new_var_name is None:
        new_var_name_list = [None] * len(var_name_list)
    elif isinstance(new_var_name, str):
        if len(var_name_list) == 1:
            new_var_name_list = [new_var_name]
        else:
            return json.dumps({
                "status": "error",
                "message": "多个变量时，new_var_name 必须为与 var_name 长度相同的列表"
            })
    elif isinstance(new_var_name, list):
        if len(new_var_name) == len(var_name_list):
            new_var_name_list = new_var_name
        else:
            return json.dumps({
                "status": "error",
                "message": f"new_var_name 列表长度({len(new_var_name)})与 var_name 列表长度({len(var_name_list)})不匹配"
            })
    else:
        new_var_name_list = [None] * len(var_name_list)
    
    # ========== 定义温度单位映射 ==========
    unit_map = {
        'kelvin_to_celsius': '°C',
        'celsius_to_kelvin': 'K',
        'celsius_to_fahrenheit': '°F',
        'fahrenheit_to_celsius': '°C',
        'kelvin_to_fahrenheit': '°F'
    }
    
    try:
        # 创建新数据集（只复制一次）
        new_ds = ds.copy()
        results = []
        
        # ========== 对每个变量执行运算 ==========
        for i, v in enumerate(var_name_list):
            data = ds[v].values.astype('float64')
            op_val = operand_list[i]
            result = None
            op_desc = ""
            
            # --- 执行运算 ---
            if operation == 'add':
                result = data + float(op_val)
                op_desc = f"{v} + {op_val}"
            elif operation == 'subtract':
                result = data - float(op_val)
                op_desc = f"{v} - {op_val}"
            elif operation == 'multiply':
                result = data * float(op_val)
                op_desc = f"{v} * {op_val}"
            elif operation == 'divide':
                divisor = float(op_val)
                if divisor == 0:
                    return json.dumps({"status": "error", "message": f"变量 '{v}' 的除数为 0"})
                result = data / divisor
                op_desc = f"{v} / {op_val}"
            elif operation == 'power':
                result = data ** float(op_val)
                op_desc = f"{v} ** {op_val}"
            elif operation == 'negate':
                result = -data
                op_desc = f"-{v}"
            elif operation == 'abs':
                result = np.abs(data)
                op_desc = f"|{v}|"
            elif operation == 'copy':
                result = data.copy()
                op_desc = f"copy of {v}"
            elif operation == 'linear':
                if isinstance(op_val, str):
                    a, b = map(float, op_val.split(','))
                elif isinstance(op_val, (list, tuple)):
                    a, b = float(op_val[0]), float(op_val[1])
                else:
                    return json.dumps({"status": "error", "message": f"变量 '{v}' 的 linear operand 格式错误"})
                result = data * a + b
                op_desc = f"{v} * {a} + {b}"
            elif operation == 'kelvin_to_celsius':
                result = data - 273.15
                op_desc = f"{v} (K → °C)"
            elif operation == 'celsius_to_kelvin':
                result = data + 273.15
                op_desc = f"{v} (°C → K)"
            elif operation == 'celsius_to_fahrenheit':
                result = data * 9.0/5.0 + 32
                op_desc = f"{v} (°C → °F)"
            elif operation == 'fahrenheit_to_celsius':
                result = (data - 32) * 5.0/9.0
                op_desc = f"{v} (°F → °C)"
            elif operation == 'kelvin_to_fahrenheit':
                result = (data - 273.15) * 9.0/5.0 + 32
                op_desc = f"{v} (K → °F)"
            
            # --- 生成新变量名 ---
            if new_var_name_list[i] is None:
                # 自动生成: var_operation 或 var_op_operand
                op_abbr = operation.replace("kelvin_to", "K2").replace("celsius_to", "C2").replace("fahrenheit_to", "F2")
                new_name = f"{v}_{op_abbr}"
            else:
                new_name = new_var_name_list[i]
            
            # --- 添加新变量到数据集 ---
            new_ds[new_name] = (ds[v].dims, result)
            new_ds[new_name].attrs = ds[v].attrs.copy()
            new_ds[new_name].attrs['history'] = f"Applied operation: {op_desc}"
            new_ds[new_name].attrs['long_name'] = f"{ds[v].attrs.get('long_name', v)} ({op_desc})"
            
            if operation in unit_map:
                new_ds[new_name].attrs['units'] = unit_map[operation]
            
            # --- 统计信息 ---
            stats = {
                "min": float(np.nanmin(result)),
                "max": float(np.nanmax(result)),
                "mean": float(np.nanmean(result)),
                "std": float(np.nanstd(result))
            }
            
            results.append({
                "source_var": v,
                "new_var_name": new_name,
                "operation": op_desc,
                "shape": list(result.shape),
                "dtype": str(result.dtype),
                "statistics": stats
            })
        
        # ========== 存入管理器 ==========
        new_key = manager.add(new_ds)
        
        return json.dumps({
            "status": "success",
            "new_cache_key": new_key,
            "total_vars_processed": len(results),
            "results": results
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": f"数学运算失败: {str(e)}"})

@tool
def inspect_variable_values(dataset_key: str, var_name: str, num_samples: int = 5) -> str:
    """
    查看变量的具体数值样本（前N个值、最后N个值及统计极值）。
    用于 Agent 无法通过元数据判断内容时，直接观察数据。
    """
    try:
        manager = get_manager()
        ds = manager.get(dataset_key)
        if var_name not in ds:
            return json.dumps({"status": "error", "message": f"变量 {var_name} 不存在"})
        
        data = ds[var_name]
        # 处理时间类型或普通类型
        values = data.values.flatten()
        total_size = values.size
        
        # 采样逻辑
        sample_indices = np.unique(np.linspace(0, total_size - 1, min(num_samples, total_size)).astype(int))
        samples = [str(v) for v in values[sample_indices]]
        
        res = {
            "status": "success",
            "var_name": var_name,
            "total_elements": int(total_size),
            "samples": samples,
            "dtype": str(data.dtype)
        }
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@tool
def get_time_info(dataset_key: str, var_name: str = "time") -> str:
    """
    专门获取时间变量的详细信息：起始日期、结束日期、频率及时间单位。
    """
    try:
        manager = get_manager()
        ds = manager.get(dataset_key)
        if var_name not in ds:
            return json.dumps({"status": "error", "message": f"未找到时间变量: {var_name}"})
        
        time_var = ds[var_name]
        if not np.issubdtype(time_var.dtype, np.datetime64):
            return json.dumps({"status": "error", "message": f"变量 {var_name} 不是标准时间类型"})
        
        t_min = time_var.min().values
        t_max = time_var.max().values
        
        return json.dumps({
            "status": "success",
            "start_time": str(t_min),
            "end_time": str(t_max),
            "time_steps": len(time_var),
            "units": time_var.attrs.get("units", "unknown"),
            "calendar": time_var.attrs.get("calendar", "standard")
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@tool
def search_variables_by_attribute(dataset_key: str, keyword: str) -> str:
    """
    通过关键字搜索变量名、长名称(long_name)或标准名称(standard_name)。
    """
    manager = get_manager()
    ds = manager.get(dataset_key)
    found = []
    for var in ds.data_vars:
        attr_str = str(ds[var].attrs).lower()
        if keyword.lower() in var.lower() or keyword.lower() in attr_str:
            found.append(var)
    return json.dumps({"status": "success", "keyword": keyword, "matched_variables": found})

@tool
def get_coordinate_info(dataset_key: str) -> str:
    """
    获取数据集的坐标轴范围（经度、纬度、高度等）及分辨率。
    """
    manager = get_manager()
    ds = manager.get(dataset_key)
    coords_info = {}
    for dim in ds.dims:
        if dim in ds.coords:
            c = ds.coords[dim]
            coords_info[dim] = {
                "range": [float(c.min()), float(c.max())],
                "size": len(c),
                "units": c.attrs.get("units", "")
            }
    return json.dumps({"status": "success", "coordinates": coords_info})


@tool
def filter_by_condition(
    dataset_key: str,
    conditions: str,
    return_mask: bool = False,
    new_var_name: Optional[str] = None
) -> str:
    """
    根据布尔条件筛选数据集，支持任意复杂逻辑表达式。

    参数：
    - dataset_key: 数据集键
    - conditions: 布尔表达式字符串，支持 & (与)、| (或)、~ (非)、
                  >、<、>=、<=、==、!= 等运算符
                  示例：
                    "(SSS < 35.0) & (SST > 25.0)"
                    "(TEMP > 20) | (PSAL < 34)"
                    "~(LATITUDE > 0)"
    - return_mask: 是否只返回布尔掩码（True/False 数组），默认 False
    - new_var_name: 若指定，将掩码作为新变量存入数据集并返回新键

    返回：
    - 统计信息：满足 / 不满足条件的样本数
    - 掩码数组（如果 return_mask=True）
    - 新数据集键（如果 new_var_name 指定）
    """
    manager = get_manager()
    ds = manager.get(dataset_key)
    if ds is None:
        return json.dumps({"status": "error", "message": f"数据集 '{dataset_key}' 不存在"})

    try:
        # ============================================================
        # 安全解析条件表达式
        # ============================================================
        # 构建安全的命名空间，只包含数据集中的变量名
        safe_namespace = {}
        for var_name in ds.data_vars:
            safe_namespace[var_name] = ds[var_name]
        # 也加入坐标变量
        for coord_name in ds.coords:
            if coord_name not in safe_namespace:
                safe_namespace[coord_name] = ds[coord_name]

        # 执行布尔表达式
        mask = eval(conditions, {"__builtins__": {}}, safe_namespace)

        # 确保结果是布尔类型
        if not np.issubdtype(mask.dtype, np.bool_):
            mask = mask.astype(bool)

        # ============================================================
        # 统计结果
        # ============================================================
        total_count = mask.size
        true_count = int(mask.sum().item())
        false_count = total_count - true_count
        true_ratio = true_count / total_count if total_count > 0 else 0.0

        result = {
            "status": "success",
            "condition": conditions,
            "total_samples": total_count,
            "matching_samples": true_count,
            "non_matching_samples": false_count,
            "matching_ratio": round(true_ratio, 4)
        }

        # ============================================================
        # 可选：返回掩码的具体值
        # ============================================================
        if return_mask:
            # 展平为一维数组返回（避免 JSON 序列化多维数组的问题）
            mask_flat = mask.values.flatten().tolist()
            # 同时保留每个剖面（或原维度）的汇总信息
            if 'N_prof' in mask.dims:
                result["per_profile"] = {
                    "profile_index": list(range(mask.sizes.get('N_prof', mask.shape[0]))),
                    "matches": mask.values.astype(int).tolist() if mask.ndim == 1 else mask.any(dim=[d for d in mask.dims if d != 'N_prof']).astype(int).tolist(),
                    "note": "1=满足条件, 0=不满足。若变量是多维的，此处显示每个剖面是否至少有一个点满足条件"
                }
            else:
                result["mask_values"] = mask_flat[:50]  # 只返回前50个，避免过长

        # ============================================================
        # 可选：将掩码存入数据集
        # ============================================================
        if new_var_name:
            new_ds = ds.copy()
            new_ds[new_var_name] = mask.astype(int)  # 存为整数（0/1）
            new_key = manager.add(new_ds)
            result["new_cache_key"] = new_key
            result["new_var_name"] = new_var_name
            result["note"] = f"布尔掩码已作为变量 '{new_var_name}' 存入新数据集，键: {new_key}"

        return json.dumps(result)

    except NameError as e:
        var_name = str(e).split("'")[1] if "'" in str(e) else str(e)
        return json.dumps({
            "status": "error",
            "message": f"条件表达式中的变量 '{var_name}' 在数据集中不存在。可用变量: {list(ds.data_vars.keys())}",
            "available_variables": list(ds.data_vars.keys())
        })
    except SyntaxError as e:
        return json.dumps({
            "status": "error",
            "message": f"条件表达式语法错误: {str(e)}。请使用 Python 布尔表达式格式，如 '(SSS < 35.0) & (SST > 25.0)'",
            "valid_operators": ["& (与)", "| (或)", "~ (非)", ">", "<", ">=", "<=", "==", "!="],
            "example": "(SSS < 35.0) & (SST > 25.0)"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"条件筛选失败: {str(e)}",
            "suggestion": "请检查条件表达式是否正确，变量名是否存在于数据集中"
        })