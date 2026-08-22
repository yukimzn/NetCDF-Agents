"""
通用JSON序列化模块
专门处理xarray、numpy、pandas等科学计算库的数据类型转换
"""

import json
import numpy as np
import xarray as xr
import pandas as pd
from typing import Any, Dict, List, Union, Optional
from datetime import datetime, date


class NumpyJSONEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，处理numpy、xarray、pandas等类型
    """
    
    def default(self, obj: Any) -> Any:
        # 处理numpy类型
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.datetime64):
            return str(obj)
        elif isinstance(obj, np.complexfloating):
            return float(obj.real) if obj.imag == 0 else str(obj)
        
        # 处理pandas类型
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        
        # 处理xarray类型
        elif isinstance(obj, xr.DataArray):
            return {
                "dims": list(obj.dims),
                "values": obj.values.tolist() if obj.size < 10000 else f"array_shape_{obj.shape}",
                "attrs": safe_serialize(obj.attrs)
            }
        elif isinstance(obj, xr.Dataset):
            return {
                "dims": dict(obj.dims),
                "data_vars": list(obj.data_vars),
                "coords": list(obj.coords),
                "attrs": safe_serialize(obj.attrs)
            }
        
        # 处理datetime
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # 处理可迭代对象但不包括字符串和字典
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, dict, list, tuple)):
            return list(obj)
        
        # 处理其他有item方法的对象（numpy scalar）
        elif hasattr(obj, 'item'):
            try:
                return obj.item()
            except:
                return str(obj)
        
        # 默认处理
        return super().default(obj)


def safe_serialize(obj: Any, max_depth: int = 10, current_depth: int = 0) -> Any:
    """
    递归地将对象转换为JSON可序列化的格式
    
    Args:
        obj: 需要转换的对象
        max_depth: 最大递归深度
        current_depth: 当前递归深度
    
    Returns:
        JSON可序列化的对象
    """
    if current_depth > max_depth:
        return str(obj)
    
    # None值处理
    if obj is None:
        return None
    
    # 基础类型直接返回
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # numpy类型转换
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        if obj.size > 10000:  # 大数组只返回形状信息
            return f"array_shape_{obj.shape}_dtype_{obj.dtype}"
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.datetime64):
        return str(obj)
    
    # xarray类型
    elif isinstance(obj, xr.DataArray):
        return {
            "type": "DataArray",
            "name": obj.name,
            "dims": list(obj.dims),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "values": safe_serialize(obj.values, max_depth, current_depth + 1),
            "attrs": safe_serialize(obj.attrs, max_depth, current_depth + 1)
        }
    elif isinstance(obj, xr.Dataset):
        return {
            "type": "Dataset",
            "dims": dict(obj.dims),
            "data_vars": list(obj.data_vars),
            "coords": list(obj.coords),
            "attrs": safe_serialize(obj.attrs, max_depth, current_depth + 1)
        }
    
    # 字典处理
    elif isinstance(obj, dict):
        return {
            safe_serialize(k, max_depth, current_depth + 1): 
            safe_serialize(v, max_depth, current_depth + 1) 
            for k, v in obj.items()
        }
    
    # 列表/元组处理
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item, max_depth, current_depth + 1) for item in obj]
    
    # pandas类型
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    
    # 其他类型转换为字符串
    else:
        try:
            # 尝试直接序列化
            json.dumps(obj, cls=NumpyJSONEncoder)
            return obj
        except:
            return str(obj)


def to_json(data: Any, ensure_ascii: bool = False, indent: Optional[int] = None) -> str:
    """
    将数据转换为JSON字符串
    
    Args:
        data: 需要转换的数据
        ensure_ascii: 是否确保ASCII
        indent: 缩进空格数
    
    Returns:
        JSON字符串
    """
    safe_data = safe_serialize(data)
    return json.dumps(safe_data, cls=NumpyJSONEncoder, 
                     ensure_ascii=ensure_ascii, indent=indent)


def to_dict(data: Any) -> Dict:
    """
    将数据转换为可JSON序列化的字典
    
    Args:
        data: 需要转换的数据
    
    Returns:
        可JSON序列化的字典
    """
    return safe_serialize(data)


def is_json_serializable(obj: Any) -> bool:
    """
    检查对象是否可JSON序列化
    
    Args:
        obj: 需要检查的对象
    
    Returns:
        是否可序列化
    """
    try:
        json.dumps(obj, cls=NumpyJSONEncoder)
        return True
    except:
        return False


class SerializableDataManager:
    """
    可序列化的数据管理器包装器
    自动处理所有数据集的JSON序列化
    """
    
    def __init__(self, manager):
        self.manager = manager
    
    def get_serializable_summary(self, key: str) -> Dict:
        """获取可序列化的数据集摘要"""
        summary = self.manager.get_summary(key)
        return safe_serialize(summary)
    
    def get_serializable_datasets_info(self) -> Dict:
        """获取所有数据集的可序列化信息"""
        keys = self.manager.get_keys()
        summaries = {}
        for key in keys:
            summary = self.manager.get_summary(key)
            summaries[key] = safe_serialize(summary)
        return {
            "keys": keys,
            "summaries": summaries,
            "count": len(keys)
        }

'''
# 使用示例
if __name__ == "__main__":
    # 测试numpy类型
    test_data = {
        "int32": np.int32(42),
        "float32": np.float32(3.14),
        "array": np.array([1, 2, 3]),
        "nested": {
            "value": np.float64(123.456)
        }
    }
    
    # 转换为JSON
    json_str = to_json(test_data, indent=2)
    print(json_str)
    
    # 检查是否可序列化
    print(f"Is serializable: {is_json_serializable(test_data)}")
'''