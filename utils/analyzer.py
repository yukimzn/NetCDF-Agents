"""
analyzer.py
NetCDF 文件分析器 - 具备完整的容错机制
当遇到无法解析的元数据时，保留原始数据而不是抛出异常
"""

import json
import numpy as np
import xarray as xr
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import warnings
import traceback
import sys
from utils.JsonProcess import safe_serialize, to_json


def simplify_metadata(metadata: Dict[str, Any], max_variables: int = 10) -> Dict[str, Any]:
    """
    简化 NetCDF 元数据，只保留核心信息
    
    Args:
        metadata: 原始元数据字典
        max_variables: 最多保留的变量数量（避免过长）
    
    Returns:
        简化后的元数据字典
    """
    simplified = {}
    
    # 1. 基本信息
    print("\n获取路径\n")
    simplified["file_path"] = metadata["file_path"]
    print("\n路径获取正确\n")
    # 2. 维度信息（保留）
    simplified["dimensions"] = metadata.get("dimensions", {})
    
    # 3. 坐标信息（简化）
    coordinates = metadata.get("coordinates", {})
    simplified_coords = {}
    for coord_name, coord_info in coordinates.items():
        simplified_coords[coord_name] = {
            "range": coord_info.get("range"),
            "units": coord_info.get("units", "unknown"),
            "resolution": coord_info.get("resolution") if "resolution" in coord_info else None
        }
    simplified["coordinates"] = simplified_coords
    
    # 4. 变量信息（只保留前 max_variables 个，并简化）
    variables = metadata.get("variables", [])
    simplified_vars = []
    for var in variables[:max_variables]:
        simplified_vars.append({
            "name": var.get("name"),
            "description": var.get("description", var.get("name")),
            "units": var.get("units", "unknown"),
            "dimensions": var.get("dimensions", []),
            "shape": var.get("shape", [])
        })
    
    simplified["variables"] = {
        "count": len(variables),
        "sample": simplified_vars,
        "note": f"共 {len(variables)} 个变量，仅显示前 {max_variables} 个" if len(variables) > max_variables else None
    }
    
    # 5. 空间覆盖范围
    spatial_coverage = metadata.get("spatial_coverage", {})
    if spatial_coverage:
        simplified["spatial_coverage"] = spatial_coverage
    else:
        # 尝试从全局属性中提取经纬度范围
        global_attrs = metadata.get("global_attributes", {})
        lon_range = None
        lat_range = None
        
        if "westernmost_longitude" in global_attrs and "easternmost_longitude" in global_attrs:
            lon_range = [global_attrs["westernmost_longitude"], global_attrs["easternmost_longitude"]]
        if "sourthenmost_latitude" in global_attrs and "northernmost_latitude" in global_attrs:
            lat_range = [global_attrs["sourthenmost_latitude"], global_attrs["northernmost_latitude"]]
        
        if lon_range and lat_range:
            simplified["spatial_coverage"] = {
                "lon_range": f"{lon_range[0]:.2f} to {lon_range[1]:.2f}",
                "lat_range": f"{lat_range[0]:.2f} to {lat_range[1]:.2f}"
            }
    
    # 6. 全局属性（只保留关键信息）
    global_attrs = metadata.get("global_attributes", {})
    key_attrs = [
        "title", "institution", "project_name", "license", "product_version",
        "time_coverage_start", "time_coverage_end", "date_created"
    ]
    simplified["global_attributes"] = {k: global_attrs.get(k) for k in key_attrs if k in global_attrs}
    
    # 添加经纬度范围（如果存在且未被提取）
    if "spatial_coverage" not in simplified:
        if "northernmost_latitude" in global_attrs:
            simplified["global_attributes"]["northernmost_latitude"] = global_attrs["northernmost_latitude"]
        if "sourthenmost_latitude" in global_attrs:
            simplified["global_attributes"]["southernmost_latitude"] = global_attrs["sourthenmost_latitude"]
        if "westernmost_longitude" in global_attrs:
            simplified["global_attributes"]["westernmost_longitude"] = global_attrs["westernmost_longitude"]
        if "easternmost_longitude" in global_attrs:
            simplified["global_attributes"]["easternmost_longitude"] = global_attrs["easternmost_longitude"]
    from utils.JsonProcess import safe_serialize
    return safe_serialize(simplified)


# print(f"\n {sys.path}\n")
class NetCDFAnalyzer:
    """NC文件分析工具类"""
    
    @staticmethod
    def extract_metadata(nc_file_path: str) -> Dict[str, Any]:
        """
        读取NC文件并提取元数据
        Args:
            nc_file_path: NC文件路径
        Returns:
            包含维度、变量、坐标等信息的字典
        """
        
        try:
            print("简单版本的nc analyzer，用来避免奇怪错误。\n")
            ds = xr.open_dataset(nc_file_path,decode_times=False, decode_cf=False)
            dimensions_dict = dict(ds.sizes)
            metadata = {
                "file_path": nc_file_path,
                "dimensions": dimensions_dict,
                "coordinates": {},
                "variables": [],
                "spatial_coverage": {},
                "global_attributes": dict(ds.attrs)
            }
            
            # 提取坐标信息
            for coord_name in ds.coords:
                coord = ds[coord_name]
                if coord.size > 0:
                    coord_info = {
                        "range": [float(coord.min().values), float(coord.max().values)],
                        "units": str(coord.attrs.get('units', 'unknown')),
                        "description": str(coord.attrs.get('long_name', coord_name))
                    }
                    # 计算分辨率（如果是数值型且一维）
                    if coord.ndim == 1 and coord.size > 1:
                        try:
                            coord_values = coord.values
                            if np.issubdtype(coord_values.dtype, np.number):
                                diff =  np.diff(coord_values)
                                coord_info["resolution"] = float(np.mean(diff))
                            else:
                                coord_info["resolution"] = "non-numeric coordinate"
                        except Exception as e:
                            coord_info["resolution"] = None
                    metadata["coordinates"][coord_name] = coord_info
            
            # 提取变量信息
            for var_name in ds.data_vars:
                var = ds[var_name]
                metadata["variables"].append({
                    "name": var_name,
                    "description": str(var.attrs.get('long_name', var_name)),
                    "units": str(var.attrs.get('units', 'unknown')),
                    "dimensions": list(var.sizes),
                    "type": str(var.dtype),
                    "shape": list(var.shape),
                    "attributes": dict(var.attrs)
                })
            
            # 提取空间覆盖范围
            lat_names = [c for c in metadata["coordinates"] if 'lat' in c.lower()]
            lon_names = [c for c in metadata["coordinates"] if 'lon' in c.lower()]
            if lat_names and lon_names:
                lat_name = lat_names[0]
                lon_name = lon_names[0]
                lat_range = metadata["coordinates"][lat_name]["range"]
                lon_range = metadata["coordinates"][lon_name]["range"]
                metadata["spatial_coverage"] = {
                    "lon_range": f"{lon_range[0]:.2f} to {lon_range[1]:.2f}",
                    "lat_range": f"{lat_range[0]:.2f} to {lat_range[1]:.2f}"
                }
            
            ds.close()
            return metadata
        except Exception as e:
            print(f"读取文件失败: {str(e)}")
            return {"error": f"读取文件失败: {str(e)}"}

    def get_compact_summary(ds_or_metadata, max_variables: int = 5) -> Dict[str, Any]:
        """
        获取极简摘要（用于 prompt）
        
        Args:
            ds_or_metadata: xr.Dataset 对象或 metadata 字典
            max_variables: 最多显示的变量数量
        
        Returns:
            极简摘要字典
        """
        # 如果是 Dataset 对象，先提取简单信息
        if hasattr(ds_or_metadata, 'sizes'):
            return {
                "dimensions": dict(ds_or_metadata.sizes),
                "variables_count": len(ds_or_metadata.data_vars),
                "coordinates": list(ds_or_metadata.coords.keys())[:5],
                "has_lat": any('lat' in c.lower() for c in ds_or_metadata.coords),
                "has_lon": any('lon' in c.lower() for c in ds_or_metadata.coords),
            }
        
        # 如果是 metadata 字典
        if isinstance(ds_or_metadata, dict):
            variables = ds_or_metadata.get("variables", [])
            coords = ds_or_metadata.get("coordinates", {})
            
            # 提取经纬度范围
            lon_range = None
            lat_range = None
            spatial = ds_or_metadata.get("spatial_coverage", {})
            if spatial:
                lon_range = spatial.get("lon_range")
                lat_range = spatial.get("lat_range")
            else:
                attrs = ds_or_metadata.get("global_attributes", {})
                if "westernmost_longitude" in attrs and "easternmost_longitude" in attrs:
                    lon_range = [attrs["westernmost_longitude"], attrs["easternmost_longitude"]]
                if "sourthenmost_latitude" in attrs and "northernmost_latitude" in attrs:
                    lat_range = [attrs["sourthenmost_latitude"], attrs["northernmost_latitude"]]
            
            return {
                "dimensions": ds_or_metadata.get("dimensions", {}),
                "variables_count": len(variables),
                "variables_sample": [v.get("name") for v in variables[:max_variables]],
                "coordinates": list(coords.keys())[:5],
                "lon_range": lon_range,
                "lat_range": lat_range,
            }
        
        return {}

