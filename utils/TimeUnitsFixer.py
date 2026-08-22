import xarray as xr
from typing import Dict, List, Optional, Union
import uuid
import json
import hashlib
import re  # 正则表达式
import xarray as xr
import numpy as np
import pandas as pd  # 用于时间处理（如果用到）
from typing import Dict, Optional, List, Literal, Union, Tuple


class TimeUnitFixer:
    """时间单位自动修复器"""

    FIX_PATTERNS = [
        {
            'pattern': r'msec since 00:00:00',
            'replacement': 'milliseconds since 1970-01-01 00:00:00',
            'description': '毫秒单位缺少参考日期'
        },
        {
            'pattern': r'sec since 00:00:00',
            'replacement': 'seconds since 1970-01-01 00:00:00',
            'description': '秒单位缺少参考日期'
        },
        {
            'pattern': r'hours? since\s*$',
            'replacement': 'hours since 1970-01-01 00:00:00',
            'description': '小时单位缺少参考日期'
        },
        {
            'pattern': r'days? since\s*$',
            'replacement': 'days since 1970-01-01 00:00:00',
            'description': '天单位缺少参考日期'
        },
        {
            'pattern': r'since 00:00:00$',
            'replacement': 'since 1970-01-01 00:00:00',
            'description': '缺少年份的参考日期'
        }
    ]

    @classmethod
    def detect_issue(cls, units: str) -> Optional[str]:
        if not units:
            return None
        for pattern_info in cls.FIX_PATTERNS:
            if re.search(pattern_info['pattern'], units, re.IGNORECASE):
                return pattern_info['description']
        return None

    @classmethod
    def fix_units(cls, units: str) -> Tuple[str, bool, str]:
        if not units:
            return units, False, "单位为空，无法修复"
        original = units
        fixed = units
        fix_description = []
        for pattern_info in cls.FIX_PATTERNS:
            if re.search(pattern_info['pattern'], fixed, re.IGNORECASE):
                fixed = re.sub(
                    pattern_info['pattern'],
                    pattern_info['replacement'],
                    fixed,
                    flags=re.IGNORECASE
                )
                fix_description.append(pattern_info['description'])
        is_fixed = (original != fixed)
        description = '; '.join(fix_description) if is_fixed else "无需修复"
        return fixed, is_fixed, description

    @classmethod
    def create_fixed_dataset(cls, ds: xr.Dataset) -> Tuple[xr.Dataset, Dict]:
        fixed_ds = ds.copy(deep=True)
        fix_report = {
            'fixed_coordinates': [],
            'fixed_variables': [],
            'fix_details': [],
            'original_units': {},
            'fixed_units': {}
        }
        all_vars = list(ds.coords.keys()) + list(ds.data_vars.keys())
        for var_name in all_vars:
            var = ds[var_name] if var_name in ds else ds.coords[var_name]
            if 'units' in var.attrs:
                units = var.attrs['units']
                if 'since' in units.lower():
                    original_units = units
                    fixed_units, is_fixed, fix_desc = cls.fix_units(original_units)
                    if is_fixed:
                        if var_name in ds.coords:
                            fix_report['fixed_coordinates'].append(var_name)
                        else:
                            fix_report['fixed_variables'].append(var_name)
                        fix_report['original_units'][var_name] = original_units
                        fix_report['fixed_units'][var_name] = fixed_units
                        fix_report['fix_details'].append(f"{var_name}: {fix_desc}")
                        if var_name in fixed_ds:
                            fixed_ds[var_name].attrs['units'] = fixed_units
                        else:
                            fixed_ds.coords[var_name].attrs['units'] = fixed_units
        return fixed_ds, fix_report
