import xarray as xr
from typing import Dict, List, Optional, Any
import uuid

class DatasetManager:
    """
    全局数据集管理器，负责：
    - 存储 xarray.Dataset 及其文件路径映射
    - 自动生成并缓存轻量级元数据（变量名、维度、标题等），供大模型理解
    - 实现会话级别的键隔离，防止多轮对话间的数据集互相污染
    """

    def __init__(self):
        # 核心存储
        self._datasets: Dict[str, xr.Dataset] = {}
        self._file_paths: Dict[str, str] = {}
        # 轻量元数据缓存：key -> 摘要字典
        self._light_metadata_cache: Dict[str, Dict] = {}
        # 会话隔离：session_id -> 持有数据集键的集合
        self._session_keys: Dict[str, set] = {}

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------
    def register_session_key(self, session_id: str, key: str) -> None:
        """将一个数据集键注册到指定会话，表示该会话正在使用它"""
        if session_id:
            self._session_keys.setdefault(session_id, set()).add(key)

    def get_session_keys(self, session_id: str) -> set:
        """返回指定会话持有的所有数据集键"""
        return self._session_keys.get(session_id, set())

    def remove_key_if_unused(self, key: str, session_id: str) -> None:
        """
        释放一个数据集键。仅当没有其他会话仍在使用此键时，才真正删除数据集。
        """
        if session_id in self._session_keys:
            self._session_keys[session_id].discard(key)
        # 检查是否还有其他会话引用
        still_used = any(key in keys for keys in self._session_keys.values())
        if not still_used:
            self.remove(key)

    # ------------------------------------------------------------------
    # 数据集增删改查
    # ------------------------------------------------------------------
    def add(self,
            dataset: xr.Dataset,
            key: Optional[str] = None,
            file_path: Optional[str] = None,
            lineage_note: str = "") -> str:
        """
        添加数据集到管理器，自动生成轻量元数据并缓存。
        返回分配或指定的数据集键。
        """
        if key is None:
            key = f"ds_{uuid.uuid4().hex[:8]}"
        self._datasets[key] = dataset
        if file_path:
            self._file_paths[key] = file_path

        # 自动生成轻量摘要
        self._light_metadata_cache[key] = self._generate_light_metadata(
            dataset, file_path, lineage_note
        )
        return key

    def _generate_light_metadata(self,
                                 ds: xr.Dataset,
                                 file_path: Optional[str] = None,
                                 lineage_note: str = "") -> Dict:
        """
        生成只供 LLM 理解的轻量级摘要，包含：
        - 变量名列表
        - 维度及大小
        - 坐标轴列表
        - 数据集标题（若存在）
        - 数据来源（文件路径或内存说明）
        """
        dims = {k: int(v) for k, v in ds.sizes.items()}  # 确保为普通 int
        vars_list = [str(v) for v in ds.data_vars.keys()]
        coords = [str(c) for c in ds.coords.keys()]
        title = ds.attrs.get('title', '') if ds.attrs else ''
        source = file_path if file_path else f"内存数据集 ({lineage_note})"

        return {
            "source": source,
            "title": title,
            "dimensions": dims,
            "variables": vars_list,
            "coordinates": coords,
            "lineage": lineage_note if lineage_note else "无",
        }

    def replace(self, key: str, dataset: xr.Dataset, lineage_note: str = "") -> None:
        """替换指定键的数据集，并更新轻量元数据"""
        if key not in self._datasets:
            raise KeyError(f"数据集键 '{key}' 不存在")
        self._datasets[key] = dataset
        # 刷新轻量元数据
        file_path = self._file_paths.get(key)
        self._light_metadata_cache[key] = self._generate_light_metadata(
            dataset, file_path, lineage_note
        )

    def remove(self, key: str) -> None:
        """彻底删除数据集及关联的元数据、文件路径映射"""
        if key in self._datasets:
            del self._datasets[key]
        if key in self._file_paths:
            del self._file_paths[key]
        if key in self._light_metadata_cache:
            del self._light_metadata_cache[key]

    def clear(self):
        """清空所有数据集（不重置会话绑定，以保证多轮对话安全）"""
        self._datasets.clear()
        self._file_paths.clear()
        self._light_metadata_cache.clear()

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    def get(self, key: str) -> xr.Dataset:
        """获取数据集，若不存在则抛出 KeyError"""
        if key not in self._datasets:
            raise KeyError(f"数据集键 '{key}' 不存在")
        return self._datasets[key]

    def get_multi(self, keys: List[str]) -> List[xr.Dataset]:
        """批量获取数据集"""
        return [self.get(k) for k in keys]

    def get_file_path(self, key: str) -> Optional[str]:
        """返回数据集关联的文件路径（可能为 None）"""
        return self._file_paths.get(key)

    def list_keys(self) -> List[str]:
        """列出所有已管理的数据集键"""
        return list(self._datasets.keys())

    def contains(self, key: str) -> bool:
        """检查键是否存在"""
        return key in self._datasets

    def find_key_by_file_path(self, file_path: str) -> Optional[str]:
        """根据文件路径反查数据集键"""
        for key, path in self._file_paths.items():
            if path == file_path:
                return key
        return None

    # ------------------------------------------------------------------
    # 轻量元数据访问
    # ------------------------------------------------------------------
    def get_all_light_metadata(self) -> Dict[str, Dict]:
        """返回所有数据集的轻量元数据缓存"""
        return self._light_metadata_cache

    def get_light_metadata(self, key: str) -> Optional[Dict]:
        """返回指定键的轻量元数据，若不存在则返回 None"""
        return self._light_metadata_cache.get(key)

    def get_summary(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        获取数据集摘要（兼容旧接口）。
        - 指定 key：返回该数据集的轻量元数据，若缓存缺失则实时生成并缓存。
        - 不指定 key：返回所有数据集的摘要字典 {key: summary}。
        """
        if key is not None:
            if key in self._light_metadata_cache:
                return self._light_metadata_cache[key]
            ds = self._datasets.get(key)
            if ds is not None:
                metadata = self._generate_light_metadata(ds, self._file_paths.get(key))
                self._light_metadata_cache[key] = metadata
                return metadata
            return {}
        else:
            return {k: self.get_summary(k) for k in self._datasets}
    
    def cache_light_metadata(self, key, light_meta):
        self._light_metadata_cache[key] = light_meta