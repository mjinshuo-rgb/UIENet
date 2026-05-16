"""
配置加载工具
使用 OmegaConf 从 config.yaml 读取所有超参数，返回可用点号访问的配置对象。

访问规则（必须在所有模块中严格遵守）：
- Phase 1 参数为顶层键，使用 config.phase1_feat_channels 访问
- Phase 2 及后续为嵌套键，使用 config.branch_swin.embed_dim 访问

使用方法:
    from utils.config import load_config
    config = load_config('config.yaml')
    print(config.phase1_feat_channels)  # 64
    print(config.branch_swin.embed_dim) # 64
"""

from pathlib import Path
from omegaconf import OmegaConf


def load_config(config_path):
    """
    加载 YAML 配置文件并返回 OmegaConf 对象

    Args:
        config_path: YAML 文件路径 (str or Path)

    Returns:
        OmegaConf DictConfig 对象，支持点号访问属性
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    conf = OmegaConf.load(config_path)
    return conf