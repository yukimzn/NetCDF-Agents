import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 核心模型配置（从环境变量读取，提供默认值仅为防止报错）
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-coder:480b-cloud")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
    
    # 系统参数
    MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "5"))
    MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "10"))
    DATA_ROOT = os.getenv("DATA_ROOT", "./data")
    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./checkpoints")