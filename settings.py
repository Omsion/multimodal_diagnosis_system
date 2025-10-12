# settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """系统配置，可被环境变量或.env文件覆盖"""
    # 模型路径
    RESNET_MODEL_PATH: str = "./models/resnet50_dr_grading.pth"
    QWEN_VL_MODEL_PATH: str = "./models/Qwen-VL"
    R1_7B_MODEL_PATH: str = "./models/R1-7B-finetuned"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG配置
    VECTOR_DB_PATH: str = "./vector_db"
    KNOWLEDGE_BASE_PATH: str = "./knowledge_base"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3
    
    # DR分级映射
    DR_GRADES: dict = {
        0: "无DR", 1: "轻度非增殖性DR", 2: "中度非增殖性DR",
        3: "重度非增殖性DR", 4: "增殖性DR"
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 创建一个全局配置实例供其他模块使用
settings = Settings()