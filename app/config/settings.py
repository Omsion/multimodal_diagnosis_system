# settings.py
"""
系统配置管理模块

使用Pydantic BaseSettings进行配置管理，支持：
- 环境变量覆盖
- .env文件配置
- 类型验证和默认值
- 配置文档生成
- 运行时配置热重载
"""

import os
from typing import Dict, List, Optional, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from enum import Enum

class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ModelConfig(BaseSettings):
    """模型配置基类"""
    model_path: str = Field(..., description="模型文件路径")
    device: str = Field("auto", description="运行设备 (auto/cpu/cuda)")
    batch_size: int = Field(1, description="批处理大小")
    max_memory_gb: Optional[float] = Field(None, description="最大内存使用量(GB)")

    @validator('device')
    def validate_device(cls, v):
        allowed_devices = ['auto', 'cpu', 'cuda']
        if v not in allowed_devices:
            raise ValueError(f'device must be one of {allowed_devices}')
        return v

class Settings(BaseSettings):
    """
    系统主配置类

    支持通过环境变量或.env文件覆盖默认配置。
    所有配置都有类型验证和文档说明。
    """

    # === 基础配置 ===
    APP_NAME: str = Field("多模态DR智能诊断系统", description="应用名称")
    VERSION: str = Field("3.0.0", description="应用版本")
    DEBUG: bool = Field(False, description="调试模式")
    LOG_LEVEL: LogLevel = Field(LogLevel.INFO, description="日志级别")

    # === 服务器配置 ===
    HOST: str = Field("0.0.0.0", description="服务器监听地址")
    PORT: int = Field(8000, description="服务器端口")
    MAX_CONCURRENT_REQUESTS: int = Field(10, description="最大并发请求数")
    REQUEST_TIMEOUT: int = Field(300, description="请求超时时间(秒)")

    # === 模型配置 ===
    RESNET_MODEL_PATH: str = Field(
        "./models/best_model_resnet152_simple_finetune.pth",
        description="DR分级ResNet模型路径"
    )
    QWEN_VL_MODEL_PATH: str = Field(
        "./models/Qwen-VL",
        description="Qwen-VL视觉语言模型路径"
    )
    R1_7B_MODEL_PATH: str = Field(
        "./models/R1-7B-finetuned",
        description="R1-7B医疗推理模型路径"
    )

    # === 模型参数配置 ===
    # DR分级模型配置
    DR_MODEL_BATCH_SIZE: int = Field(1, description="DR模型批处理大小")
    DR_MODEL_INPUT_SIZE: int = Field(224, description="DR模型输入图像尺寸")

    # Qwen-VL模型配置
    QWEN_MAX_NEW_TOKENS: int = Field(256, description="Qwen-VL最大生成长度")
    QWEN_TEMPERATURE: float = Field(0.7, description="Qwen-VL生成温度")
    QWEN_TOP_P: float = Field(0.9, description="Qwen-VL top-p采样参数")

    # === 安全配置 ===
    MAX_UPLOAD_SIZE: int = Field(50 * 1024 * 1024, description="最大上传文件大小(字节)")
    ALLOWED_IMAGE_TYPES: List[str] = Field(
        default=["image/jpeg", "image/jpg", "image/png"],
        description="允许的图像类型"
    )
    MIN_IMAGE_SIZE: int = Field(224, description="最小图像尺寸")
    MAX_IMAGE_SIZE: int = Field(4096, description="最大图像尺寸")

    # === 监控配置 ===
    ENABLE_METRICS: bool = Field(True, description="启用性能指标收集")
    METRICS_PORT: int = Field(9090, description="指标暴露端口")
    HEALTH_CHECK_INTERVAL: int = Field(30, description="健康检查间隔(秒)")

    # === DR分级映射配置 ===
    DR_GRADES: Dict[int, str] = Field(
        default={
            0: "无DR",
            1: "轻度非增殖性DR",
            2: "中度非增殖性DR",
            3: "重度非增殖性DR",
            4: "增殖性DR"
        },
        description="DR严重程度等级映射"
    )

    # === 治疗建议配置 ===
    TREATMENT_RECOMMENDATIONS: Dict[str, List[str]] = Field(
        default={
            "无DR": [
                "保持良好的血糖控制",
                "每年进行一次眼底检查",
                "控制血压和血脂",
                "保持健康的生活方式"
            ],
            "轻度非增殖性DR": [
                "严格控制血糖、血压和血脂",
                "每6个月进行一次眼底检查",
                "改善生活方式",
                "考虑使用眼科药物控制病变进展"
            ],
            "中度非增殖性DR": [
                "立即进行全面的眼科评估",
                "每3-4个月进行一次眼底检查",
                "可能需要激光治疗",
                "严格控制全身情况"
            ],
            "重度非增殖性DR": [
                "紧急眼科会诊",
                "考虑进行全视网膜光凝治疗",
                "每月密切随访",
                "严格控制全身状况"
            ],
            "增殖性DR": [
                "立即进行抗VEGF治疗或激光治疗",
                "密切监测视力变化",
                "每月或更频繁随访",
                "考虑玻璃体手术"
            ]
        },
        description="基于DR分级的治疗建议"
    )

    @validator('RESNET_MODEL_PATH')
    def validate_resnet_path(cls, v):
        """验证ResNet模型路径"""
        if not os.path.exists(v):
            import logging
            logging.warning(f"DR分级模型文件不存在: {v}")
        return v

    @validator('QWEN_VL_MODEL_PATH')
    def validate_qwen_path(cls, v):
        """验证Qwen-VL模型路径"""
        if not os.path.exists(v):
            import logging
            logging.warning(f"Qwen-VL模型路径不存在: {v}")
        return v

    @validator('R1_7B_MODEL_PATH')
    def validate_r1_7b_path(cls, v):
        """验证R1-7B模型路径"""
        if not os.path.exists(v):
            import logging
            logging.warning(f"R1-7B模型路径不存在: {v}")
        return v

    @validator('VECTOR_DB_PATH')
    def create_vector_db_dir(cls, v):
        """确保向量数据库目录存在"""
        os.makedirs(v, exist_ok=True)
        return v

    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """验证日志级别"""
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v

    def get_treatment_recommendations(self, dr_grade: int) -> List[str]:
        """
        根据DR等级获取治疗建议

        Args:
            dr_grade (int): DR等级 (0-4)

        Returns:
            List[str]: 治疗建议列表
        """
        dr_desc = self.DR_GRADES.get(dr_grade, "未知等级")
        return self.TREATMENT_RECOMMENDATIONS.get(dr_desc, [
            "建议进行专业眼科检查",
            "根据检查结果制定个性化治疗方案"
        ])

    def get_dr_grade_description(self, dr_grade: int) -> str:
        """
        获取DR等级描述

        Args:
            dr_grade (int): DR等级

        Returns:
            str: DR等级描述
        """
        return self.DR_GRADES.get(dr_grade, "未知等级")

    def is_image_type_allowed(self, content_type: str) -> bool:
        """
        检查图像类型是否允许

        Args:
            content_type (str): 文件类型

        Returns:
            bool: 是否允许
        """
        return content_type.lower() in [t.lower() for t in self.ALLOWED_IMAGE_TYPES]

    def get_model_config_dict(self) -> Dict[str, Any]:
        """
        获取模型配置字典

        Returns:
            Dict[str, Any]: 模型配置
        """
        return {
            "resnet": {
                "path": self.RESNET_MODEL_PATH,
                "batch_size": self.DR_MODEL_BATCH_SIZE,
                "input_size": self.DR_MODEL_INPUT_SIZE
            },
            "qwen_vl": {
                "path": self.QWEN_VL_MODEL_PATH,
                "max_new_tokens": self.QWEN_MAX_NEW_TOKENS,
                "temperature": self.QWEN_TEMPERATURE,
                "top_p": self.QWEN_TOP_P
            },
            "llm": {
                "path": self.R1_7B_MODEL_PATH,
                "max_tokens": self.LLM_MAX_TOKENS,
                "temperature": self.LLM_TEMPERATURE,
                "top_p": self.LLM_TOP_P
            }
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        # 允许字段别名
        allow_population_by_field_name = True

        # 环境变量前缀
        env_prefix = "DR_SYSTEM_"

# 创建全局配置实例
settings = Settings()

# 配置验证和初始化日志
import logging

def setup_logging():
    """设置日志配置"""
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': settings.LOG_LEVEL.value,
                'formatter': 'detailed',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': settings.LOG_LEVEL.value,
                'formatter': 'detailed',
                'filename': './logs/diagnosis_system.log',
                'mode': 'a',
                'encoding': 'utf-8'
            }
        },
        'root': {
            'level': settings.LOG_LEVEL.value,
            'handlers': ['console', 'file'] if not settings.DEBUG else ['console']
        }
    }

    # 确保日志目录存在
    os.makedirs('./logs', exist_ok=True)

    return log_config

# 导出配置函数
__all__ = [
    'settings',
    'LogLevel',
    'ModelConfig',
    'setup_logging'
]