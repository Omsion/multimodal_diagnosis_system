# llm_loader.py
"""
LLM加载器模块

该模块负责加载和管理R1-7B大型语言模型，并提供LangChain集成。
包含以下功能：
- 自动设备检测和模型优化
- LoRA适配器支持
- 内存优化和量化支持
- 优雅的错误处理和降级机制
"""

import os
import gc
import torch
import warnings
from typing import Optional, Dict, Any
from contextlib import contextmanager

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.language_models.chat_models import BaseChatModel

from settings import settings
import logging

logger = logging.getLogger(__name__)

# 抑制一些非关键警告
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

class LLMLoader:
    """
    大型语言模型加载器

    提供统一的模型加载接口，支持多种优化策略和错误恢复机制。
    """

    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化LLM加载器

        Args:
            model_path (str): 模型路径
            config (Optional[Dict[str, Any]]): 额外配置参数
        """
        self.model_path = model_path
        self.config = config or {}
        self.device = self._detect_device()
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def _detect_device(self) -> str:
        """
        自动检测最佳运行设备

        Returns:
            str: 设备类型 (cuda/cpu)
        """
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)

            if gpu_memory_gb >= 8:  # 至少8GB显存
                logger.info(f"检测到GPU，显存: {gpu_memory_gb:.1f}GB，使用CUDA")
                return "cuda"
            else:
                logger.warning(f"GPU显存不足({gpu_memory_gb:.1f}GB)，使用CPU")
                return "cpu"
        else:
            logger.info("未检测到GPU，使用CPU")
            return "cpu"

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        获取量化配置以减少内存使用

        Returns:
            Optional[BitsAndBytesConfig]: 量化配置
        """
        if self.device == "cuda":
            try:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            except Exception as e:
                logger.warning(f"无法启用4位量化: {e}")
                return None
        return None

    def _get_model_config(self) -> Dict[str, Any]:
        """
        获取模型加载配置

        Returns:
            Dict[str, Any]: 模型配置字典
        """
        config = {
            "device_map": "auto" if self.device == "cuda" else "cpu",
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # 添加量化配置
        quantization_config = self._get_quantization_config()
        if quantization_config:
            config["quantization_config"] = quantization_config

        return config

    @contextmanager
    def _memory_efficient_loading(self):
        """
        内存高效的加载上下文管理器
        """
        # 清理现有缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 设置优化选项
        original_gc_state = gc.isenabled()
        gc.disable()  # 暂时禁用GC以提高加载速度

        try:
            yield
        finally:
            # 恢复GC并清理内存
            if original_gc_state:
                gc.enable()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_tokenizer(self) -> bool:
        """
        加载tokenizer

        Returns:
            bool: 是否成功加载
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left",
                truncation=True
            )

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"成功加载tokenizer: {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"加载tokenizer失败: {e}")
            return False

    def load_model(self) -> bool:
        """
        加载基础模型

        Returns:
            bool: 是否成功加载
        """
        try:
            with self._memory_efficient_loading():
                config = self._get_model_config()

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **config
                )

                # 应用LoRA适配器（如果存在）
                try:
                    lora_path = os.path.join(self.model_path, "adapter_model")
                    if os.path.exists(lora_path):
                        self.model = PeftModel.from_pretrained(self.model, lora_path)
                        logger.info("成功应用LoRA适配器")
                    else:
                        # 尝试直接从模型路径加载LoRA
                        self.model = PeftModel.from_pretrained(self.model, self.model_path)
                        logger.info("成功应用LoRA适配器（直接模式）")
                except Exception as lora_e:
                    logger.info(f"未找到或加载LoRA适配器失败，使用基础模型: {lora_e}")

                # 设置评估模式
                self.model.eval()

                logger.info(f"成功加载模型: {self.model_path}")
                return True

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

    def create_pipeline(self) -> bool:
        """
        创建HuggingFace pipeline

        Returns:
            bool: 是否成功创建
        """
        try:
            # 生成配置
            generation_config = GenerationConfig(
                max_new_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # 添加更多安全配置
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                generation_config=generation_config,
                # 优化配置
                return_full_text=False,  # 只返回新生成的内容
                clean_up_tokenization_spaces=True,
                # 流式生成支持
                streaming=True,
                # 批处理配置
                batch_size=1,
                # 设备配置
                device=self.device if self.device == "cpu" else 0
            )

            logger.info("成功创建LLM pipeline")
            return True

        except Exception as e:
            logger.error(f"创建pipeline失败: {e}")
            return False

    def load_as_langchain_component(self) -> Optional[BaseChatModel]:
        """
        加载模型并返回LangChain组件

        Returns:
            Optional[BaseChatModel]: LangChain LLM组件
        """
        try:
            # 按顺序加载组件
            if not self.load_tokenizer():
                raise RuntimeError("Tokenizer加载失败")

            if not self.load_model():
                raise RuntimeError("模型加载失败")

            if not self.create_pipeline():
                raise RuntimeError("Pipeline创建失败")

            # 创建LangChain组件
            llm = HuggingFacePipeline(
                pipeline=self.pipeline,
                model_kwargs={"temperature": settings.LLM_TEMPERATURE}
            )

            logger.info("成功创建LangChain LLM组件")
            return llm

        except Exception as e:
            logger.error(f"创建LangChain组件失败: {e}")
            return None

    def unload(self):
        """卸载模型并释放内存"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("模型已卸载，内存已释放")

def load_r1_7b_llm_as_langchain_component() -> Optional[BaseChatModel]:
    """
    加载R1-7B模型作为LangChain组件的便捷函数

    Returns:
        Optional[BaseChatModel]: LangChain LLM组件，失败时返回None
    """
    loader = LLMLoader(
        model_path=settings.R1_7B_MODEL_PATH,
        config={
            "max_new_tokens": settings.LLM_MAX_TOKENS,
            "temperature": settings.LLM_TEMPERATURE,
            "top_p": settings.LLM_TOP_P
        }
    )

    return loader.load_as_langchain_component()

# 导出主要函数
__all__ = [
    'load_r1_7b_llm_as_langchain_component',
    'LLMLoader'
]