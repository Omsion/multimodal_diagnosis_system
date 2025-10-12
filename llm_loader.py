# llm_loader.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.language_models.chat_models import BaseChatModel

from settings import settings
import logging

logger = logging.getLogger(__name__)

def load_r1_7b_llm_as_langchain_component() -> BaseChatModel:
    """加载微调后的R1-7B模型，并将其封装为LangChain可用的组件。"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(settings.R1_7B_MODEL_PATH, trust_remote_code=True)
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            settings.R1_7B_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # 检查并应用LoRA适配器
        try:
            model = PeftModel.from_pretrained(base_model, settings.R1_7B_MODEL_PATH)
            logger.info(f"成功加载R1-7B模型并应用LoRA适配器: {settings.R1_7B_MODEL_PATH}")
        except ValueError:
            model = base_model
            logger.info(f"未找到LoRA适配器，加载R1-7B基础模型: {settings.R1_7B_MODEL_PATH}")

        # 创建Hugging Face Pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15
        )
        
        # 封装为LangChain LLM组件
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        logger.error(f"加载R1-7B模型失败: {e}")
        raise