# api_models.py
"""
API模型包装器模块

该模块提供对外部API模型（如Qwen-VL和DeepSeek-Chat）的统一访问接口。
旨在替代本地重量级模型，提供轻量级的运行模式。
"""

import os
import base64
import logging
from io import BytesIO
from typing import Optional, Dict, Any
from PIL import Image

from app.config.settings import settings

logger = logging.getLogger(__name__)

class QwenVLAPIClient:
    """
    Qwen-VL API客户端
    
    使用DashScope API调用Qwen-VL模型进行图像分析。
    """
    
    def __init__(self):
        self.api_key = settings.DASHSCOPE_API_KEY
        self.model_name = settings.QWEN_VL_MODEL_NAME
        
        if not self.api_key:
            logger.warning("未配置DASHSCOPE_API_KEY，Qwen-VL API将无法工作")
            
    def generate_description(self, image: Image.Image, dr_grade_desc: str) -> str:
        """
        生成眼底图像的病灶描述
        
        Args:
            image (Image.Image): 输入的眼底图像
            dr_grade_desc (str): DR分级描述
            
        Returns:
            str: 病灶描述文本
        """
        try:
            import dashscope
            from http import HTTPStatus
        except ImportError:
            logger.error("未安装dashscope库，请运行 pip install dashscope")
            return "系统错误：缺少dashscope依赖"

        if not self.api_key:
            return "配置错误：缺少DASHSCOPE_API_KEY"

        try:
            # 将图像转换为base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_data = f"data:image/png;base64,{img_str}"
            
            # 构建Prompt
            prompt = self._build_medical_prompt(dr_grade_desc)
            
            # 调用API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": img_data},
                        {"text": prompt}
                    ]
                }
            ]
            
            response = dashscope.MultiModalConversation.call(
                model=self.model_name,
                messages=messages,
                api_key=self.api_key
            )
            
            if response.status_code == HTTPStatus.OK:
                description = response.output.choices[0].message.content[0]["text"]
                return self._post_process_description(description, dr_grade_desc)
            else:
                logger.error(f"Qwen-VL API调用失败: {response.code} - {response.message}")
                return f"API调用失败: {response.message}"
                
        except Exception as e:
            logger.error(f"Qwen-VL API处理异常: {e}")
            return f"处理异常: {str(e)}"

    def _build_medical_prompt(self, dr_grade_desc: str) -> str:
        """构建专业的医学提示词"""
        return f"""作为一名专业的眼科医生，请详细分析这张诊断为'{dr_grade_desc}'的眼底图像。
请用专业术语描述图中观察到的关键病灶特征，包括：
1. 微血管瘤的数量和分布
2. 出血斑的位置和范围
3. 硬性渗出的形态特征
4. 棉绒斑的有无及分布
5. 血管异常（如静脉串珠、IRMA等）
6. 其他相关病理改变

请提供简洁、准确、专业的描述。"""

    def _post_process_description(self, description: str, dr_grade_desc: str) -> str:
        """后处理生成的描述"""
        return description.strip()


def get_deepseek_chat_model():
    """
    获取DeepSeek Chat LangChain组件
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOpenAI
        except ImportError:
            logger.error("未安装langchain-openai或langchain-community")
            return None
            
    if not settings.DEEPSEEK_API_KEY:
        logger.warning("未配置DEEPSEEK_API_KEY")
        return None
        
    return ChatOpenAI(
        model=settings.DEEPSEEK_MODEL_NAME,
        openai_api_key=settings.DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com",
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        top_p=settings.LLM_TOP_P
    )
