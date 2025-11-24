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
    
    使用OpenAI SDK调用Qwen-VL模型 (兼容模式) 进行图像分析。
    """
    
    def __init__(self):
        self.api_key = settings.DASHSCOPE_API_KEY
        self.base_url = settings.QWEN_BASE_URL
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
            from openai import OpenAI
        except ImportError:
            logger.error("未安装openai库，请运行 pip install openai")
            return "系统错误：缺少openai依赖"

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
            
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_data}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            
            description = completion.choices[0].message.content
            return self._post_process_description(description, dr_grade_desc)
                
        except Exception as e:
            logger.error(f"Qwen-VL API处理异常: {e}")
            return f"处理异常: {str(e)}"

    def generate_description_stream(
        self, 
        image: Image.Image, 
        dr_grade_desc: str,
        enable_thinking: bool = True,
        thinking_budget: int = 81920
    ):
        """
        流式生成眼底图像的病灶描述
        
        Args:
            image (Image.Image): 输入的眼底图像
            dr_grade_desc (str): DR分级描述
            enable_thinking (bool): 是否启用思考过程输出
            thinking_budget (int): 最大推理过程 Token 数
            
        Yields:
            Dict[str, str]: 包含 type ('reasoning' 或 'answer') 和 content 的字典
        """
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("未安装openai库，请运行 pip install openai")
            yield {"type": "error", "content": "系统错误：缺少openai依赖"}
            return

        if not self.api_key:
            yield {"type": "error", "content": "配置错误：缺少DASHSCOPE_API_KEY"}
            return

        try:
            # 将图像转换为base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            img_data = f"data:image/png;base64,{img_str}"
            
            # 构建Prompt
            prompt = self._build_medical_prompt(dr_grade_desc)
            
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            # 流式调用配置
            extra_body = {}
            if enable_thinking:
                extra_body['enable_thinking'] = True
                extra_body['thinking_budget'] = thinking_budget

            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_data}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                stream=True,
                extra_body=extra_body
            )
            
            # 处理流式响应
            for chunk in completion:
                # 如果chunk.choices为空，跳过（可能是usage信息）
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                # 处理思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    yield {
                        "type": "reasoning",
                        "content": delta.reasoning_content
                    }
                # 处理回复内容
                elif hasattr(delta, 'content') and delta.content:
                    yield {
                        "type": "answer",
                        "content": delta.content
                    }
                    
        except Exception as e:
            logger.error(f"Qwen-VL API流式处理异常: {e}")
            yield {"type": "error", "content": f"处理异常: {str(e)}"}


    def _build_medical_prompt(self, dr_grade_desc: str) -> str:
        """构建专业的医学提示词，包含few-shot示例"""
        
        # Few-shot learning: 不同DR等级的关键特征示例
        grade_examples = {
            "无DR": "正常眼底表现：视网膜血管走行清晰，无微动脉瘤，无出血斑，无渗出物，黄斑中心凹反光正常，视盘边界清晰。",
            "轻度非增殖性DR": "轻度病变：可见数个散在的微动脉瘤（1-5个），多位于后极部，无明显出血，无或仅有少量硬性渗出，血管走行基本正常。",
            "中度非增殖性DR": "中度病变：微动脉瘤数量增多（5-15个），散在分布；可见点状出血和/或火焰状出血，部分位于黄斑区；硬性渗出斑增多，呈黄白色；血管轻度迂曲。",
            "重度非增殖性DR": "重度病变：大量微动脉瘤和出血斑广泛分布；明显的硬性渗出，部分融合成斑块；可见棉绒斑（软性渗出）；视网膜内微血管异常（IRMA）；静脉串珠样改变。",
            "增殖性DR": "增殖期病变：除上述病变外，可见新生血管形成（视盘或视网膜表面）；玻璃体积血征象；纤维增殖膜形成；部分病例可见牵拉性视网膜脱离征象。"
        }
        
        example = grade_examples.get(dr_grade_desc, "")
        
        return f"""作为一名专业的眼科医生，请详细分析这张诊断为'{dr_grade_desc}'的眼底图像。

参考典型特征（{dr_grade_desc}）：
{example}

请根据实际观察到的图像特征，用专业术语详细描述：

1. **微动脉瘤（Microaneurysms）**：数量、大小、分布位置（后极部/周边部）
2. **出血（Hemorrhages）**：
   - 点状出血（dot hemorrhages）的位点
   - 火焰状出血（flame-shaped hemorrhages）的范围
   - 出血层次（视网膜内/视网膜前/玻璃体）
3. **渗出物（Exudates）**：
   - 硬性渗出（hard exudates）：颜色、形态、是否融合
   - 棉绒斑（cotton-wool spots）：数量和位置
4. **血管异常（Vascular abnormalities）**：
   - 静脉串珠（venous beading）
   - 视网膜内微血管异常（IRMA）
   - 血管迂曲程度
5. **黄斑区情况**：中心凹光反射、有无水肿、是否累及黄斑
6. **新生血管（Neovascularization）**：有无视盘新生血管（NVD）或视网膜表面新生血管（NVE）
7. **其他病理改变**：纤维增殖、玻璃体变化、视网膜脱离等

请提供简洁、准确、专业的描述（150-200字）。"""

    def _post_process_description(self, description: str, dr_grade_desc: str) -> str:
        """后处理生成的描述"""
        if not description:
            return "无法生成描述"
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
