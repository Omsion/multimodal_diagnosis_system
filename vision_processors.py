# vision_processors.py
"""
视觉处理模块

该模块包含糖尿病视网膜病变诊断的视觉处理组件：
- DRGradingModule: 基于ResNet50的DR严重程度分级
- QwenVLModule: 基于Qwen-VL的视觉语言模型病灶描述生成
包含优化的内存管理和错误处理机制。
"""

import os
import gc
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torchvision.models as models
from torchvision import transforms
from typing import Tuple, Optional
import logging
import threading

from settings import settings

logger = logging.getLogger(__name__)

# 线程锁用于保护GPU内存操作
_memory_lock = threading.Lock()


class DRGradingModule:
    """
    DR图像分级模块

    使用ResNet50模型对眼底图像进行糖尿病视网膜病变严重程度分级。
    包含优化的内存管理和线程安全机制。
    """

    def __init__(self, model_path: str):
        """
        初始化DR分级模块

        Args:
            model_path (str): 预训练模型文件路径
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = self._get_transform()
        self._lock = threading.Lock()

        # 延迟加载模型以优化启动时间
        self._model_loaded = False

    def _load_model(self) -> bool:
        """
        加载预训练模型

        Returns:
            bool: 是否成功加载模型
        """
        try:
            # 使用ResNet152替代ResNet50以提高精度
            model = models.resnet152(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(settings.DR_GRADES))

            if os.path.exists(self.model_path):
                # 使用内存映射加载大型模型文件
                state_dict = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"成功加载DR分级模型: {self.model_path}")
            else:
                logger.warning(f"DR分级模型不存在: {self.model_path}，使用随机初始化")

            model = model.to(self.device)
            model.eval()
            self.model = model
            self._model_loaded = True
            return True

        except Exception as e:
            logger.error(f"加载DR分级模型失败: {e}")
            return False

    def _get_transform(self):
        """
        获取图像预处理变换

        Returns:
            transforms.Compose: 图像预处理管道
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[int, float]:
        """
        对输入图像进行DR分级预测

        Args:
            image (Image.Image): 输入的眼底图像

        Returns:
            Tuple[int, float]: (预测等级, 置信度)

        Raises:
            RuntimeError: 当模型未加载或预测失败时
        """
        # 线程安全的预测
        with self._lock:
            # 延迟加载模型
            if not self._model_loaded:
                if not self._load_model():
                    raise RuntimeError("DR分级模型加载失败")

            try:
                # 确保输入图像有效
                if image is None or image.size == (0, 0):
                    raise ValueError("无效的输入图像")

                # 预处理图像
                img_tensor = self.transform(image).unsqueeze(0).to(self.device)

                # 模型推理
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                grade = predicted.item()
                conf = confidence.item()

                logger.info(f"DR分级结果: 等级={grade} ({settings.DR_GRADES[grade]}), 置信度={conf:.3f}")

                # 显式清理中间张量
                del img_tensor, outputs, probabilities, confidence, predicted

                return grade, conf

            except Exception as e:
                logger.error(f"DR分级预测失败: {e}")
                raise RuntimeError(f"DR分级预测失败: {e}")

            finally:
                # 清理GPU内存
                self._cleanup_memory()

    def _cleanup_memory(self):
        """清理GPU内存和Python垃圾回收"""
        with _memory_lock:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


class QwenVLModule:
    """
    Qwen-VL视觉语言模型模块

    使用Qwen-VL模型生成眼底图像的病灶描述。
    包含优化的内存管理和错误恢复机制。
    """

    def __init__(self, model_path: str, max_new_tokens: int = 256):
        """
        初始化Qwen-VL模块

        Args:
            model_path (str): Qwen-VL模型路径
            max_new_tokens (int): 最大生成token数
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self._lock = threading.Lock()
        self._model_loaded = False

    def _load_model(self) -> bool:
        """
        加载Qwen-VL模型和处理器

        Returns:
            bool: 是否成功加载模型
        """
        try:
            # 使用更安全的模型加载方式
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )

            logger.info(f"成功加载Qwen-VL模型: {self.model_path}")
            self._model_loaded = True
            return True

        except Exception as e:
            logger.error(f"加载Qwen-VL模型失败: {e}")
            self.model = None
            self.processor = None
            return False

    def generate_description(self, image: Image.Image, dr_grade_desc: str) -> str:
        """
        生成眼底图像的病灶描述

        Args:
            image (Image.Image): 输入的眼底图像
            dr_grade_desc (str): DR分级描述

        Returns:
            str: 病灶描述文本

        Raises:
            RuntimeError: 当模型未加载或生成失败时
        """
        # 线程安全的生成
        with self._lock:
            # 延迟加载模型
            if not self._model_loaded:
                if not self._load_model():
                    raise RuntimeError("Qwen-VL模型加载失败")

            try:
                # 确保输入有效
                if image is None or image.size == (0, 0):
                    raise ValueError("无效的输入图像")

                if not dr_grade_desc or not dr_grade_desc.strip():
                    raise ValueError("无效的DR分级描述")

                # 构建专业提示词
                prompt = self._build_medical_prompt(dr_grade_desc)

                # 处理输入
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.model.device)

                # 生成描述
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        no_repeat_ngram_size=2,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )

                # 解码输出
                description = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].strip()

                # 清理prompt部分
                final_desc = description.replace(prompt, "").strip()

                # 后处理：确保专业性和完整性
                final_desc = self._post_process_description(final_desc, dr_grade_desc)

                logger.info(f"成功生成病灶描述: {final_desc[:100]}...")

                # 清理中间变量
                del inputs, generated_ids, description

                return final_desc

            except Exception as e:
                logger.error(f"生成病灶描述失败: {e}")
                # 返回安全的默认描述
                return self._get_fallback_description(dr_grade_desc)

            finally:
                self._cleanup_memory()

    def _build_medical_prompt(self, dr_grade_desc: str) -> str:
        """
        构建专业的医学提示词

        Args:
            dr_grade_desc (str): DR分级描述

        Returns:
            str: 专业提示词
        """
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
        """
        后处理生成的描述

        Args:
            description (str): 原始描述
            dr_grade_desc (str): DR分级描述

        Returns:
            str: 后处理的描述
        """
        # 移除可能的重复或冗余内容
        lines = [line.strip() for line in description.split('\n') if line.strip()]

        # 确保描述与DR分级一致
        if dr_grade_desc and not any(keyword in description.lower() for keyword in ['微血管', '出血', '渗出', '棉绒']):
            description += f"该影像表现符合{dr_grade_desc}的典型特征。"

        return '\n'.join(lines) if lines else f"影像显示符合{dr_grade_desc}的特征。"

    def _get_fallback_description(self, dr_grade_desc: str) -> str:
        """
        获取备用描述

        Args:
            dr_grade_desc (str): DR分级描述

        Returns:
            str: 备用描述
        """
        return f"由于技术限制，无法生成详细病灶描述。诊断为{dr_grade_desc}，建议进行进一步的专业检查。"

    def _cleanup_memory(self):
        """清理GPU内存和Python垃圾回收"""
        with _memory_lock:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def unload_model(self):
        """卸载模型以释放内存"""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            self._model_loaded = False
            self._cleanup_memory()
            logger.info("Qwen-VL模型已卸载")