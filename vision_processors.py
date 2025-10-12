# vision_processors.py
import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torchvision.models as models
from torchvision import transforms
from typing import Tuple

from settings import settings
import logging

logger = logging.getLogger(__name__)

class DRGradingModule:
    """DR图像分级模块"""
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
    
    def _load_model(self, model_path: str) -> nn.Module:
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(settings.DR_GRADES))
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"成功加载DR分级模型: {model_path}")
        else:
            logger.warning(f"DR分级模型不存在: {model_path}，使用随机初始化")
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[int, float]:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        outputs = self.model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        grade = predicted.item()
        conf = confidence.item()
        logger.info(f"DR分级结果: 等级={grade}, 置信度={conf:.3f}")
        return grade, conf

class QwenVLModule:
    """Qwen-VL视觉语言模型模块"""
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
            logger.info(f"成功加载Qwen-VL模型: {model_path}")
            return model, processor
        except Exception as e:
            logger.error(f"加载Qwen-VL模型失败: {e}")
            return None, None
    
    def generate_description(self, image: Image.Image, dr_grade_desc: str) -> str:
        if self.model is None: return "Qwen-VL模型未加载"
        prompt = f"这是一张诊断为'{dr_grade_desc}'的眼底图，请用专业术语详细描述图中的关键病灶特征。"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        final_desc = description.replace(prompt, "").strip()
        logger.info("成功生成病灶描述")
        return final_desc