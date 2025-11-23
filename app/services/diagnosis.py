
import logging
from PIL import Image
import json
from typing import Tuple, Dict, Any, Optional

from app.core.vision.processors import DRGradingModule, QwenVLModule
from app.core.llm.loader import load_r1_7b_llm_as_langchain_component
from app.core.rag.chain_builder import get_retriever, create_rag_chain
from app.config.settings import settings

logger = logging.getLogger(__name__)

class DiagnosisService:
    def __init__(self):
        self.dr_grader: Optional[DRGradingModule] = None
        self.lesion_describer: Optional[QwenVLModule] = None
        self.rag_chain = None
        self.initialized = False

    def initialize(self):
        """Initialize all models."""
        if self.initialized:
            return

        logger.info("Initializing DiagnosisService models...")
        
        if settings.USE_API_MODELS:
            logger.info("使用API模式运行 (轻量级模式)")
            self._initialize_api_models()
        else:
            logger.info("使用本地模型运行 (完整模式)")
            self._initialize_local_models()

        self.initialized = True

    def _initialize_local_models(self):
        """初始化本地模型"""
        # Load DR Grading Model
        try:
            self.dr_grader = DRGradingModule(settings.RESNET_MODEL_PATH)
            self.dr_grader._load_model()
            logger.info("DR Grading model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load DR Grading model: {e}")

        # Load Qwen-VL Model
        try:
            self.lesion_describer = QwenVLModule(settings.QWEN_VL_MODEL_PATH)
            self.lesion_describer._load_model()
            logger.info("Qwen-VL model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Qwen-VL model: {e}")

        # Load RAG Chain
        try:
            from app.core.llm.loader import load_r1_7b_llm_as_langchain_component
            llm = load_r1_7b_llm_as_langchain_component()
            if llm:
                retriever = get_retriever()
                self.rag_chain = create_rag_chain(llm, retriever)
                logger.info("RAG chain built successfully.")
            else:
                logger.error("Failed to load LLM, RAG chain cannot be built.")
        except Exception as e:
            logger.error(f"Failed to build RAG chain: {e}")

    def _initialize_api_models(self):
        """初始化API模型"""
        # Load DR Grading Model (本地保留)
        try:
            self.dr_grader = DRGradingModule(settings.RESNET_MODEL_PATH)
            self.dr_grader._load_model()
            logger.info("DR Grading model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load DR Grading model: {e}")

        # Load Qwen-VL API Client
        try:
            from app.core.api_models import QwenVLAPIClient
            self.lesion_describer = QwenVLAPIClient()
            logger.info("Qwen-VL API client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Qwen-VL API client: {e}")

        # Load DeepSeek Chat API for RAG
        try:
            from app.core.api_models import get_deepseek_chat_model
            llm = get_deepseek_chat_model()
            if llm:
                retriever = get_retriever()
                self.rag_chain = create_rag_chain(llm, retriever)
                logger.info("RAG chain with DeepSeek API built successfully.")
            else:
                logger.error("Failed to load DeepSeek API, RAG chain cannot be built.")
        except Exception as e:
            logger.error(f"Failed to build RAG chain with API: {e}")

    def predict_dr(self, image: Image.Image) -> Tuple[int, float, str]:
        """
        Predict DR grade.
        Returns: (grade, confidence, grade_description)
        """
        if not self.dr_grader:
            raise RuntimeError("DR Grading model not initialized")
        
        grade, confidence = self.dr_grader.predict(image)
        grade_desc = settings.DR_GRADES.get(grade, "Unknown Grade")
        return grade, confidence, grade_desc

    def analyze_lesion(self, image: Image.Image, dr_grade_desc: str) -> str:
        """
        Generate lesion description using Qwen-VL.
        """
        if not self.lesion_describer:
            raise RuntimeError("Qwen-VL model not initialized")
        
        return self.lesion_describer.generate_description(image, dr_grade_desc)

    def rag_reasoning(self, dr_grade_desc: str, lesion_description: str) -> Dict[str, Any]:
        """
        Perform RAG reasoning.
        """
        if not self.rag_chain:
            raise RuntimeError("RAG chain not initialized")

        rag_input = {
            "dr_grade_desc": dr_grade_desc,
            "lesion_description": lesion_description
        }
        
        report_str = self.rag_chain.invoke(rag_input)
        
        try:
            structured_report = json.loads(report_str)
        except json.JSONDecodeError:
            structured_report = {
                "cot_reasoning": report_str,
                "recommendations": [],
                "traceability": "Failed to parse JSON, returning raw text"
            }
        
        return structured_report

# Global instance
diagnosis_service = DiagnosisService()
