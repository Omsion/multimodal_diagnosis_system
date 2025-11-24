
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

    async def rag_reasoning_stream(self, dr_grade_desc: str, lesion_description: str):
        """
        Perform RAG reasoning with streaming.
        Yields chunks of the response.
        """
        if not self.rag_chain:
            raise RuntimeError("RAG chain not initialized")

        rag_input = {
            "dr_grade_desc": dr_grade_desc,
            "lesion_description": lesion_description
        }
        
        async for chunk in self.rag_chain.astream(rag_input):
            yield chunk

    def rag_reasoning(self, dr_grade_desc: str, lesion_description: str) -> Dict[str, Any]:
        """
        Perform RAG reasoning (Synchronous wrapper for compatibility).
        """
        if not self.rag_chain:
            raise RuntimeError("RAG chain not initialized")

        rag_input = {
            "dr_grade_desc": dr_grade_desc,
            "lesion_description": lesion_description
        }
        
        report_str = self.rag_chain.invoke(rag_input)
        
        # Try to parse JSON from the new format (Thinking Process ... JSON Report ...)
        import re
        try:
            # Extract JSON part
            json_match = re.search(r'JSON Report:\s*(\{.*\})', report_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                structured_report = json.loads(json_str)
                
                # Extract Thinking Process
                thinking_match = re.search(r'Thinking Process:\s*(.*?)\s*JSON Report:', report_str, re.DOTALL)
                if thinking_match:
                    structured_report["cot_reasoning"] = thinking_match.group(1).strip()
            else:
                # Fallback: try to parse the whole string as JSON (old format compatibility)
                structured_report = json.loads(report_str)
                
        except (json.JSONDecodeError, AttributeError):
            structured_report = {
                "cot_reasoning": report_str,
                "recommendations": [],
                "traceability": "Failed to parse JSON, returning raw text"
            }
        
        return structured_report

    def get_loaded_models_info(self) -> Dict[str, int]:
        """
        Return counts of local models and API services.
        """
        local_count = 0
        api_count = 0

        # 1. DR Grader (Always Local)
        if self.dr_grader:
            local_count += 1

        # 2. Lesion Describer
        if self.lesion_describer:
            if settings.USE_API_MODELS:
                api_count += 1
            else:
                local_count += 1

        # 3. RAG Chain
        if self.rag_chain:
            if settings.USE_API_MODELS:
                api_count += 1
            else:
                local_count += 1
                
        return {
            "local_models_loaded": local_count,
            "api_services_active": api_count
        }

# Global instance
diagnosis_service = DiagnosisService()
