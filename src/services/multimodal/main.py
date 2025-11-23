
import sys
import os
from pathlib import Path
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import logging
from datetime import datetime

from src.core.vision.processors import QwenVLModule
from src.core.llm.loader import load_r1_7b_llm_as_langchain_component
from src.core.rag.chain_builder import get_retriever, create_rag_chain
from src.config.settings import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multimodal_service")

app = FastAPI(
    title="Multimodal Analysis Service",
    description="负责病灶描述和RAG推理的微服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例
lesion_describer = None
rag_chain = None

@app.on_event("startup")
async def startup_event():
    global lesion_describer, rag_chain
    logger.info("正在加载多模态模型...")
    
    # 加载Qwen-VL
    try:
        lesion_describer = QwenVLModule(settings.QWEN_VL_MODEL_PATH)
        lesion_describer._load_model()
        logger.info("Qwen-VL模型加载成功")
    except Exception as e:
        logger.error(f"Qwen-VL模型加载失败: {e}")

    # 加载RAG链
    try:
        llm = load_r1_7b_llm_as_langchain_component()
        if llm:
            retriever = get_retriever()
            rag_chain = create_rag_chain(llm, retriever)
            logger.info("RAG链构建成功")
        else:
            logger.error("LLM加载失败，无法构建RAG链")
    except Exception as e:
        logger.error(f"RAG链构建失败: {e}")

class LesionAnalysisRequest(BaseModel):
    dr_grade_desc: str

class RagReasoningRequest(BaseModel):
    dr_grade_desc: str
    lesion_description: str

@app.post("/analyze_lesion")
async def analyze_lesion(
    dr_grade_desc: str = Body(...),
    file: UploadFile = File(...)
):
    global lesion_describer
    if not lesion_describer:
        raise HTTPException(status_code=503, detail="Qwen-VL模型未初始化")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        description = lesion_describer.generate_description(image, dr_grade_desc)
        
        return {
            "lesion_description": description,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"病灶分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag_reasoning")
async def rag_reasoning(request: RagReasoningRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG链未初始化")

    try:
        rag_input = {
            "dr_grade_desc": request.dr_grade_desc,
            "lesion_description": request.lesion_description
        }
        
        # 调用LangChain
        report_str = rag_chain.invoke(rag_input)
        
        # 尝试解析JSON
        try:
            structured_report = json.loads(report_str)
        except json.JSONDecodeError:
            # 如果LLM返回的不是纯JSON，尝试提取或作为纯文本返回
            structured_report = {
                "cot_reasoning": report_str,
                "recommendations": [],
                "traceability": "解析失败，返回原始文本"
            }

        return {
            "structured_report": structured_report,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"RAG推理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "qwen_loaded": lesion_describer is not None and lesion_describer._model_loaded,
        "rag_loaded": rag_chain is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
