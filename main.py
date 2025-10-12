# main.py
import uvicorn
import io
import json
import uuid
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from settings import settings
from vision_processors import DRGradingModule, QwenVLModule
from llm_loader import load_r1_7b_llm_as_langchain_component
from rag_chain_builder import get_retriever, create_rag_chain
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI应用和数据模型 ---
app = FastAPI(title="多模态DR智能诊断系统 V2 (LangChain集成版)")

class DiagnosisResponse(BaseModel):
    trace_id: str
    dr_grade: int
    dr_grade_desc: str
    confidence: float
    lesion_description: str
    structured_report: dict

# --- 全局资源加载 ---
# 使用全局变量来持有加载好的模型和链，避免重复加载
dr_grader = None
lesion_describer = None
rag_chain = None

@app.on_event("startup")
def startup_event():
    """在应用启动时加载所有模型和RAG链"""
    global dr_grader, lesion_describer, rag_chain
    logger.info("开始加载系统资源...")
    dr_grader = DRGradingModule(settings.RESNET_MODEL_PATH)
    lesion_describer = QwenVLModule(settings.QWEN_VL_MODEL_PATH)
    llm = load_r1_7b_llm_as_langchain_component()
    retriever = get_retriever()
    rag_chain = create_rag_chain(llm, retriever)
    logger.info("系统资源加载完成，应用已就绪。")

@app.post("/diagnose", response_model=DiagnosisResponse, summary="一键诊断DR图像")
async def diagnose(file: UploadFile = File(..., description="上传DR眼底图像")):
    trace_id = str(uuid.uuid4())
    logger.info(f"[{trace_id}] 收到新的诊断请求: {file.filename}")
    
    # 读取和验证图像
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.error(f"[{trace_id}] 图像读取或验证失败: {e}")
        raise HTTPException(status_code=400, detail=f"无效的图像文件: {e}")

    # 1. 视觉处理
    logger.info(f"[{trace_id}] 正在进行DR分级...")
    dr_grade, confidence = dr_grader.predict(image)
    dr_grade_desc = settings.DR_GRADES.get(dr_grade, "未知等级")
    
    logger.info(f"[{trace_id}] 正在生成病灶描述...")
    lesion_description = lesion_describer.generate_description(image, dr_grade_desc)
    
    # 2. 调用RAG链生成结构化报告
    logger.info(f"[{trace_id}] 正在调用RAG链...")
    try:
        rag_input = {
            "dr_grade_desc": dr_grade_desc,
            "lesion_description": lesion_description
        }
        report_str = rag_chain.invoke(rag_input)
        structured_report = json.loads(report_str)
    except Exception as e:
        logger.error(f"[{trace_id}] RAG链处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成诊断报告时出错: {e}")
        
    logger.info(f"[{trace_id}] 诊断流程成功完成。")
    return DiagnosisResponse(
        trace_id=trace_id,
        dr_grade=dr_grade,
        dr_grade_desc=dr_grade_desc,
        confidence=confidence,
        lesion_description=lesion_description,
        structured_report=structured_report
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)