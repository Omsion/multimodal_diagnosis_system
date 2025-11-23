# main.py
"""
多模态DR智能诊断系统 FastAPI 主服务 (Gateway Mode)

该系统作为网关，协调后端微服务进行诊断：
- DR Grading Service (Port 8001): 负责DR分级
- Multimodal Service (Port 8002): 负责病灶描述和RAG推理
"""

import os
import io
import json
import uuid
import asyncio
import httpx
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import psutil

from src.config.settings import settings
import logging

# 配置结构化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# 微服务地址配置
DR_SERVICE_URL = "http://localhost:8001"
MULTIMODAL_SERVICE_URL = "http://localhost:8002"

# --- 数据模型定义 ---
class DiagnosisRequest(BaseModel):
    """诊断请求数据模型"""
    enable_rag: bool = Field(default=True, description="是否启用RAG增强推理")
    enable_cot: bool = Field(default=True, description="是否启用思维链推理")
    max_tokens: Optional[int] = Field(default=1024, description="最大生成长度")

class DiagnosisResponse(BaseModel):
    """诊断响应数据模型"""
    trace_id: str = Field(..., description="追踪ID")
    timestamp: str = Field(..., description="诊断时间戳")
    dr_grade: int = Field(..., description="DR等级 (0-4)")
    dr_grade_desc: str = Field(..., description="DR等级描述")
    confidence: float = Field(..., description="分级置信度")
    lesion_description: str = Field(..., description="病灶描述")
    structured_report: Dict[str, Any] = Field(..., description="结构化诊断报告")
    processing_time: float = Field(..., description="处理时间(秒)")
    system_status: Dict[str, str] = Field(..., description="系统状态信息")

class HealthCheck(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    timestamp: str = Field(..., description="检查时间")
    services_status: Dict[str, bool] = Field(..., description="微服务状态")

class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误信息")
    trace_id: str = Field(..., description="追踪ID")
    timestamp: str = Field(..., description="错误时间")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("正在启动多模态DR诊断系统网关...")
    
    # 检查微服务连接
    async with httpx.AsyncClient() as client:
        try:
            await client.get(f"{DR_SERVICE_URL}/health", timeout=2.0)
            logger.info("DR分级服务连接正常")
        except Exception as e:
            logger.warning(f"无法连接DR分级服务: {e}")
            
        try:
            await client.get(f"{MULTIMODAL_SERVICE_URL}/health", timeout=2.0)
            logger.info("多模态服务连接正常")
        except Exception as e:
            logger.warning(f"无法连接多模态服务: {e}")

    yield
    logger.info("系统网关已关闭")

app = FastAPI(
    title="多模态DR智能诊断系统 V3 (Gateway)",
    description="基于微服务架构的糖尿病视网膜病变智能诊断系统网关",
    version="3.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

def validate_image(file: UploadFile = File(...)):
    """验证上传的图像文件"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="文件必须是图像格式")
    return file

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="服务器内部错误",
            trace_id="",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.post("/diagnose", response_model=DiagnosisResponse, summary="一键诊断DR图像")
async def diagnose(
    file: UploadFile = Depends(validate_image),
    request: Optional[DiagnosisRequest] = None,
):
    trace_id = str(uuid.uuid4())
    request_config = request or DiagnosisRequest()
    start_time = datetime.now()
    
    logger.info(f"[{trace_id}] 收到诊断请求: {file.filename}")

    # 读取文件内容以便多次发送
    file_content = await file.read()
    
    # 验证图像
    try:
        image = Image.open(io.BytesIO(file_content))
        if image.size[0] < 224 or image.size[1] < 224:
            raise HTTPException(status_code=400, detail="图像尺寸过小")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无效图像: {e}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. 调用DR分级服务
        try:
            files = {'file': (file.filename, file_content, file.content_type)}
            dr_response = await client.post(f"{DR_SERVICE_URL}/predict_dr", files=files)
            dr_response.raise_for_status()
            dr_result = dr_response.json()
            
            dr_grade = dr_result['dr_grade']
            confidence = dr_result['confidence']
            dr_grade_desc = settings.DR_GRADES.get(dr_grade, "未知等级")
            
            logger.info(f"[{trace_id}] DR分级完成: {dr_grade_desc}")
            
        except Exception as e:
            logger.error(f"[{trace_id}] DR服务调用失败: {e}")
            raise HTTPException(status_code=503, detail=f"DR分级服务不可用: {str(e)}")

        # 2. 调用多模态服务生成病灶描述
        try:
            # 重置文件指针或重新构建files
            files = {'file': (file.filename, file_content, file.content_type)}
            data = {'dr_grade_desc': dr_grade_desc}
            
            lesion_response = await client.post(
                f"{MULTIMODAL_SERVICE_URL}/analyze_lesion",
                files=files,
                data=data
            )
            lesion_response.raise_for_status()
            lesion_result = lesion_response.json()
            lesion_description = lesion_result['lesion_description']
            
            logger.info(f"[{trace_id}] 病灶描述生成完成")
            
        except Exception as e:
            logger.error(f"[{trace_id}] 病灶分析服务调用失败: {e}")
            lesion_description = f"无法生成详细描述 (服务不可用)。诊断为{dr_grade_desc}。"

        # 3. 调用多模态服务进行RAG推理
        structured_report = {}
        if request_config.enable_rag:
            try:
                rag_payload = {
                    "dr_grade_desc": dr_grade_desc,
                    "lesion_description": lesion_description
                }
                rag_response = await client.post(
                    f"{MULTIMODAL_SERVICE_URL}/rag_reasoning",
                    json=rag_payload
                )
                rag_response.raise_for_status()
                structured_report = rag_response.json()['structured_report']
                logger.info(f"[{trace_id}] RAG推理完成")
                
            except Exception as e:
                logger.error(f"[{trace_id}] RAG服务调用失败: {e}")
                structured_report = {
                    "cot_reasoning": "RAG服务暂时不可用",
                    "recommendations": ["建议咨询医生"],
                    "traceability": "系统降级"
                }
        else:
            structured_report = {
                "cot_reasoning": f"基于AI视觉分析，诊断为{dr_grade_desc}。",
                "recommendations": ["定期复查"],
                "traceability": "视觉模型直接输出"
            }

    processing_time = (datetime.now() - start_time).total_seconds()
    
    return DiagnosisResponse(
        trace_id=trace_id,
        timestamp=datetime.now().isoformat(),
        dr_grade=dr_grade,
        dr_grade_desc=dr_grade_desc,
        confidence=confidence,
        lesion_description=lesion_description,
        structured_report=structured_report,
        processing_time=processing_time,
        system_status={
            "gateway": "healthy",
            "cpu_usage": f"{psutil.cpu_percent()}%"
        }
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    services_status = {}
    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            resp = await client.get(f"{DR_SERVICE_URL}/health")
            services_status["dr_service"] = resp.status_code == 200
        except:
            services_status["dr_service"] = False
            
        try:
            resp = await client.get(f"{MULTIMODAL_SERVICE_URL}/health")
            services_status["multimodal_service"] = resp.status_code == 200
        except:
            services_status["multimodal_service"] = False

    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        services_status=services_status
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)