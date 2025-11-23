
import io
import uuid
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import psutil
import GPUtil

from app.api.schemas import DiagnosisRequest, DiagnosisResponse, HealthCheck, ErrorResponse
from app.services.diagnosis import diagnosis_service
from app.config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)

def validate_image(file: UploadFile = File(...)):
    """Validate uploaded image file"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    return file

@router.post("/diagnose", response_model=DiagnosisResponse, summary="One-click DR Diagnosis")
async def diagnose(
    file: UploadFile = Depends(validate_image),
    request: Optional[DiagnosisRequest] = None,
):
    trace_id = str(uuid.uuid4())
    request_config = request or DiagnosisRequest()
    start_time = datetime.now()
    
    logger.info(f"[{trace_id}] Received diagnosis request: {file.filename}")

    # Read file content
    file_content = await file.read()
    
    # Validate image
    try:
        image = Image.open(io.BytesIO(file_content)).convert("RGB")
        if image.size[0] < 224 or image.size[1] < 224:
            raise HTTPException(status_code=400, detail="Image size too small")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        # 1. DR Grading
        dr_grade, confidence, dr_grade_desc = diagnosis_service.predict_dr(image)
        logger.info(f"[{trace_id}] DR Grading complete: {dr_grade_desc}")

        # 2. Lesion Description
        try:
            lesion_description = diagnosis_service.analyze_lesion(image, dr_grade_desc)
            logger.info(f"[{trace_id}] Lesion description complete")
        except Exception as e:
            logger.error(f"[{trace_id}] Lesion analysis failed: {e}")
            lesion_description = f"Could not generate description. Diagnosed as {dr_grade_desc}."

        # 3. RAG Reasoning
        structured_report = {}
        if request_config.enable_rag:
            try:
                structured_report = diagnosis_service.rag_reasoning(dr_grade_desc, lesion_description)
                logger.info(f"[{trace_id}] RAG reasoning complete")
            except Exception as e:
                logger.error(f"[{trace_id}] RAG failed: {e}")
                structured_report = {
                    "cot_reasoning": "RAG service temporarily unavailable",
                    "recommendations": ["Consult a doctor"],
                    "traceability": "System degraded"
                }
        else:
            structured_report = {
                "cot_reasoning": f"Based on AI visual analysis, diagnosed as {dr_grade_desc}.",
                "recommendations": ["Regular checkup"],
                "traceability": "Visual model output"
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
                "service": "healthy",
                "cpu_usage": f"{psutil.cpu_percent()}%"
            }
        )

    except Exception as e:
        logger.error(f"[{trace_id}] Diagnosis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthCheck)
async def health_check():
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_status={
            "dr_grader": diagnosis_service.dr_grader is not None,
            "lesion_describer": diagnosis_service.lesion_describer is not None,
            "rag_chain": diagnosis_service.rag_chain is not None
        }
    )

@router.get("/metrics", summary="System Metrics", description="Get real-time system performance metrics")
async def get_metrics():
    """Get system metrics including CPU, memory, GPU usage"""
    try:
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU (if available)
        gpu_info = {"available": False}
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_info = {
                    "available": True,
                    "load": f"{gpu.load * 100:.1f}%",
                    "memory_used": f"{gpu.memoryUsed}MB",
                    "memory_total": f"{gpu.memoryTotal}MB",
                    "temperature": f"{gpu.temperature}°C"
                }
        except:
            pass
        
        # Disk
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_usage": f"{cpu_percent}%",
            "memory_used": f"{memory.percent}%",
            "memory_available": f"{memory.available / (1024**3):.1f}GB",
            "disk_usage": f"{disk.percent}%",
            "gpu_load": gpu_info.get("load", "N/A") if gpu_info["available"] else "N/A",
            "gpu_memory": f"{(gpu_info.get('memory_used', 0) / gpu_info.get('memory_total', 1) * 100):.1f}%" if gpu_info["available"] else "N/A",
            "uptime": "Online",
            "models_loaded": sum([
                diagnosis_service.dr_grader is not None,
                diagnosis_service.lesion_describer is not None,
                diagnosis_service.rag_chain is not None
            ])
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")

@router.post("/reload-knowledge-base", summary="Reload Knowledge Base", description="Reload the RAG knowledge base from disk")
async def reload_knowledge_base():
    """Reload the RAG vector database from knowledge base files"""
    try:
        logger.info("Starting knowledge base reload...")
        
        # Check if vector DB exists
        if not os.path.exists(settings.VECTOR_DB_PATH):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "向量数据库不存在，请先运行 init_vector_db.py 初始化"
                }
            )
        
        # Reinitialize the diagnosis service (which will reload the RAG chain)
        try:
            diagnosis_service.initialize()
            logger.info("Knowledge base reloaded successfully")
            
            return {
                "status": "success",
                "message": "知识库重新加载成功",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to reload knowledge base: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"重新加载失败: {str(e)}"
                }
            )
            
    except Exception as e:
        logger.error(f"Knowledge base reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
