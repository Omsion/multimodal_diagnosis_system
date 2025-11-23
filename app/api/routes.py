
import io
import uuid
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import psutil

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
