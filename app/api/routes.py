
import io
import uuid
import logging
import os
from datetime import datetime
from typing import Optional
import asyncio

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
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
        loop = asyncio.get_event_loop()

        # 1. DR Grading
        t1 = datetime.now()
        dr_grade, confidence, dr_grade_desc = await loop.run_in_executor(
            None, diagnosis_service.predict_dr, image
        )
        t2 = datetime.now()
        logger.info(f"[{trace_id}] DR Grading complete: {dr_grade_desc} (Time: {(t2-t1).total_seconds():.2f}s)")

        # 2. Lesion Description
        try:
            t3 = datetime.now()
            lesion_description = await loop.run_in_executor(
                None, diagnosis_service.analyze_lesion, image, dr_grade_desc
            )
            t4 = datetime.now()
            logger.info(f"[{trace_id}] Lesion description complete (Time: {(t4-t3).total_seconds():.2f}s)")
        except Exception as e:
            logger.error(f"[{trace_id}] Lesion analysis failed: {e}")
            lesion_description = f"Could not generate description. Diagnosed as {dr_grade_desc}."

        # 3. RAG Reasoning
        structured_report = {}
        if request_config.enable_rag:
            try:
                t5 = datetime.now()
                structured_report = await loop.run_in_executor(
                    None, diagnosis_service.rag_reasoning, dr_grade_desc, lesion_description
                )
                t6 = datetime.now()
                logger.info(f"[{trace_id}] RAG reasoning complete (Time: {(t6-t5).total_seconds():.2f}s)")
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
        logger.info(f"[{trace_id}] Total processing time: {processing_time:.2f}s")
        
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

@router.post("/diagnose-stream", summary="Streaming DR Diagnosis")
async def diagnose_stream(
    file: UploadFile = Depends(validate_image),
    request: Optional[DiagnosisRequest] = None,
):
    """
    流式诊断端点，使用 Server-Sent Events (SSE) 实时返回诊断进度
    """
    trace_id = str(uuid.uuid4())
    request_config = request or DiagnosisRequest()
    start_time = datetime.now()
    
    logger.info(f"[{trace_id}] Received streaming diagnosis request: {file.filename}")

    # Read file content
    file_content = await file.read()
    
    # Validate image
    try:
        image = Image.open(io.BytesIO(file_content)).convert("RGB")
        if image.size[0] < 224 or image.size[1] < 224:
            raise HTTPException(status_code=400, detail="Image size too small")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    async def event_generator():
        """生成 SSE 事件流"""
        import json
        
        try:
            loop = asyncio.get_event_loop()

            # Step 1: DR Grading
            yield f"event: progress\ndata: {json.dumps({'stage': 'dr_grading', 'message': '正在进行 DR 分级...'}, ensure_ascii=False)}\n\n"
            
            t1 = datetime.now()
            dr_grade, confidence, dr_grade_desc = await loop.run_in_executor(
                None, diagnosis_service.predict_dr, image
            )
            t2 = datetime.now()
            logger.info(f"[{trace_id}] DR Grading complete: {dr_grade_desc} (Time: {(t2-t1).total_seconds():.2f}s)")
            
            # 发送DR分级结果
            yield f"event: dr_grade\ndata: {json.dumps({'grade': dr_grade, 'confidence': confidence, 'description': dr_grade_desc}, ensure_ascii=False)}\n\n"

            # Step 2: Lesion Description (Streaming)
            yield f"event: progress\ndata: {json.dumps({'stage': 'lesion_analysis', 'message': 'Qwen-VL 正在分析病灶特征...'}, ensure_ascii=False)}\n\n"
            
            try:
                t3 = datetime.now()
                
                # 检查是否使用API模式
                if settings.USE_API_MODELS and hasattr(diagnosis_service.lesion_describer, 'generate_description_stream'):
                    # 使用流式API
                    reasoning_complete = False
                    full_reasoning = ""
                    full_answer = ""
                    
                    # 在线程池中执行流式生成器
                    def stream_lesion():
                        for chunk in diagnosis_service.lesion_describer.generate_description_stream(
                            image, dr_grade_desc, enable_thinking=True, thinking_budget=81920
                        ):
                            return chunk
                    
                    # 直接调用生成器（因为它已经是同步的）
                    for chunk in diagnosis_service.lesion_describer.generate_description_stream(
                        image, dr_grade_desc, enable_thinking=True, thinking_budget=81920
                    ):
                        chunk_type = chunk.get("type")
                        chunk_content = chunk.get("content", "")
                        
                        if chunk_type == "reasoning":
                            # 第一个reasoning chunk时发送stage切换事件
                            if not reasoning_complete and not full_reasoning:
                                yield f"event: progress\ndata: {json.dumps({'stage': 'thinking', 'message': 'AI 正在思考...'}, ensure_ascii=False)}\n\n"
                            
                            full_reasoning += chunk_content
                            yield f"event: lesion_reasoning\ndata: {json.dumps({'content': chunk_content}, ensure_ascii=False)}\n\n"
                            
                        elif chunk_type == "answer":
                            # 第一个answer chunk时标记reasoning完成
                            if not reasoning_complete:
                                reasoning_complete = True
                                yield f"event: progress\ndata: {json.dumps({'stage': 'answering', 'message': '生成病灶描述...'}, ensure_ascii=False)}\n\n"
                            
                            full_answer += chunk_content
                            yield f"event: lesion_answer\ndata: {json.dumps({'content': chunk_content}, ensure_ascii=False)}\n\n"
                            
                        elif chunk_type == "error":
                            yield f"event: error\ndata: {json.dumps({'message': chunk_content}, ensure_ascii=False)}\n\n"
                            full_answer = chunk_content
                            break
                    
                    lesion_description = full_answer if full_answer else "无法生成描述"
                else:
                    # 使用非流式API（本地模型或旧版API）
                    lesion_description = await loop.run_in_executor(
                        None, diagnosis_service.analyze_lesion, image, dr_grade_desc
                    )
                    yield f"event: lesion_answer\ndata: {json.dumps({'content': lesion_description}, ensure_ascii=False)}\n\n"
                
                t4 = datetime.now()
                logger.info(f"[{trace_id}] Lesion description complete (Time: {(t4-t3).total_seconds():.2f}s)")
                
            except Exception as e:
                logger.error(f"[{trace_id}] Lesion analysis failed: {e}")
                lesion_description = f"Could not generate description. Diagnosed as {dr_grade_desc}."
                yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"

            # Step 3: RAG Reasoning (Streaming)
            structured_report = {}
            if request_config.enable_rag:
                yield f"event: progress\ndata: {json.dumps({'stage': 'rag_reasoning', 'message': 'DeepSeek-R1 正在进行思维链推理...'}, ensure_ascii=False)}\n\n"
                
                try:
                    t5 = datetime.now()
                    
                    full_response = ""
                    thinking_process = ""
                    json_part = ""
                    in_json = False
                    
                    async for chunk in diagnosis_service.rag_reasoning_stream(dr_grade_desc, lesion_description):
                        full_response += chunk
                        
                        # Simple heuristic to separate Thinking from JSON
                        if "JSON Report:" in full_response and not in_json:
                            parts = full_response.split("JSON Report:", 1)
                            thinking_process = parts[0].replace("Thinking Process:", "").strip()
                            json_part = parts[1]
                            in_json = True
                            # Send the final part of thinking if any
                            yield f"event: rag_reasoning\ndata: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                        elif in_json:
                            json_part += chunk
                        else:
                            # Still in thinking process
                            yield f"event: rag_reasoning\ndata: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

                    t6 = datetime.now()
                    logger.info(f"[{trace_id}] RAG reasoning complete (Time: {(t6-t5).total_seconds():.2f}s)")
                    
                    # Parse the accumulated JSON part
                    import re
                    try:
                        # Clean up JSON string (remove markdown code blocks if present)
                        clean_json = json_part.strip()
                        if clean_json.startswith("```json"):
                            clean_json = clean_json[7:]
                        if clean_json.endswith("```"):
                            clean_json = clean_json[:-3]
                        
                        structured_report = json.loads(clean_json)
                        structured_report["cot_reasoning"] = thinking_process # Add thinking process to report
                    except json.JSONDecodeError:
                         structured_report = {
                            "cot_reasoning": thinking_process,
                            "recommendations": [],
                            "traceability": "Failed to parse JSON report"
                        }

                    yield f"event: rag_result\ndata: {json.dumps(structured_report, ensure_ascii=False)}\n\n"
                    
                except Exception as e:
                    logger.error(f"[{trace_id}] RAG failed: {e}")
                    structured_report = {
                        "cot_reasoning": "RAG service temporarily unavailable",
                        "recommendations": ["Consult a doctor"],
                        "traceability": "System degraded"
                    }
                    yield f"event: rag_result\ndata: {json.dumps(structured_report, ensure_ascii=False)}\n\n"
            else:
                structured_report = {
                    "cot_reasoning": f"Based on AI visual analysis, diagnosed as {dr_grade_desc}.",
                    "recommendations": ["Regular checkup"],
                    "traceability": "Visual model output"
                }
                yield f"event: rag_result\ndata: {json.dumps(structured_report, ensure_ascii=False)}\n\n"

            # Final completion event
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[{trace_id}] Total streaming processing time: {processing_time:.2f}s")
            
            yield f"event: complete\ndata: {json.dumps({'trace_id': trace_id, 'processing_time': processing_time}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"[{trace_id}] Streaming diagnosis failed: {e}")
            yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


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
        # Use interval=None to avoid blocking. The first call might return 0.0 or a previous value, 
        # but it won't block the event loop.
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # GPU (if available) - Run in thread pool to avoid blocking
        gpu_info = {"available": False}
        
        async def get_gpu_info_async():
            try:
                loop = asyncio.get_event_loop()
                # Run GPUtil.getGPUs() in a separate thread
                gpus = await loop.run_in_executor(None, GPUtil.getGPUs)
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    # Calculate memory percentage from numeric values
                    gpu_memory_percent = f"{(gpu.memoryUsed / gpu.memoryTotal * 100):.1f}%"
                    return {
                        "available": True,
                        "load": f"{gpu.load * 100:.1f}%",
                        "memory_percent": gpu_memory_percent,
                        "temperature": f"{gpu.temperature}°C" if hasattr(gpu, 'temperature') else "N/A"
                    }
            except Exception as e:
                logger.debug(f"GPU info check failed: {e}")
            return {"available": False}

        # Add timeout for GPU check (e.g., 2 seconds)
        try:
            gpu_info = await asyncio.wait_for(get_gpu_info_async(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("GPU metrics check timed out")
            gpu_info = {"available": False, "error": "Timeout"}
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
        
        # Disk
        disk = psutil.disk_usage('/')
        
        model_info = diagnosis_service.get_loaded_models_info()
        
        return {
            "system_mode": "API Mode" if settings.USE_API_MODELS else "Local Mode",
            "cpu_usage": f"{cpu_percent}%",
            "memory_used": f"{memory.percent}%",
            "memory_available": f"{memory.available / (1024**3):.1f}GB",
            "disk_usage": f"{disk.percent}%",
            "gpu_load": gpu_info.get("load", "N/A") if gpu_info["available"] else "N/A",
            "gpu_memory": gpu_info.get("memory_percent", "N/A") if gpu_info["available"] else "N/A",
            "uptime": "Online",
            "local_models_loaded": model_info["local_models_loaded"],
            "api_services_active": model_info["api_services_active"]
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
