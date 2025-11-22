# main.py
"""
多模态DR智能诊断系统 FastAPI 主服务

该系统结合计算机视觉和大型语言模型，提供糖尿病视网膜病变的智能诊断服务。
包含以下功能：
- DR严重程度自动分级
- 病灶特征智能描述
- 基于知识库的RAG推理
- 完整的诊断报告生成
"""

import os
import io
import json
import uuid
import asyncio
import concurrent.futures
import threading
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from PIL import Image
import psutil
import GPUtil

from src.config.settings import settings
from src.core.vision.processors import DRGradingModule, QwenVLModule
from src.core.llm.loader import load_r1_7b_llm_as_langchain_component
from src.core.rag.chain_builder import get_retriever, create_rag_chain
import logging

# 配置结构化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] - %(message)s'
)
logger = logging.getLogger(__name__)

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
    models_loaded: bool = Field(..., description="模型是否加载")
    gpu_available: bool = Field(..., description="GPU是否可用")
    memory_usage: Dict[str, float] = Field(..., description="内存使用情况")
    active_requests: int = Field(..., description="当前活跃请求数")

class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误信息")
    trace_id: str = Field(..., description="追踪ID")
    timestamp: str = Field(..., description="错误时间")

# --- 全局状态管理 ---
class SystemState:
    """系统状态管理类"""

    def __init__(self):
        self.dr_grader: Optional[DRGradingModule] = None
        self.lesion_describer: Optional[QwenVLModule] = None
        self.rag_chain = None
        self.active_requests: Dict[str, Dict] = {}
        self.request_lock = threading.Lock()
        self._initialization_complete = False

    def add_request(self, trace_id: str) -> None:
        """添加活跃请求"""
        with self.request_lock:
            self.active_requests[trace_id] = {
                "start_time": datetime.now(),
                "status": "processing"
            }

    def complete_request(self, trace_id: str, success: bool = True) -> float:
        """完成请求并返回处理时间"""
        with self.request_lock:
            if trace_id in self.active_requests:
                start_time = self.active_requests[trace_id]["start_time"]
                processing_time = (datetime.now() - start_time).total_seconds()
                self.active_requests[trace_id]["status"] = "completed" if success else "failed"
                self.active_requests[trace_id]["processing_time"] = processing_time
                return processing_time
            return 0.0

    def cleanup_requests(self) -> None:
        """清理过期的请求记录"""
        with self.request_lock:
            current_time = datetime.now()
            expired_traces = [
                trace_id for trace_id, data in self.active_requests.items()
                if (current_time - data["start_time"]).total_seconds() > 3600  # 1小时超时
            ]
            for trace_id in expired_traces:
                del self.active_requests[trace_id]

    @property
    def is_ready(self) -> bool:
        """检查系统是否准备就绪"""
        return (
            self._initialization_complete and
            self.dr_grader is not None and
            self.lesion_describer is not None and
            self.rag_chain is not None
        )

    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        memory = psutil.virtual_memory()
        gpu_metrics = {}

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_metrics = {
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_utilization": gpu.load * 100
                }
        except:
            pass

        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "active_requests": len(self.active_requests),
            **gpu_metrics
        }

# 全局系统状态实例
system_state = SystemState()

# --- 模型加载和初始化 ---
async def load_models_parallel() -> bool:
    """
    并行加载所有模型

    Returns:
        bool: 是否成功加载所有模型
    """
    logger.info("开始并行加载系统模型...")

    def load_dr_grader():
        try:
            system_state.dr_grader = DRGradingModule(settings.RESNET_MODEL_PATH)
            logger.info("DR分级模型加载成功")
            return True
        except Exception as e:
            logger.error(f"DR分级模型加载失败: {e}")
            return False

    def load_lesion_describer():
        try:
            system_state.lesion_describer = QwenVLModule(settings.QWEN_VL_MODEL_PATH)
            logger.info("病灶描述模型加载成功")
            return True
        except Exception as e:
            logger.error(f"病灶描述模型加载失败: {e}")
            return False

    def load_rag_chain():
        try:
            llm = load_r1_7b_llm_as_langchain_component()
            retriever = get_retriever()
            system_state.rag_chain = create_rag_chain(llm, retriever)
            logger.info("RAG链构建成功")
            return True
        except Exception as e:
            logger.error(f"RAG链构建失败: {e}")
            return False

    # 使用线程池并行执行模型加载
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # 提交所有加载任务
            dr_future = executor.submit(load_dr_grader)
            qwen_future = executor.submit(load_lesion_describer)
            rag_future = executor.submit(load_rag_chain)

            # 等待所有任务完成
            dr_success = dr_future.result()
            qwen_success = qwen_future.result()
            rag_success = rag_future.result()

            if dr_success and qwen_success and rag_success:
                logger.info("所有模型加载完成")
                return True
            else:
                logger.error("部分模型加载失败")
                return False

    except Exception as e:
        logger.error(f"并行模型加载过程中发生错误: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("正在启动多模态DR诊断系统...")

    # 并行加载模型
    success = await load_models_parallel()
    if not success:
        logger.error("系统初始化失败，部分功能可能不可用")

    system_state._initialization_complete = True

    # 启动后台清理任务
    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield

    # 关闭时执行
    logger.info("正在关闭系统...")
    cleanup_task.cancel()

    # 清理资源
    if system_state.lesion_describer:
        system_state.lesion_describer.unload_model()

    logger.info("系统已安全关闭")

# --- FastAPI应用创建 ---
app = FastAPI(
    title="多模态DR智能诊断系统 V3 (优化版)",
    description="基于计算机视觉和大型语言模型的糖尿病视网膜病变智能诊断系统",
    version="3.0.0",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# --- 依赖注入 ---
def get_system_state():
    """获取系统状态依赖"""
    return system_state

def validate_image(file: UploadFile = File(...)):
    """验证上传的图像文件"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="文件必须是图像格式"
        )
    return file

# --- 错误处理 ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            message=exc.detail,
            trace_id="",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
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

# --- API端点 ---
@app.post("/diagnose", response_model=DiagnosisResponse, summary="一键诊断DR图像")
async def diagnose(
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(validate_image),
    request: Optional[DiagnosisRequest] = None,
    state: SystemState = Depends(get_system_state)
):
    """
    主要诊断端点

    对上传的眼底图像进行完整的DR诊断流程：
    1. 图像验证和预处理
    2. 并行执行DR分级和病灶描述
    3. 基于RAG的知识增强推理
    4. 生成结构化诊断报告
    """
    trace_id = str(uuid.uuid4())
    request_config = request or DiagnosisRequest()

    # 检查系统状态
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="系统尚未初始化完成，请稍后再试")

    # 记录请求开始
    state.add_request(trace_id)
    logger.info(f"[{trace_id}] 收到诊断请求: {file.filename}")

    try:
        # 读取和验证图像
        contents = await file.read()

        # 图像大小限制 (防止内存攻击)
        if len(contents) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=413, detail="图像文件过大，请上传小于50MB的图像")

        # 打开并验证图像
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            # 图像尺寸验证
            if image.size[0] < 224 or image.size[1] < 224:
                raise HTTPException(status_code=400, detail="图像尺寸过小，最小224x224像素")

            logger.info(f"[{trace_id}] 图像验证通过: {image.size}")

        except Exception as e:
            logger.error(f"[{trace_id}] 图像处理失败: {e}")
            raise HTTPException(status_code=400, detail=f"无效的图像文件: {e}")

        # 并行执行视觉处理任务
        start_time = datetime.now()

        async def process_vision_tasks():
            """并行执行视觉处理任务"""
            loop = asyncio.get_event_loop()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # 并行提交DR分级和病灶描述任务
                dr_future = loop.run_in_executor(
                    executor,
                    state.dr_grader.predict,
                    image
                )
                desc_future = loop.run_in_executor(
                    executor,
                    state.lesion_describer.generate_description,
                    image,
                    ""  # 先用空字符串，后面会填充DR描述
                )

                # 等待两个任务完成
                dr_grade, confidence = await dr_future
                lesion_description = await desc_future

                return dr_grade, confidence, lesion_description

        # 执行并行视觉处理
        dr_grade, confidence, initial_lesion_description = await process_vision_tasks()
        dr_grade_desc = settings.DR_GRADES.get(dr_grade, "未知等级")

        # 重新生成更准确的病灶描述（现在有了DR分级信息）
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            lesion_description = await loop.run_in_executor(
                executor,
                state.lesion_describer.generate_description,
                image,
                dr_grade_desc
            )

        # 生成RAG增强的结构化报告
        structured_report = {}
        if request_config.enable_rag and state.rag_chain:
            try:
                rag_input = {
                    "dr_grade_desc": dr_grade_desc,
                    "lesion_description": lesion_description
                }
                report_str = await loop.run_in_executor(
                    executor,
                    state.rag_chain.invoke,
                    rag_input
                )
                structured_report = json.loads(report_str)
                logger.info(f"[{trace_id}] RAG报告生成成功")

            except json.JSONDecodeError as e:
                logger.error(f"[{trace_id}] JSON解析失败: {e}")
                structured_report = {
                    "cot_reasoning": "报告生成时JSON解析失败",
                    "recommendations": ["建议联系医生进行进一步检查"],
                    "traceability": "系统错误"
                }
            except Exception as e:
                logger.error(f"[{trace_id}] RAG处理失败: {e}")
                structured_report = {
                    "cot_reasoning": "RAG处理暂时不可用",
                    "recommendations": ["基于DR分级结果，建议进行相应级别的医疗咨询"],
                    "traceability": "系统降级处理"
                }
        else:
            # 不使用RAG时的简化报告
            structured_report = {
                "cot_reasoning": f"基于图像分析，诊断为{dr_grade_desc}，置信度{confidence:.3f}。{lesion_description}",
                "recommendations": [
                    f"{dr_grade_desc}标准处理流程",
                    "定期复查眼底情况",
                    "控制血糖血压"
                ],
                "traceability": "基于AI视觉分析"
            }

        # 计算处理时间
        processing_time = state.complete_request(trace_id, success=True)

        # 获取系统状态
        system_metrics = state.get_system_metrics()
        system_status = {
            "cpu_usage": f"{system_metrics['cpu_percent']:.1f}%",
            "memory_usage": f"{system_metrics['memory_percent']:.1f}%",
            "active_requests": system_metrics['active_requests']
        }

        # 添加后台清理任务
        background_tasks.add_task(state.cleanup_requests)

        logger.info(f"[{trace_id}] 诊断流程完成，耗时 {processing_time:.2f}s")

        return DiagnosisResponse(
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(),
            dr_grade=dr_grade,
            dr_grade_desc=dr_grade_desc,
            confidence=confidence,
            lesion_description=lesion_description,
            structured_report=structured_report,
            processing_time=processing_time,
            system_status=system_status
        )

    except HTTPException:
        # 重新抛出HTTP异常
        state.complete_request(trace_id, success=False)
        raise

    except Exception as e:
        # 处理其他异常
        logger.error(f"[{trace_id}] 诊断过程中发生未预期错误: {e}")
        state.complete_request(trace_id, success=False)
        raise HTTPException(
            status_code=500,
            detail=f"诊断过程中发生错误: {str(e)}"
        )

@app.get("/health", response_model=HealthCheck, summary="系统健康检查")
async def health_check(state: SystemState = Depends(get_system_state)):
    """
    系统健康检查端点

    Returns:
        HealthCheck: 包含系统状态的详细信息
    """
    metrics = state.get_system_metrics()

    return HealthCheck(
        status="healthy" if state.is_ready else "initializing",
        timestamp=datetime.now().isoformat(),
        models_loaded=state.is_ready,
        gpu_available=torch.cuda.is_available() if 'torch' in globals() else False,
        memory_usage={
            "used_percent": metrics["memory_percent"],
            "available_gb": metrics["memory_available_gb"]
        },
        active_requests=metrics["active_requests"]
    )

@app.get("/metrics", summary="系统性能指标")
async def get_metrics(state: SystemState = Depends(get_system_state)):
    """获取详细系统性能指标"""
    return state.get_system_metrics()

@app.post("/reload-knowledge-base", summary="重新加载知识库")
async def reload_knowledge_base(background_tasks: BackgroundTasks, state: SystemState = Depends(get_system_state)):
    """
    重新加载知识库

    用于更新知识库内容而无需重启整个系统
    """
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="系统尚未初始化完成")

    try:
        # 异步重新加载知识库
        async def reload_rag():
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, get_retriever)  # 这会触发重新创建

                # 重新构建RAG链
                llm = load_r1_7b_llm_as_langchain_component()
                retriever = get_retriever()
                state.rag_chain = create_rag_chain(llm, retriever)

        background_tasks.add_task(reload_rag)

        return {"message": "知识库重新加载已启动", "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"知识库重新加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"知识库重新加载失败: {e}")

# --- 后台任务 ---
async def periodic_cleanup():
    """定期清理任务"""
    while True:
        try:
            await asyncio.sleep(300)  # 每5分钟执行一次
            system_state.cleanup_requests()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"定期清理任务失败: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 检查必要目录
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./temp", exist_ok=True)

    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None,  # 使用自定义日志配置
        workers=1,  # 由于使用全局状态，只能使用单个worker
        limit_concurrency=10,  # 限制并发请求数
        timeout_keep_alive=30,
        access_log=True
    )