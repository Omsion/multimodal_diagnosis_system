
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class DiagnosisRequest(BaseModel):
    """Diagnosis Request Model"""
    enable_rag: bool = Field(default=True, description="Enable RAG enhanced reasoning")
    enable_cot: bool = Field(default=True, description="Enable Chain of Thought reasoning")
    max_tokens: Optional[int] = Field(default=1024, description="Max generation length")

class DiagnosisResponse(BaseModel):
    """Diagnosis Response Model"""
    trace_id: str = Field(..., description="Trace ID")
    timestamp: str = Field(..., description="Timestamp")
    dr_grade: int = Field(..., description="DR Grade (0-4)")
    dr_grade_desc: str = Field(..., description="DR Grade Description")
    confidence: float = Field(..., description="Grade Confidence")
    lesion_description: str = Field(..., description="Lesion Description")
    structured_report: Dict[str, Any] = Field(..., description="Structured Report")
    processing_time: float = Field(..., description="Processing Time (seconds)")
    system_status: Dict[str, str] = Field(..., description="System Status Info")

class HealthCheck(BaseModel):
    """Health Check Response Model"""
    status: str = Field(..., description="Service Status")
    timestamp: str = Field(..., description="Check Time")
    models_status: Dict[str, bool] = Field(..., description="Models Status")

class ErrorResponse(BaseModel):
    """Error Response Model"""
    error: str = Field(..., description="Error Type")
    message: str = Field(..., description="Error Message")
    trace_id: str = Field(..., description="Trace ID")
    timestamp: str = Field(..., description="Error Time")
