
import asyncio
import time
from app.api.routes import get_metrics
from app.services.diagnosis import diagnosis_service

# Mock diagnosis service to avoid initialization errors if not running full app
class MockService:
    def __init__(self):
        self.dr_grader = "mock"
        self.lesion_describer = "mock"
        self.rag_chain = "mock"

diagnosis_service.dr_grader = "mock"
diagnosis_service.lesion_describer = "mock"
diagnosis_service.rag_chain = "mock"

async def test_metrics():
    print("Testing get_metrics()...")
    start = time.time()
    try:
        metrics = await get_metrics()
        duration = time.time() - start
        print(f"get_metrics() returned in {duration:.4f}s")
        print("Metrics:", metrics)
        
        if duration > 3.0:
            print("FAIL: Metrics took too long!")
        else:
            print("PASS: Metrics returned quickly.")
            
        if "error" in metrics:
             print("NOTE: GPU check timed out or failed (expected if no GPU or driver issue), but main call succeeded.")
             
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test_metrics())
