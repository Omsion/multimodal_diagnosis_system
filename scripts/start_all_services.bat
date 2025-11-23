
@echo off
echo Starting All Services...

start "DR Service" cmd /k "scripts\start_dr_service.bat"
start "Multimodal Service" cmd /k "scripts\start_multimodal_service.bat"

echo Waiting for services to initialize...
timeout /t 10

start "API Gateway" cmd /k "scripts\start_gateway.bat"

echo All services started.
echo Gateway: http://localhost:8000
echo DR Service: http://localhost:8001
echo Multimodal Service: http://localhost:8002
pause
