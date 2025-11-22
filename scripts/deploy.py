#!/usr/bin/env python3
"""
部署脚本

提供系统部署、健康检查和监控功能。
支持不同部署模式和配置选项。
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings, setup_logging
import requests

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """部署管理器"""

    def __init__(self):
        self.process = None
        self.config = {
            "host": settings.HOST,
            "port": settings.PORT,
            "workers": 1,
            "log_level": settings.LOG_LEVEL.value.lower(),
            "timeout": settings.REQUEST_TIMEOUT,
            "limit_concurrency": settings.MAX_CONCURRENT_REQUESTS
        }

    def check_system_requirements(self) -> bool:
        """检查系统要求"""
        logger.info("检查系统要求...")

        # 检查Python版本
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.error(f"Python版本过低: {python_version}，需要3.8+")
            return False

        # 检查必要的目录
        required_dirs = [
            settings.VECTOR_DB_PATH,
            settings.KNOWLEDGE_BASE_PATH,
            "./logs",
            "./temp"
        ]

        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # 检查模型文件
        model_files = [
            settings.RESNET_MODEL_PATH,
            settings.QWEN_VL_MODEL_PATH,
            settings.R1_7B_MODEL_PATH
        ]

        for model_file in model_files:
            if not os.path.exists(model_file):
                logger.warning(f"模型文件不存在: {model_file}")
                # 注意：不返回False，因为有些模型可能是可选的

        logger.info("系统要求检查完成")
        return True

    def install_dependencies(self) -> bool:
        """安装依赖包"""
        logger.info("安装依赖包...")

        try:
            # 安装requirements.txt中的依赖
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True)

            logger.info("依赖包安装完成")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"依赖包安装失败: {e}")
            return False

    def start_service(self, dev_mode: bool = False) -> bool:
        """启动服务"""
        logger.info("启动多模态DR诊断服务...")

        try:
            if dev_mode:
                # 开发模式：直接运行main.py
                cmd = [sys.executable, "main.py"]
            else:
                # 生产模式：使用uvicorn
                cmd = [
                    sys.executable, "-m", "uvicorn",
                    "main:app",
                    f"--host={self.config['host']}",
                    f"--port={self.config['port']}",
                    f"--workers={self.config['workers']}",
                    f"--log-level={self.config['log_level']}",
                    f"--timeout={self.config['timeout']}",
                    "--limit-concurrency", str(self.config['limit_concurrency']),
                    "--access-log"
                ]

            # 启动进程
            self.process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            logger.info(f"服务已启动，进程ID: {self.process.pid}")
            return True

        except Exception as e:
            logger.error(f"启动服务失败: {e}")
            return False

    def stop_service(self) -> bool:
        """停止服务"""
        if self.process is None:
            logger.info("服务未运行")
            return True

        logger.info("正在停止服务...")

        try:
            # 发送SIGTERM信号
            self.process.terminate()

            # 等待进程结束
            try:
                self.process.wait(timeout=30)
                logger.info("服务已正常停止")
            except subprocess.TimeoutExpired:
                logger.warning("服务未在30秒内停止，强制终止")
                self.process.kill()
                self.process.wait()

            return True

        except Exception as e:
            logger.error(f"停止服务失败: {e}")
            return False

    def health_check(self) -> bool:
        """健康检查"""
        try:
            url = f"http://{self.config['host']}:{self.config['port']}/health"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"健康检查通过: {health_data['status']}")
                return True
            else:
                logger.error(f"健康检查失败: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    def monitor_service(self, interval: int = 30) -> None:
        """监控服务状态"""
        logger.info(f"开始监控服务状态，检查间隔: {interval}秒")

        while True:
            try:
                if not self.health_check():
                    logger.error("服务健康检查失败")
                    # 这里可以添加告警或重启逻辑

                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("监控已停止")
                break
            except Exception as e:
                logger.error(f"监控过程中发生错误: {e}")
                time.sleep(5)  # 出错后短暂等待

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模态DR诊断系统部署工具")
    parser.add_argument("command", choices=["install", "start", "stop", "restart", "check", "monitor"],
                       help="要执行的命令")
    parser.add_argument("--dev", action="store_true", help="开发模式")
    parser.add_argument("--host", default=settings.HOST, help="服务器地址")
    parser.add_argument("--port", type=int, default=settings.PORT, help="服务器端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"],
                       default=settings.LOG_LEVEL.value.lower(), help="日志级别")
    parser.add_argument("--monitor-interval", type=int, default=30, help="监控间隔(秒)")

    args = parser.parse_args()

    # 创建部署管理器
    manager = DeploymentManager()

    # 更新配置
    manager.config.update({
        "host": args.host,
        "port": args.port,
        "workers": args.workers,
        "log_level": args.log_level
    })

    # 设置信号处理
    def signal_handler(signum, frame):
        logger.info("收到退出信号，正在清理...")
        manager.stop_service()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 执行命令
    try:
        if args.command == "install":
            success = (
                manager.check_system_requirements() and
                manager.install_dependencies()
            )
            sys.exit(0 if success else 1)

        elif args.command == "start":
            if not manager.check_system_requirements():
                sys.exit(1)

            success = manager.start_service(dev_mode=args.dev)
            if success and not args.dev:
                # 生产模式下启动监控
                manager.monitor_service(args.monitor_interval)

            sys.exit(0 if success else 1)

        elif args.command == "stop":
            success = manager.stop_service()
            sys.exit(0 if success else 1)

        elif args.command == "restart":
            manager.stop_service()
            time.sleep(5)  # 等待端口释放
            success = manager.start_service(dev_mode=args.dev)
            if success and not args.dev:
                manager.monitor_service(args.monitor_interval)
            sys.exit(0 if success else 1)

        elif args.command == "check":
            success = (
                manager.check_system_requirements() and
                manager.health_check()
            )
            sys.exit(0 if success else 1)

        elif args.command == "monitor":
            manager.monitor_service(args.monitor_interval)

    except KeyboardInterrupt:
        logger.info("程序被中断")
        manager.stop_service()
        sys.exit(0)
    except Exception as e:
        logger.error(f"执行命令时发生错误: {e}")
        manager.stop_service()
        sys.exit(1)

if __name__ == "__main__":
    main()