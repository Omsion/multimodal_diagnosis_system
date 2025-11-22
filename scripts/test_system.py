#!/usr/bin/env python3
"""
系统测试脚本

提供全面的功能测试和性能测试，确保系统稳定性和可靠性。
"""

import os
import sys
import time
import json
import asyncio
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw
import numpy as np
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemTester:
    """系统测试器"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []

    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """记录测试结果"""
        status = "PASS" if passed else "FAIL"
        logger.info(f"[{status}] {test_name}: {details}")
        self.test_results.append({
            "name": test_name,
            "passed": passed,
            "details": details,
            "timestamp": time.time()
        })

    def create_test_image(self, size: tuple = (224, 224), color: str = "red") -> bytes:
        """创建测试图像"""
        try:
            # 创建彩色测试图像
            img = Image.new('RGB', size, color=color)
            draw = ImageDraw.Draw(img)

            # 添加一些图形元素模拟眼底图像
            center_x, center_y = size[0] // 2, size[1] // 2
            radius = min(size) // 4

            # 绘制圆形模拟视盘
            draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], outline="white", width=2)

            # 添加一些点状出血模拟
            for _ in range(10):
                x = np.random.randint(0, size[0])
                y = np.random.randint(0, size[1])
                draw.ellipse([x-2, y-2, x+2, y+2], fill="darkred")

            # 转换为bytes
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)

            return img_bytes.getvalue()

        except Exception as e:
            logger.error(f"创建测试图像失败: {e}")
            raise

    def test_health_check(self) -> bool:
        """测试健康检查接口"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)

            if response.status_code == 200:
                data = response.json()
                required_fields = ["status", "timestamp", "models_loaded", "gpu_available"]
                all_fields_present = all(field in data for field in required_fields)

                self.log_test_result(
                    "健康检查接口",
                    all_fields_present,
                    f"状态: {data.get('status', 'unknown')}"
                )
                return all_fields_present
            else:
                self.log_test_result("健康检查接口", False, f"HTTP {response.status_code}")
                return False

        except Exception as e:
            self.log_test_result("健康检查接口", False, str(e))
            return False

    def test_image_diagnosis(self, image_bytes: bytes) -> bool:
        """测试图像诊断接口"""
        try:
            files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
            data = {
                "enable_rag": True,
                "enable_cot": True
            }

            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/diagnose",
                files=files,
                data=data,
                timeout=60
            )
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                required_fields = [
                    "trace_id", "dr_grade", "dr_grade_desc", "confidence",
                    "lesion_description", "structured_report", "processing_time"
                ]

                all_fields_present = all(field in result for field in required_fields)
                processing_time = end_time - start_time

                self.log_test_result(
                    "图像诊断接口",
                    all_fields_present,
                    f"DR等级: {result.get('dr_grade', 'unknown')}, "
                    f"处理时间: {processing_time:.2f}s"
                )
                return all_fields_present
            else:
                self.log_test_result("图像诊断接口", False, f"HTTP {response.status_code}")
                return False

        except Exception as e:
            self.log_test_result("图像诊断接口", False, str(e))
            return False

    def test_invalid_image(self) -> bool:
        """测试无效图像处理"""
        try:
            # 测试非图像文件
            files = {"file": ("test.txt", b"not an image", "text/plain")}

            response = requests.post(
                f"{self.base_url}/diagnose",
                files=files,
                timeout=10
            )

            # 应该返回400错误
            success = response.status_code == 400
            self.log_test_result(
                "无效图像处理",
                success,
                f"返回状态码: {response.status_code}"
            )
            return success

        except Exception as e:
            self.log_test_result("无效图像处理", False, str(e))
            return False

    def test_large_image(self) -> bool:
        """测试大图像处理"""
        try:
            # 创建大图像（超过限制）
            large_image = self.create_test_image(size=(5000, 5000))
            files = {"file": ("large.jpg", large_image, "image/jpeg")}

            response = requests.post(
                f"{self.base_url}/diagnose",
                files=files,
                timeout=10
            )

            # 应该返回413错误（文件过大）或处理成功
            success = response.status_code in [200, 413]
            self.log_test_result(
                "大图像处理",
                success,
                f"返回状态码: {response.status_code}"
            )
            return success

        except Exception as e:
            self.log_test_result("大图像处理", False, str(e))
            return False

    def test_concurrent_requests(self, num_requests: int = 5) -> bool:
        """测试并发请求"""
        try:
            import threading
            import queue

            results = queue.Queue()

            def worker():
                try:
                    test_image = self.create_test_image()
                    success = self.test_image_diagnosis(test_image)
                    results.put(success)
                except Exception as e:
                    logger.error(f"并发请求失败: {e}")
                    results.put(False)

            # 启动多个线程
            threads = []
            start_time = time.time()

            for _ in range(num_requests):
                thread = threading.Thread(target=worker)
                thread.start()
                threads.append(thread)

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            end_time = time.time()

            # 收集结果
            success_count = 0
            while not results.empty():
                if results.get():
                    success_count += 1

            success_rate = success_count / num_requests
            total_time = end_time - start_time

            success = success_rate >= 0.8  # 至少80%成功率
            self.log_test_result(
                "并发请求测试",
                success,
                f"成功率: {success_rate:.1%}, 总时间: {total_time:.2f}s"
            )
            return success

        except Exception as e:
            self.log_test_result("并发请求测试", False, str(e))
            return False

    def test_performance(self, num_tests: int = 10) -> Dict[str, Any]:
        """性能测试"""
        try:
            times = []
            success_count = 0

            for i in range(num_tests):
                logger.info(f"性能测试进度: {i+1}/{num_tests}")

                test_image = self.create_test_image()

                start_time = time.time()
                success = self.test_image_diagnosis(test_image)
                end_time = time.time()

                if success:
                    success_count += 1
                    times.append(end_time - start_time)

                # 短暂休息避免服务器过载
                time.sleep(1)

            if times:
                performance_stats = {
                    "total_tests": num_tests,
                    "success_count": success_count,
                    "success_rate": success_count / num_tests,
                    "avg_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "std_time": np.std(times),
                    "p95_time": np.percentile(times, 95),
                    "p99_time": np.percentile(times, 99)
                }

                self.log_test_result(
                    "性能测试",
                    True,
                    f"平均响应时间: {performance_stats['avg_time']:.2f}s, "
                    f"成功率: {performance_stats['success_rate']:.1%}"
                )
                return performance_stats
            else:
                self.log_test_result("性能测试", False, "没有成功的测试")
                return {}

        except Exception as e:
            self.log_test_result("性能测试", False, str(e))
            return {}

    def run_all_tests(self) -> bool:
        """运行所有测试"""
        logger.info("开始运行系统测试...")

        # 基础功能测试
        tests = [
            ("健康检查", self.test_health_check),
            ("图像诊断", lambda: self.test_image_diagnosis(self.create_test_image())),
            ("无效图像处理", self.test_invalid_image),
            ("大图像处理", self.test_large_image),
            ("并发请求", lambda: self.test_concurrent_requests(3)),
        ]

        all_passed = True
        for test_name, test_func in tests:
            logger.info(f"运行测试: {test_name}")
            try:
                result = test_func()
                all_passed = all_passed and result
            except Exception as e:
                logger.error(f"测试 {test_name} 执行失败: {e}")
                self.log_test_result(test_name, False, str(e))
                all_passed = False

            time.sleep(2)  # 测试间隔

        # 性能测试（可选）
        logger.info("运行性能测试...")
        try:
            performance_stats = self.test_performance(5)
            if performance_stats:
                logger.info(f"性能统计: {json.dumps(performance_stats, indent=2)}")
        except Exception as e:
            logger.error(f"性能测试失败: {e}")

        # 生成测试报告
        self.generate_test_report()

        return all_passed

    def generate_test_report(self):
        """生成测试报告"""
        report = {
            "test_time": time.time(),
            "total_tests": len(self.test_results),
            "passed_tests": sum(1 for r in self.test_results if r["passed"]),
            "failed_tests": sum(1 for r in self.test_results if not r["passed"]),
            "success_rate": sum(1 for r in self.test_results if r["passed"]) / len(self.test_results),
            "details": self.test_results
        }

        # 保存到文件
        report_file = project_root / "test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"测试报告已保存到: {report_file}")
        logger.info(f"测试完成: {report['passed_tests']}/{report['total_tests']} 通过")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="多模态DR诊断系统测试工具")
    parser.add_argument("--url", default="http://localhost:8000", help="服务地址")
    parser.add_argument("--test", choices=[
        "health", "diagnosis", "invalid", "large", "concurrent", "performance", "all"
    ], default="all", help="要执行的测试类型")
    parser.add_argument("--performance-tests", type=int, default=5, help="性能测试次数")

    args = parser.parse_args()

    # 创建测试器
    tester = SystemTester(args.url)

    try:
        if args.test == "all":
            success = tester.run_all_tests()
        elif args.test == "health":
            success = tester.test_health_check()
        elif args.test == "diagnosis":
            test_image = tester.create_test_image()
            success = tester.test_image_diagnosis(test_image)
        elif args.test == "invalid":
            success = tester.test_invalid_image()
        elif args.test == "large":
            success = tester.test_large_image()
        elif args.test == "concurrent":
            success = tester.test_concurrent_requests()
        elif args.test == "performance":
            tester.test_performance(args.performance_tests)
            success = True

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("测试被中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()