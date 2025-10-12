# run_diagnosis.py
import requests
import argparse
import json
import time

def run_diagnosis(image_path: str, server_url: str = "http://127.0.0.1:8000/diagnose"):
    """向诊断API发送一张图片并打印结果。"""
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            print(f"正在发送图片 '{image_path}' 到 {server_url}...")
            
            start_time = time.time()
            response = requests.post(server_url, files=files, timeout=300) # 5分钟超时
            end_time = time.time()
            
            response.raise_for_status()

            print(f"\n诊断成功！耗时: {end_time - start_time:.2f} 秒\n" + "="*80)
            
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("="*80)

    except FileNotFoundError:
        print(f"错误：找不到文件 '{image_path}'")
    except requests.exceptions.RequestException as e:
        print(f"错误：请求失败 - {e}")
        if e.response:
            print("服务器返回内容:", e.response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一键运行DR图像智能诊断客户端")
    parser.add_argument("image_path", type=str, help="需要诊断的DR图像文件路径")
    args = parser.parse_args()
    
    run_diagnosis(args.image_path)