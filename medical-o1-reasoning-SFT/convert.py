import json
import random
from tqdm import tqdm

# --- 配置区 ---
# 定义输入文件路径
input_file_path = 'data/medical-o1-reasoning-SFT/medical_o1_sft_Chinese.json'
# 定义抽样后的样本数量
SAMPLE_SIZE = 1000
# 定义输出文件路径 (文件名中自动包含样本数量)
output_file_path = f'medical_sharegpt_format_sampled_{SAMPLE_SIZE}.json'


try:
    # 1. 读取原始JSON数据
    print(f"Reading original data from '{input_file_path}'...")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    print(f"Successfully loaded {total_items} items.")

    # 2. 对数据集进行随机抽样
    # 如果原始数据量小于等于设定的样本量，则使用全部数据
    if total_items <= SAMPLE_SIZE:
        print(f"Dataset size ({total_items}) is less than or equal to the sample size ({SAMPLE_SIZE}). Using all data.")
        sampled_data = data
    else:
        print(f"Randomly sampling {SAMPLE_SIZE} items from the dataset...")
        sampled_data = random.sample(data, SAMPLE_SIZE)

    # 3. 创建并填充输出结构
    output_data = []
    actual_sample_size = len(sampled_data)
    print(f"Starting conversion for {actual_sample_size} sampled items...")

    for item in tqdm(sampled_data, desc="Processing items"):
        # 使用 .get() 方法确保即使字段缺失也不会报错
        question = item.get("Question", "")
        complex_cot = item.get("Complex_CoT", "")
        response = item.get("Response", "")

        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": f"<think>{complex_cot}</think> {response}"
                }
            ],
            "system": "您是一位医学专家，在临床推理、诊断和治疗计划方面拥有丰富的知识。请回答以下医学问题。"
        }
        output_data.append(conversation)

    # 4. 将转换后的数据写入新的JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"\nConversion completed successfully!")
    print(f"{actual_sample_size} sampled items saved to '{output_file_path}'")

except FileNotFoundError:
    print(f"Error: The file '{input_file_path}' was not found. Please check the file path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")