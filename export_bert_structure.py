import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task.BertSquad import BertSquad
from phl_code import disassemble
import torch

def main():
    # 初始化模型
    print("Initializing BERT model...")
    model = BertSquad()
    model.eval()  # 设置为评估模式

    # 创建虚拟输入
    # 注意：这里使用与BertForQuestionAnswering兼容的输入格式
    print("Creating dummy input...")
    dummy_input = (
        torch.ones(1, 128, dtype=torch.long),     # input_ids
        torch.ones(1, 128, dtype=torch.long),     # attention_mask
        torch.zeros(1, 128, dtype=torch.long)     # token_type_ids
    )

    # 调用 disassemble 函数
    print("Exporting model structure...")
    disassemble(
        model=model,
        dummy_input=dummy_input,
        onnx_temp_folder="./onnx_temp",
        result_folder="./results"
    )
    print("Export completed! Check the 'results' folder for the JSON file.")

if __name__ == "__main__":
    main()
