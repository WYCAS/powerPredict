import sys
import os
import torch
import onnx

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phl_code import disassemble

class YOLOv5s:
    """
    YOLOv5s模型包装类，用于与disassemble函数兼容
    """
    def __init__(self):
        self.name = "YOLOv5s"
    
    def eval(self):
        """设置为评估模式"""
        return self
    
    def __call__(self, x):
        """模拟前向传播"""
        # 这个函数不会被实际调用，因为我们直接使用ONNX模型
        return x

def main():
    # 初始化模型
    print("Initializing YOLOv5s model wrapper...")
    model = YOLOv5s()
    model.eval()
    
    # 创建虚拟输入 - YOLOv5s的输入是[1, 3, 640, 640]的图像张量
    print("Creating dummy input...")
    dummy_input = torch.ones(1, 3, 640, 640, dtype=torch.float32)
    
    # 检查ONNX模型是否存在
    yolo_onnx_path = "/mnt/data/phl/yqr-PowerPredict/PowerPredict/src/analyse/yolov5s.onnx"
    if not os.path.exists(yolo_onnx_path):
        print(f"Error: ONNX model file not found at {yolo_onnx_path}")
        return
    
    # 创建临时目录
    onnx_temp_folder = "./onnx_temp"
    result_folder = "./results"
    os.makedirs(onnx_temp_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    
    # 复制ONNX模型到临时目录
    import shutil
    temp_onnx = os.path.join(onnx_temp_folder, "model.onnx")
    shutil.copy(yolo_onnx_path, temp_onnx)
    
    # 调用 disassemble 函数
    print("Exporting model structure...")
    disassemble(
        model=model,
        dummy_input=dummy_input,
        onnx_temp_folder=onnx_temp_folder,
        result_folder=result_folder
    )
    
    print("Export completed! Check the 'results' folder for the JSON file.")

if __name__ == "__main__":
    main()
