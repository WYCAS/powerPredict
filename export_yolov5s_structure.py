import sys
import os
import torch
import onnx

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phl_code import disassemble

def main():
    # YOLOv5s ONNX模型路径
    yolo_onnx_path = "/mnt/data/phl/yqr-PowerPredict/PowerPredict/src/analyse/yolov5s.onnx"
    
    print(f"Loading YOLOv5s ONNX model from {yolo_onnx_path}...")
    
    # 检查文件是否存在
    if not os.path.exists(yolo_onnx_path):
        print(f"Error: ONNX model file not found at {yolo_onnx_path}")
        return
    
    # 创建临时目录
    onnx_temp_folder = "./onnx_temp"
    result_folder = "./results"
    os.makedirs(onnx_temp_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    
    # 直接使用已有的ONNX模型
    print("Analyzing YOLOv5s ONNX model structure...")
    analyze_yolo_onnx(
        onnx_path=yolo_onnx_path,
        onnx_temp_folder=onnx_temp_folder,
        result_folder=result_folder
    )
    
    print("Export completed! Check the 'results' folder for the JSON file.")

def analyze_yolo_onnx(onnx_path, onnx_temp_folder, result_folder):
    """
    分析YOLOv5s ONNX模型结构并导出为JSON
    
    参数:
        onnx_path: YOLOv5s ONNX模型路径
        onnx_temp_folder: 临时文件夹路径
        result_folder: 结果保存路径
    """
    from os.path import join
    import json
    from collections import defaultdict
    from onnx import shape_inference
    import numpy as np
    
    # 模型名称
    model_name = "YOLOv5s"
    result_path = join(result_folder, f"{model_name}.json")
    
    # 加载ONNX模型
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    
    # 运行形状推断
    print("Running shape inference...")
    try:
        inferred_model = shape_inference.infer_shapes(model_onnx)
    except Exception as e:
        print(f"Shape inference failed: {e}")
        print("Using original model without shape inference")
        inferred_model = model_onnx
    
    # 获取计算图
    graph = inferred_model.graph
    
    # 收集所有初始化的常量
    print("Collecting initializers...")
    initializers = {}
    for initializer in graph.initializer:
        # 使用 onnx.numpy_helper 来提取常量的实际值
        array = onnx.numpy_helper.to_array(initializer)
        initializers[initializer.name] = {
            'value': array,
            'shape': list(array.shape) if hasattr(array, 'shape') else [],
            'dtype': str(array.dtype)
        }
    
    # 收集 Constant 节点的值
    print("Collecting constant nodes...")
    constant_nodes = {}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    # 提取 Constant 节点的值
                    tensor = attr.t
                    array = onnx.numpy_helper.to_array(tensor)
                    constant_nodes[node.output[0]] = {
                        'value': array,
                        'shape': list(array.shape) if hasattr(array, 'shape') else [],
                        'dtype': str(array.dtype)
                    }
    
    # 打印全局输入和输出
    print("Global inputs:")
    for input in graph.input:
        print(f" Name: {input.name}, Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
    
    print("Global outputs:")
    for output in graph.output:
        print(f" Name: {output.name}, Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")
    
    # ONNX数据类型到字符串的映射
    def onnx_dtype_to_string(elem_type):
        """将ONNX数据类型枚举值转换为可读的字符串"""
        dtype_map = {
            1: "float32",
            2: "uint8",
            3: "int8",
            4: "uint16",
            5: "int16",
            6: "int32",
            7: "int64",
            8: "string",
            9: "bool",
            10: "float16",
            11: "float64",
            12: "uint32",
            13: "uint64",
            14: "complex64",
            15: "complex128",
            16: "bfloat16"
        }
        return dtype_map.get(elem_type, f"unknown({elem_type})")
    
    # 递归转换所有的 RepeatedScalarContainer 为普通列表
    def convert_repeated_containers(obj):
        from google._upb._message import RepeatedScalarContainer
        
        if isinstance(obj, RepeatedScalarContainer):
            # 直接转换为列表
            return list(obj)
        elif isinstance(obj, dict):
            # 递归处理字典
            return {key: convert_repeated_containers(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # 递归处理列表和元组
            return [convert_repeated_containers(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            # 处理numpy数组
            if obj.size < 100:  # 只保留小数组的实际值
                return obj.tolist()
            else:
                return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
        else:
            # 其他类型直接返回
            return obj
    
    all_op = defaultdict(int)
    
    # 使用列表按遍历顺序存储算子信息
    all_op_sequence = []  # 按线性顺序存储算子
    
    # 处理所有节点（计算单元）及其输入输出形状
    print("Processing nodes...")
    for index, node in enumerate(graph.node):
        op_info = {}
        op_name = node.op_type
        
        # 显式记录算子的线性顺序
        op_info["index"] = index
        op_info["op_type"] = op_name
        op_info["name"] = node.name if node.name else f"{op_name}_{index}"
        
        all_op[node.op_type] += 1
        
        if node.attribute:
            attribute_dict = {}
            for attr in node.attribute:
                # 处理不同类型的属性值
                if attr.type == 1:  # FLOAT
                    attribute_dict[attr.name] = attr.f
                elif attr.type == 2:  # INT
                    attribute_dict[attr.name] = attr.i
                elif attr.type == 3:  # STRING
                    attribute_dict[attr.name] = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else attr.s
                elif attr.type == 4:  # TENSOR
                    try:
                        tensor_value = onnx.numpy_helper.to_array(attr.t)
                        if tensor_value.size < 100:  # 只保留小张量的实际值
                            attribute_dict[attr.name] = tensor_value.tolist()
                        else:
                            attribute_dict[attr.name] = f"tensor(shape={tensor_value.shape}, dtype={tensor_value.dtype})"
                    except:
                        attribute_dict[attr.name] = "tensor(unknown)"
                elif attr.type == 5:  # GRAPH
                    attribute_dict[attr.name] = "graph"
                elif attr.type == 6:  # FLOATS
                    attribute_dict[attr.name] = list(attr.floats)
                elif attr.type == 7:  # INTS
                    attribute_dict[attr.name] = list(attr.ints)
                elif attr.type == 8:  # STRINGS
                    attribute_dict[attr.name] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in attr.strings]
                elif attr.type == 9:  # TENSORS
                    attribute_dict[attr.name] = "tensors"
                elif attr.type == 10:  # GRAPHS
                    attribute_dict[attr.name] = "graphs"
            
            op_info["attribute"] = attribute_dict
        
        # 处理输入
        input_shapes = []
        for input_name in node.input:
            input_data = {
                "name": input_name,
                "shape": None,
                "value": None,
            }
            
            # 检查是否是 Constant 节点输出
            if input_name in constant_nodes:
                const_data = constant_nodes[input_name]
                input_data["shape"] = const_data['shape']
                
                # 只保留小数组的实际值
                if isinstance(const_data['value'], np.ndarray) and const_data['value'].size < 100:
                    input_data["value"] = const_data['value'].tolist()
                else:
                    input_data["value"] = None
                
                input_data["dtype"] = const_data['dtype']
            elif input_name in initializers:
                init_data = initializers[input_name]
                input_data["shape"] = init_data['shape']
                
                # 只保留小数组的实际值
                if isinstance(init_data['value'], np.ndarray) and init_data['value'].size < 100:
                    input_data["value"] = init_data['value'].tolist()
                else:
                    input_data["value"] = None
                
                input_data["dtype"] = init_data['dtype']
            else:
                # 如果不是常量，使用原来的方法获取形状
                for value_info in graph.value_info:
                    if value_info.name == input_name:
                        shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                        input_data["shape"] = shape
                        elem_type = value_info.type.tensor_type.elem_type
                        input_data["dtype"] = onnx_dtype_to_string(elem_type)
                        break
                
                if input_data["shape"] is None:
                    for input in graph.input:
                        if input.name == input_name:
                            input_data["shape"] = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
                            elem_type = input.type.tensor_type.elem_type
                            input_data["dtype"] = onnx_dtype_to_string(elem_type)
                            break
                
                for output in graph.output:
                    if output.name == input_name:
                        input_data["shape"] = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
                        elem_type = output.type.tensor_type.elem_type
                        input_data["dtype"] = onnx_dtype_to_string(elem_type)
                        break
            
            input_shapes.append(input_data)
        
        op_info["inputs"] = input_shapes
        
        # 处理输出
        output_shapes = []
        for output_name in node.output:
            output_data = {
                "name": output_name,
                "shape": None,
                "value": None,
            }
            
            for value_info in graph.value_info:
                if value_info.name == output_name:
                    tensor_shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    output_data["shape"] = tensor_shape
                    elem_type = value_info.type.tensor_type.elem_type
                    output_data["dtype"] = onnx_dtype_to_string(elem_type)
                    break
            
            if output_data["shape"] is None:
                for output in graph.output:
                    if output.name == output_name:
                        tensor_shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
                        output_data["shape"] = tensor_shape
                        elem_type = output.type.tensor_type.elem_type
                        output_data["dtype"] = onnx_dtype_to_string(elem_type)
                        break
            
            output_shapes.append(output_data)
        
        op_info["outputs"] = output_shapes
        
        # 按遍历顺序追加到列表中
        all_op_sequence.append(op_info)
    
    print(f"Total operators: {len(all_op_sequence)}")
    print(f"Operator type statistics: {dict(all_op)}")
    
    # 转换 RepeatedScalarContainer 和处理大型数组
    all_op_sequence = convert_repeated_containers(all_op_sequence)
    
    # 序列化为 JSON
    print(f"Writing results to {result_path}...")
    with open(result_path, "w", encoding='utf-8') as f:
        json.dump(all_op_sequence, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
