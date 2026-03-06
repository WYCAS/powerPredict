from collections import defaultdict
from os import makedirs
from os.path import join

import torch
import onnx
from onnx import shape_inference
from google._upb._message import RepeatedScalarContainer
import json

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

def convert_repeated_containers(obj):
    """
    递归转换所有的 RepeatedScalarContainer 为普通列表
    """
    if isinstance(obj, RepeatedScalarContainer):
        # 直接转换为列表
        return list(obj)

    elif isinstance(obj, dict):
        # 递归处理字典
        return {key: convert_repeated_containers(value) for key, value in obj.items()}

    elif isinstance(obj, (list, tuple)):
        # 递归处理列表和元组
        return [convert_repeated_containers(item) for item in obj]

    else:
        # 其他类型直接返回
        return obj


def disassemble(model, dummy_input, onnx_temp_folder, result_folder):
    model_name = model.__class__.__name__

    temp_name = "model.onnx"
    temp_onnx = join(onnx_temp_folder, temp_name)
    result_path = join(result_folder, f"{model_name}.json")

    makedirs(onnx_temp_folder, exist_ok=True)
    makedirs(result_folder, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        temp_onnx,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
    )

    # 第二部分：解析ONNX模型，获取计算单元和输入输出形状
    model_onnx = onnx.load(temp_onnx)
    onnx.checker.check_model(model_onnx)

    # 运行形状推断
    inferred_model = shape_inference.infer_shapes(model_onnx)

    # 获取计算图
    graph = inferred_model.graph

    # 收集所有初始化的常量（包括指数值）
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
    print("全局输入：")
    for input in graph.input:
        print(f"  Name: {input.name}, Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

    print("全局输出：")
    for output in graph.output:
        print(f"  Name: {output.name}, Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")

    all_op = defaultdict(int)

    all_op_dict = defaultdict(lambda: [])
    # 打印所有节点（计算单元）及其输入输出形状
    print("节点信息：")
    for node in graph.node:
        op_info = {}
        op_name = node.op_type
        # print(f"Node: {node.name}, OpType: {op_name}")
        all_op[node.op_type] += 1
        if node.attribute:
            # print("   节点属性:")
            attribute_dict = {}
            for attr in node.attribute:
                # 处理不同类型的属性值
                if attr.i != 0:  # 整型属性
                    attribute_dict[attr.name] = attr.i
                elif attr.ints:  # 整数列表
                    attribute_dict[attr.name] = attr.ints
                elif attr.s:  # 字符串属性
                    attribute_dict[attr.name] = attr.s.decode('utf-8')
                elif attr.floats:  # 浮点数列表
                    attribute_dict[attr.name] = attr.floats
            # print(attribute_dict)
            op_info["attribute"] = attribute_dict
        else:
            pass
            # print("   节点属性: 无")

        # print("输入:")
        input_shapes = []
        for input_name in node.input:
            # 在value_info中查找张量形状
            # 先在常量中寻找
            # 首先检查是否是常量（在 initializers 中）
            input_data = {
                "shape": None,
                "value": None,
            }
            # 检查是否是 Constant 节点输出
            if input_name in constant_nodes:
                const_data = constant_nodes[input_name]
                input_data["shape"] = const_data['shape']
                input_data["value"] = const_data['value']
                if input_data["value"].ndim == 0:  # 标量
                    input_data["value"] = input_data["value"].item()
                else:
                    input_data["value"] = input_data["value"].tolist()
                input_data["dtype"] = const_data['dtype']
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
            # print(f"    {input_name}: Shape {tensor_shape}")
            input_shapes.append(input_data)

        op_info["inputs"] = input_shapes

        # print("输出:")
        output_shapes = []
        for output_name in node.output:
            output_data = {
                "shape": None,
                "value": None,
            }
            tensor_shape = None
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
            # print(f"    {output_name}: Shape {tensor_shape}")
        op_info["outputs"] = output_shapes
        all_op_dict[op_name].append(op_info)
    # print(all_op)
    print(all_op_dict)

    all_op_dict = convert_repeated_containers(all_op_dict)

    with open(result_path, "w", encoding='utf-8') as f:
        json.dump(all_op_dict, f, ensure_ascii=False, indent=2)