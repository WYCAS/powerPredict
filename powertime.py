#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 功率阈值 (Watt)，超过此值将被视为异常点剔除
POWER_THRESHOLD = 400.0 
# 平滑曲线的滑动窗口大小
SMOOTH_WINDOW = 20 
# ===========================================

# 全局集合，用于记录加载失败的模型
FAILED_MODELS_LOG = set()

# ---------------------------------------------------------
# 1. 导入 MLP
# ---------------------------------------------------------
mlp_dir = "/mnt/data/phl/power_grid/dabao_Enriched_the_input_parameters/src/train/predictor"
if mlp_dir not in sys.path:
    sys.path.append(mlp_dir)

try:
    from MLP import TorchMLP
    print(f"✅ 成功导入 TorchMLP")
except ImportError as e:
    print(f"❌ 导入 TorchMLP 失败: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 2. 算子映射表
# ---------------------------------------------------------
OP_FILENAME_MAPPING = {
    'Identity': 'multi_identity',
    'Gather': 'multi_gather',
    'Add': 'multi_add',
    'ReduceMean': 'multi_reduce_mean',
    'Sub': 'multi_sub',
    'Pow': 'multi_pow',
    'Sqrt': 'multi_sqrt',
    'Div': 'multi_div',
    'Unsqueeze': 'multi_unsqueeze',
    'ConstantOfShape': 'multi_constant_of_shape',
    'Equal': 'multi_equal',
    'Where': 'multi_where',
    'Expand': 'multi_expand',
    'Cast': 'multi_cast',
    'Reshape': 'multi_reshape',
    'Transpose': 'multi_transpose',
    'Shape': 'multi_shape',
    'Slice': 'multi_slice',
    'Softmax': 'multi_softmax',
    'MatMul': 'multi_mat',
    'Mul': 'multi_mat',
    'Erf': 'multi_relu',
    'Conv': 'multi_conv',
    'Sigmoid': 'multi_sigmoid',
    'Concat': 'multi_cat',
    'MaxPool': 'multi_max_pooling',
    'Resize': 'multi_resize',
    'Split': 'multi_split'
}

# ---------------------------------------------------------
# 3. 稳健的解析器
# ---------------------------------------------------------
def parse_operator_features_robust(txt_path):
    print(f"正在读取算子文件: {txt_path}")
    if not os.path.exists(txt_path):
        print(f"❌ 文件不存在: {txt_path}")
        return []

    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace('\\', '') 
    content = content.replace('\n', ' ')
    pattern = re.compile(r'index\s*:\s*(\d+)\s+([A-Za-z0-9_]+)\s*:\s*\[(.*?)\]')
    matches = pattern.findall(content)

    results = []
    for m in matches:
        idx_str, op_type, feat_raw = m
        feat_str = feat_raw.strip()
        features = []
        if feat_str:
            try:
                parts = re.split(r'[,\s]+', feat_str)
                features = [float(x) for x in parts if x.strip()]
            except ValueError:
                pass 
        
        results.append({'id': int(idx_str), 'type': op_type, 'features': features})
    print(f"✅ 解析完成，共提取到 {len(results)} 个算子")
    return results

# ---------------------------------------------------------
# 4. 智能模型加载器
# ---------------------------------------------------------
def load_model_smart(model_path):
    if model_path in FAILED_MODELS_LOG:
        return None, None, None, None
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            meta = state_dict
            weights = meta['model_state_dict']
            input_size = meta.get('input_size', 10)
            scaler_mean = meta.get('scaler_mean', 0.0)
            scaler_scale = meta.get('scaler_scale', 1.0)
            output_scale = meta.get('output_scale', 1.0)
        else:
            weights = state_dict
            input_size = weights[list(weights.keys())[0]].shape[1]
            scaler_mean, scaler_scale, output_scale = 0.0, 1.0, 1.0 

        model = TorchMLP(input_size)
        model.load_state_dict(weights)
        model.eval()
        return model, scaler_mean, scaler_scale, output_scale
    except Exception:
        FAILED_MODELS_LOG.add(model_path)
        return None, None, None, None

# ---------------------------------------------------------
# 5. 预测逻辑
# ---------------------------------------------------------
def predict_power_and_time(model_info, features):
    if not model_info or model_info[0] is None:
        return 0.0, 0.0
    model, s_mean, s_scale, o_scale = model_info
    try:
        expected_dim = model.net[0].in_features
        features = (features + [0.0] * expected_dim)[:expected_dim]
        feat_np = np.array(features).reshape(1, -1)
        norm_feat = (feat_np - s_mean) / s_scale
        
        with torch.no_grad():
            output = model(torch.FloatTensor(norm_feat))
            if isinstance(output, tuple):
                p, t = output[0].item(), output[1].item()
                sp = o_scale[0] if isinstance(o_scale, (list, np.ndarray)) else o_scale
                st = o_scale[1] if isinstance(o_scale, (list, np.ndarray)) and len(o_scale) > 1 else 1.0
                return abs(p/sp), abs(t/st)
            else:
                scale = o_scale[0] if isinstance(o_scale, (list, np.ndarray)) else o_scale
                return abs(output.item()/scale), 0.0
    except Exception:
        return 0.0, 0.0

# ---------------------------------------------------------
# 6. 绘图功能 (包含柱状图和平滑曲线图)
# ---------------------------------------------------------

def get_cumulative_time(data_list):
    """计算累计时间轴"""
    times = [d['time'] for d in data_list]
    start_times = [0.0]
    for t in times[:-1]:
        start_times.append(start_times[-1] + t)
    return start_times, times

def plot_bar_profile(data_list, total_energy, total_time, output_img):
    """绘制原始功率柱状图（跳过异常值）"""
    if not data_list: return
    
    start_times, durations = get_cumulative_time(data_list)
    powers = [d['power'] for d in data_list]
    ids = [d['id'] for d in data_list]

    plt.figure(figsize=(14, 7))
    plt.bar(start_times, powers, width=durations, align='edge', color='skyblue', edgecolor='navy', alpha=0.7)

    plt.title(f'Operator Power Profile (Valid Ops)\nTotal Energy: {total_energy:.4f} J | Total Execution Time: {total_time:.6f} s', fontsize=12)
    plt.xlabel('Cumulative Time (seconds)')
    plt.ylabel('Power (Watts)')
    plt.grid(True, linestyle='--', alpha=0.4)

    # 标注 Top 5 功率算子
    top_p_indices = np.argsort(powers)[-5:]
    for idx in top_p_indices:
        if powers[idx] > 0:
            plt.text(start_times[idx] + durations[idx]/2, powers[idx], f"Op:{ids[idx]}", 
                     ha='center', va='bottom', fontsize=8, color='darkred', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"📊 柱状图已保存至: {output_img}")
    plt.close()

def plot_smoothed_curve(data_list, output_img):
    """绘制平滑后的功率曲线图（跳过异常值）"""
    if not data_list or len(data_list) < 2: return

    powers = np.array([d['power'] for d in data_list])
    start_times, _ = get_cumulative_time(data_list)
    
    # 简单的移动平均平滑
    window_size = min(SMOOTH_WINDOW, len(powers))
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        # 使用 'same' 模式保持长度一致，但边缘可能会有伪影，这里处理一下
        smoothed_powers = np.convolve(powers, kernel, mode='same')
        # 修正边缘：简单的卷积会让边缘数据偏小，这里不做复杂处理，仅作趋势展示
    else:
        smoothed_powers = powers

    plt.figure(figsize=(14, 7))
    
    # 绘制原始数据的浅色背景线
    plt.plot(start_times, powers, color='lightgray', alpha=0.5, label='Raw Power')
    
    # 绘制平滑曲线
    plt.plot(start_times, smoothed_powers, color='crimson', linewidth=2, label=f'Smoothed (MA={window_size})')
    
    # 填充曲线下方区域
    plt.fill_between(start_times, smoothed_powers, color='crimson', alpha=0.1)

    plt.title(f'Smoothed Power Trend (Threshold < {POWER_THRESHOLD}W)', fontsize=12)
    plt.xlabel('Cumulative Time (seconds)')
    plt.ylabel('Power (Watts)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"📈 平滑曲线图已保存至: {output_img}")
    plt.close()

# ---------------------------------------------------------
# 7. 主程序
# ---------------------------------------------------------
def main():
    input_txt = "yolov5_operators.txt"
    
    # === 路径配置 ===
    model_configs = [
        {
            "tag": "4090time",
            "dict_dir": "/mnt/data/phl/power_grid/dabao_Enriched_the_input_parameters/src/train/dict_4090",
            "out_txt": "4090time.txt",
            "out_img_bar": "4090time_profile.png",
            "out_img_smooth": "4090time_smooth.png" # 新增平滑图路径
        },
        {
            "tag": "t4time",
            "dict_dir": "/mnt/data/phl/power_grid/dabao_Enriched_the_input_parameters/src/train/dict_t4_v2",
            "out_txt": "t4time.txt",
            "out_img_bar": "t4time_profile.png",
            "out_img_smooth": "t4time_smooth.png" # 新增平滑图路径
        }
    ]

    operators = parse_operator_features_robust(input_txt)
    if not operators: 
        print("❌ 未能解析到算子数据，请检查输入文件。")
        return

    for config in model_configs:
        print("\n" + f" 🚀 正在处理模型集: {config['tag']} ".center(80, "="))
        
        FAILED_MODELS_LOG.clear()
        loaded_models = {}
        results_data = []
        anomalies_log = [] # 用于记录异常算子

        # === 预测循环 ===
        for i, op in enumerate(operators):
            op_type = op['type']
            model_name = OP_FILENAME_MAPPING.get(op_type)
            
            if not model_name:
                for k, v in OP_FILENAME_MAPPING.items():
                    if k.lower() == op_type.lower():
                        model_name = v
                        break
            
            model_info = None
            if model_name:
                if model_name not in loaded_models:
                    pt_path = os.path.join(config['dict_dir'], f"{model_name}.pt")
                    loaded_models[model_name] = load_model_smart(pt_path) if os.path.exists(pt_path) else None
                model_info = loaded_models[model_name]
            
            p, t = predict_power_and_time(model_info, op['features'])

            # --- 异常检测逻辑 ---
            is_error = False
            if p > POWER_THRESHOLD:
                is_error = True
                anomalies_log.append({
                    'id': op['id'], 
                    'type': op_type, 
                    'power': p
                })
            
            results_data.append({
                'id': op['id'], 
                'type': op_type, 
                'power': p, 
                'time': t, 
                'energy': p * t,
                'is_error': is_error  # 标记该条数据
            })
            
            if i > 0 and i % 1000 == 0: print(f"进度: {i}/{len(operators)}...")

        # === 打印异常信息 ===
        if anomalies_log:
            print(f"\n⚠️  发现 {len(anomalies_log)} 个异常算子 (功率 > {POWER_THRESHOLD}W):")
            print(f"   {'Index':<8} {'Operator':<20} {'Power':<15}")
            print("   " + "-"*45)
            for err in anomalies_log:
                print(f"   {err['id']:<8} {err['type']:<20} {err['power']:.2f} W")
        else:
            print(f"\n✅ 未发现功率大于 {POWER_THRESHOLD}W 的异常算子。")

        # === 统计结果 (全量数据计算总能耗，或者你可以选择排除异常数据计算) ===
        # 这里为了记录真实预测情况，TXT中保留所有数据，但在统计变量中我将其分为 total_all 和 total_valid
        total_energy = sum(d['energy'] for d in results_data)
        total_time = sum(d['time'] for d in results_data)

        # 写入文件 (包含标记是否异常)
        with open(config['out_txt'], 'w', encoding='utf-8') as f:
            f.write("Index\tOperator\tPower(W)\tTime(s)\tEnergy(J)\tStatus\n")
            for d in results_data:
                status = "ERROR" if d['is_error'] else "OK"
                f.write(f"{d['id']}\t{d['type']}\t{d['power']:.9f}\t{d['time']:.9f}\t{d['energy']:.9f}\t{status}\n")
            f.write(f"\nTotal Time: {total_time:.6f} s\nTotal Energy: {total_energy:.6f} J\n")

        # === 准备绘图数据 (剔除异常算子) ===
        valid_data = [d for d in results_data if not d['is_error']]
        valid_energy = sum(d['energy'] for d in valid_data)
        valid_time = sum(d['time'] for d in valid_data)

        if valid_data:
            # 1. 绘制柱状图
            plot_bar_profile(valid_data, valid_energy, valid_time, config['out_img_bar'])
            # 2. 绘制平滑曲线图 (新增)
            plot_smoothed_curve(valid_data, config['out_img_smooth'])
        else:
            print("⚠️ 有效数据为空，跳过绘图。")
        
        print(f"✅ {config['tag']} 处理完成！有效能耗: {valid_energy:.6f} J")

    print("\n" + " 所有模型集运行结束 ".center(80, "#"))

if __name__ == "__main__":
    main()