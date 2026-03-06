import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class TinyLlama(nn.Module):
    def __init__(self, model_size="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        初始化TinyLlama模型和分词器。
        
        参数:
            model_size (str): 预训练模型标识，默认为 "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        """
        super().__init__()
        
        # 本地模型路径（如果需要）
        local_model_path = "/mnt/data/yqr/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model = AutoModelForCausalLM.from_pretrained(local_model_path)
        
        #self.model = AutoModelForCausalLM.from_pretrained(model_size)
        
        # 设置为评估模式
        self.model.eval()
        
        # 获取关键组件（便于分析）
        self.transformer = self.model.model  # Llama的transformer主体
        
    def forward(self, input_ids, attention_mask=None):
        """
        TinyLlama前向传播（仅推理模式）
        
        参数:
            input_ids: token IDs [batch_size, seq_length]
            attention_mask: 注意力掩码 [batch_size, seq_length]
            
        返回:
            logits: 预测logits [batch_size, seq_length, vocab_size]
        """
        # 生成因果注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 创建因果掩码（下三角矩阵）
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # 扩展注意力掩码以匹配因果掩码形状
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        attention_mask = attention_mask.expand(-1, -1, seq_len, -1)
        
        # 合并因果掩码和注意力掩码
        combined_mask = attention_mask * causal_mask
        
        # 前向传播
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=combined_mask,
            use_cache=False,
            return_dict=True
        )
        
        # 获取最后一层隐藏状态并通过LM头
        hidden_states = outputs.last_hidden_state
        logits = self.model.lm_head(hidden_states)
        
        return logits
    
    @staticmethod
    def _generate_causal_mask(seq_len, device='cpu'):
        """生成因果掩码（下三角矩阵）"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1, 1, seq_len, seq_len)
    
    # 可选：提供一个方便的方法来创建与模型输入匹配的虚拟输入
    @staticmethod
    def get_dummy_input(batch_size=1, seq_length=128):
        """
        生成虚拟输入
        
        参数:
            batch_size: 批大小
            seq_length: 序列长度
            
        返回:
            input_ids: 虚拟token IDs
            attention_mask: 注意力掩码
        """
        # TinyLlama的词汇表大小约为32000
        vocab_size = 32000
        input_ids = torch.randint(low=0, high=vocab_size, 
                                 size=(batch_size, seq_length), 
                                 dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return input_ids, attention_mask
    
    
    def export_to_onnx_optimized(self, dummy_input, onnx_path, use_external_data=True):
        """
        优化的ONNX导出方法，专门用于大型模型

        参数:
            dummy_input: 虚拟输入
            onnx_path: ONNX文件保存路径
            use_external_data: 是否使用外部数据存储
        """
        input_ids, attention_mask = dummy_input

        # 设置模型为评估模式
        self.eval()

        # 使用torch.no_grad()减少内存使用
        with torch.no_grad():
            if use_external_data:
                print("使用外部数据存储导出TinyLlama模型...")
                torch.onnx.export(
                    self,
                    (input_ids, attention_mask),
                    onnx_path,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["logits"],
                    opset_version=13,  # 使用13版本，更稳定
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="tinyllama_external_data",
                    size_threshold=1024,  # 1KB以上存储外部
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "seq_length"},
                        "attention_mask": {0: "batch_size", 1: "seq_length"},
                        "logits": {0: "batch_size", 1: "seq_length"}
                    },
                    verbose=False,
                )
            else:
                # 备选方案：不使用外部数据
                torch.onnx.export(
                    self,
                    (input_ids, attention_mask),
                    onnx_path,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["logits"],
                    opset_version=13,
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "seq_length"},
                        "attention_mask": {0: "batch_size", 1: "seq_length"},
                        "logits": {0: "batch_size", 1: "seq_length"}
                    },
                )
