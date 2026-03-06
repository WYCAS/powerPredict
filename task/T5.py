import torch
from torch import nn
from transformers import T5Tokenizer, T5ForConditionalGeneration


class T5(nn.Module):
    def __init__(self, model_size="t5-small"):
        """
        初始化T5模型和分词器。

        参数:
            model_size (str): 预训练模型标识，默认为 "t5-small"。
                                可选: "t5-base", "t5-large", "t5-3b", "t5-11b"。
            max_length (int): 输入文本编码后的最大长度。
        """
        super().__init__()
        # 使用条件生成模型，这是标准的T5模型[citation:10]
        
        #self.model = T5ForConditionalGeneration.from_pretrained(model_size)

        local_model_path = "/mnt/data/yqr/models/google-t5/t5-small"
        self.model = T5ForConditionalGeneration.from_pretrained(local_model_path)

        # 将模型设置为评估模式 (与YOLOv5加载后的行为一致)
        # 获取编码器
        self.encoder = self.model.encoder
        # 设置为评估模式
        self.encoder.eval()

    def forward(self, input_ids, encoder_attention_mask,decoder_input_ids):
        # 1. Encoder前向传播
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True
        )

        # 2. 创建Decoder的因果掩码（下三角矩阵）
        batch_size, decoder_seq_len = decoder_input_ids.shape
        # 生成形状为[1, 1, decoder_seq_len, decoder_seq_len]的因果掩码
        causal_mask = self._generate_causal_mask(decoder_seq_len, device=decoder_input_ids.device)

        # 3. Decoder前向传播
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=causal_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True
        )
        return decoder_outputs.last_hidden_state

    @staticmethod
    def _generate_causal_mask(seq_len, device='cpu'):
        """生成因果掩码（下三角矩阵）"""
        # 创建形状为[seq_len, seq_len]的下三角矩阵
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # 扩展维度：[1, 1, seq_len, seq_len] 用于多头注意力
        return mask.view(1, 1, seq_len, seq_len)

    # 可选：提供一个方便的方法来创建与模型输入匹配的虚拟输入
    @staticmethod
    def get_dummy_input(batch_size=1, seq_length=16, decoder_seq_len = 10):
        # 生成虚拟输入，包括input_ids和attention_mask
        input_ids = torch.randint(low=0, high=1000, size=(batch_size, seq_length), dtype=torch.long)
        encoder_attention_mask = torch.ones_like(input_ids)
        decoder_input_ids = torch.ones(batch_size, decoder_seq_len, dtype=torch.long)
        return input_ids, encoder_attention_mask, decoder_input_ids