import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class BertSquad(nn.Module):
    def __init__(self, model_size="bert-large-uncased-whole-word-masking-finetuned-squad"):
        """
        初始化Bert Squad模型和分词器。
        
        参数:
            model_size (str): 预训练模型标识，默认为 
                            "bert-large-uncased-whole-word-masking-finetuned-squad"
                            可选: "bert-base-uncased", "distilbert-base-uncased-distilled-squad"等
        """
        super().__init__()
        
        # 本地模型路径（如果需要）
        local_model_path = "/mnt/data/yqr/models/google-bert/bert-base-uncased"
        self.model = AutoModelForQuestionAnswering.from_pretrained(local_model_path)
        
        #self.model = AutoModelForQuestionAnswering.from_pretrained(model_size)
        
        # 设置为评估模式
        self.model.eval()
        
        # 获取关键组件
        self.bert = self.model.bert
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Bert Squad前向传播（问答任务）
        
        参数:
            input_ids: token IDs [batch_size, seq_length]
            attention_mask: 注意力掩码 [batch_size, seq_length]
            token_type_ids: token类型IDs [batch_size, seq_length]
                            (0=问题, 1=上下文)
            
        返回:
            start_logits: 答案开始位置logits [batch_size, seq_length]
            end_logits: 答案结束位置logits [batch_size, seq_length]
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        if token_type_ids is None:
            # 默认全部为0（问题部分）
            token_type_ids = torch.zeros_like(input_ids)
        
        # Bert前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        # QA输出头
        start_logits = self.model.qa_outputs(sequence_output)[:, :, 0]
        end_logits = self.model.qa_outputs(sequence_output)[:, :, 1]
        
        return start_logits, end_logits
    
    # 可选：提供一个方便的方法来创建与模型输入匹配的虚拟输入
    @staticmethod
    def get_dummy_input(batch_size=1, seq_length=384):
        """
        生成虚拟输入（模拟SQuAD格式）
        
        SQuAD任务典型输入：
        - 问题部分（token_type_ids=0）
        - 上下文部分（token_type_ids=1）
        
        参数:
            batch_size: 批大小
            seq_length: 序列长度（Bert最大512，但384是常用设置）
            
        返回:
            input_ids: 虚拟token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
        """
        # Bert词汇表大小约为30522
        vocab_size = 30522
        
        # 生成随机token IDs
        input_ids = torch.randint(low=0, high=vocab_size, 
                                 size=(batch_size, seq_length), 
                                 dtype=torch.long)
        
        # 注意力掩码（全部有效）
        attention_mask = torch.ones_like(input_ids)
        
        # 创建token类型IDs：前1/4为问题，后3/4为上下文
        token_type_ids = torch.zeros_like(input_ids)
        context_start = seq_length // 4
        token_type_ids[:, context_start:] = 1
        
        # 确保特殊token
        # [CLS] token = 101, [SEP] token = 102
        input_ids[:, 0] = 101  # CLS token
        if seq_length > 1:
            input_ids[:, context_start] = 102  # SEP token（问题结束）
        if seq_length > context_start + 1:
            input_ids[:, -1] = 102  # SEP token（上下文结束）
        
        return input_ids, attention_mask, token_type_ids
    
    @staticmethod
    def get_squad_like_input(batch_size=1, question_len=64, context_len=320):
        """
        更真实的SQuAD格式输入
        
        参数:
            batch_size: 批大小
            question_len: 问题长度
            context_len: 上下文长度
            （总长度 = question_len + context_len + 3个特殊token）
            
        返回:
            input_ids: 虚拟token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
        """
        total_len = question_len + context_len + 3  # [CLS], [SEP], [SEP]
        
        # Bert词汇表大小
        vocab_size = 30522
        
        input_ids = torch.randint(low=0, high=vocab_size, 
                                 size=(batch_size, total_len), 
                                 dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        
        # 设置特殊token
        # [CLS] token = 101
        input_ids[:, 0] = 101
        
        # 问题部分结束的[SEP] token = 102
        input_ids[:, question_len + 1] = 102
        
        # 上下文部分结束的[SEP] token = 102
        input_ids[:, -1] = 102
        
        # 设置token类型IDs
        # 问题部分：[CLS] + question + [SEP] = 0
        # 上下文部分：context + [SEP] = 1
        token_type_ids[:, question_len + 2:] = 1
        
        return input_ids, attention_mask, token_type_ids