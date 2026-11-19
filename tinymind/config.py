"""
TinyMind 配置类（对应 MiniMind: model/model_minimind.py -> MiniMindConfig）

与 MiniMind 的区别：这里不继承 transformers.PretrainedConfig，只是一个纯 Python 类，
方便你把注意力集中在「每个超参数控制模型的哪一部分」上。字段名与默认值与 MiniMind 完全一致，
因此 TinyMind 与 MiniMind 的权重可以互相加载（见 tests/test_ch08_model.py 的等价性测试）。
"""
import math


class TinyMindConfig:
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        # ---------------- 基础结构 ----------------
        self.hidden_size = hidden_size                     # d_model：每个 token 的向量维度
        self.num_hidden_layers = num_hidden_layers         # Transformer Block 的层数
        self.use_moe = use_moe                             # FFN 是否替换为 MoE（第 9 章）
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)   # 词表大小（第 3 章）
        self.bos_token_id = kwargs.get("bos_token_id", 1)  # <|im_start|>
        self.eos_token_id = kwargs.get("eos_token_id", 2)  # <|im_end|>
        self.pad_token_id = kwargs.get("pad_token_id", 0)  # <|endoftext|>
        # ---------------- 注意力（第 5 章） ----------------
        self.flash_attn = kwargs.get("flash_attn", True)   # 是否使用 F.scaled_dot_product_attention
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)   # Q 头数
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)   # KV 头数（GQA）
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        # ---------------- FFN（第 7 章） ----------------
        self.hidden_act = kwargs.get("hidden_act", "silu")
        # 中间层维度 ≈ hidden * pi，并向上取整到 64 的倍数（对 GPU 友好）
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        # ---------------- 位置编码 & 归一化（第 4、6 章） ----------------
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # 输入/输出 embedding 共享
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn",
        } if self.inference_rope_scaling else None
        # ---------------- MoE（第 9 章，use_moe=False 时忽略） ----------------
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        # ---------------- 初始化 ----------------
        self.initializer_range = kwargs.get("initializer_range", 0.02)

    def to_dict(self):
        return dict(self.__dict__)

    def __repr__(self):
        items = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if k != "rope_scaling")
        return f"TinyMindConfig({items})"
