from transformers import PretrainedConfig

class SASREConfig(PretrainedConfig):
    model_type = "SASREC" 
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = kwargs.get("vocab_size")
        self.d_model = kwargs.get("d_model")
        self.seq_len = kwargs.get("seq_len")
        self.dropout = kwargs.get("dropout")
        self.nhead = kwargs.get("nhead")
        self.dim_feedforward = kwargs.get("dim_feedforward")
        self.num_layers = kwargs.get("num_layers")
        
class LightGCNConfig(PretrainedConfig):
    model_type = "lightgcn"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_users = kwargs.get("n_users")
        self.m_items = kwargs.get("m_items")
        self.d_model = kwargs.get("d_model")
        self.num_layers = kwargs.get("num_layers")
        self.dropout = kwargs.get("dropout")

class mfConifg(PretrainedConfig):
    model_type = "neumf"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_users = kwargs.get("n_users")
        self.n_items = kwargs.get("n_items")
        self.d_model = kwargs.get("d_model")

class caserConfig(PretrainedConfig):
    model_type = "caser"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = kwargs.get("d_model")
        self.n_h = kwargs.get("n_h")
        self.n_v = kwargs.get("n_v")
        self.dropout = kwargs.get("dropout")
        self.max_his_len = kwargs.get("seq_len")


class hgnConfig(PretrainedConfig):
    model_type = "hgn"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = kwargs.get("d_model")
        self.pooling_type = kwargs.get("pooling_type")
        self.max_his_len = kwargs.get("seq_len")

# from transformers.models.t5.configuration_t5 import T5Config
# class tigerConfit(T5Config):
#     model_type = "tiger"
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)