from typing import Optional, Union, Unpack
from cachetools import Cache
from transformers import PreTrainedModel
from baseline_config import *
from transformers.modeling_outputs import ModelOutput
from transformers import PretrainedConfig
import torch
from torch import nn
import utils
from utils import *
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Dice(nn.Module):
    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score

class MLPLayers(nn.Module):
    def __init__(
        self,
        layers,
        dropout=0.0,
        activation="relu",
        bn=False,
        init_method=None,
        last_activation=True,
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)
        if self.activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                torch.nn.init.normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)

def activation_layer(activation_name="relu", emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "dice":
            activation = Dice(emb_dim)
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation

class PositionEmbedding(nn.Module):
    def __init__(self, seq_len=20, embed_dim=512, dropout=0.1):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 (L, D)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * 
                             (-math.log(10000.0) / embed_dim))
        
        # 创建位置编码表
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos
        
        # 将位置编码注册为buffer（不参与训练的参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 添加位置编码
        # print(x.shape,self.pe.shape,self.pe.device,x.device)
        x = x + self.pe[:x.size(1)]  # 使用切片确保只取序列实际长度部分
        return self.dropout(x)

class SASREC(PreTrainedModel):
    config_class = SASREConfig
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model,padding_idx=-1)
        self.positional_encoding = PositionEmbedding(seq_len=config.seq_len, embed_dim=config.d_model, dropout=config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout,batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        torch.backends.mha.set_fastpath_enabled(False)


        self.post_init()

    def _init_weights(self, module):
        # print("初始化被调用")  # 实际永远不会执行
        # if isinstance(module, nn.Linear):
        pass

    def forward(self,
        input_ids=None, # [batch_size, sequence_length] token_id
        attention_mask=None,# [batch_size, sequence_length] 0/1 
        pos_item=None,# [batch_size, 1]
        neg_item=None,# [batch_size, K]
        **kwargs):
        input_ids = utils.unsqueeze_input(input_ids)
        attention_mask = utils.unsqueeze_input(attention_mask)
        pos_item = utils.unsqueeze_input(pos_item)
        neg_item = utils.unsqueeze_input(neg_item)
        B,L = input_ids.shape
        # print(B,L)
        embeddings = self.embedding(input_ids)
        # print(embeddings.shape)
        embeddings = self.positional_encoding(embeddings)
        # print(embeddings.shape)
        src_key_padding_mask = (~attention_mask).bool()  # Convert attention_mask to key_padding_mask
        casual_mask = nn.Transformer.generate_square_subsequent_mask(self.config.seq_len,device=embeddings.device)
        output = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask,is_causal=True,mask=casual_mask)

        logits = output[:,-1,:]@ self.embedding.weight.t() # (B,D) @ (D,V) = (B,V)

        if neg_item[0,0]==-1:
            assert neg_item.shape[-1]==1 and torch.sum(neg_item[:,-1])==-B
            loss = F.cross_entropy(logits, torch.squeeze(pos_item,-1).long())
            # print(f"2-> {logits} {pos_item} {loss}")
        else:
            loss = compute_bce_loss(logits,pos_item,neg_item)

        return ModelOutput(
            logits=logits,
            loss=loss # torch.tensor(1)
        )
    
class LightGCN(PreTrainedModel):
    config_class = LightGCNConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_users  = self.config.n_users
        self.num_items  = self.config.m_items
        self.latent_dim = self.config.d_model
        self.n_layers = self.config.num_layers
        self.keep_prob = 1-self.config.dropout
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t() #index -> torch.Size([313218, 2])
        values = x.values()  # torch.Size([313218])
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse_coo_tensor(index.t(), values, size,device=x.device)
        return g
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        self.Graph = self.Graph.to(users_emb.device)
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.keep_prob<1 and self.training:
            g_droped = self.__dropout_x(self.Graph,self.keep_prob)
        else:
            g_droped = self.Graph    
        g_droped = g_droped.to_sparse_csr()
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1) # (34694,3,32)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1) # (34694,32)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def forward(self,
        user=None,# [batch_size,1]
        pos_item=None,# [batch_size,1]
        neg_item=None,# [batch_size,1]
        **kwargs
    ):
        B,L = user.shape
        all_users, all_items = self.computer() # [user_n,32] [item_n,32]
        users_emb = all_users[user[:,0]] # (400,32)
        logits = users_emb@ all_items.t() # (400,9922)
        if neg_item[0,0]==-1:
            assert neg_item.shape[-1]==1 and torch.sum(neg_item[:,-1])==-B
            loss = F.cross_entropy(logits, torch.squeeze(pos_item,-1).long())
            # print(f"2-> {logits} {pos_item} {loss}")
        else:
            loss = compute_bce_loss(logits,pos_item,neg_item) # bpr loss or bce loss? use bpr loss!!!
        return ModelOutput(
            logits=logits,
            loss=loss # torch.tensor(1)
        )
        
class mf(PreTrainedModel):
    config_class = mfConifg
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.user_mlp_embedding = nn.Embedding(self.config.n_users, self.config.d_model)
        self.item_mlp_embedding = nn.Embedding(self.config.m_items, self.config.d_model)
        nn.init.normal_(self.user_mlp_embedding.weight, std=0.1)
        nn.init.normal_(self.item_mlp_embedding.weight, std=0.1)
      

    def forward(self,
        user=None,# [batch_size,1]
        pos_item=None,# [batch_size,1]
        neg_item=None,# [batch_size,1]
        **kwargs
    ):
        user = utils.unsqueeze_input(user)
        pos_item = utils.unsqueeze_input(pos_item)
        neg_item = utils.unsqueeze_input(neg_item)
        user = self.user_mlp_embedding(user[:,0])
        
        B,L = user.shape
        logits = user@ self.item_mlp_embedding.weight.t()
        if neg_item[0,0]==-1:
            assert neg_item.shape[-1]==1 and torch.sum(neg_item[:,-1])==-B
            loss = F.cross_entropy(logits, torch.squeeze(pos_item,-1).long())
            # print(f"2-> {logits} {pos_item} {loss}")
        else:
            loss = utils.compute_bce_loss(logits,pos_item,neg_item)
        return ModelOutput(
            logits=logits,
            loss=loss # torch.tensor(1)
        )
    
class caser(PreTrainedModel):
    config_class=caserConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding_size = config.embedding_size
        self.n_h = config.n_h
        self.n_v = config.n_v
        self.dropout_prob = config.dropout
        self.n_items = config.m_items
        self.n_users = config.n_users
        self.max_seq_length = config.max_his_len

        # define layers and loss
        self.user_embedding = nn.Embedding(
            self.n_users+1, self.embedding_size, padding_idx=-1
        )
        self.item_embedding = nn.Embedding(
            self.n_items+1, self.embedding_size, padding_idx=-1
        )

        # vertical conv layer
        self.conv_v = nn.Conv2d(
            in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_length, 1)
        )

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_length)]
        self.conv_h = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.n_h,
                    kernel_size=(i, self.embedding_size),
                )
                for i in lengths
            ]
        )

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)
        self.fc2 = nn.Linear(
            self.embedding_size + self.embedding_size, self.embedding_size
        )

        self.dropout = nn.Dropout(self.dropout_prob)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight.data, 0, 1.0 / module.embedding_dim)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias.data, 0)

    def forward(self,
        user=None, # [batch_size, 1]
        input_ids=None, # [batch_size, sequence_length] token_id
        attention_mask=None,# [batch_size, sequence_length] 0/1 
        pos_item=None,# [batch_size, 1]
        neg_item=None,# [batch_size, K]
        **kwargs):
        user = utils.unsqueeze_input(user)
        pos_item = utils.unsqueeze_input(pos_item)
        neg_item = utils.unsqueeze_input(neg_item)
        attention_mask = utils.unsqueeze_input(attention_mask)
        input_ids = utils.unsqueeze_input(input_ids)
        B,L = input_ids.shape
        item_seq_emb = self.item_embedding(input_ids).unsqueeze(1)
        user_emb = self.user_embedding(user).squeeze(1)
        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer, length level
        if self.n_v:
            out_v = self.conv_v(item_seq_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect
        # horizontal conv layer, feature level
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) # (batch_size, self.n_h)
                out_hs.append(pool_out) 
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)
        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        seq_output = self.ac_fc(self.fc2(x)) # (B,D)
        logits = seq_output @ self.item_embedding.weight.t() # (B,V)

        if neg_item[0,0]==-1:
            assert neg_item.shape[-1]==1 and torch.sum(neg_item[:,-1])==-B
            loss = F.cross_entropy(logits, torch.squeeze(pos_item,-1).long())
            # print(f"2-> {logits} {pos_item} {loss}")
        else:
            loss = compute_bce_loss(logits,pos_item,neg_item)

        return ModelOutput(
            logits=logits,
            loss=loss # torch.tensor(1)
        )

class hgn(PreTrainedModel):
    config_class=hgnConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # load the parameter information
        self.embedding_size = config.embedding_size
        self.pool_type = config.pooling_type
        self.n_items = config.m_items
        self.n_users = config.n_users
        self.max_seq_length = config.max_his_len

        if self.pool_type not in ["max", "average"]:
            raise NotImplementedError("Make sure 'pool_type' in ['max', 'average']!")

        # define the layers and loss function
        self.item_embedding = nn.Embedding(
            self.n_items+1, self.embedding_size, padding_idx=-1
        )
        self.user_embedding = nn.Embedding(self.n_users+1, self.embedding_size, padding_idx=-1)

        # define the module feature gating need
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.w2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.b = nn.Parameter(torch.zeros(self.embedding_size), requires_grad=True)

        # define the module instance gating need
        self.w3 = nn.Linear(self.embedding_size, 1, bias=False)
        self.w4 = nn.Linear(self.embedding_size, self.max_seq_length, bias=False)

        # define item_embedding for prediction
        self.item_embedding_for_prediction = nn.Embedding(
            self.n_items, self.embedding_size
        )

        self.sigmoid = nn.Sigmoid()
        self.post_init()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight.data, 0.0, 1 / self.embedding_size)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias.data, 0)

    def feature_gating(self, seq_item_embedding, user_embedding):
        """
        choose the features that will be sent to the next stage(more important feature, more focus)
        """

        batch_size, seq_len, embedding_size = seq_item_embedding.size()
        seq_item_embedding_value = seq_item_embedding

        seq_item_embedding = self.w1(seq_item_embedding)
        # batch_size * seq_len * embedding_size
        user_embedding = self.w2(user_embedding)
        # batch_size * embedding_size
        user_embedding = user_embedding.unsqueeze(1).repeat(1, seq_len, 1)
        # batch_size * seq_len * embedding_size

        user_item = self.sigmoid(seq_item_embedding + user_embedding + self.b)
        # batch_size * seq_len * embedding_size

        user_item = torch.mul(seq_item_embedding_value, user_item)
        # batch_size * seq_len * embedding_size

        return user_item

    def instance_gating(self, user_item, user_embedding):
        """
        choose the last click items that will influence the prediction( more important more chance to get attention)
        """

        user_embedding_value = user_item

        user_item = self.w3(user_item)
        # batch_size * seq_len * 1

        user_embedding = self.w4(user_embedding).unsqueeze(2)
        # batch_size * seq_len * 1

        instance_score = self.sigmoid(user_item + user_embedding).squeeze(-1)
        # batch_size * seq_len * 1
        output = torch.mul(instance_score.unsqueeze(2), user_embedding_value)
        # batch_size * seq_len * embedding_size

        if self.pool_type == "average":
            output = torch.div(
                output.sum(dim=1), instance_score.sum(dim=1).unsqueeze(1)
            )
            # batch_size * embedding_size
        else:
            # for max_pooling
            index = torch.max(instance_score, dim=1)[1]
            # batch_size * 1
            output = self.gather_indexes(output, index)
            # batch_size * seq_len * embedding_size ==>> batch_size * embedding_size

        return output

    def forward(self,
        user=None, # [batch_size, 1]
        input_ids=None, # [batch_size, sequence_length] token_id
        attention_mask=None,# [batch_size, sequence_length] 0/1 
        pos_item=None,# [batch_size, 1]
        neg_item=None,# [batch_size, K]
        **kwargs):
        user = utils.unsqueeze_input(user)
        pos_item = utils.unsqueeze_input(pos_item)
        neg_item = utils.unsqueeze_input(neg_item)
        attention_mask = utils.unsqueeze_input(attention_mask)
        input_ids = utils.unsqueeze_input(input_ids)
        B,L = input_ids.shape
        seq_item_embedding = self.item_embedding(input_ids)
        user_embedding = self.user_embedding(user[:,0])
        feature_gating = self.feature_gating(seq_item_embedding, user_embedding)
        instance_gating = self.instance_gating(feature_gating, user_embedding)
        # batch_size * embedding_size
        item_item = torch.sum(seq_item_embedding, dim=1)
        output = user_embedding + instance_gating + item_item
        logits = output @ self.item_embedding_for_prediction.weight.t() 

        if neg_item[0,0]==-1:
            assert neg_item.shape[-1]==1 and torch.sum(neg_item[:,-1])==-B
            loss = F.cross_entropy(logits, torch.squeeze(pos_item,-1).long())
            # print(f"2-> {logits} {pos_item} {loss}")
        else:
            loss = compute_bce_loss(logits,pos_item,neg_item)

        return ModelOutput(
            logits=logits,
            loss=loss # torch.tensor(1)
        )
    
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from torch.nn import CrossEntropyLoss
from transformers.models.t5.configuration_t5 import T5Config
from transformers.modeling_outputs import BaseModelOutput,Seq2SeqLMOutput
class TIGER(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        # You can add parameters out here.
        self.temperature = 1.0

    def set_hyper(self,args,tokenizer):
        self.temperature = args.temperature
        self.args = args
        self.tokenizer = tokenizer

# 其实就是一个简简单单的cross entropy
# 唯一的改变就是这个温度。
    def ranking_loss(self, lm_logits, labels):
        if labels is not None:
            t_logits = lm_logits/self.temperature
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        return loss


    def total_loss(self, lm_logits, labels, decoder_input_ids):
        loss = self.ranking_loss(lm_logits, labels)             
        return loss
# 可能除了input_ids和labels attention_mask外其它都是做None
    def forward(
        self,
        input_ids=None, # [batch_size, sequence_length] token_id
        whole_word_ids=None,
        attention_mask=None,# [batch_size, sequence_length] 0/1 
        encoder_outputs=None, # [batch_size, sequence_length, hidden_size] decoder need it to do cross attention 
        decoder_input_ids=None, # [batch_size, sequence_length] decoder input ,use in encoder-decoder,in the training step ,it will be the label right shift.
        decoder_attention_mask=None,# decoder_attention_mask decoder attnetion mask,casual design
        cross_attn_head_mask = None,# tell model which encoder_outputs should be attentioned
        past_key_values=None, # 
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,

        **kwargs,
    ):
        r"""
            input_ids [B,L]  attention_mask [B,L] labels [B,5] (4 id + 1 eos) decoder_input_ids== labels shift
        """
        # print("1, ",input_ids,"2, ",decoder_input_ids)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, # [batch_size, sequence_length]
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
# hidden_states [B,L,D]
        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
# decoder_input_ids [B,5]
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
# 
        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)
# sequence_output [B,decoder_input_id_L,DIM]  lm_logits [B,decoder_input_id_L,vocab_size]
        lm_logits = self.lm_head(sequence_output)
        
        # ------------------------------------------
        # Loss Computing!
        loss = None
        if labels is not None:
            loss = self.total_loss(lm_logits, labels, decoder_input_ids)

        # ------------------------------------------

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM,KwargsForCausalLM,BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
import copy
class LCREC(Qwen3ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.temperature = 1.0
    def set_hyper(self,args,tokenizer):
        self.temperature = args.temperature
        self.args = args
        self.tokenizer = tokenizer
    def ranking_loss(self, shift_logits, shift_labels):
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits /self.temperature, shift_labels)
        return loss

    def total_loss(self, shift_logits, shift_labels):
        gen_loss = self.ranking_loss(shift_logits, shift_labels)
        loss = gen_loss

        return loss
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if labels is not None and labels.shape[1] != input_ids.shape[1]:

            input_ids = torch.cat([input_ids, labels,torch.tensor([[self.args.token_eos_id]]*labels.shape[0],dtype=input_ids.dtype,device=input_ids.device)], dim=-1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], labels.shape[1]+1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=-1,
            )
            valid_label_len =  labels.shape[1]+1
            labels = copy.deepcopy(input_ids)
            if self.args.only_train_response:
                labels[:, :-valid_label_len] = -100

            
            

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss,logits_to_keep==0意味着全序列都用 
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            if labels.shape[1] == logits.shape[1]:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.total_loss(shift_logits, shift_labels)
            else:
                assert False

            #     loss = torch.tensor(10.0,requires_grad=True).to(logits.device)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

