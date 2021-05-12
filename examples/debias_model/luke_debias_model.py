from .config import BertConfig
from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import device, dtype



class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # split the hidden states to num_attention_heads for multi-head attention
    # hidden_size = self.num_attention_heads * self.attention_head_size
    # x [batch_size, seq_len, hidden_size]
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # each key, query, value is of [bs, self.num_attention_heads, seq_len, self.attention_head_size]
    # eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # todo

    softmax=nn.Softmax(dim=-1)

    atten = torch.matmul(self.dropout(softmax(torch.matmul(query,key.transpose(-2,-1))/math.sqrt(query.shape[-1])+attention_mask)),value)

    atten=atten.transpose(2,1)
    return  atten.reshape(atten.shape[0],atten.shape[1],-1)  
    # raise NotImplementedError

  def forward(self, hidden_states, attention_mask):
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # self attention
    self.self_attention = BertSelfAttention(config)
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # layer out
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    input: the input
    output: the input that requires the sublayer to transform
    dense_layer, dropout: the sublayer
    ln_layer: layer norm that takes input+sublayer(output)
    """
    # todo

    return ln_layer(input+dropout(dense_layer(output)))


  def forward(self, hidden_states, attention_mask):
    # todo
    # multi-head attention
    atten=self.self_attention(hidden_states, attention_mask)
    # add-norm layer
    value = self.add_norm(hidden_states,atten,self.attention_dense, self.attention_dropout,self.attention_layer_norm)
    # feed forward
    value_interm = self.interm_af(self.interm_dense(value))
    # another add-norm layer
    final_out = self.add_norm(value,value_interm,self.out_dense,self.out_dropout,self.out_layer_norm)

    return final_out

    # raise NotImplementedError


class BertModel(nn.Module):
  def __init__(self, ):
    super().__init__()
    config = BertConfig
    self.config=config
    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()


    self.predict_layer = nn.Linear(config.hidden_size,2)
  def init_weights(self):
    # Initialize weights
    self.apply(self._init_weights)

  def _init_weights(self, module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()
    
  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]



    inputs_embeds = self.word_embedding(input_ids)


    pos_embeds = self.pos_embedding(self.position_ids[:,:seq_length])

    # get token type ids, since we are not consider token type, just a placeholder
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)
    # add three embeddings together
    embeds = inputs_embeds + tk_type_embeds + pos_embeds

    # layer norm and dropout
    embeds = self.embed_layer_norm(embeds)
    embeds = self.embed_dropout(embeds)
    return embeds
    # raise NotImplementedError
  def get_parameter_dtype(self,parameter: Union[nn.Module]):
      try:
        return next(parameter.parameters()).dtype
      except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
          tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
          return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype
  @property
  def dtype(self) -> dtype:
    return self.get_parameter_dtype(self)  
  
  def get_extended_attention_mask(self, attention_mask: Tensor, dtype) -> Tensor:
      # attention_mask [batch_size, seq_length]
      assert attention_mask.dim() == 2
      # [batch_size, 1, 1, seq_length] for multi-head attention
      extended_attention_mask = attention_mask[:, None, None, :]
      extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
      extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
      return extended_attention_mask
  def encode(self, hidden_states, attention_mask):
    # get the extended attention mask for self attention
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, self.dtype)

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask, downstream=False):
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)
    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
    # get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)
    if downstream:
        prediction = self.predict_layer(sequence_output)
        return prediction
    else:
        return sequence_output

