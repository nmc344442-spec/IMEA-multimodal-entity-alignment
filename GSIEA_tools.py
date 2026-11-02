# from __future__ import absolute_import
# from __future__ import unicode_literals
# from __future__ import division
# from __future__ import print_function

# from typing import Tuple, Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import math

# from transformers.activations import ACT2FN
# from transformers.pytorch_utils import apply_chunking_to_forward

# from .Tool_model import GAT, GCN

# class MultiModalEncoder(nn.Module):
#     def __init__(self,args,
#                  ent_num,
#                  img_feature_dim,
#                  char_feature_dim=None,
#                  use_project_head=False,
#                  attr_input_dim=1000
#                  ):
#         super(MultiModalEncoder,self).__init__()

#         self.args = args
#         attr_dim = self.args.attr_dim
#         img_dim = self.args.img_dim
#         name_dim = self.args.name_dim
#         char_dim = self.args.char_dim
#         dropout = self.args.dropout
#         self.ENT_NUM = ent_num
#         self.use_project_head = use_project_head

#         self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
#         self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
#         self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

#         self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
#         nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
#         self.entity_emb.requires_grad = True

#         self.rel_fc = nn.Linear(1000, attr_dim)
#         self.att_fc = nn.Linear(attr_input_dim, attr_dim)
#         self.img_fc = nn.Linear(img_feature_dim, img_dim)
#         self.name_fc = nn.Linear(300, char_dim)
#         self.char_fc = nn.Linear(char_feature_dim, char_dim)

#         if self.args.structure_encoder == 'gcn':
#             self.cross_graph_model = GCN(self.n_units[0],self.n_units[1],self.n_units[2],dropout=self.args.dropout)
#         elif self.args.structure_encoder == 'gat':
#             self.cross_graph_model = GAT(n_units=self.n_units,n_heads=self.n_heads,dropout=args.dropout,attn_dropout=args.attn_dropout,instance_normalization=self.args.instance_normalization,diag=True)

#         self.fusion = Fusion(args)


#     def forward(self,input_idx,adj,img_features=None,rel_features=None,att_features=None,name_features=None,char_features=None,importance_scores=None):
#         if self.args.w_gcn:
#             gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj, importance_scores=importance_scores)
#         else:
#             gph_emb = None
#         if self.args.w_img:
#             img_emb = self.img_fc(img_features)
#         else:
#             img_emb = None
#         if self.args.w_rel:
#             rel_emb = self.rel_fc(rel_features)
#         else:
#             rel_emb = None
#         if self.args.w_attr:
#             att_emb = self.att_fc(att_features)
#         else:
#             att_emb = None
#         if self.args.w_name and name_features is not None:
#             name_emb = self.name_fc(name_features)
#         else:
#             name_emb = None
#         if self.args.w_char and char_features is not None:
#             char_emb = self.char_fc(char_features)
#         else:
#             char_emb = None

#         joint_emb,joint_emb_fz,hidden_states,weight_norm = self.fusion(gph_emb,[rel_emb,att_emb,img_emb,name_emb,char_emb])

#         return gph_emb,img_emb,rel_emb,att_emb,name_emb,char_emb,joint_emb,joint_emb_fz,hidden_states,weight_norm

# class Fusion(nn.Module):
#     def __init__(self,args):
#         super().__init__()
#         self.args = args
#         self.gphLayer = GphLayer(args)
#         self.fusion = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
#         params = torch.ones(6,requires_grad=True)
#         self.weight_raw = torch.nn.Parameter(params)  # 模态全局权重

#     def forward(self,gph_emb,embs):
#         embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
#         modal_num = len(embs)
#         hidden_states = torch.stack(embs,dim=1)

#         hidden_states_gph = torch.stack((gph_emb,),dim=1)
#         gph_output,gph_attention,kvs = self.gphLayer(hidden_states_gph,output_attentions=True)

#         for i, layer_module in enumerate(self.fusion):
#             layer_outputs = layer_module(hidden_states,kvs,output_attentions=True)
#             hidden_states = layer_outputs[0]

#         attention = gph_attention + layer_outputs[1]
#         attention_pro = torch.sum(attention,dim=-3)
#         attention_pro_comb = torch.sum(attention_pro,dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
#         weight_norm = F.softmax(attention_pro_comb, dim=-1)
#         embs.insert(0,gph_emb)
#         modal_num_2 = len(embs)
#         embs = [weight_norm[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num_2)]
#         joint_emb = torch.cat(embs,dim=1)

#         hidden_states = torch.cat([hidden_states_gph,hidden_states],dim=1)

#         weight_norm_fz = F.softmax(self.weight_raw,dim=0)
#         emb_fz = [weight_norm_fz[idx] * F.normalize(embs[idx]) for idx in range(modal_num_2)]
#         joint_emb_fz = torch.cat(emb_fz,dim=1)

#         return joint_emb,joint_emb_fz,hidden_states,weight_norm

# class GphSelfAttention(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         assert config.hidden_size % config.num_attention_heads == 0
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.scaling = self.attention_head_size ** -0.5

#         self.dropout = nn.Dropout(0.1)

#     def transpose_for_scores(self,x:torch.Tensor) -> torch.Tensor:
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(
#             self,
#             hidden_states:torch.Tensor,
#             output_attentions=False,
#     ):
#         mixed_query_layer = self.query(hidden_states) * self.scaling
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         kvs = (key_layer,value_layer)

#         attention_socres = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_socres = attention_socres / math.sqrt(self.attention_head_size)
#         attention_probs = nn.functional.softmax(attention_socres, dim=-1)
#         attention_probs = self.dropout(attention_probs)
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

#         context_layer = context_layer.view(new_context_layer_shape)

#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

#         return outputs,kvs

# class GphSelfOutput(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self,hidden_states:torch.Tensor,input_tensor:torch.Tensor)->torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states

# class GphAttention(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.self = GphSelfAttention(config)
#         self.output = GphSelfOutput(config)

#     def forward(self,hidden_states:torch.Tensor,output_attentions=False):
#         self_output,kvs = self.self(
#             hidden_states,output_attentions,
#         )
#         attention_output = self.output(self_output[0],hidden_states)
#         attention_probs = self_output[1]

#         return attention_output,attention_probs,kvs

# class GphIntermediate(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         self.intermediate_act_fn = ACT2FN["gelu"]

#     def forward(self,hidden_states:torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states

# class GphOutput(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=1e-12)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self,hidden_states:torch.Tensor,input_tensor:torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states

# class GphLayer(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.config = config
#         self.chunk_size_feed_forward = 0
#         self.seq_len_dim = 1
#         self.attention = GphAttention(config)
#         if self.config.use_intermediate:
#             self.intermediate = GphIntermediate(config)
#         self.output = GphOutput(config)

#     def forward(self,hidden_states:torch.Tensor,output_attentions=False):
#         self_attention_output, self_attention_probs, kvs = self.attention(
#             hidden_states,
#             output_attentions=output_attentions
#         )
#         if not self.config.use_intermediate:
#             return self_attention_output, self_attention_probs, kvs
#         layer_output = apply_chunking_to_forward(
#             self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, self_attention_output
#         )

#         return layer_output, self_attention_probs, kvs

#     def feed_forward_chunk(self,self_attention_output):
#         intermediate_output = self.intermediate(self_attention_output)
#         layer_output = self.output(intermediate_output,self_attention_output)
#         return layer_output

# class BertSelfAttention(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.kv_weight = nn.Parameter(torch.tensor(1.0))

#         self.scaling =self.attention_head_size ** -0.5

#         self.dropout = nn.Dropout(0.1)

#     def transpose_for_scores(self,x:torch.Tensor) -> torch.Tensor:
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
#         x = x.view(new_x_shape)
#         return x.permute(0,2,1,3)

#     def forward(self,hidden_states:torch.Tensor,kvs:torch.Tensor,output_attentions=False):
#         mixed_query_layer = self.query(hidden_states) * self.scaling
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         if kvs is not None:
#             key_layer = torch.cat([self.kv_weight * kvs[0],key_layer],dim=2)
#             value_layer = torch.cat([self.kv_weight * kvs[1],value_layer],dim=2)

#         attention_socres = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_socres = attention_socres / math.sqrt(self.attention_head_size)

#         attention_probs = nn.functional.softmax(attention_socres, dim=-1)

#         attention_probs = self.dropout(attention_probs)
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

#         context_layer = context_layer.view(new_context_layer_shape)

#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
#         return outputs

# class BertSelfOutput(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size,config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=1e-12)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self,hidden_states:torch.Tensor,input_tensor:torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states

# class BertAttention(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.self = BertSelfAttention(config)
#         self.output = BertSelfOutput(config)

#     def forward(self,hidden_states:torch.Tensor,kvs,output_attentions=False):
#         self_outputs = self.self(
#             hidden_states,
#             kvs,
#             output_attentions,
#         )
#         attention_output = self.output(self_outputs[0],hidden_states)
#         outputs = (attention_output,) + self_outputs[1:]

#         return outputs

# class BertIntermediate(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size,config.intermediate_size)
#         self.intermediate_act_fn = ACT2FN["gelu"]

#     def forward(self,hidden_states:torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states

# class BertOutput(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=1e-12)
#         self.dropout = nn.Dropout(0.1)
#         self.conv = ConvModule(in_channels=3)

#     def forward(self,hidden_states:torch.Tensor,input_tensor:torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.conv(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states

# class BertLayer(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.config = config
#         self.chunk_size_feed_forward = 0
#         self.seq_len_dim = 1
#         self.attention = BertAttention(config)
#         if self.config.use_intermediate:
#             self.intermediate = BertIntermediate(config)
#         self.output = BertOutput(config)

#     def forward(self,hidden_states:torch.Tensor,kvs,output_attentions=False):
#         self_attention_outputs = self.attention(
#             hidden_states,
#             kvs,
#             output_attentions=output_attentions,
#         )
#         if not self.config.use_intermediate:
#             return (self_attention_outputs[0],self_attention_outputs[1])

#         attention_output = self_attention_outputs[0]
#         outputs = self_attention_outputs[1]
#         layer_output = apply_chunking_to_forward(
#             self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
#         )
#         outputs = (layer_output, outputs)
#         return outputs

#     def feed_forward_chunk(self,attention_output):
#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output,attention_output)
#         return layer_output

# class DepthwiseConv1d(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias=False):
#         super(DepthwiseConv1d,self).__init__()
#         assert out_channels % in_channels == 0
#         self.conv = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             groups=in_channels,
#             stride=stride,
#             padding=padding,
#             bias=bias
#         )

#     def forward(self,inputs):
#         return self.conv(inputs)

# class PointwiseConv1d(nn.Module):
#     def __init__(self,in_channels,out_channels,stride,padding,bias):
#         super(PointwiseConv1d, self).__init__()
#         self.conv = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=stride,
#             padding=padding,
#             bias=bias,
#         )

#     def forward(self,inputs):
#         return self.conv(inputs)

# class Transpose(nn.Module):
#     def __init__(self, shape: tuple):
#         super(Transpose, self).__init__()
#         self.shape = shape

#     def forward(self, x):
#         return x.transpose(*self.shape)

# class ConvModule(nn.Module):
#     def __init__(self,in_channels,kernel_size=31,expansion_factor=2,droupout=0.1):
#         super(ConvModule,self).__init__()
#         assert (kernel_size - 1) % 2 == 0
#         assert expansion_factor == 2

#         self.sequential = nn.Sequential(
#             PointwiseConv1d(in_channels,in_channels * expansion_factor,stride=1,padding=0,bias=True),
#             nn.GLU(dim=1),
#             DepthwiseConv1d(in_channels,in_channels,kernel_size,stride=1,padding=(kernel_size-1)//2),
#             nn.BatchNorm1d(in_channels),
#             PointwiseConv1d(in_channels,in_channels,stride=1,padding=0,bias=True),
#         )

#     def forward(self,inputs):
#         return self.sequential(inputs)


# from __future__ import absolute_import
# from __future__ import unicode_literals
# from __future__ import division
# from __future__ import print_function

# from typing import Tuple, Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import math

# from transformers.activations import ACT2FN
# from transformers.pytorch_utils import apply_chunking_to_forward

# from .Tool_model import GAT, GCN

# class MultiModalEncoder(nn.Module):
#     def __init__(self,args,
#                  ent_num,
#                  img_feature_dim,
#                  char_feature_dim=None,
#                  use_project_head=False,
#                  attr_input_dim=1000
#                  ):
#         super(MultiModalEncoder,self).__init__()

#         self.args = args
#         attr_dim = self.args.attr_dim
#         img_dim = self.args.img_dim
#         name_dim = self.args.name_dim
#         char_dim = self.args.char_dim
#         dropout = self.args.dropout
#         self.ENT_NUM = ent_num
#         self.use_project_head = use_project_head

#         self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
#         self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
#         self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

#         self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
#         nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
#         self.entity_emb.requires_grad = True

#         self.rel_fc = nn.Linear(1000, attr_dim)
#         self.att_fc = nn.Linear(attr_input_dim, attr_dim)
#         self.img_fc = nn.Linear(img_feature_dim, img_dim)
#         self.name_fc = nn.Linear(300, char_dim)
#         self.char_fc = nn.Linear(char_feature_dim, char_dim)

#         if self.args.structure_encoder == 'gcn':
#             self.cross_graph_model = GCN(self.n_units[0],self.n_units[1],self.n_units[2],dropout=self.args.dropout)
#         elif self.args.structure_encoder == 'gat':
#             self.cross_graph_model = GAT(n_units=self.n_units,n_heads=self.n_heads,dropout=args.dropout,attn_dropout=args.attn_dropout,instance_normalization=self.args.instance_normalization,diag=True)

#         self.fusion = Fusion(args)


#     def forward(self,input_idx,adj,img_features=None,rel_features=None,att_features=None,name_features=None,char_features=None):
#         if self.args.w_gcn:
#             gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
#         else:
#             gph_emb = None
#         if self.args.w_img:
#             img_emb = self.img_fc(img_features)
#         else:
#             img_emb = None
#         if self.args.w_rel:
#             rel_emb = self.rel_fc(rel_features)
#         else:
#             rel_emb = None
#         if self.args.w_attr:
#             att_emb = self.att_fc(att_features)
#         else:
#             att_emb = None
#         if self.args.w_name and name_features is not None:
#             name_emb = self.name_fc(name_features)
#         else:
#             name_emb = None
#         if self.args.w_char and char_features is not None:
#             char_emb = self.char_fc(char_features)
#         else:
#             char_emb = None

#         joint_emb,joint_emb_fz,hidden_states,weight_norm = self.fusion(gph_emb,[rel_emb,att_emb,img_emb,name_emb,char_emb])

#         return gph_emb,img_emb,rel_emb,att_emb,name_emb,char_emb,joint_emb,joint_emb_fz,hidden_states,weight_norm

# class Fusion(nn.Module):
#     def __init__(self,args):
#         super().__init__()
#         self.args = args
#         self.gphLayer = GphLayer(args)
#         self.fusion = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
#         params = torch.ones(6,requires_grad=True)
#         self.weight_raw = torch.nn.Parameter(params)  # 模态全局权重

#     def forward(self,gph_emb,embs):
#         embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
#         modal_num = len(embs)
#         hidden_states = torch.stack(embs,dim=1)

#         hidden_states_gph = torch.stack((gph_emb,),dim=1)
#         gph_output,gph_attention,kvs = self.gphLayer(hidden_states_gph,output_attentions=True)

#         for i, layer_module in enumerate(self.fusion):
#             layer_outputs = layer_module(hidden_states,kvs,output_attentions=True)
#             hidden_states = layer_outputs[0]

#         attention = gph_attention + layer_outputs[1]
#         attention_pro = torch.sum(attention,dim=-3)
#         attention_pro_comb = torch.sum(attention_pro,dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
#         weight_norm = F.softmax(attention_pro_comb, dim=-1)
#         embs.insert(0,gph_emb)
#         modal_num_2 = len(embs)
#         embs = [weight_norm[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num_2)]
#         joint_emb = torch.cat(embs,dim=1)

#         hidden_states = torch.cat([hidden_states_gph,hidden_states],dim=1)

#         weight_norm_fz = F.softmax(self.weight_raw,dim=0)
#         emb_fz = [weight_norm_fz[idx] * F.normalize(embs[idx]) for idx in range(modal_num_2)]
#         joint_emb_fz = torch.cat(emb_fz,dim=1)

#         return joint_emb,joint_emb_fz,hidden_states,weight_norm

# class GphSelfAttention(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         assert config.hidden_size % config.num_attention_heads == 0
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.scaling = self.attention_head_size ** -0.5

#         self.dropout = nn.Dropout(0.1)

#     def transpose_for_scores(self,x:torch.Tensor) -> torch.Tensor:
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(
#             self,
#             hidden_states:torch.Tensor,
#             output_attentions=False,
#     ):
#         mixed_query_layer = self.query(hidden_states) * self.scaling
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         kvs = (key_layer,value_layer)

#         attention_socres = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_socres = attention_socres / math.sqrt(self.attention_head_size)
#         attention_probs = nn.functional.softmax(attention_socres, dim=-1)
#         attention_probs = self.dropout(attention_probs)
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

#         context_layer = context_layer.view(new_context_layer_shape)

#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

#         return outputs,kvs

# class GphSelfOutput(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self,hidden_states:torch.Tensor,input_tensor:torch.Tensor)->torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states

# class GphAttention(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.self = GphSelfAttention(config)
#         self.output = GphSelfOutput(config)

#     def forward(self,hidden_states:torch.Tensor,output_attentions=False):
#         self_output,kvs = self.self(
#             hidden_states,output_attentions,
#         )
#         attention_output = self.output(self_output[0],hidden_states)
#         attention_probs = self_output[1]

#         return attention_output,attention_probs,kvs

# class GphIntermediate(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         self.intermediate_act_fn = ACT2FN["gelu"]

#     def forward(self,hidden_states:torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states

# class GphOutput(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=1e-12)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self,hidden_states:torch.Tensor,input_tensor:torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states

# class GphLayer(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.config = config
#         self.chunk_size_feed_forward = 0
#         self.seq_len_dim = 1
#         self.attention = GphAttention(config)
#         if self.config.use_intermediate:
#             self.intermediate = GphIntermediate(config)
#         self.output = GphOutput(config)

#     def forward(self,hidden_states:torch.Tensor,output_attentions=False):
#         self_attention_output, self_attention_probs, kvs = self.attention(
#             hidden_states,
#             output_attentions=output_attentions
#         )
#         if not self.config.use_intermediate:
#             return self_attention_output, self_attention_probs, kvs
#         layer_output = apply_chunking_to_forward(
#             self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, self_attention_output
#         )

#         return layer_output, self_attention_probs, kvs

#     def feed_forward_chunk(self,self_attention_output):
#         intermediate_output = self.intermediate(self_attention_output)
#         layer_output = self.output(intermediate_output,self_attention_output)
#         return layer_output

# class BertSelfAttention(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.kv_weight = nn.Parameter(torch.tensor(1.0))

#         self.scaling =self.attention_head_size ** -0.5

#         self.dropout = nn.Dropout(0.1)

#     def transpose_for_scores(self,x:torch.Tensor) -> torch.Tensor:
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
#         x = x.view(new_x_shape)
#         return x.permute(0,2,1,3)

#     def forward(self,hidden_states:torch.Tensor,kvs:torch.Tensor,output_attentions=False):
#         mixed_query_layer = self.query(hidden_states) * self.scaling
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         if kvs is not None:
#             key_layer = torch.cat([self.kv_weight * kvs[0],key_layer],dim=2)
#             value_layer = torch.cat([self.kv_weight * kvs[1],value_layer],dim=2)

#         attention_socres = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_socres = attention_socres / math.sqrt(self.attention_head_size)

#         attention_probs = nn.functional.softmax(attention_socres, dim=-1)

#         attention_probs = self.dropout(attention_probs)
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

#         context_layer = context_layer.view(new_context_layer_shape)

#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
#         return outputs

# class BertSelfOutput(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size,config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=1e-12)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self,hidden_states:torch.Tensor,input_tensor:torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states

# class BertAttention(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.self = BertSelfAttention(config)
#         self.output = BertSelfOutput(config)

#     def forward(self,hidden_states:torch.Tensor,kvs,output_attentions=False):
#         self_outputs = self.self(
#             hidden_states,
#             kvs,
#             output_attentions,
#         )
#         attention_output = self.output(self_outputs[0],hidden_states)
#         outputs = (attention_output,) + self_outputs[1:]

#         return outputs

# class BertIntermediate(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size,config.intermediate_size)
#         self.intermediate_act_fn = ACT2FN["gelu"]

#     def forward(self,hidden_states:torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states

# class BertOutput(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=1e-12)
#         self.dropout = nn.Dropout(0.1)
#         self.conv = ConvModule(in_channels=3)

#     def forward(self,hidden_states:torch.Tensor,input_tensor:torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.conv(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states

# class BertLayer(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.config = config
#         self.chunk_size_feed_forward = 0
#         self.seq_len_dim = 1
#         self.attention = BertAttention(config)
#         if self.config.use_intermediate:
#             self.intermediate = BertIntermediate(config)
#         self.output = BertOutput(config)

#     def forward(self,hidden_states:torch.Tensor,kvs,output_attentions=False):
#         self_attention_outputs = self.attention(
#             hidden_states,
#             kvs,
#             output_attentions=output_attentions,
#         )
#         if not self.config.use_intermediate:
#             return (self_attention_outputs[0],self_attention_outputs[1])

#         attention_output = self_attention_outputs[0]
#         outputs = self_attention_outputs[1]
#         layer_output = apply_chunking_to_forward(
#             self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
#         )
#         outputs = (layer_output, outputs)
#         return outputs

#     def feed_forward_chunk(self,attention_output):
#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output,attention_output)
#         return layer_output

# class DepthwiseConv1d(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias=False):
#         super(DepthwiseConv1d,self).__init__()
#         assert out_channels % in_channels == 0
#         self.conv = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             groups=in_channels,
#             stride=stride,
#             padding=padding,
#             bias=bias
#         )

#     def forward(self,inputs):
#         return self.conv(inputs)

# class PointwiseConv1d(nn.Module):
#     def __init__(self,in_channels,out_channels,stride,padding,bias):
#         super(PointwiseConv1d, self).__init__()
#         self.conv = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=stride,
#             padding=padding,
#             bias=bias,
#         )

#     def forward(self,inputs):
#         return self.conv(inputs)

# class Transpose(nn.Module):
#     def __init__(self, shape: tuple):
#         super(Transpose, self).__init__()
#         self.shape = shape

#     def forward(self, x):
#         return x.transpose(*self.shape)

# class ConvModule(nn.Module):
#     def __init__(self,in_channels,kernel_size=31,expansion_factor=2,droupout=0.1):
#         super(ConvModule,self).__init__()
#         assert (kernel_size - 1) % 2 == 0
#         assert expansion_factor == 2

#         self.sequential = nn.Sequential(
#             PointwiseConv1d(in_channels,in_channels * expansion_factor,stride=1,padding=0,bias=True),
#             nn.GLU(dim=1),
#             DepthwiseConv1d(in_channels,in_channels,kernel_size,stride=1,padding=(kernel_size-1)//2),
#             nn.BatchNorm1d(in_channels),
#             PointwiseConv1d(in_channels,in_channels,stride=1,padding=0,bias=True),
#         )

#     def forward(self,inputs):
#         return self.sequential(inputs)





from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward

from .Tool_model import GAT, GCN


class MultiModalEncoder(nn.Module):
    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 use_project_head=False,
                 attr_input_dim=1000
                 ):
        super(MultiModalEncoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        name_dim = self.args.name_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(attr_input_dim, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)

        if self.args.structure_encoder == 'gcn':
            self.cross_graph_model = GCN(
                self.n_units[0],
                self.n_units[1],
                self.n_units[2],
                dropout=self.args.dropout
            )
        elif self.args.structure_encoder == 'gat':
            self.cross_graph_model = GAT(
                n_units=self.n_units,
                n_heads=self.n_heads,
                dropout=args.dropout,
                attn_dropout=args.attn_dropout,
                instance_normalization=self.args.instance_normalization,
                diag=True,
                args=args
            )

        # ===== 新增：全局重要实体分支 =====
        # global_topk = 0 时，关闭全局分支，模型退化为原始 GAT
        # global_topk = 50 时，每个实体都会从全图 Top-50 重要实体中聚合信息
        self.global_topk = 30
    
        self.global_bias = 8.0

        gph_dim = self.n_units[-1]

        # 当前实体的 query 来自原始 GAT 输出 h_local
        self.global_query = nn.Linear(gph_dim, gph_dim, bias=False)

        # 全局重要实体的 key/value 来自原始 entity embedding
        self.global_key = nn.Linear(self.input_dim, gph_dim, bias=False)
        self.global_value = nn.Linear(self.input_dim, gph_dim, bias=False)

        # 门控融合 local / global
        self.global_gate = nn.Linear(gph_dim * 2, 1)

        # 初始时更偏向 local GAT，避免 global 分支一开始破坏原模型
        nn.init.constant_(self.global_gate.bias, 0.0)

        self.fusion = Fusion(args)

    def global_importance_aggregate(self, x, h_local, importance_scores):
        """
        x:
            原始实体 embedding，shape = [N, input_dim]
        h_local:
            原始 GAT 输出的局部结构嵌入，shape = [N, gph_dim]
        importance_scores:
            每个实体的全局重要性分数，shape = [N]

        返回:
            h_global，shape = [N, gph_dim]
        """
        if importance_scores is None or self.global_topk <= 0:
            return torch.zeros_like(h_local)

        device = x.device
        importance_scores = importance_scores.to(device).view(-1)

        N = x.size(0)
        topk = min(self.global_topk, N)

        # 1. 选出全图 Top-K 重要实体
        _, global_ids = torch.topk(importance_scores, topk)

        global_x = x[global_ids]  # [topk, input_dim]

        # 2. 计算 query / key / value
        q = self.global_query(h_local)      # [N, gph_dim]
        k = self.global_key(global_x)       # [topk, gph_dim]
        v = self.global_value(global_x)     # [topk, gph_dim]

        # 3. 每个实体对全局重要实体做 attention
        att_score = torch.matmul(q, k.t()) / math.sqrt(q.size(-1))  # [N, topk]

        # 4. 将重要性分数作为 bias 加入 global attention
        imp = importance_scores[global_ids]
        imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-9)

        att_score = att_score + self.global_bias * imp.view(1, -1)

        beta = F.softmax(att_score, dim=-1)

        # 5. 聚合全局重要实体表示
        h_global = torch.matmul(beta, v)  # [N, gph_dim]

        return h_global

    def forward(self, input_idx, adj, img_features=None, rel_features=None,
                att_features=None, name_features=None, char_features=None,
                importance_scores=None):

        if self.args.w_gcn:
            x = self.entity_emb(input_idx)

            # 1. local 分支：原始 GAT，只聚合邻居，不再直接混合 importance
            h_local = self.cross_graph_model(x, adj)

            # 2. global 分支：聚合全局 Top-K 重要实体
            if importance_scores is not None and self.global_topk > 0:
                h_global = self.global_importance_aggregate(
                    x,
                    h_local,
                    importance_scores
                )

                # 3. gate 融合 local 和 global
                eta = torch.sigmoid(
                    self.global_gate(
                        torch.cat([h_local, h_global], dim=-1)
                    )
                )

                gph_emb = eta * h_local + (1.0 - eta) * h_global
            else:
                # global_topk = 0 时，退化为原始 GAT
                gph_emb = h_local
        else:
            gph_emb = None

        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None

        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None

        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None

        if self.args.w_name and name_features is not None:
            name_emb = self.name_fc(name_features)
        else:
            name_emb = None

        if self.args.w_char and char_features is not None:
            char_emb = self.char_fc(char_features)
        else:
            char_emb = None

        joint_emb, joint_emb_fz, hidden_states, weight_norm = self.fusion(
            gph_emb,
            [rel_emb, att_emb, img_emb, name_emb, char_emb]
        )

        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, joint_emb_fz, hidden_states, weight_norm


class Fusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gphLayer = GphLayer(args)
        self.fusion = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        params = torch.ones(6, requires_grad=True)
        self.weight_raw = torch.nn.Parameter(params)  # 模态全局权重

    def forward(self, gph_emb, embs):
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        modal_num = len(embs)
        hidden_states = torch.stack(embs, dim=1)

        hidden_states_gph = torch.stack((gph_emb,), dim=1)
        gph_output, gph_attention, kvs = self.gphLayer(hidden_states_gph, output_attentions=True)

        for i, layer_module in enumerate(self.fusion):
            layer_outputs = layer_module(hidden_states, kvs, output_attentions=True)
            hidden_states = layer_outputs[0]

        attention = gph_attention + layer_outputs[1]
        attention_pro = torch.sum(attention, dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        weight_norm = F.softmax(attention_pro_comb, dim=-1)

        embs.insert(0, gph_emb)
        modal_num_2 = len(embs)
        embs = [weight_norm[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num_2)]
        joint_emb = torch.cat(embs, dim=1)

        hidden_states = torch.cat([hidden_states_gph, hidden_states], dim=1)

        weight_norm_fz = F.softmax(self.weight_raw, dim=0)
        emb_fz = [weight_norm_fz[idx] * F.normalize(embs[idx]) for idx in range(modal_num_2)]
        joint_emb_fz = torch.cat(emb_fz, dim=1)

        return joint_emb, joint_emb_fz, hidden_states, weight_norm


class GphSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.scaling = self.attention_head_size ** -0.5

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states) * self.scaling
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        kvs = (key_layer, value_layer)

        attention_socres = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_socres = attention_socres / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_socres, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs, kvs


class GphSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GphAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = GphSelfAttention(config)
        self.output = GphSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions=False):
        self_output, kvs = self.self(
            hidden_states, output_attentions,
        )
        attention_output = self.output(self_output[0], hidden_states)
        attention_probs = self_output[1]

        return attention_output, attention_probs, kvs


class GphIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class GphOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GphLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = GphAttention(config)
        if self.config.use_intermediate:
            self.intermediate = GphIntermediate(config)
        self.output = GphOutput(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions=False):
        self_attention_output, self_attention_probs, kvs = self.attention(
            hidden_states,
            output_attentions=output_attentions
        )
        if not self.config.use_intermediate:
            return self_attention_output, self_attention_probs, kvs

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            self_attention_output
        )

        return layer_output, self_attention_probs, kvs

    def feed_forward_chunk(self, self_attention_output):
        intermediate_output = self.intermediate(self_attention_output)
        layer_output = self.output(intermediate_output, self_attention_output)
        return layer_output


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.kv_weight = nn.Parameter(torch.tensor(1.0))

        self.scaling = self.attention_head_size ** -0.5

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, kvs: torch.Tensor, output_attentions=False):
        mixed_query_layer = self.query(hidden_states) * self.scaling
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if kvs is not None:
            key_layer = torch.cat([self.kv_weight * kvs[0], key_layer], dim=2)
            value_layer = torch.cat([self.kv_weight * kvs[1], value_layer], dim=2)

        attention_socres = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_socres = attention_socres / math.sqrt(self.attention_head_size)

        attention_probs = nn.functional.softmax(attention_socres, dim=-1)

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, kvs, output_attentions=False):
        self_outputs = self.self(
            hidden_states,
            kvs,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.conv = ConvModule(in_channels=3)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        if self.config.use_intermediate:
            self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, kvs, output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states,
            kvs,
            output_attentions=output_attentions,
        )
        if not self.config.use_intermediate:
            return (self_attention_outputs[0], self_attention_outputs[1])

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output
        )
        outputs = (layer_output, outputs)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, inputs):
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, bias):
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs):
        return self.conv(inputs)


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class ConvModule(nn.Module):
    def __init__(self, in_channels, kernel_size=31, expansion_factor=2, droupout=0.1):
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        assert expansion_factor == 2

        self.sequential = nn.Sequential(
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            nn.GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
        )

    def forward(self, inputs):
        return self.sequential(inputs)