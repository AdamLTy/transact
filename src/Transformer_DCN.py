import torch
from fuxictr.utils import not_in_whitelist
from torch import nn
import random
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2


class Transformer_DCN(BaseModel):
    def __init__(self,
                 feature_map,                    # 特征映射，定义输入特征的结构和配置
                 model_id="Transformer_DCN",     # 模型标识符
                 gpu=-1,                         # GPU设备ID，-1表示使用CPU
                 hidden_activations="ReLU",      # 隐藏层激活函数类型
                 dcn_cross_layers=3,             # DCNv2交叉网络层数
                 dcn_hidden_units=[1024, 512, 256],  # DCN并行DNN各隐藏层单元数
                 mlp_hidden_units=[64, 32],      # 最终MLP各隐藏层单元数
                 num_heads=1,                    # Transformer多头注意力头数
                 transformer_layers=2,           # Transformer编码器层数
                 transformer_dropout=0.2,        # Transformer内部dropout率
                 dim_feedforward=256,            # Transformer前馈网络隐藏层维度
                 learning_rate=5e-4,             # 学习率
                 embedding_dim=64,               # 特征嵌入维度
                 net_dropout=0.2,                # 网络dropout率
                 first_k_cols=16,                # 取Transformer输出最后k个位置
                 batch_norm=False,               # 是否使用批归一化
                 concat_max_pool=True,           # 是否拼接max pooling结果
                 accumulation_steps=1,           # 梯度累积步数
                 embedding_regularizer=None,     # 嵌入层正则化器
                 net_regularizer=None,           # 网络正则化器
                 **kwargs):
        super().__init__(feature_map,
                         model_id=model_id,
                         gpu=gpu,
                         embedding_regularizer=embedding_regularizer,
                         net_regularizer=net_regularizer,
                         **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        # 计算物品信息的总嵌入维度
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)

        # Transformer输入维度 = item_info_dim * 2 (目标item + 序列item拼接)
        transformer_in_dim = self.item_info_dim * 2

        self.accumulation_steps = accumulation_steps
        # 特征嵌入层，将原始特征转换为嵌入向量
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        # Transformer编码器，处理序列信息
        self.transformer_encoder = Transformer(
            transformer_in_dim,                # 输入维度: item_info_dim * 2
            dim_feedforward=dim_feedforward,   # 前馈网络隐藏层维度
            num_heads=num_heads,              # 注意力头数
            dropout=transformer_dropout,      # dropout率
            transformer_layers=transformer_layers,  # 编码器层数
            first_k_cols=first_k_cols,       # 取最后k个位置的输出
            concat_max_pool=concat_max_pool  # 是否拼接max pooling
        )
        # Transformer输出维度: (first_k_cols + max_pool) * transformer_in_dim
        seq_out_dim = (first_k_cols + int(concat_max_pool)) * transformer_in_dim

        # DCN输入维度 = 所有特征嵌入维度 + Transformer输出维度
        dcn_in_dim = feature_map.sum_emb_out_dim() + seq_out_dim
        
        # DCNv2交叉网络，学习特征间的高阶交互
        self.crossnet = CrossNetV2(dcn_in_dim, dcn_cross_layers)
        
        # 并行DNN网络，与交叉网络并行处理特征
        self.parallel_dnn = MLP_Block(input_dim=dcn_in_dim,
                                      output_dim=None,  # 输出隐藏层而非最终预测
                                      hidden_units=dcn_hidden_units,
                                      hidden_activations=hidden_activations,
                                      output_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm)
        
        # DCN输出维度 = 交叉网络输出维度 + 并行DNN最后一层维度
        dcn_out_dim = dcn_in_dim + dcn_hidden_units[-1]
        
        # 最终MLP，输出CTR预测结果
        self.mlp = MLP_Block(input_dim=dcn_out_dim,
                             output_dim=1,  # 输出1维，表示点击概率
                             hidden_units=mlp_hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.output_activation)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """前向传播
        Args:
            inputs: 包含batch_dict, item_dict, mask的输入数据
        Returns:
            dict: 包含y_pred的预测结果
        """
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict:  # 如果batch特征不为空
            # 获取批次特征嵌入: [batch_size, total_emb_dim]
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(feature_emb)
        feat_emb = torch.cat(emb_list, dim=-1)  # [batch_size, total_emb_dim]
        
        # 获取物品特征嵌入并重塑形状
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)  # [batch_size*seq_len, item_info_dim]
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)  # [batch_size, seq_len, item_info_dim]

        # 分离目标物品和序列物品嵌入
        target_emb = item_feat_emb[:, -1, :]      # [batch_size, item_info_dim] 目标物品
        sequence_emb = item_feat_emb[:, 0:-1, :]  # [batch_size, seq_len-1, item_info_dim] 历史序列
        
        # Transformer处理序列信息
        # 输出形状: [batch_size, (first_k_cols + concat_max_pool) * transformer_in_dim]
        transformer_emb = self.transformer_encoder(
            target_emb, sequence_emb, mask=mask
        )

        # 拼接所有特征作为DCN输入
        # 形状: [batch_size, total_emb_dim + item_info_dim + transformer_out_dim]
        dcn_in_emb = torch.cat([feat_emb, target_emb, transformer_emb], dim=-1)
        
        # DCN交叉网络和并行DNN处理
        cross_out = self.crossnet(dcn_in_emb)    # [batch_size, dcn_in_dim]
        dnn_out = self.parallel_dnn(dcn_in_emb)  # [batch_size, dcn_hidden_units[-1]]
        
        # 最终MLP输出预测结果
        y_pred = self.mlp(torch.cat([cross_out, dnn_out], dim=-1))  # [batch_size, 1]
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

    def get_labels(self, inputs):
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss


class Transformer(nn.Module):
    def __init__(self,
                 transformer_in_dim,        # Transformer输入维度，等于item_dim*2（序列+目标item拼接后的维度）
                 dim_feedforward=64,        # Transformer前馈网络的隐藏层维度
                 num_heads=1,              # 多头注意力机制的头数
                 dropout=0,                # Dropout比例
                 transformer_layers=1,     # Transformer编码器层数
                 first_k_cols=16,         # 取Transformer输出的最后k个位置用于后续处理
                 concat_max_pool=True):   # 是否使用max pooling并拼接到输出中
        super(Transformer, self).__init__()
        self.concat_max_pool = concat_max_pool
        self.first_k_cols = first_k_cols
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_in_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 输入张量格式为[batch_size, seq_len, features]而非默认的[seq_len, batch_size, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        if self.concat_max_pool:
            # 线性投影层用于对max pooling结果进行可学习的特征变换
            # 提供参数化的特征聚合能力，比简单max pooling有更强的表达能力
            self.out_linear = nn.Linear(transformer_in_dim, transformer_in_dim)

    def forward(self, target_emb, sequence_emb, mask=None):
        # target_emb: [batch_size, item_dim], sequence_emb: [batch_size, seq_len, item_dim]
        # concat action sequence emb with target emb
        seq_len = sequence_emb.size(1)  # 获取序列长度
        # 将target_emb扩展并与sequence_emb拼接: [batch_size, seq_len, item_dim*2]
        concat_seq_emb = torch.cat([sequence_emb,
                                    target_emb.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        
        # get sequence mask (1's are masked)
        key_padding_mask = self.adjust_mask(mask).bool()  # [batch_size, seq_len]
        
        # Transformer编码器处理拼接后的序列: [batch_size, seq_len, transformer_in_dim]
        tfmr_out = self.transformer_encoder(src=concat_seq_emb,
                                            src_key_padding_mask=key_padding_mask)
        
        # 将被mask的位置填充为0: [batch_size, seq_len, transformer_in_dim]
        tfmr_out = tfmr_out.masked_fill(
            key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), 0.
        )
        
        # process the transformer output
        output_concat = []
        # 取最后first_k_cols个位置的输出并展平: [batch_size, first_k_cols * transformer_in_dim]
        output_concat.append(tfmr_out[:, -self.first_k_cols:].flatten(start_dim=1))
        
        if self.concat_max_pool:
            # Apply max pooling to the transformer output
            # 将被mask的位置填充为极小值用于max pooling
            tfmr_out = tfmr_out.masked_fill(
                key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), -1e9
            )
            # 在序列维度上进行max pooling: [batch_size, transformer_in_dim]
            pooled_out = self.out_linear(tfmr_out.max(dim=1).values)
            output_concat.append(pooled_out)
        
        # 最终输出维度: [batch_size, (first_k_cols + concat_max_pool) * transformer_in_dim]
        return torch.cat(output_concat, dim=-1)

    def adjust_mask(self, mask):
        # make sure not all actions in the sequence are masked
        # Check if all elements in a sequence are masked (all 1's): [batch_size]
        fully_masked = mask.all(dim=-1)
        # For sequences that are fully masked, unmask the last position to prevent empty sequences
        mask[fully_masked, -1] = 0
        return mask
