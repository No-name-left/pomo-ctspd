
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CTSPdModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = CTSPd_Encoder(**model_params)
        self.decoder = CTSPd_Decoder(**model_params)
        self.encoded_nodes: Optional[torch.Tensor] = None
        self.group_ids: Optional[torch.Tensor] = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        problems = reset_state.problems
        self.group_ids = problems[:, :, 2].long()
        self.encoded_nodes = self.encoder(problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def set_training_epoch(self, epoch, total_epochs):
        self.encoder.set_training_epoch(epoch, total_epochs)

    def forward(self, state):
        batch_size = int(state.BATCH_IDX.size(0))
        pomo_size = int(state.BATCH_IDX.size(1))
        encoded_nodes = self.encoded_nodes
        if encoded_nodes is None:
            raise RuntimeError("Model.pre_forward must be called before forward.")
        current_node = state.current_node

        if current_node is None:
            selected = _select_initial_nodes(state, batch_size, pomo_size, encoded_nodes.device)
            prob = torch.ones(size=(batch_size, pomo_size), device=encoded_nodes.device)

            encoded_first_node = _get_encoding(encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            ninf_mask = state.ninf_mask
            if ninf_mask is None:
                raise RuntimeError("Step state has no ninf_mask.")

            encoded_last_node = _get_encoding(encoded_nodes, current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(
                encoded_last_node,
                ninf_mask=ninf_mask,
                group_ids=self.group_ids,
                current_min_priority=state.current_min_priority,
            )
            # shape: (batch, pomo, problem)

            if self.training or self.model_params['eval_type'] == 'softmax':
                sample_probs = _apply_sampling_controls(probs, self.model_params, self.training)
                while True:
                    selected = sample_probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = sample_probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None


        return selected, prob


def _select_initial_nodes(state, batch_size, pomo_size, device):
    ninf_mask = state.ninf_mask
    if ninf_mask is None:
        return torch.arange(pomo_size, device=device)[None, :].expand(batch_size, pomo_size)

    legal_mask = torch.isfinite(ninf_mask[:, 0, :]).to(device=device)
    selected = torch.empty((batch_size, pomo_size), dtype=torch.long, device=device)
    pomo_slots = torch.arange(pomo_size, device=device)

    for batch_idx in range(batch_size):
        legal_nodes = torch.nonzero(legal_mask[batch_idx], as_tuple=False).flatten()
        if legal_nodes.numel() == 0:
            raise RuntimeError("Initial CTSPd priority mask has no legal node.")
        selected[batch_idx] = legal_nodes[pomo_slots % legal_nodes.numel()]

    return selected


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = int(node_index_to_pick.size(0))
    pomo_size = int(node_index_to_pick.size(1))
    embedding_dim = int(encoded_nodes.size(2))

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


def _apply_sampling_controls(probs, model_params, training):
    if training:
        return probs

    adjusted = probs
    temperature = float(model_params.get('sampling_temperature', 1.0))
    if temperature <= 0:
        raise ValueError("sampling_temperature must be positive.")
    if abs(temperature - 1.0) > 1e-9:
        adjusted = adjusted.clamp_min(1e-12).pow(1.0 / temperature)

    top_k = int(model_params.get('sampling_top_k', 0) or 0)
    if 0 < top_k < int(adjusted.size(-1)):
        values, indices = adjusted.topk(top_k, dim=-1)
        filtered = torch.zeros_like(adjusted)
        filtered.scatter_(dim=-1, index=indices, src=values)
        adjusted = filtered

    denom = adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return adjusted / denom


def _inverse_softplus(value):
    clamped_value = max(float(value), 1e-6)
    value_tensor = torch.tensor(clamped_value, dtype=torch.float32)
    return torch.log(torch.expm1(value_tensor))


########################################
# ENCODER
########################################

class CTSPd_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        num_groups = self.model_params.get('num_groups', 5)  # 从 env_params 传入，5是安全设置值
        self.use_group_embedding = self.model_params.get('use_group_embedding', True)
        self.use_group_fusion_gate = self.model_params.get('use_group_fusion_gate', True)

        self.coord_embedding = nn.Linear(2, embedding_dim)
        self.group_embedding = nn.Embedding(num_groups + 1, embedding_dim)
        
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fusion_gate = nn.Linear(embedding_dim * 2, embedding_dim)

        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def set_training_epoch(self, epoch, total_epochs):
        for layer in self.layers:
            if not isinstance(layer, EncoderLayer):
                raise TypeError("CTSPd_Encoder.layers must contain EncoderLayer instances.")
            layer.set_training_epoch(epoch, total_epochs)

    def forward(self, data):
        # data: (batch, problem, 3) = [x, y, priority]
        
        group_ids = data[:, :, 2].long()  # (batch, problem)
        
        coords = data[:, :, :2]
        coord_feat = self.coord_embedding(coords)

        if self.use_group_embedding:
            group_feat = self.group_embedding(group_ids)
            combined = torch.cat([coord_feat, group_feat], dim=-1)
            fused = self.fusion(combined)

            if self.use_group_fusion_gate:
                gate = torch.sigmoid(self.fusion_gate(combined))
                embedded_input = coord_feat + gate * (fused - coord_feat)
            else:
                embedded_input = fused
        else:
            embedded_input = coord_feat
        
        out = embedded_input
        for layer in self.layers:
            if not isinstance(layer, EncoderLayer):
                raise TypeError("CTSPd_Encoder.layers must contain EncoderLayer instances.")
            out = layer(out, group_ids=group_ids)
        
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        num_groups = int(self.model_params.get('num_groups', 5))
        self.cluster_bias_mode = self.model_params.get('cluster_bias_mode', 'scheduled').lower()
        self.same_group_bias_init = float(self.model_params.get('same_group_bias_init', 0.1))
        self.same_group_bias_final = float(self.model_params.get('same_group_bias_final', 1.25))
        self.same_group_bias_max = float(self.model_params.get('same_group_bias_max', 2.0))
        self.learnable_bias_start_epoch = max(
            1,
            int(self.model_params.get('learnable_bias_start_epoch', 1)),
        )
        self.learnable_bias_warmup_epochs = max(
            1,
            int(self.model_params.get('learnable_bias_warmup_epochs', 1)),
        )
        self.learnable_bias_scale_max = float(self.model_params.get('learnable_bias_scale_max', 1.0))
        self.same_group_bias_warmup_epochs = max(
            1,
            int(self.model_params.get('same_group_bias_warmup_epochs', 30)),
        )
        self.priority_distance_bias = float(self.model_params.get('priority_distance_bias', 0.15))
        self.priority_distance_tau = max(1e-6, float(self.model_params.get('priority_distance_tau', 1.0)))
        self.exclude_self_group_bias = self.model_params.get('exclude_self_group_bias', True)
        self.relation_bias_mode = self.model_params.get('relation_bias_mode', 'none').lower()
        self.relation_bias_max_distance = max(0, int(self.model_params.get('relation_bias_max_distance', num_groups - 1)))
        self.relation_bias_init = float(self.model_params.get('relation_bias_init', 0.2))
        self.relation_bias_tau = max(1e-6, float(self.model_params.get('relation_bias_tau', 1.0)))

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

        self.same_group_bias_runtime: torch.Tensor
        self.same_group_bias_param: Optional[nn.Parameter]
        self.register_buffer(
            'same_group_bias_runtime',
            torch.tensor(self.same_group_bias_init, dtype=torch.float32),
        )
        if self.cluster_bias_mode == 'learnable':
            self.same_group_bias_param = nn.Parameter(_inverse_softplus(self.same_group_bias_init))
        elif self.cluster_bias_mode in ('signed_learnable', 'signed_residual'):
            self.same_group_bias_param = nn.Parameter(torch.tensor(self.same_group_bias_init, dtype=torch.float32))
        elif self.cluster_bias_mode in ('scheduled_residual', 'scheduled_plus_learnable'):
            self.same_group_bias_param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.same_group_bias_param = None
        self.register_buffer(
            'learnable_bias_scale_runtime',
            torch.tensor(self.learnable_bias_scale_max, dtype=torch.float32),
            persistent=False,
        )
        self.relation_attention_bias: Optional[nn.Parameter]
        if self.relation_bias_mode in ('learnable', 'headwise', 'relative', 'relative_learnable'):
            distances = torch.arange(self.relation_bias_max_distance + 1, dtype=torch.float32)
            init = self.relation_bias_init * torch.exp(-distances / self.relation_bias_tau)
            self.relation_attention_bias = nn.Parameter(init[None, :].repeat(head_num, 1))
        else:
            self.relation_attention_bias = None

    def set_training_epoch(self, epoch, total_epochs):
        learnable_progress = float(epoch - self.learnable_bias_start_epoch + 1) / float(self.learnable_bias_warmup_epochs)
        learnable_progress = min(1.0, max(0.0, learnable_progress))
        self.learnable_bias_scale_runtime.fill_(learnable_progress * self.learnable_bias_scale_max)

        if self.cluster_bias_mode in ('scheduled', 'scheduled_residual', 'scheduled_plus_learnable'):
            progress = min(1.0, max(0.0, float(epoch - 1) / float(self.same_group_bias_warmup_epochs)))
            value = self.same_group_bias_init + progress * (self.same_group_bias_final - self.same_group_bias_init)
            self.same_group_bias_runtime.fill_(value)

    def forward(self, input1, group_ids=None):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        attention_bias = self._make_group_attention_bias(group_ids, input1.device)

        # 传入multi_head_attention
        out_concat = multi_head_attention(q, k, v, group_bias=attention_bias)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)

    def _current_same_group_bias(self) -> Optional[torch.Tensor]:
        if self.cluster_bias_mode in ('none', 'off', 'disabled'):
            return None
        if self.cluster_bias_mode == 'learnable':
            same_group_bias_param = self.same_group_bias_param
            if same_group_bias_param is None:
                raise RuntimeError("cluster_bias_mode='learnable' requires same_group_bias_param.")
            bias = F.softplus(same_group_bias_param)
        elif self.cluster_bias_mode in ('signed_learnable', 'signed_residual'):
            same_group_bias_param = self.same_group_bias_param
            if same_group_bias_param is None:
                raise RuntimeError("cluster_bias_mode='signed_learnable' requires same_group_bias_param.")
            scale = self.learnable_bias_scale_runtime.to(device=same_group_bias_param.device)
            bias = same_group_bias_param * scale
        elif self.cluster_bias_mode in ('scheduled_residual', 'scheduled_plus_learnable'):
            same_group_bias_param = self.same_group_bias_param
            if same_group_bias_param is None:
                raise RuntimeError("cluster_bias_mode='scheduled_residual' requires same_group_bias_param.")
            bias = self.same_group_bias_runtime + same_group_bias_param
        elif self.cluster_bias_mode == 'fixed':
            bias = torch.tensor(self.same_group_bias_final, device=self.same_group_bias_runtime.device)
        else:
            bias = self.same_group_bias_runtime
        if self.cluster_bias_mode in ('signed_learnable', 'signed_residual'):
            return torch.clamp(bias, min=-self.same_group_bias_max, max=self.same_group_bias_max)
        return torch.clamp(bias, min=0.0, max=self.same_group_bias_max)

    def _make_group_attention_bias(self, group_ids, device):
        if group_ids is None:
            return None

        same_group_bias = self._current_same_group_bias()
        use_same_group = False
        if same_group_bias is not None:
            use_same_group = abs(float(same_group_bias.detach().cpu())) > 0
        use_distance_bias = self.priority_distance_bias > 0
        use_relation_bias = self.relation_attention_bias is not None
        if not use_same_group and not use_distance_bias and not use_relation_bias:
            return None

        g_i = group_ids.unsqueeze(dim=1).unsqueeze(dim=3)
        g_j = group_ids.unsqueeze(dim=1).unsqueeze(dim=2)
        problem_size = int(group_ids.size(1))

        if self.exclude_self_group_bias:
            eye = torch.eye(problem_size, dtype=torch.bool, device=device)[None, None, :, :]
        else:
            eye = None

        attention_bias = torch.zeros(
            group_ids.size(0),
            1,
            problem_size,
            problem_size,
            device=device,
            dtype=torch.float32,
        )

        if use_same_group:
            if same_group_bias is None:
                raise RuntimeError("same_group_bias is unexpectedly unset.")
            same_group_mask = (g_i == g_j)
            if eye is not None:
                same_group_mask = same_group_mask & (~eye)
            attention_bias = attention_bias + same_group_mask.float() * same_group_bias.to(device)

        if use_distance_bias:
            group_distance = (g_i - g_j).abs().float()
            distance_mask = torch.exp(-group_distance / self.priority_distance_tau)
            if eye is not None:
                distance_mask = distance_mask.masked_fill(eye, 0.0)
            attention_bias = attention_bias + distance_mask * self.priority_distance_bias

        if use_relation_bias:
            relation_attention_bias = self.relation_attention_bias
            if relation_attention_bias is None:
                raise RuntimeError("relation_attention_bias is unexpectedly unset.")
            relation_attention_bias = relation_attention_bias * self.learnable_bias_scale_runtime.to(
                device=relation_attention_bias.device
            )
            relation_bias = self._make_relation_attention_bias(
                group_ids,
                relation_attention_bias,
                problem_size,
                device,
            )
            if eye is not None:
                relation_bias = relation_bias.masked_fill(eye, 0.0)
            attention_bias = attention_bias + relation_bias

        return attention_bias

    def _make_relation_attention_bias(self, group_ids, relation_attention_bias, problem_size, device):
        group_delta = (group_ids[:, :, None] - group_ids[:, None, :]).abs()
        group_delta = group_delta.clamp(max=self.relation_bias_max_distance).long()
        flat_delta = group_delta.reshape(-1).to(device=relation_attention_bias.device)

        head_num = int(relation_attention_bias.size(0))
        relation_bias = relation_attention_bias.index_select(dim=1, index=flat_delta)
        relation_bias = relation_bias.reshape(head_num, group_ids.size(0), problem_size, problem_size)
        return relation_bias.permute(1, 0, 2, 3).to(device)


########################################
# DECODER
########################################

class CTSPd_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        num_groups = int(self.model_params.get('num_groups', 5))
        self.use_decoder_priority_bias = bool(self.model_params.get('use_decoder_priority_bias', False))
        self.decoder_priority_bias_mode = self.model_params.get('decoder_priority_bias_mode', 'learnable').lower()
        self.decoder_priority_bias_max_delta = max(
            0,
            int(self.model_params.get('decoder_priority_bias_max_delta', num_groups - 1)),
        )
        self.decoder_priority_bias_init = float(self.model_params.get('decoder_priority_bias_init', 0.2))
        self.decoder_priority_bias_tau = max(
            1e-6,
            float(self.model_params.get('decoder_priority_bias_tau', 1.0)),
        )

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k: Optional[torch.Tensor] = None  # saved key, for multi-head attention
        self.v: Optional[torch.Tensor] = None  # saved value, for multi-head_attention
        self.single_head_key: Optional[torch.Tensor] = None  # saved, for single-head attention
        self.q_first: Optional[torch.Tensor] = None  # saved q1, for multi-head attention
        self.decoder_priority_bias_table: Optional[nn.Parameter]
        if self.use_decoder_priority_bias and self.decoder_priority_bias_mode == 'learnable':
            deltas = torch.arange(self.decoder_priority_bias_max_delta + 1, dtype=torch.float32)
            init = self.decoder_priority_bias_init * torch.exp(-deltas / self.decoder_priority_bias_tau)
            self.decoder_priority_bias_table = nn.Parameter(init)
        else:
            self.decoder_priority_bias_table = None

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(dim0=1, dim1=2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask, group_ids=None, current_min_priority=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']
        q_first = self.q_first
        k = self.k
        v = self.v
        single_head_key = self.single_head_key
        if q_first is None or k is None or v is None or single_head_key is None:
            raise RuntimeError("Decoder.set_kv and set_q1 must be called before forward.")

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = q_first + q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, k, v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        priority_bias = self._make_decoder_priority_bias(group_ids, current_min_priority, ninf_mask)
        if priority_bias is not None:
            score_clipped = score_clipped + priority_bias

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs

    def _make_decoder_priority_bias(self, group_ids, current_min_priority, ninf_mask):
        if not self.use_decoder_priority_bias:
            return None
        if group_ids is None or current_min_priority is None:
            return None

        priority_delta = group_ids[:, None, :].to(current_min_priority.device) - current_min_priority[:, :, None]
        priority_delta = priority_delta.clamp(min=0, max=self.decoder_priority_bias_max_delta).long()

        if self.decoder_priority_bias_mode == 'learnable':
            bias_table = self.decoder_priority_bias_table
            if bias_table is None:
                raise RuntimeError("decoder_priority_bias_table is unexpectedly unset.")
            flat_delta = priority_delta.reshape(-1).to(device=bias_table.device)
            priority_bias = bias_table.index_select(dim=0, index=flat_delta).reshape(priority_delta.shape)
            priority_bias = priority_bias.to(device=ninf_mask.device)
        elif self.decoder_priority_bias_mode == 'fixed':
            priority_bias = self.decoder_priority_bias_init * torch.exp(
                -priority_delta.float() / self.decoder_priority_bias_tau
            )
        else:
            raise ValueError("Unsupported decoder_priority_bias_mode: {}".format(self.decoder_priority_bias_mode))

        legal_mask = torch.isfinite(ninf_mask)
        return priority_bias.masked_fill(~legal_mask, 0.0)


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = int(qkv.size(0))
    n = int(qkv.size(1))

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(dim0=1, dim1=2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None, group_bias=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = int(q.size(0))
    head_num = int(q.size(1))
    n = int(q.size(2))
    key_dim = int(q.size(3))
    input_s = int(k.size(2))

    score = torch.matmul(q, k.transpose(dim0=2, dim1=3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float, device=score.device))
    # 新增：应用聚类感知偏置（在softmax之前）
    if group_bias is not None:
        # group_bias: (batch, 1, problem, problem) 或兼容形状
        # 广播到所有头：(batch, head_num, n, problem)
        score_scaled = score_scaled + group_bias
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(dim0=1, dim1=2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(dim0=1, dim1=2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(dim0=1, dim1=2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
