# coding: utf-8
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in (None, ''):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from losses.flow_matching import flow_matching_mse_loss


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = int(time_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

    def forward(self, t):
        # t: [B, 1]
        half_dim = max(self.time_dim // 2, 1)
        freq_ids = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        scale = -math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(freq_ids * scale).view(1, -1)
        angles = t * freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.size(-1) < self.time_dim:
            emb = F.pad(emb, (0, self.time_dim - emb.size(-1)))
        elif emb.size(-1) > self.time_dim:
            emb = emb[:, :self.time_dim]
        return self.proj(emb)  # [B, T]


class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, flow_input):
        # flow_input: [B, F]
        return self.net(flow_input)  # [B, D]


class MoERouter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, dropout=0.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_experts),
        )

    def forward(self, flow_input):
        # flow_input: [B, F]
        logits = self.router(flow_input)  # [B, E]
        gate = torch.softmax(logits, dim=-1)
        return logits, gate


class MLPVelocityNet(nn.Module):
    def __init__(self, embed_dim, condition_dim, hidden_dim, time_dim, expert_dropout=0.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.time_embedding = TimeEmbedding(self.time_dim)
        self.input_dim = self.embed_dim + self.time_dim + self.condition_dim
        self.expert = MLPExpert(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.embed_dim,
            dropout=expert_dropout,
        )

    def forward(self, z_t, t, cond):
        # z_t: [B, D]
        # t: [B, 1]
        # cond: [B, C]
        time_emb = self.time_embedding(t)  # [B, T]
        flow_input = torch.cat([z_t, time_emb, cond], dim=-1)  # [B, D + T + C]
        velocity = self.expert(flow_input)  # [B, D]
        return velocity, None


class MoEVelocityNet(nn.Module):
    def __init__(
        self,
        embed_dim,
        condition_dim,
        hidden_dim,
        time_dim,
        num_experts=4,
        router_hidden_dim=128,
        router_dropout=0.0,
        expert_dropout=0.0,
        dense_moe=True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.num_experts = int(num_experts)
        self.dense_moe = bool(dense_moe)
        self.time_embedding = TimeEmbedding(self.time_dim)
        self.input_dim = self.embed_dim + self.time_dim + self.condition_dim
        self.experts = nn.ModuleList([
            MLPExpert(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.embed_dim,
                dropout=expert_dropout,
            )
            for _ in range(self.num_experts)
        ])
        self.router = MoERouter(
            input_dim=self.input_dim,
            hidden_dim=router_hidden_dim,
            num_experts=self.num_experts,
            dropout=router_dropout,
        )

    def forward(self, z_t, t, cond):
        # z_t: [B, D]
        # t: [B, 1]
        # cond: [B, C]
        time_emb = self.time_embedding(t)  # [B, T]
        flow_input = torch.cat([z_t, time_emb, cond], dim=-1)  # [B, D + T + C]
        expert_outputs = torch.stack([expert(flow_input) for expert in self.experts], dim=1)  # [B, E, D]
        _, gate = self.router(flow_input)  # [B, E]
        if not self.dense_moe:
            top_idx = gate.argmax(dim=-1, keepdim=True)
            hard_gate = torch.zeros_like(gate).scatter_(1, top_idx, 1.0)
            gate = gate + (hard_gate - gate).detach()
        velocity = torch.sum(gate.unsqueeze(-1) * expert_outputs, dim=1)  # [B, D]
        return velocity, gate


class PrototypeFlow(nn.Module):
    def __init__(
        self,
        embed_dim,
        condition_dim,
        hidden_dim=128,
        time_dim=32,
        velocity_type='moe',
        num_experts=4,
        moe_router_hidden_dim=128,
        moe_router_dropout=0.0,
        expert_dropout=0.0,
        dense_moe=True,
        normalize_output=True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.velocity_type = str(velocity_type).lower()
        self.num_experts = int(num_experts)
        self.normalize_output = bool(normalize_output)
        self.eps = 1e-12

        if self.velocity_type == 'mlp':
            self.velocity_net = MLPVelocityNet(
                embed_dim=self.embed_dim,
                condition_dim=self.condition_dim,
                hidden_dim=self.hidden_dim,
                time_dim=self.time_dim,
                expert_dropout=expert_dropout,
            )
        elif self.velocity_type == 'moe':
            self.velocity_net = MoEVelocityNet(
                embed_dim=self.embed_dim,
                condition_dim=self.condition_dim,
                hidden_dim=self.hidden_dim,
                time_dim=self.time_dim,
                num_experts=self.num_experts,
                router_hidden_dim=moe_router_hidden_dim,
                router_dropout=moe_router_dropout,
                expert_dropout=expert_dropout,
                dense_moe=dense_moe,
            )
        else:
            raise ValueError(f'Unsupported velocity_type: {self.velocity_type}')

    @staticmethod
    def build_condition(
        stream_repr,
        user_out,
        user_rep,
        user_activity,
        router_weights=None,
        detach_condition=False,
    ):
        # stream_repr: [B, D]
        # user_out: [B, D]
        # user_rep: [B, D]
        # user_activity: [B, 1]
        pieces = [stream_repr, user_out, user_rep]
        if user_activity.dim() == 1:
            user_activity = user_activity.unsqueeze(-1)
        pieces.append(user_activity)
        if router_weights is not None:
            pieces.append(router_weights)
        condition = torch.cat(pieces, dim=-1)
        if detach_condition:
            condition = condition.detach()
        return condition  # [B, C]

    @staticmethod
    def flow_matching_loss(v_pred, v_target):
        return flow_matching_mse_loss(v_pred, v_target)

    def _compute_moe_regularization(self, gate, ref_tensor):
        zero = ref_tensor.new_zeros(())
        if gate is None:
            return zero, zero
        entropy = -(gate * torch.log(gate + self.eps)).sum(dim=-1).mean()
        entropy_loss = -entropy
        expected = gate.new_full((gate.size(-1),), 1.0 / gate.size(-1))
        mean_gate = gate.mean(dim=0)
        balance_loss = torch.mean((mean_gate - expected) ** 2)
        return entropy_loss, balance_loss

    def generate_one_step(self, x0, condition):
        # x0: [B, D]
        # condition: [B, C]
        t0 = x0.new_zeros(x0.size(0), 1)
        velocity, moe_gate = self.velocity_net(x0, t0, condition)
        z_pos = x0 + velocity  # [B, D]
        if self.normalize_output:
            z_pos = F.normalize(z_pos, dim=-1)
        return {
            'z_pos': z_pos,
            'moe_gate': moe_gate,
        }

    def forward(self, x0, x1, condition):
        # x0: [B, D]
        # x1: [B, D]
        # condition: [B, C]
        batch_size = x0.size(0)
        t = torch.rand(batch_size, 1, device=x0.device, dtype=x0.dtype)  # [B, 1]
        z_t = (1.0 - t) * x0 + t * x1  # [B, D]
        v_target = x1 - x0  # [B, D]
        v_pred, moe_gate = self.velocity_net(z_t, t, condition)  # [B, D], [B, E] or None
        flow_loss = self.flow_matching_loss(v_pred, v_target)
        z_gen = self.generate_one_step(x0, condition)
        moe_entropy_loss, moe_balance_loss = self._compute_moe_regularization(moe_gate, x0)
        return {
            'z_pos': z_gen['z_pos'],
            'flow_loss': flow_loss,
            'v_pred': v_pred,
            'v_target': v_target,
            'moe_gate': z_gen['moe_gate'],
            'moe_entropy_loss': moe_entropy_loss,
            'moe_balance_loss': moe_balance_loss,
        }


PositivePrototypeFlow = PrototypeFlow


if __name__ == '__main__':
    torch.manual_seed(2026)

    batch_size = 5
    embed_dim = 12
    route_dim = 3
    condition_dim = embed_dim * 3 + 1 + route_dim

    x0 = torch.randn(batch_size, embed_dim)
    x1 = torch.randn(batch_size, embed_dim)
    stream_repr = torch.randn(batch_size, embed_dim)
    user_out = torch.randn(batch_size, embed_dim)
    user_rep = torch.randn(batch_size, embed_dim)
    user_activity = torch.rand(batch_size, 1)
    router_weights = torch.softmax(torch.randn(batch_size, route_dim), dim=-1)

    condition = PrototypeFlow.build_condition(
        stream_repr=stream_repr,
        user_out=user_out,
        user_rep=user_rep,
        user_activity=user_activity,
        router_weights=router_weights,
        detach_condition=False,
    )

    for velocity_type in ('mlp', 'moe'):
        flow = PrototypeFlow(
            embed_dim=embed_dim,
            condition_dim=condition_dim,
            hidden_dim=16,
            time_dim=8,
            velocity_type=velocity_type,
            num_experts=4,
            moe_router_hidden_dim=12,
            moe_router_dropout=0.0,
            expert_dropout=0.0,
            dense_moe=True,
            normalize_output=True,
        )
        outputs = flow(x0, x1, condition)
        generated = flow.generate_one_step(x0, condition)
        assert outputs['z_pos'].shape == (batch_size, embed_dim)
        assert outputs['v_pred'].shape == (batch_size, embed_dim)
        assert outputs['v_target'].shape == (batch_size, embed_dim)
        assert outputs['flow_loss'].dim() == 0
        assert outputs['moe_entropy_loss'].dim() == 0
        assert outputs['moe_balance_loss'].dim() == 0
        assert generated['z_pos'].shape == (batch_size, embed_dim)
        if velocity_type == 'moe':
            assert outputs['moe_gate'].shape == (batch_size, 4)
            assert generated['moe_gate'].shape == (batch_size, 4)
        else:
            assert outputs['moe_gate'] is None
            assert generated['moe_gate'] is None

    print('PrototypeFlow smoke test passed for mlp and moe velocity fields.')
