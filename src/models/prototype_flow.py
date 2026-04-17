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
        return self.proj(emb)


class ConditionalVelocityNet(nn.Module):
    def __init__(self, embed_dim, condition_dim, hidden_dim, time_dim):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)

        self.time_embedding = TimeEmbedding(self.time_dim)
        input_dim = self.embed_dim + self.condition_dim + self.time_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

    def forward(self, z_t, t, condition):
        # z_t: [B, D]
        # t: [B, 1]
        # condition: [B, C]
        time_emb = self.time_embedding(t)  # [B, T]
        velocity_input = torch.cat([z_t, time_emb, condition], dim=-1)  # [B, D + T + C]
        return self.mlp(velocity_input)  # [B, D]


class PositivePrototypeFlow(nn.Module):
    def __init__(self, embed_dim, condition_dim, hidden_dim=128, time_dim=32, normalize_output=True):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.normalize_output = bool(normalize_output)
        self.velocity_net = ConditionalVelocityNet(
            embed_dim=self.embed_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.hidden_dim,
            time_dim=self.time_dim,
        )

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
        pieces.append(user_activity)  # [B, 1]

        if router_weights is not None:
            pieces.append(router_weights)  # [B, M]

        condition = torch.cat(pieces, dim=-1)
        if detach_condition:
            condition = condition.detach()
        return condition

    @staticmethod
    def flow_matching_loss(v_pred, v_target):
        return flow_matching_mse_loss(v_pred, v_target)

    def generate(self, x0, condition):
        # x0: [B, D]
        # condition: [B, C]
        t0 = x0.new_zeros(x0.size(0), 1)
        velocity = self.velocity_net(x0, t0, condition)  # [B, D]
        z_pos = x0 + velocity  # [B, D]
        if self.normalize_output:
            z_pos = F.normalize(z_pos, dim=-1)
        return z_pos

    def forward(self, x0, x1, condition):
        # x0: [B, D]
        # x1: [B, D]
        # condition: [B, C]
        batch_size = x0.size(0)
        t = torch.rand(batch_size, 1, device=x0.device, dtype=x0.dtype)  # [B, 1]
        z_t = (1.0 - t) * x0 + t * x1  # [B, D]
        v_target = x1 - x0  # [B, D]
        v_pred = self.velocity_net(z_t, t, condition)  # [B, D]
        flow_loss = self.flow_matching_loss(v_pred, v_target)
        z_pos = self.generate(x0, condition)  # [B, D]
        return z_pos, flow_loss


if __name__ == '__main__':
    torch.manual_seed(2026)

    batch_size = 5
    embed_dim = 12
    route_dim = 3
    condition_dim = embed_dim * 3 + 1 + route_dim

    flow = PositivePrototypeFlow(
        embed_dim=embed_dim,
        condition_dim=condition_dim,
        hidden_dim=16,
        time_dim=8,
        normalize_output=True,
    )

    stream_repr = torch.randn(batch_size, embed_dim)
    user_out = torch.randn(batch_size, embed_dim)
    user_rep = torch.randn(batch_size, embed_dim)
    user_activity = torch.rand(batch_size, 1)
    router_weights = torch.softmax(torch.randn(batch_size, route_dim), dim=-1)

    condition = flow.build_condition(
        stream_repr=stream_repr,
        user_out=user_out,
        user_rep=user_rep,
        user_activity=user_activity,
        router_weights=router_weights,
        detach_condition=False,
    )
    x0 = torch.randn(batch_size, embed_dim)
    x1 = torch.randn(batch_size, embed_dim)

    z_pos, flow_loss = flow(x0, x1, condition)
    z_gen = flow.generate(x0, condition)

    assert condition.shape == (batch_size, condition_dim)
    assert z_pos.shape == (batch_size, embed_dim)
    assert z_gen.shape == (batch_size, embed_dim)
    assert flow_loss.dim() == 0
    assert torch.isfinite(flow_loss)
    print('PositivePrototypeFlow smoke test passed.')
