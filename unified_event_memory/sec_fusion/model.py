#!/usr/bin/env python3

import torch
import torch.nn as nn


class SECNumericEncoder(nn.Module):

    def __init__(
        self,
        input_dim=3,
        hidden_dim=64,
        output_dim=200,
        dropout=0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):

        if x.shape[-1] != 3:
            raise ValueError(
                "SEC feature must have exactly "
                "3 dimensions"
            )

        return self.network(x)


class SECGraphFusion(nn.Module):

    def __init__(
        self,
        kgt_dim=200,
        sec_hidden_dim=64,
        dropout=0.1,
    ):
        super().__init__()

        self.kgt_dim = kgt_dim

        self.encoder = SECNumericEncoder(
            input_dim=3,
            hidden_dim=sec_hidden_dim,
            output_dim=kgt_dim,
            dropout=dropout,
        )

        self.gate = nn.Sequential(
            nn.Linear(kgt_dim * 2 + 1, kgt_dim),
            nn.Sigmoid(),
        )

        # Small initial SEC contribution.
        # Still learnable through backpropagation.
        self.fusion_scale = nn.Parameter(
            torch.tensor(-3.0)
        )

    def forward(
        self,
        kgt_emb,
        sec_features,
        sec_mask,
        sec_age_days=None,
        return_gate=False,
    ):
        if kgt_emb.shape[-1] != self.kgt_dim:
            raise ValueError(
                "Unexpected KGT embedding dimension"
            )

        if sec_features.shape != (
            kgt_emb.shape[0], 3
        ):
            raise ValueError(
                "SEC features must be [batch, 3]"
            )

        if sec_mask.shape not in (
            (kgt_emb.shape[0],),
            (kgt_emb.shape[0], 1),
        ):
            raise ValueError(
                "SEC mask shape mismatch"
            )

        sec_mask = sec_mask.reshape(-1, 1).to(
            device=kgt_emb.device,
            dtype=kgt_emb.dtype,
        )

        sec_features = sec_features.to(
            device=kgt_emb.device,
            dtype=kgt_emb.dtype,
        )

        # Ensure masked values never enter encoder.
        safe_sec = sec_features * sec_mask

        sec_emb = self.encoder(safe_sec)

        if sec_age_days is None:
            age = torch.zeros(
                (kgt_emb.shape[0], 1),
                device=kgt_emb.device,
                dtype=kgt_emb.dtype,
            )
        else:
            age = sec_age_days.reshape(-1, 1).to(
                device=kgt_emb.device,
                dtype=kgt_emb.dtype,
            )

            # Age is normalized to [0, 1].
            age = age.clamp(
                min=0.0,
                max=180.0,
            ) / 180.0

        gate_input = torch.cat(
            [
                kgt_emb,
                sec_emb,
                age,
            ],
            dim=-1
        )

        gate = self.gate(gate_input)

        scale = torch.sigmoid(
            self.fusion_scale
        )

        fused = (
            kgt_emb
            +
            sec_mask
            * scale
            * gate
            * sec_emb
        )

        if return_gate:
            return fused, gate * scale * sec_mask

        return fused


def validate_fusion():

    torch.manual_seed(41)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = SECGraphFusion(
        kgt_dim=200,
    ).to(device)

    # Synthetic KGT embeddings are used ONLY
    # to verify shapes, gradients and masking.
    # These are not trained company embeddings.

    kgt = torch.randn(
        5, 200,
        device=device,
        requires_grad=True,
    )

    sec = torch.tensor(
        [
            [0.5, 0.2, 0.9],
            [-0.4, 0.8, 0.7],
            [0.1, 0.3, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    mask = torch.tensor(
        [1, 1, 1, 0, 0],
        device=device,
        dtype=torch.float32,
    )

    age = torch.tensor(
        [2, 8, 30, -1, -1],
        device=device,
        dtype=torch.float32,
    )

    fused, gate = model(
        kgt,
        sec,
        mask,
        age,
        return_gate=True,
    )

    assert fused.shape == (5, 200)
    assert gate.shape == (5, 200)

    # Missing SEC signal must reproduce
    # the original KGT embedding exactly.
    assert torch.equal(
        fused[3:],
        kgt[3:],
    )

    # Check that available SEC signals
    # actually affect the embedding.
    delta = (
        fused[:3] - kgt[:3]
    ).abs().sum().item()

    assert delta > 0

    # Test actual fusion gradients.
    loss = fused.square().mean()
    loss.backward()

    encoder_grad = sum(
        int(
            p.grad is not None
            and torch.count_nonzero(p.grad).item() > 0
        )
        for p in model.encoder.parameters()
    )

    gate_grad = sum(
        int(
            p.grad is not None
            and torch.count_nonzero(p.grad).item() > 0
        )
        for p in model.gate.parameters()
    )

    assert encoder_grad > 0
    assert gate_grad > 0
    assert kgt.grad is not None
    assert torch.isfinite(kgt.grad).all()

    print("=" * 72)
    print("SEC NUMERIC + KGT FUSION SMOKE")
    print("=" * 72)
    print("Device:", device)
    print("KGT input:", tuple(kgt.shape))
    print("SEC input:", tuple(sec.shape))
    print("Fused output:", tuple(fused.shape))
    print("Available SEC rows:", int(mask.sum()))
    print("Missing SEC preserved:", True)
    print("SEC contribution L1:", delta)
    print("Encoder gradient tensors:", encoder_grad)
    print("Gate gradient tensors:", gate_grad)
    print()
    print("SEC FUSION GRADIENT PASS")
    print("NOTE: This uses synthetic KGT inputs.")
    print("NOTE: Fusion weights are not trained yet.")


if __name__ == "__main__":
    validate_fusion()
