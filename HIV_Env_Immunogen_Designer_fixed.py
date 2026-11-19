"""
HIV Env Immunogen Designer — Structural GNO Phase Field
Based on: structural_gno_fold_v4.py (Real Fold ONE surrogate)
Purpose: Lock Env Trimer in Pre-fusion state to expose bNAb epitopes
"""
import torch
import torch.nn as nn


class HIV_Env_Immunogen_Designer(nn.Module):
    def __init__(self, gno_fold_model):
        super().__init__()
        self.gno_fold = gno_fold_model  # StructuralGNOFold จาก AGI ONE Ecosystem

        # Learned stabilizing mutations (Amino Acid substitutions)
        # 20 Amino Acids + 1 No-mutation
        self.mutation_net = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 21)  # Predicts AA substitution probability
        )

    def forward(self, env_trimer_graph, target_epitope_mask):
        """
        env_trimer_graph: Graph representation of HIV Env gp160
            Must have .node_feats (B,N,C) and .sigma (B,N) attributes,
            OR be a raw tensor (B,N,C).
        target_epitope_mask: Mask ของจุดที่เราต้องการเปิด (เช่น CD4 binding site)
            Shape: (B,N) or (N,)
        """
        # [B3c FIX] Handle both graph objects and raw tensors
        if hasattr(env_trimer_graph, 'node_feats'):
            node_feats = env_trimer_graph.node_feats
            sigma = getattr(env_trimer_graph, 'sigma', None)
        else:
            node_feats = env_trimer_graph
            sigma = None

        # 1. จำลอง Phase Field ของโปรตีนด้วย GNO Fold
        if sigma is not None:
            u_pred = self.gno_fold.forward_phase_field(node_feats, sigma)
        else:
            u_pred = self.gno_fold(node_feats)

        # Ensure u_pred is (B, N, C)
        if u_pred.dim() == 2:
            u_pred = u_pred.unsqueeze(0)

        # 2. คำนวณ Free Energy ของการเปลี่ยนรูป (Conformational Energy)
        if target_epitope_mask.dim() == 1:
            target_epitope_mask = target_epitope_mask.unsqueeze(0)
        # Expand mask for broadcasting with u_pred (B, N, C)
        mask_expanded = target_epitope_mask.unsqueeze(-1)  # (B, N, 1)
        pre_fusion_energy = (u_pred * mask_expanded).sum()

        # 3. เสนอการกลายพันธุ์ (SOSIP mutations) เพื่อล็อกโครงสร้าง
        # [B3b FIX] per-position mutation prediction
        # u_pred: (B, N, C) → mutation_net applied on last dim → (B, N, 21)
        # This gives per-residue mutation probabilities
        if u_pred.shape[-1] != 256:
            # Project to expected dimension if needed
            proj = nn.Linear(u_pred.shape[-1], 256).to(u_pred.device)
            u_for_mutation = proj(u_pred)
        else:
            u_for_mutation = u_pred

        mutation_logits = self.mutation_net(u_for_mutation)  # (B, N, 21)
        mutation_probs = torch.softmax(mutation_logits, dim=-1)

        return {
            "stabilized_structure": u_pred,
            "energy_barrier": pre_fusion_energy,
            "proposed_mutations": mutation_probs,  # (B, N, 21)
        }


# --- smoke test ---
if __name__ == "__main__":
    torch.manual_seed(0)

    # Mock GNO fold model
    class MockGNOFold(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 256)

        def forward_phase_field(self, node_feats, sigma):
            return self.proj(node_feats)

        def forward(self, x):
            return self.proj(x)

    gno = MockGNOFold()
    designer = HIV_Env_Immunogen_Designer(gno)

    # Test with raw tensor input
    B, N, C = 2, 50, 64
    node_feats = torch.randn(B, N, C)
    epitope_mask = torch.zeros(B, N)
    epitope_mask[:, 10:20] = 1.0  # CD4 binding site region

    out = designer(node_feats, epitope_mask)
    loss = out["energy_barrier"] + out["proposed_mutations"].sum()
    loss.backward()

    print(f"Env Designer OK:")
    print(f"  stabilized_structure: {tuple(out['stabilized_structure'].shape)}")
    print(f"  energy_barrier: {out['energy_barrier'].item():.4f}")
    print(f"  proposed_mutations: {tuple(out['proposed_mutations'].shape)}")
    print(f"  mutation probs sum to 1: {out['proposed_mutations'].sum(-1).mean().item():.4f}")
