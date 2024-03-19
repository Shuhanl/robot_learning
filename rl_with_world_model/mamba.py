import torch.nn as nn
try:
    from mamba_ssm.models.mixer_seq_simple import create_block
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    print("Failed to import Triton LayerNorm / RMSNorm kernels")


class MambaNet(nn.Module):
    def __init__(
        self,
        model_dim,
        nlayers,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    model_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                )
                for i in range(nlayers)
            ]
        )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            model_dim, eps=norm_epsilon
        )

    def forward(self, hidden_states, lengths=None, inference_params=None):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MambaModel(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim=128,
        nlayers=1,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        fused_add_norm=False,
        residual_in_fp32=False,
    ) -> None:
        super().__init__()
        self.model_type = "mamba"
        self.embed = nn.Linear(input_dim, model_dim)
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.mamba = MambaNet(
            model_dim=model_dim,
            nlayers=nlayers,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )


    def forward(self, x, inference_params=None):
        hidden_states = self.embed(x)
        output = self.mamba(hidden_states)
        return output
