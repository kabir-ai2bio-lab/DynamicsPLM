import torch
from torch import nn
from transformers import EsmConfig, EsmForMaskedLM
from config import config
from huggingface_hub import PyTorchModelHubMixin


class DynamicsPLM(nn.Module, PyTorchModelHubMixin):

    def __init__(self, esm2_select='model_35M'):
        super().__init__()

        self.esm_config = config[esm2_select]

        esm_for_mlm   = EsmForMaskedLM.from_pretrained(self.esm_config['model_id'])
        self.encoder  = esm_for_mlm.esm
        self.mlm_head = esm_for_mlm.lm_head

        hf_cfg = EsmConfig.from_pretrained(self.esm_config['model_id'])

        self.embed_dim  = self.esm_config['embed_dim']
        self.vocab_size = hf_cfg.vocab_size
        self.num_heads  = hf_cfg.num_attention_heads
        self.num_layers = hf_cfg.num_hidden_layers

        self.res_out_dim      = self.esm_config['res_out_dim']
        self.pair_out_dim     = self.esm_config['pair_out_dim']
        self.res_feature_idx  = config['training']['res_feature_idx']
        self.pair_feature_idx = config['training']['pair_feature_idx']
        dropout_rate          = config['training']['dropout']

        self.residue_dynamics_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim, self.res_out_dim),
        )

        self.pairwise_transform = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim, self.embed_dim * 2),
        )

        atten_dim           = self.esm_config['atten_dim']
        total_attention_dim = self.num_heads * self.num_layers
        self.pairwise_middle = nn.Linear(self.embed_dim * 2, atten_dim)
        self.pairwise_output = nn.Linear(atten_dim + total_attention_dim, self.pair_out_dim)

        self.gelu     = nn.GELU()
        self.softplus = nn.Softplus(beta=1.0, threshold=2.0)
        self.sigmoid  = nn.Sigmoid()
        self.softmax  = nn.Softmax(dim=-1)

        self._init_bias_zero()

    def _init_bias_zero(self):
        for module in [self.residue_dynamics_head,
                       self.pairwise_transform,
                       self.pairwise_middle,
                       self.pairwise_output]:
            for m in module.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs, return_mlm=True, return_res_dynamics=True,
                return_pair_dynamics=True, return_embeddings=False,
                return_attentions=False):
        output = {}

        encoder_output = self.encoder(**inputs, output_attentions=True)
        embeddings     = encoder_output['last_hidden_state']

        # (batch, total_heads, seq_len, seq_len) → (batch, seq_len, seq_len, total_heads)
        attention_maps = torch.cat(encoder_output['attentions'], dim=1)
        attention_maps = attention_maps.permute(0, 2, 3, 1)

        if return_embeddings:
            output['embeddings'] = embeddings
        if return_attentions:
            output['attentions'] = attention_maps

        if return_mlm:
            output['mlm_logits'] = self.mlm_head(embeddings)

        if return_res_dynamics:
            res_raw = self.residue_dynamics_head(embeddings)
            for feature_name, feature_indices in self.res_feature_idx.items():
                if feature_name == 'rmsf_nor':
                    output[feature_name] = self.sigmoid(res_raw[:, :, feature_indices])
                elif feature_name in ['ss', 'chi', 'phi', 'psi']:
                    output[feature_name] = self.softmax(res_raw[:, :, feature_indices])
                else:
                    output[feature_name] = self.softplus(res_raw[:, :, feature_indices])

        if return_pair_dynamics:
            s    = self.pairwise_transform(embeddings)
            q, k = s.chunk(2, dim=-1)

            prod              = q[:, None, :, :] * k[:, :, None, :]
            diff              = q[:, None, :, :] - k[:, :, None, :]
            pairwise_features = self.gelu(self.pairwise_middle(torch.cat([prod, diff], dim=-1)))

            pair_raw = self.pairwise_output(torch.cat([pairwise_features, attention_maps], dim=-1))

            for feature_name, feature_indices in self.pair_feature_idx.items():
                if feature_name in ['corr', 'nma_pair1', 'nma_pair2', 'nma_pair3']:
                    output[feature_name] = self.sigmoid(pair_raw[:, :, :, feature_indices]) * 2 - 1.0
                else:
                    output[feature_name] = self.softplus(pair_raw[:, :, :, feature_indices])

        return output

    def get_num_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
