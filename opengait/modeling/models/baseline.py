import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from einops import rearrange

from .TCL_utils.cluster import PhaseClusterAdapter

class Baseline(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval


class Baseline_TCL(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.HPP = SetBlockWrapper(HorizontalPoolingPyramid(bin_num=model_cfg['bin_num']))
        self.gm_adapter = PhaseClusterAdapter(**model_cfg['cluster_cfg'])
        self.pretrain_weights_path = model_cfg['pretrain_weights_path']

    def load_pretrain_weights(self):
        pretrain_dict = torch.load(self.pretrain_weights_path, map_location=torch.device('cpu'))
        # 删除 BNNecks 相关的权重
        pretrain_model = pretrain_dict['model']
        # filtered_model = {k: v for k, v in pretrain_model.items() if not k.startswith('BNNecks.') and not k.startswith('FCs.')}
        msg = self.load_state_dict(pretrain_model, strict=False)
        self.msg_mgr.log_info('Missing keys: {}'.format(msg.missing_keys))
        self.msg_mgr.log_info('Unexpected keys: {}'.format(msg.unexpected_keys))

    def init_parameters(self):
        super().init_parameters()
        if self.pretrain_weights_path is not None:
            self.load_pretrain_weights()
            self.gm_adapter._init_weights()
            for name, parameter in self.named_parameters():
                if 'gm_adapter' in name or 'FCs' in name or 'BNNecks' in name:
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False        

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, s, p]

        outs, cluster_indices = self.gm_adapter(feat)  # [n, c, p], [n, s]

        embed_1 = self.FCs(outs)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        cluster_loss = self.gm_adapter.diversity_loss()
        balance_loss = self.gm_adapter.balance_loss()
        
        # vis
        clustered_frames = {}
        cluster_assignments = cluster_indices  # [n, s]

        for k in range(self.gm_adapter.num_clusters):
            mask = (cluster_assignments == k)  # [n, s] bool
            frames_list = []

            for i in range(sils.size(0)):  
                idx = mask[i].nonzero(as_tuple=True)[0] 
                if idx.numel() > 0:
                    selected_frames = sils[i, :, idx, :, :]  # [c, num_frames, h, w]
                    selected_frames = selected_frames.permute(1, 0, 2, 3).contiguous()  # [num_frames, c, h, w]
                    frames_list.append(selected_frames)

            if frames_list:
                clustered_frames[f'image/cluster_{k}'] = torch.cat(frames_list, dim=0)  # [N_k, c, h, w]
            else:
                clustered_frames[f'image/cluster_{k}'] = torch.zeros(1, sils.size(1), sils.size(3), sils.size(4))
        
        
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
                'cluster_loss': cluster_loss,
                'balance_loss': balance_loss 
            },
            'visual_summary': clustered_frames,
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval