import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from typing import Optional, Sequence, Tuple, Union
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
import warnings

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.hierarchical_gnn import HGNNModule
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import MODELS
from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from .rtmcc_head import RTMCCHead

@MODELS.register_module()
class CausalInterventionModule(nn.Module):
    """
    Implements the Causal Intervention Module (CIM).
    This module identifies confounded keypoint features and replaces them
    with learned, normalized "ideal" features.
    """
    def __init__(self,
                 num_keypoints: int,
                 feature_dim: int,
                 intervention_strategy: str = 'topk',
                 intervention_k: int = 10,
                 intervention_k_val: int = 1,
                 intervention_threshold: float = 0.5):
        super().__init__()
        if intervention_strategy not in ['topk', 'threshold']:
            raise ValueError("intervention_strategy must be 'topk' or 'threshold'")
            
        self.num_keypoints = num_keypoints
        self.feature_dim = feature_dim
        self.intervention_strategy = intervention_strategy
        
        # Treat intervention_k as the value for training
        self.intervention_k = intervention_k
        self.intervention_k_val = intervention_k_val
        
        self.intervention_threshold = intervention_threshold
        self.canonical_features = nn.Embedding(num_keypoints, feature_dim)

    def forward(self, f_kpts: Tensor, h_initial: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            f_kpts (Tensor): Keypoint features with shape [B, K, C].
            h_initial (Tuple[Tensor, Tensor]): Initial SimCC predictions (x, y),
                                                 with shapes [B, K, W] and [B, K, H].
        Returns:
            Tuple[Tensor, Tensor]:
            - f_prime_kpts (Tensor): Intervened keypoint features [B, K, C].
            - intervened_mask (Tensor): Boolean mask for intervened keypoints [B, K].
        """
        h_initial_x, h_initial_y = h_initial
        device = f_kpts.device

        # Identify confounded nodes using prediction confidence
        with torch.no_grad():
            conf_x = h_initial_x.softmax(dim=-1).max(dim=-1)[0]
            conf_y = h_initial_y.softmax(dim=-1).max(dim=-1)[0]
            confidence = (conf_x + conf_y) / 2.0  # Shape: [B, K]
            
            confound_score = 1.0 - confidence
            intervened_mask = torch.zeros_like(confidence, dtype=torch.bool, device=device)

            if self.intervention_strategy == 'topk':
                # Select k value based on the self.training state
                k = self.intervention_k if self.training else self.intervention_k_val
                
                # Only perform top-k operation if k > 0
                if k > 0:
                    # Ensure k does not exceed the total number of keypoints
                    k = min(k, self.num_keypoints)
                    _, topk_indices = torch.topk(confound_score, k, dim=1)
                    intervened_mask.scatter_(1, topk_indices, True)
            else: # 'threshold'
                intervened_mask = confound_score > self.intervention_threshold

        # Generate and replace with counterfactual (canonical) features
        intervened_indices = torch.where(intervened_mask)
        # If no nodes are intervened, return the original features directly
        if intervened_indices[0].numel() == 0:
            return f_kpts.clone(), intervened_mask

        kpt_indices_to_lookup = intervened_indices[1]
        canonical_feats = self.canonical_features(kpt_indices_to_lookup)

        f_prime_kpts = f_kpts.clone()
        f_prime_kpts[intervened_indices] = canonical_feats

        return f_prime_kpts, intervened_mask

@MODELS.register_module()
class CIGHead(RTMCCHead):
    def __init__(
        self,
        in_channels,
        out_channels,
        input_size,
        in_featuremap_size,
        in_featuremap_num=4,
        simcc_split_ratio=2.0,
        final_layer_kernel_size=1,
        gau_cfg=dict(),
        loss=dict(),
        loss_cf=dict(),
        decoder=None,
        gcn_cfg=dict(),
        cim_cfg=dict(),
        keypoint_connections=None,
        keypoint_groups=None,
        init_cfg=None,
        **kwargs
    ):
        if init_cfg is None:
            init_cfg = self.default_init_cfg
        super().__init__(
            in_channels,
            out_channels,
            input_size,
            in_featuremap_size,
            simcc_split_ratio,
            final_layer_kernel_size,
            gau_cfg,
            loss,
            decoder,
            init_cfg,
            **kwargs,
        )
        self.loss_cf_module = MODELS.build(loss_cf)

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        hidden_dims = gau_cfg['hidden_dims']
        in_mlp = 0
        for i in range(in_featuremap_num):
            in_mlp += 4 ** i
        flatten_dims_GCN = in_mlp * (self.in_featuremap_size[0] * self.in_featuremap_size[1])
        self.mlp_pre_gau = nn.Sequential(
            ScaleNorm(flatten_dims_GCN),
            nn.Linear(flatten_dims_GCN, gau_cfg['hidden_dims'], bias=False))

        self.initial_pred_x = nn.Linear(hidden_dims, W)
        self.initial_pred_y = nn.Linear(hidden_dims, H)

        cim_cfg.update(num_keypoints=out_channels, feature_dim=hidden_dims)
        self.cim = CausalInterventionModule(**cim_cfg)

        self.gcn = HGNNModule(gcn_cfg, keypoint_connections, keypoint_groups)

        self.cls_gx = nn.Linear(gcn_cfg['hidden_dim'], W)
        self.cls_gy = nn.Linear(gcn_cfg['hidden_dim'], H)
        
    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        """Forward pass (for predict). This method executes the complete CI-GNN inference pipeline."""
        # Initial feature processing
        x = self.final_layer(feats[-1]) # B, K, H', W'
        x = torch.flatten(x, 2) # B, K, H'*W'
        
        f_kpts_initial = self.mlp_pre_gau(x) # B, K, hidden_dims
        
        f_kpts = self.gau(f_kpts_initial) # [B, K, C_hidden]

        # 2. Get initial predictions for intervention
        h_initial_x = self.initial_pred_x(f_kpts)
        h_initial_y = self.initial_pred_y(f_kpts)
        
        # 3. Apply causal intervention
        f_prime_kpts, _ = self.cim(f_kpts, (h_initial_x, h_initial_y))
        
        # 4. GCN refinement on intervened features
        gcn_out = self.gcn(f_prime_kpts) # [B, K, C_gcn_hidden]
        
        # 5. Final prediction
        pred_x = self.cls_gx(gcn_out)
        pred_y = self.cls_gy(gcn_out)

        return pred_x, pred_y

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: OptConfigType = {}) -> dict:
        """Calculate losses. This method implements the complete training logic,
        including the counterfactual consistency loss."""
        gt_x = torch.cat([d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples], dim=0)
        gt_y = torch.cat([d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples], dim=0)
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples], dim=0)
        gt_simcc = (gt_x, gt_y)
        
        # --- Common feature extraction ---
        x = self.final_layer(feats[-1])
        x = torch.flatten(x, 2)
        f_kpts_initial = self.mlp_pre_gau(x)
        f_kpts = self.gau(f_kpts_initial)

        # --- Counterfactual Path (for the main loss L_kpt) ---
        h_initial_x = self.initial_pred_x(f_kpts.detach()) # Detach gradients from this branch, used only for CIM decision-making
        h_initial_y = self.initial_pred_y(f_kpts.detach())
        f_prime_kpts, intervened_mask = self.cim(f_kpts, (h_initial_x, h_initial_y))
        gcn_out_cf = self.gcn(f_prime_kpts)
        pred_x_cf = self.cls_gx(gcn_out_cf)
        pred_y_cf = self.cls_gy(gcn_out_cf)
        pred_simcc_cf = (pred_x_cf, pred_y_cf)

        # --- Observational Path (for the consistency loss L_cf) ---
        with torch.no_grad():
            gcn_out_obs = self.gcn(f_kpts)
            pred_x_obs = self.cls_gx(gcn_out_obs)
            pred_y_obs = self.cls_gy(gcn_out_obs)
            pred_simcc_obs = (pred_x_obs, pred_y_obs)

        # --- Loss Calculation ---
        losses = dict()
        loss_kpt = self.loss_module(pred_simcc_cf, gt_simcc, keypoint_weights)
        losses['loss_kpt'] = loss_kpt

        loss_cf = self.loss_cf_module(pred_simcc_obs, pred_simcc_cf, intervened_mask, keypoint_weights)
        losses['loss_cf'] = loss_cf
        
        # --- Accuracy Calculation ---
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc_cf),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0)
        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses['acc_pose'] = acc_pose

        return losses
    
    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        
        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y = self.forward(_feats)
            _batch_pred_x_flip, _batch_pred_y_flip = self.forward(_feats_flip)
            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip, _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)

        preds = self.decode((batch_pred_x, batch_pred_y))
        return preds

    @property
    def default_init_cfg(self):
        return [
            dict(type='Normal', layer=['Conv2d', 'Linear'], std=0.01, bias=0),
            dict(type='Constant', layer=['BatchNorm2d', 'BatchNorm1d'], val=1),
            dict(type='Normal', layer=['Embedding'], std=0.01)]