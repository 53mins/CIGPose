import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS
from typing import Optional, Sequence, Tuple, Union

@MODELS.register_module()
class CounterfactualConsistencyLoss(nn.Module):
    """Counterfactual Consistency Loss.

    This loss enforces consistency between the observational and counterfactual
    predictions for stable keypoints that have not been intervened upon.
    This encourages the model to learn meaningful causal interventions.

    Args:
        use_kl (bool): Whether to use KL divergence to measure the distance
            between two probability distributions. If False, L2 loss is used.
            Defaults to True.
        loss_weight (float): The weight of the loss. Defaults to 1.0.
    """

    def __init__(self, use_kl: bool = True, loss_weight: float = 1.0):
        super().__init__()
        self.use_kl = use_kl
        self.loss_weight = loss_weight

    def forward(self,
                h_obs: Tuple[torch.Tensor, torch.Tensor],
                h_cf: Tuple[torch.Tensor, torch.Tensor],
                intervened_mask: torch.Tensor,
                keypoint_weights: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h_obs (Tuple[Tensor, Tensor]): The observational predictions (x, y),
                with each tensor having a shape of (B, K, W) or (B, K, H).
            h_cf (Tuple[Tensor, Tensor]): The counterfactual predictions (x, y).
            intervened_mask (Tensor): A boolean mask of shape (B, K), where
                `True` indicates that the keypoint has been intervened upon.
            keypoint_weights (Tensor): The weights for each keypoint, with a
                shape of (B, K).

        Returns:
            Tensor: The calculated loss value.
        """
        h_obs_x, h_obs_y = h_obs
        h_cf_x, h_cf_y = h_cf
        
        # We only compute the loss on stable keypoints that are visible and
        # have not been intervened upon
        stable_mask = (~intervened_mask) & (keypoint_weights > 0)
        
        # If there are no stable keypoints in the batch, return a loss of 0
        if stable_mask.sum() == 0:
            return h_obs_x.new_tensor(0.)

        if self.use_kl:
            # KL divergence D_kl(P || Q) requires inputs of log P and Q.
            # We treat the observational prediction as the target distribution Q
            # and the counterfactual prediction as P.
            log_p_cf_x = F.log_softmax(h_cf_x, dim=-1)
            # Treat the observational prediction as a fixed target, so we
            # don't compute its gradient
            p_obs_x = F.softmax(h_obs_x.detach(), dim=-1) 
            loss_x = F.kl_div(log_p_cf_x, p_obs_x, reduction='none').sum(-1)

            log_p_cf_y = F.log_softmax(h_cf_y, dim=-1)
            p_obs_y = F.softmax(h_obs_y.detach(), dim=-1)
            loss_y = F.kl_div(log_p_cf_y, p_obs_y, reduction='none').sum(-1)
        else:
            # Calculate the L2 distance on the probability distributions
            p_cf_x = F.softmax(h_cf_x, dim=-1)
            p_obs_x = F.softmax(h_obs_x.detach(), dim=-1)
            loss_x = torch.sum((p_cf_x - p_obs_x)**2, dim=-1)

            p_cf_y = F.softmax(h_cf_y, dim=-1)
            p_obs_y = F.softmax(h_obs_y.detach(), dim=-1)
            loss_y = torch.sum((p_cf_y - p_obs_y)**2, dim=-1)

        loss = (loss_x + loss_y) * stable_mask
        
        # Average the loss over the number of stable keypoints
        return self.loss_weight * (loss.sum() / stable_mask.sum())