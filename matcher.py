import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    def __init__(self, cost_mask: float = 1, cost_dice: float = 1):
        super().__init__()
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_mask != 0 or cost_dice != 0, "all costs can't be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, num_queries = outputs["pred_masks"].shape[:2]
        num_targets = targets[0]["masks"].shape[0]  # Using the first element to get num_targets

        indices = []

        for b in range(bs):
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            tgt_mask = targets[b]["masks"]  # [num_targets, H_tgt, W_tgt]
                
            # Downsample gt masks to match pred mask size
            tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest").squeeze(1)

            # Flatten spatial dimension
            out_mask = out_mask.flatten(1)  # [num_queries, H*W]
            tgt_mask = tgt_mask.flatten(1)  # [num_targets, H*W]
            # Compute the focal loss between masks (dummy example)
            cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)

            # Compute the dice loss between masks
            cost_dice = batch_sigmoid_focal_loss(out_mask, tgt_mask)

            # Final cost matrix
            C = self.cost_mask * cost_mask + self.cost_dice * cost_dice
            C = C.cpu()
            C[torch.isinf(C)] = 1e3
            if torch.isnan(C).any() or torch.isinf(C).any():
                raise ValueError("Cost matrix contains NaN or Inf values")
            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def batch_sigmoid_focal_loss(inputs, targets,alpha: float = 0.25, gamma: float = 2):
    hw = inputs.shape[1]
    if inputs.dtype != targets.dtype:
        targets = targets.to(inputs.dtype)
    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets))
        
    return loss / hw

def batch_dice_loss(inputs, targets):
    smooth = 1e-5
    
    # 디버깅 출력을 추가하여 텐서의 형태와 값을 확인합니다.
    if inputs.dtype != targets.dtype:
        targets = targets.to(inputs.dtype)
        
    intersection = torch.matmul(inputs, targets.T)
    dice_score = (2. * intersection + smooth) / (inputs.sum(-1, keepdim=True) + targets.sum(-1, keepdim=True).T + smooth)

    # Debugging output for dice loss shape

    # Ensure the output is [num_queries, num_targets]
    return 1 - dice_score
