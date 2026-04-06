import torch
import torch.nn as nn
from utils import img2mse


class Criterion(nn.Module):
    def __init__(self):
    # def __init__(self, lambda_o=0.01):

    #     """
    #     Loss criterion for GNT with transient handling.
        
    #     Args:
    #         lambda_o: Weight for visibility mask regularization loss. 
    #                  Default 0.01 encourages model to predict binary masks.
    #     """
        super().__init__()
        # self.lambda_o = lambda_o

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion with visibility mask support
        """
        pred_rgb = outputs["rgb"]  # [N_rays, 3]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]  # [N_rays, 3]

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)
        # # Compute RGB loss using standard MSE
        # rgb_loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        # # Check if visibility mask is provided for transient handling
        # visibility_mask = ray_batch.get("visibility_mask")  # [N_rays, 1]
        
        # if visibility_mask is not None:
        #     # Transform visibility mask from [0, 1] to soft weights
        #     # M_t close to 1 -> high weight for rgb loss (static pixels)
        #     # M_t close to 0 -> low weight for rgb loss (transient pixels)
            
        #     # Compute per-pixel RGB loss (without mask weighting)
        #     per_pixel_loss = (pred_rgb - gt_rgb) ** 2  # [N_rays, 3]
        #     per_pixel_loss = torch.mean(per_pixel_loss, dim=-1, keepdim=True)  # [N_rays, 1]
            
        #     # Weighted RGB loss: weight by visibility mask
        #     weighted_rgb_loss = torch.mean(visibility_mask * per_pixel_loss)
            
        #     # Transient regularization: encourage mask to be binary (0 or 1)
        #     # (1 - M_t)^2 is small when M_t is 0 or 1
        #     transient_reg_loss = torch.mean((1.0 - visibility_mask) ** 2)
            
        #     # Total loss
        #     loss = weighted_rgb_loss + self.lambda_o * transient_reg_loss
            
        #     # Log individual loss components
        #     scalars_to_log["loss/rgb"] = weighted_rgb_loss.item()
        #     scalars_to_log["loss/transient_reg"] = transient_reg_loss.item()
        # else:
        #     # Fallback to standard loss if visibility mask not available
        #     loss = rgb_loss

        return loss, scalars_to_log
