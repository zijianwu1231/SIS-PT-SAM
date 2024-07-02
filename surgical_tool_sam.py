import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mobile_sam.build_sam import sam_model_registry

class SurgicalToolSAM(nn.Module):
    """Adapt from MobileSAM
    Args:
        sam_model: a vision transformer model, see base_vit.py
    Examples:
        For inference, freeze_image_encoder, freeze_prompt_encoder, and freeze_mask_decoder all should be True.
    """

    def __init__(
        self, 
        ckpt, 
        freeze_image_encoder=True, 
        freeze_prompt_encoder=True, 
        freeze_mask_decoder=True,
    ):
        super(SurgicalToolSAM, self).__init__()
        sam_model = sam_model_registry["vit_t"](checkpoint=ckpt) # model - MobileSAM, slightly change its mask decoder

        self.freeze_image_encoder = freeze_image_encoder
        self.freeze_prompt_encoder = freeze_prompt_encoder

        if freeze_image_encoder: # freeze the image encoder
            print('freeze the image encoder!')
            for param in sam_model.image_encoder.parameters():
                param.requires_grad = False
        
        if freeze_prompt_encoder: # freeze the prompt encoder
            print('freeze the prompt encoder!')
            for param in sam_model.prompt_encoder.parameters():
                param.requires_grad = False

        if freeze_mask_decoder: # freeze the mask decoder
            print('freeze the mask decoder!')
            for param in sam_model.mask_decoder.parameters():
                param.requires_grad = False
                            
        self.sam = sam_model
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder
   
    def forward(self, image, point_prompt, status="training"):
        if self.freeze_image_encoder:
            with torch.no_grad():
                image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        else:
            image_embedding = self.image_encoder(image)
        if self.freeze_prompt_encoder:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=point_prompt,
                    boxes=None,
                    masks=None,
                )
        else:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=point_prompt,
                boxes=None,
                masks=None,
            )
        low_res_logits, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        ) # (B, 1, 256, 256)

        if status == "training":
            return low_res_logits
        elif status == "inference":
            low_res_pred = torch.sigmoid(low_res_logits)
            low_res_pred = F.interpolate(
                low_res_pred,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
            )
            low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()  # (256, 256)
            surgicaltoolsam_seg = (low_res_pred > 0.5).astype(np.uint8)
            return surgicaltoolsam_seg
    