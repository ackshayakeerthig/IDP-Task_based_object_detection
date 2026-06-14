"""
Knowledge Distillation Framework: CLIP Teacher Model
Generates semantic heatmaps for student model alignment

Teacher (Expert): CLIP (ViT-B/32) - Pre-trained on 400M image-text pairs
Student (Learner): TaskAwareYOLO with gating
Goal: Student learns to generate task-aware feature maps aligned with CLIP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np


class CLIPTeacherModel(nn.Module):
    """
    CLIP-based Teacher Model for Semantic Affordance Heatmaps.
    
    Process:
    1. Image -> CLIP Vision Encoder -> [B, 768] image embedding
    2. Task text -> CLIP Text Encoder -> [B, 768] text embedding
    3. Compute pixel-wise similarity -> [B, 7, 7] or [B, 14, 14] heatmap
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, etc.)
            device: torch device
        """
        super().__init__()
        
        try:
            import clip
        except ImportError:
            raise ImportError("Install clip: pip install openai-clip")
        
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()  # Teacher is frozen
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get embedding dimensions
        self.embed_dim = self.model.text_projection.shape[1]  # Usually 512
        self.vision_embed_dim = self.model.visual.output_dim  # Usually 768
        
    @torch.no_grad()
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features using CLIP vision encoder.
        
        Args:
            images: [B, 3, H, W] normalized images
        
        Returns:
            features: [B, 768] image embeddings
        """
        return self.model.encode_image(images)
    
    @torch.no_grad()
    def get_text_features(self, task_texts: List[str]) -> torch.Tensor:
        """
        Extract task text features using CLIP text encoder.
        
        Args:
            task_texts: List of task descriptions (e.g., ["pouring", "cutting"])
        
        Returns:
            features: [B, 512] text embeddings (after projection)
        """
        import clip
        
        # Tokenize text
        tokens = clip.tokenize(task_texts, context_length=77).to(self.device)
        
        # Encode text
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        
        return text_features
    
    @torch.no_grad()
    def generate_semantic_heatmap(
        self,
        images: torch.Tensor,
        task_texts: List[str],
        grid_size: Tuple[int, int] = (14, 14)
    ) -> torch.Tensor:
        """
        Generate semantic affordance heatmaps: visual-textual similarity.
        
        Process:
        1. Extract vision features from CLIP ViT patches
        2. Extract text features from task description
        3. Compute cosine similarity -> heatmap
        4. Normalize and reshape to grid
        
        Args:
            images: [B, 3, 224, 224] or [B, 3, 336, 336] (CLIP input)
            task_texts: List of task strings (length B)
            grid_size: Output heatmap grid (14, 14) or (7, 7)
        
        Returns:
            heatmap: [B, 1, grid_h, grid_w] semantic affordance scores
        """
        batch_size = images.shape[0]
        
        # Get image features from ViT patches
        image_features = self.get_image_features(images)  # [B, 768]
        
        # Get task text features
        text_features = self.get_text_features(task_texts)  # [B, 512]
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity: [B, 512] x [B, 512]^T -> [B, B]
        # For each image, compute similarity with corresponding task
        similarity = torch.diagonal(
            torch.matmul(image_features, text_features.t())
        )  # [B]
        
        # Reshape to heatmap: replicate similarity across spatial grid
        h, w = grid_size
        heatmap = similarity.view(batch_size, 1, 1, 1)
        heatmap = heatmap.expand(batch_size, 1, h, w)
        
        return heatmap
    
    @torch.no_grad()
    def generate_spatial_heatmap(
        self,
        images: torch.Tensor,
        task_texts: List[str],
        grid_size: Tuple[int, int] = (14, 14)
    ) -> torch.Tensor:
        """
        Advanced: Generate spatial heatmap using ViT patch embeddings.
        
        Extracts intermediate patch embeddings from ViT and computes
        patch-wise similarity with task embedding.
        
        Args:
            images: [B, 3, 224, 224]
            task_texts: List of tasks
            grid_size: Target output size
        
        Returns:
            heatmap: [B, 1, grid_h, grid_w] spatial similarity map
        """
        batch_size = images.shape[0]
        
        # Get text features
        text_features = self.get_text_features(task_texts)  # [B, 512]
        text_features = F.normalize(text_features, dim=-1)  # [B, 512]
        
        # Extract ViT patch embeddings
        # For ViT-B/32: 7x7 = 49 patches + 1 class token = 50 tokens
        # Intermediate features before projection
        with torch.no_grad():
            # Process through vision encoder up to final layer
            x = self.model.visual.conv1(images)  # [B, 768, 7, 7]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 768, 49]
            x = x.permute(0, 2, 1)  # [B, 49, 768]
            
            # Add class token
            cls_tokens = self.model.visual.class_embedding.unsqueeze(0).expand(
                x.shape[0], -1, -1
            )  # [B, 1, 768]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, 50, 768]
            
            # Add positional embedding
            x = x + self.model.visual.positional_embedding
            x = self.model.visual.ln_pre(x)
            
            # Transformer blocks
            x = self.model.visual.transformer(x)
            x = self.model.visual.ln_post(x)
        
        # Remove class token
        patch_features = x[:, 1:, :]  # [B, 49, 768]
        
        # Project text features to vision space
        text_features_expanded = F.linear(
            text_features,
            self.model.visual.proj.weight if hasattr(self.model.visual, 'proj') else torch.eye(512)
        )  # [B, 768]
        
        # Compute patch-wise cosine similarity
        text_norm = F.normalize(text_features_expanded, dim=-1)  # [B, 768]
        patch_norm = F.normalize(patch_features, dim=-1)  # [B, 49, 768]
        
        similarity = torch.matmul(patch_norm, text_norm.unsqueeze(-1))  # [B, 49, 1]
        similarity = similarity.squeeze(-1)  # [B, 49]
        
        # Reshape to 7x7 grid
        heatmap = similarity.view(batch_size, 1, 7, 7)
        
        # Interpolate to target grid size
        heatmap = F.interpolate(
            heatmap,
            size=grid_size,
            mode='bilinear',
            align_corners=False
        )
        
        return heatmap


class DistillationLoss(nn.Module):
    """
    Composite loss for knowledge distillation.
    
    L_total = L_YOLO + λ * L_Distill
    
    Where L_Distill = MSE(student_heatmap, teacher_heatmap)
    """
    
    def __init__(
        self,
        yolo_weight: float = 1.0,
        distill_weight: float = 0.5,
        mse_reduction: str = 'mean'
    ):
        super().__init__()
        self.yolo_weight = yolo_weight
        self.distill_weight = distill_weight
        self.mse_loss = nn.MSELoss(reduction=mse_reduction)
    
    def forward(
        self,
        yolo_loss: torch.Tensor,
        student_heatmap: torch.Tensor,
        teacher_heatmap: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            yolo_loss: Standard YOLO loss (box + conf + cls)
            student_heatmap: [B, 1, H, W] from TaskAwareYOLO
            teacher_heatmap: [B, 1, H, W] from CLIP
        
        Returns:
            dict with:
                - 'total_loss': Weighted sum
                - 'yolo_loss': Component
                - 'distill_loss': Component
        """
        
        # MSE distillation loss
        distill_loss = self.mse_loss(student_heatmap, teacher_heatmap)
        
        # Weighted total
        total_loss = (
            self.yolo_weight * yolo_loss +
            self.distill_weight * distill_loss
        )
        
        return {
            'total_loss': total_loss,
            'yolo_loss': yolo_loss,
            'distill_loss': distill_loss
        }


class DistillationLossV2(nn.Module):
    """
    Enhanced distillation with multiple objectives.
    
    L_total = L_YOLO + λ_mse * L_MSE + λ_cos * L_Cosine + λ_kl * L_KL
    """
    
    def __init__(
        self,
        yolo_weight: float = 1.0,
        mse_weight: float = 0.3,
        cosine_weight: float = 0.2,
        kl_weight: float = 0.1
    ):
        super().__init__()
        self.yolo_weight = yolo_weight
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.kl_weight = kl_weight
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        yolo_loss: torch.Tensor,
        student_heatmap: torch.Tensor,
        teacher_heatmap: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-objective distillation loss.
        """
        
        # 1. MSE on heatmaps
        mse = self.mse_loss(student_heatmap, teacher_heatmap)
        
        # 2. Cosine similarity on features (if provided)
        cosine = torch.tensor(0.0, device=yolo_loss.device)
        if student_features is not None and teacher_features is not None:
            # Flatten features
            s_feat = student_features.view(student_features.shape[0], -1)
            t_feat = teacher_features.view(teacher_features.shape[0], -1)
            
            # Normalize
            s_feat = F.normalize(s_feat, dim=-1)
            t_feat = F.normalize(t_feat, dim=-1)
            
            # Cosine loss (maximize similarity)
            target = torch.ones(s_feat.shape[0], device=s_feat.device)
            cosine = 1.0 - torch.diagonal(torch.matmul(s_feat, t_feat.t())).mean()
        
        # 3. KL divergence on normalized heatmaps
        student_prob = F.softmax(student_heatmap.flatten(1), dim=-1)
        teacher_prob = F.softmax(teacher_heatmap.flatten(1), dim=-1)
        kl = self.kl_loss(
            torch.log(student_prob + 1e-8),
            teacher_prob
        )
        
        # Total loss
        total_loss = (
            self.yolo_weight * yolo_loss +
            self.mse_weight * mse +
            self.cosine_weight * cosine +
            self.kl_weight * kl
        )
        
        return {
            'total_loss': total_loss,
            'yolo_loss': yolo_loss,
            'mse_loss': mse,
            'cosine_loss': cosine,
            'kl_loss': kl
        }


# ==================== Testing ====================

if __name__ == "__main__":
    print("Knowledge Distillation Framework Test")
    print("=" * 60)
    
    # Test DistillationLoss
    print("\n1. Testing DistillationLoss...")
    loss_fn = DistillationLoss(yolo_weight=1.0, distill_weight=0.5)
    
    yolo_loss = torch.tensor(2.5, requires_grad=True)
    student_hm = torch.randn(4, 1, 14, 14)
    teacher_hm = torch.randn(4, 1, 14, 14)
    
    losses = loss_fn(yolo_loss, student_hm, teacher_hm)
    
    print(f"   YOLO Loss: {losses['yolo_loss'].item():.4f}")
    print(f"   Distill Loss: {losses['distill_loss'].item():.4f}")
    print(f"   Total Loss: {losses['total_loss'].item():.4f}")
    print("   ✓ DistillationLoss OK")
    
    # Test DistillationLossV2
    print("\n2. Testing DistillationLossV2...")
    loss_fn_v2 = DistillationLossV2()
    
    student_feat = torch.randn(4, 512, 20, 20)
    teacher_feat = torch.randn(4, 512, 20, 20)
    
    losses_v2 = loss_fn_v2(
        yolo_loss,
        student_hm,
        teacher_hm,
        student_features=student_feat,
        teacher_features=teacher_feat
    )
    
    print(f"   YOLO Loss: {losses_v2['yolo_loss'].item():.4f}")
    print(f"   MSE Loss: {losses_v2['mse_loss'].item():.4f}")
    print(f"   Cosine Loss: {losses_v2['cosine_loss'].item():.4f}")
    print(f"   KL Loss: {losses_v2['kl_loss'].item():.4f}")
    print(f"   Total Loss: {losses_v2['total_loss'].item():.4f}")
    print("   ✓ DistillationLossV2 OK")
    
    print("\n" + "=" * 60)
    print("Knowledge Distillation tests passed! ✓")
