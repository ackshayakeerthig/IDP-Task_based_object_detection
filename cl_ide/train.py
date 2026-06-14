"""
Training Pipeline: Task-Aware YOLO with Knowledge Distillation & QAT

Features:
- Knowledge distillation from CLIP teacher
- Quantization-Aware Training (QAT) for INT8 precision
- Kaggle TPU + GPU support
- Checkpoint management
- Tensorboard logging
- Sparsity regularization for FPGA efficiency
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Custom modules
from task_aware_yolo import TaskAwareYOLOWithHooks
from clip_teacher import CLIPTeacherModel, DistillationLossV2
from coco_dataloader import create_coco_dataloader, FUNCTIONAL_TASKS


class TrainingConfig:
    """Training hyperparameters."""
    
    def __init__(self):
        self.epochs = 100
        self.warmup_epochs = 5
        self.qat_epochs = 10  # Last 10 epochs with QAT
        self.batch_size = 32
        self.img_size = 640
        
        # Learning rate
        self.initial_lr = 1e-3
        self.min_lr = 1e-5
        
        # Loss weights
        self.yolo_weight = 1.0
        self.mse_weight = 0.3
        self.cosine_weight = 0.2
        self.kl_weight = 0.1
        self.sparsity_weight = 0.01  # Encourage sparse gating
        
        # Optimizer
        self.optimizer = "adamw"  # adamw or sgd
        self.weight_decay = 0.0005
        self.momentum = 0.937
        
        # Mixed precision training
        self.use_amp = True
        
        # Checkpointing
        self.save_freq = 5  # Save every 5 epochs
        self.checkpoint_dir = Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Logging
        self.log_dir = Path("./logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_freq = 100  # Log every N batches


class TaskAwareYOLOTrainer:
    """Main training orchestrator."""
    
    def __init__(
        self,
        model: nn.Module,
        teacher_model: CLIPTeacherModel,
        train_loader,
        val_loader,
        config: TrainingConfig,
        device: str = "cuda",
        resume_from: Optional[str] = None
    ):
        self.model = model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Loss functions
        self.distill_loss = DistillationLossV2(
            yolo_weight=config.yolo_weight,
            mse_weight=config.mse_weight,
            cosine_weight=config.cosine_weight,
            kl_weight=config.kl_weight
        ).to(device)
        
        # Learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        self.use_amp = config.use_amp
        
        # Tensorboard
        self.writer = SummaryWriter(str(config.log_dir))
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_lr': []
        }
        
        # Current epoch
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Resume if checkpoint provided
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config.optimizer == "adamw":
            optimizer = optim.AdamW(
                params,
                lr=self.config.initial_lr,
                weight_decay=self.config.weight_decay
            )
        else:  # sgd
            optimizer = optim.SGD(
                params,
                lr=self.config.initial_lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        
        return optimizer
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Setup learning rate scheduler with warmup."""
        
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        total_steps = len(self.train_loader) * (self.config.epochs - self.config.warmup_epochs)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.min_lr
        )
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        yolo_loss_sum = 0.0
        distill_loss_sum = 0.0
        sparsity_loss_sum = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            task_tokens = batch['task_tokens'].to(self.device)
            task_names = batch['task_names']
            labels = [lbl.to(self.device) for lbl in batch['labels']]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Student forward
                    student_output = self.model(
                        images=images,
                        task_tokens=task_tokens,
                        return_heatmap=True
                    )
                    
                    # YOLO loss (from detections)
                    # Note: Simplified - in practice, extract YOLO loss properly
                    yolo_loss = torch.tensor(0.5, device=self.device)  # Placeholder
                    
                    # Teacher heatmap
                    with torch.no_grad():
                        teacher_heatmap = self.teacher_model.generate_semantic_heatmap(
                            images,
                            task_names,
                            grid_size=(14, 14)
                        )
                    
                    # Distillation loss
                    distill_losses = self.distill_loss(
                        yolo_loss=yolo_loss,
                        student_heatmap=student_output['heatmap'],
                        teacher_heatmap=teacher_heatmap,
                        student_features=student_output['gated_features']
                    )
                    
                    # Sparsity regularization
                    sparsity_loss = self._compute_sparsity_loss(student_output)
                    
                    # Total loss
                    loss = (
                        distill_losses['total_loss'] +
                        self.config.sparsity_weight * sparsity_loss
                    )
            else:
                # Standard forward
                student_output = self.model(
                    images=images,
                    task_tokens=task_tokens,
                    return_heatmap=True
                )
                
                yolo_loss = torch.tensor(0.5, device=self.device)
                
                with torch.no_grad():
                    teacher_heatmap = self.teacher_model.generate_semantic_heatmap(
                        images,
                        task_names,
                        grid_size=(14, 14)
                    )
                
                distill_losses = self.distill_loss(
                    yolo_loss=yolo_loss,
                    student_heatmap=student_output['heatmap'],
                    teacher_heatmap=teacher_heatmap,
                    student_features=student_output['gated_features']
                )
                
                sparsity_loss = self._compute_sparsity_loss(student_output)
                loss = (
                    distill_losses['total_loss'] +
                    self.config.sparsity_weight * sparsity_loss
                )
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Metrics
            total_loss += loss.item()
            yolo_loss_sum += distill_losses['yolo_loss'].item()
            distill_loss_sum += distill_losses['distill_loss'].item()
            sparsity_loss_sum += sparsity_loss.item()
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % self.config.log_freq == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
                
                self.writer.add_scalar(
                    'train/batch_loss',
                    loss.item(),
                    epoch * len(self.train_loader) + batch_idx
                )
        
        # Epoch metrics
        avg_loss = total_loss / num_batches
        self.metrics['train_loss'].append(avg_loss)
        self.metrics['train_lr'].append(self.optimizer.param_groups[0]['lr'])
        
        self.writer.add_scalar('train/loss', avg_loss, epoch)
        self.writer.add_scalar('train/yolo_loss', yolo_loss_sum / num_batches, epoch)
        self.writer.add_scalar('train/distill_loss', distill_loss_sum / num_batches, epoch)
        self.writer.add_scalar('train/sparsity_loss', sparsity_loss_sum / num_batches, epoch)
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
        
        return {
            'loss': avg_loss,
            'yolo_loss': yolo_loss_sum / num_batches,
            'distill_loss': distill_loss_sum / num_batches,
            'sparsity_loss': sparsity_loss_sum / num_batches
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validating")
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            task_tokens = batch['task_tokens'].to(self.device)
            task_names = batch['task_names']
            
            student_output = self.model(
                images=images,
                task_tokens=task_tokens,
                return_heatmap=True
            )
            
            teacher_heatmap = self.teacher_model.generate_semantic_heatmap(
                images,
                task_names,
                grid_size=(14, 14)
            )
            
            yolo_loss = torch.tensor(0.5, device=self.device)
            
            distill_losses = self.distill_loss(
                yolo_loss=yolo_loss,
                student_heatmap=student_output['heatmap'],
                teacher_heatmap=teacher_heatmap,
                student_features=student_output['gated_features']
            )
            
            sparsity_loss = self._compute_sparsity_loss(student_output)
            loss = (
                distill_losses['total_loss'] +
                self.config.sparsity_weight * sparsity_loss
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.metrics['val_loss'].append(avg_loss)
        
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        # Save if best
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(epoch, is_best=True)
        
        return {'loss': avg_loss}
    
    def enable_qat(self):
        """Enable Quantization-Aware Training."""
        try:
            from pytorch_quantization import nn as quant_nn
            from pytorch_quantization.nn.modules import _utils as quant_utils
            
            print("Enabling QAT...")
            quant_nn.TensorQuantizer.use_fake_quant = True
            quant_nn.TensorQuantizer.use_static_arange = False
            
            # Prepare model for QAT
            self.model.apply(torch.quantization.convert)
            
        except ImportError:
            print("pytorch_quantization not installed. Skipping QAT.")
    
    def disable_qat(self):
        """Disable QAT and finalize quantization."""
        try:
            from pytorch_quantization import nn as quant_nn
            quant_nn.TensorQuantizer.use_fake_quant = False
        except:
            pass
    
    def _compute_sparsity_loss(self, output: Dict) -> torch.Tensor:
        """
        Compute sparsity regularization loss.
        Encourages gating module to produce sparse gates (many zeros).
        """
        # This is a simplified version - in practice, hook into gating module
        return torch.tensor(0.0, device=self.device)
    
    def train(self):
        """Full training loop."""
        
        for epoch in range(self.start_epoch, self.config.epochs):
            # Enable QAT in last epochs
            if epoch >= (self.config.epochs - self.config.qat_epochs):
                self.enable_qat()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch)
        
        self.writer.close()
        print("\nTraining complete!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        ckpt_path = (
            self.config.checkpoint_dir / f"model_epoch_{epoch+1}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config.__dict__
        }, ckpt_path)
        
        if is_best:
            best_path = self.config.checkpoint_dir / "model_best.pt"
            torch.save(self.model.state_dict(), best_path)
    
    def load_checkpoint(self, ckpt_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.metrics = checkpoint['metrics']
        self.start_epoch = checkpoint['epoch'] + 1
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def setup_kaggle_tpu():
    """Setup Kaggle TPU if available."""
    try:
        import kaggle_datasets
        print("Kaggle TPU environment detected!")
        
        # Use XLA devices
        os.environ['TPU_NAME'] = '/job:localhost/replica:0/task:0/device:TPU:0'
        device = 'tpu:0'
        print(f"Using device: {device}")
        
        return device
    except:
        # Fall back to GPU/CPU
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("Using CPU")
        
        return device


# ==================== Main ====================

def main():
    """Main training script."""
    
    # Setup device
    device = setup_kaggle_tpu()
    
    # Config
    config = TrainingConfig()
    
    print("\nTask-Aware YOLO Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Initial LR: {config.initial_lr}")
    print("=" * 60 + "\n")
    
    # Create dummy model
    # In practice, load YOLOv8-Small from ultralytics
    try:
        from ultralytics import YOLO
        yolo_base = YOLO('yolov8s.pt')
    except:
        print("Warning: ultralytics YOLO not available. Using dummy model.")
        yolo_base = None
    
    # Initialize models
    model = TaskAwareYOLOWithHooks(
        yolo_model=yolo_base,
        semantic_dim=512,
        vocab_size=10000
    )
    
    # CLIP teacher
    teacher = CLIPTeacherModel(model_name="ViT-B/32", device=device)
    
    # Create dummy dataloaders for demo
    # In practice, use actual COCO dataset
    print("Creating dummy dataloaders for demo...")
    from coco_dataloader import COCODatasetWithTasks, TaskTokenizer
    from torch.utils.data import DataLoader
    
    # Minimal dummy dataset
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'images': torch.randn(1, 3, 640, 640).squeeze(0),
                'task_tokens': torch.tensor(idx % 14),
                'task_names': list(FUNCTIONAL_TASKS.values())[idx % 14],
                'labels': torch.zeros((0, 5))
            }
    
    def dummy_collate(batch):
        return {
            'images': torch.stack([item['images'] for item in batch]),
            'task_tokens': torch.stack([item['task_tokens'] for item in batch]),
            'task_names': [item['task_names'] for item in batch],
            'labels': [item['labels'] for item in batch]
        }
    
    train_dataset = DummyDataset(size=50)
    val_dataset = DummyDataset(size=10)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=dummy_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=dummy_collate
    )
    
    # Trainer
    trainer = TaskAwareYOLOTrainer(
        model=model,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
