"""
Testing & Visualization Utilities for Task-Aware YOLO

Includes:
- Unit tests for all modules
- Model architecture visualization
- TensorBoard logging utilities
- Debugging tools
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime


class ModelArchitectureVisualizer:
    """Visualize model architecture and information flow."""
    
    @staticmethod
    def print_architecture(model: nn.Module, input_size: Tuple = (1, 3, 640, 640)):
        """Print detailed model architecture."""
        print("\n" + "=" * 70)
        print("MODEL ARCHITECTURE")
        print("=" * 70)
        
        # Use torchsummary if available
        try:
            from torchsummary import summary
            summary(model, input_size=input_size[1:])  # Exclude batch dim
        except ImportError:
            print("Note: Install torchsummary for detailed summary:")
            print("  pip install torchsummary")
            
            # Manual summary
            total_params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\nTotal parameters: {total_params:,.0f}")
            print(f"Trainable parameters: {trainable:,.0f}")
            print(f"Non-trainable parameters: {total_params - trainable:,.0f}")
        
        print("=" * 70)
    
    @staticmethod
    def plot_information_flow(output_path: str = "model_flow.png"):
        """Create and save a visualization of the model's information flow."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'Task-Aware YOLO Information Flow', 
                ha='center', fontsize=16, fontweight='bold')
        
        # Define boxes
        boxes = [
            (5, 8.5, 'Input Image\n640×640', 'lightblue'),
            (2.5, 7, 'Task Text\nTokenizer', 'lightgreen'),
            (7.5, 7, 'YOLOv8\nBackbone', 'lightyellow'),
            (2.5, 5.5, 'TinyLinear\nMapper', 'lightgreen'),
            (7.5, 5.5, 'P4 Features\n20×20', 'lightyellow'),
            (5, 4, 'TaskGating\nModule', 'lightcoral'),
            (7.5, 2.5, 'YOLO Head\nDetections', 'lightyellow'),
            (2.5, 2.5, 'Feature\nProjection', 'lightcyan'),
            (2.5, 0.5, 'MSE Loss\nvs CLIP', 'lightgray'),
        ]
        
        # Draw boxes
        for x, y, text, color in boxes:
            rect = patches.FancyBboxPatch(
                (x - 0.8, y - 0.4), 1.6, 0.8,
                boxstyle="round,pad=0.05",
                linewidth=2,
                edgecolor='black',
                facecolor=color
            )
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw arrows
        arrows = [
            ((5, 8.1), (7.5, 7.4)),      # Input to backbone
            ((5, 8.1), (2.5, 7.4)),      # Input to task
            ((2.5, 6.6), (2.5, 5.9)),    # Task to mapper
            ((7.5, 6.6), (7.5, 5.9)),    # Backbone to P4
            ((2.5, 5.1), (5, 4.4)),      # Mapper to gating
            ((7.5, 5.1), (5, 4.4)),      # P4 to gating
            ((5, 3.6), (7.5, 2.9)),      # Gating to YOLO head
            ((5, 3.6), (2.5, 2.9)),      # Gating to projection
            ((2.5, 2.1), (2.5, 0.9)),    # Projection to loss
        ]
        
        for (x1, y1), (x2, y2) in arrows:
            ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.15, head_length=0.1,
                    fc='black', ec='black', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Architecture visualization saved to {output_path}")
        plt.close()


class PerformanceAnalyzer:
    """Analyze and log model performance metrics."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_inference_stats(self, latencies: List[float], sparsities: List[float]):
        """Log inference performance statistics."""
        latencies = np.array(latencies)
        sparsities = np.array(sparsities)
        
        stats = {
            'mean_latency_ms': latencies.mean() * 1000,
            'median_latency_ms': np.median(latencies) * 1000,
            'std_latency_ms': latencies.std() * 1000,
            'min_latency_ms': latencies.min() * 1000,
            'max_latency_ms': latencies.max() * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000,
            'mean_sparsity': sparsities.mean(),
            'std_sparsity': sparsities.std(),
        }
        
        self.metrics['inference'] = stats
        
        print("\n" + "=" * 50)
        print("INFERENCE PERFORMANCE")
        print("=" * 50)
        for key, value in stats.items():
            if 'latency' in key:
                print(f"{key:20s}: {value:7.2f} ms")
            else:
                print(f"{key:20s}: {value:7.4f}")
        print("=" * 50)
        
        return stats
    
    def plot_loss_curves(self, train_losses: List[float], val_losses: List[float],
                        output_path: str = "loss_curves.png"):
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        ax.fill_between(epochs, train_losses, alpha=0.2, color='blue')
        ax.fill_between(epochs, val_losses, alpha=0.2, color='red')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Loss curves saved to {output_path}")
        plt.close()
    
    def plot_sparsity_evolution(self, sparsities: List[float],
                               output_path: str = "sparsity_evolution.png"):
        """Plot how sparsity changes during training."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(sparsities, 'g-', linewidth=2, marker='o')
        ax.fill_between(range(len(sparsities)), sparsities, alpha=0.3, color='green')
        
        ax.set_xlabel('Batch', fontsize=12)
        ax.set_ylabel('Gate Sparsity', fontsize=12)
        ax.set_title('Sparsity Evolution During Training', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Sparsity evolution saved to {output_path}")
        plt.close()
    
    def save_metrics_json(self, output_path: Optional[str] = None):
        """Save metrics to JSON file."""
        if output_path is None:
            output_path = self.log_dir / f"metrics_{self.timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✓ Metrics saved to {output_path}")


class ModelDebugger:
    """Debug tools for model analysis."""
    
    @staticmethod
    def check_gradient_flow(model: nn.Module, loss: torch.Tensor):
        """Check gradient flow through the model."""
        print("\n" + "=" * 50)
        print("GRADIENT FLOW ANALYSIS")
        print("=" * 50)
        
        loss.backward()
        
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                if param_norm > 0:
                    print(f"✓ {name:40s} | grad: {param_norm.item():.6f}")
                else:
                    print(f"✗ {name:40s} | grad: {param_norm.item():.6f} (NO GRADIENT)")
        
        total_norm = total_norm ** 0.5
        print(f"\nTotal gradient norm: {total_norm:.6f}")
        print("=" * 50)
    
    @staticmethod
    def check_activations(model: nn.Module, x: torch.Tensor):
        """Analyze activation statistics through the model."""
        print("\n" + "=" * 50)
        print("ACTIVATION STATISTICS")
        print("=" * 50)
        
        activation_stats = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if torch.is_tensor(output):
                    stats = {
                        'shape': tuple(output.shape),
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'has_nan': bool(torch.isnan(output).any().item()),
                        'has_inf': bool(torch.isinf(output).any().item()),
                    }
                    activation_stats[name] = stats
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.Sigmoid)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Print stats
        for name, stats in activation_stats.items():
            print(f"\n{name}")
            print(f"  Shape: {stats['shape']}")
            print(f"  Mean: {stats['mean']:8.4f} | Std: {stats['std']:8.4f}")
            print(f"  Min:  {stats['min']:8.4f} | Max: {stats['max']:8.4f}")
            if stats['has_nan']:
                print(f"  ⚠️  NaN detected!")
            if stats['has_inf']:
                print(f"  ⚠️  Inf detected!")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        print("=" * 50)


class UnitTester:
    """Unit tests for framework components."""
    
    @staticmethod
    def test_task_mapper():
        """Test TinyLinearTaskMapper."""
        from task_aware_yolo import TinyLinearTaskMapper
        
        print("\nTesting TinyLinearTaskMapper...")
        mapper = TinyLinearTaskMapper()
        
        # Test 1D input
        tokens = torch.tensor([42, 100, 200])
        output = mapper(tokens)
        assert output.shape == (3, 512), f"Expected shape (3, 512), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        
        print("✓ TinyLinearTaskMapper passed")
    
    @staticmethod
    def test_gating_module():
        """Test TaskGatingModule."""
        from task_aware_yolo import TaskGatingModule
        
        print("\nTesting TaskGatingModule...")
        gating = TaskGatingModule(feature_channels=512)
        
        task_vec = torch.randn(4, 512)
        features = torch.randn(4, 512, 20, 20)
        
        output = gating(task_vec, features)
        
        assert output.shape == features.shape, f"Shape mismatch: {output.shape} vs {features.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        
        # Check that output is actually modulated
        assert not torch.allclose(output, features), "Gating did not modulate features"
        
        print("✓ TaskGatingModule passed")
    
    @staticmethod
    def test_distillation_loss():
        """Test DistillationLoss."""
        from clip_teacher import DistillationLoss
        
        print("\nTesting DistillationLoss...")
        loss_fn = DistillationLoss()
        
        yolo_loss = torch.tensor(2.5, requires_grad=True)
        student_hm = torch.randn(4, 1, 14, 14, requires_grad=True)
        teacher_hm = torch.randn(4, 1, 14, 14)
        
        losses = loss_fn(yolo_loss, student_hm, teacher_hm)
        
        assert 'total_loss' in losses, "Missing total_loss"
        assert 'yolo_loss' in losses, "Missing yolo_loss"
        assert 'distill_loss' in losses, "Missing distill_loss"
        
        # Check backward
        losses['total_loss'].backward()
        assert yolo_loss.grad is not None, "Gradient not computed"
        
        print("✓ DistillationLoss passed")
    
    @staticmethod
    def run_all_tests():
        """Run all unit tests."""
        print("\n" + "=" * 50)
        print("RUNNING UNIT TESTS")
        print("=" * 50)
        
        try:
            UnitTester.test_task_mapper()
            UnitTester.test_gating_module()
            UnitTester.test_distillation_loss()
            
            print("\n" + "=" * 50)
            print("✓ ALL TESTS PASSED")
            print("=" * 50)
            return True
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {e}")
            return False
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            return False


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Testing & Visualization Utilities')
    parser.add_argument('action', choices=['test', 'visualize', 'analyze'],
                       help='Action to perform')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if args.action == 'test':
        UnitTester.run_all_tests()
    
    elif args.action == 'visualize':
        output = args.output or 'model_architecture.png'
        ModelArchitectureVisualizer.plot_information_flow(output)
    
    elif args.action == 'analyze':
        analyzer = PerformanceAnalyzer()
        # Example metrics
        latencies = np.random.exponential(scale=0.005, size=1000)
        sparsities = np.random.uniform(0.2, 0.8, size=1000)
        analyzer.log_inference_stats(latencies, sparsities)
        analyzer.plot_loss_curves(
            np.random.randn(100).cumsum() + 2.5,
            np.random.randn(100).cumsum() + 2.0,
            output=args.output or 'loss_curves.png'
        )
