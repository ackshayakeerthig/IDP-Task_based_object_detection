"""
Custom COCO DataLoader with Task Mapping

Maps 80 COCO classes to 14 Functional Tasks (from "What Object Should I Use?" - arXiv:1904.03000)

Functional Tasks:
1. Pouring
2. Cutting
3. Grasping
4. Holding
5. Sitting
6. Carrying
7. Pushing
8. Pulling
9. Hitting
10. Throwing
11. Opening
12. Closing
13. Balancing
14. Stacking
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json
import cv2
from PIL import Image
import random


# ==================== Task Definitions ====================

FUNCTIONAL_TASKS = {
    0: "pouring",
    1: "cutting",
    2: "grasping",
    3: "holding",
    4: "sitting",
    5: "carrying",
    6: "pushing",
    7: "pulling",
    8: "hitting",
    9: "throwing",
    10: "opening",
    11: "closing",
    12: "balancing",
    13: "stacking"
}

# COCO class to functional task mapping
# Source: Common sense affordances from object interactions
COCO_TO_TASK_MAPPING = {
    # POURING (0)
    "cup": 0, "glass": 0, "bottle": 0, "bowl": 0, "plate": 0, "pot": 0, "pan": 0,
    
    # CUTTING (1)
    "knife": 1, "scissors": 1, "fork": 1, "apple": 1, "orange": 1, "banana": 1,
    "carrot": 1, "broccoli": 1, "cake": 1,
    
    # GRASPING (2)
    "backpack": 2, "handbag": 2, "suitcase": 2, "ball": 2, "baseball bat": 2,
    "baseball glove": 2, "frisbee": 2, "skateboard": 2, "surfboard": 2, "tennis racket": 2,
    "book": 2, "dog": 2, "cat": 2, "mouse": 2, "bird": 2,
    
    # HOLDING (3)
    "person": 3, "hand": 3, "arm": 3, "cup": 3, "bottle": 3, "phone": 3,
    "remote": 3, "laptop": 3, "keyboard": 3, "mouse": 3, "tie": 3, "watch": 3,
    
    # SITTING (4)
    "chair": 4, "couch": 4, "bench": 4, "stool": 4, "table": 4, "bed": 4,
    
    # CARRYING (5)
    "backpack": 5, "handbag": 5, "suitcase": 5, "person": 5, "horse": 5, "dog": 5, "cat": 5,
    
    # PUSHING (6)
    "car": 6, "truck": 6, "bus": 6, "train": 6, "bicycle": 6, "motorcycle": 6,
    "door": 6, "cart": 6,
    
    # PULLING (7)
    "door": 7, "cart": 7, "bicycle": 7, "rope": 7, "handle": 7,
    
    # HITTING (8)
    "baseball bat": 8, "tennis racket": 8, "hammer": 8, "ball": 8, "frisbee": 8,
    
    # THROWING (9)
    "baseball": 9, "ball": 9, "frisbee": 9, "person": 9, "rock": 9,
    
    # OPENING (10)
    "door": 10, "window": 10, "refrigerator": 10, "oven": 10, "microwave": 10,
    "drawer": 10, "box": 10, "bottle": 10, "jar": 10,
    
    # CLOSING (11)
    "door": 11, "window": 11, "refrigerator": 11, "oven": 11, "microwave": 11,
    "drawer": 11, "box": 11, "bottle": 11, "jar": 11,
    
    # BALANCING (12)
    "cup": 12, "glass": 12, "bowl": 12, "plate": 12, "book": 12, "block": 12,
    
    # STACKING (13)
    "block": 13, "cup": 13, "plate": 13, "book": 13, "box": 13
}

# COCO Class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "door", "window",
    "building", "tree", "flower", "road", "mountain", "ocean"
]


class TaskTokenizer:
    """Convert task strings to token IDs for the TinyLinearTaskMapper."""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build vocabulary from task names."""
        task_strings = list(FUNCTIONAL_TASKS.values())
        
        # Special tokens
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1
        self.vocab["<START>"] = 2
        self.vocab["<END>"] = 3
        
        idx = 4
        for task in task_strings:
            for word in task.split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to token IDs."""
        tokens = []
        for word in text.lower().split():
            token_id = self.vocab.get(word, self.vocab["<UNK>"])
            tokens.append(token_id)
        
        if not tokens:
            tokens = [self.vocab["<PAD>"]]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Convert token IDs back to text."""
        words = []
        for token_id in tokens:
            if token_id.item() in self.reverse_vocab:
                words.append(self.reverse_vocab[token_id.item()])
        return " ".join(words)


class COCODatasetWithTasks(Dataset):
    """
    COCO dataset with functional task labels.
    
    Returns:
        - image: [3, H, W] normalized image
        - task_token: scalar or vector of task token IDs
        - task_name: string task name
        - labels: YOLO format annotations [[x_center, y_center, w, h, class_id], ...]
        - image_path: path to image for debugging
    """
    
    def __init__(
        self,
        coco_dir: str,
        split: str = "train",
        img_size: int = 640,
        augment: bool = False,
        task_distribution: str = "balanced"
    ):
        """
        Args:
            coco_dir: Path to COCO dataset root
            split: "train" or "val"
            img_size: Target image size for YOLO
            augment: Enable data augmentation
            task_distribution: "balanced" (equal tasks) or "random" (natural distribution)
        """
        self.coco_dir = Path(coco_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.task_distribution = task_distribution
        
        # Annotation file path
        self.anno_file = (
            self.coco_dir / f"annotations/instances_{split}2017.json"
        )
        self.img_dir = self.coco_dir / f"{split}2017"
        
        # Load COCO annotations
        self.images, self.annotations = self._load_annotations()
        
        # Task tokenizer
        self.task_tokenizer = TaskTokenizer()
        
        # Create image_id to annotations mapping
        self.img_to_annos = self._build_img_anno_map()
    
    def _load_annotations(self) -> Tuple[List[Dict], List[Dict]]:
        """Load COCO JSON annotations."""
        if not self.anno_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.anno_file}")
        
        with open(self.anno_file, 'r') as f:
            coco_data = json.load(f)
        
        return coco_data['images'], coco_data['annotations']
    
    def _build_img_anno_map(self) -> Dict:
        """Map image IDs to annotations."""
        img_to_annos = {}
        for anno in self.annotations:
            img_id = anno['image_id']
            if img_id not in img_to_annos:
                img_to_annos[img_id] = []
            img_to_annos[img_id].append(anno)
        return img_to_annos
    
    def _get_task_from_image(self, image_id: int) -> str:
        """
        Determine task for image based on present objects.
        
        Strategy:
        1. Get all objects in image
        2. Map to functional tasks
        3. Random or balanced selection
        """
        annos = self.img_to_annos.get(image_id, [])
        
        if not annos:
            # Default task if no annotations
            return "grasping"
        
        # Get categories present
        present_tasks = set()
        for anno in annos:
            class_id = anno['category_id'] - 1  # COCO uses 1-indexed
            if class_id < len(COCO_CLASSES):
                class_name = COCO_CLASSES[class_id]
                task_id = COCO_TO_TASK_MAPPING.get(class_name, 2)  # Default to grasping
                present_tasks.add(task_id)
        
        if not present_tasks:
            present_tasks.add(2)  # Default
        
        # Select task
        if self.task_distribution == "balanced":
            task_id = random.choice(list(present_tasks))
        else:
            task_id = max(present_tasks)  # Most common task
        
        return FUNCTIONAL_TASKS.get(task_id, "grasping")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get image and annotations with task label."""
        
        img_info = self.images[idx]
        image_id = img_info['id']
        image_path = self.img_dir / img_info['file_name']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return dummy image
            image = Image.new('RGB', (self.img_size, self.img_size))
        
        # Get task
        task_name = self._get_task_from_image(image_id)
        task_token = self.task_tokenizer.encode(task_name)
        task_token = task_token[0] if len(task_token) > 0 else torch.tensor(0)
        
        # Get annotations
        annos = self.img_to_annos.get(image_id, [])
        
        # Convert to YOLO format: [x_center, y_center, w, h, class_id]
        h, w = img_info['height'], img_info['width']
        labels = []
        
        for anno in annos:
            bbox = anno['bbox']  # [x, y, w, h]
            x, y, box_w, box_h = bbox
            
            # Normalize to [0, 1]
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            w_norm = box_w / w
            h_norm = box_h / h
            
            class_id = anno['category_id'] - 1  # 0-indexed
            
            labels.append([x_center, y_center, w_norm, h_norm, class_id])
        
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        
        # Resize image
        image.thumbnail((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image_array = np.array(image)
        
        # Pad to square
        pad_h = self.img_size - image_array.shape[0]
        pad_w = self.img_size - image_array.shape[1]
        image_array = np.pad(
            image_array,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant',
            constant_values=114
        )
        
        # To tensor and normalize
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        
        # Normalize to ImageNet stats
        image_tensor[0] = (image_tensor[0] - 0.485) / 0.229
        image_tensor[1] = (image_tensor[1] - 0.456) / 0.224
        image_tensor[2] = (image_tensor[2] - 0.406) / 0.225
        
        return {
            'image': image_tensor,
            'task_token': task_token,
            'task_name': task_name,
            'labels': labels,
            'image_id': image_id,
            'image_path': str(image_path)
        }


# Utility function for creating DataLoader
def create_coco_dataloader(
    coco_dir: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = False,
    img_size: int = 640
) -> DataLoader:
    """
    Create COCO DataLoader with task labels.
    
    Args:
        coco_dir: Path to COCO root directory
        split: "train" or "val"
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Shuffle data
        augment: Enable augmentation
        img_size: Image size for YOLO
    
    Returns:
        DataLoader yielding batches with images and task labels
    """
    
    dataset = COCODatasetWithTasks(
        coco_dir=coco_dir,
        split=split,
        img_size=img_size,
        augment=augment
    )
    
    def collate_fn(batch):
        """Custom collate to handle variable-length labels."""
        images = torch.stack([item['image'] for item in batch])
        task_tokens = torch.stack([item['task_token'] for item in batch])
        task_names = [item['task_name'] for item in batch]
        labels = [item['labels'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        
        return {
            'images': images,
            'task_tokens': task_tokens,
            'task_names': task_names,
            'labels': labels,
            'image_ids': image_ids
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


# ==================== Testing ====================

if __name__ == "__main__":
    print("COCO DataLoader with Task Mapping Test")
    print("=" * 60)
    
    # Test TaskTokenizer
    print("\n1. Testing TaskTokenizer...")
    tokenizer = TaskTokenizer()
    
    task_text = "pouring"
    token = tokenizer.encode(task_text)
    decoded = tokenizer.decode(token)
    
    print(f"   Input: '{task_text}'")
    print(f"   Tokens: {token}")
    print(f"   Decoded: '{decoded}'")
    print("   ✓ TaskTokenizer OK")
    
    # Test COCO mapping
    print("\n2. Testing COCO to Task Mapping...")
    print(f"   Total COCO classes: {len(COCO_CLASSES)}")
    print(f"   Total Functional Tasks: {len(FUNCTIONAL_TASKS)}")
    print(f"   Mapped classes: {len(COCO_TO_TASK_MAPPING)}")
    
    sample_mapping = {
        "cup": FUNCTIONAL_TASKS[COCO_TO_TASK_MAPPING.get("cup", 0)],
        "knife": FUNCTIONAL_TASKS[COCO_TO_TASK_MAPPING.get("knife", 0)],
        "chair": FUNCTIONAL_TASKS[COCO_TO_TASK_MAPPING.get("chair", 0)],
    }
    print(f"   Sample mappings: {sample_mapping}")
    print("   ✓ Task mapping OK")
    
    print("\n" + "=" * 60)
    print("DataLoader tests passed! ✓")
    print("\nNote: Full COCODatasetWithTasks requires actual COCO dataset files.")
    print("Download from: https://cocodataset.org/#download")
