import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torchvision import transforms
from ultralytics import YOLO

TASKS = [
    "pouring", "cutting", "grasping", "holding", "sitting", "carrying", 
    "pushing", "pulling", "hitting", "throwing", "opening", "closing", 
    "balancing", "stacking"
]

task_to_id = {task: i for i, task in enumerate(TASKS)}
id_to_task = {i: task for i, task in enumerate(TASKS)}

def get_task_tensor(task_name, device):
    if task_name not in task_to_id:
        print(f"⚠️ Warning: '{task_name}' not in vocabulary. Defaulting to 'pouring'.")
        tid = 0
    else:
        tid = task_to_id[task_name]
    return torch.tensor([tid]).to(device)

class TaskGatingModule(nn.Module):
    def __init__(self, feature_channels=512, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_channels),
            nn.Sigmoid()
        )

    def forward(self, task_vector, feature_map):
        gates = self.mlp(task_vector)
        gates = gates.to(feature_map.device) 
        gates = gates.unsqueeze(-1).unsqueeze(-1)
        gated_features = feature_map * gates
        return gated_features

class FeatureProjectionHead(nn.Module):
    def __init__(self, in_channels=512, target_size=(14, 14)):
        super().__init__()
        self.proj_conv = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)
        self.target_size = target_size

    def forward(self, student_features):
        student_features = student_features.to(self.proj_conv.weight.device)
        x = self.proj_conv(student_features)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return x

class TinyLinearTaskMapper(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=256, semantic_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, semantic_dim)
        self.norm = nn.LayerNorm(semantic_dim)

    def forward(self, task_tokens):
        if task_tokens.dim() == 1:
            x = self.embedding(task_tokens)
            x = x.unsqueeze(1)
            x = x.mean(dim=1)
        else:
            x = self.embedding(task_tokens)
            x = x.mean(dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x

class TaskAwareYOLOWithHooks(nn.Module):
    def __init__(self, yolo_model, semantic_dim=512, target_grid_size=(14, 14), vocab_size=14):
        super().__init__()
        self.yolo_model = yolo_model
        self.task_mapper = TinyLinearTaskMapper(vocab_size=vocab_size, semantic_dim=semantic_dim)
        self.gating_p5 = TaskGatingModule(feature_channels=256, hidden_dim=semantic_dim)
        self.feature_projection = FeatureProjectionHead(
            in_channels=256, target_size=target_grid_size
        )
        self.captured_p5 = None

    def forward(self, images, task_tokens):
        task_vector = self.task_mapper(task_tokens)
        
        def hook_fn(module, input, output):
            self.captured_p5 = output

        handle = self.yolo_model.model[21].register_forward_hook(hook_fn)
        detections = self.yolo_model(images)
        handle.remove()

        gated_features = self.gating_p5(task_vector, self.captured_p5)
        student_heatmap = self.feature_projection(gated_features)

        return student_heatmap, detections

def load_model(weights_path, device):
    yolo_skeleton = YOLO('yolov8n.pt').model.to(device)
    model = TaskAwareYOLOWithHooks(yolo_skeleton, vocab_size=14).to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

def extract_verb_clip(question_text, clip_model, clip_preprocess, device):
    """
    Zero-shot matches the user question to one of the 14 trained tasks using CLIP.
    """
    text_tokens = clip.tokenize([question_text] + TASKS, truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features.float(), dim=-1)
        
    question_feat = text_features[0].unsqueeze(0)
    tasks_feat = text_features[1:]
    
    similarities = torch.matmul(question_feat, tasks_feat.T).squeeze()
    best_idx = torch.argmax(similarities).item()
    return TASKS[best_idx]
