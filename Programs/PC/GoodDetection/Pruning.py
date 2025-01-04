import torch
import torch.nn.utils.prune as prune
from models.experimental import attempt_load

# Wczytaj model YOLOv7
model_path = r"C:\Users\obiwa\Desktop\Studia\Inzynierka\Inzynierkav2\GoodDetection\runs\train\ball_detection6\weights\best.pt"
device = 'cpu'

model = attempt_load(model_path, map_location=device)

# Pruning modelu
for module_name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)  # Usuwamy 30% najmniej istotnych połączeń

# Zapisz model po pruningu
pruned_model_path = r"C:\Users\obiwa\Desktop\Studia\Inzynierka\Inzynierkav2\GoodDetection\runs\train\ball_detection6\weights\best_pruned.pt"
torch.save(model.state_dict(), pruned_model_path)
print(f"Pruned model został zapisany pod: {pruned_model_path}")
