import torch
from models.experimental import attempt_load

# Wczytaj model YOLOv7
model_path = r"C:\Users\obiwa\Desktop\Studia\Inzynierka\Inzynierkav2\GoodDetection\runs\train\ball_detection6\weights\best.pt"
device = 'cpu'

# Załaduj model
model = attempt_load(model_path, map_location=device)

# Kwantyzacja dynamiczna
quantized_model = torch.quantization.quantize_dynamic(
    model,  # Model, który chcemy skwantyzować
    {torch.nn.Linear},  # Typy warstw, które chcemy kwantyzować (np. Linear)
    dtype=torch.qint8  # Typ kwantyzacji
)

# Zapisz skwantyzowany model
quantized_model_path = r"C:\Users\obiwa\Desktop\Studia\Inzynierka\Inzynierkav2\GoodDetection\runs\train\ball_detection6\weights\best_quantized.pt"
torch.save(quantized_model.state_dict(), quantized_model_path)
print(f"Skwantyzowany model został zapisany pod: {quantized_model_path}")
