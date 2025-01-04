import torch
import cv2
import time
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

# Ścieżka do modelu
model_path = r"C:\Users\obiwa\Desktop\Studia\Inzynierka\Inzynierkav2\GoodDetection\runs\train\ball_detection6\weights\best.pt"  # Zmień na rzeczywistą ścieżkę do modelu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(model_path, map_location=device)
model.eval()

# Parametry obrazu i modelu
frame_width = 640
frame_height = 480
square_size = 150
square_x1 = int((frame_width / 2) - (square_size / 2))
square_y1 = int((frame_height / 2) - (square_size / 2))
square_x2 = int((frame_width / 2) + (square_size / 2))
square_y2 = int((frame_height / 2) + (square_size / 2))

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

if not cap.isOpened():
    print("Nie udało się otworzyć kamery.")
    exit()

# Zmienna do obliczania FPS
prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nie udało się odczytać obrazu z kamery.")
            break

        # Przeskalowanie obrazu do wymagań modelu
        img = cv2.resize(frame, (320, 320))
        img_tensor = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        # Detekcja za pomocą modelu
        with torch.no_grad():
            predictions = model(img_tensor)[0]

        predictions = non_max_suppression(predictions, conf_thres=0.10, iou_thres=0.45)

        # Wyświetlanie detekcji na obrazie
        for det in predictions:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = f"Conf: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Rysowanie "kwadratu" w centrum obrazu
        cv2.rectangle(frame, (square_x1, square_y1), (square_x2, square_y2), (255, 0, 0), 2)

        # Obliczanie FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Wyświetlanie FPS w konsoli
        print(f"FPS: {fps:.2f}")

        # Wyświetlanie obrazu
        cv2.imshow("Detection", frame)

        # Wyjście po naciśnięciu klawisza 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Zwolnienie zasobów
    cap.release()
    cv2.destroyAllWindows()
