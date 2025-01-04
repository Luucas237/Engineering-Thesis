import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from utils.general import scale_coords

# Ścieżka do modelu YOLOv8
path = r'C:\Users\obiwa\Desktop\Studia\Inzynierka\Inzynierkav2\BetterDetection\best_320.pt'

# Wczytanie modelu
model = YOLO(path)

# Parametry obrazu
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

        # Wykonywanie detekcji za pomocą YOLOv8
        results = model(frame, conf=0.3, verbose=False)

        # Wyświetlanie detekcji na obrazie
        for result in results:
            if result.boxes is not None and len(result.boxes):
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Współrzędne prostokąta
                    confidence = box.conf[0]  # Pewność detekcji
                    label = f"Conf: {confidence:.2f}"
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
