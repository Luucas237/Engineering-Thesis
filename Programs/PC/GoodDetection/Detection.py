import torch
import cv2
import time
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

model_path = r"C:\Users\obiwa\Desktop\Studia\Inzynierka\Inzynierkav2\GoodDetection\runs\train\ball_detection6\weights\best.pt"
device = select_device('cpu') 

model = attempt_load(model_path, map_location=device)
model.eval()  

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No camera found")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_width = frame.shape[1]

    left_line_x = int(frame_width * 0.25)  
    right_line_x = int(frame_width * 0.75) 

    # Tworzenie kopii ramki do detekcji o rozdzielczości 320x320
    img = cv2.resize(frame, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # Detekcja
    with torch.no_grad():
        predictions = model(img)[0]
    predictions = non_max_suppression(predictions, conf_thres=0.10, iou_thres=0.45)

    # Skalowanie współrzędnych do oryginalnego rozmiaru obrazu
    for i, det in enumerate(predictions):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                # Rysowanie ramki
                label = f'{model.names[int(cls)]} {conf:.2f}'
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Obliczenie pozycji środkowej ramki
                box_center_x = (x1 + x2) / 2 

                if box_center_x < left_line_x:
                    position = "left"
                elif left_line_x <= box_center_x <= right_line_x:
                    position = "middle"
                else:
                    position = "right"
                
                print(position)
    cv2.line(frame, (left_line_x, 0), (left_line_x, frame.shape[0]), (255, 0, 0), 2)    # Linia pionowa dla "left"
    cv2.line(frame, (right_line_x, 0), (right_line_x, frame.shape[0]), (255, 0, 0), 2)  # Linia pionowa dla "right"

    cv2.imshow('YOLOv7 Ball Detection', frame)

    fps = 1 / (time.time() - prev_time)
    prev_time = current_time
    print(f"FPS: {fps:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
