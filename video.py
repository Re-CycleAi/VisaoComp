from ultralytics import YOLO
import cv2


model = YOLO("runs/detect/cigarrovapedetect/weights/best.pt")


video_path = "C:/Python/meuvideo.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter('saida_detectada.mp4', fourcc, fps, (width, height))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

# Libera os recursos
cap.release()
out.release()
print("Processamento concluído. Vídeo salvo como 'saidasalva.mp4'")
