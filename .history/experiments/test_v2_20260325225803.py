from ultralytics import YOLO
import cv2

MODEL_PATH = '../models/best_v2.pt'
VIDEO_SOURCE = '../videos/asfalt_kare.mp4'
GUVEN_ESIGI = 0.5

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=GUVEN_ESIGI)

    resim_sonucu = results[0].plot(labels=True, conf=True)

    cv2.imshow("Sadelestirilmis Test", resim_sonucu)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()