from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO("../models/best_yeni.pt.pt")
cap = cv2.VideoCapture("../videos/coklu_asfalt_zemin.mp4")

while True:
    success, frame = cap.read()
    if not success:
        break
    results = model(frame, conf=0.4, verbose=False)

    tespitler = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            alan = w * h
            tespitler.append({'coords': (x1, y1, x2, y2), 'alan': alan})

    if len(tespitler) >= 2:
        en_buyuk_kare = max(tespitler, key=lambda x: x['alan'])

        for t in tespitler:
            coords = t['coords']
            alan = t['alan']

            if t == en_buyuk_kare:
                etiket = f"BUYUK KARE ({alan})"
                renk = (0, 255, 0)
            else:
                etiket = f"KUCUK KARE ({alan})"
                renk = (0, 255, 255)

            # Çizim
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), renk, 3)
            cvzone.putTextRect(frame, etiket, (max(0, coords[0]), max(35, coords[1])), scale=1.5, thickness=2,
                               colorR=renk)

    elif len(tespitler) == 1:
        t = tespitler[0]
        alan = t['alan']
        coords = t['coords']

        durum = "Kare"
        renk = (0, 255, 0)

        etiket = f"{durum} Alan: {alan}"
        cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), renk, 3)
        cvzone.putTextRect(frame, etiket, (max(0, coords[0]), max(35, coords[1])), scale=1.5, thickness=2, colorR=renk)


    cv2.imshow("TEKNOFEST IHA GORUS", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()