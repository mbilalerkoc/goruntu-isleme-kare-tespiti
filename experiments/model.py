import cv2
from ultralytics import YOLO
import hesaplamalar as fizik # Kendi yazdığımız dosyayı içe aktarıyoruz

# --- AYARLAR ---
MODEL_YOLU = '../models/best_v5.pt'
VIDEO_YOLU = '../videos/asfalt_kare.mp4'
GERCEK_W_CM = 50.0
ODAK_UZAKLIGI = 700
YUKSEKLIK_M = 20.0
VFOV = 45.0
RESIM_H = 1080
HIZ = 15.0
RUZGAR = 2.0

# 1. Modeli ve Menzili Hazırla
model = YOLO(MODEL_YOLU)
menzil = fizik.atis_menzili_hesapla(YUKSEKLIK_M, HIZ, RUZGAR)

def process_ai_video():
    cap = cv2.VideoCapture(VIDEO_YOLU)
    print(f"Otomasyon Başladı. Menzil: {menzil:.2f}m")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, conf=0.7, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w_p = x2 - x1
                m_x, m_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # --- HESAPLAMALAR (Modülden çağırıyoruz) ---
                optik_d = fizik.optik_mesafe_hesapla(GERCEK_W_CM, ODAK_UZAKLIGI, w_p)
                yatay_d = fizik.yatay_mesafe_hesapla(m_y, RESIM_H, VFOV, YUKSEKLIK_M)

                # --- DURUM ---
                if yatay_d <= menzil:
                    color, msg = (0, 0, 255), "!!! BIRAK !!!"
                else:
                    color, msg = (0, 255, 0), f"Kalan: {yatay_d - menzil:.1f}m"

                # --- GÖRSELLEŞTİRME ---
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"Kare | D: {optik_d:.1f}cm | XY: ({m_x},{m_y})"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, msg, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("KAPSUL-ALTAY Moduler Analiz", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_ai_video()