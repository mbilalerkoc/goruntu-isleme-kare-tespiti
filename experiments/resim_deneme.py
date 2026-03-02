from ultralytics import YOLO
import cv2
import os

# 1. Modeli ve Klasör Yolunu Belirle
model_yolu = r'../models/best_yeni.pt'
klasor_yolu = r'../images'
model = YOLO(model_yolu)

# 2. Klasördeki Resimleri Listele
# Sadece resim formatlarını (png, jpg, jpeg) filtreliyoruz
resim_uzantilari = ('.png', '.jpg', '.jpeg', '.bmp')
resimler = [f for f in os.listdir(klasor_yolu) if f.lower().endswith(resim_uzantilari)]

if not resimler:
    print(f"HATA: '{klasor_yolu}' klasöründe uygun resim bulunamadı!")
else:
    print(f"{len(resimler)} adet resim bulundu. Gezinmek için bir tuşa basın, çıkmak için 'q'ya basın.")

    # 3. Klasör İçinde Döngü
    for resim_adi in resimler:
        tam_yol = os.path.join(klasor_yolu, resim_adi)
        frame = cv2.imread(tam_yol)

        if frame is None:
            print(f"Atlanıyor: {resim_adi} okunamadı.")
            continue

        # 4. Tahmin Yap
        results = model(frame, conf=0.4)

        # 5. Çizim Yap
        resim_sonucu = results[0].plot(labels=True, conf=True)

        # Pencere Başlığına Dosya Adını Yazdıralım
        pencere_basligi = f"Nesne Tespit Sonucu - {resim_adi}"
        cv2.namedWindow("Navigasyon", cv2.WINDOW_NORMAL)
        cv2.imshow("Navigasyon", resim_sonucu)

        # 6. Tuş Kontrolü
        print(f"Görüntüleniyor: {resim_adi}")
        key = cv2.waitKey(0) & 0xFF

        # 'q' tuşuna basılırsa döngüden çık
        if key == ord('q'):
            print("Çıkış yapılıyor...")
            break

    cv2.destroyAllWindows()
    print("İşlem tamamlandı.")