import math

# Sabitler
G = 9.81

def yatay_mesafe_hesapla(hedef_y_pikseli, resim_h, vfov, yukseklik_m):
    """Görüntüdeki pikselden yerdeki yatay mesafeyi bulur."""
    merkez_y = resim_h / 2
    pixel_offset = merkez_y - hedef_y_pikseli
    aci_radyan = (pixel_offset / resim_h) * math.radians(vfov)
    return yukseklik_m * math.tan(aci_radyan)

def optik_mesafe_hesapla(gercek_w_cm, odak_uzakligi, piksel_w):
    """Kameradan hedefe olan doğrudan mesafeyi (cm) hesaplar."""
    if piksel_w == 0: return 0
    return (gercek_w_cm * odak_uzakligi) / piksel_w

def atis_menzili_hesapla(yukseklik_m, hiz_uav, ruzgar_hizi):
    """Kara aracının bırakıldıktan sonra düşeceği yatay mesafeyi hesaplar."""
    dusme_zamani = math.sqrt((2 * yukseklik_m) / G)
    dusme_araligi = (hiz_uav + ruzgar_hizi) * dusme_zamani
    return dusme_araligi