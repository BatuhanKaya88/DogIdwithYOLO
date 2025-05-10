import cv2
import numpy as np
from ultralytics import YOLO

# Modeli yükle
model = YOLO('C:/Users/Sibel Kaya/Desktop/DogIdwithYOLO/.venv/runs/train/dog_detector4/weights/best.pt')

video_path = 'C:/Users/Sibel Kaya/Desktop/DogIdwithYOLO/.venv/videos/deneme_videosu.mp4'
cap = cv2.VideoCapture(video_path)

# Çözünürlüğü artır
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Genişlik
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Yükseklik

prev_center = None
alert_state = "normal"  # normal, yellow, red
threat_level = "none"  # none, attack, fast

# Hız limitleri
speed_threshold_normal = 20  # Normal hız için limit
speed_threshold_fast = 50   # Hızlı hareket için limit

# Siluet tespiti için fonksiyon
def is_likely_human_silhouette(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h / float(w)
    return aspect_ratio > 1.6  # 2 ayaklılar için yüksek oran

# Uyarı mesajlarını göstermek için fonksiyon
def show_alert(message, color=(0, 255, 255)):
    """Uyarı mesajını küçük boyutlu ekranda gösterir."""
    alert_img = np.zeros((100, 500, 3), dtype=np.uint8)  # Uyarı ekranını oluştur
    cv2.putText(alert_img, message, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)  # Yazı boyutunu küçült
    cv2.imshow("Uyarı", alert_img)  # Ekranda uyarı mesajını göster

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # Video frame'inde köpek algıla
    dog_detected = False
    current_threat_level = "none"  # Geçici tehdit seviyesi

    # Hareket ve siluet tespiti için konturları bul
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if is_likely_human_silhouette(contour):  # Eğer insan tespiti yapılmışsa
            current_threat_level = "human"
            alert_state = "yellow"
            break  # İnsan tespit edilirse köpek algılamasına geçme

    for box in results.boxes:
        cls = int(box.cls[0])  # Sınıfı al (0 köpek sınıfı)
        if cls == 0:  # Köpek algılandıysa
            dog_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Çerçeve koordinatları
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Merkez koordinatları

            # Hareket kontrolü
            if prev_center:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                distance = (dx ** 2 + dy ** 2) ** 0.5

                # Hız analizi: Hızlı hareket için sarı uyarı
                if distance > speed_threshold_fast:
                    current_threat_level = "fast"
                    alert_state = "yellow"
                elif distance > speed_threshold_normal:
                    current_threat_level = "normal"

                # Kırmızı uyarı: saldırgan yaklaşım
                if (x2 - x1) > 200 and current_threat_level != "attack":  # Yaklaşan köpek büyük ve hızlı
                    current_threat_level = "attack"
                    alert_state = "red"

            prev_center = (cx, cy)

            # Video üzerine çerçeve çizme
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Çerçeveyi çiz
            cv2.putText(frame, "Kopek", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Hız ve tehdit seviyesini ekrana yazma
            if current_threat_level == "attack":
                cv2.putText(frame, "Tehdit: Saldiri", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif current_threat_level == "fast":
                cv2.putText(frame, "Tehdit: HIZLI", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            break

    # Uyarı ekranı
    if alert_state == "yellow":
        show_alert("SARI UYARI: Koşuyor!", (0, 255, 255))  # Sarı uyarı göster
    elif alert_state == "red":
        show_alert("KIRMIZI UYARI: SALDIRI!", (0, 0, 255))  # Kırmızı uyarı göster
    elif alert_state == "normal" and dog_detected:
        show_alert("Durum Normal", (0, 255, 0))  # Durum normalse yeşil uyarı göster

    # Durum normalleştirme
    if alert_state == "yellow" and not dog_detected:
        alert_state = "normal"
        prev_center = None

    # Video görüntüsünü göster
    cv2.imshow("Video", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):  # 'q' tuşuna basılınca çık
        break

cap.release()
cv2.destroyAllWindows()  # Tüm pencereleri kapat
