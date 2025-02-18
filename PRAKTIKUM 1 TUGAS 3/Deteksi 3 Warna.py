import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rentang warna dalam HSV
    color_ranges = {
        'Merah': ([0, 120, 70], [10, 255, 255], (0, 0, 255)),  # BGR Merah
        'Hijau': ([40, 40, 40], [80, 255, 255], (0, 255, 0)),  # BGR Hijau
        'Biru': ([90, 50, 50], [130, 255, 255], (255, 0, 0))   # BGR Biru
    }

    for color, (lower, upper, bbox_color) in color_ranges.items():
        lower_np = np.array(lower, np.uint8)
        upper_np = np.array(upper, np.uint8)
        mask = cv2.inRange(hsv, lower_np, upper_np)
        
        # Temukan kontur
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter area kecil
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 2)
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

    # Menampilkan hasil hanya dalam satu tampilan
    cv2.imshow("Deteksi Warna", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
