import cv2
import joblib
import numpy as np
from collections import deque, Counter

model = joblib.load("color_model.pkl")

history = deque(maxlen=10)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    cx = w // 2
    cy = h // 2

    # Vẽ dấu cộng
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

    # Vẽ khung nhận diện
    cv2.rectangle(
        frame,
        (cx - 50, cy - 50),
        (cx + 50, cy + 50),
        (0, 255, 0),
        2
    )

    # Chỉ lấy vùng giữa
    roi = frame[
        cy - 50:cy + 50,
        cx - 50:cx + 50
    ]

    if roi.size > 0:

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [18, 16],
            [0, 180, 0, 256]
        )

        hist = cv2.normalize(hist, hist)

        dac_trung = hist.flatten()

        prediction = model.predict(
            dac_trung.reshape(1, -1)
        )[0]

        # chống nháy
        history.append(prediction)

        mau = Counter(history).most_common(1)[0][0]

        cv2.putText(
            frame,
            f"Mau: {mau}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Nhan dang mau", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()