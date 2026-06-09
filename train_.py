import os
import cv2
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Đường dẫn dataset
DATASET = "dataset"

# Tự động lấy tên các thư mục màu
mau_sac = [
    folder for folder in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, folder))
]

print("Các màu :", mau_sac)

X = []
y = []

for mau in mau_sac:

    folder = os.path.join(DATASET, mau)

    print("Đang đọc:", folder)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            print("Không đọc được:", path)
            continue

        img = cv2.resize(img, (100, 100))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
         [hsv],
         [0, 1],
         None,
         [18, 16],
         [0, 180, 0, 256]
       )

        hist = cv2.normalize(hist, hist)

        dac_trung = hist.flatten()
    

        X.append(dac_trung)
        y.append(mau)

      

# Kiểm tra trước khi train
if len(X) == 0:
    raise Exception(
        "Không tìm thấy ảnh nào trong dataset. "
        "Kiểm tra lại thư mục và định dạng ảnh."
    )
X = np.array(X)

print("Tổng mẫu:", len(X))
print("Tổng nhãn:", len(y))
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

du_doan = model.predict(X_test)
print(classification_report(y_test, du_doan))

do_chinh_xac = accuracy_score(y_test, du_doan)

print("Độ chính xác =", do_chinh_xac)

joblib.dump(model, "color_model.pkl")

print("Đã lưu color_model.pkl")
print(classification_report(y_test, du_doan))