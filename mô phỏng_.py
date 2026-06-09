import cv2
import numpy as np
import joblib
import pygame
from collections import deque, Counter
# Load model
model = joblib.load("color_model.pkl")

# Chống nháy
history = deque(maxlen=10)

# Nhớ màu trước đó
last_color = None


# Webcam
cap = cv2.VideoCapture(0)

# Pygame
pygame.init()

WIDTH = 800
HEIGHT = 500

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Simulation")

font = pygame.font.SysFont(None, 40)
clock = pygame.time.Clock()

# Robot Positions

A = (400, 100)   # RED
B = (200, 350)   # GREEN
C = (600, 350)   # BLUE

robot_x, robot_y = A

last_color = None
running = True

while running:
    # Webcam
    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    cx = w // 2
    cy = h // 2

    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

    cv2.rectangle(
        frame,
        (cx - 50, cy - 50),
        (cx + 50, cy + 50),
        (0, 255, 0),
        2
    )

    roi = frame[
        cy - 50:cy + 50,
        cx - 50:cx + 50
    ]

    mau = "unknown"

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

        feature = hist.flatten()

        prediction = model.predict(
    feature.reshape(1, -1)
)[0]

        history.append(prediction)

        mau = Counter(history).most_common(1)[0][0]

        cv2.putText(
            frame,
            f"Color: {mau}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        
    # Robot Action
    action = "WAIT"
    
    if mau != last_color:
    
        if mau == "red":
    
            robot_x, robot_y = A
            action = "GO TO A"
    
        elif mau == "green":
    
            robot_x, robot_y = B
            action = "GO TO B"
    
        elif mau == "blue":
    
            robot_x, robot_y = C
            action = "GO TO C"
    
        last_color = mau    

    # Webcam Window
    cv2.imshow(
        "Color Detection",
        frame
    )
    
    # Robot Simulation Window
    screen.fill((240, 240, 240))
    
    # Tam giác
    pygame.draw.polygon(
        screen,
        (0, 0, 0),
        [A, B, C],
        3
    )
    
    font_small = pygame.font.SysFont(None, 30)
    
    # Nhãn đỉnh
    screen.blit(
        font_small.render("A - RED", True, (255, 0, 0)),
        (A[0] - 40, A[1] - 30)
    )
    
    screen.blit(
        font_small.render("B - GREEN", True, (0, 180, 0)),
        (B[0] - 50, B[1] + 15)
    )
    
    screen.blit(
        font_small.render("C - BLUE", True, (0, 0, 255)),
        (C[0] - 45, C[1] + 15)
    )
    
    # Robot
    pygame.draw.circle(
        screen,
        (255, 140, 0),
        (robot_x, robot_y),
        20
    )
    
    # Thông tin
    text1 = font.render(
        f"Color: {mau}",
        True,
        (0, 0, 0)
    )
    
    text2 = font.render(
        f"Action: {action}",
        True,
        (255, 0, 0)
    )
    
    screen.blit(text1, (20, 20))
    screen.blit(text2, (20, 80))
    pygame.display.flip()

    # Event
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    if cv2.waitKey(1) & 0xFF == 27:
        running = False

    clock.tick(60)

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
mean_h = np.mean(hsv[:, :, 0])


