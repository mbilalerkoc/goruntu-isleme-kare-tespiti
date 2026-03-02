import cv2
import numpy as np
import math

SIZE = 1024
FPS = 30
DURATION = 20
FRAMES = FPS * DURATION

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("iha_mission_simulation.mp4", fourcc, FPS, (SIZE,SIZE))

for i in range(FRAMES):
    frame = np.full((SIZE,SIZE,3), 110, dtype=np.uint8)

    # Sonsuz manevra hareketi
    x = int(400 + 200 * math.sin(i/20))
    y = int(500 + 150 * math.sin(i/10))

    cv2.rectangle(frame, (x,y), (x+200,y+200), (255,0,0), -1)
    cv2.rectangle(frame, (600-x//3,600-y//4), (650-x//3,650-y//4), (0,0,255), -1)

    video.write(frame)

video.release()
print("VIDEO TAMAMLANDI")