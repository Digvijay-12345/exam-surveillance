"""Scripted classroom simulation (12 students) for demos without real footage.
It produces person/phone boxes and feeds the SAME tracker + rule engine as real video."""
import cv2, random
import numpy as np

W, H = 960, 540
PERIOD = 520

class ClassroomSim:
    def __init__(self, seed=1):
        random.seed(seed)
        self.t = 0
        self.centers = [(140 + c * 220, 130 + r * 170) for r in range(3) for c in range(4)]

    def step(self):
        ph = self.t % PERIOD
        boxes, phones = [], []
        for i, (cx, cy) in enumerate(self.centers):
            x, top, bot = cx + random.gauss(0, .8), cy - 55, cy + 55
            if i == 2 and 170 <= ph < 240:                      # student 3 stands up
                top -= 55 * min(1, (ph - 170) / 10)
            if i == 9 and 290 <= ph < 420:                      # student 10 leans to student 11
                x += 200 * (min(ph - 290, 40) / 40 if ph < 380 else (420 - ph) / 40)
            if i == 7 and 440 <= ph < 500:                      # student 8 fidgets a lot
                x += 9 if self.t % 2 else -9
            boxes.append([x - 30, top, x + 30, bot])
            if i == 4 and 60 <= ph < 140:                       # student 5 holds a phone
                phones.append([x + 18, cy + 5, x + 34, cy + 33])
        self.t += 1
        return self._draw(boxes, phones), boxes, phones

    def _draw(self, boxes, phones):
        f = np.full((H, W, 3), (38, 34, 30), "uint8")
        for x1, y1, x2, y2 in boxes:
            cx = int((x1 + x2) / 2)
            cv2.rectangle(f, (int(x1) - 22, int(y2) - 40), (int(x2) + 22, int(y2) + 6), (70, 90, 110), -1)  # desk
            cv2.rectangle(f, (int(x1) + 4, int(y1) + 34), (int(x2) - 4, int(y2) - 40), (150, 110, 80), -1)  # body
            cv2.circle(f, (cx, int(y1) + 18), 16, (170, 200, 230), -1)                                       # head
        for x1, y1, x2, y2 in phones:
            cv2.rectangle(f, (int(x1), int(y1)), (int(x2), int(y2)), (230, 230, 230), -1)
        cv2.putText(f, "SIMULATED CLASSROOM (demo)", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (200, 200, 200), 1)
        return f
