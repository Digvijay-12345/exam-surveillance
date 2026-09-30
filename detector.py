import os, time, math, cv2, numpy as np
from collections import deque
import db

PERSON, PHONE = 0, 67  # COCO class ids
GREEN, ORANGE, RED = (0, 200, 0), (0, 165, 255), (0, 0, 255)
os.makedirs("evidence", exist_ok=True)


def enhance_low_light(frame, thresh=90):
    """Brighten dark frames (gamma + CLAHE). Returns (frame, brightness). No-op if already bright."""
    small = cv2.resize(frame, (80, 60))
    mean = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
    if mean >= thresh:
        return frame, mean
    gamma = 1.0 + (thresh - mean) / thresh * 1.2
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], "uint8")
    frame = cv2.LUT(frame, lut)
    l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), mean


class Tracker:
    """Simple centroid tracker: gives each student a stable ID."""
    def __init__(self, max_dist=80, max_missing=30):
        self.tracks, self.next_id = {}, 1
        self.max_dist, self.max_missing = max_dist, max_missing

    def update(self, boxes):
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        unmatched = list(range(len(boxes)))
        for t in self.tracks.values():
            t["missing"] += 1
        for tid, t in self.tracks.items():
            best, bd = None, self.max_dist
            for i in unmatched:
                d = math.dist(t["c"], centers[i])
                if d < bd:
                    best, bd = i, d
            if best is not None:
                unmatched.remove(best)
                t.update(c=centers[best], box=boxes[best], missing=0)
                t["hist"].append(centers[best])
                t["heights"].append(boxes[best][3] - boxes[best][1])
        for i in unmatched:
            self.tracks[self.next_id] = dict(c=centers[i], box=boxes[i], missing=0,
                                             hist=deque([centers[i]], maxlen=15),
                                             heights=deque([boxes[i][3] - boxes[i][1]], maxlen=60))
            self.next_id += 1
        self.tracks = {k: v for k, v in self.tracks.items() if v["missing"] <= self.max_missing}
        return {k: v for k, v in self.tracks.items() if v["missing"] == 0}


class ExamMonitor:
    def __init__(self, conf=0.35, move_thresh=1.5, stand_ratio=1.35, cooldown=10,
                 imgsz=480, skip=2, low_light=True):
        self.model = None  # YOLOv10 loads lazily on first real-video frame
        self.conf, self.move_thresh, self.stand_ratio, self.cooldown = conf, move_thresh, stand_ratio, cooldown
        self.imgsz, self.skip, self.low_light = imgsz, skip, low_light
        self.brightness = 255.0
        self.reset()

    def reset(self):
        self.tracker, self.last_alert, self.frame_i = Tracker(), {}, 0
        self.draw, self.pending = [], []

    # ---------- real video ----------
    def process(self, frame):
        if self.low_light:
            frame, self.brightness = enhance_low_light(frame)
        self.frame_i += 1
        if (self.frame_i - 1) % max(self.skip, 1) != 0:   # frame skipping for weak CPUs
            self._render(frame)
            return frame, []
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO("yolov10n.pt")  # auto-downloads on first run
        conf = max(0.2, self.conf - (0.05 if self.brightness < 90 else 0))
        res = self.model.predict(frame, conf=conf, imgsz=self.imgsz,
                                 classes=[PERSON, PHONE], verbose=False)[0]
        persons, phones = [], []
        for b in res.boxes:
            (persons if int(b.cls[0]) == PERSON else phones).append(b.xyxy[0].tolist())
        return self.analyze(frame, persons, phones)

    # ---------- rule engine (also used by the simulated demo) ----------
    def _alert(self, sid, etype, details):
        key = (sid, etype)
        if time.time() - self.last_alert.get(key, 0) < self.cooldown:
            return
        self.last_alert[key] = time.time()
        self.pending.append((sid, etype, details))

    def analyze(self, frame, persons, phones):
        self.draw, self.pending = [], []
        self.tracker.max_dist = 0.09 * frame.shape[1]
        tracked = self.tracker.update(persons)

        for sid, t in tracked.items():
            x1, y1, x2, y2 = map(int, t["box"])
            h, color = y2 - y1, GREEN
            pts = list(t["hist"])
            if len(pts) >= 10:  # 1) excessive movement (normalised by body height and frame skip)
                path = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
                ratio = path / max(h, 1) / max(self.skip, 1)
                if ratio > self.move_thresh:
                    color = ORANGE
                    self._alert(sid, "EXCESSIVE_MOVEMENT", f"path/height={ratio:.2f}")
            if len(t["heights"]) >= 30:  # 2) standing up
                base = sorted(list(t["heights"])[:30])[15]
                if h > base * self.stand_ratio:
                    color = ORANGE
                    self._alert(sid, "STANDING_UP", f"h={h}, baseline={base:.0f}")
            self.draw.append((x1, y1, x2, y2, color, f"Student {sid}"))

        ids = list(tracked)  # 3) students too close
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = tracked[ids[i]], tracked[ids[j]]
                w = ((a["box"][2] - a["box"][0]) + (b["box"][2] - b["box"][0])) / 2
                if math.dist(a["c"], b["c"]) < 0.6 * w:
                    self._alert(ids[i], "STUDENTS_TOO_CLOSE", f"with student {ids[j]}")

        for p in phones:  # 4) phone visible
            x1, y1, x2, y2 = map(int, p)
            self.draw.append((x1, y1, x2, y2, RED, "PHONE"))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            near = min(tracked.items(), key=lambda kv: math.dist(kv[1]["c"], (cx, cy)), default=(0, None))[0]
            self._alert(near, "PHONE_DETECTED", "cell phone in frame")

        self._render(frame)  # draw first so evidence snapshots contain the boxes
        events = []
        for sid, etype, d in self.pending:
            path = f"evidence/{time.strftime('%Y%m%d_%H%M%S')}_{etype}_{sid}.jpg"
            cv2.imwrite(path, frame)
            db.log_event(sid, etype, d, path)
            events.append((sid, etype, d))
        return frame, events

    def _render(self, frame):
        for x1, y1, x2, y2, c, label in self.draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        pos = (10, frame.shape[0] - 10)
        cv2.putText(frame, ts, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, ts, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
