import tempfile, threading, time, cv2, pandas as pd, streamlit as st
from detector import ExamMonitor
from sim_classroom import ClassroomSim
import db

st.set_page_config(page_title="Anti-Cheating Classroom Monitor", layout="wide")
st.title("🎓 Anti-Cheating Classroom Monitor")
st.caption("YOLOv10 + OpenCV · real-time flags for invigilators · timestamped evidence")

SIM = "Classroom demo (simulated, no video needed)"
PROFILES = {
    "Low power (old laptop / CPU)": dict(imgsz=320, skip=3, width=640),
    "Balanced": dict(imgsz=480, skip=2, width=640),
    "Accurate (better PC / GPU)": dict(imgsz=640, skip=1, width=960),
}


class Stream:
    """Reads frames in a background thread for live sources so lag never builds up."""
    def __init__(self, src, live):
        self.live, self.idx, self.frame, self.ok, self._stop = live, 0, None, True, False
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if src == 0 else cv2.VideoCapture(src)
        if src == 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.opened = self.cap.isOpened()
        if live and self.opened:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False; break
            self.frame, self.idx = f, self.idx + 1

    def read(self):
        if self.live:
            return self.ok, self.frame, self.idx
        ok, f = self.cap.read()
        self.idx += 1
        return ok, f, self.idx

    def release(self):
        self._stop = True; time.sleep(0.1); self.cap.release()


@st.cache_resource
def get_monitor():
    return ExamMonitor()


with st.sidebar:
    profile = st.selectbox("Performance profile", list(PROFILES), index=1)
    low_light = st.checkbox("Low-light enhancement", value=True)
    mode = st.radio("Input", [SIM, "Webcam", "Upload video", "RTSP / CCTV URL"])
    src = 0
    if mode == "Upload video":
        f = st.file_uploader("Video", type=["mp4", "avi", "mov"])
        if f:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp.write(f.read()); tmp.close(); src = tmp.name
        else:
            src = None
    elif mode == "RTSP / CCTV URL":
        src = st.text_input("URL (rtsp:// or http://)", "http://")
    conf = st.slider("Detection confidence", 0.2, 0.9, 0.35)
    move = st.slider("Movement threshold", 0.5, 3.0, 1.5)
    c1, c2 = st.columns(2)
    start = c1.button("▶ Start", type="primary")
    if c2.button("🗑 Clear log"):
        db.clear_events()
    if mode == SIM:
        st.info("Scripted: Student 5 phone · 3 stands · 10 leans to 11 · 8 fidgets. Loops every ~30 s.")

left, right = st.columns([3, 2])
video_ph, stat_ph = left.empty(), left.empty()
banner, table_ph = right.empty(), right.empty()


def render_alerts():
    rows = db.fetch_events(50)
    table_ph.dataframe(pd.DataFrame(rows, columns=["Time", "Student", "Event", "Details", "Evidence"]),
                       use_container_width=True, height=480)


def show_events(events):
    for sid, etype, d in events:
        banner.error(f"🚨 Student {sid}: {etype.replace('_', ' ')} ({d})")


render_alerts()

if start and src is not None:
    if mode == SIM:
        mon, sim = ExamMonitor(conf=conf, move_thresh=move, skip=1, low_light=False), ClassroomSim()
        while True:
            frame, persons, phones = sim.step()
            frame, events = mon.analyze(frame, persons, phones)
            video_ph.image(frame, channels="BGR"); show_events(events)
            if events or sim.t % 20 == 0:
                render_alerts()
            time.sleep(0.05)
    else:
        p = PROFILES[profile]
        mon = get_monitor(); mon.reset()
        mon.conf, mon.move_thresh, mon.low_light = conf, move, low_light
        mon.imgsz, mon.skip = p["imgsz"], p["skip"]
        live = src == 0 or (isinstance(src, str) and src.startswith(("rtsp", "http")))
        stream = Stream(src, live)
        if not stream.opened:
            st.error("Could not open the video source. Close other apps using the camera or check the URL.")
        else:
            last_idx, n, t0 = -1, 0, time.time()
            try:
                while True:
                    ok, frame, idx = stream.read()
                    if not ok:
                        break
                    if frame is None or idx == last_idx:
                        time.sleep(0.005); continue
                    last_idx = idx
                    if frame.shape[1] != p["width"]:
                        frame = cv2.resize(frame, (p["width"], int(frame.shape[0] * p["width"] / frame.shape[1])))
                    frame, events = mon.process(frame)
                    video_ph.image(frame, channels="BGR"); show_events(events)
                    n += 1
                    if events or n % 30 == 0:
                        render_alerts()
                    if n % 10 == 0:
                        fps = 10 / (time.time() - t0); t0 = time.time()
                        lit = "dark → enhanced" if (low_light and mon.brightness < 90) else "normal"
                        stat_ph.caption(f"⚡ {fps:.1f} FPS · light: {lit} ({mon.brightness:.0f}/255) · {profile}")
            finally:
                stream.release()
            if n == 0:
                st.error("No frames could be read from this video. Its codec may be unsupported; re-encode it with ffmpeg.")
            else:
                st.info(f"Stream ended. Processed {n} frames.")
