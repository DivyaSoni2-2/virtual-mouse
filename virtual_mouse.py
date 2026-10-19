"""
Virtual Mouse - control your cursor with hand gestures (webcam + MediaPipe).

Gesture            Action
-----------------  ----------------------------------------------------------
FIST               Stop everything (deactivate, cursor stops)
PALM (open hand)   Activate tracking
INDEX finger up    Move the cursor with your fingertip
PINCH              Right-click at the current cursor position
INDEX + MIDDLE     Scroll: raise hand above start point = up, lower = down
Hand not visible   Deactivate (cursor stops responding)

Install:  pip install opencv-python mediapipe pyautogui
Quit:     press Q, Esc or Enter while the preview window is focused
"""

import platform
import threading
import time
from enum import Enum

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# =============================== CONFIG ===============================
CAMERA_INDEX = 0
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 640, 480, 30
SHOW_PREVIEW = True

# --- Cursor movement ---
# Screen-widths the cursor travels per frame-width of finger travel.
# 1.8 means ~55% of the camera view covers the whole screen. Raise = faster.
CURSOR_GAIN = 1.8
# One-Euro filter: lower MIN_CUTOFF = steadier when slow; higher BETA = less lag when fast.
FILTER_MIN_CUTOFF = 1.5
FILTER_BETA = 0.02

# --- Gesture detection (distances are relative to hand size) ---
FINGER_EXTEND_RATIO = 1.08   # tip-to-wrist vs knuckle-to-wrist, for "finger is up"
PINCH_ON = 0.24              # thumb-index distance that starts a pinch
PINCH_OFF = 0.38             # distance that ends a pinch (hysteresis avoids flicker)
PINCH_FREEZE = 0.45          # cursor freezes as fingers approach, so the click doesn't drift
PINCH_MIN_REACH = 1.10       # index tip must be extended outward (separates pinch from fist)
STABLE_FRAMES = 2            # frames a gesture must persist before it counts
LOST_GRACE_FRAMES = 8        # frames without a hand before deactivating

# --- Scrolling (joystick style: further from start point = faster) ---
IS_WINDOWS = platform.system() == "Windows"
SCROLL_DEADZONE = 0.04                         # fraction of frame height with no scroll
SCROLL_SPEED = 6000.0 if IS_WINDOWS else 50.0  # wheel units / second per unit of offset
SCROLL_INVERT = False                          # True flips the direction
# ======================================================================

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False  # cursor is clamped to the screen instead; quit with Q/Esc


class Gesture(Enum):
    NONE = "NONE"
    FIST = "FIST"
    PALM = "PALM"
    INDEX = "INDEX"
    PINCH = "PINCH"
    SCROLL = "SCROLL"


class OneEuroFilter:
    """Adaptive low-pass filter: smooth when moving slowly, responsive when fast."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t = self.x = None
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.t, self.x, self.dx = t, x, 0.0
            return x
        dt = max(t - self.t, 1e-3)
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * ((x - self.x) / dt) + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


class Camera:
    """Grabs frames on a background thread so processing always sees the newest frame."""

    def __init__(self, index, width, height, fps):
        backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame, self._seq, self._stop = None, 0, False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1

    def read(self):
        with self._lock:
            return self._seq, self._frame

    def release(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self.cap.release()


class HandAnalyzer:
    """Turns 21 hand landmarks into a gesture. Size-independent and rotation-tolerant."""

    TIPS = (8, 12, 16, 20)
    PIPS = (6, 10, 14, 18)

    def __init__(self):
        self.pinching = False

    def reset(self):
        self.pinching = False

    def analyze(self, pts):
        """pts: (21, 2) landmark array in pixels. Returns (Gesture, pinch_ratio)."""
        def d(a, b):
            return float(np.linalg.norm(pts[a] - pts[b]))

        size = max(d(0, 9), 1e-6)  # wrist -> middle knuckle
        index, middle, ring, pinky = (
            d(t, 0) > d(p, 0) * FINGER_EXTEND_RATIO for t, p in zip(self.TIPS, self.PIPS)
        )

        pinch_ratio = d(4, 8) / size
        reach = d(8, 0) / size
        limit = PINCH_OFF if self.pinching else PINCH_ON
        self.pinching = pinch_ratio < limit and reach > PINCH_MIN_REACH

        if self.pinching:
            return Gesture.PINCH, pinch_ratio
        if index and middle and ring and pinky:
            return Gesture.PALM, pinch_ratio
        if not (index or middle or ring or pinky):
            return Gesture.FIST, pinch_ratio
        if index and middle and not ring and not pinky:
            return Gesture.SCROLL, pinch_ratio
        if index and not (middle or ring or pinky):
            return Gesture.INDEX, pinch_ratio
        return Gesture.NONE, pinch_ratio


class GestureStabilizer:
    """Only accepts a gesture after it has been seen for N consecutive frames."""

    def __init__(self, frames):
        self.frames = frames
        self.reset()

    def reset(self):
        self.candidate, self.count, self.stable = Gesture.NONE, 0, Gesture.NONE

    def update(self, gesture):
        if gesture == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = gesture, 1
        if self.count >= self.frames:
            self.stable = gesture
        return self.stable


class VirtualMouse:
    """State machine that turns stable gestures into mouse actions."""

    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.fx = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
        self.fy = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
        self.active = False
        self.prev_gesture = Gesture.NONE
        self.right_clicked = False
        self.last_pos = None
        self._reset_motion()

    # ---- state helpers ----
    def _reset_motion(self):
        self.cursor_anchor = None   # cursor position when tracking began
        self.finger_anchor = None   # filtered fingertip position at that moment
        self.fx.reset()
        self.fy.reset()
        self.scroll_anchor = None
        self.scroll_y = None
        self.scroll_acc = 0.0
        self.scroll_t = None

    def deactivate(self):
        self.active = False
        self.prev_gesture = Gesture.NONE
        self.right_clicked = False
        self._reset_motion()

    # ---- main entry ----
    def update(self, gesture, norm, pinch_ratio, now):
        """norm: (21, 2) landmarks normalised to 0..1."""
        if gesture != self.prev_gesture:
            self._reset_motion()  # re-anchor on every gesture change => no cursor jumps
            self.prev_gesture = gesture

        if gesture == Gesture.FIST:
            self.deactivate()
            return
        if gesture == Gesture.PALM:
            self.active = True
            return
        if not self.active:
            return

        if gesture == Gesture.INDEX:
            self._move(norm, pinch_ratio, now)
        elif gesture == Gesture.PINCH:
            self._right_click()
        elif gesture == Gesture.SCROLL:
            self._scroll(norm, now)

        if gesture != Gesture.PINCH:
            self.right_clicked = False

    # ---- actions ----
    def _move(self, norm, pinch_ratio, now):
        if pinch_ratio < PINCH_FREEZE:  # fingers closing for a click: hold still
            self._reset_motion()
            return

        tip = norm[8]
        fx = self.fx(tip[0] * self.screen_w * CURSOR_GAIN, now)
        fy = self.fy(tip[1] * self.screen_h * CURSOR_GAIN, now)

        if self.cursor_anchor is None:
            pos = pyautogui.position()
            self.cursor_anchor = (float(pos[0]), float(pos[1]))
            self.finger_anchor = (fx, fy)
            return

        tx = self.cursor_anchor[0] + (fx - self.finger_anchor[0])
        ty = self.cursor_anchor[1] + (fy - self.finger_anchor[1])
        cx = min(max(tx, 0), self.screen_w - 1)
        cy = min(max(ty, 0), self.screen_h - 1)
        if cx != tx or cy != ty:  # hit a screen edge: re-anchor so there's no dead travel
            self.cursor_anchor = (cx, cy)
            self.finger_anchor = (fx, fy)

        pos = (int(round(cx)), int(round(cy)))
        if pos != self.last_pos:
            pyautogui.moveTo(pos[0], pos[1], _pause=False)
            self.last_pos = pos

    def _right_click(self):
        if not self.right_clicked:  # one click per pinch
            pyautogui.rightClick(_pause=False)
            self.right_clicked = True

    def _scroll(self, norm, now):
        y = (norm[8][1] + norm[12][1]) / 2.0
        if self.scroll_anchor is None:
            self.scroll_anchor = self.scroll_y = y
            self.scroll_t = now
            return

        self.scroll_y = 0.4 * y + 0.6 * self.scroll_y  # light smoothing
        dt = min(now - self.scroll_t, 0.1)
        self.scroll_t = now

        offset = self.scroll_anchor - self.scroll_y  # hand raised => positive => scroll up
        if SCROLL_INVERT:
            offset = -offset
        if abs(offset) < SCROLL_DEADZONE:
            self.scroll_acc = 0.0
            return

        speed = (abs(offset) - SCROLL_DEADZONE) * SCROLL_SPEED
        self.scroll_acc += speed * dt * (1 if offset > 0 else -1)
        clicks = int(self.scroll_acc)
        if clicks:
            pyautogui.scroll(clicks, _pause=False)
            self.scroll_acc -= clicks


def draw_hud(frame, mouse, gesture, fps):
    color = (0, 220, 0) if mouse.active else (0, 0, 255)
    state = "ACTIVE" if mouse.active else "INACTIVE"
    cv2.putText(frame, f"{state} | {gesture.value} | {fps:.0f} fps", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if not mouse.active:
        cv2.putText(frame, "Show an open PALM to activate", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def main():
    cam = Camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    analyzer = HandAnalyzer()
    stabilizer = GestureStabilizer(STABLE_FRAMES)
    mouse = VirtualMouse()
    drawing = mp.solutions.drawing_utils
    hands_mod = mp.solutions.hands

    gesture, lost, last_seq = Gesture.NONE, 0, -1
    fps, t_prev = 0.0, time.perf_counter()

    try:
        with hands_mod.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0,
                             min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
            while True:
                seq, frame = cam.read()
                if frame is None or seq == last_seq:
                    time.sleep(0.002)
                    continue
                last_seq = seq

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
                t_prev = now

                frame = cv2.flip(frame, 1)  # mirror so movement feels natural
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    lost = 0
                    lm = result.multi_hand_landmarks[0]
                    norm = np.array([(p.x, p.y) for p in lm.landmark], dtype=np.float32)
                    pts = norm * np.array([w, h], dtype=np.float32)

                    raw, pinch_ratio = analyzer.analyze(pts)
                    gesture = stabilizer.update(raw)
                    mouse.update(gesture, norm, pinch_ratio, now)

                    if SHOW_PREVIEW:
                        drawing.draw_landmarks(frame, lm, hands_mod.HAND_CONNECTIONS)
                else:
                    lost += 1
                    if lost >= LOST_GRACE_FRAMES:
                        mouse.deactivate()
                        analyzer.reset()
                        stabilizer.reset()
                        gesture = Gesture.NONE

                if SHOW_PREVIEW:
                    draw_hud(frame, mouse, gesture, fps)
                    cv2.imshow("Virtual Mouse", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27, 13):
                        break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
