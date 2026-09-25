"""Aplica los efectos de vfx_spec.json al video, guiados por la pose de MediaPipe.

Uso: python3 render_vfx.py <video.mp4> <pose_landmarker.task> [salida.mp4]
"""
import json
import subprocess
import sys

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

VIDEO, MODEL = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "output/bailecito_vfx.mp4"
SPEC = json.load(open("vfx_spec.json"))
FADE = 0.2  # segundos de crossfade entre segmentos

HANDS = [15, 16, 19, 20]
HEAD = [0, 7, 8]
TORSO = [11, 12, 24, 23]
rng = np.random.default_rng(7)


def hex_bgr(h):
    h = h.lstrip("#")
    return np.array([int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)], np.float32)


def hue_shift(bgr, deg):
    px = np.uint8([[np.clip(bgr, 0, 255)]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).astype(int)
    hsv[..., 0] = (hsv[..., 0] + deg // 2) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


def segment_at(t):
    for i, s in enumerate(SPEC):
        if s["inicio_s"] <= t < s["fin_s"]:
            return i
    return len(SPEC) - 1


# ---------- 1) Pose + segmentación (una sola pasada) ----------
opt = vision.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=MODEL, delegate=mpp.BaseOptions.Delegate.CPU),
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.PoseLandmarker.create_from_options(opt)
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
frames, poses = [], []
i = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    H, W = f.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    r = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
    if r.pose_landmarks:
        poses.append(np.array([[l.x * W, l.y * H] for l in r.pose_landmarks[0]], np.float32))
    else:
        poses.append(poses[-1] if poses else np.zeros((33, 2), np.float32))
    frames.append(f)
    i += 1

# Suavizado temporal de landmarks (EMA) para que los efectos no tiemblen.
for k in range(1, len(poses)):
    poses[k] = 0.6 * poses[k] + 0.4 * poses[k - 1]


# ---------- 2) Máscaras por zona ----------
# Silueta aproximada a partir del esqueleto: la máscara de segmentación de
# mediapipe 1.0.1 falla al copiarse a numpy (Check failed en CopyToBuffer).
BONES = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), (26, 28), (15, 19), (16, 20)]


def skeleton_mask(p, shape):
    m = np.zeros(shape, np.float32)
    w = np.linalg.norm(p[11] - p[12]) + 1
    cv2.fillPoly(m, [p[TORSO].astype(np.int32)], 1)
    for a, b in BONES:
        thick = 0.30 if b in (25, 26, 27, 28) or a in (23, 24, 25, 26) else 0.20
        cv2.line(m, tuple(int(v) for v in p[a]), tuple(int(v) for v in p[b]), 1, max(2, int(w * thick)), cv2.LINE_AA)
    cv2.circle(m, tuple(int(v) for v in p[0]), int(w * 0.33), 1, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), 3)


def zone_mask(zona, p, sil):
    m = np.zeros(sil.shape, np.float32)
    scale = np.linalg.norm(p[11] - p[12]) + 1
    if zona == "silueta_completa":
        return sil
    if zona == "fondo":
        return 1 - sil
    if zona == "manos":
        for j in HANDS:
            cv2.circle(m, tuple(int(v) for v in p[j]), int(scale * 0.35), 1, -1)
    elif zona == "cabeza":
        cv2.circle(m, tuple(int(v) for v in p[0]), int(scale * 0.55), 1, -1)
    elif zona == "torso":
        cv2.fillPoly(m, [p[TORSO].astype(np.int32)], 1)
        m = cv2.dilate(m, np.ones((25, 25), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), 9)


# ---------- 3) Efectos (cada uno devuelve una capa aditiva float32 BGR) ----------
trail_hist = []  # historial de posiciones de manos
particles = []   # [x, y, vx, vy, vida, b, g, r]


def fx_glow(t, p, sil, zmask, col, I):
    pulse = 0.55 + 0.45 * np.sin(2 * np.pi * 1.6 * t)
    edge = np.clip(cv2.dilate(zmask, np.ones((15, 15), np.uint8)) - cv2.erode(zmask, np.ones((5, 5), np.uint8)), 0, 1)
    halo = cv2.GaussianBlur(edge, (0, 0), 6 + 14 * I) * (1.2 + 2.5 * I) * pulse
    return halo[..., None] * col[None, None, :] / 255.0 * 255


def fx_trail(t, p, sil, zmask, col, I):
    layer = np.zeros((*sil.shape, 3), np.float32)
    n = int(6 + 22 * I)
    hist = trail_hist[-n:]
    for a in range(1, len(hist)):
        fade = a / len(hist)
        c = hue_shift(col, int(40 * (1 - fade)))
        for j in (15, 16):
            p0 = tuple(int(v) for v in hist[a - 1][j])
            p1 = tuple(int(v) for v in hist[a][j])
            cv2.line(layer, p0, p1, (c * fade).tolist(), max(1, int((3 + 10 * I) * fade)), cv2.LINE_AA)
    glow = cv2.GaussianBlur(layer, (0, 0), 8)
    return (layer + glow * 1.8) * (0.6 + 0.6 * I)


def fx_particles(t, p, sil, zmask, col, I, emit_pts):
    vel = trail_hist[-1] - trail_hist[-2] if len(trail_hist) > 1 else np.zeros_like(p)
    for j in emit_pts:
        speed = np.linalg.norm(vel[j])
        for _ in range(int(2 + I * 6 + speed * 0.4 * I)):
            c = hue_shift(col, int(rng.integers(-50, 50)))
            v = vel[j] * 0.3 + rng.normal(0, 1.5 + 3 * I, 2)
            particles.append([*p[j], *v, 1.0, *c])
    return None  # se dibujan en draw_particles (persisten entre segmentos)


def draw_particles(shape):
    layer = np.zeros((*shape, 3), np.float32)
    alive = []
    for q in particles:
        q[0] += q[2]; q[1] += q[3]; q[3] += 0.15; q[4] -= 0.035
        if q[4] > 0:
            cv2.circle(layer, (int(q[0]), int(q[1])), max(1, int(3 * q[4])), (np.array(q[5:8]) * q[4]).tolist(), -1, cv2.LINE_AA)
            alive.append(q)
    particles[:] = alive[-1500:]
    return layer + cv2.GaussianBlur(layer, (0, 0), 5) * 2


def fx_wave(t, frame, zmask, I):
    H, W = zmask.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    amp = (4 + 18 * I) * zmask
    dx = amp * np.sin(yy / 18.0 + t * 9)
    dy = amp * 0.5 * np.cos(xx / 22.0 + t * 7)
    warped = cv2.remap(frame, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def fx_fractal(t, p, sil, zmask, col, I):
    layer = np.zeros((*sil.shape, 3), np.float32)
    center = p[TORSO].mean(0)
    # Ecos concéntricos de la silueta (auto-similitud a distintas escalas)
    m8 = (sil > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for k in range(1, int(3 + 5 * I)):
        s = 1 + 0.12 * k * (0.8 + 0.2 * np.sin(t * 4))
        ang = t * 40 * (1 if k % 2 else -1)
        M = cv2.getRotationMatrix2D(tuple(center.tolist()), ang * 0.15 * k, s)
        c = hue_shift(col, 35 * k)
        for cnt in cnts:
            pts = cv2.transform(cnt.astype(np.float32), M).astype(np.int32)
            cv2.polylines(layer, [pts], True, (c * (1 - k * 0.08)).tolist(), 2, cv2.LINE_AA)
    # Polígonos estrellados anidados que rotan
    R = np.linalg.norm(p[11] - p[23]) * 1.2
    for k in range(6):
        r = R * (0.7 ** k) * (1 + 0.5 * I)
        n = 6
        a0 = t * (1.5 + k) * (-1) ** k
        pts = np.array([[center[0] + r * np.cos(a0 + 2 * np.pi * v / n), center[1] + r * np.sin(a0 + 2 * np.pi * v / n)] for v in range(n)], np.int32)
        cv2.polylines(layer, [pts], True, hue_shift(col, -30 * k).tolist(), 2, cv2.LINE_AA)
    return (layer + cv2.GaussianBlur(layer, (0, 0), 7) * 2) * (0.5 + 0.7 * I)


def render_effect(seg, t, frame, p, sil):
    s = SPEC[seg]
    col, I, efecto = hex_bgr(s["color_base"]), s["intensidad"], s["efecto"]
    zmask = zone_mask(s["zona"], p, sil)
    base, add = frame, np.zeros_like(frame)
    if efecto == "distorsion_onda":
        base = fx_wave(t, frame, zmask, I)
        add = fx_glow(t, p, sil, zmask, col, I * 0.6)
    elif efecto == "glow_pulsante":
        add = fx_glow(t, p, sil, zmask, col, I)
    elif efecto == "trail_neon":
        add = fx_trail(t, p, sil, zmask, col, I)
    elif efecto == "particulas":
        pts = HANDS if s["zona"] == "manos" else [0, 11, 12, 15, 16, 23, 24, 27, 28]
        fx_particles(t, p, sil, zmask, col, I, pts)
        add = fx_glow(t, p, sil, sil, col, I * 0.4)
    elif efecto == "geometria_fractal":
        add = fx_fractal(t, p, sil, zmask, col, I)
    return base, add, I


# ---------- 4) Render ----------
H, W = frames[0].shape[:2]
ff = imageio_ffmpeg.get_ffmpeg_exe()
enc = subprocess.Popen(
    [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
     "-i", VIDEO, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
     "-c:a", "aac", "-shortest", "-movflags", "+faststart", OUT],
    stdin=subprocess.PIPE,
)
for n, f in enumerate(frames):
    t = n / fps
    p = poses[n]
    sil = skeleton_mask(p, (H, W))
    trail_hist.append(p.copy())
    del trail_hist[:-40]
    frame = f.astype(np.float32)

    seg = segment_at(t)
    base, add, I = render_effect(seg, t, frame, p, sil)
    # Crossfade con el segmento anterior al inicio de cada tramo
    since = t - SPEC[seg]["inicio_s"]
    if seg > 0 and since < FADE:
        a = since / FADE
        pb, pa, pI = render_effect(seg - 1, t, frame, p, sil)
        base, add, I = base * a + pb * (1 - a), add * a + pa * (1 - a), I * a + pI * (1 - a)

    # Gradación psicodélica: saturación proporcional a la intensidad del movimiento
    hsv = cv2.cvtColor(np.clip(base, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.25 + 0.5 * I), 0, 255)
    hsv[..., 0] = (hsv[..., 0] + 6 * I * np.sin(t * 2)) % 180
    graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    out = graded + add + draw_particles((H, W))
    enc.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
enc.stdin.close()
enc.wait()
print("listo:", OUT)
