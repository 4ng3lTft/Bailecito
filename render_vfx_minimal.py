"""Renderer minimalista para vfx_spec_minimal.json.

Un solo sistema visual continuo: cada efecto tiene un peso que entra y sale con la
curva de easing del segmento, los colores se interpolan en HSL y la modulación
(senoidal / Perlin) es siempre una función continua del tiempo.

Uso: python3 render_vfx_minimal.py <video.mp4> <pose_landmarker.task> [salida.mp4]
"""
import colorsys
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
OUT = sys.argv[3] if len(sys.argv) > 3 else "output/bailecito_minimal.mp4"
SPEC = json.load(open("vfx_spec_minimal.json"))
T_FADE = 0.4  # ventana de transición centrada en cada frontera de segmento
EFFECTS = ["glow_pulsante", "trail_neon", "particulas", "distorsion_onda"]

WRISTS = [15, 16]
TORSO = [11, 12, 24, 23]
BONES = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (15, 19), (16, 20)]


# ---------- Curvas y modulación ----------
def ease(x, curva):
    x = min(max(x, 0.0), 1.0)
    if curva == "ease_in_out_cubic":
        return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
    if curva == "senoidal":
        return 0.5 - 0.5 * np.cos(np.pi * x)
    return x


_rng = np.random.default_rng(3)
_lattice = _rng.random(512)


def noise1d(x):
    """Value noise 1D con interpolación suave (aprox. Perlin), rango [0, 1]."""
    i = int(np.floor(x)); f = x - i
    u = f * f * (3 - 2 * f)
    return _lattice[i % 512] * (1 - u) + _lattice[(i + 1) % 512] * u


def modulate(kind, t, seed=0.0):
    if kind == "onda_senoidal":
        return 0.5 + 0.5 * np.sin(2 * np.pi * 0.45 * t + seed)
    if kind == "perlin_noise":
        return noise1d(t * 1.3 + seed * 17)
    return 1.0


# ---------- Color en HSL ----------
def hex_hls(h):
    h = h.lstrip("#")
    return colorsys.rgb_to_hls(*(int(h[k:k + 2], 16) / 255 for k in (0, 2, 4)))


def hls_mix(items):
    """Promedio ponderado en HLS; el tono se promedia sobre el círculo."""
    wsum = sum(w for w, _ in items) or 1
    hx = sum(w * np.cos(2 * np.pi * c[0]) for w, c in items)
    hy = sum(w * np.sin(2 * np.pi * c[0]) for w, c in items)
    h = (np.arctan2(hy, hx) / (2 * np.pi)) % 1
    l = sum(w * c[1] for w, c in items) / wsum
    s = sum(w * c[2] for w, c in items) / wsum
    return h, l, s


def hls_bgr(c):
    r, g, b = colorsys.hls_to_rgb(*c)
    return np.array([b, g, r], np.float32)


def hls_lerp(a, b, x):
    return hls_mix([(1 - x, a), (x, b)])


# ---------- Pesos por segmento (suman 1 en cada frontera) ----------
def segment_weights(t):
    out = []
    for k, s in enumerate(SPEC):
        a, b = s["inicio_s"], s["fin_s"]
        w_in = 1.0 if k == 0 else ease((t - (a - T_FADE / 2)) / T_FADE, s["curva_easing"])
        nxt = SPEC[k + 1]["curva_easing"] if k + 1 < len(SPEC) else s["curva_easing"]
        w_out = 1.0 if k == len(SPEC) - 1 else 1 - ease((t - (b - T_FADE / 2)) / T_FADE, nxt)
        w = w_in * w_out
        if w > 1e-3:
            out.append((w, s))
    return out


def frame_state(t):
    """Estado continuo del sistema en el instante t."""
    segs = segment_weights(t)
    st = {e: 0.0 for e in EFFECTS}
    zones = {e: {} for e in EFFECTS}
    for w, s in segs:
        m = modulate(s["modulacion"], t, seed=len(s["zona"]))
        gain = s["intensidad"] * (1.0 if s["protagonismo"] == "focal" else 0.7) * (0.75 + 0.25 * m)
        st[s["efecto"]] += w * gain
        zones[s["efecto"]][s["zona"]] = zones[s["efecto"]].get(s["zona"], 0) + w
    base = hls_mix([(w, hex_hls(s["color_base"])) for w, s in segs])
    sec = hls_mix([(w, hex_hls(s["color_secundario"])) for w, s in segs])
    # Aura de fondo permanente: el sistema nunca "se apaga", solo respira.
    st["glow_pulsante"] = max(st["glow_pulsante"], 0.06 + 0.04 * np.sin(2 * np.pi * 0.3 * t))
    if not zones["glow_pulsante"]:
        zones["glow_pulsante"] = {"silueta_completa": 1.0}
    return st, zones, base, sec


# ---------- Pasada 1: solo landmarks (no guardamos frames en RAM) ----------
opt = vision.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=MODEL, delegate=mpp.BaseOptions.Delegate.CPU),
    running_mode=vision.RunningMode.VIDEO,
)
lmk = vision.PoseLandmarker.create_from_options(opt)
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(3)), int(cap.get(4))
poses, i = [], 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    rgb = np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    r = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
    if r.pose_landmarks:
        poses.append(np.array([[l.x * W, l.y * H] for l in r.pose_landmarks[0]], np.float32))
    else:
        poses.append(poses[-1] if poses else np.full((33, 2), [W / 2, H / 2], np.float32))
    i += 1
cap.release()
poses = np.array(poses)
# Suavizado bidireccional (EMA ida y vuelta): sin retraso de fase.
for k in range(1, len(poses)):
    poses[k] = 0.55 * poses[k] + 0.45 * poses[k - 1]
for k in range(len(poses) - 2, -1, -1):
    poses[k] = 0.55 * poses[k] + 0.45 * poses[k + 1]


# ---------- Máscaras ----------
def pt(v):
    return tuple(int(x) for x in v)


def silhouette(p):
    m = np.zeros((H, W), np.float32)
    w = np.linalg.norm(p[11] - p[12]) + 1
    cv2.fillPoly(m, [p[TORSO].astype(np.int32)], 1)
    for a, b in BONES:
        thick = 0.30 if a in (23, 24, 25, 26) else 0.20
        cv2.line(m, pt(p[a]), pt(p[b]), 1, max(2, int(w * thick)), cv2.LINE_AA)
    cv2.circle(m, pt(p[0]), int(w * 0.33), 1, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), 4)


def zone_mask(zona, p, sil):
    if zona == "silueta_completa":
        return sil
    m = np.zeros((H, W), np.float32)
    w = np.linalg.norm(p[11] - p[12]) + 1
    if zona == "manos":
        for j in WRISTS:
            cv2.circle(m, pt(p[j]), int(w * 0.28), 1, -1, cv2.LINE_AA)
    elif zona == "cabeza":
        cv2.circle(m, pt(p[0]), int(w * 0.45), 1, -1, cv2.LINE_AA)
    elif zona == "torso":
        cv2.fillPoly(m, [p[TORSO].astype(np.int32)], 1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def screen(a, b):
    """Blend 'screen' en [0,1]: ilumina sin quemar, como luz sobre el cuerpo."""
    return 1 - (1 - a) * (1 - b)


# ---------- Sistema de partículas con resortes ----------
N = 48
anchors = np.concatenate([np.full(N // 2, 15), np.full(N // 2, 16)])
body_anchor = np.array([0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 19, 20] * 4)[:N]
phase = _rng.random(N) * 2 * np.pi
radius = 0.4 + 0.6 * _rng.random(N)
px = poses[0][anchors].copy()
pv = np.zeros_like(px)
K, C = 38.0, 7.5  # rigidez y amortiguamiento (sub-amortiguado: rebota suave)

# Mallas precomputadas una sola vez (antes se recalculaban cada frame).
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)

# ---------- Pasada 2: render en streaming ----------
ff = imageio_ffmpeg.get_ffmpeg_exe()
enc = subprocess.Popen(
    [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
     "-i", VIDEO, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
     "-c:a", "aac", "-shortest", "-movflags", "+faststart", OUT],
    stdin=subprocess.PIPE,
)
cap = cv2.VideoCapture(VIDEO)
dt = 1 / fps
for n in range(len(poses)):
    ok, frame = cap.read()
    if not ok:
        break
    t = n * dt
    p = poses[n]
    st, zones, cbase, csec = frame_state(t)
    col_a, col_b = hls_bgr(cbase), hls_bgr(csec)
    sil = silhouette(p)
    img = frame.astype(np.float32) / 255
    light = np.zeros((H, W, 3), np.float32)
    shoulder = np.linalg.norm(p[11] - p[12]) + 1

    # 1) Distorsión: una onda concéntrica desde la cadera, amortiguada por distancia.
    if st["distorsion_onda"] > 0.01:
        hip = p[[23, 24]].mean(0)
        zm = sum(w * zone_mask(z, p, sil) for z, w in zones["distorsion_onda"].items())
        zm = cv2.GaussianBlur(zm, (0, 0), 20)
        r = np.hypot(XX - hip[0], YY - hip[1])
        amp = 7 * st["distorsion_onda"] * zm * np.exp(-r / (shoulder * 2.5))
        d = amp * np.sin(r / 14 - t * 6)
        mx = (XX + d * (XX - hip[0]) / (r + 1)).astype(np.float32)
        my = (YY + d * (YY - hip[1]) / (r + 1)).astype(np.float32)
        img = cv2.remap(img, mx, my,
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 2) Aura: halo suave del borde de la zona, gradiente base→secundario en vertical.
    g = st["glow_pulsante"]
    zm = sum(w * zone_mask(z, p, sil) for z, w in zones["glow_pulsante"].items())
    # La silueta del esqueleto es más angosta que la ropa: se dilata para que el
    # halo quede solo fuera del cuerpo y lo rodee sin teñirlo.
    k = max(3, int(shoulder * 0.45)) | 1
    zm = cv2.dilate(zm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    halo = np.clip(np.minimum(cv2.GaussianBlur(zm, (0, 0), 12 + 10 * g) * 1.6, 1) - zm, 0, 1) * g * 3.2
    grad = (YY / H)[..., None]
    light += halo[..., None] * (col_a * (1 - grad) + col_b * grad)

    # 3) Estela: una línea fina por muñeca, con grosor y opacidad que se afinan hacia la cola.
    tr = st["trail_neon"]
    if tr > 0.01:
        layer = np.zeros((H, W, 3), np.float32)
        L = int(6 + 18 * tr)
        for j in WRISTS:
            hist = poses[max(0, n - L):n + 1, j]
            for a in range(1, len(hist)):
                u = a / len(hist)
                c = hls_bgr(hls_lerp(csec, cbase, u))
                wobble = 0.8 + 0.4 * noise1d(t * 3 + a * 0.3 + j)
                cv2.line(layer, pt(hist[a - 1]), pt(hist[a]), (c * ease(u, "ease_in_out_cubic")).tolist(),
                         max(1, int(1 + 4 * tr * u * wobble)), cv2.LINE_AA)
        light += (layer + cv2.GaussianBlur(layer, (0, 0), 5) * 0.8) * min(1.0, tr * 1.8)

    # 4) Partículas con resorte: cada una busca su ancla + un offset que "respira".
    pw = st["particulas"]
    body = sum(w for z, w in zones["particulas"].items() if z == "silueta_completa")
    anc = p[anchors] * (1 - body) + p[body_anchor] * body
    open_r = shoulder * (0.15 + 0.55 * pw)
    ang = phase + t * 0.9 + np.array([noise1d(t * 0.7 + k) * 2 for k in range(N)])
    target = anc + np.stack([np.cos(ang), np.sin(ang)], 1) * (open_r * radius)[:, None]
    acc = K * (target - px) - C * pv
    pv += acc * dt
    px += pv * dt
    if pw > 0.01:
        layer = np.zeros((H, W, 3), np.float32)
        for k in range(N):
            c = hls_bgr(hls_lerp(cbase, csec, radius[k]))
            s = 1.5 + 1.5 * modulate("onda_senoidal", t * 2, phase[k])
            cv2.circle(layer, pt(px[k]), int(round(s)), c.tolist(), -1, cv2.LINE_AA)
        light += (layer + cv2.GaussianBlur(layer, (0, 0), 4) * 1.2) * min(1.0, pw * 1.6)

    out = screen(img, np.clip(light, 0, 1))
    enc.stdin.write((np.clip(out, 0, 1) * 255).astype(np.uint8).tobytes())
enc.stdin.close()
enc.wait()
print("listo:", OUT)
