import math
import time
import threading
from dataclasses import dataclass

import numpy as np
import librosa
import soundfile as sf

import tensorflow as tf
import joblib

import moderngl
import moderngl_window as mglw
from pyrr import Matrix44


def icosahedron(radius=1.0):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    v = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= radius
    f = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ], dtype=np.uint32)
    return v, f

def midpoint_cache_key(a, b):
    return tuple(sorted((int(a), int(b))))

def subdivide(vertices, faces, radius=1.0):
    verts = vertices.tolist()
    cache = {}
    new_faces = []

    def midpoint(i, j):
        key = midpoint_cache_key(i, j)
        if key in cache:
            return cache[key]
        p = (vertices[i] + vertices[j]) * 0.5
        p /= np.linalg.norm(p)
        p *= radius
        verts.append(p.tolist())
        idx = len(verts) - 1
        cache[key] = idx
        return idx

    for a, b, c in faces:
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces += [
            [a,  ab, ca],
            [b,  bc, ab],
            [c,  ca, bc],
            [ab, bc, ca],
        ]

    v = np.array(verts, dtype=np.float32)
    f = np.array(new_faces, dtype=np.uint32)
    return v, f

def build_icosphere(subdiv=2, radius=1.0):
    v, f = icosahedron(radius)
    for _ in range(subdiv):
        v, f = subdivide(v, f, radius)

    edges = set()
    for a, b, c in f:
        for e in ((a, b), (b, c), (c, a)):
            i, j = int(e[0]), int(e[1])
            if i > j:
                i, j = j, i
            edges.add((i, j))

    edge_idx = np.array([e for e in edges], dtype=np.uint32).ravel()
    return v, edge_idx


VERT = r"""
#version 330
in vec3 in_pos;

uniform mat4 u_mvp;
uniform float u_time;

uniform float u_amp;    // spikes intensity
uniform float u_freq;   // spatial frequency
uniform float u_speed;  // animation speed

float n3(vec3 p){
    return fract(sin(dot(p, vec3(12.9898,78.233,37.719))) * 43758.5453);
}
float n3s(vec3 p){ return n3(p) * 2.0 - 1.0; }

void main() {
    vec3 dir = normalize(in_pos);
    float t = u_time * u_speed;

    float d = 0.0;
    vec3 q = dir * u_freq;

    d += 0.55 * n3s(q + vec3(t,0,0));
    d += 0.28 * n3s(q*2.1 + vec3(0,t,0));
    d += 0.17 * n3s(q*3.7 + vec3(0,0,t));

    float disp = u_amp * d;
    vec3 pos = dir * (1.0 + disp);

    gl_Position = u_mvp * vec4(pos, 1.0);
}
"""

FRAG = r"""
#version 330
out vec4 fragColor;

uniform float u_alpha;
uniform float u_valence; // [-1,1]

// mood color from valence
vec3 moodColor(float v) {
    // sad -> neutral -> happy
    vec3 sad     = vec3(0.2, 0.45, 1.0);
    vec3 neutral = vec3(0.85, 0.2, 0.95);
    vec3 happy   = vec3(1.0, 0.9, 0.25);

    if (v < 0.0) {
        return mix(neutral, sad, clamp(-v, 0.0, 1.0));
    } else {
        return mix(neutral, happy, clamp(v, 0.0, 1.0));
    }
}

void main() {
    vec3 c = moodColor(u_valence);
    fragColor = vec4(c, u_alpha);
}
"""

def clamp(x, a, b):
    return max(a, min(b, x))

def lerp(a, b, t):
    return a + (b - a) * t

class EmotionSmoother:
    def __init__(self, speed=0.08):
        self.v = 0.0
        self.a = 0.0
        self.speed = float(speed)

    def update(self, target_v, target_a):
        self.v += (target_v - self.v) * self.speed
        self.a += (target_a - self.a) * self.speed
        return self.v, self.a


class AudioRingBuffer:
    def __init__(self, sr=44100, seconds=1.0):
        self.sr = int(sr)
        self.size = int(sr * seconds)
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.lock = threading.Lock()

    def push(self, samples: np.ndarray):
        samples = samples.astype(np.float32, copy=False)
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        n = len(samples)
        if n <= 0:
            return
        if n >= self.size:
            samples = samples[-self.size:]
            n = len(samples)
        with self.lock:
            self.buf = np.roll(self.buf, -n)
            self.buf[-n:] = samples

    def snapshot(self) -> np.ndarray:
        with self.lock:
            return self.buf.copy()

@dataclass
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.0

class AudioFileStreamer(threading.Thread):
    def __init__(self, path, ring: AudioRingBuffer, sr=44100, block_size=1024):
        super().__init__(daemon=True)
        self.path = path
        self.ring = ring
        self.sr = int(sr)
        self.block_size = int(block_size)
        self._stop = threading.Event()

        try:
            import sounddevice as sd
        except Exception as e:
            raise RuntimeError(
                "Manca sounddevice. Installa: pip install sounddevice"
            ) from e
        self.sd = sd

    def stop(self):
        self._stop.set()

    def run(self):
        # Leggiamo con soundfile (supporta wav/flac/ogg; per mp3 spesso funziona se hai ffmpeg/gstreamer)
        data, file_sr = sf.read(self.path, dtype='float32', always_2d=True)
        x = data.mean(axis=1)  # mono

        # resample se necessario
        if int(file_sr) != self.sr:
            x = librosa.resample(x, orig_sr=int(file_sr), target_sr=self.sr)

        idx = 0
        n = len(x)

        def callback(outdata, frames, time_info, status):
            nonlocal idx
            if self._stop.is_set():
                raise self.sd.CallbackStop()

            end = idx + frames
            if end <= n:
                chunk = x[idx:end]
                idx = end
            else:
                remain = max(0, n - idx)
                chunk = np.zeros(frames, dtype=np.float32)
                if remain > 0:
                    chunk[:remain] = x[idx:n]
                idx = n

            self.ring.push(chunk)

            out = np.column_stack([chunk, chunk]).astype(np.float32)
            outdata[:] = out

            if idx >= n:
                self._stop.set()
                raise self.sd.CallbackStop()

        with self.sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            blocksize=self.block_size,
            dtype='float32',
            callback=callback
        ):
            while not self._stop.is_set():
                time.sleep(0.02)


class EmotionInference(threading.Thread):
    def __init__(
        self,
        ring: AudioRingBuffer,
        state: EmotionState,
        model_path="best_model_dynamic_persecond_old.keras",
        norm_path="input_norm.npz",
        scaler_path="label_scaler.joblib",
        sr=44100,
        hop_seconds=0.25
    ):
        super().__init__(daemon=True)
        self.ring = ring
        self.state = state
        self.sr = int(sr)
        self.hop_seconds = float(hop_seconds)
        self._stop = threading.Event()
        self.lock = threading.Lock()

        self.model = tf.keras.models.load_model(model_path)

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            audio_1s = self.ring.snapshot()

            mel = librosa.feature.melspectrogram(
                y=audio_1s,
                sr=self.sr,
                n_mels=128,
                n_fft=2048,
                hop_length=512
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            x = np.expand_dims(mel_db, axis=(0, -1)).astype(np.float32)

            pred = self.model.predict(x, verbose=0)[0]

            v = pred[0] * 2.0 - 1.0  # [-1,1]
            a = pred[1]

            v = float(pred[0])
            a = float(pred[1])


            v = clamp(v, -1.0, 1.0)
            a = clamp(a, 0.0, 1.0)

            with self.lock:
                self.state.valence = v
                self.state.arousal = a

            time.sleep(self.hop_seconds)

    def get(self):
        with self.lock:
            return self.state.valence, self.state.arousal


def emotion_to_visual(v, a):
    v = float(np.clip(v, -1.0, 1.0))
    a = float(np.clip(a, 0.0, 1.0))


    amp = lerp(0.015, 0.22, a)

    freq = lerp(0.9, 4.0, a)

    speed = lerp(0.25, 2.8, a)

    alpha = lerp(0.35, 1.0, a)

    return amp, freq, speed, alpha


class EmotionWireSpikySphere(mglw.WindowConfig):
    title = "Emotion WireSpikySphere (CNN+GRU Valence/Arousal)"
    window_size = (1280, 720)
    resource_dir = "."
    aspect_ratio = None
    samples = 4

    # mesh density
    SUBDIV = 3  # 2..4

    # audio/model config
    AUDIO_PATH = "InTheEnd.wav"  # <-- cambia qui
    SR = 44100

    MODEL_PATH = "best_model_dynamic_persecond.keras"
    NORM_PATH = "input_norm.npz"
    SCALER_PATH = "label_scaler.joblib"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.line_width = 1.2

        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

        verts, line_idx = build_icosphere(subdiv=self.SUBDIV, radius=1.0)
        self.vbo = self.ctx.buffer(verts.astype("f4").tobytes())
        self.ibo = self.ctx.buffer(line_idx.astype("u4").tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f", "in_pos")],
            self.ibo,
            mode=moderngl.LINES,
        )

        eye = (0.0, 0.0, 3.0)
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        self.view = Matrix44.look_at(eye, target, up, dtype=np.float32)
        self.proj = Matrix44.perspective_projection(
            45.0, self.wnd.aspect_ratio, 0.1, 100.0, dtype=np.float32
        )

        self.ring = AudioRingBuffer(sr=self.SR, seconds=1.0)
        self.state = EmotionState()
        self.smoother = EmotionSmoother(speed=0.10)

        try:
            self.audio_thread = AudioFileStreamer(
                self.AUDIO_PATH,
                self.ring,
                sr=self.SR,
                block_size=1024
            )
        except Exception as e:
            raise RuntimeError(
                "Errore audio. Assicurati di avere sounddevice e un backend audio funzionante.\n"
                "Installa: pip install sounddevice\n"
                f"Dettaglio: {e}"
            )

        try:
            self.infer_thread = EmotionInference(
                ring=self.ring,
                state=self.state,
                model_path=self.MODEL_PATH,
                norm_path=self.NORM_PATH,
                scaler_path=self.SCALER_PATH,
                sr=self.SR,
                hop_seconds=0.25
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "File mancanti. Nella stessa cartella devi avere:\n"
                f"- {self.MODEL_PATH}\n"
                f"- {self.NORM_PATH}  (np.savez con mean/std)\n"
                f"- {self.SCALER_PATH} (joblib.dump del MinMaxScaler)\n\n"
                "Nel tuo training aggiungi:\n"
                "  np.savez('input_norm.npz', mean=X_mean, std=X_std)\n"
                "  joblib.dump(scaler, 'label_scaler.joblib')\n"
            ) from e

        self.audio_thread.start()
        self.infer_thread.start()

        # init uniforms (valori default)
        self.prog["u_amp"].value = 0.05
        self.prog["u_freq"].value = 1.5
        self.prog["u_speed"].value = 1.0
        self.prog["u_alpha"].value = 0.9
        self.prog["u_valence"].value = 0.0

        self._t0 = time.time()

    def close(self):
        # stop threads
        try:
            if hasattr(self, "infer_thread"):
                self.infer_thread.stop()
            if hasattr(self, "audio_thread"):
                self.audio_thread.stop()
        except Exception:
            pass
        super().close()

    def on_render(self, time_now: float, frame_time: float):
        v_raw, a_raw = self.infer_thread.get()
        v, a = self.smoother.update(v_raw, a_raw)

        amp, freq, speed, alpha = emotion_to_visual(v, a)

        self.prog["u_amp"].value = amp
        self.prog["u_freq"].value = freq
        self.prog["u_speed"].value = speed
        self.prog["u_alpha"].value = alpha
        self.prog["u_valence"].value = v

        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        rot = Matrix44.from_y_rotation((time_now * 0.25) + (v * 0.2), dtype=np.float32)
        m = rot

        mvp = self.proj * self.view * m
        self.prog["u_mvp"].write(mvp.astype("f4").tobytes())
        self.prog["u_time"].value = time_now

        self.vao.render()

        # se audio finito, chiudi
        if getattr(self.audio_thread, "_stop", None) is not None and self.audio_thread._stop.is_set():
            self.wnd.close()


if __name__ == "__main__":
    mglw.run_window_config(EmotionWireSpikySphere)
