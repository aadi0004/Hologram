import sys
import threading
import numpy as np
import cv2
import mediapipe as mp
import moderngl
from moderngl_window import WindowConfig, run_window_config
from pyrr import Matrix44
import requests
from dotenv import load_dotenv
import os
import time
import tkinter as tk
import queue
import json
import math
import random
import io

# ---- SAFE AUDIO IMPORTS ----
try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# ---- CONFIG ----
AUDIO_SAMPLE_RATE = 22050  # Lower rate for stability
AUDIO_CHANNELS = 1
AUDIO_FILENAME = "input.wav"
TTS_FILENAME = "reply.wav"
ELEVENLABS_VOICE_ID = "NYkjXRso4QIcgWakN1Cr"  # Silas Vane voice

# ---- VISEME MAPPING (from reference file) ----
PHONEME_TO_VISEME = {
    "AA":"A","AE":"A","AH":"A","AX":"A","AW":"A",
    "EH":"E","EY":"E",
    "IH":"I","IY":"I",
    "AO":"O","OW":"O","OY":"O",
    "UH":"U","UW":"U",
    "F":"FV","V":"FV",
    "M":"MBP","B":"MBP","P":"MBP",
    "L":"L",
    "S":"SZ","Z":"SZ",
    "CH":"CHJ","JH":"CHJ","SH":"CHJ","ZH":"CHJ",
    "R":"RW","W":"RW",
    "TH":"TH","DH":"TH",
}

VOWEL_VISEMES = {"a":"A","e":"E","i":"I","o":"O","u":"U"}

VISEME_TO_MOUTH = {
    # (open, wide, round, closed) each 0..1, used to deform mouth landmarks
    "REST": (0.05, 0.1, 0.0, 0.5),
    "A": (0.95, 0.25, 0.05, 0.0),
    "E": (0.55, 0.70, 0.0, 0.0),
    "I": (0.45, 0.85, 0.0, 0.0),
    "O": (0.65, 0.10, 0.85, 0.0),
    "U": (0.38, 0.10, 0.75, 0.0),
    "FV": (0.15, 0.25, 0.0, 0.9),
    "MBP": (0.00, 0.20, 0.0, 1.0),
    "L": (0.40, 0.35, 0.0, 0.1),
    "SZ": (0.15, 0.45, 0.0, 0.2),
    "CHJ": (0.45, 0.30, 0.0, 0.0),
    "RW": (0.30, 0.10, 0.55, 0.0),
    "TH": (0.35, 0.20, 0.0, 0.1),
}

VISEME_PRIORITY = {
    "MBP": 10, "FV": 9, "TH": 8, "L": 7, "CHJ": 6, "SZ": 5,
    "O": 4, "U": 4, "A": 3, "E": 3, "I": 3, "RW": 2, "REST": 1
}

# ---- Load API key ----
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if ELEVENLABS_API_KEY is None:
    print("Please set ELEVENLABS_API_KEY in your .env file.")
    sys.exit(1)

# ---- GLOBALS ----
face_points_global = None
holo_instance = None
speech_manager = None
gui_queue = queue.Queue()

def capture_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.7
    )
    cap = cv2.VideoCapture(0)
    print("Position your face in view. Press SPACE to capture.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.multi_face_landmarks[0],
                mp.solutions.face_mesh.FACEMESH_TESSELATION
            )
        
        cv2.imshow("Face Mesh Capture - Press SPACE to capture", frame)
        key = cv2.waitKey(1)
        
        if key == 32 and results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            points = []
            for lm in landmarks:
                x = (lm.x - 0.5) * 2.0
                y = -((lm.y - 0.5) * 2.0)
                z = lm.z * 0.6
                points.append([x, y, z])
            
            points = np.array(points, dtype=np.float32)
            center = points.mean(axis=0)
            points -= center
            scale = np.max(points.ptp(axis=0))
            if scale > 0:
                points /= scale / 1.1
            
            cap.release()
            cv2.destroyAllWindows()
            print(f"✅ Captured {len(points)} facial points.")
            return points
            
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

def get_face_wireframe_indices():
    """Get all face mesh connections for wireframe effect"""
    face_connections = list(mp.solutions.face_mesh.FACEMESH_TESSELATION)
    return np.array(face_connections, dtype=np.uint32).flatten()

def get_mouth_indices():
    """Get mouth landmark indices for realistic deformation"""
    # From reference file - precise mouth landmarks
    MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314]
    
    # Combine and filter for valid indices
    all_mouth = list(set(MOUTH_OUTER + MOUTH_INNER))
    valid_mouth = [idx for idx in all_mouth if 0 <= idx < 468]
    
    return np.array(valid_mouth, dtype=np.int32)

# ---- ELEVENLABS WITH ALIGNMENT (from reference) ----
def elevenlabs_tts_with_alignment(text: str):
    """Enhanced TTS with alignment data for precise lip sync"""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("Set ELEVENLABS_API_KEY in your environment")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    
    # 1) Try to fetch alignment data
    alignment = None
    try:
        headers_json = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "accept": "application/json",
            "content-type": "application/json",
        }
        payload_json = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.8,
                "similarity_boost": 0.9,
                "style": 0.3,
                "use_speaker_boost": True
            },
            "alignment": True
        }
        r = requests.post(url, headers=headers_json, json=payload_json, timeout=30)
        if r.status_code == 200:
            data = r.json()
            alignment = data.get("alignment") or data.get("timings")
    except Exception as e:
        print(f"Alignment fetch failed: {e}")
        alignment = None

    # 2) Fetch MP3 and decode safely
    headers_mp3 = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.8,
            "similarity_boost": 0.9,
            "style": 0.3,
            "use_speaker_boost": True
        }
    }
    
    rr = requests.post(url, headers=headers_mp3, json=payload, timeout=30)
    rr.raise_for_status()
    
    # Safe audio decoding
    if PYDUB_AVAILABLE:
        try:
            mp3_bytes = io.BytesIO(rr.content)
            seg = AudioSegment.from_file(mp3_bytes, format="mp3")
            seg = seg.set_channels(1)  # mono
            seg = seg.set_frame_rate(AUDIO_SAMPLE_RATE)
            sample_rate = seg.frame_rate
            audio = np.array(seg.get_array_of_samples(), dtype=np.int16)
        except Exception as e:
            print(f"Pydub decode failed: {e}")
            return None, None, []
    else:
        # Save to file and use soundfile
        with open(TTS_FILENAME, 'wb') as f:
            f.write(rr.content)
        try:
            audio, sample_rate = sf.read(TTS_FILENAME, dtype='int16')
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # Take first channel
            audio = audio.astype(np.int16)
        except Exception as e:
            print(f"Soundfile decode failed: {e}")
            return None, None, []
    
    # Build timeline from alignment
    total_audio_sec = len(audio) / sample_rate
    timeline = build_viseme_timeline_from_alignment(alignment or {"words": []}, total_audio_sec)
    
    return audio, sample_rate, timeline

def normalize_phoneme(p: str) -> str:
    p = p.strip().upper()
    return "".join([c for c in p if not c.isdigit()])

def words_to_viseme_guess(word: str):
    """Fallback viseme generation from word"""
    seq = []
    for ch in word.lower():
        if ch in VOWEL_VISEMES: 
            seq.append(VOWEL_VISEMES[ch])
        elif ch in ["m","b","p"]: 
            seq.append("MBP")
        elif ch in ["f","v"]: 
            seq.append("FV")
        elif ch == "l": 
            seq.append("L")
        elif ch in ["s","z"]: 
            seq.append("SZ")
        elif ch in ["r","w"]: 
            seq.append("RW")
        elif ch in ["t","d","h"]: 
            seq.append("TH")
        else: 
            seq.append("REST")
    return seq or ["REST"]

def build_viseme_timeline_from_alignment(aln: dict, total_audio_sec: float):
    """Build timeline from ElevenLabs alignment data"""
    entries = []
    
    # Try explicit phonemes first
    if isinstance(aln, dict) and isinstance(aln.get("phonemes"), list):
        for ph in aln["phonemes"]:
            phn = normalize_phoneme(ph.get("phoneme",""))
            vis = PHONEME_TO_VISEME.get(phn)
            if not vis: continue
            start = float(ph.get("start", 0.0))
            end = float(ph.get("end", start + 0.08))
            entries.append({"start": start, "end": end, "viseme": vis})
        
        if entries:
            return sorted(entries, key=lambda e: e["start"])
    
    # Try words with phonemes
    if isinstance(aln, dict) and isinstance(aln.get("words"), list):
        for w in aln["words"]:
            wstart = float(w.get("start", 0.0))
            wend = float(w.get("end", max(0.08, wstart + 0.2)))
            duration = max(0.08, wend - wstart)
            
            phs = w.get("phonemes")
            if isinstance(phs, list) and phs:
                seg = len(phs)
                for i, ph in enumerate(phs):
                    phn = normalize_phoneme(ph.get("phoneme",""))
                    vis = PHONEME_TO_VISEME.get(phn)
                    if not vis: continue
                    s = wstart + (i/seg) * duration
                    e = wstart + ((i+1)/seg) * duration
                    entries.append({"start": s, "end": e, "viseme": vis})
            else:
                # Guess from word
                guess = words_to_viseme_guess(w.get("word",""))
                seg = max(1, len(guess))
                for i, vis in enumerate(guess):
                    s = wstart + (i/seg) * duration
                    e = wstart + ((i+1)/seg) * duration
                    entries.append({"start": s, "end": e, "viseme": vis})
        
        if entries:
            return sorted(entries, key=lambda e: e["start"])
    
    # Fallback synthetic cycle
    t = 0.0
    step = 0.12
    cycle = ["A","E","I","O","U","MBP","FV","REST"]
    ci = 0
    while t < total_audio_sec:
        entries.append({"start": t, "end": min(total_audio_sec, t+step), "viseme": cycle[ci % len(cycle)]})
        t += step
        ci += 1
    
    return entries

def active_viseme(timeline, t: float) -> str:
    """Get active viseme at time t with priority"""
    window = 0.05
    best, pr = "REST", -1
    
    for e in timeline:
        if e["start"] - window <= t <= e["end"] + window:
            v = e["viseme"]
            p = VISEME_PRIORITY.get(v, 0)
            if p > pr:
                best, pr = v, p
        if e["start"] > t + 0.2:
            break
    
    return best

class HologramFace(WindowConfig):
    gl_version = (3, 3)
    title = "3D Holographic Face Assistant - Realistic Lip Sync"
    window_size = (1200, 900)
    resource_dir = '.'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global holo_instance, face_points_global
        holo_instance = self
        
        self.mesh = face_points_global.copy()
        self.base_mesh = face_points_global.copy()
        self.vertex_count = len(self.mesh)
        self.time = 0.0
        self.speaking = False
        
        # ---- REALISTIC LIP SYNC SYSTEM ----
        self.mouth_indices = get_mouth_indices()
        self.wireframe_indices = get_face_wireframe_indices()
        self.current_viseme = "REST"
        
        # Blink system
        self.blink_t = 0.0
        self.next_blink_gap = random.uniform(3.0, 6.0)
        
        # Eye indices (rough)
        self.eyelid_upper = [159, 145, 160, 161, 246]
        self.eyelid_lower = [23, 27, 159, 158, 157]
        
        print(f"✅ Using {len(self.mouth_indices)} mouth indices for realistic viseme lip sync")

        # Rendering programs
        self.point_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                uniform float time;
                out float brightness;
                void main() {
                    vec3 pos = in_position;
                    brightness = 0.9 + 0.1 * sin(time * 3.0 + pos.x * 8.0);
                    gl_Position = mvp * vec4(pos, 1.0);
                    gl_PointSize = 5.0;
                }
            """,
            fragment_shader="""
                #version 330
                in float brightness;
                out vec4 fragColor;
                void main() {
                    float d = length(gl_PointCoord - vec2(0.5));
                    if(d > 0.5) discard;
                    float alpha = 1.0 - smoothstep(0.0, 0.5, d);
                    vec3 color = vec3(0.0, 0.9, 1.0) * brightness;
                    fragColor = vec4(color, alpha * 0.95);
                }
            """
        )

        self.line_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                uniform float time;
                out float glow;
                void main() {
                    glow = 0.7 + 0.3 * sin(time * 2.0);
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in float glow;
                out vec4 fragColor;
                void main() {
                    vec3 color = vec3(0.0, 0.85, 1.0) * glow;
                    fragColor = vec4(color, 0.4);
                }
            """
        )

        # Create buffers
        self.vbo_points = self.ctx.buffer(self.mesh.tobytes())
        self.vbo_lines = self.ctx.buffer(self.mesh.tobytes())
        self.ibo_wireframe = self.ctx.buffer(self.wireframe_indices.tobytes())
        
        self.vao_points = self.ctx.simple_vertex_array(self.point_prog, self.vbo_points, 'in_position')
        self.vao_lines = self.ctx.vertex_array(
            self.line_prog, [(self.vbo_lines, '3f', 'in_position')], self.ibo_wireframe
        )

    def start_speaking(self):
        print("🗣️ Realistic viseme lip animation started")
        self.speaking = True

    def stop_speaking(self):
        print("🔇 Realistic viseme lip animation stopped")
        self.speaking = False
        self.current_viseme = "REST"
        self.reset_mesh()

    def set_viseme(self, viseme: str):
        """Set current viseme for lip sync"""
        self.current_viseme = viseme if viseme in VISEME_TO_MOUTH else "REST"

    def reset_mesh(self):
        """Reset mesh to original positions"""
        try:
            self.mesh = self.base_mesh.copy()
            self.vbo_points.write(self.mesh.tobytes())
            self.vbo_lines.write(self.mesh.tobytes())
        except Exception as e:
            print(f"Reset error: {e}")

    def apply_mouth_deformation(self):
        """Apply realistic mouth deformation based on current viseme"""
        try:
            open_v, wide_v, round_v, closed_v = VISEME_TO_MOUTH.get(self.current_viseme, VISEME_TO_MOUTH["REST"])
            
            # Get mouth points
            mouth_points = self.mesh[self.mouth_indices]
            center = mouth_points.mean(axis=0, keepdims=True)
            
            # Apply deformation (from reference file algorithm)
            v = mouth_points - center
            v[:, 1] *= (1.0 + (open_v * 1.6 - closed_v * 0.8))
            v[:, 0] *= (1.0 + (wide_v * 0.8))
            v *= (1.0 - round_v * 0.45)
            
            # Update mesh
            self.mesh[self.mouth_indices] = center + v
            
            # Apply blinking
            blink_phase = self.get_blink_amount()
            for i in self.eyelid_upper:
                if i < len(self.mesh):
                    self.mesh[i, 1] += 0.02 * blink_phase
            for i in self.eyelid_lower:
                if i < len(self.mesh):
                    self.mesh[i, 1] -= 0.02 * blink_phase
                    
        except Exception as e:
            print(f"Mouth deformation error: {e}")

    def get_blink_amount(self):
        """Calculate blink animation"""
        t = self.blink_t
        if t < 0.18:
            x = t / 0.18
            return 1.0 - abs(2*x - 1.0)
        return 0.0

    def update(self, dt: float):
        """Update animations"""
        # Blink system
        self.blink_t += dt
        if self.blink_t > self.next_blink_gap:
            self.blink_t = 0.0
            self.next_blink_gap = random.uniform(3.0, 6.0)

    def on_render(self, time, frame_time):
        try:
            self.time = time
            self.update(frame_time)
            
            # Reset to base mesh
            self.mesh = self.base_mesh.copy()
            
            # Apply mouth deformation
            if self.speaking:
                self.apply_mouth_deformation()

            # Static positioning
            model = Matrix44.identity()
            view = Matrix44.look_at(
                eye=(0, 0, 2.8),
                target=(0, 0, 0),
                up=(0, 1, 0)
            )
            proj = Matrix44.perspective_projection(
                fovy=50.0,
                aspect=self.wnd.aspect_ratio,
                near=0.1,
                far=100.0
            )
            mvp = proj * view * model

            # Update uniforms
            self.point_prog['mvp'].write(mvp.astype('f4').tobytes())
            self.point_prog['time'].value = self.time
            self.line_prog['mvp'].write(mvp.astype('f4').tobytes())
            self.line_prog['time'].value = self.time

            # Update buffers
            self.vbo_points.write(self.mesh.tobytes())
            self.vbo_lines.write(self.mesh.tobytes())

            # Render
            self.ctx.clear(0, 0, 0, 1)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            
            self.vao_lines.render(mode=moderngl.LINES)
            self.vao_points.render(mode=moderngl.POINTS)
            
        except Exception as e:
            print(f"Render error: {e}")

# ---- SAFE AUDIO FUNCTIONS ----
def safe_play_audio(audio_data, sample_rate):
    """Play audio safely without crashes"""
    try:
        if sd is not None:
            # Normalize to prevent clipping
            if len(audio_data) > 0:
                audio_norm = audio_data.astype(np.float32) / 32768.0
                sd.play(audio_norm, samplerate=sample_rate)
                sd.wait()
        else:
            print("Audio playback unavailable")
    except Exception as e:
        print(f"Audio playback error: {e}")

def safe_record_audio(filename, duration=5):
    """Record audio safely"""
    try:
        if sd is not None and sf is not None:
            print("🎤 Recording... Speak clearly now.")
            audio = sd.rec(int(duration * AUDIO_SAMPLE_RATE), 
                          samplerate=AUDIO_SAMPLE_RATE, 
                          channels=AUDIO_CHANNELS, 
                          dtype='float32')
            sd.wait()
            
            # Simple noise gate
            audio_abs = np.abs(audio)
            noise_floor = np.percentile(audio_abs, 20)
            audio = np.where(audio_abs < noise_floor * 2, audio * 0.1, audio)
            
            # Normalize and convert to int16
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            audio_int16 = (audio * 32767).astype(np.int16)
            sf.write(filename, audio_int16, AUDIO_SAMPLE_RATE)
            print("✅ Recording complete.")
            return True
        else:
            print("Recording unavailable")
            return False
    except Exception as e:
        print(f"Recording error: {e}")
        return False

def simple_asr(audiofile_path):
    """Simple fallback ASR"""
    # This is a placeholder - you could integrate with Google Speech-to-Text
    # or other services here
    return "Hello"

class SpeechManager:
    def __init__(self, holo):
        self.holo = holo
        self.listening = False
        self.timeline = []
        self.audio_start_time = None

    def speak_with_lipsync(self, text):
        """Enhanced speak with proper viseme lip sync"""
        def tts_and_play():
            try:
                self.holo.start_speaking()
                
                # Get audio and timeline from ElevenLabs
                audio_data, sample_rate, timeline = elevenlabs_tts_with_alignment(text)
                
                if audio_data is not None and timeline:
                    self.timeline = timeline
                    self.audio_start_time = time.time()
                    
                    # Start lip sync in separate thread
                    def lip_sync_thread():
                        while self.holo.speaking and self.audio_start_time:
                            t_now = time.time() - self.audio_start_time
                            viseme = active_viseme(self.timeline, t_now)
                            self.holo.set_viseme(viseme)
                            time.sleep(0.02)  # 50 FPS lip sync
                    
                    threading.Thread(target=lip_sync_thread, daemon=True).start()
                    
                    # Play audio
                    safe_play_audio(audio_data, sample_rate)
                else:
                    # Fallback timing
                    time.sleep(max(2, len(text) * 0.08))
                
                self.audio_start_time = None
                self.holo.stop_speaking()
                
            except Exception as e:
                print(f"TTS error: {e}")
                self.audio_start_time = None
                self.holo.stop_speaking()
        
        threading.Thread(target=tts_and_play, daemon=True).start()

    def listen(self, on_text):
        if self.listening:
            print("Already listening...")
            return
        
        self.listening = True

        def listen_workflow():
            try:
                if safe_record_audio(AUDIO_FILENAME, duration=5):
                    # Here you could use Vosk or other STT
                    recognized = simple_asr(AUDIO_FILENAME)
                else:
                    recognized = ""
                
                print(f"👤 User said: '{recognized}'")
                self.listening = False
                on_text(recognized)
                
            except Exception as e:
                print(f"Listen error: {e}")
                self.listening = False
                on_text("")

        threading.Thread(target=listen_workflow, daemon=True).start()

class ControlPanel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Lumen AI - Realistic Viseme Lip Sync")
        self.root.geometry("440x280")
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(False, False)
        
        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(expand=True, fill='both', padx=20, pady=15)
        
        title_label = tk.Label(main_frame, text="🤖 LUMEN + SILAS VANE", 
                             fg='#00ffff', bg='#0a0a0a', font=('Arial', 22, 'bold'))
        title_label.pack(pady=15)
        
        self.speak_button = tk.Button(main_frame, text="🎤 HOLD TO SPEAK", 
                                     bg='#1a4d66', fg='white', 
                                     font=('Arial', 18, 'bold'),
                                     relief='raised', borderwidth=4,
                                     activebackground='#ff4444',
                                     cursor='hand2')
        self.speak_button.pack(pady=20, padx=20, fill='x', ipady=15)
        
        self.status_label = tk.Label(main_frame, text="🟢 REALISTIC VISEME LIP SYNC READY", 
                                   fg='#00ff00', bg='#0a0a0a', font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=12)
        
        info_label = tk.Label(main_frame, text="Advanced viseme-based lip animation", 
                            fg='#888888', bg='#0a0a0a', font=('Arial', 11))
        info_label.pack()
        
        tech_label = tk.Label(main_frame, text="✨ ElevenLabs alignment + phoneme mapping", 
                            fg='#ffaa00', bg='#0a0a0a', font=('Arial', 10))
        tech_label.pack(pady=5)
        
        self.speak_button.bind('<Button-1>', self.start_speaking)
        self.speak_button.bind('<ButtonRelease-1>', self.stop_speaking)
        
        self.speaking = False
        self.process_gui_updates()
        
    def process_gui_updates(self):
        """Thread-safe GUI updates"""
        try:
            while True:
                try:
                    action, data = gui_queue.get_nowait()
                    if action == 'status':
                        self.status_label.config(text=data['text'], fg=data['color'])
                    elif action == 'button':
                        self.speak_button.config(text=data['text'], bg=data['color'])
                except queue.Empty:
                    break
        except:
            pass
        self.root.after(100, self.process_gui_updates)
        
    def update_status_safe(self, text, color):
        gui_queue.put(('status', {'text': text, 'color': color}))
        
    def update_button_safe(self, text, color):
        gui_queue.put(('button', {'text': text, 'color': color}))
    
    def start_speaking(self, event):
        global speech_manager
        if not self.speaking and speech_manager:
            self.speaking = True
            self.update_button_safe("🔴 RECORDING...", '#cc2222')
            self.update_status_safe("🎧 LISTENING...", '#ffaa00')
            speech_manager.listen(self.process_speech)
    
    def stop_speaking(self, event):
        if self.speaking:
            self.speaking = False
            self.update_button_safe("🎤 HOLD TO SPEAK", '#1a4d66')
            self.update_status_safe("🔄 PROCESSING...", '#ff8800')
    
    def process_speech(self, text):
        global speech_manager
        
        RESPONSES = {
            "hello": "Hello! I'm Lumen with realistic viseme-based lip synchronization using ElevenLabs alignment data and Silas Vane's voice!",
            "hi": "Hi there! My lip sync now uses proper viseme mapping with A, E, I, O, U, MBP, FV phonemes for ultra-realistic speech!",
            "test": "Perfect! My viseme lip sync with ElevenLabs alignment creates incredibly realistic mouth movements synchronized to speech!",
            "viseme": "Yes! I use advanced viseme mapping where each phoneme creates specific mouth shapes - A for open, MBP for closed, O for rounded!",
            "realistic": "My lip animation uses ElevenLabs phoneme alignment data to create visemes that match real human speech patterns!",
            "phoneme": "I map phonemes to visemes: A-E-I-O-U for vowels, MBP for lips closed, FV for lip-teeth contact, and more!",
            "voice": "I speak with Silas Vane's voice and my lips sync perfectly using timeline-based viseme animation!",
            "mouth": "My mouth deformation uses mathematical models with open, wide, round, and closed parameters for each viseme!",
            "sync": "My lip sync is frame-accurate using ElevenLabs alignment timestamps matched to viseme priorities!",
            "amazing": "Thank you! This advanced viseme system creates the most realistic lip animation possible!"
        }
        
        print(f"👤 User said: '{text}'")
        response = "I can hear you perfectly! My realistic viseme lip sync with ElevenLabs alignment is ready to demonstrate!"
        text_lower = text.lower().strip()
        
        if text_lower:
            for key, reply in RESPONSES.items():
                if key in text_lower:
                    response = reply
                    break
        
        print(f"🤖 Lumen (Viseme) responds: '{response}'")
        self.update_status_safe("🗣️ REALISTIC VISEME SPEAKING...", '#00ffff')
        
        def reset_status():
            time.sleep(max(4, len(response) * 0.08))
            self.update_status_safe("🟢 REALISTIC VISEME LIP SYNC READY", '#00ff00')
        
        speech_manager.speak_with_lipsync(response)
        threading.Thread(target=reset_status, daemon=True).start()
    
    def run(self):
        self.root.mainloop()

def main():
    global face_points_global, holo_instance, speech_manager
    
    print("🚀 Initializing REALISTIC viseme-based hologram with ElevenLabs alignment...")
    
    face_points_global = capture_face_mesh()
    
    def run_holo():
        run_window_config(HologramFace)
    
    holo_thread = threading.Thread(target=run_holo, daemon=True)
    holo_thread.start()
    
    print("⏳ Starting realistic hologram renderer...")
    count = 0
    while holo_instance is None:
        time.sleep(0.1)
        count += 1
        if count > 120:
            print("❌ Failed to initialize hologram window.")
            sys.exit(1)
    
    speech_manager = SpeechManager(holo_instance)
    
    print("✅ REALISTIC viseme-based Lumen hologram is ready!")
    control_panel = ControlPanel()
    control_panel.run()

if __name__ == "__main__":
    main()
