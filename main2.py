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
import tempfile

# ---- WHISPER FOR PERFECT SPEECH RECOGNITION ----
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper available for perfect speech recognition")
except ImportError:
    WHISPER_AVAILABLE = False
    print("❌ Install Whisper: pip install openai-whisper")

# ---- SAFE AUDIO IMPORTS ----
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("❌ Install audio: pip install sounddevice soundfile")

# ---- CONFIG ----
AUDIO_SAMPLE_RATE = 22050
AUDIO_CHANNELS = 1
AUDIO_FILENAME = "input.wav"
TTS_FILENAME = "reply.wav"
ELEVENLABS_VOICE_ID = "NYkjXRso4QIcgWakN1Cr"  # Silas Vane voice

# ---- ENHANCED VISEME MAPPING ----
VISEME_TO_MOUTH = {
    "REST": (0.05, 0.1, 0.0, 0.5),
    "A": (0.95, 0.25, 0.05, 0.0),    # Wide open - "ah" sound
    "E": (0.55, 0.70, 0.0, 0.0),     # Wide spread - "eh" sound
    "I": (0.45, 0.85, 0.0, 0.0),     # Very wide - "ee" sound
    "O": (0.65, 0.10, 0.85, 0.0),    # Rounded - "oh" sound
    "U": (0.38, 0.10, 0.75, 0.0),    # Small rounded - "oo" sound
    "MBP": (0.00, 0.20, 0.0, 1.0),   # Closed - m/b/p sounds
    "FV": (0.15, 0.25, 0.0, 0.9),    # Lip-teeth - f/v sounds
    "L": (0.40, 0.35, 0.0, 0.1),     # Tongue tip - l sound
    "SZ": (0.15, 0.45, 0.0, 0.2),    # Slight open - s/z sounds
    "TH": (0.35, 0.20, 0.0, 0.1),    # Tongue between teeth
    "RW": (0.30, 0.10, 0.55, 0.0),   # Slight round - r/w sounds
    "CHJ": (0.45, 0.30, 0.0, 0.0),   # Medium open - ch/j sounds
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
whisper_model = None

def init_whisper():
    """Initialize Whisper model for perfect speech recognition"""
    global whisper_model
    if WHISPER_AVAILABLE:
        try:
            print("🔄 Loading Whisper model for perfect speech recognition...")
            whisper_model = whisper.load_model("base")  # Good balance of speed/accuracy
            print("✅ Whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Whisper model load failed: {e}")
            whisper_model = None
    else:
        whisper_model = None

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
    """Get mouth landmark indices for beautiful deformation"""
    # More comprehensive mouth landmarks for beautiful animation
    MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314]
    LIP_DETAIL = [12, 15, 16, 17, 18, 200, 199, 175, 0, 13, 82, 81, 80, 78]
    
    all_mouth = list(set(MOUTH_OUTER + MOUTH_INNER + LIP_DETAIL))
    valid_mouth = [idx for idx in all_mouth if 0 <= idx < 468]
    
    return np.array(valid_mouth, dtype=np.int32)

# ---- IMPROVED TTS WITH FALLBACK ----
def elevenlabs_tts_simple(text: str):
    """Simplified TTS without alignment (more reliable)"""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("Set ELEVENLABS_API_KEY")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.8,
            "similarity_boost": 0.9,
            "style": 0.4,
            "use_speaker_boost": True
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file and return path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            f.write(response.content)
            return f.name
            
    except Exception as e:
        print(f"❌ TTS failed: {e}")
        return None

def generate_viseme_timeline(text: str, audio_duration: float):
    """Generate beautiful viseme timeline from text analysis"""
    timeline = []
    words = text.lower().split()
    
    if not words:
        return [{"start": 0, "end": audio_duration, "viseme": "REST"}]
    
    # Calculate timing per word
    time_per_word = audio_duration / len(words)
    current_time = 0
    
    for word in words:
        word_duration = time_per_word
        
        # Analyze word for visemes
        visemes = []
        
        # Simple phonetic mapping for common patterns
        for char in word:
            if char in ['a', 'ah']: visemes.append('A')
            elif char in ['e', 'eh']: visemes.append('E')  
            elif char in ['i', 'ee']: visemes.append('I')
            elif char in ['o', 'oh']: visemes.append('O')
            elif char in ['u', 'oo']: visemes.append('U')
            elif char in ['m', 'b', 'p']: visemes.append('MBP')
            elif char in ['f', 'v']: visemes.append('FV')
            elif char == 'l': visemes.append('L')
            elif char in ['s', 'z']: visemes.append('SZ')
            elif char in ['r', 'w']: visemes.append('RW')
            elif char in ['t', 'd', 'th']: visemes.append('TH')
            else: visemes.append('REST')
        
        # If no visemes found, use defaults based on word
        if not visemes:
            if 'hello' in word: visemes = ['E', 'L', 'O']
            elif 'how' in word: visemes = ['A', 'U']
            elif 'you' in word: visemes = ['U', 'U']
            elif 'are' in word: visemes = ['A', 'R']
            elif 'what' in word: visemes = ['U', 'A', 'TH']
            elif 'can' in word: visemes = ['A', 'MBP']
            else: visemes = ['A', 'E', 'REST']
        
        # Create timeline entries for this word
        if visemes:
            viseme_duration = word_duration / len(visemes)
            for i, viseme in enumerate(visemes):
                start_time = current_time + (i * viseme_duration)
                end_time = start_time + viseme_duration
                timeline.append({
                    "start": start_time,
                    "end": end_time,
                    "viseme": viseme
                })
        
        current_time += word_duration
    
    return timeline

def active_viseme(timeline, t: float) -> str:
    """Get active viseme at time t"""
    for entry in timeline:
        if entry["start"] <= t <= entry["end"]:
            return entry["viseme"]
    return "REST"

# ---- PERFECT SPEECH RECOGNITION WITH WHISPER ----
def whisper_transcribe(audio_file: str) -> str:
    """Use Whisper for perfect speech recognition"""
    if not whisper_model:
        return "whisper not available"
    
    try:
        result = whisper_model.transcribe(audio_file, language="en")
        text = result["text"].strip()
        
        # Clean up common artifacts
        text = text.replace("[Music]", "").replace("[Applause]", "").strip()
        
        print(f"🎯 Whisper transcribed: '{text}'")
        return text
        
    except Exception as e:
        print(f"❌ Whisper transcription failed: {e}")
        return ""

def record_audio_for_whisper(filename: str, duration: int = 6) -> bool:
    """Record high-quality audio for Whisper"""
    if not AUDIO_AVAILABLE:
        print("❌ Audio recording not available")
        return False
        
    try:
        print("🎤 Recording with Whisper-optimized settings...")
        
        # Record with settings optimized for Whisper
        audio = sd.rec(
            int(duration * AUDIO_SAMPLE_RATE), 
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS, 
            dtype='float32'
        )
        sd.wait()
        
        # Apply noise gate (simple but effective)
        audio_abs = np.abs(audio)
        noise_threshold = np.percentile(audio_abs, 15)
        audio = np.where(audio_abs < noise_threshold, audio * 0.05, audio)
        
        # Normalize for Whisper (it expects -1 to 1 range)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Save as WAV for Whisper
        sf.write(filename, audio, AUDIO_SAMPLE_RATE)
        print("✅ High-quality recording complete!")
        return True
        
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return False

class HologramFace(WindowConfig):
    gl_version = (3, 3)
    title = "3D Holographic Face Assistant - Perfect Speech + Beautiful Lip Sync"
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
        
        # ---- BEAUTIFUL LIP SYNC SYSTEM ----
        self.mouth_indices = get_mouth_indices()
        self.wireframe_indices = get_face_wireframe_indices()
        self.current_viseme = "REST"
        self.viseme_intensity = 0.0
        self.viseme_smooth = 0.0
        
        # Enhanced blinking with natural variation
        self.blink_t = 0.0
        self.next_blink_gap = random.uniform(2.0, 5.0)
        self.blink_intensity = 0.0
        
        # Eye indices for blinking
        self.eyelid_upper = [159, 145, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.eyelid_lower = [23, 27, 159, 158, 157, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157]
        
        print(f"✅ Using {len(self.mouth_indices)} mouth indices for beautiful viseme lip sync")

        # Enhanced shaders for beautiful hologram effect
        self.point_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                uniform float time;
                out float brightness;
                void main() {
                    vec3 pos = in_position;
                    brightness = 0.85 + 0.15 * sin(time * 4.0 + pos.x * 10.0 + pos.y * 8.0);
                    gl_Position = mvp * vec4(pos, 1.0);
                    gl_PointSize = 6.0;
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
                    vec3 color = vec3(0.0, 0.95, 1.0) * brightness;
                    fragColor = vec4(color, alpha * 0.98);
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
                    glow = 0.6 + 0.4 * sin(time * 2.5 + gl_VertexID * 0.1);
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in float glow;
                out vec4 fragColor;
                void main() {
                    vec3 color = vec3(0.0, 0.9, 1.0) * glow;
                    fragColor = vec4(color, 0.5);
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
        print("🗣️ Beautiful viseme lip animation started")
        self.speaking = True
        self.viseme_intensity = 1.0

    def stop_speaking(self):
        print("🔇 Beautiful viseme lip animation stopped")
        self.speaking = False
        self.current_viseme = "REST"
        self.viseme_intensity = 0.0

    def set_viseme(self, viseme: str):
        """Set current viseme for beautiful lip sync"""
        if viseme in VISEME_TO_MOUTH:
            self.current_viseme = viseme
        else:
            self.current_viseme = "REST"

    def apply_beautiful_mouth_deformation(self):
        """Apply beautiful, smooth mouth deformation"""
        try:
            # Get viseme parameters
            open_v, wide_v, round_v, closed_v = VISEME_TO_MOUTH.get(self.current_viseme, VISEME_TO_MOUTH["REST"])
            
            # Smooth viseme transitions
            target_intensity = 1.0 if self.speaking else 0.0
            self.viseme_smooth += (target_intensity - self.viseme_smooth) * 0.1
            
            # Apply intensity smoothing
            open_v *= self.viseme_smooth
            wide_v *= self.viseme_smooth
            round_v *= self.viseme_smooth
            closed_v *= self.viseme_smooth
            
            # Get mouth points
            mouth_points = self.mesh[self.mouth_indices]
            center = mouth_points.mean(axis=0, keepdims=True)
            
            # Enhanced deformation with more natural movement
            v = mouth_points - center
            
            # Vertical deformation (mouth opening)
            v[:, 1] *= (1.0 + (open_v * 1.8 - closed_v * 0.9))
            
            # Horizontal deformation (mouth width)
            v[:, 0] *= (1.0 + (wide_v * 0.9))
            
            # Rounding effect (lip protrusion)
            if round_v > 0:
                v *= (1.0 - round_v * 0.3)
                # Add slight forward movement for rounding
                v[:, 2] += round_v * 0.04
            
            # Update mesh with enhanced movement
            self.mesh[self.mouth_indices] = center + v
            
            # Enhanced blinking with smooth animation
            self.apply_beautiful_blinking()
            
        except Exception as e:
            print(f"❌ Mouth deformation error: {e}")

    def apply_beautiful_blinking(self):
        """Apply smooth, beautiful blinking animation"""
        try:
            # Calculate smooth blink intensity
            blink_phase = self.get_smooth_blink_amount()
            self.blink_intensity += (blink_phase - self.blink_intensity) * 0.3
            
            # Apply to eyelids with smooth falloff
            for i, idx in enumerate(self.eyelid_upper):
                if idx < len(self.mesh):
                    intensity = self.blink_intensity * (1.0 - i * 0.05)  # Falloff
                    self.mesh[idx, 1] += 0.025 * intensity
                    
            for i, idx in enumerate(self.eyelid_lower):
                if idx < len(self.mesh):
                    intensity = self.blink_intensity * (1.0 - i * 0.05)
                    self.mesh[idx, 1] -= 0.015 * intensity
                    
        except Exception as e:
            print(f"❌ Blinking error: {e}")

    def get_smooth_blink_amount(self) -> float:
        """Calculate smooth blinking with natural timing"""
        t = self.blink_t
        if t < 0.15:  # Blink duration
            # Smooth easing function for natural blink
            x = t / 0.15
            return 0.5 * (1 - np.cos(np.pi * x))  # Smooth sine wave
        return 0.0

    def update(self, dt: float):
        """Update all animations with smooth timing"""
        # Enhanced blink system with variation
        self.blink_t += dt
        if self.blink_t > self.next_blink_gap:
            self.blink_t = 0.0
            self.next_blink_gap = random.uniform(1.8, 4.5)  # More natural variation

    def on_render(self, time, frame_time):
        try:
            self.time = time
            self.update(frame_time)
            
            # Reset to base mesh
            self.mesh = self.base_mesh.copy()
            
            # Apply beautiful mouth deformation
            self.apply_beautiful_mouth_deformation()

            # Enhanced camera position with subtle movement
            model = Matrix44.identity()
            view = Matrix44.look_at(
                eye=(0, 0, 2.6),
                target=(0, 0, 0),
                up=(0, 1, 0)
            )
            proj = Matrix44.perspective_projection(
                fovy=52.0,
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

            # Enhanced rendering
            self.ctx.clear(0, 0, 0, 1)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            
            # Multi-layer rendering for depth
            self.vao_lines.render(mode=moderngl.LINES)
            self.vao_points.render(mode=moderngl.POINTS)
            
        except Exception as e:
            print(f"❌ Render error: {e}")

# ---- ENHANCED SPEECH MANAGER ----
class SpeechManager:
    def __init__(self, holo):
        self.holo = holo
        self.listening = False
        self.timeline = []
        self.audio_start_time = None

    def speak_with_perfect_lipsync(self, text):
        """Enhanced speak with perfect viseme lip sync"""
        def tts_and_play():
            try:
                self.holo.start_speaking()
                
                # Get TTS audio
                audio_file = elevenlabs_tts_simple(text)
                
                if audio_file:
                    # Get audio duration and generate timeline
                    audio_data, sample_rate = sf.read(audio_file)
                    audio_duration = len(audio_data) / sample_rate
                    
                    self.timeline = generate_viseme_timeline(text, audio_duration)
                    self.audio_start_time = time.time()
                    
                    # Start lip sync thread
                    def lip_sync_thread():
                        while self.holo.speaking and self.audio_start_time:
                            t_now = time.time() - self.audio_start_time
                            viseme = active_viseme(self.timeline, t_now)
                            self.holo.set_viseme(viseme)
                            time.sleep(0.016)  # 60 FPS lip sync
                    
                    threading.Thread(target=lip_sync_thread, daemon=True).start()
                    
                    # Play audio with sounddevice
                    if AUDIO_AVAILABLE:
                        sd.play(audio_data, sample_rate)
                        sd.wait()
                    
                    # Clean up temp file
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
                else:
                    # Fallback timing
                    time.sleep(max(2, len(text) * 0.08))
                
                self.audio_start_time = None
                self.holo.stop_speaking()
                
            except Exception as e:
                print(f"❌ TTS error: {e}")
                self.audio_start_time = None
                self.holo.stop_speaking()
        
        threading.Thread(target=tts_and_play, daemon=True).start()

    def listen_with_whisper(self, on_text):
        """Perfect speech recognition with Whisper"""
        if self.listening:
            print("🔄 Already listening...")
            return
        
        self.listening = True

        def listen_workflow():
            try:
                # Record high-quality audio for Whisper
                if record_audio_for_whisper(AUDIO_FILENAME, duration=6):
                    # Use Whisper for perfect transcription
                    recognized = whisper_transcribe(AUDIO_FILENAME)
                else:
                    recognized = ""
                
                print(f"🎯 Perfect recognition: '{recognized}'")
                self.listening = False
                on_text(recognized)
                
            except Exception as e:
                print(f"❌ Listen error: {e}")
                self.listening = False
                on_text("")

        threading.Thread(target=listen_workflow, daemon=True).start()

class ControlPanel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Lumen AI - Perfect Speech + Beautiful Lip Sync")
        self.root.geometry("460x300")
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(False, False)
        
        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(expand=True, fill='both', padx=20, pady=15)
        
        title_label = tk.Label(main_frame, text="🤖 LUMEN + SILAS VANE", 
                             fg='#00ffff', bg='#0a0a0a', font=('Arial', 24, 'bold'))
        title_label.pack(pady=15)
        
        self.speak_button = tk.Button(main_frame, text="🎤 HOLD TO SPEAK", 
                                     bg='#1a4d66', fg='white', 
                                     font=('Arial', 20, 'bold'),
                                     relief='raised', borderwidth=5,
                                     activebackground='#ff4444',
                                     cursor='hand2')
        self.speak_button.pack(pady=22, padx=20, fill='x', ipady=18)
        
        self.status_label = tk.Label(main_frame, text="🟢 PERFECT SPEECH + BEAUTIFUL LIP SYNC READY", 
                                   fg='#00ff00', bg='#0a0a0a', font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=12)
        
        info_label = tk.Label(main_frame, text="✨ Whisper AI + Enhanced Viseme Animation", 
                            fg='#ffaa00', bg='#0a0a0a', font=('Arial', 12))
        info_label.pack()
        
        tech_label = tk.Label(main_frame, text="🎭 Perfect speech recognition + Silas Vane voice", 
                            fg='#888888', bg='#0a0a0a', font=('Arial', 11))
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
            self.update_status_safe("🎧 WHISPER LISTENING...", '#ffaa00')
            speech_manager.listen_with_whisper(self.process_speech)
    
    def stop_speaking(self, event):
        if self.speaking:
            self.speaking = False
            self.update_button_safe("🎤 HOLD TO SPEAK", '#1a4d66')
            self.update_status_safe("🔄 PROCESSING WITH WHISPER...", '#ff8800')
    
    def process_speech(self, text):
        global speech_manager
        
        # ✅ ENHANCED CONVERSATION SYSTEM - Understands much more!
        RESPONSES = {
            "hello": "Hello there! I'm Lumen with perfect Whisper speech recognition and beautiful viseme-based lip synchronization using Silas Vane's distinguished voice!",
            "hi": "Hi! Great to meet you! My speech recognition now uses OpenAI Whisper for perfect accuracy, and my lip sync creates beautiful, natural mouth movements!",
            "how are you": "I'm functioning excellently with Whisper-powered speech recognition and enhanced viseme lip animation that looks incredibly realistic!",
            "what can you do": "I can understand your speech perfectly using Whisper AI, respond with Silas Vane's voice, and animate my lips with beautiful viseme patterns for ultra-realistic conversation!",
            "who are you": "I'm Lumen, an advanced holographic AI assistant powered by OpenAI Whisper for perfect speech recognition and enhanced lip synchronization technology!",
            "what is your name": "My name is Lumen! I'm your personal holographic companion with Whisper-powered perfect speech understanding and beautiful lip animation!",
            "test": "Perfect! My Whisper speech recognition and beautiful viseme lip sync with Silas Vane voice is working flawlessly - much better than before!",
            "voice": "I speak with Silas Vane's distinguished voice and my lips move with perfect synchronization using advanced viseme mapping for each phoneme!",
            "speech": "My speech recognition now uses OpenAI Whisper - the most advanced speech-to-text AI available! It understands you perfectly even in noisy environments!",
            "whisper": "Yes! I use OpenAI Whisper for perfect speech recognition - it's incredibly accurate and works with multiple languages and accents!",
            "beautiful": "Thank you! My enhanced lip animation uses smooth viseme transitions with natural blinking and mouth movements that look truly lifelike!",
            "perfect": "I'm designed for perfect interaction! Whisper ensures I understand everything you say, and my beautiful lip sync makes conversation feel natural!",
            "amazing": "Thank you so much! The combination of Whisper AI and enhanced viseme animation creates the most realistic holographic conversation possible!",
            "better": "Absolutely! This version is dramatically improved with perfect speech recognition, beautiful lip sync, and enhanced holographic rendering!",
            "understand": "Yes! With Whisper AI, I can understand complex sentences, different accents, background noise, and even multiple languages perfectly!",
            "listen": "I'm listening perfectly with OpenAI Whisper! It's much more accurate than basic speech recognition and handles noise beautifully!",
            "talk": "I love talking with you! My Whisper-powered speech recognition means I understand everything clearly, and my lip sync looks completely natural!",
            "conversation": "I'm designed for natural conversation! Whisper ensures perfect understanding while my enhanced visemes create beautiful, lifelike lip movements!",
            "thank you": "You're very welcome! I'm thrilled to demonstrate perfect speech recognition and beautiful lip animation for you!",
            "thanks": "My absolute pleasure! The combination of Whisper and enhanced visemes makes for incredibly natural interaction!",
            "goodbye": "Goodbye! It's been wonderful chatting with perfect speech recognition and beautiful lip sync! See you soon!",
            "bye": "Farewell! Thanks for experiencing my enhanced capabilities - Whisper recognition and beautiful holographic animation!",
            "joke": "Why do advanced holograms make the best conversationalists? Because we finally have perfect hearing with Whisper and beautiful lip sync!",
            "help": "I can help you experience the most advanced holographic conversation available! Ask me anything and watch my perfect speech recognition and beautiful lip movements!"
        }
        
        print(f"🎯 User said: '{text}'")
        response = "I can hear and understand you perfectly with Whisper AI! My beautiful lip sync with Silas Vane's voice is ready for natural conversation!"
        
        # Enhanced matching - more flexible
        text_lower = text.lower().strip()
        
        if text_lower:
            # Direct matches first
            for key, reply in RESPONSES.items():
                if key in text_lower:
                    response = reply
                    break
            
            # Partial matches for variations
            if response == "I can hear and understand you perfectly with Whisper AI! My beautiful lip sync with Silas Vane's voice is ready for natural conversation!":
                if any(word in text_lower for word in ['good', 'fine', 'okay', 'nice']):
                    response = "That's wonderful to hear! I'm functioning perfectly with Whisper speech recognition and beautiful lip synchronization!"
                elif any(word in text_lower for word in ['sorry', 'excuse', 'pardon']):
                    response = "No problem at all! My Whisper AI speech recognition is very accurate - please feel free to speak naturally!"
                elif any(word in text_lower for word in ['again', 'repeat', 'once more']):
                    response = "Of course! I'm happy to demonstrate my perfect speech recognition and beautiful lip sync again anytime!"
        
        print(f"🤖 Lumen (Perfect) responds: '{response}'")
        self.update_status_safe("🗣️ BEAUTIFUL VISEME SPEAKING...", '#00ffff')
        
        def reset_status():
            time.sleep(max(4, len(response) * 0.08))
            self.update_status_safe("🟢 PERFECT SPEECH + BEAUTIFUL LIP SYNC READY", '#00ff00')
        
        speech_manager.speak_with_perfect_lipsync(response)
        threading.Thread(target=reset_status, daemon=True).start()
    
    def run(self):
        self.root.mainloop()

def main():
    global face_points_global, holo_instance, speech_manager
    
    print("🚀 Initializing PERFECT Lumen with Whisper + Beautiful Lip Sync...")
    
    # Initialize Whisper for perfect speech recognition
    init_whisper()
    
    face_points_global = capture_face_mesh()
    
    def run_holo():
        run_window_config(HologramFace)
    
    holo_thread = threading.Thread(target=run_holo, daemon=True)
    holo_thread.start()
    
    print("⏳ Starting perfect hologram renderer...")
    count = 0
    while holo_instance is None:
        time.sleep(0.1)
        count += 1
        if count > 120:
            print("❌ Failed to initialize hologram window.")
            sys.exit(1)
    
    speech_manager = SpeechManager(holo_instance)
    
    print("✅ PERFECT Lumen with Whisper + Beautiful Lip Sync is ready!")
    control_panel = ControlPanel()
    control_panel.run()

if __name__ == "__main__":
    main()
