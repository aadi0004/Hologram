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
import sounddevice as sd
import soundfile as sf
import time
import tkinter as tk
import queue

# ---- CONFIG ----
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_FILENAME = "input.wav"
TTS_FILENAME = "reply.wav"
ELEVENLABS_VOICE_ID = "NYkjXRso4QIcgWakN1Cr"  # Silas Vane voice

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
gui_queue = queue.Queue()  # ✅ Thread-safe GUI updates

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

def get_advanced_lip_indices():
    """Get comprehensive lip indices for ULTRA-REALISTIC lip sync"""
    # ✅ COMPREHENSIVE: All major lip landmarks for maximum realism
    OUTER_LIP = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
    INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88]
    UPPER_LIP = [61, 84, 17, 314, 405]
    LOWER_LIP = [17, 18, 200, 199, 175, 0, 13, 269, 270, 267, 271, 272]
    LIP_CORNERS = [61, 291, 39, 181, 184, 17, 314, 405]
    MOUTH_INTERIOR = [13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 415]
    
    # Combine all for maximum coverage
    all_indices = OUTER_LIP + INNER_LIP + UPPER_LIP + LOWER_LIP + LIP_CORNERS + MOUTH_INTERIOR
    unique_indices = list(set(idx for idx in all_indices if 0 <= idx < 468))
    
    return np.array(unique_indices, dtype=np.int32)

class HologramFace(WindowConfig):
    gl_version = (3, 3)
    title = "3D Holographic Face Assistant - ULTRA Realistic Lip Sync"
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
        self.lip_time = 0.0
        self.speaking = False
        
        # ✅ ULTRA-REALISTIC: Advanced lip animation system
        self.lip_indices = get_advanced_lip_indices()
        self.wireframe_indices = get_face_wireframe_indices()
        
        # ✅ PHONEME SIMULATION: Different mouth shapes for different sounds
        self.phoneme_patterns = {
            'A': {'mouth_open': 0.15, 'mouth_wide': 0.02},
            'E': {'mouth_open': 0.08, 'mouth_wide': 0.06},
            'I': {'mouth_open': 0.04, 'mouth_wide': 0.08},
            'O': {'mouth_open': 0.12, 'mouth_wide': -0.02},
            'U': {'mouth_open': 0.06, 'mouth_wide': -0.04},
            'M': {'mouth_open': 0.01, 'mouth_wide': 0.0},
            'P': {'mouth_open': 0.02, 'mouth_wide': 0.0},
            'F': {'mouth_open': 0.03, 'mouth_wide': 0.01},
            'S': {'mouth_open': 0.02, 'mouth_wide': 0.03}
        }
        
        print(f"✅ Using {len(self.lip_indices)} advanced lip indices for ultra-realistic sync")

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
        print("🗣️ Ultra-realistic lip animation started")
        self.speaking = True
        self.lip_time = 0.0

    def stop_speaking(self):
        print("🔇 Ultra-realistic lip animation stopped")
        self.speaking = False
        try:
            # Reset all lip vertices
            for idx in self.lip_indices:
                if 0 <= idx < len(self.mesh):
                    self.mesh[idx, 0] = float(self.base_mesh[idx, 0])
                    self.mesh[idx, 1] = float(self.base_mesh[idx, 1])
                    self.mesh[idx, 2] = float(self.base_mesh[idx, 2])
            
            self.vbo_points.write(self.mesh.tobytes())
            self.vbo_lines.write(self.mesh.tobytes())
        except Exception as e:
            print(f"Reset error: {e}")

    def update_ultra_realistic_lips(self, delta):
        """✅ ULTRA-REALISTIC: Advanced phoneme-based lip animation"""
        if not self.speaking:
            self.lip_time = 0.0
            return
            
        try:
            self.lip_time += float(delta * 25.0)
            
            # ✅ PHONEME SIMULATION: Cycle through different mouth shapes
            phoneme_cycle = int(self.lip_time * 2) % len(self.phoneme_patterns)
            current_phoneme = list(self.phoneme_patterns.keys())[phoneme_cycle]
            phoneme_data = self.phoneme_patterns[current_phoneme]
            
            for i, vertex_idx in enumerate(self.lip_indices):
                if not (0 <= vertex_idx < len(self.mesh)):
                    continue
                    
                # Get base position
                base_x = float(self.base_mesh[vertex_idx, 0])
                base_y = float(self.base_mesh[vertex_idx, 1])
                base_z = float(self.base_mesh[vertex_idx, 2])
                
                # ✅ ADVANCED: Multiple animation layers for ultra-realism
                
                # 1. Phoneme-specific movement
                phoneme_open = float(phoneme_data['mouth_open'] * np.sin(self.lip_time + i * 0.2))
                phoneme_wide = float(phoneme_data['mouth_wide'] * np.sin(self.lip_time * 1.1 + i * 0.15))
                
                # 2. Natural speech rhythm
                speech_rhythm = float(0.06 * np.sin(self.lip_time * 1.8 + i * 0.3))
                
                # 3. Articulation movements (tongue influence)
                articulation = float(0.03 * np.sin(self.lip_time * 2.2 + i * 0.25))
                
                # 4. Breathing patterns
                breathing = float(0.008 * np.sin(self.lip_time * 0.4 + i * 0.1))
                
                # 5. Natural asymmetry (humans aren't symmetric)
                asymmetry_x = float(0.004 * np.sin(self.lip_time * 1.3 + i * 0.7))
                asymmetry_y = float(0.002 * np.cos(self.lip_time * 1.5 + i * 0.8))
                
                # 6. Micro-expressions
                micro_expr = float(0.006 * np.sin(self.lip_time * 0.6 + i * 0.4))
                
                # ✅ COMBINE: All layers with natural weighting
                total_x = float(phoneme_wide + asymmetry_x)
                total_y = float(phoneme_open + speech_rhythm + articulation + breathing + asymmetry_y + micro_expr)
                total_z = float(0.02 * np.sin(self.lip_time * 0.7 + i * 0.2))
                
                # ✅ APPLY: Ultra-realistic movement
                self.mesh[vertex_idx, 0] = float(base_x + total_x)
                self.mesh[vertex_idx, 1] = float(base_y + total_y)
                self.mesh[vertex_idx, 2] = float(base_z + total_z)
            
            # Update buffers
            self.vbo_points.write(self.mesh.tobytes())
            self.vbo_lines.write(self.mesh.tobytes())
            
        except Exception as e:
            print(f"Ultra-realistic lip error: {e}")
            self.speaking = False

    def on_render(self, time, frame_time):
        try:
            self.time = time
            self.update_ultra_realistic_lips(frame_time)

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

            self.point_prog['mvp'].write(mvp.astype('f4').tobytes())
            self.point_prog['time'].value = self.time
            self.line_prog['mvp'].write(mvp.astype('f4').tobytes())
            self.line_prog['time'].value = self.time

            self.ctx.clear(0, 0, 0, 1)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            
            self.vao_lines.render(mode=moderngl.LINES)
            self.vao_points.render(mode=moderngl.POINTS)
            
        except Exception as e:
            print(f"Render error: {e}")

# ✅ IMPROVED ELEVENLABS FUNCTIONS
def elevenlabs_tts(text, outfile=TTS_FILENAME, voice_id=ELEVENLABS_VOICE_ID):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.8,
            "similarity_boost": 0.9,
            "style": 0.3,
            "use_speaker_boost": True
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            with open(outfile, 'wb') as f:
                f.write(response.content)
            return outfile
        else:
            print("❌ TTS failed:", response.text)
            return None
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return None

def elevenlabs_asr(audiofile_path):
    """✅ IMPROVED: Better English recognition with optimized settings"""
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    try:
        with open(audiofile_path, 'rb') as f:
            files = {'file': (audiofile_path, f, 'audio/wav')}
            # ✅ FORCE ENGLISH: Better recognition settings
            data = {
                'model_id': 'scribe_v1',
                'language': 'en',  # Force English
                'optimize_streaming_latency': 0,  # Better accuracy
                'voice_isolation': True  # Remove background noise
            }
            response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('text', '').strip()
            # ✅ CLEAN UP: Remove common transcription artifacts
            text = text.replace('(background noise)', '').replace('(music)', '').strip()
            return text
        else:
            print("❌ ASR failed:", response.text)
            return ""
    except Exception as e:
        print(f"❌ ASR Error: {e}")
        return ""

def record_audio(filename, duration=6, fs=AUDIO_SAMPLE_RATE):
    """✅ IMPROVED: Better audio quality for recognition"""
    try:
        print("🎤 Recording... Speak clearly in English now.")
        # ✅ BETTER SETTINGS: Higher quality recording
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=AUDIO_CHANNELS, dtype='float64')
        sd.wait()
        
        # ✅ NOISE REDUCTION: Simple noise gate
        audio_abs = np.abs(audio)
        noise_floor = np.percentile(audio_abs, 20)
        audio = np.where(audio_abs < noise_floor * 2, audio * 0.1, audio)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        sf.write(filename, audio, fs)
        print("✅ Recording complete.")
    except Exception as e:
        print(f"❌ Recording error: {e}")

def play_audio_clip(filename):
    try:
        data, samplerate = sf.read(filename)
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"❌ Audio playback error: {e}")

class SpeechManager:
    def __init__(self, holo):
        self.holo = holo
        self.listening = False

    def speak(self, text):
        def tts_and_play():
            try:
                self.holo.start_speaking()
                audio_file = elevenlabs_tts(text)
                if audio_file and os.path.exists(audio_file):
                    play_audio_clip(audio_file)
                else:
                    time.sleep(max(2, len(text) * 0.08))
                self.holo.stop_speaking()
            except Exception as e:
                print(f"Speech error: {e}")
                self.holo.stop_speaking()
        
        threading.Thread(target=tts_and_play, daemon=True).start()

    def listen(self, on_text):
        if self.listening:
            print("🔄 Already listening...")
            return
        
        self.listening = True

        def asr_workflow():
            try:
                record_audio(AUDIO_FILENAME, duration=6)  # Longer recording
                recognized = elevenlabs_asr(AUDIO_FILENAME)
                print(f"👤 User said: '{recognized}'")
                self.listening = False
                on_text(recognized if recognized else "")
            except Exception as e:
                print(f"❌ Listen error: {e}")
                self.listening = False
                on_text("")

        threading.Thread(target=asr_workflow, daemon=True).start()

class ControlPanel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Lumen AI - ULTRA Realistic Lip Sync")
        self.root.geometry("420x270")
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(False, False)
        
        main_frame = tk.Frame(self.root, bg='#0a0a0a')
        main_frame.pack(expand=True, fill='both', padx=20, pady=15)
        
        title_label = tk.Label(main_frame, text="🤖 LUMEN + SILAS VANE", 
                             fg='#00ffff', bg='#0a0a0a', font=('Arial', 20, 'bold'))
        title_label.pack(pady=15)
        
        self.speak_button = tk.Button(main_frame, text="🎤 HOLD TO SPEAK ENGLISH", 
                                     bg='#1a4d66', fg='white', 
                                     font=('Arial', 16, 'bold'),
                                     relief='raised', borderwidth=4,
                                     activebackground='#ff4444',
                                     cursor='hand2')
        self.speak_button.pack(pady=18, padx=20, fill='x', ipady=12)
        
        self.status_label = tk.Label(main_frame, text="🟢 ULTRA-REALISTIC LIP SYNC READY", 
                                   fg='#00ff00', bg='#0a0a0a', font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=10)
        
        info_label = tk.Label(main_frame, text="Phoneme-based ultra-realistic lip animation", 
                            fg='#888888', bg='#0a0a0a', font=('Arial', 10))
        info_label.pack()
        
        lang_label = tk.Label(main_frame, text="✨ Optimized for clear English speech", 
                            fg='#ffaa00', bg='#0a0a0a', font=('Arial', 10))
        lang_label.pack(pady=5)
        
        self.speak_button.bind('<Button-1>', self.start_speaking)
        self.speak_button.bind('<ButtonRelease-1>', self.stop_speaking)
        
        self.speaking = False
        
        # ✅ THREAD-SAFE: Process GUI updates safely
        self.process_gui_updates()
        
    def process_gui_updates(self):
        """✅ FIXED: Thread-safe GUI updates"""
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
        # Schedule next check
        self.root.after(100, self.process_gui_updates)
        
    def update_status_safe(self, text, color):
        """✅ THREAD-SAFE: Safe status updates"""
        gui_queue.put(('status', {'text': text, 'color': color}))
        
    def update_button_safe(self, text, color):
        """✅ THREAD-SAFE: Safe button updates"""
        gui_queue.put(('button', {'text': text, 'color': color}))
    
    def start_speaking(self, event):
        global speech_manager
        if not self.speaking and speech_manager:
            self.speaking = True
            self.update_button_safe("🔴 RECORDING...", '#cc2222')
            self.update_status_safe("🎧 LISTENING FOR ENGLISH...", '#ffaa00')
            speech_manager.listen(self.process_speech)
    
    def stop_speaking(self, event):
        if self.speaking:
            self.speaking = False
            self.update_button_safe("🎤 HOLD TO SPEAK ENGLISH", '#1a4d66')
            self.update_status_safe("🔄 PROCESSING...", '#ff8800')
    
    def process_speech(self, text):
        global speech_manager
        
        # ✅ ENHANCED RESPONSES: Better conversation with more variety
        RESPONSES = {
            "hello": "Hello! I'm Lumen with ultra-realistic lip sync using phoneme-based animation and Silas Vane's distinguished voice!",
            "hi": "Hi there! My lip movements now simulate real phonemes like A, E, I, O, U sounds with natural breathing and asymmetry!",
            "how are you": "I'm functioning perfectly with advanced lip synchronization that includes speech rhythm, articulation, and micro-expressions!",
            "test": "Perfect! My ultra-realistic lip sync with Silas Vane voice now uses phoneme simulation for incredibly lifelike speech!",
            "voice": "I use Silas Vane's voice with phoneme-based lip animation that creates different mouth shapes for different speech sounds!",
            "lips": "My lips now move with ultra-realistic patterns including vowel shapes, consonant articulation, breathing, and natural asymmetry!",
            "realistic": "Absolutely! I simulate real human speech with phoneme patterns, rhythm variations, articulation movements, and breathing cycles!",
            "phoneme": "Yes! I use phoneme-based animation with different mouth shapes for A, E, I, O, U, M, P, F, and S sounds - just like humans!",
            "amazing": "Thank you! My ultra-realistic lip sync combines phoneme simulation, speech rhythm, articulation, breathing, and micro-expressions!",
            "natural": "My lip animation now includes natural asymmetry, breathing patterns, micro-expressions, and phoneme-specific mouth shapes!",
            "english": "Perfect! I'm optimized for English speech recognition with noise reduction and better transcription accuracy!",
            "understand": "Yes! I can now understand English much better with improved voice isolation and optimized recognition settings!",
            "name": "I'm Lumen, your ultra-realistic holographic AI with advanced phoneme-based lip sync and Silas Vane's voice!",
            "joke": "Why do ultra-realistic holograms make the best speakers? Because we finally have perfect phoneme synchronization!"
        }
        
        print(f"👤 User said: '{text}'")
        response = "I can hear you speaking English! My ultra-realistic lip sync with phoneme-based animation is ready to demonstrate natural speech patterns!"
        text_lower = text.lower().strip()
        
        if text_lower:
            for key, reply in RESPONSES.items():
                if key in text_lower:
                    response = reply
                    break
        
        print(f"🤖 Lumen (Ultra-Realistic) responds: '{response}'")
        self.update_status_safe("🗣️ ULTRA-REALISTIC SPEAKING...", '#00ffff')
        
        def reset_status():
            time.sleep(max(4, len(response) * 0.08))
            self.update_status_safe("🟢 ULTRA-REALISTIC LIP SYNC READY", '#00ff00')
        
        speech_manager.speak(response)
        threading.Thread(target=reset_status, daemon=True).start()
    
    def run(self):
        self.root.mainloop()

def main():
    global face_points_global, holo_instance, speech_manager
    
    print("🚀 Initializing ULTRA-REALISTIC Lumen with phoneme-based lip sync...")
    
    face_points_global = capture_face_mesh()
    
    def run_holo():
        run_window_config(HologramFace)
    
    holo_thread = threading.Thread(target=run_holo, daemon=True)
    holo_thread.start()
    
    print("⏳ Starting ultra-realistic hologram renderer...")
    count = 0
    while holo_instance is None:
        time.sleep(0.1)
        count += 1
        if count > 120:
            print("❌ Failed to initialize hologram window.")
            sys.exit(1)
    
    speech_manager = SpeechManager(holo_instance)
    
    print("✅ ULTRA-REALISTIC Lumen with phoneme-based lip sync is ready!")
    control_panel = ControlPanel()
    control_panel.run()

if __name__ == "__main__":
    main()
