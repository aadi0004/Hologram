import sys
import threading
import numpy as np
import cv2
import mediapipe as mp
from pyrr import Matrix44
import moderngl
import moderngl_window as mglw
from moderngl_window import WindowConfig, run_window_config
import pyttsx3
import speech_recognition as sr

# Global face mesh points (populated after capture)
face_points_global = None

# ----------------- FACE MESH CAPTURE -----------------
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
            print(f"Captured {len(points)} facial points.")
            return points
        elif key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

# ----------------- 3D HOLOGRAM FACE -----------------
class HologramFace(WindowConfig):
    gl_version = (3, 3)
    title = "3D Holographic Face Assistant"
    window_size = (1200, 900)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global face_points_global
        self.mesh = face_points_global.copy()
        self.base_mesh = face_points_global.copy()
        self.vertex_count = len(self.mesh)
        self.rotation = 0.0
        self.time = 0.0
        self.lip_time = 0.0
        self.speaking = False

        # Last ~20 points cover the mouth region
        self.mouth_indices = range(self.vertex_count - 20, self.vertex_count)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                uniform float time;
                out float glow;
                void main() {
                    vec3 pos = in_position;
                    pos.x += sin(time * 3.0 + pos.y * 10.0) * 0.015;
                    pos.y += sin(time * 2.0 + pos.x * 6.0) * 0.01;
                    pos.z += sin(time * 4.0) * 0.02;
                    glow = 0.8 + 0.2 * sin(time * 7.0);
                    gl_Position = mvp * vec4(pos, 1.0);
                    gl_PointSize = 20.0 + 15.0 * glow;
                }
            """,
            fragment_shader="""
                #version 330
                in float glow;
                uniform float time;
                out vec4 fragColor;
                void main() {
                    vec2 coord = gl_PointCoord - vec2(0.5);
                    float dist = length(coord);
                    if (dist > 0.5) discard;
                    float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
                    float flicker = 0.9 + 0.1 * sin(time * 15.0);
                    vec3 color = vec3(0.1, 0.85, 1.0) * glow * flicker;
                    fragColor = vec4(color, alpha * 0.9);
                }
            """
        )

        self.vbo = self.ctx.buffer(self.mesh.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_position')

    def start_speaking(self):
        self.speaking = True

    def stop_speaking(self):
        self.speaking = False
        self.mesh[self.mouth_indices] = self.base_mesh[self.mouth_indices]

    def update_lips(self, delta):
        if self.speaking:
            self.lip_time += delta * 20.0
            amplitude = 0.1
            for i, idx in enumerate(self.mouth_indices):
                offset = amplitude * np.sin(self.lip_time + i)
                self.mesh[idx][1] = self.base_mesh[idx][1] + offset
        else:
            self.lip_time = 0

    def on_render(self, time, frame_time):
        self.time = time
        self.rotation += frame_time * 0.5
        self.update_lips(frame_time)

        model = Matrix44.from_y_rotation(self.rotation)
        view = Matrix44.look_at(
            eye=(0, 0, 3),
            target=(0, 0, 0),
            up=(0, 1, 0)
        )
        proj = Matrix44.perspective_projection(
            fovy=60.0,
            aspect=self.wnd.aspect_ratio,
            near=0.1,
            far=100.0
        )
        mvp = proj * view * model

        self.vbo.write(self.mesh.tobytes())
        self.prog['mvp'].write(mvp.astype('f4').tobytes())
        self.prog['time'].value = self.time

        self.ctx.clear(0, 0, 0, 1)
        # Use OpenGL constant directly
        self.vao.render(mode=moderngl.POINTS)

# ----------------- SPEECH MANAGER & MAIN LOGIC -----------------
class SpeechManager:
    def __init__(self, holo):
        self.tts = pyttsx3.init()
        self.holo = holo
        self.listening = False
        self.tts.connect('started-utterance', lambda name, loc, length: self.holo.start_speaking())
        self.tts.connect('finished-utterance', lambda name, completed: self.holo.stop_speaking())

    def speak(self, text):
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        self.tts.say(text)
        self.tts.runAndWait()

    def listen(self, on_text):
        if self.listening:
            print("Already listening...")
            return
        self.listening = True
        threading.Thread(target=self._listen_thread, args=(on_text,), daemon=True).start()

    def _listen_thread(self, on_text):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            audio = recognizer.listen(source, phrase_time_limit=5)
        text = ""
        try:
            text = recognizer.recognize_google(audio).lower()
            print(f"Recognized: {text}")
        except Exception as e:
            print(f"Recognition error: {e}")
        self.listening = False
        on_text(text)

def main():
    global face_points_global
    face_points_global = capture_face_mesh()

    def run_holo():
        run_window_config(HologramFace)

    # Launch OpenGL hologram in thread to keep prompt alive
    holo_thread = threading.Thread(target=run_holo, daemon=True)
    holo_thread.start()
    # Wait for window setup
    import time
    time.sleep(2)

    # Dummy object; .holo will be set by moderngl-window
    class HoloWrapper:
        def start_speaking(self): pass
        def stop_speaking(self): pass

    speech_manager = SpeechManager(HoloWrapper())

    # Simple loop to let user interact
    RESPONSES = {
        "hello": "Hello! How can I assist you?",
        "how are you": "I'm a holographic assistant, always ready.",
        "name": "I am Lumen, your holographic AI.",
        "joke": "Why did the hologram smile? Because it was looking sharp!",
        "bye": "Goodbye! Have a great day!"
    }

    # Patch in lips-moving to the real window after creation
    def patch_lipcontrol():
        # Find live configured window
        from moderngl_window import get_running_window_configs
        while True:
            configs = get_running_window_configs()
            if configs:
                real_holo = configs[0]
                speech_manager.holo = real_holo
                break
            time.sleep(0.1)
    patcher = threading.Thread(target=patch_lipcontrol, daemon=True)
    patcher.start()

    def process(text):
        print("User said:", text)
        response = "Sorry, I did not understand."
        for key in RESPONSES:
            if key in text:
                response = RESPONSES[key]
                break
        speech_manager.speak(response)

    while True:
        input("Press Enter and speak...")
        speech_manager.listen(process)

if __name__ == "__main__":
    main()
