import cv2
import mediapipe as mp
import numpy as np
import moderngl
import moderngl_window as mglw
from pyrr import Matrix44, Vector3

# ---------- Step 1: Capture face landmarks ----------
def capture_face_mesh():
    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    print("Press [SPACE] to capture your face, [Q] to quit...")

    mesh_points = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)

        # Draw landmarks
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks[0].landmark:
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACE = capture
            if results.multi_face_landmarks:
                mesh_points = np.array([
                    [lm.x - 0.5, -(lm.y - 0.5), lm.z]  # shift center, flip Y
                    for lm in results.multi_face_landmarks[0].landmark
                ])
                print("✅ Face captured! Closing camera...")
                break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return mesh_points

# ---------- Step 2: Hologram renderer ----------
class HoloFaceApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Hologram Face"
    resource_dir = '.'
    window_size = (900, 700)
    aspect_ratio = window_size[0] / window_size[1]
    vsync = True
    samples = 4

    def __init__(self, mesh_points, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

        # Load shaders
        with open('shaders/hologram.vert') as f:
            vs_src = f.read()
        with open('shaders/hologram.frag') as f:
            fs_src = f.read()
        self.prog = self.ctx.program(vertex_shader=vs_src, fragment_shader=fs_src)

        # Fake normals
        normals = np.tile([0, 0, 1], (len(mesh_points), 1))
        vbo_pos = self.ctx.buffer(mesh_points.astype('f4').tobytes())
        vbo_nrm = self.ctx.buffer(normals.astype('f4').tobytes())

        self.vao = self.ctx.vertex_array(self.prog, [
            (vbo_pos, '3f', 'in_position'),
            (vbo_nrm, '3f', 'in_normal'),
        ])

        # Camera setup
        self.zoom = 2.5
        self.theta = 0.0
        self.time_accum = 0.0

        # Shader uniforms
        self.prog['HoloColor'].value = (0.2, 0.9, 1.0)  # cyan glow
        self.prog['Alpha'].value = 0.8
        self.prog['ScanFreq'].value = 60.0
        self.prog['FlickerSpeed'].value = 3.5
        self.prog['FresnelPower'].value = 2.5

    def on_render(self, time: float, frame_time: float):
        self.time_accum = time
        self.ctx.clear(0.02, 0.02, 0.04)

        # Rotate slowly
        self.theta += frame_time * 0.4
        rot = Matrix44.from_y_rotation(self.theta, dtype='f4')
        model = rot

        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.05, 100.0, dtype='f4')
        view = Matrix44.look_at((0.0, 0.0, self.zoom), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), dtype='f4')
        mvp = proj * view * model

        # Update uniforms
        self.prog['Mvp'].write(mvp.astype('f4').tobytes())
        self.prog['Model'].write(model.astype('f4').tobytes())
        self.prog['CamPos'].value = (0.0, 0.0, self.zoom)
        self.prog['Time'].value = self.time_accum

        # Render as glowing points
        self.vao.render(moderngl.POINTS)

# ---------- Step 3: Run ----------
if __name__ == "__main__":
    points = capture_face_mesh()
    if points is not None:
        print("Launching hologram window...")

        # Create a subclass that injects mesh_points
        class App(HoloFaceApp):
            def __init__(self, **kwargs):
                super().__init__(points, **kwargs)

        mglw.run_window_config(App)

    else:
        print("❌ No face captured.")

