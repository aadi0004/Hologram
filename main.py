import cv2
import mediapipe as mp
import numpy as np
import moderngl
import moderngl_window as mglw
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

mp_face_mesh = mp.solutions.face_mesh

# Global storage for face points
FACE_POINTS = None


def capture_face_mesh():
    cap = cv2.VideoCapture(0)
    print("Position your face. Press SPACE to capture, Q to quit.")
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pts = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                pts = []
                face = results.multi_face_landmarks[0]
                for lm in face.landmark:
                    x = (lm.x - 0.5) * 2.0   # scale up
                    y = -(lm.y - 0.5) * 2.0
                    z = lm.z * 0.8
                    pts.append([x, y, z])
                print(f"[✅] Captured {len(pts)} landmarks")
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return np.array(pts, dtype=np.float32) if pts is not None else None


class App(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "User Hologram"
    window_size = (800, 800)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        global FACE_POINTS
        pts = FACE_POINTS
        if pts is None:
            raise RuntimeError("No face points loaded!")

        self.prog = self.load_program(
            vertex_shader="shaders/hologram.vert",
            fragment_shader="shaders/hologram.frag"
        )

        # VBO
        self.vbo = self.ctx.buffer(pts.astype("f4").tobytes())

        # Index buffer for lines
        indices = []
        for a, b in FACEMESH_TESSELATION:
            indices += [a, b]
        self.ibo = self.ctx.buffer(np.array(indices, dtype=np.int32).tobytes())

        self.vao_points = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_position")
        self.vao_lines = self.ctx.vertex_array(self.prog, [(self.vbo, "3f", "in_position")], self.ibo)

        if "HoloColor" in self.prog:
            self.prog["HoloColor"].value = (0.0, 1.0, 1.0)
        if "Alpha" in self.prog:
            self.prog["Alpha"].value = 0.9

        self.ctx.point_size = 5.0

    def render(self, time, frame_time):
        self.ctx.clear(0.0, 0.0, 0.05)
        mvp = np.eye(4, dtype="f4")
        if "Mvp" in self.prog:
            self.prog["Mvp"].write(mvp.tobytes())

        self.vao_points.render(moderngl.POINTS)
        self.vao_lines.render(moderngl.LINES)


def main():
    print("[Main] Starting...")
    global FACE_POINTS
    FACE_POINTS = capture_face_mesh()
    if FACE_POINTS is None:
        print("[❌] No face captured")
        return

    print("[App] Launching hologram window...")
    mglw.run_window_config(App)


if __name__ == "__main__":
    main()
