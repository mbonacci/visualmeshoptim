import queue

import numpy as np

from visualmeshoptim.utils import TunableParameter
from visualmeshoptim.visualizer import MeshVisualizer


class DummyMesh:
    def __init__(self):
        self.updated = 0
        self.quantities = {}

    def set_edge_width(self, width):
        self.width = width

    def set_edge_color(self, color):
        self.color = color

    def add_scalar_quantity(self, name, values, **kwargs):
        self.quantities[name] = np.array(values)

    def update_vertex_positions(self, verts):
        self.updated += 1
        self.latest = np.array(verts)


class DummyPolyscope:
    def __init__(self):
        self.user_callback = None
        self.mesh = DummyMesh()
        self.closed = False

    def set_max_fps(self, fps):
        self.fps = fps

    def set_enable_vsync(self, flag):
        self.vsync = flag

    def init(self):
        pass

    def register_surface_mesh(self, name, vertices, faces, smooth_shade=True):
        self.mesh = DummyMesh()
        return self.mesh

    def set_user_callback(self, callback):
        self.user_callback = callback

    def frame_tick(self):
        if self.user_callback:
            self.user_callback()
        self.closed = True

    def window_requests_close(self):
        return self.closed

    def shutdown(self):
        pass


class DummyImGui:
    def Begin(self, name, opened):
        return True, opened

    def End(self):
        pass

    def TextUnformatted(self, text):
        self.last_text = text

    def InputInt(self, name, value):
        return False, value

    def InputFloat(self, name, value):
        return False, value

    def Checkbox(self, name, value):
        return False, value

    def Button(self, label):
        return False

    def SameLine(self):
        pass

    def Separator(self):
        pass

    def NewLine(self):
        pass


def test_mesh_visualizer_updates_vertices():
    backend = (DummyPolyscope(), DummyImGui())
    viz = MeshVisualizer(backend=backend)
    vertex_queue: "queue.Queue[np.ndarray]" = queue.Queue()
    vertex_queue.put(np.ones((3, 3), dtype=np.float32))
    stats = {"iteration": 0, "objective": 0.0, "grad_norm": 0.0}
    params = {"alpha": TunableParameter(1.0)}
    viz.run(
        initial_vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        quantities={},
        vertex_queue=vertex_queue,
        stats=stats,
        problem_parameters=params,
        optimizer_parameters=params,
        on_start=lambda: None,
        on_stop=lambda: None,
        is_running=lambda: False,
        refresh_quantities=lambda: {},
        on_reset=None,
    )
    assert backend[0].mesh.updated >= 1
