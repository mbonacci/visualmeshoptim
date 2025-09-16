import numpy as np
import torch

from visualmeshoptim.mesh_io import MeshData
from visualmeshoptim.optimizer import VisualMeshOptimizer
from visualmeshoptim.problem import MeshProblem


class QuadraticProblem(MeshProblem):
    def __init__(self, target: np.ndarray):
        self._target = torch.as_tensor(target, dtype=torch.double)
        super().__init__(device="cpu", dtype=torch.double)
        super().initialize()

    def load_mesh(self):
        vertices = np.zeros_like(self._target.numpy())
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        return MeshData(vertices=vertices, faces=faces)

    def prepare_data(self, data):
        return None

    def objective(self, vertices: torch.Tensor) -> torch.Tensor:
        diff = vertices - self._target.to(vertices.device)
        return torch.sum(diff**2)


def test_optimizer_converges_on_quadratic():
    target = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    problem = QuadraticProblem(target)
    optimizer = VisualMeshOptimizer(problem)
    optimizer.set_optimizer_parameter("optimizer", "Adam")
    optimizer.set_optimizer_parameter("learning_rate", 0.2)
    optimizer.set_optimizer_parameter("fix_centroid", False)
    result = optimizer.optimize_headless(max_iterations=200, tolerance=1e-8, verbose=False)
    final = problem.get_vertices().detach().cpu().numpy()
    assert result["grad_norm"] < 1e-3
    assert np.allclose(final, target, atol=1e-2)
