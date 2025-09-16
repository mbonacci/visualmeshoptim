import numpy as np
import pytest
import torch

from visualmeshoptim.mesh_io import MeshData
from visualmeshoptim.problem import MeshProblem


def test_mesh_problem_requires_implementation():
    class IncompleteProblem(MeshProblem):
        def load_mesh(self):
            return MeshData(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

    with pytest.raises(TypeError):
        IncompleteProblem()


def test_mesh_problem_initialization_and_reset():
    class SimpleProblem(MeshProblem):
        def __init__(self):
            self._calls = {"prepare": 0, "objective": 0}
            super().__init__(device="cpu", dtype=torch.double)
            super().initialize()

        def load_mesh(self):
            vertices = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
            return MeshData(vertices=vertices, faces=faces)

        def prepare_data(self, data):
            self._calls["prepare"] += 1

        def objective(self, vertices):
            self._calls["objective"] += 1
            return torch.sum(vertices**2)

    problem = SimpleProblem()
    assert problem.is_initialized()
    assert problem.get_vertices().shape == (4, 3)
    assert problem._calls["prepare"] == 1
    # mutate vertices and ensure reset restores them
    initial = problem.get_vertices().clone()
    mutated = initial + 1.0
    problem.update_vertices(mutated)
    problem.reset_vertices()
    assert torch.allclose(problem.get_vertices(), initial)
