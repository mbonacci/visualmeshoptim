"""Optimization entry point for :mod:`visualmeshoptim`."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

from .problem import MeshProblem
from .utils import TunableParameter, get_logger


Callback = Callable[[int, float, np.ndarray, float], None]


@dataclass
class _IterationStats:
    iteration: int
    objective_value: float
    grad_norm: float


class VisualMeshOptimizer:
    """Run optimization in headless or interactive mode."""

    def __init__(
        self,
        problem: MeshProblem,
        *,
        visualizer: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> None:
        if not problem.is_initialized():
            problem.initialize()
        self.problem = problem
        self.visualizer = visualizer
        self.logger = logger or get_logger()
        self.callbacks: List[Callback] = []
        self.optimizer_parameters: Dict[str, TunableParameter] = {
            "optimizer": TunableParameter("LBFGS", dtype="choice", options=["LBFGS", "Adam"], description="Optimization algorithm."),
            "learning_rate": TunableParameter(1e-2, dtype="float", min_value=1e-6, max_value=1.0, description="Step size for the optimizer."),
            "max_iterations": TunableParameter(200, dtype="int", min_value=1, description="Maximum number of iterations."),
            "tolerance": TunableParameter(1e-6, dtype="float", min_value=1e-12, description="Stopping tolerance on gradient norm."),
            "fix_centroid": TunableParameter(True, dtype="bool", description="Keep centroid anchored to the initial location."),
            "lbfgs_history": TunableParameter(40, dtype="int", min_value=1, max_value=200, description="History size for LBFGS."),
        }

    # ------------------------------------------------------------------
    def add_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def set_optimizer_parameter(self, name: str, value: Any) -> None:
        if name not in self.optimizer_parameters:
            raise KeyError(name)
        self.optimizer_parameters[name].update(value)

    def get_optimizer_parameter(self, name: str) -> TunableParameter:
        return self.optimizer_parameters[name]

    # ------------------------------------------------------------------
    def optimize_headless(
        self,
        *,
        max_iterations: Optional[int] = None,
        learning_rate: Optional[float] = None,
        tolerance: Optional[float] = None,
        optimizer_type: Optional[str] = None,
        callbacks: Optional[Iterable[Callback]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute the optimization loop without visualization."""
        options = self._collect_options(
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            tolerance=tolerance,
            optimizer_type=optimizer_type,
        )
        return self._run_optimization(
            options,
            callbacks=list(callbacks or []) + self.callbacks,
            stop_event=None,
            enable_logging=verbose,
        )

    # ------------------------------------------------------------------
    def optimize_interactive(self) -> None:
        if self.visualizer is None:
            from .visualizer import MeshVisualizer

            self.visualizer = MeshVisualizer()

        problem_params = self.problem.parameters()
        opt_params = self.optimizer_parameters

        initial_vertices = self.problem.get_vertices().detach().cpu().numpy()
        faces = self.problem.get_faces().detach().cpu().numpy()
        quantities = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in self.problem.get_visualization_quantities().items()
        }

        vertex_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        stats = {"iteration": 0, "objective": 0.0, "grad_norm": 0.0}
        run_state = {"running": False, "thread": None, "stop_event": threading.Event()}

        def _step_callback(iter_idx: int, value: float, verts_np: np.ndarray, grad_norm: float) -> None:
            stats["iteration"] = iter_idx
            stats["objective"] = value
            stats["grad_norm"] = grad_norm
            try:
                if vertex_queue.full():
                    vertex_queue.get_nowait()
                vertex_queue.put_nowait(verts_np.astype(np.float32, copy=False))
            except queue.Full:
                pass

        def start_requested() -> None:
            if run_state["running"]:
                return
            run_state["stop_event"].clear()

            def runner() -> None:
                try:
                    options = self._collect_options()
                    self._run_optimization(
                        options,
                        callbacks=[_step_callback] + list(self.callbacks),
                        stop_event=run_state["stop_event"],
                        enable_logging=False,
                    )
                finally:
                    run_state["running"] = False
                    run_state["thread"] = None

            thread = threading.Thread(target=runner, daemon=True)
            thread.start()
            run_state["thread"] = thread
            run_state["running"] = True

        def stop_requested() -> None:
            if run_state["running"]:
                run_state["stop_event"].set()

        def is_running() -> bool:
            return run_state["running"]

        def refresh_quantities() -> Dict[str, np.ndarray]:
            return {
                name: tensor.detach().cpu().numpy()
                for name, tensor in self.problem.get_visualization_quantities().items()
            }

        def reset_requested() -> None:
            if run_state["running"]:
                return
            self.problem.reset_vertices()
            refreshed = self.problem.get_vertices().detach().cpu().numpy().astype(np.float32, copy=False)
            stats["iteration"] = 0
            stats["objective"] = 0.0
            stats["grad_norm"] = 0.0
            try:
                while not vertex_queue.empty():
                    vertex_queue.get_nowait()
                vertex_queue.put_nowait(refreshed)
            except queue.Full:
                pass

        self.visualizer.run(  # type: ignore[call-arg]
            initial_vertices=initial_vertices,
            faces=faces,
            quantities=quantities,
            vertex_queue=vertex_queue,
            stats=stats,
            problem_parameters=problem_params,
            optimizer_parameters=opt_params,
            on_start=start_requested,
            on_stop=stop_requested,
            is_running=is_running,
            refresh_quantities=refresh_quantities,
            on_reset=reset_requested,
        )

    # ------------------------------------------------------------------
    def _collect_options(
        self,
        *,
        max_iterations: Optional[int] = None,
        learning_rate: Optional[float] = None,
        tolerance: Optional[float] = None,
        optimizer_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        opts = {name: param.value for name, param in self.optimizer_parameters.items()}
        if max_iterations is not None:
            opts["max_iterations"] = int(max_iterations)
        if learning_rate is not None:
            opts["learning_rate"] = float(learning_rate)
        if tolerance is not None:
            opts["tolerance"] = float(tolerance)
        if optimizer_type is not None:
            opts["optimizer"] = optimizer_type
        return opts

    def _apply_constraints(self, parameter: torch.nn.Parameter, *, fix_centroid: bool, initial_centroid: torch.Tensor) -> None:
        with torch.no_grad():
            data = parameter.data
            if fix_centroid:
                data -= data.mean(dim=0)
                data += initial_centroid
            constrained = self.problem.apply_constraints(data)
            if constrained is not data:
                constrained = torch.as_tensor(constrained, device=data.device, dtype=data.dtype)
                data.copy_(constrained)

    def _run_optimization(
        self,
        options: Dict[str, Any],
        *,
        callbacks: Iterable[Callback],
        stop_event: Optional[threading.Event],
        enable_logging: bool,
    ) -> Dict[str, Any]:
        self.problem.require_initialized()
        vertices = torch.nn.Parameter(self.problem.get_vertices().detach().clone())
        vertices = vertices.to(device=self.problem.device, dtype=self.problem.dtype)
        optimizer_name = str(options.get("optimizer", "LBFGS")).lower()
        lr = float(options.get("learning_rate", 1e-2))
        max_iters = int(options.get("max_iterations", 200))
        tol = float(options.get("tolerance", 1e-6))
        fix_centroid = bool(options.get("fix_centroid", True))
        history = int(options.get("lbfgs_history", 40))
        if optimizer_name == "lbfgs":
            opt = torch.optim.LBFGS([vertices], lr=lr, history_size=history, line_search_fn="strong_wolfe")
        elif optimizer_name == "adam":
            opt = torch.optim.Adam([vertices], lr=lr)
        else:
            raise ValueError(f"Unknown optimizer '{optimizer_name}'")

        initial_centroid = vertices.detach().mean(dim=0)
        grad_norm = float("inf")
        final_value = float("nan")
        stats = _IterationStats(iteration=0, objective_value=0.0, grad_norm=float("inf"))

        for iteration in range(1, max_iters + 1):
            if stop_event is not None and stop_event.is_set():
                break

            def closure() -> torch.Tensor:
                opt.zero_grad()
                self._apply_constraints(vertices, fix_centroid=fix_centroid, initial_centroid=initial_centroid)
                value = self.problem.objective(vertices)
                value.backward()
                return value

            if optimizer_name == "lbfgs":
                opt.step(closure)
            else:
                closure()
                opt.step()

            self._apply_constraints(vertices, fix_centroid=fix_centroid, initial_centroid=initial_centroid)
            opt.zero_grad()
            value = self.problem.objective(vertices)
            value.backward()
            grad = vertices.grad.detach()
            grad_norm = float(torch.linalg.vector_norm(grad)) if grad is not None else 0.0
            final_value = float(value.detach().cpu().numpy())
            stats = _IterationStats(iteration=iteration, objective_value=final_value, grad_norm=grad_norm)

            verts_np = vertices.detach().cpu().numpy()
            if enable_logging:
                self.logger.info("iter %d | obj=%.6g | grad=%.3e", iteration, final_value, grad_norm)
            self.problem.on_step(iteration, vertices.detach(), value.detach())
            for cb in callbacks:
                cb(iteration, final_value, verts_np, grad_norm)
            if grad_norm < tol:
                break

        self.problem.update_vertices(vertices.detach())
        return {"iterations": stats.iteration, "objective": stats.objective_value, "grad_norm": stats.grad_norm}
