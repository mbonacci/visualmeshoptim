"""Example: optimize a cloth draping configuration using a mass-spring model.

Run with:
    python -m visualmeshoptim.examples.example_cloth_drape [--no-visual]
"""
from __future__ import annotations

import argparse

import torch

from visualmeshoptim import VisualMeshOptimizer
from visualmeshoptim.examples.cloth_drape import ClothDrapeProblem


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static cloth draping example")
    parser.add_argument("--rows", type=int, default=20, help="Number of vertices along the vertical direction")
    parser.add_argument("--cols", type=int, default=30, help="Number of vertices along the horizontal direction")
    parser.add_argument("--spacing", type=float, default=0.05, help="Rest spacing between neighbouring vertices")
    parser.add_argument("--mass", type=float, default=1.0, help="Mass assigned to each vertex")
    parser.add_argument("--stretch-k", type=float, default=250.0, help="Structural spring stiffness")
    parser.add_argument("--bend-k", type=float, default=25.0, help="Bending spring stiffness")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravitational acceleration (positive scalar)")
    parser.add_argument("--no-diagonals", action="store_true", help="Disable diagonal structural springs")
    parser.add_argument("--no-bending", action="store_true", help="Disable bending springs")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Torch device used for the simulation",
    )
    parser.add_argument("--no-visual", action="store_true", help="Run headless without launching Polyscope")
    parser.add_argument("--optimizer", type=str, default="LBFGS", choices=["LBFGS", "Adam"], help="Optimizer backend")
    parser.add_argument("--learning-rate", type=float, default=5e-2, help="Learning rate for the optimizer")
    parser.add_argument("--max-iterations", type=int, default=200, help="Maximum number of optimization iterations")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Gradient norm stopping tolerance")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    problem = ClothDrapeProblem(
        rows=int(args.rows),
        cols=int(args.cols),
        spacing=float(args.spacing),
        vertex_mass=float(args.mass),
        include_diagonals=not bool(args.no_diagonals),
        include_bending=not bool(args.no_bending),
        stretch_stiffness=float(args.stretch_k),
        bend_stiffness=float(args.bend_k),
        gravity=float(args.gravity),
        device=str(args.device),
        dtype=torch.double,
        fixed_indices=[0, args.cols-1],
    )

    optimizer = VisualMeshOptimizer(problem)
    if args.no_visual:
        result = optimizer.optimize_headless(
            optimizer_type=str(args.optimizer),
            learning_rate=float(args.learning_rate),
            max_iterations=int(args.max_iterations),
            tolerance=float(args.tolerance),
        )
        print("Optimization finished:")
        print(f"  iterations : {result['iterations']}")
        print(f"  objective  : {result['objective']:.6g}")
        print(f"  grad_norm  : {result['grad_norm']:.3e}")
        return 0

    try:
        optimizer.optimize_interactive()
    except RuntimeError as exc:
        print(f"Error starting interactive viewer: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
