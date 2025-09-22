"""Interactive example: optimize a pollen shell in Polyscope.

Run with:
    python -m visualmeshoptim.examples.example_pollen [--mesh PATH]

The viewer opens with panels to tweak problem and optimizer parameters,
and start/stop the optimization thread.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import torch

from visualmeshoptim.examples.pollen_shell import PollenShellProblem
from visualmeshoptim import VisualMeshOptimizer


def _default_mesh_path() -> Path:
    """Select a reasonable default mesh file in the current directory."""
    cwd = Path.cwd()
    for name in ("pollen.pt", "pollen.npz", "pollen.obj", "pollen.off", "pollen.stl"):
        candidate = cwd / name
        if candidate.exists():
            return candidate
    # Fallback: look one level up (useful when running from package dir)
    for name in ("pollen.pt", "pollen.npz"):
        candidate = cwd.parent / name
        if candidate.exists():
            return candidate
    return cwd / "pollen.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive pollen shell optimization example")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=_default_mesh_path(),
        help="Path to mesh file (.pt, .npz, .obj, .off, .stl)",
    )
    parser.add_argument("--edge-k", type=float, default=1.0, help="Edge stiffness scale factor")
    parser.add_argument("--bend-k", type=float, default=0.1, help="Bending stiffness scale factor")
    parser.add_argument(
        "--rest-bend-from-mesh",
        action="store_true",
        help="Use current mesh dihedrals as rest angles (default: false)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Torch device (default: auto)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Mesh not found: {mesh_path}")
        return 2

    problem = PollenShellProblem(
        str(mesh_path),
        edge_k_scale=float(args.edge_k),
        bend_k_scale=float(args.bend_k),
        rest_bend_from_mesh=bool(args.rest_bend_from_mesh),
        device=str(args.device),
        dtype=torch.double,
    )

    optimizer = VisualMeshOptimizer(problem)
    try:
        optimizer.optimize_interactive()
    except RuntimeError as exc:
        # Likely Polyscope not available
        print(f"Error starting interactive viewer: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

