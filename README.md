# visualmeshoptim

`visualmeshoptim` ia an attempt to create a reusable
mesh-optimization toolkit.  It provides a thin abstraction layer (`MeshProblem`)
for describing geometry-driven objectives, an optimization engine that runs in
headless or interactive modes, and optional Polyscope visualization helpers.

This library is built using AI coding agents, and is intended to be a POC for 
using AI to accelerate scientific software development as well as useful tool for
physics students. It is not yet a mature library, but may be useful for prototyping 
new mesh optimization problems.


## Features

- **Problem abstraction** – Subclass `MeshProblem` to describe how a mesh is
  loaded, which auxiliary tensors are required, and how the scalar objective is
  evaluated.
- **Configurable optimization loop** – `VisualMeshOptimizer` handles device
  management, supports both LBFGS and Adam optimizers, and exposes tunable
  parameters through callbacks and the interactive UI.
- **Visualization integration** – `MeshVisualizer` encapsulates Polyscope
  registration, updating vertex buffers in real time and exposing problem and
  optimizer parameters through ImGui panels.
- **Reusable utilities** – Mesh I/O helpers, Blender export tooling, and
  geometry utilities simplify preparing datasets for new problems.
- **Examples and tests** – The `examples` package ports the historical pollen
  shell problem; the test suite validates the core abstractions and optimizer.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[visual,io]
pytest
```

## Creating a new problem

1. Subclass `MeshProblem` and implement `load_mesh`, `prepare_data`, and
   `objective`.
2. Instantiate your problem and pass it to `VisualMeshOptimizer`.
3. Call `optimize_headless()` for scripted runs or `optimize_interactive()` to
   launch the Polyscope UI (requires the optional `visual` dependency group).

See `src/visualmeshoptim/examples/pollen_shell.py` for a complete example.
