# ActivationScope Multi-Agent System Core Blueprint

This repository is powered by an automated, multi-agent OpenCode system. This file acts as the invariant law of the repository. All agents must read and strictly adhere to these protocols.

---

## 1. Setup & Agent Roles

The AI engineering team consists of specialized local and cloud models configured under `.opencode/agents/`. Task allocation is determined dynamically by OpenCode's semantic router or explicitly by the `@orchestrator`.

### Agents
* **`@orchestrator`** (`qwen3.6-27b:q4`): The central operational hub. Breaks down multi-file objectives into step-by-step linear pipelines and drives execution.
* **`@architect`** (`qwen3.6-27b:q4`): High-cognitive anchor. Solves complex mathematical requirements, API boundary safety, and memory lifecycle design.
* **`@engineer`** (`qwen3.6-27b:q4`): The builder. Writes the raw Python, C++/ATen hooks, and Pybind11 bindings.
* **`@reviewer`** (`qwen3.6-27b:q4`): The gatekeeper. Evaluates engineer patches specifically looking for silent memory leaks or PyTorch graph retention traps.
* **`@tester`** (`qwen3.6-27b:q4`): The validator. Authors and fires automated test runners, verifying stability against the PyTorch matrix.
* **`@devops`** (`qwen3.6-27b:q4`): The environment manager. Tunes Conda recipes, setup scripts, and CUDA container execution.
* **`@docs`** (`qwen3.6-27b:q4`): The technical writer. Maintains inline documentation, docstrings, and architectural markdown records.

---

## 2. Universal Code Generation Constraints

No agent may bypass these architectural rules during file updates:
1. **Detach-on-Store Rule:** Forward hooks operating in `"store"` mode must capture an independent copy of the output via `out.detach().clone()`. This prevents the tracker from holding live references into PyTorch's autograd graph while still preserving tensor data for later inspection. Online-stats mode is unaffected — it only reads scalars/tensors and never retains activation copies.
2. **No-Grad Safety:** All inline C++ statistics modifications inside `csrc/` *must* run within a `torch::NoGradGuard no_grad;` block to ensure the tracker does not mutate or bloat the training graph.
3. **Memory Cleanup:** Activations accumulate across batches inside `track()`. The user calls `.clear()` **explicitly** when ready to reset (e.g., after reading out data). Context exit runs `.remove()` for full teardown — hooks detached, activations wiped — so nothing leaks past the block.
4. **Per-Element Reductions:** Online min/max/mean statistics reduce only over the batch dimension (dim 0). The resulting shape `[C, H, W]` (or `[C, SeqLen]`, etc.) is preserved across forward passes and accumulates element-wise running stats per layer component.

---

## 3. Communication & Handoff Protocols

- **Orchestrator is the default entry point.** Unless a specialist is addressed
  directly, every task starts at `@orchestrator`, which decomposes the work and
  dispatches steps to the right specialists via the `task` tool.
- **Agents do the work — on disk.** Builder agents (`@orchestrator`,
  `@architect`, `@engineer`, `@tester`, `@devops`, `@docs`) MUST apply their
  changes by using the `write`/`edit`/`bash` tools directly. Producing a text
  description of a change without writing it to disk does NOT count as done.
- **Verify, don't trust.** After any agent reports a change, the orchestrator
  (and reviewers) MUST read the file back to confirm the change actually landed
  and is correct. Reported intent is not verified work.
- **Isolation of Duties:** Each agent stays in its domain.
  - `@architect` designs and specifies; `@engineer` implements.
  - `@reviewer` audits read-only (`edit: deny`) and routes every fix to the
    responsible agent — code → `@engineer`, infra/CI → `@devops`,
    tests → `@tester`, docs → `@docs`. It never edits source itself.
  - `@tester` authors and runs tests but never patches library code; it reports
    failures back to the orchestrator for routing.
- **Cooperation loop:** dispatch → apply on disk → verify by reading → build/test
  → review → route fixes → re-verify. Repeat until the step is genuinely done.
- **No silent human gate.** Agents are authorized to edit files and run commands
  within the permissions defined in `.opencode/`. They should still summarize
  what they changed, but they act rather than wait for per-step approval.
