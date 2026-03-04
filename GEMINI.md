# Agent Guidelines

<!-- Do not restructure or delete sections. Update individual values in-place when they change. -->

## Core Principles

- **Do NOT maintain backward compatibility** unless explicitly requested. Break things boldly.
- **Keep this file under 20-30 lines of instructions.** Every line competes for the agent's limited context budget (~150-200 total).

---

## Project Overview

<!-- Update this section as the project takes shape -->

**Project type:** Machine Learning / Federated Learning (IFCA)
**Primary language:** Python 3.13+
**Key dependencies:** `torch`, `torchvision`, managed by `uv`.

---

## Commands

<!-- Update these as your workflow evolves - commands change frequently -->

```bash
# Execution
uv run <Python script>

# Dependency management
uv add <package>

# If uv is not available
mise trust
```

---

## Code Conventions

<!-- Keep this minimal - let tools like linters handle formatting -->

- Follow the existing patterns in the codebase
- Prefer explicit over clever
- Delete dead code immediately

---

## Architecture

<!-- Major architecture changes MUST trigger a rewrite of this section -->

```
[Add directory structure overview when it stabilizes]
```

---

## Maintenance Notes

<!-- This section is permanent. Do not delete. -->

**Keep this file lean and current:**

1. **Remove placeholder sections** (sections still containing `[To be determined]` or `[Add your ... here]`) once you fill them in
2. **Review regularly** - stale instructions poison the agent's context
3. **CRITICAL: Keep total under 20-30 lines** - move detailed docs to separate files and reference them
4. **Update commands immediately** when workflows change
5. **Rewrite Architecture section** when major architectural changes occur
6. **Delete anything the agent can infer** from your code

**Remember:** Coding agents learn from your actual code. Only document what's truly non-obvious or critically important.
