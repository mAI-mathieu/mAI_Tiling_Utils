# Codex instructions for this ComfyUI custom node pack

This repository is a ComfyUI custom node pack.

## Scope

Codex must work only inside this repository.
Do not modify ComfyUI core files.
Do not assume this repository is the full ComfyUI install.
Do not touch sibling folders in `custom_nodes`.

## Architecture rules

- Keep nodes small and explicit.
- Do not create one huge Python file.
- Put node classes in `nodes/`.
- Put shared pure Python helpers in `utils/`.
- Keep frontend code in `web/js/` only when needed.
- Avoid clever abstractions unless they remove real duplication.
- Never hardcode local paths.
- Never commit models, checkpoints, generated images, logs, secrets, or local output files.
- Do not add automatic shell execution.
- Avoid `install.py` unless explicitly requested.
- Prefer `requirements.txt` for pip dependencies.
- Keep `requirements.txt` minimal.
- Do not add or pin torch unless explicitly requested.

## ComfyUI node rules

Each node class must define:

- `INPUT_TYPES`
- `RETURN_TYPES`
- `FUNCTION`
- `CATEGORY`

Expose nodes through `NODE_CLASS_MAPPINGS` in the root `__init__.py`.
Expose readable names through `NODE_DISPLAY_NAME_MAPPINGS`.

Optional dependencies must not crash ComfyUI at import time.
If a dependency is missing, raise a clear error inside the node function.

## UI and naming rules

- Use clear category names.
- Use stable node internal names.
- Do not rename public node inputs or outputs unless the task asks for it.
- When changing public inputs or outputs, update README and example workflows.

## Testing

- Add small pure Python tests for logic when possible.
- Do not require a full ComfyUI launch for basic tests.
- Keep test data tiny.
- Do not depend on local absolute paths.

## Response expectations

After every task, explain:

1. What changed
2. Which files changed
3. How to test it in ComfyUI
4. Any limitation or risk
