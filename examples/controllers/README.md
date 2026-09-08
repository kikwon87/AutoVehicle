# Controller templates / 제어기 템플릿

Copy one of these, edit it, and point the platform at it — the UI's
"Controller file" field, or `RunConfig(controller="/path/to/file.py")`.

The interface they implement is documented in `docs/controller_api.md`.

| File | What it shows |
|---|---|
| `template_minimal.py` | The smallest thing that drives: proportional lane keeping plus a speed PI. No imports from `avsim`. |
| `template_stanley.py` | A real geometric controller with a car-following gap policy, signal handling and `diagnostics()`. |
| `template_ml.py` | The shape a learned policy takes: feature vector in, normalized command out, weights loaded from a file. |

Run one from the command line without the UI:

```bash
python -m avsim.cli platform --headless \
    --preset unprotected_left --controller examples/controllers/template_stanley.py
```
