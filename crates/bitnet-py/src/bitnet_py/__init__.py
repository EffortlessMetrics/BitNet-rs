"""Minimal stub of bitnet_py for tests.

This stub provides just enough structure for import-time checks in the
llama_compat module. It intentionally raises FileNotFoundError when a model
is instantiated, mirroring the behavior when the actual native extension is
missing or a model path is invalid.
"""

class Model:  # pragma: no cover - stub used only for import-time behavior
    def __init__(self, *args, **kwargs):
        raise FileNotFoundError("Model not found")
