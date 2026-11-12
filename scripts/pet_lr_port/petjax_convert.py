"""pet-jax's own metatrain -> Flax key-renaming functions (petjax/convert.py, c3f216c), loaded from his source with
jax / marathon stubbed, for the bit-for-bit round-trip check of iris_to_gguf.py. No metatrain."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# pet-jax c3f216c petjax/convert.py (his source, not vendored here); override with PETJAX_CONVERT
PETJAX_CONVERT = Path(__import__("os").environ.get(
    "PETJAX_CONVERT", Path.home() / "data/plato-lr-scratch/pet_lr_port/src/petjax/convert.py"))


def load_petjax_convert(path):
    """pet-jax's convert.py with its JAX / marathon imports stubbed (only numpy is needed by the renames)."""
    jnp = types.ModuleType("jax.numpy")
    jnp.array = np.array
    jnp.asarray = np.asarray
    jax = types.ModuleType("jax")
    jax.numpy = jnp
    mio = types.ModuleType("marathon.io")
    mio.write_msgpack = mio.write_yaml = mio.read_msgpack = mio.read_yaml = None
    marathon = types.ModuleType("marathon")
    marathon.io = mio
    saved = {k: sys.modules.get(k) for k in ("jax", "jax.numpy", "marathon", "marathon.io")}
    sys.modules.update({"jax": jax, "jax.numpy": jnp, "marathon": marathon, "marathon.io": mio})
    try:
        spec = importlib.util.spec_from_file_location("petjax_convert", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod
