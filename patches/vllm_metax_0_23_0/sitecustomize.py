"""Redirect vllm_metax module imports to patched files in this directory.

Loaded automatically by Python's `site` module at interpreter startup
(put this directory on PYTHONPATH). The finder below is consulted on the
first import of each `vllm_metax.*` submodule; files mirroring the package
layout under ./vllm_metax/ override the installed versions.
"""
try:
    import os
    import sys
    from importlib.abc import MetaPathFinder
    from importlib.machinery import SourceFileLoader
    from importlib.util import spec_from_file_location

    _ROOT = os.path.dirname(os.path.abspath(__file__))

    class _VerlVllmPatchFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in sys.modules:
                return None
            if not fullname.startswith("vllm_metax."):
                return None
            # vllm_metax.platform -> <root>/vllm_metax/platform.py
            f = os.path.join(_ROOT, *fullname.split(".")) + ".py"
            if os.path.isfile(f):
                return spec_from_file_location(
                    fullname, f, loader=SourceFileLoader(fullname, f)
                )
            return None

    sys.meta_path.insert(0, _VerlVllmPatchFinder())
except Exception:  # never break interpreter startup
    pass
