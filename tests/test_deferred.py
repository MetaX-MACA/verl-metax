import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

from verl_metax import deferred


class DeferredWorkerSetupTests(unittest.TestCase):
    def tearDown(self):
        sys.meta_path[:] = [
            finder for finder in sys.meta_path if not getattr(finder, "__verl_metax_worker_finder__", False)
        ]

    def test_install_is_lazy_and_idempotent(self):
        before = set(sys.modules)

        self.assertTrue(deferred.install())
        self.assertFalse(deferred.install())

        imported = set(sys.modules) - before
        self.assertFalse(imported & deferred.TARGET_MODULES)

    def test_wrapped_loader_patches_only_after_module_execution(self):
        events = []
        wrapped = SimpleNamespace(exec_module=lambda _module: events.append("imported"))
        apply = mock.Mock(side_effect=lambda _name: events.append("patched"))
        workarounds = SimpleNamespace(apply_imported_worker_workaround=apply)
        module = ModuleType("verl.utils.distributed")

        with mock.patch.dict(sys.modules, {"verl_metax.workarounds.verl_0_9": workarounds}):
            loader = deferred._PatchLoader("verl.utils.distributed", wrapped)
            loader.exec_module(module)

        self.assertEqual(events, ["imported", "patched"])
        apply.assert_called_once_with("verl.utils.distributed")
