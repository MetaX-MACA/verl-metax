import os
import sys
import unittest
import warnings
from pathlib import Path
from types import ModuleType
from unittest import mock

from verl_metax.sleep_mode_patch import _core, _verify


class SleepModePatchTests(unittest.TestCase):
    def setUp(self):
        self.meta_path = list(sys.meta_path)
        self.pythonpath = os.environ.get("PYTHONPATH")

    def tearDown(self):
        sys.meta_path[:] = self.meta_path
        if self.pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = self.pythonpath
        for module in _core.patch_module_paths():
            sys.modules.pop(module, None)

    def test_auto_activates_for_vllm_metax_0_23(self):
        with mock.patch.object(_core, "_installed_vllm_metax_version", return_value="0.23.0+maca1"):
            report = _core.activate("auto")

        self.assertTrue(report.active)
        self.assertEqual(report.reason, "activated")
        finder = sys.meta_path[0]
        spec = finder.find_spec("vllm_metax.platform")
        self.assertIsNotNone(spec)
        origin = Path(spec.origin)
        self.assertEqual(origin.name, "platform.py")
        self.assertEqual(origin.parent.name, "_vllm_metax_0_23_0")

    def test_activation_is_idempotent(self):
        with mock.patch.object(_core, "_installed_vllm_metax_version", return_value="0.23.0"):
            first = _core.activate("auto")
            second = _core.activate("auto")

        self.assertTrue(first.active)
        self.assertTrue(second.active)
        self.assertEqual(second.reason, "already active")

    def test_activation_propagates_sitecustomize_directory(self):
        with (
            mock.patch.object(_core, "_installed_vllm_metax_version", return_value="0.23.0"),
            mock.patch.dict(os.environ, {"PYTHONPATH": "/existing"}),
        ):
            _core.activate("auto")
            first = os.environ["PYTHONPATH"].split(os.pathsep)[0]

        self.assertEqual(os.path.normcase(first), os.path.normcase(str(_core._PACKAGE_DIR)))

    def test_off_never_installs_finder(self):
        with mock.patch.object(_core, "_installed_vllm_metax_version", return_value="0.23.0"):
            report = _core.activate("off")

        self.assertFalse(report.active)
        self.assertEqual(report.reason, "disabled by configuration")
        self.assertFalse(_core._finder_installed())

    def test_auto_skips_other_vllm_metax_versions(self):
        with mock.patch.object(_core, "_installed_vllm_metax_version", return_value="0.24.0"):
            report = _core.activate("auto")

        self.assertFalse(report.active)
        self.assertIn("outside the supported range", report.reason)

    def test_on_rejects_other_vllm_metax_versions(self):
        with mock.patch.object(_core, "_installed_vllm_metax_version", return_value="0.24.0"):
            with self.assertRaisesRegex(RuntimeError, "outside the supported range"):
                _core.activate("on")

    def test_auto_refuses_partial_activation_after_target_import(self):
        sys.modules["vllm_metax.platform"] = ModuleType("vllm_metax.platform")
        with (
            mock.patch.object(_core, "_installed_vllm_metax_version", return_value="0.23.0"),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            report = _core.activate("auto")

        self.assertFalse(report.active)
        self.assertIn("already imported", report.reason)
        self.assertTrue(caught)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "VERL_METAX_SLEEP_PATCH"):
            _core.activate("sometimes")

    def test_verifier_compares_actual_import_locations(self):
        paths = _core.patch_module_paths()
        modules = {}
        for name, path in paths.items():
            module = ModuleType(name)
            module.__file__ = path
            modules[name] = module

        report = _core.SleepModePatchReport("on", "0.23.0", True, "already active")
        with (
            mock.patch.object(_verify, "activate", return_value=report),
            mock.patch.object(_verify, "patch_module_paths", return_value=paths),
            mock.patch.dict(sys.modules, modules),
        ):
            result = _verify._inspect_process()

        self.assertTrue(result["matches"])
        self.assertEqual(result["imported"], result["expected"])


if __name__ == "__main__":
    unittest.main()
