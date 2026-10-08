import importlib
import sys
import unittest
from unittest import mock

from _stubs import FakePlatformRegistry, install


class PluginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install()
        for name in ("verl_metax.platform", "verl_metax.plugin"):
            sys.modules.pop(name, None)
        cls.plugin = importlib.import_module("verl_metax.plugin")

    def test_platform_is_registered(self):
        registered = FakePlatformRegistry.get("metax")
        self.assertIs(registered, self.plugin.PlatformMetaX)

    def test_registration_is_idempotent(self):
        first = self.plugin.register()
        second = self.plugin.register()
        self.assertIs(first, second)

    def test_initial_registration_checks_exact_verl_version(self):
        self.plugin._registered = False
        try:
            with (
                mock.patch("verl_metax.plugin.check_verl_compatibility") as check,
                mock.patch("verl_metax.plugin.activate_sleep_mode_patch") as activate,
            ):
                self.plugin.register()
            check.assert_called_once_with(strict=True)
            activate.assert_called_once_with()
        finally:
            self.plugin._registered = True

    def test_platform_identity(self):
        platform = self.plugin.PlatformMetaX()
        self.assertEqual(platform.device_name, "cuda")
        self.assertEqual(platform.vendor_name, "metax")
        self.assertEqual(platform.communication_backend_name(), "nccl")
        self.assertTrue(platform.is_platform_available(use_smi_check=True))


if __name__ == "__main__":
    unittest.main()
