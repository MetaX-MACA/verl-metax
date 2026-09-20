import unittest
from unittest import mock

from packaging.version import Version

from verl_metax import __version__
from verl_metax.compat import SUPPORTED_VERL, check_verl_compatibility


class CompatibilityTests(unittest.TestCase):
    def test_package_version_tracks_supported_verl_release(self):
        self.assertEqual(Version(__version__).base_version, "0.9.0")
        self.assertEqual(str(SUPPORTED_VERL), f"=={Version(__version__).base_version}")

    def test_supported_verl_version(self):
        with mock.patch("verl_metax.compat._read_verl_version", return_value="0.9.0"):
            report = check_verl_compatibility(strict=True)
        self.assertTrue(report.compatible)

    def test_unsupported_verl_version_fails_in_strict_mode(self):
        with mock.patch("verl_metax.compat._read_verl_version", return_value="0.10.0"):
            with self.assertRaises(RuntimeError):
                check_verl_compatibility(strict=True)


if __name__ == "__main__":
    unittest.main()
