import unittest
from unittest import mock

from packaging.version import Version

from verl_metax import __version__
from verl_metax._build_version import package_version
from verl_metax.compat import SUPPORTED_VERL, check_verl_compatibility


class CompatibilityTests(unittest.TestCase):
    def test_package_version_tracks_supported_verl_release(self):
        self.assertEqual(Version(__version__).base_version, "0.9.0")
        self.assertEqual(str(SUPPORTED_VERL), f"=={Version(__version__).base_version}")

    def test_build_version_includes_maca_ai_version(self):
        version = Version(package_version("3.8.0.5"))

        self.assertEqual(str(version), "0.9.0+maca3.8.0.5")
        self.assertEqual(version.base_version, "0.9.0")

    def test_build_version_reads_maca_ai_environment(self):
        with mock.patch.dict("os.environ", {"MACA_AI_VERSION": "3.8.0.5"}):
            self.assertEqual(package_version(), "0.9.0+maca3.8.0.5")

    def test_build_version_without_maca_ai_version_is_public_version(self):
        self.assertEqual(package_version(""), "0.9.0")

    def test_build_version_normalizes_valid_local_separators(self):
        self.assertEqual(package_version("3_8-0.5"), "0.9.0+maca3.8.0.5")

    def test_build_version_rejects_invalid_local_identifier(self):
        with self.assertRaisesRegex(ValueError, "MACA_AI_VERSION"):
            package_version("3.8/0")

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
