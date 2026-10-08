import unittest
from unittest import mock

from verl_metax import worker_setup


class WorkerSetupTests(unittest.TestCase):
    def test_setup_installs_lazy_worker_workarounds(self):
        with (
            mock.patch("verl_metax.deferred.install") as install,
            mock.patch("verl_metax.sleep_mode_patch.activate") as activate,
        ):
            worker_setup.setup()

        install.assert_called_once_with()
        activate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
