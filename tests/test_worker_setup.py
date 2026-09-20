import unittest
from unittest import mock

from verl_metax import worker_setup


class WorkerSetupTests(unittest.TestCase):
    def test_setup_installs_lazy_worker_workarounds(self):
        with mock.patch("verl_metax.deferred.install") as install:
            worker_setup.setup()

        install.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
