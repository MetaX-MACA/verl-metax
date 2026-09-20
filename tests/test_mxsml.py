import unittest
from types import ModuleType, SimpleNamespace

from verl_metax.integrations import mxsml


class MXSMLTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        module = ModuleType("pymxsml")
        module.MXSML_INIT_FLAG_REINIT = 7
        module.mxSmlInitWithFlags = lambda flags: self.calls.append(("init", flags))
        module.mxSmlGetDeviceInfo = lambda index: SimpleNamespace(uuid=f"MX-{index}".encode())
        module.mxSmlSetCpuAffinity = lambda index: self.calls.append(("affinity", index))
        mxsml._module = module
        mxsml._initialized = False

    def tearDown(self):
        mxsml._module = None
        mxsml._initialized = False

    def test_uuid_is_decoded_and_library_is_initialized_once(self):
        self.assertEqual(mxsml.get_device_uuid("2"), "MX-2")
        self.assertEqual(mxsml.get_device_uuid(3), "MX-3")
        self.assertEqual(self.calls, [("init", 7)])

    def test_cpu_affinity_uses_integer_device_index(self):
        mxsml.set_cpu_affinity("4")
        self.assertEqual(self.calls, [("init", 7), ("affinity", 4)])


if __name__ == "__main__":
    unittest.main()
