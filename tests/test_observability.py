import pathlib
import unittest

from verl_metax.observability.mxsmi import parse_mxsmi_output, split_gpu_blocks


class MXSMIParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = pathlib.Path(__file__).parent / "fixtures" / "mx_smi_2_2_9.txt"
        cls.text = fixture.read_text(encoding="utf-8")

    def test_split_gpu_blocks(self):
        self.assertEqual(set(split_gpu_blocks(self.text)), {0, 1})

    def test_parse_all_fields(self):
        metrics = parse_mxsmi_output(self.text)
        self.assertEqual(metrics[0].utilization_percent, 67.5)
        self.assertEqual(metrics[0].memory_used_bytes, 1024 * 1024)
        self.assertEqual(metrics[0].memory_total_bytes, 65536 * 1024)
        self.assertEqual(metrics[0].temperature_celsius, 54)
        self.assertEqual(metrics[0].power_watts, 245.5)
        self.assertEqual(metrics[1].power_watts, 270)

    def test_missing_field_is_none(self):
        metrics = parse_mxsmi_output("GPU#0 MXC550\n  Utilization\n    GPU : 10 %\n")
        self.assertEqual(metrics[0].utilization_percent, 10)
        self.assertIsNone(metrics[0].power_watts)


if __name__ == "__main__":
    unittest.main()

