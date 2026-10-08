import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
HOW_TO_RUN = ROOT / "docs" / "how-to-run"
RL_INSIGHT_RECIPE = ROOT / "tools" / "rl_insight" / "train_deepseek_r1_qwen_32b_2nodes.sh"
VLLM_METAX_PATCH = ROOT / "verl_metax" / "sleep_mode_patch" / "_vllm_metax_0_23_0"


class RecipeTests(unittest.TestCase):
    def test_training_recipes_use_metax_launcher_and_overridable_paths(self):
        recipes = sorted(path for path in EXAMPLES.rglob("*.sh") if "env" not in path.parts)
        recipes.append(RL_INSIGHT_RECIPE)
        self.assertGreaterEqual(len(recipes), 18)

        for path in recipes:
            with self.subTest(recipe=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                self.assertNotRegex(source, r"python(?:3)?\s+-m\s+verl(?:\.|\s)")
                self.assertTrue(
                    "verl-metax-ppo" in source or "verl-metax-run" in source or "exec bash" in source,
                    "recipe does not enter through a verl-metax launcher",
                )
                for variable in ("MODEL_PATH", "TRAIN_FILES", "VAL_FILES"):
                    self.assertRegex(
                        source,
                        rf"(?:export\s+)?{variable}=.*\$\{{{variable}:-",
                        f"{variable} must have an overridable default",
                    )

    def test_sleep_mode_recipe_loads_complete_vllm_metax_replacement(self):
        expected = (
            VLLM_METAX_PATCH.parent / "sitecustomize.py",
            VLLM_METAX_PATCH / "platform.py",
            VLLM_METAX_PATCH / "device_allocator" / "cumem.py",
            VLLM_METAX_PATCH / "patch" / "plugin_enhancement" / "device_allocator.py",
        )
        for path in expected:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(path.is_file())

        muon = (EXAMPLES / "grpo" / "qwen2_5_7b_muon_megatron.sh").read_text(encoding="utf-8")
        self.assertFalse((EXAMPLES / "env" / "vllm-metax-0.23-sleep-mode.sh").exists())
        self.assertNotIn("vllm-metax-0.23-sleep-mode.sh", muon)
        self.assertIn("enable_sleep_mode=True", muon)

    def test_sleep_mode_instructions_are_centralized(self):
        guide = (EXAMPLES / "env" / "README.md").read_text(encoding="utf-8")
        self.assertIn("## vllm-metax 0.23.0 sleep mode", guide)
        self.assertIn("VERL_METAX_SLEEP_PATCH", guide)
        self.assertNotIn("source examples/env/vllm-metax-0.23-sleep-mode.sh", guide)

        for path in HOW_TO_RUN.glob("*.md"):
            with self.subTest(path=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("source examples/env/vllm-metax-0.23-sleep-mode.sh", source)

    def test_focused_recipe_guides_name_their_entry_points(self):
        guides = {
            ROOT / "docs" / "recipes-pd-disaggregation.md": "examples/grpo/qwen3_4b_pd.sh",
            ROOT / "docs" / "recipes-megatron.md": "examples/grpo/qwen3_8b_megatron.sh",
            ROOT / "docs" / "recipes-muon.md": "examples/grpo/qwen2_5_7b_muon_megatron.sh",
        }
        for path, recipe in guides.items():
            with self.subTest(path=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                self.assertIn(recipe, source)
                self.assertIn(f"bash {recipe}", source)

        self.assertFalse((HOW_TO_RUN / "run_qwen3_4b_pd.md").exists())
        self.assertFalse((HOW_TO_RUN / "run_qwen3_8b_megatron.md").exists())
        self.assertFalse((HOW_TO_RUN / "muon_usage.md").exists())

    def test_megatron_path_is_centralized_and_overridable(self):
        env = (EXAMPLES / "env" / "megatron.sh").read_text(encoding="utf-8")
        self.assertRegex(env, r'MEGATRON_PATH="\$\{MEGATRON_PATH:-[^}]+\}"')

        for path in EXAMPLES.rglob("*.sh"):
            if path.name == "metax.sh":
                continue
            with self.subTest(recipe=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                self.assertNotRegex(source, re.compile(r"Megatron-LM-0\.(?:12|16)"))

    def test_environment_helpers_have_stable_names_and_document_versions(self):
        env_dir = EXAMPLES / "env"
        metax = (env_dir / "metax.sh").read_text(encoding="utf-8")
        megatron = (env_dir / "megatron.sh").read_text(encoding="utf-8")

        self.assertIn("MACA 3.8.0", metax)
        self.assertIn("Megatron-LM 0.18.0", megatron)
        self.assertIn('source "${SCRIPT_DIR}/metax.sh"', megatron)
        self.assertFalse((env_dir / "metax-3.8.sh").exists())
        self.assertFalse((env_dir / "megatron-0.18.sh").exists())


if __name__ == "__main__":
    unittest.main()
