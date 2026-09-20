import importlib
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

from _stubs import install

install()
verl_0_9 = importlib.import_module("verl_metax.workarounds.verl_0_9")


class WorkaroundTests(unittest.TestCase):
    def test_asm_environment_restores_existing_values(self):
        original = {"MHA_USE_BLAS": "custom"}
        with mock.patch.dict(os.environ, original, clear=True):
            with verl_0_9.asm_environment():
                self.assertEqual(os.environ["MHA_USE_BLAS"], "ON")
                self.assertEqual(os.environ["MHA_BWD_NO_ATOMIC_F64"], "1")
            self.assertEqual(dict(os.environ), original)

    def test_uses_asm_only_for_megatron_asm_backend(self):
        config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(
                    strategy="megatron",
                    megatron=SimpleNamespace(override_transformer_config={"te_flash_backend": "asm"}),
                )
            )
        )
        self.assertTrue(verl_0_9._uses_asm(config))
        config.actor_rollout_ref.actor.strategy = "fsdp"
        self.assertFalse(verl_0_9._uses_asm(config))

    def test_unknown_workaround_is_rejected(self):
        with self.assertRaises(ValueError):
            verl_0_9.apply_workarounds("does-not-exist")

    def test_none_applies_nothing(self):
        self.assertEqual(verl_0_9.apply_workarounds("none"), {})

    def test_device_flops_patch_is_idempotent(self):
        module = SimpleNamespace(_DEVICE_FLOPS={"C500": 1})
        with mock.patch.object(verl_0_9.importlib, "import_module", return_value=module):
            self.assertTrue(verl_0_9.patch_device_flops())
            self.assertFalse(verl_0_9.patch_device_flops())
        self.assertEqual(module._DEVICE_FLOPS, verl_0_9.DEVICE_FLOPS)

    def test_numa_affinity_delegates_to_metax_platform(self):
        original = mock.Mock()
        module = SimpleNamespace(set_numa_affinity=original)
        affinity = mock.Mock()
        platform = SimpleNamespace(
            vendor_name="metax",
            ray_resource_name=lambda: "GPU",
            set_cpu_affinity=affinity,
        )
        ray = ModuleType("ray")
        ray.is_initialized = lambda: False
        manager = sys.modules["verl.plugin.platform.platform_manager"]

        with (
            mock.patch.object(verl_0_9.importlib, "import_module", return_value=module),
            mock.patch.object(manager, "get_platform", return_value=platform, create=True),
            mock.patch.dict(sys.modules, {"ray": ray}),
            mock.patch.dict(os.environ, {"LOCAL_RANK": "3"}),
        ):
            self.assertTrue(verl_0_9.patch_numa_affinity())
            module.set_numa_affinity()

        affinity.assert_called_once_with("3")
        original.assert_not_called()

    def test_trtllm_uuid_delegates_to_active_platform(self):
        original = mock.Mock()
        module = SimpleNamespace(get_device_uuid=original)
        platform = SimpleNamespace(get_device_uuid=mock.Mock(return_value="MX-uuid"))
        manager = sys.modules["verl.plugin.platform.platform_manager"]

        with (
            mock.patch.object(verl_0_9.importlib.util, "find_spec", return_value=object()),
            mock.patch.object(verl_0_9.importlib, "import_module", return_value=module),
            mock.patch.object(manager, "get_platform", return_value=platform, create=True),
        ):
            self.assertTrue(verl_0_9.patch_trtllm_device_uuid())
            self.assertEqual(module.get_device_uuid("2"), "MX-uuid")

        platform.get_device_uuid.assert_called_once_with("2")
        original.assert_not_called()

    def test_vllm_headless_patch_uses_function_after_plugin_discovery(self):
        target = SimpleNamespace(run_headless=lambda *_args, **_kwargs: "stale")
        serve = SimpleNamespace(run_headless=mock.Mock(return_value="live"))

        def import_module(name):
            if name == "verl.workers.rollout.vllm_rollout.vllm_async_server":
                return target
            if name == "vllm.entrypoints.cli.serve":
                return serve
            raise AssertionError(name)

        with (
            mock.patch.object(verl_0_9.importlib.util, "find_spec", return_value=object()),
            mock.patch.object(verl_0_9.importlib, "import_module", side_effect=import_module),
        ):
            self.assertTrue(verl_0_9.patch_vllm_run_headless())
            self.assertEqual(target.run_headless("args"), "live")

        serve.run_headless.assert_called_once_with("args")

    def test_dataloader_cleanup_releases_unique_iterators_on_runner_failure(self):
        shared_shutdown = mock.Mock()
        val_shutdown = mock.Mock(side_effect=RuntimeError("worker already exited"))
        shared_iterator = SimpleNamespace(_shutdown_workers=shared_shutdown)
        val_iterator = SimpleNamespace(_shutdown_workers=val_shutdown)

        class PPOTrainer:
            pass

        class Trainer(PPOTrainer):
            def __init__(self):
                self.train_dataloader_it = shared_iterator
                self.train_dataloader = SimpleNamespace(_iterator=shared_iterator)
                self.val_dataloader = SimpleNamespace(_iterator=val_iterator)

        class TaskRunnerV1:
            def __init__(self):
                self.trainer = Trainer()

            def run(self):
                raise RuntimeError("training failed")

        class ModifiedTaskRunnerV1(TaskRunnerV1):
            __ray_actor_class__ = TaskRunnerV1

        original_actor_class = SimpleNamespace(
            __ray_metadata__=SimpleNamespace(modified_class=ModifiedTaskRunnerV1)
        )

        def ray_remote(source_class):
            class Modified(source_class):
                __ray_actor_class__ = source_class

            return SimpleNamespace(
                __ray_metadata__=SimpleNamespace(modified_class=Modified)
            )

        runner_module = SimpleNamespace(TaskRunnerV1=original_actor_class)
        modules = {
            "verl.trainer.ppo.v1.trainer_base": SimpleNamespace(PPOTrainer=PPOTrainer),
            "verl.trainer.main_ppo": runner_module,
            "ray": SimpleNamespace(remote=mock.Mock(side_effect=ray_remote)),
        }
        with mock.patch.object(verl_0_9.importlib, "import_module", side_effect=modules.__getitem__):
            self.assertTrue(verl_0_9.patch_dataloader_worker_cleanup())
            self.assertFalse(verl_0_9.patch_dataloader_worker_cleanup())

        self.assertIsNot(runner_module.TaskRunnerV1, original_actor_class)
        modules["ray"].remote.assert_called_once_with(TaskRunnerV1)

        # Execute the user class that the rebuilt ActorClass will export. This
        # models execution in a fresh Ray worker, where PPOTrainer.close was
        # not monkey-patched by the driver process.
        del PPOTrainer.close
        runner = TaskRunnerV1()
        with self.assertLogs(verl_0_9.logger, level="WARNING") as logs:
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                runner.run()

        shared_shutdown.assert_called_once_with()
        val_shutdown.assert_called_once_with()
        self.assertIn("Failed to shut down DataLoader workers", logs.output[0])
        self.assertIsNone(runner.trainer.train_dataloader_it)
        self.assertIsNone(runner.trainer.train_dataloader._iterator)
        self.assertIsNone(runner.trainer.val_dataloader._iterator)

    def test_dataloader_cleanup_defers_to_native_close(self):
        native_close = mock.Mock()

        class PPOTrainer:
            close = native_close

        trainer_module = SimpleNamespace(PPOTrainer=PPOTrainer)
        with mock.patch.object(verl_0_9.importlib, "import_module", return_value=trainer_module) as import_module:
            self.assertFalse(verl_0_9.patch_dataloader_worker_cleanup())

        import_module.assert_called_once_with("verl.trainer.ppo.v1.trainer_base")

    def test_ray_init_installs_worker_bootstrap_and_selection(self):
        original_init = mock.Mock(return_value="context")
        ray = SimpleNamespace(init=original_init)

        with mock.patch.object(verl_0_9.importlib, "import_module", return_value=ray):
            self.assertTrue(verl_0_9.patch_ray_worker_setup("device-flops,numa-affinity"))
            self.assertFalse(verl_0_9.patch_ray_worker_setup("device-flops,numa-affinity"))

        result = ray.init(
            address="auto",
            runtime_env={"env_vars": {"EXISTING": "1"}, "working_dir": "/workspace"},
        )

        self.assertEqual(result, "context")
        original_init.assert_called_once_with(
            address="auto",
            runtime_env={
                "env_vars": {
                    "EXISTING": "1",
                    "VERL_METAX_WORKAROUNDS": "device-flops,numa-affinity",
                },
                "working_dir": "/workspace",
                "worker_process_setup_hook": verl_0_9.RAY_WORKER_SETUP_HOOK,
            },
        )

    def test_ray_init_rejects_conflicting_worker_bootstrap(self):
        ray = SimpleNamespace(init=mock.Mock())
        with mock.patch.object(verl_0_9.importlib, "import_module", return_value=ray):
            self.assertTrue(verl_0_9.patch_ray_worker_setup("all"))

        with self.assertRaisesRegex(RuntimeError, "another hook is configured"):
            ray.init(runtime_env={"worker_process_setup_hook": "custom.setup"})

    def test_megatron_patch_scopes_asm_environment(self):
        seen = []

        class Trainer:
            def __init__(self):
                self.config = SimpleNamespace(
                    actor_rollout_ref=SimpleNamespace(
                        actor=SimpleNamespace(
                            strategy="megatron",
                            megatron=SimpleNamespace(override_transformer_config={"te_flash_backend": "asm"}),
                        )
                    )
                )

            def _compute_old_log_prob(self):
                seen.append((os.environ.get("MHA_USE_BLAS"), os.environ.get("MHA_BWD_NO_ATOMIC_F64")))

            def _update_actor(self):
                seen.append((os.environ.get("MHA_USE_BLAS"), os.environ.get("MHA_BWD_NO_ATOMIC_F64")))

        class V1Trainer(Trainer):
            pass

        modules = {
            "verl.trainer.ppo.v1.trainer_base": SimpleNamespace(PPOTrainer=V1Trainer),
            "verl.trainer.ppo.ray_trainer": SimpleNamespace(RayPPOTrainer=Trainer),
        }
        with (
            mock.patch.object(verl_0_9.importlib, "import_module", side_effect=modules.__getitem__),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            self.assertTrue(verl_0_9.patch_megatron_asm_scopes())
            trainer = Trainer()
            trainer._compute_old_log_prob()
            trainer._update_actor()
            v1_trainer = V1Trainer()
            v1_trainer._compute_old_log_prob()
            v1_trainer._update_actor()
            self.assertNotIn("MHA_USE_BLAS", os.environ)
            self.assertNotIn("MHA_BWD_NO_ATOMIC_F64", os.environ)

        self.assertEqual(seen, [("ON", "1")] * 4)

    def test_entropy_kernel_patch_installs_metax_tile(self):
        cache = mock.Mock()
        cache.clear = mock.Mock()
        kernel = SimpleNamespace(configs=[SimpleNamespace(kwargs={"BLOCK_SIZE_M": 128})], cache=cache)
        kernels = SimpleNamespace(efficient_entropy_kernel_general_mainloop=kernel)
        triton = SimpleNamespace(
            Config=lambda kwargs, **options: SimpleNamespace(kwargs=kwargs, options=options),
        )

        def import_module(name):
            return triton if name == "triton" else kernels

        with (
            mock.patch.object(verl_0_9.importlib.util, "find_spec", return_value=object()),
            mock.patch.object(verl_0_9.importlib, "import_module", side_effect=import_module),
        ):
            self.assertTrue(verl_0_9.patch_entropy_kernel_tile())

        self.assertEqual(
            kernel.configs[0].kwargs,
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        )
        cache.clear.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
