from __future__ import annotations

import sys
from types import ModuleType


class FakePlatformBase:
    @staticmethod
    def check_smi_command(command: str) -> bool:
        return command == "mx-smi"


class FakePlatformRegistry:
    _platforms: dict[str, type] = {}

    @classmethod
    def register(cls, platform: str):
        def decorator(platform_cls: type) -> type:
            cls._platforms[platform] = platform_cls
            return platform_cls

        return decorator

    @classmethod
    def get(cls, name: str):
        return cls._platforms.get(name)


def install() -> None:
    torch = ModuleType("torch")
    cuda = ModuleType("torch.cuda")
    cuda.is_available = lambda: True
    cuda.current_device = lambda: 0
    cuda.device_count = lambda: 8
    cuda.set_device = lambda _index: None
    cuda.synchronize = lambda *_args: None
    cuda.manual_seed = lambda _seed: None
    cuda.manual_seed_all = lambda _seed: None
    cuda.empty_cache = lambda: None
    cuda.get_device_capability = lambda _index=0: (9, 0)
    cuda.cudart = lambda: object()
    cuda.memory = type("Memory", (), {"_set_allocator_settings": staticmethod(lambda _settings: None)})()
    cuda.nvtx = type(
        "Nvtx",
        (),
        {"range_push": staticmethod(lambda _msg: None), "range_pop": staticmethod(lambda: None)},
    )()
    torch.cuda = cuda

    verl = ModuleType("verl")
    verl.__version__ = "0.9.0"
    verl.__path__ = []
    plugin_package = ModuleType("verl.plugin")
    plugin_package.__path__ = []
    platform_package = ModuleType("verl.plugin.platform")
    platform_package.__path__ = []
    base_module = ModuleType("verl.plugin.platform.platform_base")
    base_module.PlatformBase = FakePlatformBase
    manager_module = ModuleType("verl.plugin.platform.platform_manager")
    manager_module.PlatformRegistry = FakePlatformRegistry

    sys.modules.update(
        {
            "torch": torch,
            "torch.cuda": cuda,
            "verl": verl,
            "verl.plugin": plugin_package,
            "verl.plugin.platform": platform_package,
            "verl.plugin.platform.platform_base": base_module,
            "verl.plugin.platform.platform_manager": manager_module,
        }
    )
