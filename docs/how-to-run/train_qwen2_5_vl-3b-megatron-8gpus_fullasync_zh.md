### 1. 进入工程目录

cd /workspace/verl-metax/

### 2. 设置cpu performance

使用部署环境的系统工具将 CPU governor 设置为 performance 模式。

### 3. 修改脚本

使用 `examples/full_async/qwen2_5_vl_3b_megatron.sh`，需要时覆盖以下环境变量：
MODEL_PATH: 模型路径
TRAIN_FILES / VAL_FILES: 数据集路径

### 4. 可选的 sleep mode

统一配置和验证方法见[共享环境说明](../../examples/env/README.md#vllm-metax-0230-sleep-mode)。

### 5. 运行脚本

bash examples/full_async/qwen2_5_vl_3b_megatron.sh

### 6. 版本对齐

版本不对齐会引起报错, 运行前应进行版本检查
已验证版本信息：Megatron-LM-0.18.0 + vllm-0.23.0 + transformers-5.8.1
