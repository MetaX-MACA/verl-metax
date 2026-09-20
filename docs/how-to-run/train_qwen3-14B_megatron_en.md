### 1. 进入工程目录

cd /workspace/verl-metax/

### 2. 设置cpu performance

Configure the host CPU governor for performance mode using the deployment's system tool.

### 3. 修改脚本

Use `examples/grpo/qwen3_14b_megatron.sh` and override these environment variables when needed:
actor_rollout_ref.model.path: 模型路径
data.train_files: 训练数据集路径
data.val_files: 验证数据集路径

### 4. Optional sleep mode

See the [shared environment guide](../../examples/env/README.md#vllm-metax-0230-sleep-mode).

### 5. 运行脚本

bash examples/grpo/qwen3_14b_megatron.sh

### 6. Version Info

Packages' version should be aligned.
Aligned versions：Megatron-LM-0.18.0 + vllm-0.23.0 + transformers-5.8.1
Note: CPU OOM will occur if CPU memory <= 1T
