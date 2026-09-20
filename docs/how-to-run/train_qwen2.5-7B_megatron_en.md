### 1. Enter the project directory

cd /workspace/verl-metax/

### 2. Set the server's CPU mode to performance

Configure the host CPU governor for performance mode using the deployment's system tool.

### 3. Modify the train script

Use `examples/grpo/qwen2_5_7b_megatron.sh` and override these environment variables when needed:
- actor_rollout_ref.model.path: model path
- data.train_files: train dara path
- data.val_files: val data path

### 4. Optional sleep mode

See the [shared environment guide](../../examples/env/README.md#vllm-metax-0230-sleep-mode).

### 5. Run training scripts

bash examples/grpo/qwen2_5_7b_megatron.sh

### 6. Version Info

Packages' version should be aligned.
Aligned versions：Megatron-LM-0.18.0 + vllm-0.23.0 + transformers-5.8.1
