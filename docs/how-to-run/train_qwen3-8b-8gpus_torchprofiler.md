# verl 0.9 Profiling Extensions

基于 verl 0.9 的 Profiling 能力，对大模型强化学习训练过程中的性能分析功能进行了整理与验证，重点覆盖两个相关特性：
* **Torch Profiler 支持与输出文件命名优化**
* **Rollout 按 response token 区间进行 Profiling**
这些功能主要用于定位训练过程中 CPU、GPU、CUDA Kernel、算子、通信以及同步等待等性能问题。

## 已验证版本
verl：0.9
## 特殊依赖
无
## 基本路径配置
actor_rollout_ref.model.path="['/metax0402/m01686/models/Qwen3-8B']"
data.train_files="['/metax0402/datasets/verl/gsm8k/train.parquet']"
data.val_files="['/metax0402/datasets/verl/gsm8k/test.parquet']"

## 1. 功能概览

在大模型强化学习训练中，一个完整的训练 Step 通常包含多个阶段，仅通过训练日志通常只能判断“哪个阶段耗时较长”，而 Profiler 可以进一步定位到阶段内部的：
* CPU 操作
* GPU Kernel
* CUDA 活动
* PyTorch 算子
* 通信操作
* 同步与等待
* 内存分配与释放
* 算子输入 Shape
* Python/C++ 调用栈
最终生成 Chrome Trace 格式的 `.json.gz` 文件，可以使用 Chrome Tracing 或 Perfetto 进行可视化分析。


## 2. 主要特性介绍
### 2.1 Torch Profiler 支持与输出命名优化

Torch Profiler支持三种模式：
1. Continuous / E2E Profiling：E2E模式将所有阶段数据采集到同一文件夹下。
2. Discrete Profiling：一次训练步里的每个关键阶段（计算 log_prob、更新参数、ref 前向）各自生成独立的 trace 文件，文件按阶段名分目录。
3. Schedule Profiling：每个周期仅记录几个 mini-batch，按 wait → warmup → active 循环采一部分。

#### 全局性能分析控制
在 global_profiler 中，可以全局控制性能分析的发生时间和方式。
global_profiler.tool=torch \
global_profiler.steps=[2] \
global_profiler.save_path=./outputs/profile \
global_profiler.profile_continuous_steps=False \   
● tool为采集工具选择，支持nsys、npu、torch、torch_memory等
● steps为指定采集的步数列表，如[2,4]表示采集第2步和第4步，设置为null则不采集
● save_path为保存性能分析结果的目录
● profile_continuous_steps为相邻 global profiling step是否拼成同一个 profiling session

#### 角色级性能分析控制：
每个强化学习（RL）角色（Actor、Critic 等）都有其自己的 profiler 配置。
● enable：是否为此角色启用性能分析。
● all_ranks：是否从所有rank收集数据。
● ranks：如果 all_ranks 为 False，则指定采集数据的rank列表。
● tool_config.torch：专门用于 PyTorch Profiler 的具体配置。

#### 具体采集行为配置PyTorch Profiler选项（tool_config.torch）
Torch Profiler 配置位于：
actor_rollout_ref.<role>.profiler.tool_config.torch
常见选型包括：
● contents：要进行性能分析的内容列表。
    ○ cpu：分析 CPU 活动。
    ○ cuda：分析 CUDA 活动。
    ○ memory：跟踪张量内存的分配/释放。
    ○ shapes：记录算子输入的形状。
    ○ stack：记录源代码文件和行号。
如果主要分析GPU kernel 和 CPU/GPU 时间线，常见配置是：contents=[cuda,cpu]。
如果需要进一步定位内存、输入 shape 或调用栈，可以增加：contents=[cuda,cpu,memory,shapes,stack]。
需要注意，增加采集内容通常会增加 profiling 开销和 trace 数据量。
● discrete: 离散模式开关（False为e2e模式，True为离散模式）
● schedule：启用 torch.profiler.schedule，以便仅记录每个性能分析窗口的一部分。仅在 active > 0 时生效；否则分析器会持续收集（默认值）。verl 通过在 Actor 更新循环中每个 mini-batch 调用一次 profiler.step()（以及在 SFT 中每步调用一次）来推进计划，因此计划周期的单位是 mini-batch。具体配置见3.3


#### 输出文件命名优化：
<role>-<scope>_rank<r>-of-<world>_tp<..>-pp<..>-dp<..>-cp<..>_pid<pid>_<timestamp>_cycle<N>.json.gz

以实际生成的文件名为例：actor-rollout-ref_e2e_rank0-of-8_pid749760_20260904135320921.json.gz
| 字段      | 示例值           | 含义说明                                                                          |
| role      |actor-rollout-ref| 工作者角色。如 actor、ref、critic 等，确保相同 rank 下不同角色的结果可区分。          |
| scope     | e2e             | 被分析区域。传递给start_profile/annotate的区域名称，同时作为 `save_path` 下的子目录。 |
| rank/world| rank0-of-8      | 全局分布式信息。表示当前是全局 torch.distributed 中的第 0 个 rank，总共 8 个进程。    |
|tp/pp/dp/cp| 示例中省略       | 分别代表张量/流水线/数据/上下文并行rank，Megatron 初始化时包含，示例为FSDP仅报告rank。 |
| pid       | 749760          | 进程 ID。生成该文件的本地系统进程号。                                                |
| timestamp |20260904135320921| 时间戳。精确到微秒的文件生成时间。                                                  |
| cycle     | 示例中省略       | 周期后缀。格式为 `cycle<N>`，仅在计划运行的第 2 个及更多周期时添加。                  |

这对于多卡、多进程以及多角色的 RL 训练场景尤其重要，可以避免不同 Rank、不同角色产生的 trace 文件难以区分的问题。


### 2.2 Rollout Token 区间 Profiling

LLM Rollout 的生成过程通常包含大量连续的 Decode Iteration。
如果直接对完整 Rollout 过程进行 Profiling，容易产生大量高度重复的 trace 数据，
对于较长的 Response，这会导致：
* Trace 文件过大
* 事件数量过多
* 重复信息严重
* Perfetto 加载和分析困难
因此，本特性支持通过：
profile_token_start和profile_token_end指定需要分析的 Response Token 区间。

注意：**Token 区间只影响 Profiler 的采集范围，不影响模型实际的生成范围。**
模型仍然会完整生成 Response，只是 Profiler 只记录指定区间。


## 3. Torch Profiler 配置

Torch Profiler 主要涉及三种使用方式：
1. Continuous / E2E Profiling
2. Discrete Profiling
3. Schedule Profiling
当前脚本 `examples/profiling/qwen3_8b_torch.sh` 默认使用 1. Continuous / E2E Profiling 作为示例。脚本中的 `global_profiler` 和 `actor_rollout_ref.actor.profiler` 参数是 Continuous 模式配置。
如需切换至Torch Profiler的其他模式或者按Rollout Token 区间 Profiling，请遵循以下步骤：
● 移除旧配置
在脚本中，删除第 74-80 行与 Continuous 模式相关的配置参数。
● 添加新配置
在脚本的末尾，$@ 变量之前，添加你希望启用的新模式配置参数。
重要提示：请务必将新配置添加在 $@ 之前，以确保参数能被正确传递给训练程序。


### 3.1 Continuous 模式

Continuous 模式适合观察完整训练阶段的端到端执行过程。
#### 相关配置参数：
global_profiler.tool=torch \
global_profiler.steps=[2] \
global_profiler.save_path=./outputs/profile \
actor_rollout_ref.actor.profiler.enable=True \
actor_rollout_ref.actor.profiler.all_ranks=True \
actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cuda,cpu] \
actor_rollout_ref.actor.profiler.tool_config.torch.discrete=False

仅采集指定 Rank
如果只需要 Rank 0：可以将脚本配置中的actor_rollout_ref.actor.profiler.all_ranks=True \替换为
actor_rollout_ref.actor.profiler.all_ranks=False \
actor_rollout_ref.actor.profiler.ranks=[0]

#### 参数说明
● global_profiler.steps=[2] \    # 只采集step 2的数据
● discrete=False \               # 同一个 rank 内，不按每个 profiling step 单独切 trace。对每一个被 profiling 的进程/rank 来说，把一次连续 profiling 区间写入一个 trace，而不是把整个分布式作业的所有 rank 物理合并成一个 JSON trace。
● all_ranks=True \               # 表示所有rank都会启动profiler。

#### 典型输出：
outputs/profiles/e2e/
├── actor-rollout-ref_e2e_rank0-of-8_pid495451_20260904130656676.json.gz
├── actor-rollout-ref_e2e_rank1-of-8_pid495452_20260904130656676.json.gz
├── actor-rollout-ref_e2e_rank2-of-8_pid495453_20260904130656676.json.gz
├── actor-rollout-ref_e2e_rank3-of-8_pid495454_20260904130656676.json.gz
├── actor-rollout-ref_e2e_rank4-of-8_pid495455_20260904130656676.json.gz
├── actor-rollout-ref_e2e_rank5-of-8_pid495456_20260904130656676.json.gz
├── actor-rollout-ref_e2e_rank6-of-8_pid495457_20260904130656676.json.gz
└── actor-rollout-ref_e2e_rank7-of-8_pid495458_20260904130656676.json.gz


### 3.2 Discrete 模式

Discrete 模式适合分别分析训练过程中的不同阶段。
#### 相关配置参数：
global_profiler.tool=torch \
global_profiler.steps=[2] \
global_profiler.save_path=./outputs/profile \
actor_rollout_ref.actor.profiler.enable=True \
actor_rollout_ref.rollout.profiler.enable=False \
actor_rollout_ref.ref.profiler.enable=False \
actor_rollout_ref.actor.profiler.all_ranks=False \
actor_rollout_ref.actor.profiler.ranks=[0] \
actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cuda,cpu] \
actor_rollout_ref.actor.profiler.tool_config.torch.discrete=True

#### 典型输出：
outputs/profile/
├── actor_compute_log_prob/
│   └── actor-rollout-ref_actor-compute-log-prob_rank0-of-8_pid12179_20260904114828074.json.gz
├── actor_update/
│   └── actor-rollout-ref_actor-update_rank0-of-8_pid12179_20260904114951565.json.gz
└── ref_compute_log_prob/
    └── actor-rollout-ref_ref-compute-log-prob_rank0-of-8_pid12179_20260904114909772.json.gz


### 3.3 Schedule 模式

Schedule 模式适用于包含大量 Mini-Batch 的训练更新循环。
例如一个 Actor Update 可能包含多个 Mini-Batch：如果全部采集，会产生大量 Trace 数据。
通过 Schedule 可以按照：skip_first → wait → warmup → active → repeat控制实际采集窗口。

#### 相关配置参数：
global_profiler.tool=torch \
global_profiler.steps=[2] \
global_profiler.save_path=./outputs/profile \
actor_rollout_ref.actor.profiler.enable=True \
actor_rollout_ref.actor.profiler.all_ranks=True \
actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cuda,cpu] \
actor_rollout_ref.actor.profiler.tool_config.torch.discrete=False \
actor_rollout_ref.actor.profiler.tool_config.torch.schedule.skip_first=0 \
actor_rollout_ref.actor.profiler.tool_config.torch.schedule.wait=1 \
actor_rollout_ref.actor.profiler.tool_config.torch.schedule.warmup=0 \
actor_rollout_ref.actor.profiler.tool_config.torch.schedule.active=1 \
actor_rollout_ref.actor.profiler.tool_config.torch.schedule.repeat=0

#### 参数说明：
| 参数          | 说明                                         |
| ------------ | -----------------------------------------------|
| `skip_first` | 第一个周期开始之前忽略的步数                     |
| `wait`       | 每个周期开始时不采集的步数                       |
| `warmup`     | 采集但丢弃的步数                                |
| `active`     | 正式记录的步数                                  |
| `repeat`     | 周期重复次数，`0` 表示持续重复直到 Profiling 停止 |


verl 会在 Actor 更新循环中以 Mini-Batch 为粒度调用 `profiler.step()`，因此 Schedule 的一个 step 对应一个 Mini-Batch。
注意：Schedule 仅在 `discrete=False` 的连续模式下生效。


## 4. Rollout Token 区间 Profiling

#### 相关配置参数：
如果需要只分析 Rollout 中某一段 Response Token，可以配置：
global_profiler.tool=torch \
global_profiler.steps=[2] \
global_profiler.save_path=./outputs/profile \
actor_rollout_ref.actor.profiler.enable=False \
actor_rollout_ref.rollout.profiler.enable=True \
actor_rollout_ref.rollout.profiler.all_ranks=False \
actor_rollout_ref.rollout.profiler.ranks=[0] \
actor_rollout_ref.rollout.profiler.tool=torch \
actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True \
actor_rollout_ref.rollout.profiler.tool_config.torch.profile_token_start=12 \
actor_rollout_ref.rollout.profiler.tool_config.torch.profile_token_end=22 \
actor_rollout_ref.rollout.profiler.save_path=/abs/outputs/profile

#### 参数说明
torch.profile_token_start=12:从第 12 个 response token 开始
torch.profile_token_end=22:到第 22 个为止（不含）
Rollout Engine 的 Profiling 输出路径建议使用**绝对路径**：/abs/outputs/profile
使用相对路径时，Profiling 文件可能被保存到 Ray 临时目录中，导致用户在预期目录下找不到结果。

#### 输出结果
outputs/profiles/agent_loop_rollout_replica_0/
├── profiler_out_0.txt                    # vLLM 的 profiler 运行日志
├── profiler_out_1.txt
├── rank0.1788503587595189774.pt.trace.json.gz   # TP rank0 的 trace
└── rank1.1788503587598647522.pt.trace.json.gz   # TP rank1 的 trace
文件名格式是 vLLM 的：rank{TP卡号}.{时间戳}.pt.trace.json.gz，每张 TP 卡一份。

#### Profiling 结果验证
在实际测试中，使用：profile_token_start=12、profile_token_end=22进行 Rollout Profiling。
采集窗口：[12,22)，共包含10个 Decode Step。
测试结果显示，窗口化 Profiling 能显著减少 Trace 数据量。

| 指标           | 全量 Profiling | Token `[12,22)`  |
| ------------- | -------------: | ---------------: |
| Trace 时间跨度 |   约 306.45 s  |       约 0.140 s |
| 采样算子次数   |         3143   |              10  |
| 事件总数       |    5,928,833   |          22,697  |
| Trace 文件大小 |      86.4 MB   |          0.3 MB  |

可以看到，指定 Token 区间后：
* Trace 时间跨度大幅缩小
* 事件数量显著下降
* 文件大小从约 86.4 MB 降低到约 0.3 MB
* 采样算子次数与目标 Decode Step 数量一致
