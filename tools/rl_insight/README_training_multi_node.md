# 多机训练脚本使用 RL-Insight

对应脚本：`tools/rl_insight/train_deepseek_r1_qwen_32b_2nodes.sh`

（DeepSeek-R1-Distill-Qwen-32B + Megatron，2 节点 × 8 卡 GRPO 训练，训练指标统一上报到 Server 节点。）

相关文档：

- 多机监控环境配置（exporter / targets / 硬件指标）：[`README_multi_node.md`](README_multi_node.md)
- 监控服务端单机部署：[`README.md`](README.md)

> 本文档只讲**如何运行多机训练脚本并验证训练指标上报**；监控环境（exporter / targets / Grafana 硬件面板）的配置见上表第一个链接。

## 一、前提条件

1. 已按 [`README_multi_node.md`](README_multi_node.md) 完成多机监控环境配置（或采用其"轻量模式"：只要训练指标、不配 exporter）；
2. 所有节点的训练环境都安装了 rl-insight client：

```bash
pip install "rl-insight==0.2.1"
```

3. 各节点到 Server 的 18080 端口网络可达（在各节点上检查）：

```bash
curl http://<server_ip>:18080/healthz
```

## 二、运行

先起 Ray 集群，再从 Server 节点跑训练脚本（`RL_INSIGHT_SERVER_URL` 指向 Server）：

```bash
# Server 节点 + 各 worker 分别启动 ray（或使用已有集群管理方式）
ray start --head --port=6379        # Server 节点
ray start --address=<server_ip>:6379  # 各 worker

# Server 节点执行训练脚本
RL_INSIGHT_SERVER_URL=http://<server_ip>:18080 \
  bash tools/rl_insight/train_deepseek_r1_qwen_32b_2nodes.sh
```

脚本内与 RL-Insight 相关的关键配置（均可通过环境变量或 Hydra 参数覆盖）：

```bash
export RL_INSIGHT_SERVER_URL="http://<server_ip>:18080"   # 必须指向 Server 节点 IP
trainer.n_gpus_per_node=8
trainer.nnodes=2
trainer.logger='["console","rl_insight"]'
```

说明：

- **不要写 `127.0.0.1`**：Ray worker 继承 driver 的环境变量，写 127.0.0.1 会在 worker 本机回环、连不到 Server；
- 多机下 worker 不需要单独 export，继承 driver 环境变量即可；
- 换 project / experiment 名用 hydra 透传：`trainer.project_name=xxx trainer.experiment_name=yyy`。

## 三、验证训练指标上报

浏览器打开 `http://<server_ip>:3000` → `RL-Insight → verl_trainer_v1_with_vllm_engine`：

- **training metric**：全部节点的训练指标汇聚在同一个图里（各节点训练进程统一推到 Server，天然合并，按 project_name / experiment_name 过滤）；
- 训练启动后 1~2 分钟内应能看到 actor loss / reward 等曲线；training metric 没有数据时：

| 排查 | 命令 |
| --- | --- |
| Server 可达性 | 在任意训练节点执行 `curl http://<server_ip>:18080/healthz` |
| client 是否安装 | 各节点 `python3 -c "import rl_insight"` |
| 环境变量是否正确 | 训练日志里确认 `RL_INSIGHT_SERVER_URL` 指向 Server IP 而非 127.0.0.1 |

硬件曲线（GPU 利用率 / VRAM / 功耗 / 温度）不显示时，属于监控环境问题，排查方法见 [`README_multi_node.md`](README_multi_node.md) 第四节。
