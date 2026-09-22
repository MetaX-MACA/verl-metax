# verl-metax RL-Insight 使用说明

## 1. 简介

RL-Insight 是 Verl 强化学习训练过程观测和分析的工具。在训练过程中，可以通过 RL-Insight 查看 Trainer、Rollout、Transfer Queue、Timeline 等信息，帮助定位训练过程中时间花在哪里、Rollout 是否正常、训练和推理之间的数据传输是否存在瓶颈。

RL-Insight 使用 Prometheus 采集和保存监控指标，通过 Grafana 展示。

本目录是完整、自包含的 MetaX RL-Insight 运维入口：安装脚本、MetaX
exporter、Grafana Dashboard、单机/多机说明和双机训练样例都放在这里。
初次接触本项目时从本文开始即可，所需操作信息都在本目录内。

| 指标 | 主要用途 |
| --- | --- |
| Trainer | 查看训练 step、loss、学习率、训练耗时等 |
| Rollout | 查看 vLLM / SGLang 生成阶段的运行情况 |
| Transfer Queue | 查看 Trainer 和 Rollout 之间的数据传输情况 |
| Timeline | 查看训练、Rollout、权重同步等阶段的时间分布 |
| CPU / Host | 查看 CPU、内存、网络使用情况 |
| GPU | 查看 GPU 利用率、显存、温度、功耗等硬件状态 |

### 1.1 MetaX GPU 监控

RL-Insight 已经提供训练过程监控能力。在 MetaX GPU 环境下，为了查看 GPU 硬件状态，增加了 MetaX GPU 指标采集支持。

通过 mx-smi 定时读取 GPU 状态，并转换成 Prometheus 可以采集的指标，最终在 Grafana 中展示。

目前支持：

- GPU 利用率
- 显存利用率
- 已使用显存
- 总显存
- GPU 温度
- 板卡功耗

## 2. 安装说明

本文介绍 Verl 在 MetaX GPU 环境下使用 RL-Insight 的配置方法，包括：

- RL-Insight 安装和启动
- CPU / Host 监控配置
- MetaX GPU 监控配置
- Grafana Dashboard 加载
- Prometheus target 注册
- Verl 训练脚本配置
- 单机使用（多机部署见 [`README_multi_node.md`](README_multi_node.md)）
- 常见问题排查

配置提供两种方式：

- 一键配置
- 手动配置

### 2.1 使用前准备

一键配置脚本不会自动安装 RL-Insight 和 node_exporter。第一次使用前，请先准备这两个依赖。

#### 安装 RL-Insight

MetaX 适配已在 `rl-insight 0.2.1` 上完成验证：

```bash
python3 -m pip install "rl-insight==0.2.1" --no-deps
python3 -m pip install "prometheus-client>=0.20"
```

如果发行镜像已经预装这些依赖，可以跳过安装，但应先确认版本和导入正常：

```bash
python3 -c "import importlib.metadata as m; print(m.version('rl-insight')); import prometheus_client"
```

#### 准备 RL-Insight Server 组件（离线环境可选）

RL-Insight Server 运行需要 Prometheus、Tempo 和 Grafana。

在线环境不需要提前准备这三个包；一键脚本会直接调用 RL-Insight 官方的 `server install` 机制下载安装。

内网或 GitHub 访问受限时，建议提前准备下面三个离线包，并放到 `tools/rl_insight/rl_insight_archives/` 目录：

```text
prometheus-2.54.1.linux-amd64.tar.gz
tempo_2.6.1_linux_amd64.tar.gz
grafana-13.0.0.linux-amd64.tar.gz
```

不要手工解压这些包，也不要自己复制到 `~/.rl-insight/services`。

`setup_metax_monitoring.sh` 会把离线包目录交给 RL-Insight 官方 installer，由 RL-Insight 自己完成解压、安装目录、manifest 和版本检查。

如果离线包统一放在共享目录，也可以通过环境变量指定：

```bash
RL_INSIGHT_ARCHIVE_DIR=/share/rl_insight_archives \
  bash tools/rl_insight/setup_metax_monitoring.sh
```

#### 安装系统依赖

一键配置脚本会使用 `ss` 命令检查服务端口状态。

如果环境中缺少 `ss`，需要安装 `iproute2`：

```bash
apt-get update
apt-get install -y iproute2
```

如果缺少该依赖执行脚本时会提示：

```text
[ERROR] 缺少 ss 命令
```

#### 安装 node_exporter

先检查：

```bash
command -v prometheus-node-exporter
command -v node_exporter
```

Ubuntu 环境如果没有安装：

```bash
apt-get update
apt-get install -y prometheus-node-exporter
```

安装完成后，可以使用一键配置，也可以按手动流程配置。

### 2.2 一键配置

适合依赖已经准备好的环境。

```bash
bash tools/rl_insight/setup_metax_monitoring.sh
```

脚本会自动完成：

1. 检查 RL-Insight；
2. 检查并启动 `node_exporter`；
3. 启动 `metax_metrics_exporter`；
4. 安装 MetaX Grafana Dashboard；
5. 必要时重新启动 RL-Insight；
6. 注册 Prometheus hardware targets；
7. 检查 CPU 和 MetaX GPU target 是否正常。
默认 exporter 端口：

| 服务 | 默认端口 |
| --- | --- |
| node_exporter | 9100 |
| MetaX exporter | 9101 |

exporter 端口支持通过环境变量修改：

```bash
NODE_EXPORTER_PORT=19100 \
METAX_EXPORTER_PORT=19101 \
bash tools/rl_insight/setup_metax_monitoring.sh
```

如果不设置环境变量，则使用默认端口。

脚本不会自动安装 RL-Insight 或 node_exporter。如果缺少依赖，会直接提示安装方式。

`setup_metax_monitoring.sh` 当前主要用于单机快速配置，并按可重复执行的方式设计。已经正常运行的组件不会重复启动；如果某个监控服务异常，也可以重新执行该脚本进行检查和恢复。

一键配置完成后，可以直接按第 3 章配置训练脚本并启动训练。

#### 2.2.1 日常使用和重启后的处理

同一容器内只重启 RL-Insight：

```bash
rl-insight server start --detach
```

hardware target 已写入持久配置，RL-Insight 启动时会自动重新生成到 runtime 配置中；只要 node_exporter 和 MetaX exporter 进程还在，CPU / MetaX GPU 指标会继续显示。

容器 stop / start 后，容器内文件通常还在，但 RL-Insight、node_exporter、MetaX exporter 等进程都会停止。

如果需要完整查看 CPU / MetaX GPU 硬件指标，推荐重新执行一键脚本：

```bash
bash tools/rl_insight/setup_metax_monitoring.sh
```

如果暂时不看硬件指标，只需要 Trainer / Rollout / Transfer Queue 等训练指标，可以只启动 RL-Insight Server，再正常启动训练任务。

训练脚本仍需要配置 `RL_INSIGHT_SERVER_URL`，并开启对应的 RL-Insight logger / metrics 参数。

如果不清楚具体是哪个监控组件异常，也可以直接重新执行 `setup_metax_monitoring.sh`。脚本会检查 exporter、RL-Insight、Dashboard 和 Prometheus target，并尽量恢复到可用状态。

### 2.3 手动配置

如果需要逐项确认每一步是否正常，可以按照本文后面的手动流程配置。

建议顺序：

```text
安装并启动 RL-Insight
        ↓
准备 node_exporter
        ↓
启动 MetaX exporter
        ↓
复制 MetaX Dashboard
        ↓
统一重启 RL-Insight 一次
        ↓
注册 Prometheus target
        ↓
检查 Prometheus 和 Grafana
```

Dashboard 和 exporter 都准备完成后再统一重启 RL-Insight，避免配置过程中反复重启。

#### 2.3.1 RL-Insight Server

第一次使用：

```bash
rl-insight server install
```

启动：

```bash
rl-insight server start --detach
```

常用端口：

| 端口 | 服务 |
| --- | --- |
| 18080 | RL-Insight Server |
| 9090 | Prometheus |
| 3000 | Grafana |
| 3200 | Tempo |

检查这些端口有没有监听：

```bash
ss -lntp | grep -E '18080|9090|3000|3200'
```

如果几个端口都在监听，说明 RL-Insight 的基础服务已经正常启动。

#### 2.3.2 准备 CPU / Host 监控

以下端口为默认配置。一键配置模式支持通过 `NODE_EXPORTER_PORT` 和 `METAX_EXPORTER_PORT` 修改端口。

CPU、内存、网络依赖标准 node_exporter。

检查 node_exporter 是否已经安装：

```bash
command -v prometheus-node-exporter
command -v node_exporter
```

Ubuntu 环境如果没有，可以安装：

```bash
apt-get update
apt-get install -y prometheus-node-exporter
```

启动服务：

```bash
nohup /usr/bin/prometheus-node-exporter \
  --web.listen-address=:9100 \
  >/tmp/node_exporter.log 2>&1 &
```

检查端口有没有被监听：

```bash
ss -lntp | grep 9100
```

验证 node_exporter 是否真的有数据：

```bash
curl -s http://127.0.0.1:9100/metrics \
  | grep '^node_cpu_seconds_total' \
  | head
```

如果能正常返回 CPU 指标，说明 node_exporter 已经启动，9100 端口可以访问，CPU 指标已经正常提供给 Prometheus。

#### 2.3.3 启动 MetaX GPU exporter

执行：

```bash
nohup python3 tools/rl_insight/metax_metrics_exporter.py \
  --port 9101 \
  --interval 5 \
  >/tmp/metax_metrics_exporter.log 2>&1 &
```

检查进程是否存在：

```bash
ps -ef | grep metax_metrics_exporter.py | grep -v grep
```

例如：

```text
root   123456  ... python .../metax_metrics_exporter.py --port 9101 --interval 5
```

说明 MetaX exporter 进程已经启动。

再检查 9101 端口是否正在监听：

```bash
ss -lntp | grep 9101
```

验证 MetaX GPU 指标是否真的有数据：

```bash
curl -s http://127.0.0.1:9101/metrics \
  | grep '^metax_gpu_' \
  | head -20
```

#### 2.3.4 加载 MetaX Grafana Dashboard

`metax_gpu_metrics.json` 是 MetaX GPU 的 Grafana Dashboard 配置文件。

文件放到 RL-Insight 自带的 Dashboard 目录后，RL-Insight 启动时会按已有的 Dashboard 加载机制自动加载。

先找到当前 Python 环境中的 RL-Insight Dashboard 目录：

```bash
python - <<'PY'
import pathlib
import rl_insight

p = pathlib.Path(rl_insight.__file__).parent
print(p / "config/services/grafana/dashboards")
PY
```

目录只有当前 Python 环境已经安装 rl-insight 后才会存在。如果命令无法 import `rl_insight`，需要先完成 RL-Insight 安装。

把 `metax_gpu_metrics.json` 复制到上面查询到的 Dashboard 目录，例如：

```bash
DASHBOARD_DIR=$(python - <<'PY'
import pathlib
import rl_insight
print(pathlib.Path(rl_insight.__file__).parent / "config/services/grafana/dashboards")
PY
)

cp tools/rl_insight/metax_gpu_metrics.json \
  "${DASHBOARD_DIR}/"
```

Dashboard 和 exporter 都准备完成后，再统一重启一次 RL-Insight Server。

Grafana 会重新加载 Dashboard，避免配置过程中反复重启。

```bash
rl-insight server stop
rl-insight server start --detach
```

#### 2.3.5 Prometheus target 配置

##### 单机

如果 RL-Insight Server 和训练都在同一台机器：

```yaml
jobs:
  - job_name: node-exporter
    targets:
      - target: "127.0.0.1:9100"
        labels:
          node: "node-0"

  - job_name: metax_gpu_metrics
    targets:
      - target: "127.0.0.1:9101"
        labels:
          node: "node-0"
```

保存为：

```text
hardware_targets.yaml
```

注册：

```bash
rl-insight server targets add hardware_targets.yaml
```

##### 多机

多机 targets 配置（含带 `node` label 的 YAML 示例）和完整部署步骤，见 [`README_multi_node.md`](README_multi_node.md)。

#### 2.3.5 多机配置

多机（一台 Server 节点 + 多个训练节点）场景下的架构、部署步骤、Worker exporter 启动、远端 target 注册与持久化、targets 示例 YAML，统一见专门文档：

[`README_multi_node.md`](README_multi_node.md)

要点速览：

- RL-Insight Server / Prometheus / Grafana 只在 Server 节点部署一份（即本节的手动单机流程）；
- 其他节点按需只启动 node_exporter(9100) 和 MetaX exporter(9101)，由 Server 的 Prometheus 拉取；
- 训练指标由训练进程通过 `RL_INSIGHT_SERVER_URL` 推送到 Server，与 Ray 多机训练相互独立；
- 远端 targets 在 Server 重启后可能需要重新注册，建议保留固定的 targets YAML 文件。

## 3. 在训练脚本中配置

### 3.1 配置 RL-Insight Server 地址

在脚本前面加上以下命令，IP 要改成实际运行 RL-Insight Server 的机器 IP。

例如：

```bash
export RL_INSIGHT_SERVER_URL=http://192.168.16.4:18080
```

### 3.2 打开 Trainer 的 RL-Insight 日志

需要在 Verl 启动参数中加入：

```bash
trainer.logger='["console","rl_insight"]'
```

### 3.3 打开 Rollout 指标

如果使用 vLLM / SGLang rollout，需要保证：

```bash
actor_rollout_ref.rollout.disable_log_stats=False
```

作用：允许 rollout engine 输出自己的运行指标。

### 3.4 打开 Transfer Queue 指标

如果当前 Verl 配置支持 Transfer Queue metrics，可以加入：

```bash
transfer_queue.metrics.enabled=True
```

作用：采集训练和 rollout 之间数据传输相关的指标。

### 完整示例

完整的可运行训练脚本（含 MACA 环境变量、RL-Insight 配置、vLLM/Megatron 全部参数）见仓库中的示例脚本。模型和数据路径可以通过环境变量覆盖：

```text
tools/rl_insight/train_deepseek_r1_qwen_32b_2nodes.sh
```

双机训练的完整启动和验证步骤见 [`README_training_multi_node.md`](README_training_multi_node.md)。

其中与 RL-Insight 相关的只有 3.1～3.4 节列出的配置：`RL_INSIGHT_SERVER_URL`、`trainer.logger` 加 `rl_insight`，以及 `disable_log_stats=False` / `transfer_queue.metrics.enabled=True` 两个指标开关；其余均为模型/并行/环境配置，与监控无关。

## 4. Grafana 查看可视化数据

如果访问端与 RL-Insight Server 网络互通，可以直接访问：

`http://<RL-Insight Server IP>:3000`

访问端无法直接访问该节点的 3000 端口，可以通过 SSH 端口转发访问。请在访问端机器（例如本地电脑 Windows PowerShell / Linux 终端）执行：

```bash
ssh -p <ssh_port> -L 3000:127.0.0.1:3000 <user>@<node_ip>
```

参数说明：

- `<node_ip>`：RL-Insight Server 所在节点的 SSH 登录地址；
- `<ssh_port>`：SSH 登录端口；
- `<user>`：SSH 登录用户名。

保持 SSH 连接窗口开启，然后在浏览器访问：

`http://127.0.0.1:3000`

### 4.1 SSH 转发示例

#### 容器节点

如果通过 SSH 登录容器环境：

```bash
ssh -i "<Your private key path>" -p <ssh_port> -L 3000:127.0.0.1:3000 <user>@<node_ip>
```

例如：

```bash
ssh -i "<Your private key path>" -p 31901 -L 3000:127.0.0.1:3000 root@10.13.71.11
```

#### 裸金属节点

如果直接登录裸金属节点：

```bash
ssh -p 22 -L 3000:127.0.0.1:3000 <user>@<node_ip>
```

例如：

```bash
ssh -p 22 -L 3000:127.0.0.1:3000 m01xxx@10.0.81.61
```

主要查看以下指标。

### 4.2 Trainer 指标

Trainer 指标主要用来看训练是否稳定，以及训练过程中的资源使用情况。

| 指标 / 面板 | 含义 |
| --- | --- |
| actor_entropy | 策略输出的随机性/多样性。过低可能说明策略过早变得单一；需要结合任务和训练阶段一起看。 |
| actor_grad_norm | Actor 梯度范数。主要用来观察梯度是否突然变大，辅助判断训练是否不稳定。 |
| actor_kl_loss / actor_kl_coef | KL loss 表示当前策略和参考策略的偏离程度；kl_coef 是 KL 惩罚系数。 |
| actor_loss | Actor 的优化目标值。主要看变化趋势，不能只根据单个数值判断好坏。 |
| actor_lr | Actor 当前学习率，用于确认学习率调度是否符合预期。 |
| actor_perf_memory | Actor 进程的 CPU/GPU 内存相关指标，用来观察训练过程中内存占用和峰值。 |

### 4.3 Rollout 指标

| 指标 / 面板 | 含义 |
| --- | --- |
| Token Throughput | 单位时间内处理的 token 数，通常以 tokens/s 表示。用于观察生成吞吐。 |
| Time Per Output Token Latency (TPOT) | 首 token 之后，每生成一个输出 token 的平均耗时。通常越低越好。 |
| Cache Utilization | KV Cache 使用率。持续接近 100% 时说明 KV Cache 空间比较紧张。 |
| Time To First Token Latency (TTFT) | 从请求进入到生成第一个 token 的时间。反映请求排队、prefill 等阶段的等待时间。 |
| E2E Request Latency | 一次完整生成请求从开始到结束的总耗时。 |
| Scheduler State | 当前运行、等待等请求数量，用来判断 rollout engine 是否出现排队或调度压力。 |

### 4.4 Transfer Queue 指标

| 指标 / 面板 | 含义 |
| --- | --- |
| Controller Uptime | Transfer Queue Controller 已经运行的时间。 |
| Controller RSS Memory | Controller 进程实际占用的主机内存。 |
| Active Partitions | 当前正在使用的 partition 数量。 |
| Global Indexes Allocated / Reusable Indexes | 已分配和可复用的索引数量，用于观察队列内部资源使用情况。 |
| Controller Request Rate | 不同 Transfer Queue 操作每秒的请求次数。 |
| Controller Request Latency P50 / P99 | 请求延迟的中位数和高分位延迟。P99 明显升高时需要关注长尾阻塞。 |

### 4.5 Timeline

Timeline 用来观察训练、Rollout、权重同步等阶段什么时候开始、持续多久。

色块越宽表示该阶段耗时越长；如果不同阶段之间存在明显空白，可以进一步排查等待、同步或资源空闲。

### 4.6 CPU / Host 指标

| 指标 / 面板 | 含义 |
| --- | --- |
| CPU Utilization | 主机 CPU 使用率。持续较高时需要关注数据处理、通信或其他 CPU 侧瓶颈。 |
| Memory Used | 主机已经使用的内存。接近机器总内存时需要警惕 Host OOM。 |
| Network Throughput | 主机网络接收（RX）和发送（TX）吞吐，用来观察多机通信和数据传输压力。 |

### 4.7 MetaX GPU 指标

| 指标 / 面板 | 含义 |
| --- | --- |
| GPU Utilization | GPU 计算利用率。用于观察训练或 Rollout 阶段 GPU 是否真正处于计算状态。 |
| VRAM Utilization | 显存使用比例。持续接近上限时容易出现显存不足。 |
| VRAM Used | 当前已经使用的显存容量。 |
| VRAM Total | GPU 可用总显存容量，用于和 VRAM Used 一起判断显存余量。 |
| Board Power | 板卡当前功耗。可以辅助判断 GPU 是否处于高负载，以及不同阶段负载变化。 |
| Temperature | GPU 温度。用于观察长时间训练过程中温度是否处于正常范围。 |

即使当前没有跑训练，只要 MetaX exporter 和 Prometheus 正常，MetaX GPU Dashboard 会有硬件数据。

## 5. Question & Answer

### 5.1 是否每次都要重新启动和注册

如果这些服务和进程还在：

- RL-Insight Server
- Prometheus
- Grafana
- Tempo
- node_exporter
- metax_metrics_exporter

如果这些服务和进程都正常，重新跑训练时可以直接使用，不需要每次重新启动 exporter 或重新注册 hardware target。

### 5.2 容器或服务重启后怎么办

只结束训练任务：不用重新配置监控，下一次训练直接使用即可。

只重启 RL-Insight、容器没有重启：直接重新启动 RL-Insight。hardware target 会从持久配置自动恢复；如果 node_exporter 和 MetaX exporter 仍在运行，硬件指标会继续显示。

容器 stop / start 后：进程会停止。如果需要完整的 CPU / MetaX GPU 硬件监控，直接重新执行：

```bash
bash tools/grafana/setup_metax_monitoring.sh
```

让脚本重新拉起 exporter、RL-Insight 并检查 target。

如果暂时只看训练指标，不看 CPU / MetaX GPU，可以只执行：

```bash
rl-insight server start --detach
```

如果服务状态不清楚或某个监控组件异常，优先重新执行 `setup_metax_monitoring.sh`，作为一键检查和恢复入口。

因此建议把一键配置脚本作为用户侧统一入口：首次部署、容器重启后需要完整硬件监控、或服务异常时都可以重复执行。

### 5.3 检查监控是否正常

CPU / Host：

```bash
curl -sG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=up{job="node-exporter"}'
```

MetaX GPU：

```bash
curl -sG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=up{job="metax_gpu_metrics"}'
```

正常情况下结果应为 `1`。

### 5.4 Dashboard 显示 No data 怎么查

先检查 exporter：

```bash
curl http://127.0.0.1:9100/metrics
curl http://127.0.0.1:9101/metrics
```

再检查 Prometheus：

```bash
curl -sG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=up{job="node-exporter"}'
```

```bash
curl -sG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=up{job="metax_gpu_metrics"}'
```

如果 exporter 有数据，但 Prometheus 查不到，优先检查：

- target 是否已注册；
- target IP / 端口是否正确；
- Server 到 exporter 节点网络是否能访问。

## 6. 文件说明

```text
tools/rl_insight/
├── metax_metrics_exporter.py
├── metax_gpu_metrics.json
├── setup_metax_monitoring.sh
├── train_deepseek_r1_qwen_32b_2nodes.sh
├── README.md
├── README_multi_node.md
└── README_training_multi_node.md
```

| 文件 | 用途 |
| --- | --- |
| metax_metrics_exporter.py | 从 mx-smi 采集 MetaX GPU 数据并提供 Prometheus 指标 |
| metax_gpu_metrics.json | 定义 MetaX GPU Grafana Dashboard |
| setup_metax_monitoring.sh | 单机一键配置脚本 |
| train_deepseek_r1_qwen_32b_2nodes.sh | 双机 8 卡/节点的 RL-Insight 训练样例 |
| README.md | 单机使用说明 |
| README_multi_node.md | 多机监控配置（Server + Worker）使用说明 |
| README_training_multi_node.md | 双机训练启动和训练指标验证说明 |

> Prometheus targets YAML 不提供仓库内示例文件：一键配置会在运行时自动生成；手动配置直接复制 2.3.4（单机）或 [`README_multi_node.md`](README_multi_node.md) 中的 YAML 示例。
