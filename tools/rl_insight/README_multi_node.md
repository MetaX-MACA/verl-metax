# RL-Insight 多机监控配置（Server + Worker）

适用场景：多机训练时，只在**一台 Server 节点**部署 RL-Insight Server / Prometheus / Grafana，其余节点只启动 exporter。

> - 本文档只覆盖**监控环境配置**；多机训练脚本如何运行、训练指标如何验证，见 [`README_training_multi_node.md`](README_training_multi_node.md)。
> - 本手动配置流程已在实际 2 节点环境中验证通过，可按步骤直接执行；每一步都附带检查命令和预期输出。
> - 单机部署见 [`README.md`](README.md)。

## 一、说明

1. RL-Insight Server / Prometheus / Grafana **只在 Server 节点部署一份**；
2. 每台机器（含 Server）都跑 node_exporter 和 MetaX exporter，硬件指标由 Server 的 Prometheus 统一拉取；
3. 训练指标由训练进程通过 `RL_INSIGHT_SERVER_URL` 推送到 Server，与 Ray 多机训练相互独立；
4. Grafana 中一个 dashboard 即可查看全部节点，通过 `node` 标签区分来源；
5. 多机监控与 Ray 多机训练是两件事：Ray 负责训练任务调度，RL-Insight 只负责收集和展示指标，互不依赖。

## 二、配置步骤

### 第 1 步：Server 节点部署监控服务（一次性）

```bash
bash tools/rl_insight/setup_metax_monitoring.sh
```

Server 节点会运行：

```text
RL-Insight Server :18080
Prometheus        :9090
Grafana           :3000
Tempo             :3200
node_exporter     :9100
MetaX exporter    :9101
```

### 第 2 步：Worker 节点启动 exporter

> **轻量模式**：如果只需要训练指标（loss / reward / rollout 等），不需要各节点 CPU / GPU 硬件监控，worker 节点可以完全不装 exporter，只要网络能访问 Server 的 18080 即可。

每台 worker 机器上（在装好 MetaX 驱动/容器内）执行。

**先检查端口是否已在监听——已监听则不要重复启动：**

```bash
# CPU / Host exporter（Ubuntu 包名 prometheus-node-exporter）
ss -lntp | grep ":9100"
# 未监听时再启动：
nohup /usr/bin/prometheus-node-exporter \
  --web.listen-address=:9100 \
  >/tmp/node_exporter.log 2>&1 &

# MetaX GPU exporter：在 verl 仓库根目录执行
ss -lntp | grep ":9101"
# 未监听时再启动：
nohup python3 tools/rl_insight/metax_metrics_exporter.py \
  --port 9101 \
  --interval 5 \
  >/tmp/metax_metrics_exporter.log 2>&1 &
```

Worker 节点本机检查（两条都有输出说明本机 exporter 正常）：

```bash
curl -s http://127.0.0.1:9100/metrics | grep '^node_cpu_seconds_total' | head
curl -s http://127.0.0.1:9101/metrics | grep '^metax_gpu_' | head
```

Server 节点自身的 exporter 由一键脚本自动启动，不用重复操作。

### 第 3 步：Server 节点注册所有节点的 targets

**先做网络预检**（在 Server 上执行，确认能拉到 worker 的 exporter 数据，两条都有输出再继续；不通则先查防火墙 / 容器端口映射，不要继续注册）：

```bash
curl -s http://<worker_ip>:9100/metrics | grep '^node_cpu_seconds_total' | head -1
curl -s http://<worker_ip>:9101/metrics | grep '^metax_gpu_' | head -1
```

然后创建 targets 文件 `hardware_targets_multi.yaml`。以验证过的两节点为例（node-0 = Server 192.168.16.4，node-1 = worker 192.168.16.5），节点更多时按相同格式继续增加：

```yaml
jobs:
  - job_name: node-exporter
    targets:
      - target: "192.168.16.4:9100"
        labels: {node: "node-0"}
      - target: "192.168.16.5:9100"
        labels: {node: "node-1"}

  - job_name: metax_gpu_metrics
    targets:
      - target: "192.168.16.4:9101"
        labels: {node: "node-0"}
      - target: "192.168.16.5:9101"
        labels: {node: "node-1"}
```

说明：

- IP / 主机名按实际训练节点填写，不要写死本文示例值；
- **显式写 `node` 标签**：它是 Grafana 图例区分节点、按节点过滤的依据；
- 每个 target 的端口与第 2 步 exporter 监听端口一一对应（9100 / 9101）。

在 Server 节点注册：

```bash
rl-insight server targets add hardware_targets_multi.yaml
```

> - `targets add` 是按 job 覆盖式注册（upsert），重复执行安全；
> - **持久化注意**：`targets add` 写入的是当前 runtime 的 Prometheus 配置。RL-Insight Server 重启后，本机 target 会按持久化机制自动恢复，但**远端节点的 targets 可能需要重新注册一次**——保留固定的 `hardware_targets_multi.yaml`，需要时直接重新执行上面这条命令即可。

### 第 4 步：检查多机 target

```bash
curl -sG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=up{job="node-exporter"}'
```

```bash
curl -sG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=up{job="metax_gpu_metrics"}'
```

正常情况下，**每个节点**都应返回一条 `value=1`，并通过 `node` 标签区分不同机器；哪个节点缺了就只排查那一台。

## 三、Grafana 查看硬件指标

打开 `http://<server_ip>:3000` → `RL-Insight → verl_trainer_v1_with_vllm_engine`：

- **hardware metric → MetaX GPU Metrics**：所有节点的 GPU 曲线画在同一面板，按 `node` 标签（主机名）区分图例；
- 节点多时可在面板 query 中加 `{node="<hostname>"}` 过滤单台机器，或在 dashboard 中增加 `$node` 变量做下拉切换。

训练指标（training metric）的查看与多机汇聚说明见 [`README_training_multi_node.md`](README_training_multi_node.md)。

## 四、常见问题（硬件监控相关）

| 现象 | 排查 |
| --- | --- |
| 某台节点 GPU 曲线缺失 | 在该节点执行 `curl http://127.0.0.1:9101/metrics \| grep '^metax_gpu_'` 确认 exporter 正常；再到 Server 执行 `curl -s http://<worker_ip>:9101/metrics \| grep '^metax_gpu_'` 确认网络可达；最后确认 `up{job="metax_gpu_metrics", node="<节点名>"}` 为 1；都正常则检查 targets YAML 是否包含该节点 |
| worker 重启后 exporter 丢失 | exporter 用 nohup 启动，不随机器自启；建议纳入节点初始化脚本或用 systemd 托管 |
| Server 重启后远端节点硬件曲线消失 | 远端 targets 不随 Server 持久化自动恢复，重新执行 `rl-insight server targets add hardware_targets_multi.yaml` |
| 想扩到更多节点 | 新节点跑 exporter（第 2 步）→ `hardware_targets_multi.yaml` 增加该节点 → 重新 `targets add` |
