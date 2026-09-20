#!/bin/bash
set -e


# --------------------------------------------------
# Repository paths
# --------------------------------------------------
# 当前脚本:
#   <verl-metax>/tools/rl_insight/setup_metax_monitoring.sh
#
# 无论 Verl 仓库位于 /metax0402/...、/workspace/... 或其他路径，
# 都从脚本自身位置自动计算仓库根目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# RL-Insight Server 离线包默认位置:
#   <verl-metax>/tools/rl_insight/rl_insight_archives/
#
# 也允许用户通过环境变量指定共享目录：
#   RL_INSIGHT_ARCHIVE_DIR=/share/rl_insight_archives \
#       bash tools/rl_insight/setup_metax_monitoring.sh
RL_INSIGHT_ARCHIVE_DIR="${RL_INSIGHT_ARCHIVE_DIR:-${SCRIPT_DIR}/rl_insight_archives}"
# RL-Insight runtime dashboard directory
RUNTIME_DASHBOARD_DIR="${HOME}/.rl-insight/runtime/dashboards"
METAX_EXPORTER="${SCRIPT_DIR}/metax_metrics_exporter.py"
METAX_DASHBOARD="${SCRIPT_DIR}/metax_gpu_metrics.json"

NODE_NAME="${NODE_NAME:-$(hostname)}"
TARGET_HOST="${TARGET_HOST:-127.0.0.1}"

# configurable exporter ports
NODE_EXPORTER_PORT="${NODE_EXPORTER_PORT:-9100}"
METAX_EXPORTER_PORT="${METAX_EXPORTER_PORT:-9101}"


generate_metax_verl_dashboard()
{
    echo "[INFO] 合并 MetaX 面板到 verl_trainer_v1_with_vllm_engine ..."

    export METAX_DASHBOARD_SRC="${METAX_DASHBOARD}"
    export RL_INSIGHT_RUNTIME_DASHBOARDS="${RUNTIME_DASHBOARD_DIR}"

    python3 <<'PY'
import json
import os
import shutil
from pathlib import Path

import pathlib
import rl_insight

dashboard_dir = (
    pathlib.Path(rl_insight.__file__).parent
    / "config/services/grafana/dashboards"
)

src = dashboard_dir / "verl_tainer_v1_with_vllm_engine.json"
runtime_dir = Path(os.environ["RL_INSIGHT_RUNTIME_DASHBOARDS"])
metax = Path(os.environ["METAX_DASHBOARD_SRC"])

# NOTE: Grafana file provisioning watches runtime_dir and provisions EVERY
# *.json inside as its own dashboard. The merged dashboard therefore MUST be
# written as verl_tainer_v1_with_vllm_engine.json (same metadata.name/UID as
# the original) so it REPLACES the original dashboard. Writing it under a new
# filename creates a second dashboard and, with the original file still
# present, a duplicate-UID conflict that blocks provisioning updates.
dst_runtime = runtime_dir / "verl_tainer_v1_with_vllm_engine.json"
dst_config = src

stale_files = [
    runtime_dir / "verl_tainer_v1_with_vllm_engine_metax.json",
    runtime_dir / "metax_gpu_metrics.json",
    dashboard_dir / "metax_gpu_metrics.json",
]

print("source :", src)
print("output :", dst_runtime)
print("panels :", metax)

if not src.exists():
    raise SystemExit("missing source dashboard")
if not metax.exists():
    raise SystemExit("missing metax dashboard")

with open(src) as f:
    d = json.load(f)
with open(metax) as f:
    m = json.load(f)

elements = d["spec"]["elements"]

# ---- idempotency: never merge twice ---------------------------------
already_merged = any(
    str(el.get("spec", {}).get("title", "")).startswith("MetaX ")
    for el in elements.values()
)

if not already_merged:

    # 1) drop Ascend NPU panel elements (match by title/query, not ids)
    def is_npu_panel(el):
        if el.get("kind") != "Panel":
            return False
        sp = el.get("spec", {})
        if "NPU" in str(sp.get("title", "")):
            return True
        try:
            for q in sp["data"]["spec"]["queries"]:
                expr = str(q["spec"]["query"]["spec"].get("expr", ""))
                if "npu" in expr.lower():
                    return True
        except (KeyError, TypeError):
            pass
        return False

    removed = [name for name, el in elements.items() if is_npu_panel(el)]
    for name in removed:
        elements.pop(name, None)
    print("removed NPU panels:", removed)

    # 2) add MetaX panels as dashboard.grafana.app/v2 Panel elements.
    #    Pasting a classic-schema panel body (type/targets/options/...)
    #    into a v2 element is LOSSY when Grafana migrates it (queries
    #    become null), so convert explicitly into data/vizConfig exactly
    #    like the original rl-insight panels.
    max_id = max(
        int(x.split("-")[1])
        for x in elements
        if x.startswith("panel-") and x.split("-")[1].isdigit()
    )

    new_panels = []
    for panel in m["panels"]:
        max_id += 1
        name = f"panel-{max_id}"

        queries = []
        for t in panel.get("targets", []):
            queries.append(
                {
                    "kind": "PanelQuery",
                    "spec": {
                        "query": {
                            "kind": "DataQuery",
                            "group": "prometheus",
                            "version": "v0",
                            "spec": {
                                "editorMode": "code",
                                "expr": t.get("expr", ""),
                                "legendFormat": t.get("legendFormat", ""),
                                "range": True,
                            },
                            "labels": {
                                "grafana.app/export-label": "prometheus-1"
                            },
                        },
                        "refId": t.get("refId", "A"),
                        "hidden": False,
                    },
                }
            )

        options = {"annotations": {"clustering": -1, "multiLane": False}}
        options.update(panel.get("options", {}))
        if isinstance(options.get("tooltip"), dict):
            options["tooltip"].setdefault("hideZeros", False)

        elements[name] = {
            "kind": "Panel",
            "spec": {
                "id": max_id,
                "title": panel.get("title", ""),
                "description": panel.get("description", ""),
                "links": [],
                "data": {
                    "kind": "QueryGroup",
                    "spec": {
                        "queries": queries,
                        "transformations": [],
                        "queryOptions": {},
                    },
                },
                "vizConfig": {
                    "kind": "VizConfig",
                    "group": panel.get("type", "timeseries"),
                    "version": "13.0.2",
                    "spec": {
                        "options": options,
                        "fieldConfig": panel.get("fieldConfig", {}),
                    },
                },
            },
        }
        new_panels.append(name)

    print("added MetaX panels:", new_panels)

    # 3) purge stale references from the nested layout tree so that no
    #    GridLayoutItem / TabsLayoutTab / RowsLayoutRow points at a panel
    #    that no longer exists (dangling refs break dashboard loading).
    def clean_layout(layout):
        kind = layout.get("kind")
        spec = layout.setdefault("spec", {})
        if kind in ("GridLayout", "AutoGridLayout"):
            items = spec.get("items", [])
            spec["items"] = [
                it
                for it in items
                if it.get("spec", {}).get("element", {}).get("name") in elements
            ]
            return bool(spec["items"])
        if kind == "TabsLayout":
            kept = []
            for tab in spec.get("tabs", []):
                tab_lay = tab.get("spec", {}).get("layout")
                if tab_lay is None or clean_layout(tab_lay):
                    kept.append(tab)
            spec["tabs"] = kept
            return bool(kept)
        if kind == "RowsLayout":
            kept = []
            for row in spec.get("rows", []):
                row_lay = row.get("spec", {}).get("layout")
                if row_lay is None or clean_layout(row_lay):
                    kept.append(row)
            spec["rows"] = kept
            return bool(kept)
        return True

    clean_layout(d["spec"]["layout"])
    print("layout cleaned (stale NPU references removed)")


    # 4) put the MetaX grid inside the existing "hardware metric" section
    def find_row(rows, title):
        for row in rows:
            if row.get("kind") != "RowsLayoutRow":
                continue
            if row.get("spec", {}).get("title") == title:
                return row
            lay = row.get("spec", {}).get("layout")
            if lay and lay.get("kind") == "RowsLayout":
                found = find_row(lay.get("spec", {}).get("rows", []), title)
                if found is not None:
                    return found
        return None

    metax_row = {
        "kind": "RowsLayoutRow",
        "spec": {
            "title": "MetaX GPU Metrics",
            "collapse": False,
            "layout": {
                "kind": "GridLayout",
                "spec": {
                    "items": [
                        {
                            "kind": "GridLayoutItem",
                            "spec": {
                                "x": (i % 2) * 12,
                                "y": (i // 2) * 8,
                                "width": 12,
                                "height": 8,
                                "element": {
                                    "kind": "ElementReference",
                                    "name": name,
                                },
                            },
                        }
                        for i, name in enumerate(new_panels)
                    ]
                },
            },
        },
    }

    top_rows = d["spec"]["layout"]["spec"]["rows"]
    hardware = find_row(top_rows, "hardware metric")
    if hardware is not None:
        inner = hardware["spec"]["layout"]["spec"]["rows"]
        inner[:] = [
            r
            for r in inner
            if r.get("spec", {}).get("title") != "Ascend NPU Metrics"
        ]
        inner.append(metax_row)
        print("MetaX row added inside 'hardware metric'")
    else:
        top_rows.append(metax_row)
        print("'hardware metric' row not found -> appended top-level row")

# ---- write merged dashboard under the ORIGINAL filename/UID ----------
# A backup goes to ~/.rl-insight/dashboard_backups (never into the
# provisioned directories: every *.json there becomes a dashboard).
backup_dir = Path.home() / ".rl-insight" / "dashboard_backups"
backup_dir.mkdir(parents=True, exist_ok=True)
backup = backup_dir / src.name
if not backup.exists():
    shutil.copy2(src, backup)
    print("original dashboard backed up:", backup)

with open(dst_config, "w") as f:
    json.dump(d, f, indent=2)
with open(dst_runtime, "w") as f:
    json.dump(d, f, indent=2)

print("dashboard written:", dst_config)
print("dashboard written:", dst_runtime)

# ---- remove standalone MetaX dashboard files -------------------------
# (metax_gpu_metrics.json registers as its own "MetaX GPU Metrics"
#  dashboard; the old *_metax.json duplicates the UID of the original)
for stale in stale_files:
    if stale.exists():
        stale.unlink()
        print("removed standalone dashboard file:", stale)
PY
}


echo "========================================"
echo " MetaX RL-Insight Monitoring Setup"
echo "========================================"
echo "Node: ${NODE_NAME}"
echo


# --------------------------------------------------
# 0. 前置依赖检查
# --------------------------------------------------
echo "[0/7] 检查基础依赖"

MISSING=0

if ! command -v ss >/dev/null 2>&1; then
    echo "[ERROR] 缺少 ss 命令"
    echo "请安装 iproute2:"
    echo "  apt-get update && apt-get install -y iproute2"
    MISSING=1
else
    echo "[OK] ss"
fi

for cmd in python3 curl grep nohup; do
    if command -v "${cmd}" >/dev/null 2>&1; then
        echo "[OK] ${cmd}"
    else
        echo "[ERROR] 缺少命令: ${cmd}"
        MISSING=1
    fi
done

# 检查 MetaX mx-smi
if command -v mx-smi >/dev/null 2>&1; then
    echo "[OK] mx-smi: $(command -v mx-smi)"
elif [ -x /opt/mxdriver/bin/mx-smi ]; then
    echo "[OK] mx-smi: /opt/mxdriver/bin/mx-smi"
else
    echo "[ERROR] 找不到 mx-smi"
    echo "请确认 MetaX Driver 已正确安装或挂载到当前容器。"
    MISSING=1
fi

# 检查 prometheus_client
if python3 -c "import prometheus_client" >/dev/null 2>&1; then
    echo "[OK] Python prometheus_client"
else
    echo "[ERROR] Python 缺少 prometheus_client"
    echo "安装命令："
    echo "  python3 -m pip install prometheus-client"
    MISSING=1
fi

if [ "${MISSING}" -ne 0 ]; then
    echo
    echo "[ERROR] 基础依赖检查未通过，请先安装缺失依赖后重新运行。"
    exit 1
fi

echo "[OK] 基础依赖检查通过"

# --------------------------------------------------
# 1. 检查 RL-Insight
# --------------------------------------------------
echo "[1/7] 检查 RL-Insight"

if ! command -v rl-insight >/dev/null 2>&1; then
    echo "[ERROR] 当前环境没有安装 rl-insight。"
    echo "当前适配验证版本安装命令："
    echo '  python3 -m pip install "rl-insight==0.2.1"'
    echo "安装完成后重新运行本脚本。"
    exit 1
fi

if ! python3 -c "import rl_insight" >/dev/null 2>&1; then
    echo "[ERROR] rl-insight 命令存在，但 Python 无法 import rl_insight。"
    exit 1
fi

echo "[OK] RL-Insight 已安装"

# --------------------------------------------------
# 2. 检查并启动 node_exporter
# --------------------------------------------------
echo
echo "[2/7] 检查 CPU / Host exporter"

NODE_EXPORTER=""

if command -v prometheus-node-exporter >/dev/null 2>&1; then
    NODE_EXPORTER="$(command -v prometheus-node-exporter)"
elif command -v node_exporter >/dev/null 2>&1; then
    NODE_EXPORTER="$(command -v node_exporter)"
fi

if [ -z "${NODE_EXPORTER}" ]; then
    echo "[ERROR] 当前环境没有安装 node_exporter。"
    echo
    echo "Ubuntu 环境可以安装："
    echo "  apt-get update"
    echo "  apt-get install -y prometheus-node-exporter"
    echo
    echo "安装完成后重新运行本脚本。"
    exit 1
fi

echo "[OK] node_exporter 已安装: ${NODE_EXPORTER}"

if ss -lnt 2>/dev/null | grep -q ":${NODE_EXPORTER_PORT} "; then
    echo "[OK] node_exporter 已在监听 ${NODE_EXPORTER_PORT}"
else
    echo "[INFO] node_exporter 未启动，正在启动..."
    nohup "${NODE_EXPORTER}" \
        --web.listen-address=:${NODE_EXPORTER_PORT} \
        >/tmp/node_exporter.log 2>&1 &

    sleep 2

    if ss -lnt 2>/dev/null | grep -q ":${NODE_EXPORTER_PORT} "; then
        echo "[OK] node_exporter 启动成功"
    else
        echo "[ERROR] node_exporter 启动失败"
        echo "日志：/tmp/node_exporter.log"
        exit 1
    fi
fi

if curl -fsS http://127.0.0.1:${NODE_EXPORTER_PORT}/metrics \
    | grep -q '^node_cpu_seconds_total'; then
    echo "[OK] CPU 指标正常"
else
    echo "[ERROR] ${NODE_EXPORTER_PORT} 已启动，但没有读到 CPU 指标"
    exit 1
fi

# --------------------------------------------------
# 3. 检查并启动 MetaX exporter
# --------------------------------------------------
echo
echo "[3/7] 检查 MetaX GPU exporter"

if [ ! -f "${METAX_EXPORTER}" ]; then
    echo "[ERROR] 找不到：${METAX_EXPORTER}"
    exit 1
fi

if ss -lnt 2>/dev/null | grep -q ":${METAX_EXPORTER_PORT} "; then
    echo "[OK] MetaX exporter 已在监听 ${METAX_EXPORTER_PORT}"
else
    echo "[INFO] MetaX exporter 未启动，正在启动..."

    nohup python3 "${METAX_EXPORTER}" \
        --port ${METAX_EXPORTER_PORT} \
        --interval 5 \
        >/tmp/metax_metrics_exporter.log 2>&1 &

    sleep 6

    if ss -lnt 2>/dev/null | grep -q ":${METAX_EXPORTER_PORT} "; then
        echo "[OK] MetaX exporter 启动成功"
    else
        echo "[ERROR] MetaX exporter 启动失败"
        echo "日志：/tmp/metax_metrics_exporter.log"
        exit 1
    fi
fi

if curl -fsS http://127.0.0.1:${METAX_EXPORTER_PORT}/metrics \
    | grep -q '^metax_gpu_'; then
    echo "[OK] MetaX GPU 指标正常"
else
    echo "[ERROR] ${METAX_EXPORTER_PORT} 已启动，但没有读到 MetaX GPU 指标"
    echo "请检查 /tmp/metax_metrics_exporter.log 和 mx-smi"
    exit 1
fi

# --------------------------------------------------
# 4. 合并 MetaX 面板到 verl_trainer_v1_with_vllm_engine
# --------------------------------------------------
echo
echo "[4/7] 合并 MetaX 面板到 verl_trainer_v1_with_vllm_engine"

generate_metax_verl_dashboard

echo "[OK] 合并版 dashboard 已生成（UID 不变，直接覆盖原 dashboard）"
echo "[OK] 独立 MetaX GPU Metrics dashboard 文件已清理，避免重复注册"

# --------------------------------------------------
# 5. 启动 / 重启 RL-Insight
# --------------------------------------------------
echo
echo
echo "[5/7] 检查 RL-Insight Server"

PROM_ARCHIVE="prometheus-2.54.1.linux-amd64.tar.gz"
TEMPO_ARCHIVE="tempo_2.6.1_linux_amd64.tar.gz"
GRAFANA_ARCHIVE="grafana-13.0.0.linux-amd64.tar.gz"

echo "[INFO] RL-Insight 离线包目录:"
echo "       ${RL_INSIGHT_ARCHIVE_DIR}"

LOCAL_ARCHIVE_COUNT=0

for archive in \
    "${PROM_ARCHIVE}" \
    "${TEMPO_ARCHIVE}" \
    "${GRAFANA_ARCHIVE}"
do
    if [ -f "${RL_INSIGHT_ARCHIVE_DIR}/${archive}" ]; then
        echo "[OK] 找到 ${archive}"
        LOCAL_ARCHIVE_COUNT=$((LOCAL_ARCHIVE_COUNT + 1))
    else
        echo "[WARN] 未找到 ${archive}"
    fi
done

echo

if [ "${LOCAL_ARCHIVE_COUNT}" -gt 0 ]; then
    echo "[INFO] 使用 RL-Insight 官方离线安装机制"
    echo "[INFO] 本地有的包直接使用；缺失的包会尝试在线下载"

    if ! rl-insight server install \
        --local-archive "${RL_INSIGHT_ARCHIVE_DIR}"
    then
        echo
        echo "======================================================"
        echo "[ERROR] RL-Insight Server 组件安装失败"
        echo
        echo "当前离线包目录:"
        echo "  ${RL_INSIGHT_ARCHIVE_DIR}"
        echo
        echo "建议准备完整的三个安装包:"
        echo "  ${PROM_ARCHIVE}"
        echo "  ${TEMPO_ARCHIVE}"
        echo "  ${GRAFANA_ARCHIVE}"
        echo
        echo "如果离线包不在当前 Verl 仓库，可以这样指定:"
        echo
        echo "  RL_INSIGHT_ARCHIVE_DIR=/your/archive/path \\"
        echo "    bash tools/rl_insight/setup_metax_monitoring.sh"
        echo
        echo "当前环境也尝试过在线下载；如果失败，请检查"
        echo "GitHub / dl.grafana.com 网络访问。"
        echo "======================================================"
        exit 1
    fi
else
    echo "[WARN] 当前仓库没有找到 RL-Insight Server 离线包"
    echo "[INFO] 尝试通过官方地址在线安装..."

    if ! rl-insight server install
    then
        echo
        echo "======================================================"
        echo "[ERROR] RL-Insight Server 在线安装失败"
        echo
        echo "当前节点可能无法访问外网。"
        echo
        echo "请准备以下三个离线包:"
        echo "  ${PROM_ARCHIVE}"
        echo "  ${TEMPO_ARCHIVE}"
        echo "  ${GRAFANA_ARCHIVE}"
        echo
        echo "然后放到:"
        echo "  ${RL_INSIGHT_ARCHIVE_DIR}/"
        echo
        echo "再重新执行:"
        echo "  bash tools/rl_insight/setup_metax_monitoring.sh"
        echo "======================================================"
        exit 1
    fi
fi

echo
echo "[INFO] 检查 RL-Insight Server 是否已经运行"

if curl -fsS \
    http://127.0.0.1:18080/healthz \
    >/dev/null 2>&1
then
    echo "[OK] RL-Insight Server 已经运行"
else
    echo "[INFO] RL-Insight Server 未运行，正在启动..."

    if ! rl-insight server start --detach
    then
        echo "[ERROR] RL-Insight Server 启动失败"
        echo "[INFO] 请执行以下命令查看状态:"
        echo "       rl-insight server status"
        exit 1
    fi
fi

echo "[INFO] 等待 RL-Insight Server 就绪..."
sleep 5

if curl -fsS \
    http://127.0.0.1:18080/healthz \
    >/dev/null 2>&1
then
    echo "[OK] RL-Insight API 正常: 18080"
else
    echo "[ERROR] RL-Insight API :18080 未就绪"
    rl-insight server status || true
    exit 1
fi

if curl -fsS \
    http://127.0.0.1:9090/-/ready \
    >/dev/null 2>&1
then
    echo "[OK] Prometheus 正常: 9090"
else
    echo "[ERROR] Prometheus :9090 未就绪"
    exit 1
fi

if curl -fsS \
    http://127.0.0.1:3000/api/health \
    >/dev/null 2>&1
then
    echo "[OK] Grafana 正常: 3000"
else
    echo "[ERROR] Grafana :3000 未就绪"
    exit 1
fi


echo "[6/7] 注册 Prometheus hardware targets"

TARGET_FILE="/tmp/metax_hardware_targets.yaml"

cat > "${TARGET_FILE}" <<YAML
jobs:
  - job_name: node-exporter
    targets:
      - target: "${TARGET_HOST}:${NODE_EXPORTER_PORT}"
        labels:
          node: "${NODE_NAME}"

  - job_name: metax_gpu_metrics
    targets:
      - target: "${TARGET_HOST}:${METAX_EXPORTER_PORT}"
        labels:
          node: "${NODE_NAME}"
YAML

rl-insight server targets add "${TARGET_FILE}"

echo "[OK] hardware targets 已注册"

# --------------------------------------------------
# 7. 最终检查
# --------------------------------------------------
echo
echo "[7/7] 检查 Prometheus 是否采到数据"

sleep 5

CPU_UP="$(curl -sG http://127.0.0.1:9090/api/v1/query \
    --data-urlencode 'query=up{job="node-exporter"}')"

METAX_UP="$(curl -sG http://127.0.0.1:9090/api/v1/query \
    --data-urlencode 'query=up{job="metax_gpu_metrics"}')"

if echo "${CPU_UP}" | grep -q '"1"'; then
    echo "[OK] node-exporter target: UP"
else
    echo "[WARN] node-exporter target 暂未显示 UP"
fi

if echo "${METAX_UP}" | grep -q '"1"'; then
    echo "[OK] metax_gpu_metrics target: UP"
else
    echo "[WARN] metax_gpu_metrics target 暂未显示 UP"
fi

SERVER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
SERVER_IP="${SERVER_IP:-127.0.0.1}"

echo
echo "========================================"
echo " Setup finished"
echo "========================================"
echo
echo "Grafana:"
echo "  http://${SERVER_IP}:3000"
echo
echo "训练脚本中设置："
echo "  export RL_INSIGHT_SERVER_URL=http://${SERVER_IP}:18080"
echo
echo "Merged Dashboard:"
echo "  verl_trainer_v1_with_vllm_engine  (training metric + MetaX GPU Metrics)"
echo
echo "训练脚本接入 RL-Insight（单机/多机）:"
echo "  tools/rl_insight/README.md"
echo "  tools/rl_insight/README_training_multi_node.md"
echo
echo "Exporter:"
echo "  CPU/Host : http://${TARGET_HOST}:${NODE_EXPORTER_PORT}/metrics"
echo "  MetaX GPU: http://${TARGET_HOST}:${METAX_EXPORTER_PORT}/metrics"
echo
