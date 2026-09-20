# MetaX observability

The optional exporter runs independently from the verl trainer and converts one
`mx-smi` sample per interval into Prometheus gauges.

```bash
pip install 'verl-metax[observability]'
verl-metax-metrics --address 0.0.0.0 --port 9101 --interval 5
```

Configure Prometheus to scrape every MetaX node:

```yaml
scrape_configs:
  - job_name: metax_gpu_metrics
    static_configs:
      - targets: ["worker-0:9101", "worker-1:9101"]
```

Import `verl_metax/observability/assets/metax_gpu_metrics.json` into Grafana.
The parser is validated against the checked-in `mx-smi` 2.2.9 fixture. Add a
new anonymized fixture and test before claiming support for another output
format.

For RL-Insight server installation, Grafana dashboard setup, multi-node target
registration, and a complete two-node training example, use the
[RL-Insight tool bundle](../tools/rl_insight/README.md). The
integration scripts are explicit operator commands and are never run during
package import.
