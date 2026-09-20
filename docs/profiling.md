# Profiling on MetaX

The validated verl 0.9.0 stack can collect CPU and CUDA-compatible MetaX traces
through verl's existing torch profiler configuration. Use
`examples/profiling/qwen3_8b_torch.sh` for a GRPO actor-only smoke run, or
`examples/profiling/qwen2_5_7b_gae_torch.sh` for actor/ref/critic coverage.

The important controls are:

| Setting | Meaning |
| --- | --- |
| `global_profiler.steps=[2]` | training steps to capture |
| `global_profiler.save_path` | trace output directory |
| `actor_rollout_ref.actor.profiler.all_ranks` | capture every rank or use an explicit rank list |
| `tool_config.torch.contents=[cuda,cpu]` | activity types; adding memory, shapes, or stacks increases overhead |
| `tool_config.torch.discrete` | split actor/ref/update phases into separate traces |

For schedule profiling, pass the 0.9.0 schedule fields as extra Hydra arguments:

```bash
bash examples/profiling/qwen3_8b_torch.sh \
  actor_rollout_ref.actor.profiler.tool_config.torch.schedule.wait=1 \
  actor_rollout_ref.actor.profiler.tool_config.torch.schedule.warmup=0 \
  actor_rollout_ref.actor.profiler.tool_config.torch.schedule.active=1 \
  actor_rollout_ref.actor.profiler.tool_config.torch.schedule.repeat=0
```

Use an absolute output directory for rollout profiling because Ray workers may
otherwise write relative paths under their temporary directories.

For rollout token-window profiling, enable the rollout profiler and set
`profile_token_start` and `profile_token_end`. The interval is half-open and
only limits trace collection; generation still produces the full response.
For example:

```bash
bash examples/profiling/qwen3_8b_torch.sh \
  actor_rollout_ref.actor.profiler.enable=False \
  actor_rollout_ref.rollout.profiler.enable=True \
  actor_rollout_ref.rollout.profiler.tool=torch \
  actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True \
  actor_rollout_ref.rollout.profiler.tool_config.torch.profile_token_start=12 \
  actor_rollout_ref.rollout.profiler.tool_config.torch.profile_token_end=22 \
  actor_rollout_ref.rollout.profiler.save_path=/abs/outputs/profile
```

Profiler lifecycle changes are not patched by this package. Any such change
belongs upstream and requires an end-to-end test.
