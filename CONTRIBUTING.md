# Contributing

Changes should preserve the boundary between hardware integration and the verl
framework:

- MetaX-specific runtime behavior belongs in this repository.
- Hardware-neutral fixes and missing extension points should be proposed to
  `verl-project/verl`.
- General multi-chip reference improvements may be proposed to
  `verl-project/verl-hardware-plugin`.
- Examples and tuning values must state the tested hardware and software stack.
- Recipes may retain non-sensitive validated filesystem paths as overridable
  defaults. Credentials, private registry addresses, and Gerrit URLs must not
  be committed.
- Treat `tools/rl_insight/` as one operator-facing unit: keep its setup
  script, exporter, dashboard, training example, and three runbooks consistent.
- Preserve detailed operational steps in `docs/how-to-run/`; when a recipe is
  renamed or reorganized, update its runbook and the how-to-run index together.
- Never silently broaden a versioned directory under `patches/`. Verify its
  files against the exact vllm-metax version, and create a new version directory
  when the target stack changes.
- Keep shared launch and sleep-mode instructions in `examples/env/README.md`;
  recipe runbooks should link to it instead of copying activation steps.

Run before submitting a change:

```bash
ruff check verl_metax tests tools/verify_workarounds.py
PYTHONPATH=. python -m unittest discover -s tests -v
```

Workaround changes must satisfy
[`docs/workaround-policy.md`](docs/workaround-policy.md). A workaround without
affected versions, a reproducer, a regression test, and a removal condition is
not accepted.

When a change is also proposed to verl, follow verl's own contribution policy,
including duplicate issue/PR checks and disclosure of AI assistance.
