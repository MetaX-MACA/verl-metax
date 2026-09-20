# Compatibility

The compatibility matrix is a release contract, not a list of optimistic
version ranges.

| verl-metax | verl | MACA | torch+metax | vllm-metax | Status |
| --- | --- | --- | --- | --- | --- |
| 0.9.0 | 0.9.0 | 3.8.0 | 2.10 | 0.23.0 | validated |

Additional versions validated with this stack:

| Component | Version |
| --- | --- |
| Megatron-LM | 0.18.0 |
| transformers | 5.8.1 |
| NumPy | <2 |
| Ray | >=2.56.1 |

The Muon recipe is an exception to the Megatron-LM row: it requires a
development checkout containing `use_layer_wise_param_layout`; see
[the Muon guide](recipes-muon.md).

The compatibility row describes the validated software contract, not every
recipe in the repository. See the [validation matrix](validation.md) for the
scenarios and workaround paths exercised on hardware.

## Versioning policy

The public verl-metax version tracks the verl release it supports. This makes
the compatibility boundary visible in package listings and deployment logs:

- `verl-metax==0.9.0` supports `verl==0.9.0`;
- plugin-only fixes that keep the same verl boundary use PEP 440 post releases,
  such as `0.9.0.post1`;
- release candidates use `0.9.0rc1`, `0.9.0rc2`, and so on;
- support for a new verl release moves the matching version component, for
  example to `verl-metax==0.10.0` for `verl==0.10.0`.

MACA, torch, vllm-metax, and other stack versions remain in this compatibility
matrix instead of being encoded into the verl-metax package version.

## NumPy policy

NumPy `<2` has been validated in the MetaX environment with the exact stack in
the table above. It is therefore a supported MetaX runtime configuration, not
an unresolved runtime blocker.

Upstream verl 0.9.0 nevertheless declares NumPy `>=2` in its distribution
metadata. Declaring both requirements in this package would make dependency
resolution impossible even though the tested runtime works. The MetaX
deployment constraint is kept in `constraints/verl-0.9.0-metax.txt`; use it
with a verl 0.9.0 wheel whose metadata has been corrected, or install the
validated stack without re-resolving verl's dependencies.
