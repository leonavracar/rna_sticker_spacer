# Test strategy

1. **Unit and regression tests** in `analysis/tests/` run with synthetic
   arrays and small temporary SQLite databases, without HOOMD or GSD.
   Run `python -m pytest analysis/tests -q`.
2. **End-to-end smoke fixture**: the short mixture example in
   `examples/mixture/eatt2/` should exercise GSD -> pairing -> graph ->
   analysis -> CSV export. It is not representative scientific evidence
   and must never be combined with full-sweep results.
3. **Full-sweep validation** on HPC: compare results against a verified
   reference, fixed analysis window, and chosen runs after any scientific
   algorithm changes.

The short GSD example is a useful integration fixture, but the full
GSD-to-CSV test has *not* yet been wired into automated CI. It requires the
optional trajectory dependencies; keep it as a separate, explicitly
documented integration test rather than silently making the lightweight
test suite depend on GSD.

A manual smoke test from the project root can run with:

```bash
python analysis/run_pipeline.py \
  --sweep examples/mixture/eatt2 \
  --output-root /tmp/rna-analysis-smoke \
  --experiment mixture \
  --stage all --overwrite-npz
```

**Before treating this command as validated**, confirm that this example
has the run-directory layout, `params.json`, full configuration metadata,
and required GSD frames expected by `iter_trajectories`; if not, package
a dedicated fixture with that layout. Assertions should check output
`replicates.csv` and `summary.csv`, finite measurements, and expected
replicate counts. Do not replace the scientific regression tests with a
mere successful-exit assertion.
