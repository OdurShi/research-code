# Minimal example

Validate the example taxonomy:

```bash
sbls validate-taxonomy examples/minimal/taxonomy.json
```

The example stream contains six records, fewer than the default residual batch size. To execute the full pipeline on this input, copy `configs/default.yaml`, set `residual_batch_size: 3`, and configure compatible model identifiers available in the execution environment.
