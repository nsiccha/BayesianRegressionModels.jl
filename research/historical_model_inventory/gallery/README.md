# Historical inventory gallery

This directory is an isolated runtime consumer of `../model_matrix.tsv`. The
single mounted `HistoricalInventory` semantic graph owns the matrix rows, four
filter parameters, their evaluated option domains, the filtered rows, the card
tree, and its GET operation. `HistoricalInventoryGalleryApp` only mounts that
graph through `semantic_app`; it has no second declaration, manual route/form
mirror, `AppData`/`AppContext` layer, or DynamicObjects-versus-HTMXObjects
shadow model.

The row evidence remains authoritative. The descriptor-to-semantic-card
adapter is marked experimental everywhere it appears and cannot promote an
unsupported or unresolved validation tier.

The validation filter and primary badge use the current audited/inferred
translation tier. Each card shows the exact-metadata tier separately;
historical example stamps are never treated as current capability receipts.

Validate without opening a network listener:

```sh
julia --project=web-macro research/historical_model_inventory/gallery/validate.jl
```

The browser route uses HTMXObjects' standard runtime and default operation
policy: ordinary navigation returns the framework's HTMX-enabled page shell,
whose load operation mounts the complete semantic surface. Subsequent filter
submissions use the same authoritative graph and its inferred option domains.

Serve explicitly when desired:

```sh
julia --project=web-macro research/historical_model_inventory/gallery/serve.jl 8127 127.0.0.1
```

The optional third serving argument selects another compatible matrix path.
`served_smoke.tsv` records the exact Strato2 commit, matrix/source hashes,
listener URL, restart, HTTP statuses, and observed card/filter counts.
