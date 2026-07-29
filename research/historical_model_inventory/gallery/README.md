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

The browser shell embeds the vendored HTMX 2.0.8 runtime from
`vendor/htmx-2.0.8.min.js` so the public gallery does not depend on a
third-party CDN request. The upstream Zero-Clause BSD license is retained in
`vendor/HTMX-LICENSE`; the payload was retrieved from
`https://cdn.jsdelivr.net/npm/htmx.org@2.0.8/dist/htmx.min.js` and its upstream
SHA-256 is `22283ef68cb7545914f0a88a1bdedc7256a703d1d580c1d255217d0a50d31313`.

Serve explicitly when desired:

```sh
julia --project=web-macro research/historical_model_inventory/gallery/serve.jl 8127 127.0.0.1
```

The optional third serving argument selects another compatible matrix path.
`served_smoke.tsv` records the exact Strato2 commit, matrix/source hashes,
listener URL, restart, HTTP statuses, and observed card/filter counts.
