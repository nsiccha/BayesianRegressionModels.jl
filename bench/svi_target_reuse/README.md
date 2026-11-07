# SVI target reuse and what an image can hold

Public toy models only. Timings move with host load, so compare within one batch.

- `split.jl <family> [out]`: one fresh process per family (`gauss`, `owned`,
  `connected`, `spacetime`). First-call preparation, compilation inside it, warm
  per-iteration cost, the repeated same-program rebuild, reused against rebuilt
  continuation, the size of the retained target, and in one process dataset A, then
  B (same sizes), C (other sizes) and A again: bind, model, first value, target,
  first prior and unit gradient, worker plan and its first gradient, first step,
  warm step, with fresh optimizer state each time. Returning to A checks that its
  first step reproduces A's bit for bit. Numeric priors at fixed shape are not
  varied here.
- `scale.jl [out]`: the repeated same-program target build on the connected toy, up
  to about the coordinate count of a large connected fit, piece by piece.
- `image.jl <dir> [out]`: throwaway packages whose images hold progressively more
  of a fixed model and its optimization loop, then fresh processes that fit
  datasets the workload did or did not see, with every phase timed.
