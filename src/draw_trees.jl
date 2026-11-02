"""
    brm_output_tree(descriptor, constrained_draws, constrained_names; logical, role=nothing)

Label a logical output without copying its draws. Requires `using TreeArrays`.
Rows are draws; a vector of per-chain matrices adds an outer `:chain` axis.
`logical` may also be a discovered `BRMOutput`, for example a coefficient
carrier from `brm_outputs`. Coefficients use their descriptor labels;
observation outputs use their element positions. Scalar outputs have only
the `:draw` axis. Labels follow the supplied constrained-name order for these
single-axis outputs.

Random-effect carriers use the fitted `:level` labels and `(predictor,
coefficient)` addresses on `:margin`. Standardized draws and effects have
`:draw × :level × :margin` (omit `:margin` for a one-margin block), scales
have `:margin`, and Cholesky factors have `:margin × :margin2`. Stratified
scales and factors also carry `:stratum`. These carriers use fitted coordinate
order, independently of the supplied name order, with views into the input.

Per-row block contributions use `:observation`. Model-scope assignments also
inherit `:observation` through elementary pointwise arithmetic and functions
of row outputs and scalars, including chains of assignments. Vector length
alone does not establish row alignment.

Ragged predictions use flat `:observation` positions `1:N`, where `N` is
the total number of elements. Their aggregate log likelihood uses `1:G`
on its own `:observation` axis, one contribution per group, not per element.
The output's `segments` retains the inclusive group ends of the predictive
element axis; it does not partition the aggregate likelihood's `G` columns.

The returned backing aliases the input: treat both as read-only. Unknown
output-axis semantics are reported by name, never replaced by guessed labels.
"""
function brm_output_tree end

"""
    brm_ranef_tree(descriptor, constrained_draws, constrained_names; id=nothing, group=nothing)
    brm_ranef_tree(entries::AbstractVector{<:NamedTuple})

Random-effect draws as a TreeArrays tree of block records. Each record has
`sd`, `cor`, `effects`, and, for level-correlated blocks, `level_cor`.
The axes are `:draw`, `:margin` and `:level`; correlation matrices distinguish
their second coordinate with `:margin2` / `:level2`. Margin coordinates are
`(predictor, coefficient)` records, and levels are the fitted group labels.
An outer `:block` axis identifies each binding and stratum. Per-chain input
adds `:chain`. Requires `using TreeArrays`.

Stored effects and scales alias constrained draws. Derived correlations and
scales are computed once by the same extraction as `brm_ranef_draws`; adding
labels does not copy them. Treat the returned backing as read-only.
The second form labels existing `brm_ranef_draws` entries without recomputing.
"""
function brm_ranef_tree end

"""
    brm_predictive_tree(descriptor, unconstrained_draws; problem, seed)
    brm_predictive_tree(descriptor, predicted::NamedTuple)

A TreeArrays record of posterior-predictive responses, with a leading `:draw`
axis and descriptor-owned response coordinates. The second form labels the
arrays already returned by `brm_predictive_draws`, without copying or drawing
again. Per-chain matrix input retains chains separately. Requires TreeArrays.
Ragged responses use flat `:observation` positions `1:N`; group boundaries
remain available as `segments` on the descriptor's predictive output.
"""
function brm_predictive_tree end

"""
    brm_conditional_tree(descriptor, unconstrained_draws, grid; kwargs...)
    brm_conditional_tree(result::NamedTuple)

Label `brm_conditional_draws` on `:draw × :grid`, retaining its result metadata
and replacing only `result.draws` with a TreeData over the same matrix. Grid
coordinates are records of the actual grid values (`focal` columns when
specified). Per-chain inputs add `:chain`. Requires `using TreeArrays`.
"""
function brm_conditional_tree end

"""
    rk_draw_tree(rk::RKBRMI, U, names)
    rk_draw_tree(values::NamedTuple)

The named values returned by `rk_draw_values` as a TreeArrays record of
per-name `:draw` trees. Like `rk_draw_values`, **columns** of `U` are draws.
Scalar values are stored directly; vector-valued draws stay separate, with
their actual element indices, rather than being stacked. A vector of
per-chain matrices adds `:chain`. Requires `using TreeArrays`.
The second form labels an existing `rk_draw_values` result without querying
the model again or copying its values.
"""
function rk_draw_tree end
