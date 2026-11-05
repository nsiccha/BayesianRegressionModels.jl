# test/downstream_group_block_term.jl — contract for the
# downstream grouped-term pattern (snag brm-parametric-t-ae5e02f9).
#
# A downstream package ships grouped parametric terms through BRM's public
# seams ONLY: its own markers plus small methods on
# `_sb_term_group_block` (fields form), `_sb_is_direct_term`,
# `_brm_prepares_term` / `_brm_prepare_term` (via
# `_brm_prepare_structured_term`), `_brm_replay_term`,
# `_brm_native_structured_effect` (pure-Julia row math), and
# `_sb_predictor_term!` (Stan emit reusing StanBlocks builtins). No BRM
# vocabulary change, no custom @slic/@deffun. This file ACTS as that
# downstream module with an example pair — `transient` (single-peak
# bump, 3 per-group params) and `saturating` (0-to-1 dose multiplier, 2
# per-group params) — plus their additive and multiplicative compositions,
# and pins exact behavior:
#   - SBBRMI lowers to stanc-clean Stan calling the shared builtins;
#   - the shared preparation hands `_brm_native_structured_effect` the term's
#     rows and group indices, so its row math is bit-exact versus independent
#     Julia math (max abs diff 0.0).
# Core openings this relies on: `_sb_is_direct_term` routing and the
# group-block branch of `_sb_emit_direct!` (sbimpl), and the shared
# structured-term preparation. Promotion of such terms into BRM core waits on
# multi-package adoption; see docs/src/formula-terms.md "Downstream grouped terms".

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using LinearAlgebra: Diagonal
using Distributions: Exponential, Normal
import Distributions: logpdf
import StanBlocks.stan: transpiles

const BRM = BayesianRegressionModels

module DownstreamTerms
using StanBlocks
import BayesianRegressionModels
const BRM = BayesianRegressionModels
import BayesianRegressionModels: _sb_term_group_block, _sb_predictor_term!,
    _sb_is_direct_term, _sb_find_group_block, _sb_inner_data,
    _brm_prepares_term, _brm_prepare_term, _brm_replay_term,
    _brm_native_structured_effect, _brm_term_data, _brm_apply_levels,
    _brm_prepare_structured_term

# Independent Julia reference math for the two shapes.
bump_math(x, loc, ls, mag) = begin
    xi = (x - loc) * exp(ls)
    s = 1 / (1 + exp(-xi))
    sm = 1 / (1 + exp(xi))
    exp(log(s) + log(sm)) * mag
end
sigmoid_math(x, loc, ls) = 1 / (1 + exp(-((x - loc) * exp(ls))))

function transient end
_sb_is_direct_term(::typeof(transient)) = true
_sb_term_group_block(::typeof(transient)) = (; fields=[
    (; name=:transient, n_per_group=3, group=(; kwarg=:series),
       prior=:correlated_normal),
])

function saturating end
_sb_is_direct_term(::typeof(saturating)) = true
_sb_term_group_block(::typeof(saturating)) = (; fields=[
    (; name=:saturating, n_per_group=2, group=(; kwarg=:series),
       prior=:correlated_normal),
])

for marker in (transient, saturating)
    @eval begin
        _brm_prepares_term(::BRM.ExprColumn{typeof($marker)}) = true
        function _brm_prepare_term(term::BRM.ExprColumn{typeof($marker)},
                                   target, context)
            base = BRM._brm_prepare_structured_term(term, target, context,
                _sb_term_group_block($marker, term))
            xkey, xraw = BRM._brm_term_data(
                nameof($marker), only(BRM.getargs(term)), context)
            BRM._BRMPreparedTerm(base.callable, base.source,
                merge(base.state, (; xname=xkey, x=collect(Float64, xraw))),
                base.dependencies)
        end
        function _brm_replay_term(::typeof($marker), training, fresh, context)
            fields = map(training.state.fields) do field
                raw = context.data[field.source]
                merge(field, (; idx=BRM._brm_apply_levels(field.levels, raw)))
            end
            xraw = context.data[training.state.xname]
            BRM._BRMPreparedTerm(training.callable, training.source,
                merge(training.state,
                    (; fields=Tuple(fields), x=collect(Float64, xraw))),
                training.dependencies)
        end
    end
end

function _brm_native_structured_effect(
        term::BRM._BRMPreparedTerm{typeof(transient)}, block, field)
    st = term.state
    [bump_math(st.x[i], block[field.idx[i], 1], block[field.idx[i], 2],
               block[field.idx[i], 3]) for i in eachindex(field.idx)]
end
function _brm_native_structured_effect(
        term::BRM._BRMPreparedTerm{typeof(saturating)}, block, field)
    st = term.state
    [sigmoid_math(st.x[i], block[field.idx[i], 1], block[field.idx[i], 2])
     for i in eachindex(field.idx)]
end

function _sb_predictor_term!(stmts, data, ::typeof(transient), t;
                             target::Symbol, group_block_lookup=Dict(),
                             kwargs...)
    xname, xraw = BRM._sb_inner_data(:transient, only(BRM.getargs(t)))
    data[xname] = collect(Float64, xraw)
    info = BRM._sb_find_group_block(transient, t, group_block_lookup)
    isnothing(info) && error("sbimpl: `transient` found no allocated block")
    (; block_name, idx_name) = info
    loc = Symbol(:transient_, target, :_loc)
    ls = Symbol(:transient_, target, :_ls)
    mag = Symbol(:transient_, target, :_mag)
    col = Symbol(:transient_, target, :_, xname)
    push!(stmts, :($loc = $(block_name)[$(idx_name), 1]))
    push!(stmts, :($ls = $(block_name)[$(idx_name), 2]))
    push!(stmts, :($mag = $(block_name)[$(idx_name), 3]))
    push!(stmts, :($col = biomarker_time_response($xname, $loc, $ls, $mag)))
    col
end
function _sb_predictor_term!(stmts, data, ::typeof(saturating), t;
                             target::Symbol, group_block_lookup=Dict(),
                             kwargs...)
    xname, xraw = BRM._sb_inner_data(:saturating, only(BRM.getargs(t)))
    data[xname] = collect(Float64, xraw)
    info = BRM._sb_find_group_block(saturating, t, group_block_lookup)
    isnothing(info) && error("sbimpl: `saturating` found no allocated block")
    (; block_name, idx_name) = info
    loc = Symbol(:saturating_, target, :_loc)
    ls = Symbol(:saturating_, target, :_ls)
    col = Symbol(:saturating_, target, :_, xname)
    push!(stmts, :($loc = $(block_name)[$(idx_name), 1]))
    push!(stmts, :($ls = $(block_name)[$(idx_name), 2]))
    push!(stmts, :($col = exp(biomarker_dose_response($xname, $loc, $ls))))
    col
end
end

using .DownstreamTerms: transient, saturating
import .DownstreamTerms: bump_math, sigmoid_math

# Long-format data: 2 series x 6 observations.
term_df() = (;
    logt=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0],
    logd=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    series=[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    y=[0.05, 0.2, 0.5, 0.4, 0.2, 0.1, 0.03, 0.15, 0.6, 0.45, 0.22, 0.12],
)

nested_model(df=term_df()) = @brm df begin
    sigma ~ Exponential(1)
    mu ~ 1 + transient(logt; series)
    y ~ Normal(mu, sigma)
end

composed_model(df=term_df()) = @brm df begin
    sigma ~ Exponential(1)
    base ~ Normal(0, 1)
    bump ~ 0 + transient(logt; series)
    resp ~ 0 + saturating(logd; series)
    mu = base + bump * resp
    y ~ Normal(mu, sigma)
end

# The prepared term of the downstream callable `f` on predictor `target`.
function prepared_downstream_term(brmi, target, f)
    context = BRM._brm_backend_context(brmi)
    rhs = last(BRM.getargs(BRM.linear_predictor_op(brmi, target), 2))
    term = only(t for t in BRM._brm_additive_terms(rhs)
                if t isa BRM.ExprColumn && BRM.getf(t) === f)
    BRM._brm_prepare_term(term, target, context)
end

@testset "nested downstream transient lowers to stanc-clean Stan" begin
    sb = SBBRMI(nested_model(); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("b_transient_series_L ~ lkj_corr_cholesky", code)
    @test occursin("transient_mu_logt", code)
    @test occursin("biomarker_time_response", code)
    @test sb.data[:n_series] == 2
    @test sb.data[:n_terms_transient_series] == 3
    @test sb.data[:series_idx] == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "nested downstream transient row math is exact" begin
    df = term_df()
    term = prepared_downstream_term(nested_model(df), :mu, transient)
    field = only(term.state.fields)
    # A per-group block (row = group, column = per-group parameter).
    block = randn(MersenneTwister(11), 2, 3)
    effect = BRM._brm_native_structured_effect(term, block, field)
    expected = [bump_math(df.logt[i], block[df.series[i], 1],
                          block[df.series[i], 2], block[df.series[i], 3])
                for i in eachindex(df.logt)]
    @test effect ≈ expected
    @test maximum(abs.(effect .- expected)) == 0.0
end

@testset "saturating pair and multiplicative composition lower to Stan" begin
    df = term_df()
    sb = SBBRMI(composed_model(df); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("biomarker_time_response", code)
    @test occursin("biomarker_dose_response", code)
    @test occursin("bump .* resp", code)
    @test occursin("b_transient_series_L ~ lkj_corr_cholesky", code)
    @test occursin("b_saturating_series_L ~ lkj_corr_cholesky", code)
    @test sb.data[:n_terms_transient_series] == 3
    @test sb.data[:n_terms_saturating_series] == 2
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # The saturating row math, through the same shared preparation.
    term = prepared_downstream_term(composed_model(df), :resp, saturating)
    block = randn(MersenneTwister(3), 2, 2)
    effect = BRM._brm_native_structured_effect(term, block, only(term.state.fields))
    @test maximum(abs.(effect .- [sigmoid_math(df.logd[i], block[df.series[i], 1],
        block[df.series[i], 2]) for i in eachindex(df.logd)])) == 0.0
end

@testset "downstream terms keep descriptor and frozen replay" begin
    sb = SBBRMI(nested_model(); mod=@__MODULE__)
    @test !isnothing(brm_descriptor(sb))
    fitted = (; logt=[-1.5, 0.25], logd=[0.0, 0.5], series=[1, 2], y=[0.2, 0.4])
    sb2 = reprocess(sb, fitted)
    @test sb2.data[:n_series] == 2
    @test sb2.data[:series_idx] == [1, 2]
    # Unseen group levels stay guarded by the shared ranef floor: the fitted
    # model has no coordinate for a new group, on any backend.
    unseen = (; logt=[-1.5, 0.25], logd=[0.0, 0.5], series=[1, 3], y=[0.2, 0.4])
    @test_throws "unseen level" reprocess(sb, unseen)
end
