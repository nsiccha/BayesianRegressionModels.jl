# test/shared_preparation.jl — the backend-neutral preparation layer that
# SBBRMI and RKBRMI share: parameter reference axes, fitted s/t2/me term
# preparation and its replay on new data, structured-field normalization and
# grouping, and the explicit refusal of a structured term without a native
# Julia implementation.
#
# Run: julia --project=test test/shared_preparation.jl
using Test
using BayesianRegressionModels
using Distributions
using LinearAlgebra
using Random

const BRM = BayesianRegressionModels

@testset "parameter axes are distribution shapes" begin
    @test BRM._brm_parameter_reference_axis(ExprColumn(Normal, 0, 1)) === :scalar
    @test BRM._brm_parameter_reference_axis(ExprColumn(Dirichlet, 3, 1)) === :whole
    @test BRM._brm_parameter_reference_axis(ExprColumn(MvNormal, zeros(2), I)) === :whole
    @test BRM._brm_parameter_reference_axis(
        ExprColumn(LKJCovarianceFactor, 2)) === :whole
end

function prepared_term(builder, data, target, callable)
    brmi = builder(data)
    context = BRM._brm_backend_context(brmi)
    rhs = last(BRM.getargs(BRM.linear_predictor_op(brmi, target), 2))
    term = only(t for t in BRM._brm_additive_terms(rhs)
                if t isa BRM.ExprColumn && BRM.getf(t) === callable)
    BRM._brm_prepare_term(term, target, context), term
end

@testset "shared fitted s/t2/me preparation" begin
    x = collect(range(-1.0, 1.0; length=12))
    z = x .^ 2 .+ 0.2 .* x
    data = (; x, z, y=zeros(12))

    s_builder = @brm begin mu ~ 1 + s(x); y ~ Normal(mu, 1) end
    spline, spline_expr = prepared_term(s_builder, data, :mu, s)
    @test size(spline.state.Xnull) == (12, 1)
    @test size(spline.state.Zpen) == (12, 8)

    t2_builder = @brm begin
        mu ~ 1 + t2(x, z; k=(5, 5), basis=(:cr, :cr), full=false)
        y ~ Normal(mu, 1)
    end
    tensor, tensor_expr = prepared_term(t2_builder, data, :mu, t2)
    @test size(tensor.state.Xfixed) == (12, 3)
    @test size(tensor.state.Zrr, 1) == 12

    me_builder = @brm begin mu ~ 1 + me(x, 0.2); y ~ Normal(mu, 1) end
    measured, _ = prepared_term(me_builder, data, :mu, me)
    @test measured.state.x_obs ≈ data.x
    @test measured.state.sd_x == 0.2

    prior_builder = @brm begin
        sd(mu, s(x)) ~ Exponential(2)
        latent(mu, me(x)) ~ Normal(1, 3)
        mu ~ 1 + s(x) + me(x, 0.2)
        y ~ Normal(mu, 1)
    end
    prior_spline, _ = prepared_term(prior_builder, data, :mu, s)
    prior_measured, _ = prepared_term(prior_builder, data, :mu, me)
    @test BRM.getf(prior_spline.state.sd_prior) <: Exponential
    @test BRM.getargs(prior_spline.state.sd_prior) == (2,)
    @test BRM.getf(prior_measured.state.latent_prior) <: Normal
    @test BRM.getargs(prior_measured.state.latent_prior) == (1, 3)

    # Replay applies the fitted basis before considering the new data's
    # degeneracy. A fresh fit would reject each constant margin.
    replay_data = (; x=fill(0.25, 5), z=fill(-0.1, 5), y=zeros(5))
    replay_context = BRM._brm_backend_context(s_builder(replay_data))
    replayed = BRM._brm_replay_term(spline, spline_expr, replay_context)
    @test size(replayed.state.Xnull) == (5, 1)
    @test size(replayed.state.Zpen) == (5, 8)
    @test all(isfinite, replayed.state.Zpen)

    tensor_replay_context = BRM._brm_backend_context(t2_builder(replay_data))
    tensor_replayed = BRM._brm_replay_term(
        tensor, tensor_expr, tensor_replay_context)
    @test size(tensor_replayed.state.Xfixed) == (5, 3)
    @test all(isfinite, tensor_replayed.state.Zrr)
end

@testset "shared structured-field normalization and grouping" begin
    legacy = (; n_per_group=2, group_arg_pos=1)
    fields = BRM._brm_structured_fields(legacy, BRM.sb_group_demo)
    @test only(fields).name === :sb_group_demo
    @test only(fields).prior === :correlated_normal
    @test only(fields).group == (; arg_pos=1)

    data = (; g=["a", "b", "a"], x=[1.0, 2.0, 3.0])
    brmi = @brm data begin
        mu ~ sb_group_demo(g)
        x ~ Normal(mu, 1)
    end
    call = BRM.getargs(BRM.parent(brmi.operations.mu), 2)[2]
    column = BRM._brm_structured_group_column(
        only(fields).group, call, Dict(:g => data.g))
    @test BRM.name(column) === :g
    @test BRM._brm_structured_group_name(only(fields).group, call) === :g
    sb = SBBRMI(brmi; mod=@__MODULE__)
    @test haskey(sb.data, :n_terms_sb_group_demo_g)
    @test haskey(sb.data, :g_idx)
end

function native_missing_structured end

@testset "SLIC-only structured terms fail explicitly in native execution" begin
    term = BRM._BRMPreparedTerm(
        native_missing_structured, (:g,), (;), (:g,))
    @test_throws "has no native Julia implementation" BRM._brm_native_structured_effect(
        term, zeros(1, 1), (; idx=[1]))
end
