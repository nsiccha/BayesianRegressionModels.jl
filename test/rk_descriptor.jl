# test/rk_descriptor.jl — `brm_descriptor(rk::RKBRMI)`: the content identity and
# input inventory of the program an RKBRMI runs, with no Stan program (snag
# `rkbrmi-has-no-de-65a4d1fc`, design decision `0c2ol3s`).
#
# Run: julia --startup-file=no --project=test test/rk_descriptor.jl
#
# A whole-series law registered only through `_rk_observation_source!` has no
# Stan form, so the Stan descriptor of its model fails at transpile (the
# control). The RK descriptor of the same model reads the backend's own
# program. Its id is the hash of that program and its callables: the same for
# another build, another name and other data values of the same model, and
# different for a held-out response, a non-centered completion and another
# formula. Its inputs are the backend's bound inputs, with the dataframe column
# each one is and the kind of value BRM generates for the rest.

using Test, BayesianRegressionModels, Distributions, StanBlocks
using ReactiveKernels, ReactiveKernelsPPL
using CategoricalArrays
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

module RKOnlyLaws
using BayesianRegressionModels, Distributions
import ReactiveKernels
import BayesianRegressionModels: _rk_observation_source!, brm_distribution_type

# The Gaussian log likelihood of a whole series around one level, as Julia code
# the law binds as a leaf. No `@lpxf` family is registered, so no Stan form.
function series_loglik(y, level, noise)
    total = 0.0
    for v in y
        total += -log(noise) - log(2pi) / 2 - ((v - level) / noise)^2 / 2
    end
    total
end

function series_law end
brm_distribution_type(::typeof(series_law)) = ContinuousMultivariateDistribution
function _rk_observation_source!(definitions, bindings, entry, ::typeof(series_law))
    loglik = Symbol(entry, :_loglik)
    push!(bindings, loglik => series_loglik)
    push!(definitions, :(ReactiveKernels.@kernel $entry(y, level, noise) = begin
        ll = $loglik(y, level, noise)
        return ll
    end))
    :done
end
end
using .RKOnlyLaws: series_law

series_model(data) = @brm data begin
    shift ~ Normal(0, 1)
    noise ~ Exponential(1)
    @plate for s in eachindex(y)
        y[s] ~ series_law(base[s] + shift, noise)
    end
end
series_data = (; y=[[0.5, -0.2, 0.1], [1.3, 0.9], [0.0]], base=[0.1, 1.0, -0.3])
# Other values of the same shape.
series_other = (; y=[[0.4, -0.1, 0.3], [1.0, 0.2], [0.5]], base=[0.0, 0.7, -0.1])

input(d, name) = only(i for i in d.inputs if i.name === name)
input_names(d) = Set(i.name for i in d.inputs)
# Every input is a dataframe column, a value BRM generates or the program
# computes (its kind in `transform`), or an extent of another input.
classified(d) = all(i -> i.column !== nothing || i.transform !== nothing || i.derived,
    d.inputs)

@stestset "an RK-only law: the Stan descriptor fails, the RK descriptor reads the backend" begin
    @test_throws "lpxf_expr" brm_descriptor(SBBRMI(series_model(series_data);
        mod=@__MODULE__))
    rk = RKBRMI(series_model(series_data))
    d = brm_descriptor(rk; name=:series)
    @test d isa BRMRKDescriptor
    @test startswith(d.id, "rk:") && length(d.id) == 3 + 64
    @test d.name === :series
    @test d.formula == sprint(show, brm_source(rk))
    @test d.held_out == ()
    # One input per bound input, in its order.
    @test Tuple(i.name for i in d.inputs) == Tuple(keys(rk_bound_data(rk)))
    y = input(d, :y)
    @test y.observed && !y.held_out && y.column === :y && y.transform === nothing
    @test y.size == (3,) && y.type === Symbol(string(eltype(series_data.y)))
    base = input(d, :base)
    @test !base.observed && base.column === :base && base.transform === nothing
    # The plate cell reads the series' nested values through a port of its own,
    # which is the response's column too.
    ports = [i for i in d.inputs if i.column === :y && i.name !== :y]
    @test !isempty(ports) && all(i -> !i.observed && i.transform === nothing, ports)
    @test classified(d)
    @test (:base, :y) ⊆ d.columns
    @test brm_columns(d) == d.columns
    @test :y in required_brm_inputs(d)
    # The program the id hashes, as a copy of the backend's.
    @test d.program.main == BRM._rk_retained_program(rk).main
    @test d.program.main !== BRM._rk_retained_program(rk).main
    @test any(b -> last(b) === RKOnlyLaws.series_loglik, d.program.bindings)
    # Identity: not the name, not the build, not the data's values.
    @test brm_descriptor(rk).name === :model
    @test brm_descriptor(rk).id == d.id
    @test brm_descriptor(RKBRMI(series_model(series_data))).id == d.id
    @test brm_descriptor(RKBRMI(series_model(series_other))).id == d.id
    # A Stan-only reader has no method for it.
    @test_throws MethodError brm_output(d, :shift)
    @test occursin("BRMRKDescriptor `series`", sprint(show, d))
end

plain_model(data) = @brm data begin
    sigma ~ Exponential(1)
    tau ~ Exponential(1)
    mu ~ 1 + x + g
    y ~ Normal(mu, sigma)
    nu ~ 1 + x
    z ~ Normal(nu, tau)
end
plain_fewer(data) = @brm data begin
    sigma ~ Exponential(1)
    tau ~ Exponential(1)
    mu ~ 1 + x + g
    y ~ Normal(mu, sigma)
    nu ~ 1
    z ~ Normal(nu, tau)
end
plain_data = (; x=[-1.0, 0.5, 2.0, 0.25, 1.0, -0.3],
    g=categorical(["a", "b", "a", "c", "b", "c"]),
    y=[0.2, 1.1, -0.4, 0.8, 1.6, 0.3], z=[1.0, 0.4, 0.9, -0.2, 0.0, 0.6])
plain_other = (; x=[0.3, -0.5, 1.0, 0.75, -1.2, 0.1],
    g=categorical(["c", "a", "b", "a", "c", "b"]),
    y=[0.5, -1.0, 0.4, 0.0, 0.6, 1.3], z=[0.1, 0.2, -0.9, 0.2, 1.0, 0.0])

@stestset "a model with a Stan form: same declaration, another program" begin
    rk = RKBRMI(plain_model(plain_data))
    d = brm_descriptor(rk)
    sb = brm_descriptor(SBBRMI(plain_model(plain_data); mod=@__MODULE__))
    @test d.columns == sb.columns
    @test d.formula == sb.formula
    @test d.id != sb.id
    # The graph reads raw columns and preprocesses them itself.
    for name in (:x, :g, :y, :z)
        i = input(d, name)
        @test i.column === name && i.transform === nothing
        @test i.observed == (name in (:y, :z))
    end
    @test input(d, :g).type === Symbol(string(eltype(plain_data.g)))
    # The factor's level table and count are inputs BRM generates.
    generated = [i for i in d.inputs if i.column === nothing]
    @test any(i -> i.transform === :levels, generated)
    @test any(i -> i.transform === :level_count, generated)
    @test all(i -> !i.observed, generated)
    # Data-only definitions RK preparation evaluates (the design matrix, the
    # group indices) are bound as computed values.
    computed = [i for i in d.inputs if i.transform === :computed]
    @test !isempty(computed) && all(i -> i.column === nothing && !i.observed, computed)
    @test classified(d)
    # Other values of the same shape, levels in another order: the same program.
    @test brm_descriptor(RKBRMI(plain_model(plain_other))).id == d.id
    @test brm_descriptor(RKBRMI(plain_fewer(plain_data))).id != d.id
end

@stestset "a held-out response is no input of the program" begin
    d = brm_descriptor(RKBRMI(plain_model(plain_data)))
    h = brm_descriptor(RKBRMI(plain_model(plain_data); held_out=(:z,)))
    @test h.held_out == (:z,)
    @test !(:z in input_names(h))
    @test input(h, :y).observed
    @test :z in h.columns
    @test h.id != d.id
    @test occursin("held out: z", sprint(show, h))
end

mi_model(data) = @brm data begin
    mx ~ Normal(0, 1)
    mi(x) ~ Normal(mx, 1.0)
    sigma ~ Exponential(1)
    mu ~ 1 + x
    y ~ Normal(mu, sigma)
end
mi_data = (; x=Union{Missing,Float64}[0.3, missing, -0.5, 1.2, missing, 0.1],
    y=[0.2, 1.1, -0.4, 0.8, 1.6, 0.3])

@stestset "a non-centered completion is another program" begin
    centred = brm_descriptor(RKBRMI(mi_model(mi_data)))
    noncentred = brm_descriptor(RKBRMI(mi_model(mi_data); completions=:noncentered))
    @test centred.id != noncentred.id
    @test centred.columns == noncentred.columns
end

@stestset "a frozen replay runs the fit's program on the new data" begin
    rk = RKBRMI(plain_model(plain_data))
    d = brm_descriptor(rk)
    replay = reprocess(rk, plain_other)
    r = brm_descriptor(replay)
    @test r.id == d.id
    @test Tuple(i.name for i in r.inputs) == Tuple(keys(rk_bound_data(replay)))
    @test input(r, :x).column === :x
    @test classified(r)
end
