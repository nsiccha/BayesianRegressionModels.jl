# test/rk_cell_latents.jl — latents declared in a `@plate` cell on RKBRMI.
#
# A `~` on a bare cell name that is neither data nor bound by `=` declares a
# latent of the cell (brm-use kernel.md "Latents in a cell"): one draw per cell,
# one per entry where the law's arguments are arrays, whose length may differ
# from cell to cell. RKBRMI declares it at model scope as RKPPL's dotted cell
# latent, `name[i] .~ Family.(args...)` (rkppl-use §13), under its authored
# name. SBBRMI declares one ragged Stan parameter. Both backends give the same
# density and gradient under `brm_coordinate_transport`, `brm_start` takes the
# same address on both, and a frozen replay sizes the latent to the new cells.
#
# Run: julia --startup-file=no --project=test test/rk_cell_latents.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random

const CELLS = (; mu=[[10.0, 20.0, 5.0], [7.0, 3.0]], obs=[[1, 2], [1]],
    y=[[11.0, 19.0], [6.5]], total=[36.0, 10.5], x=[0.3, -0.4])

scalar_latent(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        a ~ Normal(log(total[i]), 1.0)
        total[i] ~ Normal(exp(a), sigma)
    end
end

vector_latent(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        loc = log.(mu[i])
        log_T ~ Normal(loc, 0.5)
        T = exp.(log_T)
        y[i] ~ Normal(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end

bounded_latent(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        T ~ LogNormal(log.(mu[i]), 0.5)
        y[i] ~ Normal(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end

# The law reads sampled scalars: a location shift and a positive scale.
parameter_law_latent(data) = @brm data begin
    sigma ~ Exponential(1.0)
    b ~ Normal(0.0, 1.0)
    s ~ Exponential(2.0)
    @plate for i in eachindex(total)
        log_T ~ Normal(log.(mu[i]) .+ b, s)
        T = exp.(log_T)
        y[i] ~ Normal(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end

# The law reads a per-cell formula predictor as `m[i]`: a model value read at
# the index (rkppl-use §13; ReactiveKernels snag `rkppl-cell-laten-b18e668a`).
predictor_law_latent(data) = @brm data begin
    sigma ~ Exponential(1.0)
    m ~ 1 + x
    @plate for i in eachindex(total)
        log_T ~ Normal(log.(mu[i]) .+ m[i], 0.5)
        T = exp.(log_T)
        y[i] ~ Normal(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end

# The law reads a top-level vector parameter at the cell, `b[i]`. RKPPL admits
# it; BRM's RK preparation still refuses it as a per-subject linear predictor
# (todo 2026-10-26T08-15-30-153-1n962h6).
vector_parameter_law_latent(data) = @brm data begin
    sigma ~ Exponential(1.0)
    b[1:2] .~ Normal(0.0, 1.0)
    @plate for i in eachindex(total)
        log_T ~ Normal(log.(mu[i]) .+ b[i], 0.5)
        T = exp.(log_T)
        y[i] ~ Normal(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end

# A latent nothing observes is a generated draw: neither backend samples it.
unread_latent(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        extra ~ Normal(0.0, 2.0)
        total[i] ~ Normal(40.0, sigma)
    end
end

# The vector model's log density at RK's packed point `u` (sigma, then the
# cells' log_T entries cell after cell), written independently.
function vector_oracle(u, data=CELLS)
    σ = exp(u[1])
    lp = logpdf(Exponential(1.0), σ) + u[1]
    at = 1
    for i in eachindex(data.total)
        n = length(data.mu[i])
        log_T = u[(1 + at):(at + n)]
        at += n
        lp += sum(logpdf.(Normal.(log.(data.mu[i]), 0.5), log_T))
        T = exp.(log_T)
        lp += sum(logpdf.(Normal.(T[data.obs[i]], σ), data.y[i]))
        lp += logpdf(Normal(sum(T), 0.01 * data.total[i]), data.total[i])
    end
    lp
end

# RK against the compiled Stan model under the semantic transport, at a few
# random points; returns the transport.
function check_against_stan(brmi, name; points=3, scale=0.3)
    rk = RKBRMI(brmi)
    problem = rk_logdensity_problem(check_rk_source_roundtrip(rk);
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(), "brm-rk-cell-latents", name * ".stan"))
    names = String.(BridgeStan.param_unc_names(stan.model))
    t = brm_coordinate_transport(rk, sb, names)
    rng = Xoshiro(17)
    n = LogDensityProblems.dimension(problem)
    for _ in 1:points
        u = scale .* randn(rng, n)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model,
            brm_rk_to_stan(t, u); propto=false, jacobian=true)
        @test value ≈ stan_value atol=1e-10 rtol=1e-11
        @test gradient ≈ brm_stan_to_rk(t, stan_gradient) atol=1e-9 rtol=1e-10
        @test brm_check_coordinate_transport(t, rk, stan.model, u) isa NamedTuple
    end
    (; rk, sb, stan, t, problem)
end

@stestset "a scalar cell latent: one draw per cell" begin
    c = check_against_stan(scalar_latent(CELLS), "scalar_latent")
    @test coordinate_names(c.rk.model.layout) == [:sigma, Symbol("a.1"), Symbol("a.2")]
    # The emitted cell declares the latent and reads its own entry back.
    source = string(BRM._rk_retained_program(c.rk))
    @test occursin("a[i] .~ Normal.(", source)
    restored = rk_restore_draws(c.rk, reshape(collect(0.1:0.1:0.3), :, 1))
    @test restored.a ≈ reshape([0.2, 0.3], 2, 1)
end

@stestset "a vector cell latent whose length varies by cell" begin
    c = check_against_stan(vector_latent(CELLS), "vector_latent")
    @test coordinate_names(c.rk.model.layout) == [:sigma,
        Symbol("log_T.1.1"), Symbol("log_T.1.2"), Symbol("log_T.1.3"),
        Symbol("log_T.2.1"), Symbol("log_T.2.2")]
    for u in ([0.0, 2.3, 3.0, 1.6, 1.9, 1.1], 0.2 .* randn(Xoshiro(3), 6))
        check_consumer_point(c.problem, u, vector_oracle)
    end
    # Draws restore one entries × draws matrix per cell.
    U = randn(Xoshiro(4), 6, 4)
    restored = rk_restore_draws(c.rk, U)
    @test length(restored.log_T) == 2
    @test restored.log_T[1] ≈ U[2:4, :]
    @test restored.log_T[2] ≈ U[5:6, :]
end

@stestset "a bounded cell latent keeps its family's transform" begin
    c = check_against_stan(bounded_latent(CELLS), "bounded_latent")
    values = rk_restore_draws(c.rk, reshape(fill(-0.2, 6), :, 1))
    @test values.T[1] ≈ fill(exp(-0.2), 3, 1)
end

@stestset "the law may read sampled scalars" begin
    c = check_against_stan(parameter_law_latent(CELLS), "parameter_law_latent")
    @test count(n -> startswith(String(n), "log_T."), coordinate_names(c.rk.model.layout)) == 5
end

@stestset "the law may read a per-cell predictor" begin
    c = check_against_stan(predictor_law_latent(CELLS), "predictor_law_latent")
    @test count(n -> startswith(String(n), "log_T."), coordinate_names(c.rk.model.layout)) == 5
end

@stestset "known gap: the law reads a vector parameter at the cell" begin
    @test_broken (RKBRMI(vector_parameter_law_latent(CELLS)).model; true)
end

@stestset "a latent nothing observes is not sampled" begin
    c = check_against_stan(unread_latent(CELLS), "unread_latent")
    @test coordinate_names(c.rk.model.layout) == [:sigma]
end

@stestset "brm_start sets a cell latent by its name on both backends" begin
    for (label, model, latent, values) in (
            (:normal, vector_latent, :log_T, [log.([9.0, 21.0, 6.0]), log.([7.5, 3.0])]),
            (:bounded, bounded_latent, :T, [[9.0, 21.0, 6.0], [7.5, 3.0]]),
            (:scalar, scalar_latent, :a, [3.6, 2.4]))
        brmi = model(CELLS)
        rk = RKBRMI(brmi)
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        stan = BRM.stan_instantiate(sb;
            path=joinpath(tempdir(), "brm-rk-cell-latents", "start_$label.stan"))
        t = brm_coordinate_transport(rk, sb, String.(BridgeStan.param_unc_names(stan.model)))
        inits = NamedTuple{(latent, :sigma)}((values, 0.5))
        u = brm_start(rk, inits)
        @test brm_rk_to_stan(t, u) ≈ brm_start(sb, stan.model, inits) rtol=1e-12 atol=1e-14
        @test getproperty(BRM._rk_constrained_values(rk, u), latent) ≈ values rtol=1e-12
        # One flat vector, cell after cell, is the same point.
        label === :scalar || @test brm_start(rk,
            NamedTuple{(latent, :sigma)}((reduce(vcat, values), 0.5))) ≈ u rtol=1e-14
        # Unaddressed coordinates keep a random base (through RK's constrain
        # and unconstrain round trip).
        base = 0.3 .* randn(Xoshiro(5), length(u))
        v = brm_start(rk, NamedTuple{(latent,)}((values,)); base)
        @test v[1] ≈ base[1] rtol=1e-12
    end
    rk = RKBRMI(vector_latent(CELLS))
    per_cell = [log.([9.0, 21.0, 6.0]), log.([7.5, 3.0])]
    # refused: the cells' counts and lengths are the model's.
    @test_throws DimensionMismatch brm_start(rk, (; log_T=per_cell[1:1]))
    @test_throws DimensionMismatch brm_start(rk, (; log_T=[per_cell[1], [per_cell[2]; 0.0]]))
    @test_throws DimensionMismatch brm_start(rk, (; log_T=zeros(4)))
    # refused: per-cell vectors address a cell latent; `sigma` is one value.
    @test_throws ErrorException brm_start(rk, (; sigma=[[0.5]]))
end

@stestset "a frozen replay keeps the fit's cells and their latents" begin
    rk = RKBRMI(vector_latent(CELLS))
    # New values on the fit's cells: the same coordinates, the new data's density.
    new = merge(CELLS, (; y=[[12.0, 18.0], [6.0]], total=[37.0, 10.0]))
    replay = reprocess(rk, new)
    @test coordinate_names(replay.model.layout) == coordinate_names(rk.model.layout)
    problem = rk_logdensity_problem(replay; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    u = 0.2 .* randn(Xoshiro(9), 6)
    @test LogDensityProblems.logdensity(problem, u) ≈ vector_oracle(u, new) atol=1e-10 rtol=1e-11
    # refused: new cells give the latent other draws than the fit's.
    more = (; mu=[[10.0, 20.0], [7.0, 3.0, 4.0], [2.0]], obs=[[1], [1, 3], [1]],
        y=[[11.0], [6.5, 4.2], [2.1]], total=[30.0, 14.0, 2.0], x=[0.1, 0.2, 0.3])
    @test_throws "cell latent(s) `log_T`" reprocess(rk, more)
    # A rebuild on new cells sizes the latent to them.
    rebuilt = reprocess(rk, more; freeze_constants=false)
    @test count(n -> startswith(String(n), "log_T."),
        coordinate_names(rebuilt.model.layout)) == 6
end

@stestset "refused cell latents" begin
    # A cell latent needs a univariate family.
    @test_throws "univariate" RKBRMI(@brm CELLS begin
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(total)
            v ~ MvNormal(log.(mu[i]), 1.0)
            total[i] ~ Normal(sum(exp.(v)), sigma)
        end
    end)
end
