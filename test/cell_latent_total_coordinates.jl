# test/cell_latent_total_coordinates.jl — `@plate` cell latents in total
# coordinates: `x ~ Law(…; coordinates = :logsumexp | :sum)`.
#
# The same latent as without the keyword, sampled per cell as its total
# (`log(sum(exp.(x)))`, `log(sum(T))` for a positive latent, or `sum(x)`) plus
# K - 1 isometric sum-to-zero coordinates of the centred values. The map has a
# constant Jacobian, so the emitted density at matched points equals the
# density of the latent's own coordinates, Jacobians included (snag
# exact-sum-constr-7ba55c76, decision 18wcqyi).
#
# Run: julia --startup-file=no --project=test test/cell_latent_total_coordinates.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, LinearAlgebra

# Cells of 3, 2 and 1 entries (one entry: the total is the latent itself).
const CELLS = (; mu=[[10.0, 20.0, 5.0], [7.0, 3.0], [4.0]], obs=[[1, 2], [1], [1]],
    y=[[11.0, 19.0], [6.5], [4.2]], total=[36.0, 10.5, 4.1])

eta_latent(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        loc = log.(mu[i])
        log_T ~ Normal(loc, 0.5)
        T = exp.(log_T)
        y[i] ~ Normal(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end
logsumexp_latent(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        loc = log.(mu[i])
        log_T ~ Normal(loc, 0.5; coordinates = :logsumexp)
        T = exp.(log_T)
        y[i] ~ Normal(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end
# A scale per entry that reads a sampled scalar.
vector_scale_own(data) = @brm data begin
    sigma ~ Exponential(1.0)
    s ~ Exponential(2.0)
    @plate for i in eachindex(total)
        loc = log.(mu[i])
        scl = s .+ 0.1 .* loc
        log_T ~ Normal(loc, scl)
        T = exp.(log_T)
        y[i] ~ Normal(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end
vector_scale_total(data) = @brm data begin
    sigma ~ Exponential(1.0)
    s ~ Exponential(2.0)
    @plate for i in eachindex(total)
        loc = log.(mu[i])
        scl = s .+ 0.1 .* loc
        log_T ~ Normal(loc, scl; coordinates = :logsumexp)
        T = exp.(log_T)
        y[i] ~ Normal(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end
positive_own(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        T ~ LogNormal(log.(mu[i]), 0.5)
        y[i] ~ Normal(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end
positive_total(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        T ~ LogNormal(log.(mu[i]), 0.5; coordinates = :sum)
        y[i] ~ Normal(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end
real_sum_own(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        x ~ Normal(mu[i], 2.0)
        y[i] ~ Normal(x[obs[i]], sigma)
        total[i] ~ Normal(sum(x), 0.01 * total[i])
    end
end
real_sum_total(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        x ~ Normal(mu[i], 2.0; coordinates = :sum)
        y[i] ~ Normal(x[obs[i]], sigma)
        total[i] ~ Normal(sum(x), 0.01 * total[i])
    end
end

function compiled(brmi, label)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    folder = joinpath(tempdir(), "brm-total-coordinates")
    mkpath(folder)
    stan = BRM.stan_instantiate(sb; path=joinpath(folder, label * ".stan"))
    (; sb, stan, names=BridgeStan.param_unc_names(stan.model))
end

block(names, latent) = findall(n -> occursin(Regex("_$(latent)\\.\\d+\$"), n), names)

# The total-coordinate model's unconstrained point matching a point of the
# latent's own coordinates (`own` holds the latent's unconstrained values).
function matched(own_names, total_names, u, latent, kind, data; value=identity)
    io, it = block(own_names, latent), block(total_names, "$(latent)_coordinates")
    @test length(io) == length(it) == sum(length, data.mu)
    v = copy(u)
    at = 0
    coords = Float64[]
    for m in data.mu
        append!(coords, BRM._brm_total_coordinates(kind, value(u[io][at+1:at+length(m)])))
        at += length(m)
    end
    v[it] = coords
    v
end

@stestset "Julia maps: inverse pairs and a constant Jacobian" begin
    rng = Xoshiro(1)
    for kind in (:logsumexp, :sum), K in (1, 2, 5, 12)
        x = 2 .* randn(rng, K)
        c = BRM._brm_total_coordinates(kind, x)
        @test BRM._brm_total_values(kind, c) ≈ x rtol=1e-12 atol=1e-12
        kind === :logsumexp && @test c[1] ≈ log(sum(exp.(x)))
        kind === :sum && @test c[1] ≈ sum(x)
    end
    T = exp.(randn(rng, 6))
    @test BRM._brm_total_values(:logsum, BRM._brm_total_coordinates(:logsum, T)) ≈ T
    @test BRM._brm_total_coordinates(:logsum, T)[1] ≈ log(sum(T))
    for kind in (:logsumexp, :sum), K in (2, 7)
        f = c -> BRM._brm_total_values(kind, c)
        dets = map(1:5) do _
            c = randn(rng, K)
            J = reduce(hcat, [(f(c .+ 1e-6 .* (1:K .== j)) .- f(c .- 1e-6 .* (1:K .== j))) ./ 2e-6
                for j in 1:K])
            logabsdet(J)[1]
        end
        @test maximum(dets) - minimum(dets) < 1e-6
    end
end

@stestset "logsumexp: the latent's density at matched points" begin
    own = compiled(eta_latent(CELLS), "own_eta")
    total = compiled(logsumexp_latent(CELLS), "total_logsumexp")
    @test isempty(block(total.names, "log_T"))
    rng = Xoshiro(3)
    for _ in 1:10
        u = 0.3 .* randn(rng, length(own.names))
        io = block(own.names, "log_T")
        u[io] .+= log.(reduce(vcat, CELLS.mu))
        v = matched(own.names, total.names, u, "log_T", :logsumexp, CELLS)
        @test BridgeStan.log_density(total.stan.model, v; propto=false) ≈
            BridgeStan.log_density(own.stan.model, u; propto=false) atol=1e-8 rtol=1e-12
        lp, g = BridgeStan.log_density_gradient(total.stan.model, v; propto=false)
        @test all(isfinite, g)
        # The emitted map is the Julia map: the constrained `log_T` read back.
        θ = BridgeStan.param_constrain(total.stan.model, v; include_tp=true)
        names = BridgeStan.param_names(total.stan.model; include_tp=true)
        values = θ[findall(n -> occursin(r"_log_T\.\d+$", n), names)]
        @test values ≈ u[io] rtol=1e-10
    end
    # The total sits on one coordinate per cell: exp(c[1]) == sum(T).
    d = brm_descriptor(total.sb)
    @test brm_output(d, :log_T_coordinates).segments == [3, 5, 6]
    @test !isempty(brm_outputs(d; logical=:log_T))
end

@stestset "logsumexp with a scale per entry that reads a parameter" begin
    own = compiled(vector_scale_own(CELLS), "own_vscale")
    total = compiled(vector_scale_total(CELLS), "total_vscale")
    rng = Xoshiro(4)
    for _ in 1:5
        u = 0.3 .* randn(rng, length(own.names))
        u[block(own.names, "log_T")] .+= log.(reduce(vcat, CELLS.mu))
        v = matched(own.names, total.names, u, "log_T", :logsumexp, CELLS)
        @test BridgeStan.log_density(total.stan.model, v; propto=false) ≈
            BridgeStan.log_density(own.stan.model, u; propto=false) atol=1e-8 rtol=1e-12
    end
end

@stestset "sum of a positive latent: log(sum(T)) and the positivity Jacobian" begin
    own = compiled(positive_own(CELLS), "own_positive")
    total = compiled(positive_total(CELLS), "total_positive")
    rng = Xoshiro(5)
    for _ in 1:10
        u = 0.3 .* randn(rng, length(own.names))
        u[block(own.names, "T")] .+= log.(reduce(vcat, CELLS.mu))
        # The own model's unconstrained `T` is log(T); its density holds the
        # log-transform's Jacobian, which the total model's density carries too.
        v = matched(own.names, total.names, u, "T", :logsum, CELLS; value=x -> exp.(x))
        @test BridgeStan.log_density(total.stan.model, v; propto=false) ≈
            BridgeStan.log_density(own.stan.model, u; propto=false) atol=1e-8 rtol=1e-12
    end
end

@stestset "sum of a real latent" begin
    own = compiled(real_sum_own(CELLS), "own_real_sum")
    total = compiled(real_sum_total(CELLS), "total_real_sum")
    rng = Xoshiro(6)
    for _ in 1:10
        u = 0.5 .* randn(rng, length(own.names))
        u[block(own.names, "x")] .+= reduce(vcat, CELLS.mu)
        v = matched(own.names, total.names, u, "x", :sum, CELLS)
        @test BridgeStan.log_density(total.stan.model, v; propto=false) ≈
            BridgeStan.log_density(own.stan.model, u; propto=false) atol=1e-8 rtol=1e-12
    end
end

@stestset "brm_start takes the latent's values" begin
    total = compiled(logsumexp_latent(CELLS), "start_logsumexp")
    values = [log.([9.0, 21.0, 6.0]), log.([7.5, 3.0]), [log(4.0)]]
    for given in (values, reduce(vcat, values))
        u = brm_start(total.sb, total.stan.model, (; log_T=given, sigma=0.5))
        θ = BridgeStan.param_constrain(total.stan.model, u; include_tp=true)
        names = BridgeStan.param_names(total.stan.model; include_tp=true)
        @test θ[findall(n -> occursin(r"_log_T\.\d+$", n), names)] ≈ reduce(vcat, values) rtol=1e-10
    end
    @test_throws DimensionMismatch brm_start(total.sb, total.stan.model,
        (; log_T=values[1:2]))
    positive = compiled(positive_total(CELLS), "start_positive")
    Tvals = [[9.0, 21.0, 6.0], [7.5, 3.0], [4.0]]
    u = brm_start(positive.sb, positive.stan.model, (; T=Tvals))
    θ = BridgeStan.param_constrain(positive.stan.model, u; include_tp=true)
    names = BridgeStan.param_names(positive.stan.model; include_tp=true)
    @test θ[findall(n -> occursin(r"_T\.\d+$", n), names)] ≈ reduce(vcat, Tvals) rtol=1e-10
    # refused: a positive latent's values must be positive.
    @test_throws "positive values" brm_start(positive.sb, positive.stan.model,
        (; T=[[9.0, -1.0, 6.0], [7.5, 3.0], [4.0]]))
end

@stestset "refused spellings" begin
    sb(brmi) = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    # refused: an unknown coordinate system.
    @test_throws "not a coordinate system" sb(@brm CELLS begin
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(total)
            x ~ Normal(mu[i], 2.0; coordinates = :mean)
            total[i] ~ Normal(sum(x), sigma)
        end
    end)
    # refused: a positive latent's total is its own sum.
    @test_throws "coordinates = :sum" sb(@brm CELLS begin
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(total)
            T ~ LogNormal(log.(mu[i]), 0.5; coordinates = :logsumexp)
            total[i] ~ Normal(sum(T), sigma)
        end
    end)
    # refused: a latent on [0, 1] has no total coordinate.
    @test_throws "total coordinates need a real or a positive latent" sb(@brm CELLS begin
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(total)
            p ~ Beta(fill(2.0, length(mu[i])), 2.0; coordinates = :sum)
            total[i] ~ Normal(sum(p), sigma)
        end
    end)
end

@stestset "descriptor, plan and description read the model" begin
    sb = SBBRMI(logsumexp_latent(CELLS); mod=@__MODULE__, total_groups=())
    d = brm_descriptor(sb)
    plan = generative_plan(sb)
    @test any(decl -> decl.target === :log_T_coordinates || occursin("log_T_coordinates", string(decl.target)),
        plan.declarations)
    text = string(brm_description(d))
    @test occursin("log_T", text)
end

@stestset "known gap: total coordinates on RKBRMI" begin
    # RKPPL has no per-cell vector latent with a joint density yet (ReactiveKernels
    # snag rkppl-per-index-5e9df9fe); RKBRMI refuses the keyword naming it.
    @test_broken (RKBRMI(logsumexp_latent(CELLS)).model; true)
end
