# test/rk_plate_invariant_output.jl — a @plate output whose cell reads no per-index value.
#
# RUN: julia --project=test test/rk_plate_invariant_output.jl [testset-filter...]
#
# In `@plate for g in eachindex(path); decay[g] = exp(log_a); end` the cell reads a
# population scalar and nothing per index. As in the Julia loop, `decay` holds one
# value per index, so a later plate's `decay[g]`, `sum(decay)` and `length(decay)`
# see every entry. ReactiveKernelsPPL before 0.1.129 collected such an output as one
# number (ReactiveKernels snag `plate-output-of-c4a0ca87`, fixed in RK `b6196f114`):
# a later plate's `decay[2]` raised `BoundsError: attempt to access Float64 at index
# [2]`, and `sum`/`length` returned one entry's value with no error. BRM emits the
# plate as written (BRM snag `retained-pointwi-fa27bc37`, ODLPlaus v10's
# `washout_decay[g] = exp(washout_log_decay)`).
#
# Every case but the one-cell control has three cells: one cell cannot tell a
# number from a one-entry array. These are toy data; the likelihoods are checked
# against `Distributions.logpdf`.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems

# Whether the RKPPL under test keeps the plate's axis, so a move of the test pin
# turns the `broken` lines into passing ones without an edit. The landed number is
# read from ReactiveKernelsPPL's `Project.toml` at RK `b6196f114`.
const RK_KEEPS_INVARIANT_PLATE_AXIS = pkgversion(ReactiveKernelsPPL) >= v"0.1.129"

const INVARIANT_SUBJECTS = (; subject=["a", "b", "c"], path=[1, 2, 3],
    t=[[0.5, 1.0, 2.0], [0.5, 3.0], [1.0, 2.0, 4.0, 8.0]],
    y=[[18.0, 15.0, 11.0], [20.0, 8.0], [16.0, 12.0, 7.0, 3.0]])

# The reporter's shape: one plate computes `decay` from the population scalar
# `log_a`, a second plate reads it per subject.
invariant_decay_model(data) = @brm data begin
    sigma ~ Exponential(1)
    log_a ~ Normal(0, 1)
    log_rate ~ 1 + (1 | blk | subject)
    @plate for g in eachindex(path)
        decay[g] = exp(log_a)
    end
    @plate for g in eachindex(path)
        m[g] = exp(log_rate[g]) .* exp.(-decay[g] .* t[g])
        y[g] ~ Normal(m[g], sigma)
    end
end

# The model-scope spelling of the same values, correct on every RKPPL: the
# reference for the whole-model density and gradient.
fill_decay_model(data) = @brm data begin
    sigma ~ Exponential(1)
    log_a ~ Normal(0, 1)
    log_rate ~ 1 + (1 | blk | subject)
    decay = fill(exp(log_a), length(path))
    @plate for g in eachindex(path)
        m[g] = exp(log_rate[g]) .* exp.(-decay[g] .* t[g])
        y[g] ~ Normal(m[g], sigma)
    end
end

# Whole reads of the plate output at model scope.
invariant_total_model(data) = @brm data begin
    sigma ~ Exponential(1)
    log_a ~ Normal(0, 1)
    @plate for g in eachindex(path)
        decay[g] = exp(log_a)
    end
    total = sum(decay)
    n_decay = length(decay)
    w ~ Normal(total, sigma)
end

invariant_draw(rk, u, names) = map(only, rk_draw_values(rk, reshape(u, :, 1), names))

# Subject `s`'s log-likelihood at `u`, from the model's own `log_rate` and the
# sampled scalars, with `decay` computed here.
function invariant_subject_oracle(rk, u, s; data=INVARIANT_SUBJECTS)
    v = invariant_draw(rk, u, (:sigma, :log_a, :log_rate))
    m = exp(v.log_rate[s]) .* exp.(-exp(v.log_a) .* data.t[s])
    sum(logpdf.(Normal.(m, v.sigma), data.y[s]))
end

function invariant_fd(f, u, coordinates; h=1e-5)
    map(coordinates) do j
        plus = copy(u); minus = copy(u)
        plus[j] += h; minus[j] -= h
        (f(plus) - f(minus)) / (2h)
    end
end

invariant_point(rk) = 0.1 .* sin.(1:rk.model.layout.total)

@stestset "a whole read of an invariant plate output sees every cell" begin
    for (cells, w) in ((3, [3.0, 2.0]), (1, [1.2, 0.9]))
        @testset "$cells cells" begin
            rk = RKBRMI(invariant_total_model((; path=collect(1:cells), w)))
            u = invariant_point(rk)
            v = invariant_draw(rk, u, (:sigma, :log_a, :decay, :total, :n_decay))
            oracle = sum(logpdf.(Normal(cells * exp(v.log_a), v.sigma), w))
            likelihood = Base.invokelatest(likelihood_query(rk), u)
            # One cell is the positive control: index 1 of a number is valid and one
            # value is the right count, so it passed before RKPPL 0.1.129 too.
            broken = cells > 1 && !RK_KEEPS_INVARIANT_PLATE_AXIS
            @test v.n_decay == cells broken=broken
            @test v.total ≈ cells * exp(v.log_a) rtol=1e-14 broken=broken
            @test likelihood ≈ oracle rtol=1e-12 broken=broken
            @test (v.decay == fill(exp(v.log_a), cells)) broken=!RK_KEEPS_INVARIANT_PLATE_AXIS
        end
    end
end

@stestset "a later plate reads each cell of an invariant plate output" begin
    rk = RKBRMI(invariant_decay_model(INVARIANT_SUBJECTS))
    control = RKBRMI(fill_decay_model(INVARIANT_SUBJECTS))
    @test coordinate_names(rk.model.layout) == coordinate_names(control.model.layout)
    u = invariant_point(rk)
    if RK_KEEPS_INVARIANT_PLATE_AXIS
        @test invariant_draw(rk, u, (:decay,)).decay ==
            fill(exp(invariant_draw(rk, u, (:log_a,)).log_a), 3)
        pointwise = Base.invokelatest(pointwise_query(rk), u).y
        @test length(pointwise) == 3
        for s in 1:3
            @test sum(pointwise[s]) ≈ invariant_subject_oracle(rk, u, s) rtol=1e-12
        end
        # The whole model's density and native Reverse gradient, which reach
        # `log_a` only through `decay`.
        full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
        reference = rk_logdensity_problem(control; ad_backend=NLME_AD)
        value, gradient = LogDensityProblems.logdensity_and_gradient(full, u)
        reference_value, reference_gradient =
            LogDensityProblems.logdensity_and_gradient(reference, u)
        @test value ≈ reference_value rtol=1e-12
        @test gradient ≈ reference_gradient rtol=1e-10
        @test gradient ≈ invariant_fd(x -> LogDensityProblems.logdensity(full, x),
            u, eachindex(u)) rtol=1e-6 atol=1e-7
    else
        # Before RKPPL 0.1.129 the second plate reads index 2 of a number.
        @test_broken (Base.invokelatest(pointwise_query(rk), u); true)
    end
end

@stestset "brm_nlme_model evaluates each subject past an invariant plate output" begin
    rk = RKBRMI(invariant_decay_model(INVARIANT_SUBJECTS))
    if RK_KEEPS_INVARIANT_PLATE_AXIS
        m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:subject)
        # `log_a` is one of the view's population scalars, which every subject reads
        # through `decay`.
        @test :log_a in m.view.coordinates[m.sigma]
        θ = 0.1 .* sin.(eachindex(m.theta))
        σ = fill(log(0.8), length(m.sigma))
        @test length(m.view.levels) == 3
        H = reshape(0.1 .* cos.(1:3length(m.view.block_margins)),
            length(m.view.block_margins), 3)
        values, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
        @test values == brm_nlme_loglikelihoods(m, θ, σ, H)
        u = BRM._brm_nlme_point(m, θ, σ, H)
        for s in 1:3
            @test values[s] ≈ invariant_subject_oracle(rk, u, s) rtol=1e-12
            @test brm_nlme_loglikelihood(m, s, θ, σ, H[:, s]) ≈ values[s] rtol=1e-12
            value, gradient = brm_nlme_loglikelihood_and_gradient(m, s, θ, σ, H[:, s])
            @test value ≈ values[s] rtol=1e-12
            @test gradient ≈ G[:, s] rtol=1e-10
            @test gradient ≈ invariant_fd(η -> brm_nlme_loglikelihood(m, s, θ, σ, η),
                H[:, s], eachindex(H[:, s])) rtol=1e-6 atol=1e-7
        end
    else
        # Before RKPPL 0.1.129 the model's pointwise attribution raises the
        # `BoundsError` the reporter saw.
        @test_broken (brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:subject); true)
    end
end
