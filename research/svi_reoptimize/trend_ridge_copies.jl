# What the re-optimized copies did on test/nlme_svi_trend.jl's ridge (8 units of 100
# rows, data seed 101; ELBO-only stop, centering and coupling off) at
# unit_reoptimize = 1, fit seed 1, before a copy's step had to raise its unit's
# objective (todo 12maq21). `_brm_svi_reoptimized_terms` is replaced by that earlier
# form, every step standing, and records per copy: the unit's objective at the
# iteration's draw of the globals (`value` + the factor's entropy) for its own factor
# and for the copy, the copy's mean step in the old factor's sds and whether the
# four-sd trust bound held it, and how many precision directions sat at the ×/÷100
# bound. Printed per 100 iterations, with the globals' means. Measured: half the
# copies in the first 900 iterations were worse than the unit's own factor, and the
# globals walked to a noise sd of 2.3 and exp(m) 0.29.
#
#     [ITERATIONS=3000] julia --project=test research/svi_reoptimize/trend_ridge_copies.jl
const ROOT = get(ENV, "BRM_ROOT", pwd())
include(joinpath(@__DIR__, "..", "..", "test", "fitting_packages.jl"))
include(joinpath(ROOT, "test", "nlme_fixtures.jl"))
using LinearAlgebra, Random, Statistics, Printf
const BRM = BayesianRegressionModels

include(joinpath(ROOT, "test", "nlme_svi_ridge_fixture.jl"))
const ITERATIONS = parse(Int, get(ENV, "ITERATIONS", "3000"))
const LOG = NamedTuple[]

BRM.eval(quote
    function _brm_svi_reoptimized_terms(t, q, i, g, r, designs, plan, work, touched, D;
            centering, frame, zmean, positive=true)
        U, m = q.U[i], copy(zmean)
        own = r.value - sum(log, diag(LowerTriangular(U)))
        c = r
        for E in designs
            L, Δ = _brm_svi_newton_factor(U, c.gz, c.S; positive)
            Ŝ = Symmetric((c.S .+ c.S') ./ 2)
            F = eigen(positive ? _brm_svi_positive_part(Ŝ) : Ŝ)
            b = _BRM_SVI_REOPT_PRECISION_BOUND
            clipped = count(v -> v <= 1 / b || v >= b, F.values)
            stepsd = norm(LowerTriangular(U)' * Δ)
            U, m = L, m .+ Δ
            c = _brm_svi_unit_design_terms(t, q, i, g, E, plan, work, touched, D;
                centering, frame, zmean=m, U)
            copy_obj = c.value - sum(log, diag(LowerTriangular(U)))
            push!(Main.LOG, (; own, copy=copy_obj, stepsd, clipped,
                finite=c.problem === nothing && all(isfinite, c.S) && all(isfinite, c.gz)))
            c.problem === nothing && all(isfinite, c.S) && all(isfinite, c.gz) || return c
        end
        c
    end
end)

rk = RKBRMI(svi_ridge_model(svi_ridge_data(Xoshiro(101), 8)))
marks = Int[]
μs = Vector{Float64}[]
save = state -> (push!(marks, length(LOG)); push!(μs, copy(state.run.q.μ)))
fit = Base.invokelatest(brm_svi_fit, save, rk; rng=Xoshiro(1), subject_group=:unit,
    batch_size=4, ndraws=20, stationarity=false, centering=nothing, coupling=nothing,
    unit_reoptimize=1, max_iterations=ITERATIONS)
println("globals: ", fit.coordinates[fit.diagnostics.globals])
println("window | globals' means | copies: worse than own (share), median own→copy objective change, median step (old sds), at trust (share), clipped directions (mean)")
lo = 0
for (w, hi) in enumerate(marks)
    rows = LOG[lo+1:hi]
    global lo = hi
    isempty(rows) && continue
    Δobj = [r.copy - r.own for r in rows]
    @printf("%5d | %s | %.2f %9.2f %6.2f %.2f %.2f\n", 100w,
        join((@sprintf("%6.2f", v) for v in μs[w]), " "),
        mean(Δobj .< 0), median(Δobj), median(r.stepsd for r in rows),
        mean(r.stepsd >= 3.999 for r in rows), mean(r.clipped for r in rows))
end
