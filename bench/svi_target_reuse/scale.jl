# bench/svi_target_reuse/scale.jl — how the repeated same-program target build
# grows with the model, on the public connected toy of test/nlme_svi_owned_fixture.jl.
#
#     julia --startup-file=no --project=test bench/svi_target_reuse/scale.jl [out.tsv]
#
# Each size is a new model. Its first target compiles what the size needs, so the
# rows that answer the question are the `repeat` medians, taken after it, piece by
# piece. Times below a second vary with host load; read the growth, not the digits.
using BayesianRegressionModels
import ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels
const AD = AutoEnzyme(; mode=Enzyme.Reverse)
const OUT = get(ARGS, 1, "")
include(joinpath(@__DIR__, "..", "..", "test", "nlme_svi_owned_fixture.jl"))

median_of(f, n=3) = (x = [(@elapsed f()) for _ in 1:n]; sort!(x); x[(n + 1) ÷ 2])
linked(n, chain) = merge(svi_owned_data(collect(1:n)), (; left=collect(1:n),
    right=[i % chain == 0 ? i : i + 1 for i in 1:n], weight=fill(0.3, n)))

println(join(("subjects", "units", "packed", "bytes_retained", "model", "first_target", "ownership",
    "plan", "prior", "whole", "worker", "identity", "reused_phase", "rebuilt_phase", "iteration"), '\t'))
for (n, chain) in ((120, 12), (480, 48), (1920, 192), (2480, 496))
    rk = RKBRMI(svi_owned_connected_model(linked(n, chain)))
    model_s = @elapsed m = brm_nlme_model(rk; ad_backend=AD, unit_partition=:connected)
    first_s = @elapsed t = BRM._brm_svi_target(m)
    o = BRM._brm_svi_ownership(m)
    ownership = median_of(() -> BRM._brm_svi_ownership(m))
    plan = median_of(() -> BRM._brm_nlme_partition_plan(m.cell_gradient, m.unit, o.globals, m.view; units=o.units))
    prior = median_of(() -> BRM._brm_svi_authored_prior(m, o.globals, o.units; factors=o.factors))
    whole = median_of(() -> BRM._brm_svi_target(m))
    worker = median_of(() -> BRM._brm_nlme_worker_plan(t.plan, m.cell_gradient, m.unit))
    identity_s = median_of(() -> BRM._brm_svi_identity(t))
    before = Base.summarysize(m)
    fit = brm_svi_fit(m; batch_size=2, threads=1, stop_after=1, ndraws=0)
    retained = Base.summarysize(m) - before
    warm = brm_svi_resume(m, fit; stop_after=20, ndraws=0)
    state = brm_svi_optimizer_checkpoint(warm)
    phase() = brm_svi_resume(m, state; stop_after=0, ndraws=0).diagnostics.target.seconds
    reused = sort!([phase() for _ in 1:3])[2]
    rebuilt = sort!([(m.svi_target.prepared = nothing; phase()) for _ in 1:3])[2]
    row = (n, length(m.view.levels), length(m.unit), retained, model_s, first_s, ownership, plan, prior,
        whole, worker, identity_s, reused, rebuilt, warm.diagnostics.seconds / 20)
    line = join((x isa AbstractFloat ? round(x; sigdigits=4) : x for x in row), '\t')
    println(line)
    flush(stdout)
    isempty(OUT) || open(io -> println(io, line), OUT, "a")
end
