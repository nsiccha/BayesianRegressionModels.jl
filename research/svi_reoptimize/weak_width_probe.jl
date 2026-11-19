# The weak-data width regression of `unit_reoptimize = 1` (todo 12maq21): the
# random-intercept fixture of test/nlme_svi_default_centering.jl (32 groups x one
# row in the weak case) fitted at k = 0 and k = 1 under centering and coupling
# variants, against the same four-chain NUTS reference. Prints every global's
# mean and sd against NUTS, the learned centering controls, unit widths, and a
# per-window trajectory of the global factor, controls and coupling norm.
#
#     julia --project=test research/svi_reoptimize/weak_width_probe.jl OUTDIR case[,case...]
#
# case = label:k:centering:coupling[:counts], centering in default|none|nodelay,
# coupling in default|none, counts in weak|strong|unbalanced. Writes
# OUTDIR/nuts_<counts>.jls once and OUTDIR/<label>.jls per case. As the test does,
# the NUTS reference runs after the first SVI fit of its data, on the same backend;
# a reference whose completion quorum fails is reported and left NaN, and
# PROBE_NUTS=0 skips it.
const OUT = ARGS[1]
mkpath(OUT)
const CASES = split(ARGS[2], ",")
# test/testset_filter.jl reads ARGS as testset filters.
empty!(ARGS)
const ROOT = get(ENV, "BRM_ROOT", pwd())
include(joinpath(ROOT, "test", "nlme_fixtures.jl"))
using Random, Statistics, Serialization, WarmupHMC, MCMCDiagnosticTools, Printf

function default_centering_data(counts; seed=1)
    rng = Xoshiro(seed)
    u = randn(rng, length(counts))
    g = reduce(vcat, [fill(i, n) for (i, n) in enumerate(counts)])
    (; g, y=0.5 .+ u[g] .+ randn(rng, length(g)), prior_sd=5.0)
end
default_centering_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | g)
    effect(mu, Intercept) ~ Normal(0, prior_sd)
    y ~ Normal(mu, sigma)
end
const COUNTS = Dict("weak" => fill(1, 32), "strong" => fill(50, 32),
    "unbalanced" => repeat([1, 50], 16))

function nuts_reference(name, rk, n)
    path = joinpath(OUT, "nuts_$name.jls")
    isfile(path) && return deserialize(path)
    get(ENV, "PROBE_NUTS", "1") == "0" &&
        return (; mean=fill(NaN, n), sd=fill(NaN, n), draws=zeros(n, 0))
    reference = try
        nuts_run(rk)
    catch err
        println("NUTS reference for $name failed: ", sprint(showerror, err))
        return (; mean=fill(NaN, n), sd=fill(NaN, n), draws=zeros(n, 0))
    end
    X = hcat([Matrix{Float64}(r.posterior_position) for r in reference.results]...)
    ref = (; mean=vec(mean(X; dims=2)), sd=vec(std(X; dims=2)), draws=X)
    serialize(path, ref)
    ref
end

function nuts_run(rk)
    problems = [rk_logdensity_problem(rk; ad_backend=NLME_AD) for _ in 1:4]
    init = brm_start(rk; chains=4, rng=Xoshiro(3))
    reference = completion_warmup_mcmc([Xoshiro(10 + c) for c in 1:4], problems;
        min_completed=4, n_draws=1000, blas_threads=1, init)
end

control(x) = BRM._brm_svi_control(x)

for spec in CASES
    parts = split(spec, ":")
    label, k, cmode, pmode = parts[1], parse(Int, parts[2]), parts[3], parts[4]
    cname = length(parts) >= 5 ? parts[5] : "weak"
    rk = RKBRMI(default_centering_model(default_centering_data(COUNTS[cname])))
    kw = Dict{Symbol,Any}()
    cmode == "none" && (kw[:centering] = nothing)
    cmode == "nodelay" && (kw[:centering] = LearnedCentering(; delay=false))
    pmode == "none" && (kw[:coupling] = nothing)
    trajectory = NamedTuple[]
    save = state -> begin
        run = state.run
        q = run.q
        cs = hasproperty(q, :centering) && q.centering !== nothing ?
            [control(x) for xs in q.centering.x for x in xs] : Float64[]
        cp = BRM._brm_svi_coupling(q)
        vn = cp === nothing ? 0.0 : sqrt(sum(sum(abs2, V) for V in cp.V; init=0.0) / length(cp.V))
        Msd = std([isempty(m) ? NaN : m[1] for m in q.M])
        usd = mean(sqrt(only(BRM._brm_svi_unit_variances(U))) for U in q.U)
        push!(trajectory, (; iteration=run.iteration,
            phase=first(split(BRM._brm_svi_phase_text(run, state.controls), ",")),
            μ=copy(q.μ), sd=exp.(q.ρ), c=cs, vnorm=vn, Msd, usd,
            elbo=isempty(run.windows) ? NaN : last(run.windows).center))
    end
    fit = brm_svi_fit(save, rk; subject_group=:g, batch_size=8, rng=Xoshiro(2), ndraws=4000,
        unit_reoptimize=k, kw...)
    ref = nuts_reference(cname, rk, length(fit.coordinates))
    names = fit.coordinates[fit.diagnostics.globals]
    gpos = fit.diagnostics.globals
    println("CASE $label k=$k centering=$cmode coupling=$pmode counts=$cname " *
        "iterations=$(fit.diagnostics.iterations) converged=$(fit.diagnostics.converged) " *
        "stages=$(fit.diagnostics.stages)")
    for (j, p) in enumerate(gpos)
        sv, sm = std(fit.draws[p, :]), mean(fit.draws[p, :])
        @printf("  %-20s svi mean %8.4f sd %8.4f | nuts mean %8.4f sd %8.4f | ratio %6.3f err %6.3f\n",
            names[j], sm, sv, ref.mean[p], ref.sd[p], sv / ref.sd[p], (sm - ref.mean[p]) / ref.sd[p])
    end
    cen = fit.diagnostics.centering
    if cen !== nothing
        cs = [x.c for x in cen.cells]
        @printf("  centering c: mean %.3f min %.3f max %.3f activation=%s\n", mean(cs),
            minimum(cs), maximum(cs), string(get(cen, :activation_window, missing)))
    end
    # Units: compare mean and sd of each unit's coordinate with NUTS.
    units = fit.diagnostics.units
    um = [mean(fit.draws[only(u), :]) for u in units]
    us = [std(fit.draws[only(u), :]) for u in units]
    nm = [ref.mean[only(u)] for u in units]
    ns = [ref.sd[only(u)] for u in units]
    @printf("  units: sd ratio median %.3f [%.3f, %.3f]; mean err/sd median %.3f max %.3f\n",
        median(us ./ ns), minimum(us ./ ns), maximum(us ./ ns),
        median(abs.(um .- nm) ./ ns), maximum(abs.(um .- nm) ./ ns))
    # Trajectory, every 5th window and the last.
    println("  trajectory (iteration phase | per-global mean/sd | c mean | ‖V‖ | sd(M) | unit sd | elbo)")
    for (n, s) in enumerate(trajectory)
        (n % 5 == 1 || n == length(trajectory)) || continue
        gl = join([@sprintf("%s=%.3f/%.3f", names[j], s.μ[j], s.sd[j]) for j in eachindex(s.μ)], " ")
        @printf("   %6d %-10s %s c=%.3f V=%.3f Msd=%.3f usd=%.3f elbo=%.2f\n", s.iteration,
            s.phase, gl, isempty(s.c) ? NaN : mean(s.c), s.vnorm, s.Msd, s.usd, s.elbo)
    end
    serialize(joinpath(OUT, "$label.jls"), (; spec, names, gpos, trajectory,
        diagnostics=(; fit.diagnostics.iterations, fit.diagnostics.converged,
            fit.diagnostics.stages, fit.diagnostics.q, fit.diagnostics.centering,
            fit.diagnostics.coupling, fit.diagnostics.elbo_windows_trimmed),
        draw_mean=vec(mean(fit.draws; dims=2)), draw_sd=vec(std(fit.draws; dims=2))))
    flush(stdout)
end
