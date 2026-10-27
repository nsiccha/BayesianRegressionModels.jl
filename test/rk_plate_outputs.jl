# test/rk_plate_outputs.jl — every indexed output of an `@plate for` cell is
# an RKBRMI graph value of its own name, not only the cell's last (snag
# `rk-graph-value-f-06f92c07`).
#
# The lambda a cell lowers to binds a non-last output `conc[i] = …` exactly as
# it binds a bare local `conc = …`, so both spellings emit the same Stan
# program and the same RK density. RKBRMI additionally collects the indexed one
# as `conc = flatten_cells(conc_cells)`, which RKPPL runs only in a query that
# reads it. Oracles: each queried value is recomputed from the graph's own
# predictor and parameter values, never from the plate. Public synthetic
# models only.
#
# Run: julia --startup-file=no --project=test test/rk_plate_outputs.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PlateOutputModels
using BayesianRegressionModels, Distributions
# Two indexed outputs; the observation reads the first.
two_outputs(d) = @brm d begin
    sigma ~ Exponential(1)
    log_k ~ Normal(0, 0.5)
    log_V ~ 1 + (1 | p | subject)
    effect(log_V, Intercept) ~ Normal(1, 0.5)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_V)
        conc[i] = dose[i] / exp(log_V[i]) .* exp.(-exp(log_k) .* t[i])
        half[i] = 0.5 .* conc[i]
        y[i] ~ Normal(conc[i], sigma)
    end
end
# The same cell with `conc` a bare local.
two_outputs_local(d) = @brm d begin
    sigma ~ Exponential(1)
    log_k ~ Normal(0, 0.5)
    log_V ~ 1 + (1 | p | subject)
    effect(log_V, Intercept) ~ Normal(1, 0.5)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_V)
        conc = dose[i] / exp(log_V[i]) .* exp.(-exp(log_k) .* t[i])
        half[i] = 0.5 .* conc
        y[i] ~ Normal(conc, sigma)
    end
end
# A joint cell: the location of an in-cell observation on another axis is an
# indexed output before the cell's collected one, and a weighted observation
# reads it.
joint(d) = @brm d begin
    sigma ~ Exponential(1)
    qt_scale ~ Exponential(1)
    qt_base ~ Normal(0, 1)
    qt_slope ~ Normal(0, 1)
    log_k ~ Normal(0, 0.5)
    log_V ~ 1 + (1 | p | subject)
    effect(log_V, Intercept) ~ Normal(1, 0.5)
    sd(:, p) ~ Exponential(1)
    ecg = groups(ecg_subject => subject)
    @plate for i in eachindex(log_V)
        conc = dose[i] / exp(log_V[i]) .* exp.(-exp(log_k) .* t[i])
        ecg_conc = dose[i] / exp(log_V[i]) .* exp.(-exp(log_k) .* ecg_t[ecg[i]])
        qt_loc[i] = qt_base .+ qt_slope .* ecg_conc
        ecg_y[ecg[i]] ~ weighted(Normal(qt_loc[i], qt_scale), aweights(ecg_k[ecg[i]]))
        pk_loc[i] = conc
        y[i] ~ Normal(pk_loc[i], sigma)
    end
end
joint_local(d) = @brm d begin
    sigma ~ Exponential(1)
    qt_scale ~ Exponential(1)
    qt_base ~ Normal(0, 1)
    qt_slope ~ Normal(0, 1)
    log_k ~ Normal(0, 0.5)
    log_V ~ 1 + (1 | p | subject)
    effect(log_V, Intercept) ~ Normal(1, 0.5)
    sd(:, p) ~ Exponential(1)
    ecg = groups(ecg_subject => subject)
    @plate for i in eachindex(log_V)
        conc = dose[i] / exp(log_V[i]) .* exp.(-exp(log_k) .* t[i])
        ecg_conc = dose[i] / exp(log_V[i]) .* exp.(-exp(log_k) .* ecg_t[ecg[i]])
        qt_loc = qt_base .+ qt_slope .* ecg_conc
        ecg_y[ecg[i]] ~ weighted(Normal(qt_loc, qt_scale), aweights(ecg_k[ecg[i]]))
        pk_loc[i] = conc
        y[i] ~ Normal(pk_loc[i], sigma)
    end
end
# Two cells over the same subjects that both write `shared[i]`.
const SHARED_CALLS = Ref(0)
counted(x) = (SHARED_CALLS[] += 1; x)
two_cells(d) = @brm d begin
    sigma ~ Exponential(1)
    log_k ~ Normal(0, 0.5)
    log_V ~ 1 + (1 | p | subject)
    effect(log_V, Intercept) ~ Normal(1, 0.5)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_V)
        shared[i] = exp.(-exp(log_k) .* t[i])
        first_only[i] = 2 .* shared[i]
        loc[i] = dose[i] / exp(log_V[i]) .* shared[i]
        y[i] ~ Normal(loc[i], sigma)
    end
    @plate for i in eachindex(log_V)
        shared[i] = exp.(-exp(log_k) .* u[i])
        unread[i] = counted(3 .* shared[i])
        loc2[i] = dose[i] / exp(log_V[i]) .* shared[i]
        z[i] ~ Normal(loc2[i], sigma)
    end
end
end
const PM = PlateOutputModels

const PK = (; subject=["s1", "s2", "s3", "s4"], dose=[100.0, 120.0, 80.0, 150.0],
    t=[[0.5, 1.0, 2.0], [0.25, 1.5], [0.5, 1.0, 2.0, 4.0], [1.0]],
    u=[[0.2, 0.9], [1.1], [0.3, 0.6, 3.0], [0.5, 2.5]],
    y=[[1.0, 0.8, 0.5], [1.2, 0.7], [0.9, 0.7, 0.4, 0.2], [0.6]],
    z=[[0.9, 0.6], [1.1], [1.0, 0.8, 0.3], [1.2, 0.4]],
    # ECG reads on their own axis, listed out of subject order.
    ecg_subject=["s2", "s1", "s3", "s1", "s4", "s3", "s2"],
    ecg_t=[0.5, 1.0, 2.0, 3.0, 0.75, 4.0, 2.5],
    ecg_y=[0.3, 0.2, -0.1, 0.1, 0.4, 0.0, -0.2],
    ecg_k=[1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0])

function graph_values(rk, wants, u)
    bound = rk_bound_data(rk)
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, keys(bound)...), want=Tuple(wants), bound)
    value = Base.invokelatest(query, u)
    value isa Tuple ? value : (value,)
end
point(rk) = collect(range(-0.6, 0.8; length=rk.model.layout.total))
problem(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
emitted_main(brmi) = sprint(Base.show_unquoted, BRM._rk_emit_ast(BRM._brm_rk_plan(brmi)).main)

# Subject concentrations at `times(s)` from the graph's own `log_V`, `log_k`.
function concentrations(rk, u, times)
    log_V, log_k = graph_values(rk, (:log_V, :log_k), u)
    [PK.dose[s] / exp(log_V[s]) .* exp.(-exp(log_k) .* times(s)) for s in eachindex(PK.subject)]
end

@stestset "a cell's non-last indexed output is a graph value of its name" begin
    rk = RKBRMI(PM.two_outputs(PK))
    u = point(rk)
    conc, half = graph_values(rk, (:conc, :half), u)
    expected = reduce(vcat, concentrations(rk, u, s -> PK.t[s]))
    @test length(conc) == sum(length, PK.t)
    @test conc ≈ expected atol=1e-12 rtol=1e-12
    @test half == 0.5 .* conc
    # Queried alone, too.
    @test only(graph_values(rk, (:conc,), u)) == conc
    # The bare-local spelling has the same density and gradient, bit for bit,
    # and no `conc` value.
    other = RKBRMI(PM.two_outputs_local(PK))
    @test coordinate_names(other.model.layout) == coordinate_names(rk.model.layout)
    for v in (u, zeros(length(u)), fill(0.2, length(u)))
        @test LogDensityProblems.logdensity_and_gradient(problem(rk), v) ==
              LogDensityProblems.logdensity_and_gradient(problem(other), v)
    end
    @test occursin("conc = flatten_cells(conc_cells)", emitted_main(PM.two_outputs(PK)))
    @test !occursin("conc_cells", emitted_main(PM.two_outputs_local(PK)))
end

@stestset "an indexed output keeps the Stan program and descriptor of the bare local" begin
    for (indexed, bare, name) in ((PM.two_outputs, PM.two_outputs_local, :conc),
                                  (PM.joint, PM.joint_local, :qt_loc))
        a, b = SBBRMI(indexed(PK); mod=@__MODULE__), SBBRMI(bare(PK); mod=@__MODULE__)
        @test stan_code(a) == stan_code(b)
        da, db = brm_descriptor(a), brm_descriptor(b)
        @test da.id == db.id
        oa, ob = brm_output(da, name), brm_output(db, name)
        @test (oa.name, oa.logical, oa.role) == (ob.name, ob.logical, ob.role)
        @test oa.segments == ob.segments
    end
end

@stestset "an in-cell observation's location on another axis is a graph value" begin
    rk = RKBRMI(PM.joint(PK))
    u = point(rk)
    qt_loc, pk_loc = graph_values(rk, (:qt_loc, :pk_loc), u)
    qt_base, qt_slope = graph_values(rk, (:qt_base, :qt_slope), u)
    # Each subject's ECG rows in their authored order, subjects in cell order.
    rows = [findall(==(s), PK.ecg_subject) for s in PK.subject]
    ecg = concentrations(rk, u, s -> PK.ecg_t[rows[s]])
    @test length(qt_loc) == length(PK.ecg_subject)
    @test qt_loc ≈ reduce(vcat, [qt_base .+ qt_slope .* c for c in ecg]) atol=1e-12 rtol=1e-12
    @test pk_loc ≈ reduce(vcat, concentrations(rk, u, s -> PK.t[s])) atol=1e-12 rtol=1e-12
    other = RKBRMI(PM.joint_local(PK))
    for v in (u, zeros(length(u)))
        @test LogDensityProblems.logdensity_and_gradient(problem(rk), v) ==
              LogDensityProblems.logdensity_and_gradient(problem(other), v)
    end
    # The same value on a frozen replay of other subjects.
    keep = [3, 1]
    picked = [r for s in keep for r in rows[s]]
    replay = reprocess(rk, (; subject=PK.subject[keep], dose=PK.dose[keep], t=PK.t[keep],
        y=PK.y[keep], ecg_subject=PK.ecg_subject[picked], ecg_t=PK.ecg_t[picked],
        ecg_y=PK.ecg_y[picked], ecg_k=PK.ecg_k[picked]))
    ends = cumsum([0; length.(rows)])
    @test only(graph_values(replay, (:qt_loc,), u)) ≈
          reduce(vcat, [qt_loc[(ends[s] + 1):ends[s + 1]] for s in keep]) atol=1e-12 rtol=1e-12
end

@stestset "a name two cells write stays a cell value; unread outputs never run" begin
    brmi = PM.two_cells(PK)
    rk = RKBRMI(brmi)
    u = point(rk)
    first_only, unread = graph_values(rk, (:first_only, :unread), u)
    shared_t = reduce(vcat, [exp.(-exp(only(graph_values(rk, (:log_k,), u))) .* t) for t in PK.t])
    shared_u = reduce(vcat, [exp.(-exp(only(graph_values(rk, (:log_k,), u))) .* t) for t in PK.u])
    @test first_only ≈ 2 .* shared_t atol=1e-12 rtol=1e-12
    @test unread ≈ 3 .* shared_u atol=1e-12 rtol=1e-12
    # refused: two cells write `shared`, so no one value has that name; SBBRMI's
    # descriptor leaves it ambiguous likewise (decision 1tpze5q).
    @test !occursin("shared = flatten_cells", emitted_main(brmi))
    @test_throws "kernel has no port :shared" graph_values(rk, (:shared,), u)
    # A value no density reads runs only when queried.
    p = problem(rk)
    LogDensityProblems.logdensity_and_gradient(p, u)
    PM.SHARED_CALLS[] = 0
    for v in (u, zeros(length(u)))
        LogDensityProblems.logdensity(p, v)
        LogDensityProblems.logdensity_and_gradient(p, v)
    end
    @test PM.SHARED_CALLS[] == 0
end
