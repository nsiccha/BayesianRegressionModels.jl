# Strict unit partitions: structural ownership, including dependencies hidden
# at the attribution point, and units with an empty observation group.
# Run: julia --project=test test/nlme_partition.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_panel_fixture.jl"))
using Random
using SparseArrays

@stestset "curvature patterns retain only pairs sharing a cell" begin
    # Packed coordinates need not follow the unit's local coordinate order.
    # Repeated cells and repeated reads do not add edges; an unobserved effect
    # and an empty unit keep their zero rows/columns.
    blocks = [[70, 10, 50, 90, 30], [400, 200], Int[]]
    reads = [([[70, 50], [50, 30]], [[50, 70], Int[], [90]]),
        ([[400, 200, 400]], Vector{Int}[]), (Vector{Int}[], Vector{Int}[])]
    saved = deepcopy(reads)
    patterns = BRM._brm_nlme_eta_patterns(blocks, reads)
    @test patterns[1] == Bool[1 0 1 0 0; 0 0 0 0 0; 1 0 1 0 1;
        0 0 0 1 0; 0 0 1 0 1]
    @test patterns[2] == trues(2, 2)
    @test size(patterns[3]) == (0, 0)
    @test all(P -> P isa SparseMatrixCSC{Bool,Int}, patterns)
    @test reads == saved
end

@stestset "repeated observations do not accumulate duplicate curvature pairs" begin
    blocks = [collect(10:10:5120)]
    # 4,000 observations share 128 effects. Their curvature still has only
    # 128² entries; retaining all cell pairs would allocate over 1 GB.
    support = blocks[1][1:4:end]
    reads = [([copy(support) for _ in 1:4000],)]
    patterns = BRM._brm_nlme_eta_patterns(blocks, reads)
    @test size(only(patterns)) == (512, 512)
    @test nnz(only(patterns)) == 128^2
    @test only(patterns)[1, 509]
    @test !only(patterns)[1, 2]
    # Warmed allocation budget allows the linear support copies and the
    # sparse output, but rules out accumulating every repeated cell pair.
    bytes = @allocated BRM._brm_nlme_eta_patterns(blocks, reads)
    @test bytes < 32 * 1024^2
end

@stestset "global scalar expressions remain authorable with strict partitions" begin
    # The shared predictor has its own row domain. Its literal-index reads
    # are numbers, even when the observation consumes them beside a vector.
    # Keep the natural formula: no annotations, row-gather respelling, or
    # relaxed locality to work around the planner's scalar inference.
    data = (; x=[-1.0, 0.5, 2.0], subject=["a", "b", "a", "b"],
        y=[0.2, -0.1, 0.7, 0.4])
    rk = RKBRMI(@brm data begin
        shared ~ 1 + x
        mu ~ 1 + (1 | subject)
        weight = exp(shared[1])
        shift = shared[2] - shared[1]
        location = weight + shift + mu
        y ~ Normal(location, 1)
    end)
    @test Base.invokelatest(likelihood_query(rk), zeros(rk.model.layout.total)) ≈
        sum(logpdf.(Normal(1, 1), data.y))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    rng = Xoshiro(18)
    for _ in 1:2
        θ = 0.1 .* randn(rng, length(m.theta))
        σ = 0.1 .* randn(rng, length(m.sigma))
        H = [0.2 .* randn(rng, length(b)) for b in m.view.subject_coordinates]
        expected, gradients = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
        for i in eachindex(H)
            value, gradient = brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, H[i])
            @test value ≈ expected[i] atol=1e-9 rtol=1e-10
            @test gradient ≈ gradients[i] atol=1e-8 rtol=1e-9
            @test brm_nlme_loglikelihood(m, i, θ, σ, H[i]) ≈ expected[i] atol=1e-9 rtol=1e-10
            step = 1e-5
            independent = map(eachindex(H[i])) do j
                plus, minus = copy(H[i]), copy(H[i])
                plus[j] += step
                minus[j] -= step
                (brm_nlme_loglikelihood(m, i, θ, σ, plus) -
                 brm_nlme_loglikelihood(m, i, θ, σ, minus)) / (2step)
            end
            @test gradient ≈ independent atol=1e-7 rtol=1e-7
        end
    end
end

@stestset "a dormant cross-unit branch is refused before returning a model" begin
    data = (; subject=["a", "b", "c"], other=[2, 3, 1], y=[0.1, 0.2, 0.3])
    rk = RKBRMI(@brm data begin
        mu ~ 1 + (1 | subject)
        all_mu = mu
        @plate for i in eachindex(y)
            y[i] ~ Normal(ifelse(all_mu[other[i]] > 10, all_mu[other[i]], mu[i]), 1)
        end
    end)
    @test Base.invokelatest(likelihood_query(rk), zeros(rk.model.layout.total)) ≈
        sum(logpdf.(Normal(), data.y))
    # The branch is dormant at the unit point and the ordinary attribution
    # perturbations. Its structural footprint still includes the other unit.
    error = try
        brm_nlme_model(rk; ad_backend=NLME_AD)
        nothing
    catch e
        e
    end
    @test error isa BRMNLMEViewError
    @test occursin("another subject", sprint(showerror, error))
    @test occursin("b_subject", sprint(showerror, error))
end

# Elementwise in value, but written as a scan, which has no row form: a cell
# reading it at its rows would compute it whole, every unit's effects included.
halved_by_scan(x) = 0.5 .* x
function BRM._rk_callable_source!(definitions, bindings, entry, ::typeof(halved_by_scan))
    push!(definitions, :(ReactiveKernels.@kernel $entry(x) = begin
        values = ReactiveKernels.scan(x; init=0.0) do carry, value
            next = 0.5 * value
            (next, next)
        end
        return values
    end))
    :done
end

@stestset "a value computed whole from the units' effects is refused by name" begin
    # Guards the ArbeitsplatzAtlas stall (snag saem-s-nuts-warm-28e5f1d9, todo
    # 0y5jynw): cells that computed a field whole made one unit's gradient cost
    # more than the whole model's. The user asked for a check that errors out,
    # not a threshold (decision 1vncxdq). Each density reads only its own
    # subject's effect, so attribution accepts the model; the refusal is RK's
    # strict locality check at preparation, naming the value and its producer.
    data = (; subject=["a", "b", "c", "a", "b", "c"], y=[0.1, 0.2, 0.3, -0.1, 0.4, 0.0])
    whole = RKBRMI(@brm data begin
        mu ~ 1 + (1 | subject)
        h = halved_by_scan(mu)
        y ~ Normal(h, 1)
    end)
    local_h = RKBRMI(@brm data begin
        mu ~ 1 + (1 | subject)
        h = 0.5 .* mu
        y ~ Normal(h, 1)
    end)
    u = 0.3 .* randn(Xoshiro(3), whole.model.layout.total)
    @test whole.model.layout.total == local_h.model.layout.total
    @test Base.invokelatest(likelihood_query(whole), u) ≈
        Base.invokelatest(likelihood_query(local_h), u) atol=1e-12
    refusal = try
        brm_nlme_model(whole; ad_backend=NLME_AD)
        nothing
    catch e
        e
    end
    @test refusal isa ReactiveKernels.PartitionLocalityError
    text = sprint(showerror, refusal)
    @test occursin("`h` is computed whole", text)
    @test occursin("ReactiveKernels.scan", text)
    # The same model with a row form prepares.
    @test brm_nlme_model(local_h; ad_backend=NLME_AD) isa BRM.BRMNLMEModel
end

@stestset "a unit can have no cells of one observation group" begin
    data = panel_data()
    x1 = copy(data.x1)
    x1[data.region .== 1] .= missing
    rk = RKBRMI(panel(merge(data, (; x1))); completions=:noncentered)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:region,
        nested_groups=PANEL_NESTED)
    observed = only(filter(c -> c.names == (:x1_obs,), m.cell_query))
    @test isempty(observed.positions[1])
    rng = Xoshiro(42)
    θ = 0.1 .* randn(rng, length(m.theta))
    σ = 0.1 .* randn(rng, length(m.sigma))
    H = [0.2 .* randn(rng, length(b)) for b in m.view.subject_coordinates]
    expected, gradients = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    for i in eachindex(H)
        value, gradient = brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, H[i])
        @test value ≈ expected[i] atol=1e-9 rtol=1e-10
        @test gradient ≈ gradients[i] atol=1e-8 rtol=1e-9
        @test brm_nlme_loglikelihood(m, i, θ, σ, H[i]) ≈ expected[i] atol=1e-9 rtol=1e-10
    end
end

@stestset "one-unit calls write a dense point in place and hold θ and σ fixed for η" begin
    # The same three subjects repeated: 100 times the units multiplies the
    # packed coordinates but not one unit's work.
    function plate_data(n)
        base = NLME_PLATE_DATA
        pick = [mod1(i, length(base.subject)) for i in 1:n]
        (; subject=["s$i" for i in 1:n], dose=base.dose[pick],
            weight=base.weight[pick] .+ 0.01 .* (1:n), t=base.t[pick], dv=base.dv[pick])
    end
    rng = Xoshiro(7)
    allocations = map((3, 300)) do n
        m = brm_nlme_model(RKBRMI(nlme_plate_pk(plate_data(n))); ad_backend=NLME_AD)
        θ = 0.1 .* randn(rng, length(m.theta))
        σ = 0.1 .* randn(rng, length(m.sigma))
        H = [0.2 .* randn(rng, length(b)) for b in m.view.subject_coordinates]
        plan = BRM._brm_nlme_cell_state(m).plan
        for i in eachindex(H)
            # The η route differentiates only the unit's own effects; it
            # agrees with the η entries of the full θ, σ and η gradient.
            value, gradient = brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, H[i])
            full, indices, selected, block =
                BRM._brm_nlme_cell_value_and_selected_gradient(m, i, θ, σ, H[i])
            @test value ≈ full rtol=1e-12
            @test gradient ≈ selected[[searchsortedfirst(indices, j) for j in block]] rtol=1e-10 atol=1e-12
            @test issubset(plan.local_footprints[i], block)
            @test brm_nlme_loglikelihood(m, i, θ, σ, H[i]) ≈ value rtol=1e-12
        end
        # Outside a call the work points equal the unit point again, also
        # after an argument error.
        @test_throws DimensionMismatch brm_nlme_loglikelihood(m, 1, θ, σ, H[1][1:end-1])
        @test plan.point == m.unit
        @test m.cell_gradient.point == m.unit
        brm_nlme_loglikelihood_and_gradient(m, 1, θ, σ, H[1])
        brm_nlme_loglikelihood(m, 1, θ, σ, H[1])
        # Control, logged and not asserted: the lazy overlay 0.2.122 passed,
        # which RK's partition gradient copies into a full packed vector.
        overlay = BRM._BRMNLMEPoint(m.unit, plan.indices[1], m.unit[plan.indices[1]],
            Int[], Float64[], Int[], Float64[])
        BRM._rk_nlme_partition_value_and_gradient!(plan.sampler, plan.gradients[1], overlay, 1)
        (; packed=length(m.unit),
            gradient=@allocated(brm_nlme_loglikelihood_and_gradient(m, 1, θ, σ, H[1])),
            value=@allocated(brm_nlme_loglikelihood(m, 1, θ, σ, H[1])),
            overlay=@allocated(BRM._rk_nlme_partition_value_and_gradient!(plan.sampler,
                plan.gradients[1], overlay, 1)))
    end
    @info "one unit's allocations (bytes) at 3 and 300 units" allocations
    @test allocations[2].packed > 40allocations[1].packed
    @test allocations[2].gradient <= 1.2allocations[1].gradient
    @test allocations[2].value <= 1.2allocations[1].value
end
