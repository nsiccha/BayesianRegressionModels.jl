include("gmrf_plate_priors.jl")

@testset "Independent field ownership follows emitted declaration contexts" begin
    input = merge(data, (; unit=[10, 20, 10, 10], batch=[1, 1, 2, 2], other=zeros(4)))
    model = @brm input begin
        @plate for p in groups(unit)
            field ~ 0 + gmrf((east[p], north[p]), time[p];
                mesh=(vertices=xy, cells=triangles))
            y[p] ~ Normal(field, 0.2)
        end
        @plate for q in groups(batch)
            field ~ 0 + gmrf((east[q], north[q]), time[q];
                mesh=(vertices=xy, cells=triangles))
            other[q] ~ Normal(field, 0.3)
        end
    end
    sb = SBBRMI(model; mod=@__MODULE__)
    plan = generative_plan(sb)
    fits = [e.const_ for e in values(plan.preproc) if e.kind === :cell_formulas]
    @test length(fits) == 2
    @test length(unique(fit.context for fit in fits)) == 2
    owned = Set{Tuple}()
    for fit in fits
        predictor = only(fit.predictors)
        @test predictor.logical === :field
        @test predictor.link === identity
        target, binding = only(fit.bindings)
        @test binding.term_predictor === predictor.emitted
        @test binding.term_parameters ==
            (; sd=:sigma, spatial_length=:spatial_length, temporal_length=:temporal_length,
               field=:field)
        # The retained context names a real nested declaration of this plan.
        sites = [d for d in plan.declarations if d.context == fit.context && d.target === target]
        @test length(sites) == 1
        @test only(sites).role === :prior
        push!(owned, (fit.context..., target))
        for cell in fit.cells
            prepared = only(e.const_.prepared for e in values(cell.preproc) if e.kind === :gmrf)
            nkey = only(sites).keywords.n_field
            @test cell.data[nkey] == prepared.state.layout.ncoordinates
            @test prepared.state.target === predictor.emitted
        end
        @test !haskey(plan.data, BRM._SB_BINDINGS_KEY)
        @test all(!haskey(cell.data, BRM._SB_BINDINGS_KEY) for cell in fit.cells)
    end
    @test length(owned) == 2
    # Removing all observations of one unit keeps both fitted field cells and
    # their semantic owners; a second plate still owns its own two cells.
    rows = [2]
    replay_rows = (; (k => getproperty(input, k)[rows] for k in
        (:east, :north, :time, :y, :other, :unit, :batch))...)
    replay = generative_plan(reprocess(sb, replay_rows))
    fresh = Dict(e.const_.context => e.const_ for e in values(replay.preproc)
        if e.kind === :cell_formulas)
    for fit in fits
        after = fresh[fit.context]
        @test after.labels == fit.labels
        @test after.bindings == fit.bindings
        @test after.predictors == fit.predictors
        @test [c.data[k] for c in after.cells for k in keys(c.data) if endswith(string(k), "_n_field")] ==
              [c.data[k] for c in fit.cells for k in keys(c.data) if endswith(string(k), "_n_field")]
    end
end
