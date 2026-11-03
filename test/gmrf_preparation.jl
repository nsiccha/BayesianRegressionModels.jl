using Test, BayesianRegressionModels, LinearAlgebra, Distributions
const BRM = BayesianRegressionModels

data = (; east=[0.1, 0.9, 0.7, 0.3], north=[0.2, 0.4, 0.8, 0.5],
    time=[0.0, 0.3, 0.3, 1.7], channel=["b", "a", "b", "a"], y=zeros(4),
    xy=[0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5], triangles=[1 2 5; 2 3 5; 3 4 5; 4 1 5],
    Omega=[1.0 0.4; 0.4 1.0])

@testset "GMRF formula identity, priors and local prepared state" begin
    model = @brm data begin
        field ~ 0 + gmrf((east, north), time, channel;
            mesh=(vertices=xy, cells=triangles), cor=Omega)
        sd(field, gmrf((east, north), time, channel)) = 1.2
        length_scale(field, gmrf((east, north), time, channel), space) = 0.4
        length_scale(field, gmrf((east, north), time, channel), time) ~ LogNormal(0.5, 0.7)
        y ~ Normal(field, 0.2)
    end
    terms = BRM._brm_term_address_map(model, :field)
    @test only(keys(terms)) == Symbol("gmrf((east,north),time,channel)")
    term = only(only(values(terms)))
    context = BRM._brm_backend_context(model)
    prepared = BRM._brm_prepare_term(term, :field, context)
    @test prepared.source == (:east, :north, :time, :channel)
    @test prepared.state.times == [0, 0.3, 1.7]
    @test prepared.state.channels == ["a", "b"]
    @test prepared.state.layout.ncoordinates == 30
    @test prepared.state.layout.times === prepared.state.times
    @test prepared.state.layout.channels === prepared.state.channels
    @test prepared.state.correlation.factor ≈ Matrix(cholesky(Symmetric(data.Omega)).L)
    @test size(prepared.state.indices) == (4, 3)
    @test size(prepared.state.weights) == (4, 3)
    @test all(1 .<= prepared.state.indices .<= 30)
    priors = BRM._brm_term_prior_expressions(prepared)
    @test priors.sigma isa BRM._BRMKnownTermValue
    @test priors.spatial_length isa BRM._BRMKnownTermValue
    @test BRM.getf(priors.temporal_length) === LogNormal
    @test BRM.getargs(priors.temporal_length) == (0.5, 0.7)
    # Geometry stays unevaluated for the RK host planner.
    metadata = BRM._brm_prepare_term(term, :field, context; geometry=false)
    @test !haskey(metadata.state, :mesh)
    @test !haskey(metadata.state, :indices)
    @test metadata.state.nvertices == 5
    @test !haskey(BRM._rk_prepare_term(term, :field, context).state, :mesh)
    rows = Dict{Symbol,Any}(:east=>[0.9, 0.1], :north=>[0.4, 0.2],
        :time=>[0.3, 0.0], :channel=>["a", "b"])
    replay = BRM._brm_replay_gmrf(prepared, (; data=rows))
    @test replay.state.mesh === prepared.state.mesh
    @test replay.state.layout === prepared.state.layout
    @test replay.state.indices == prepared.state.indices[[2, 1], :]
    @test replay.state.weights ≈ prepared.state.weights[[2, 1], :]
    rows[:time] = [2.0, 0.0]
    @test_throws ArgumentError BRM._brm_replay_gmrf(prepared, (; data=rows))
    # Off-row mesh arrays and correlations do not become observation axes.
    lengths = Tuple{Symbol,Int}[]
    BRM._brm_collect_data_lengths!(lengths, term)
    @test Set(first.(lengths)) == Set((:east, :north, :time, :channel))
end
