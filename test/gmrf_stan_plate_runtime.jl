include("gmrf_plate_models.jl")
import BridgeStan as BS

@testset "Scoped GMRF descriptors and compiled independent joint laws" begin
    input = merge(data, (; unit=[10, 20, 10, 10], batch=[1, 1, 2, 2], other=zeros(4)))
    model = @brm input begin
        @plate for p in groups(unit)
            field ~ 0 + gmrf((east[p], north[p]), time[p];
                mesh=(vertices=xy, cells=triangles))
            sd(field, gmrf((east[p], north[p]), time[p])) ~ Exponential(0.7)
            length_scale(field, gmrf((east[p], north[p]), time[p]), space) ~ LogNormal(0.5, 0.8)
            length_scale(field, gmrf((east[p], north[p]), time[p]), time) ~ LogNormal(0.7, 0.6)
            y[p] ~ Normal(field, 0.2)
        end
        @plate for q in groups(batch)
            field ~ 0 + gmrf((east[q], north[q]), time[q];
                mesh=(vertices=xy, cells=triangles))
            sd(field, gmrf((east[q], north[q]), time[q])) ~ Exponential(0.7)
            length_scale(field, gmrf((east[q], north[q]), time[q]), space) ~ LogNormal(0.5, 0.8)
            length_scale(field, gmrf((east[q], north[q]), time[q]), time) ~ LogNormal(0.7, 0.6)
            other[q] ~ Normal(field, 0.2)
        end
    end
    sb = SBBRMI(model; mod=@__MODULE__)
    descriptor = brm_descriptor(sb; highlights=())
    problem = BRM.stan_instantiate(sb)
    names = BS.param_unc_names(problem.model)
    fits = [e.const_ for e in values(descriptor.plan.preproc) if e.kind === :cell_formulas]
    @test length(fits) == 2
    fields = brm_outputs(descriptor; predictor=:field, component=:field, kind=:parameter)
    @test length(fields) == 2
    @test length(unique(o.declaration.context for o in fields)) == 2
    roles = (:field, :sd, :spatial_length, :temporal_length)
    selections = map(fits) do fit
        selection = NamedTuple{roles}(map(roles) do role
            matches = [o for o in brm_outputs(descriptor; predictor=:field,
                component=role, kind=:parameter) if o.declaration.context == fit.context]
            @test length(matches) == 1
            output = only(matches)
            @test output.term === only(values(fit.bindings)).term_label
            @test output.predictors == (:field,)
            brm_output_coordinates(output, names)
        end)
        @test length(selection.field) == 20
        @test all(length(getproperty(selection, r)) == 2 for r in roles[2:end])
        selection
    end
    owned = reduce(vcat, [getproperty(s, r) for s in selections for r in roles])
    @test sort(owned) == collect(1:BS.param_unc_num(problem.model))
    @test length(unique(owned)) == length(owned)
    point = zeros(length(names))
    for s in selections
        point[s.field] = 0.03 .* sin.(1:length(s.field))
        point[s.sd] .= log(0.9)
        point[s.spatial_length] .= log(0.6)
        point[s.temporal_length] .= log(1.1)
    end
    function oracle(u)
        lp = 0.0
        for (fit, selection) in zip(fits, selections)
            offset = 0
            for (i, cell) in enumerate(fit.cells)
                prepared = only(e.const_.prepared for e in values(cell.preproc) if e.kind === :gmrf)
                state = prepared.state
                width = state.layout.ncoordinates
                z = u[selection.field[(offset + 1):(offset + width)]]
                offset += width
                sigma = exp(u[selection.sd[i]])
                spatial = exp(u[selection.spatial_length[i]])
                temporal = exp(u[selection.temporal_length[i]])
                kappa2 = 2 / spatial^2
                K = Matrix(state.mesh.stiffness) + kappa2 * Diagonal(state.mesh.mass)
                Q = K * Diagonal(1 ./ state.mesh.mass) * K / (4pi * kappa2)
                Ct = [exp(-abs(t-s)/temporal) for t in state.times, s in state.times]
                covariance = kron(Ct, inv(Symmetric(Q)))
                lp += logpdf(MvNormal(zeros(width), Symmetric(covariance)), z)
                mu = sigma .* vec(sum(state.weights .* z[state.indices]; dims=2))
                lp += sum(logpdf.(Normal.(mu, 0.2), zeros(length(mu))))
                lp += logpdf(Exponential(0.7), sigma) + log(sigma)
                lp += logpdf(LogNormal(0.5, 0.8), spatial) + log(spatial)
                lp += logpdf(LogNormal(0.7, 0.6), temporal) + log(temporal)
            end
        end
        lp
    end
    value, gradient = BS.log_density_gradient(problem.model, point; propto=false, jacobian=true)
    @test value ≈ oracle(point) atol=2e-8 rtol=2e-10
    for j in eachindex(point)
        plus, minus = copy(point), copy(point)
        h = 2e-6
        plus[j] += h
        minus[j] -= h
        @test gradient[j] ≈ (oracle(plus)-oracle(minus))/(2h) atol=3e-5 rtol=3e-6
    end
    rows = [2]
    replay_data = (; (k => getproperty(input, k)[rows] for k in
        (:east, :north, :time, :y, :other, :unit, :batch))...)
    replay = brm_descriptor(reprocess(sb, replay_data); highlights=())
    replay_fields = brm_outputs(replay; predictor=:field, component=:field, kind=:parameter)
    @test length(replay_fields) == 2
    for field in fields
        after = only(o for o in replay_fields if o.declaration.context == field.declaration.context)
        @test after.dims == field.dims
        @test after.term === field.term
        @test brm_output_coordinates(after, names) == brm_output_coordinates(field, names)
    end
end
