include("gmrf_preparation.jl")

@testset "Independent plate formulas retain scoped priors and fitted axes" begin
    input = merge(data, (; unit=[10, 20, 10, 10]))
    model = @brm input begin
        @plate for p in groups(unit)
            field ~ 0 + gmrf((east[p], north[p]), time[p];
                mesh=(vertices=xy, cells=triangles))
            sd(field, gmrf((east[p], north[p]), time[p])) = 1.2
            length_scale(field, gmrf((east[p], north[p]), time[p]), space) = 0.4
            length_scale(field, gmrf((east[p], north[p]), time[p]), time) ~ LogNormal(0.5, 0.7)
            y[p] ~ Normal(field, 0.2)
        end
    end
    operations = (BRM._named_op(raw) for (_, raw) in BRM._brm_operation_entries(model))
    kernel_op = only(op for op in operations
        if op isa BRM.ExprColumn && BRM.getf(op) === (~) &&
           BRM.getf(last(BRM.getargs(op))) === BRM.kernel)
    body = first(BRM.getargs(last(BRM.getargs(kernel_op))))
    _, statements = BRM._sb_kernel_lambda_parts(body)
    program = BRM._brm_cell_formula_program(statements, BRM._brm_inline_scope(body))
    @test program.targets == (:field,)
    @test length(program.positions) == 1
    @test length(program.prior_positions) == 3
    prepared = map(([1, 3, 4], [2])) do rows
        cell_data = (; east=data.east[rows], north=data.north[rows], time=data.time[rows],
            xy=data.xy, triangles=data.triangles)
        BRM._brm_prepare_cell_formulas(program, cell_data)
    end
    first_field, second_field = map(p -> only(p.terms.field), prepared)
    @test first_field.state.layout.ncoordinates == 15
    @test second_field.state.layout.ncoordinates == 5
    @test first_field.state.times == [0.0, 0.3, 1.7]
    @test second_field.state.times == [0.3]
    @test first_field.state.layout !== second_field.state.layout
    for p in prepared
        field = only(p.terms.field)
        priors = BRM._brm_term_prior_expressions(field)
        @test priors.sigma isa BRM._BRMKnownTermValue
        @test priors.spatial_length isa BRM._BRMKnownTermValue
        @test BRM.getf(priors.temporal_length) === LogNormal
        @test BRM.getargs(priors.temporal_length) == (0.5, 0.7)
    end
end
