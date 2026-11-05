# Offset-only formulas keep their producer rows before an explicit observation
# gather. Run: julia --project=test test/offset_only_axes.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

@brm function offset_axis_rows(x)
    theta ~ Normal(0, 1)
    @plate for i in eachindex(x)
        value[i] = theta * x[i]
    end
    return value
end

function offset_axis_builder(component, form; observed=true)
    producer = component ? :(latent ~ offset_axis_rows(x)) : quote
        theta ~ Normal(0, 1)
        @plate for i in eachindex(x)
            latent[i] = theta * x[i]
        end
    end
    formula = form === :direct ?
        :(log(quantity) ~ 0 + offset(level) + offset(latent) + offset(fixed)) :
        form === :expression ?
        :(log(quantity) ~ 0 + offset(level) + offset(latent) + offset(-0.02 * fixed)) :
        :(log(quantity) ~ 0 + offset(level) + offset(latent))
    reader = observed ? quote
        readout = quantity[position]
        y ~ Normal(readout, 0.4)
    end : quote
        @plate for j in eachindex(position)
            y[j] ~ Normal(quantity[position[j]], 0.4)
        end
    end
    brm(quote
        $producer
        level ~ Normal(0, 1)
        $formula
        $reader
    end; mod=@__MODULE__)
end

offset_axis_fixed(form, data) = form === :direct ? data.fixed :
    form === :expression ? -0.02 .* data.fixed : zeros(length(data.x))

@stestset "offset-only producer axes survive a differently sized reader" begin
    for component in (false, true), form in (:direct, :expression, :field_only)
        builder = offset_axis_builder(component, form)
        data = (; x=[0.2, -0.4, 0.7], fixed=[1.0, 2.0, 3.0],
            position=[3, 1, 1, 2], y=[1.0, 1.1, 0.9, 1.2])
        saved = deepcopy(data)
        rk, problem = consumer_problem(builder(data))
        @test length(coordinate_names(rk.model.layout)) == 2
        for (theta, level) in ((0.3, 0.2), (-0.2, 0.1), (0.0, -0.3))
            parent_starts = component ? (; level) : (; theta, level)
            local_starts = component ? [:latent => (; theta)] : Pair[]
            u = brm_start(rk, parent_starts; component_inits=local_starts)
            # Locate coordinates from explicit public starts, independent of
            # the component compiler's generated parameter names.
            itheta = only(findall(==(theta), u))
            ilevel = only(findall(==(level), u))
            oracle(v) = begin
                quantity = exp.(v[ilevel] .+ v[itheta] .* data.x .+
                    offset_axis_fixed(form, data))
                logpdf(Normal(), v[ilevel]) + logpdf(Normal(), v[itheta]) +
                    sum(logpdf.(Normal.(quantity[data.position], 0.4), data.y))
            end
            expected = exp.(level .+ theta .* data.x .+ offset_axis_fixed(form, data))
            values = rk_draw_values(rk, reshape(u, :, 1), (:quantity, :readout))
            @test only(values[:quantity]) ≈ expected
            @test only(values[:readout]) ≈ expected[data.position]
            check_consumer_point(problem, u, oracle)
        end
        @test isequal(data, saved)
        # Rebuild with a changed producer size, still distinct from the reader.
        changed = (; x=[-0.2, 0.6], fixed=[2.0, -1.0],
            position=[2, 1, 2], y=[0.7, 1.2, 0.9])
        replay = reprocess(rk, changed)
        starts = component ? (; level=0.2) : (; theta=0.3, level=0.2)
        u = brm_start(replay, starts;
            component_inits=component ? [:latent => (; theta=0.3)] : Pair[])
        expected = exp.(0.2 .+ 0.3 .* changed.x .+ offset_axis_fixed(form, changed))
        values = rk_draw_values(replay, reshape(u, :, 1), (:quantity, :readout))
        @test only(values[:quantity]) ≈ expected
        @test only(values[:readout]) ≈ expected[changed.position]
    end
end

@stestset "offset-only axes need no model-scope observed row source" begin
    for component in (false, true)
        data = (; x=[0.2, -0.4, 0.7], position=[3, 1, 1, 2],
            y=[1.0, 1.1, 0.9, 1.2])
        rk = RKBRMI(offset_axis_builder(component, :field_only; observed=false)(data))
        starts = component ? (; level=0.2) : (; level=0.2, theta=0.3)
        u = brm_start(rk, starts;
            component_inits=component ? [:latent => (; theta=0.3)] : Pair[])
        expected = exp.(0.2 .+ 0.3 .* data.x)
        values = rk_draw_values(rk, reshape(u, :, 1), (:quantity,))
        @test only(values[:quantity]) ≈ expected
    end
end

@stestset "fixed-expression-only offsets retain their data rows" begin
    data = (; fixed=[1.0, 2.0, 3.0], position=[3, 1, 1, 2],
        y=[1.0, 1.1, 0.9, 1.2])
    builder = @brm begin
        level ~ Normal(0, 1)
        log(quantity) ~ 0 + offset(level) + offset(-0.02 * fixed)
        readout = quantity[position]
        y ~ Normal(readout, 0.4)
    end
    rk = RKBRMI(builder(data))
    u = brm_start(rk, (; level=0.2))
    expected = exp.(0.2 .- 0.02 .* data.fixed)
    values = rk_draw_values(rk, reshape(u, :, 1), (:quantity, :readout))
    @test only(values[:quantity]) ≈ expected
    @test only(values[:readout]) ≈ expected[data.position]
end

@stestset "offset-only row validation preserves scalar-cell and length checks" begin
    builder = offset_axis_builder(false, :direct)
    @test_throws r"mixes row axes|aligned values" RKBRMI(builder((;
        x=[0.2, -0.4, 0.7], fixed=[1.0, 2.0],
        position=[3, 1, 1, 2], y=[1.0, 1.1, 0.9, 1.2])))
    # Three cells and three flattened values do not prove three scalar rows.
    invalid = @brm begin
        theta ~ Normal(0, 1)
        @plate for i in eachindex(x)
            latent[i] = theta .* x[i]
        end
        quantity ~ 0 + offset(latent)
        y ~ Normal(quantity, 0.4)
    end
    @test_throws r"offset of a plate output requires one real scalar per cell" begin
        rk = RKBRMI(invalid((; x=[[0.2, 0.4], Float64[], [0.8]], y=ones(3))))
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        LogDensityProblems.logdensity(problem, brm_start(rk, (; theta=0.3)))
    end
end

@stestset "offset-only gathers use index rows rather than input dependencies" begin
    builder = @brm begin
        level ~ Normal(0, 1)
        log(quantity) ~ 0 + offset(level) + offset(fixed[producer_row])
        readout = quantity[position]
        y ~ Normal(readout, 0.4)
    end
    data = (; fixed=[0.2, -0.4, 0.7, 0.5, -0.1], producer_row=[5, 2, 2],
        position=[3, 1, 1, 2], y=[1.0, 1.1, 0.9, 1.2])
    rk = RKBRMI(builder(data))
    u = brm_start(rk, (; level=0.2))
    values = rk_draw_values(rk, reshape(u, :, 1), (:quantity, :readout))
    expected = exp.(0.2 .+ data.fixed[data.producer_row])
    @test only(values[:quantity]) ≈ expected
    @test only(values[:readout]) ≈ expected[data.position]
end

@stestset "offset-only physical points agree with compiled Stan" begin
    data = (; x=[0.2, -0.4, 0.7], fixed=[1.0, 2.0, 3.0],
        position=[3, 1, 1, 2], y=[1.0, 1.1, 0.9, 1.2])
    for component in (false, true), form in (:expression, :field_only)
        model = offset_axis_builder(component, form)(data)
        rk = RKBRMI(model)
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        sb = SBBRMI(model; mod=@__MODULE__)
        stan = BRM.stan_instantiate(sb;
            path=joinpath(mktempdir(), "offset-only-$component-$form.stan"))
        transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
        for (theta, level) in ((0.3, 0.2), (-0.2, 0.1))
            starts = component ? (; level) : (; theta, level)
            local_starts = component ? [:latent => (; theta)] : Pair[]
            r = brm_start(rk, starts; component_inits=local_starts)
            s = brm_start(sb, stan.model, starts; component_inits=local_starts)
            @test brm_rk_point_to_stan(transport, r) ≈ s
            rv, rg = LogDensityProblems.logdensity_and_gradient(problem, r)
            sv, sg = BridgeStan.log_density_gradient(stan.model, s;
                propto=false, jacobian=true)
            @test rv ≈ sv atol=2e-11 rtol=2e-11
            # Both parameters are unconstrained Normal scalars; the semantic
            # transport is therefore a permutation with no Jacobian terms.
            @test brm_rk_point_to_stan(transport, rg) ≈ sg atol=2e-10 rtol=2e-10
        end
    end
end
