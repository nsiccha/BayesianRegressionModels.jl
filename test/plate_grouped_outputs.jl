# Grouped reads of every indexed output from another plate, including a
# reader written before its producer. Public synthetic model and oracle.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

function grouped_outputs(data; swap_outputs=false, reader_first=false)
    outputs = [:(location[i] = shift + x[i]),
        :(variance[i] = scale^2 * (0.5 + x[i]^2))]
    swap_outputs && reverse!(outputs)
    producer = quote
        @plate for i in eachindex(x)
            $(outputs...)
        end
    end
    reader = quote
        @plate for k in eachindex(y)
            y[k] ~ Normal(sum(location[rows[k]]), sqrt(1 + sum(variance[rows[k]])))
        end
    end
    cells = reader_first ? (reader, producer) : (producer, reader)
    brm(quote
        shift ~ Normal(0.2, 0.7)
        scale ~ LogNormal(-0.1, 0.3)
        rows = groups(group => labels)
        $(cells...)
    end; mod=@__MODULE__)(data)
end

const GROUPED_OUTPUT_DATA = (; x=[-0.5, 1.2, 0.8, -0.1],
    group=["b", "a", "b", "a"], labels=["a", "empty", "b"], y=[0.7, -0.2, 1.1])

function grouped_output_oracle(data, shift, logscale)
    scale = exp(logscale)
    result = logpdf(Normal(0.2, 0.7), shift) +
        logpdf(Normal(-0.1, 0.3), logscale)
    for (label, y) in zip(data.labels, data.y)
        indices = findall(==(label), data.group)
        location = sum((shift + data.x[i] for i in indices); init=0.0)
        variance = 1 + sum((scale^2 * (0.5 + data.x[i]^2) for i in indices); init=0.0)
        result += logpdf(Normal(location, sqrt(variance)), y)
    end
    result
end

@stestset "grouped plate outputs: construction on both backends" begin
    target = grouped_outputs(GROUPED_OUTPUT_DATA)
    @test SBBRMI(target; mod=@__MODULE__) isa SBBRMI
    @test RKBRMI(target) isa RKBRMI
end

@stestset "grouped plate outputs: bare cell locals stay local" begin
    # A bare scratch binding is not an indexed output. Its name must not
    # become a runtime gather just because another plate mentions it.
    target = @brm GROUPED_OUTPUT_DATA begin
        rows = groups(group => labels)
        @plate for i in eachindex(x)
            scratch = x[i] + 1
            result[i] = 2 * x[i]
        end
        @plate for k in eachindex(y)
            y[k] ~ Normal(sum(scratch[rows[k]]), 1)
        end
    end
    for backend in (SBBRMI, RKBRMI)
        err = try backend(target); nothing catch e; e end
        @test err isa ErrorException
        @test occursin("scratch", sprint(showerror, err))
        @test occursin("MissingColumn", sprint(showerror, err))
    end
end

function check_grouped_output_density(sb, rk, data, tag)
    stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(), tag * ".stan"))
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    point(shift, logscale) = brm_start(rk, (; shift, scale=exp(logscale)))
    for (shift, logscale) in ((0.0, 0.0), (0.4, -0.3), (-0.6, 0.2))
        u = point(shift, logscale)
        us = brm_rk_point_to_stan(transport, u)
        gs = similar(us)
        ls, _ = BridgeStan.log_density_gradient!(stan.model, us, gs;
            propto=false, jacobian=true)
        lr, gr = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test ls ≈ grouped_output_oracle(data, shift, logscale) atol=2e-10
        @test lr ≈ ls atol=2e-10
        @test gr ≈ brm_stan_gradient_to_rk(transport, u, gs) atol=2e-9 rtol=2e-9
        h = 1e-5
        for (ds, dl) in ((h, 0.0), (0.0, h))
            direction = (point(shift + ds, logscale + dl) -
                point(shift - ds, logscale - dl)) / (2h)
            expected = (grouped_output_oracle(data, shift + ds, logscale + dl) -
                grouped_output_oracle(data, shift - ds, logscale - dl)) / (2h)
            @test sum(gr .* direction) ≈ expected atol=2e-7 rtol=2e-7
        end
    end
end

@stestset "grouped plate outputs: density, gradients and statement order" begin
    for swap_outputs in (false, true), reader_first in (false, true)
        target = grouped_outputs(GROUPED_OUTPUT_DATA; swap_outputs, reader_first)
        sb = SBBRMI(target; mod=@__MODULE__)
        rk = check_rk_source_roundtrip(RKBRMI(target))
        check_grouped_output_density(sb, rk, GROUPED_OUTPUT_DATA,
            "plate_grouped_outputs_$(swap_outputs)_$(reader_first)")
    end
end

@stestset "grouped plate outputs: replay onto changed groups and row counts" begin
    target = grouped_outputs(GROUPED_OUTPUT_DATA)
    sb, rk = SBBRMI(target; mod=@__MODULE__), RKBRMI(target)
    more = (; x=[0.2, -0.3, 0.9, 1.1, -0.7], group=["c", "a", "c", "b", "a"],
        labels=["b", "c", "a", "empty"], y=[0.5, 1.3, -0.4, 0.1])
    replayed_sb, replayed_rk = reprocess(sb, more), reprocess(rk, more)
    @test stan_code(replayed_sb) == stan_code(sb)
    check_grouped_output_density(replayed_sb, replayed_rk, more, "plate_grouped_outputs_replay")
end
