include("gmrf_plate_replay.jl")

@testset "Independent GMRF cells retain lexical sampled priors" begin
    input = merge(data, (; unit=[10, 10, 20, 20]))
    for local_priors in (false, true)
        priors = quote
            Omega ~ LKJCholesky(2, 2.0)
            hyper_scale ~ Exponential(0.7)
            hyper ~ Exponential(hyper_scale)
        end
        outside, inside = local_priors ? (Expr(:block), priors) : (priors, Expr(:block))
        source = quote
            $(outside.args...)
            @plate for p in groups(unit)
                $(inside.args...)
                field ~ 0 + gmrf((east[p], north[p]), time[p], channel[p];
                    mesh=(vertices=xy, cells=triangles), cor=Omega)
                sd(field, gmrf((east[p], north[p]), time[p], channel[p])) ~ Exponential(hyper)
                length_scale(field, gmrf((east[p], north[p]), time[p], channel[p]), space) = 0.4
                length_scale(field, gmrf((east[p], north[p]), time[p], channel[p]), time) = 1.7
                y[p] ~ Normal(field, 0.2)
            end
        end
        # An outer data matrix with the same name must not turn the local
        # correlation declaration into an observed matrix.
        rows = local_priors ? input : (; (k => v for (k, v) in pairs(input) if k !== :Omega)...)
        model = BRM.brm(source; mod=@__MODULE__)(rows)
        sb = SBBRMI(model; mod=@__MODULE__)
        fit = only(e.const_ for e in values(sb.preproc) if e.kind === :cell_formulas)
        @test fit.labels == [10, 20]
        @test haskey(fit.owned, :Omega_n) == local_priors
        for c in fit.cells
            state = only(e.const_.prepared.state for e in values(c.preproc) if e.kind === :gmrf)
            @test state.correlation.kind === :parameter
            @test state.correlation.name === :Omega
            @test state.correlation.K == 2
            @test state.layout.ncoordinates == 20
            @test BRM.getf(state.sigma_prior) === Exponential
            hyper = only(BRM.getargs(state.sigma_prior))
            @test BRM.name(hyper) === :hyper
            hyper_decl = BRM._named_op(hyper)
            @test BRM.getf(hyper_decl) === (~)
            hyper_prior = last(BRM.getargs(hyper_decl))
            @test BRM.getf(hyper_prior) === Exponential
            @test BRM.name(only(BRM.getargs(hyper_prior))) === :hyper_scale
            local_priors && @test c.data[:Omega_n] == 2
        end
        subset = (; (k => getproperty(input, k)[[4, 3]] for k in
            (:east, :north, :time, :channel, :y, :unit))...)
        replay = reprocess(sb, subset)
        fresh = only(e.const_ for e in values(replay.preproc) if e.kind === :cell_formulas)
        @test fresh.labels == [10, 20]
        for (i, c) in enumerate(fresh.cells)
            state = only(e.const_.prepared.state for e in values(c.preproc) if e.kind === :gmrf)
            @test state.correlation.kind === :parameter
            @test state.layout.ncoordinates == 20
            @test size(state.indices, 1) == (i == 1 ? 0 : 2)
            local_priors && @test c.data[:Omega_n] == 2
        end
    end
end
