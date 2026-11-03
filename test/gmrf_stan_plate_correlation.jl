include("gmrf_plate_models.jl")
import BridgeStan as BS
using SpecialFunctions: logbeta

function gmrf_stan_plate_correlations(; shadow=false)
@testset "Compiled GMRF cells preserve shared and local channel correlations" begin
    input = merge(data, (; unit=[10, 10, 20, 20]))
    for local_correlation in (false, true)
        prior = quote
            Omega ~ LKJCholesky(2, 2.0)
        end
        outside, inside = local_correlation ? (Expr(:block), prior) : (prior, Expr(:block))
        source = quote
            $(outside.args...)
            @plate for p in groups(unit)
                $(inside.args...)
                field ~ 0 + gmrf((east[p], north[p]), time[p], channel[p];
                    mesh=(vertices=xy, cells=triangles), cor=Omega)
                sd(field, gmrf((east[p], north[p]), time[p], channel[p])) ~ Exponential(0.7)
                length_scale(field, gmrf((east[p], north[p]), time[p], channel[p]), space) = 0.4
                length_scale(field, gmrf((east[p], north[p]), time[p], channel[p]), time) = 1.7
                y[p] ~ Normal(field, 0.2)
            end
        end
        rows = local_correlation && shadow ? input :
            (; (k => v for (k, v) in pairs(input) if k !== :Omega)...)
        sb = SBBRMI(BRM.brm(source; mod=@__MODULE__)(rows); mod=@__MODULE__)
        descriptor = brm_descriptor(sb; highlights=())
        problem = BRM.stan_instantiate(sb)
        names = BS.param_unc_names(problem.model)
        field = brm_output_coordinates(only(brm_outputs(descriptor;
            predictor=:field, component=:field, kind=:parameter)), names)
        scales = brm_output_coordinates(only(brm_outputs(descriptor;
            predictor=:field, component=:sd, kind=:parameter)), names)
        correlation = setdiff(1:length(names), vcat(field, scales))
        @test length(field) == 40
        @test length(scales) == 2
        @test length(correlation) == (local_correlation ? 2 : 1)
        point = zeros(length(names))
        point[field] = 0.03 .* sin.(1:40)
        point[scales] .= log(0.9)
        point[correlation] .= atanh.(local_correlation ? [0.35, -0.2] : [0.35])
        fit = only(e.const_ for e in values(descriptor.plan.preproc) if e.kind === :cell_formulas)
        function oracle(u)
            rhos = tanh.(u[correlation])
            lp = sum(rho -> 2log1p(-rho^2) - logbeta(0.5, 2.0), rhos)
            for (i, cell) in enumerate(fit.cells)
                state = only(e.const_.prepared.state for e in values(cell.preproc) if e.kind === :gmrf)
                rho = rhos[local_correlation ? i : 1]
                omega = [1.0 rho; rho 1.0]
                kappa2 = 2 / 0.4^2
                K = Matrix(state.mesh.stiffness) + kappa2 * Diagonal(state.mesh.mass)
                Q = K * Diagonal(1 ./ state.mesh.mass) * K / (4pi * kappa2)
                Ct = [exp(-abs(t-s)/1.7) for t in state.times, s in state.times]
                covariance = kron(Ct, kron(omega, inv(Symmetric(Q))))
                z = u[field[(20(i-1)+1):(20i)]]
                lp += logpdf(MvNormal(zeros(20), Symmetric(covariance)), z)
                sigma = exp(u[scales[i]])
                lp += logpdf(Exponential(0.7), sigma) + log(sigma)
                mu = sigma .* vec(sum(state.weights .* z[state.indices]; dims=2))
                lp += sum(logpdf.(Normal.(mu, 0.2), zeros(length(mu))))
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
        # A string channel is fitted support metadata, including when its
        # cell has no observations in a replay. It is never a Stan input.
        fresh = (; (k => getproperty(input, k)[[3]] for k in
            (:east, :north, :time, :channel, :y, :unit))...)
        replay = brm_descriptor(reprocess(sb, fresh); highlights=())
        replay_fit = only(e.const_ for e in values(replay.plan.preproc) if e.kind === :cell_formulas)
        states = [only(e.const_.prepared.state for e in values(c.preproc) if e.kind === :gmrf)
            for c in replay_fit.cells]
        @test all(s.channels == ["a", "b"] for s in states)
        @test size(first(states).indices) == (0, 3)
        @test all(s.layout.ncoordinates == 20 for s in states)
        @test brm_output_coordinates(only(brm_outputs(replay;
            predictor=:field, component=:field, kind=:parameter)), names) == field
    end
end
end

gmrf_stan_plate_correlations()
