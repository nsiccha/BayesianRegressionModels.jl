# Shared RK cell-field models and dense oracles for the native GMRF tests.
include("gmrf_plate_models.jl")
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
using SpecialFunctions: logbeta
import DifferentiationInterface as DI

function gmrf_rk_cell_model(; sampled_correlation)
    input = merge(data, (; unit=[10, 10, 20, 20]))
    if sampled_correlation
        input = (; (k => v for (k, v) in pairs(input) if k !== :Omega)...)
        return @brm input begin
            hyper_scale ~ Exponential(0.7)
            Omega ~ LKJCholesky(2, 2.0)
            @plate for p in groups(unit)
                hyper ~ Exponential(hyper_scale)
                field ~ 0 + gmrf((east[p], north[p]), time[p], channel[p];
                    mesh=(vertices=xy, cells=triangles), cor=Omega)
                sd(field, gmrf((east[p], north[p]), time[p], channel[p])) ~ Exponential(hyper)
                length_scale(field, gmrf((east[p], north[p]), time[p], channel[p]), space) = 0.4
                length_scale(field, gmrf((east[p], north[p]), time[p], channel[p]), time) ~ LogNormal(0.7, 0.6)
                y[p] ~ Normal(field, 0.2)
            end
        end
    end
    @brm input begin
        hyper_scale ~ Exponential(0.7)
        @plate for p in groups(unit)
            hyper ~ Exponential(hyper_scale)
            field ~ 0 + gmrf((east[p], north[p]), time[p], channel[p];
                mesh=(vertices=xy, cells=triangles), cor=Omega)
            sd(field, gmrf((east[p], north[p]), time[p], channel[p])) ~ Exponential(hyper)
            length_scale(field, gmrf((east[p], north[p]), time[p], channel[p]), space) = 0.4
            length_scale(field, gmrf((east[p], north[p]), time[p], channel[p]), time) ~ LogNormal(0.7, 0.6)
            y[p] ~ Normal(field, 0.2)
        end
    end
end

# The dense prior of one cell's field: state order vertex, channel, time.
function gmrf_cell_field_logpdf(z, ell_s, ell_t, L, times)
    mesh = BRM._brm_gmrf_mesh(data.xy, data.triangles)
    kappa2 = 2 / ell_s^2
    K = Matrix(mesh.stiffness) + kappa2 * Diagonal(mesh.mass)
    Q = K * Diagonal(1 ./ mesh.mass) * K / (4pi * kappa2)
    Ct = [exp(-abs(t - s) / ell_t) for t in times, s in times]
    logpdf(MvNormal(zeros(length(z)), Symmetric(kron(Ct, kron(L * L', inv(Symmetric(Q)))))), z)
end

# One cell's observation term: three-vertex reads of its own support.
function gmrf_cell_rows_logpdf(z, sigma, input, rows, channels)
    times = sort!(unique(input.time[rows]))
    support = BRM.brm_gmrf_support(data.xy, data.triangles, times,
        channels === nothing ? ones(Int, length(rows)) : input.channel[rows])
    reads = channels === nothing ?
        BRM.brm_gmrf_rows(support, input.east[rows], input.north[rows], input.time[rows]) :
        BRM.brm_gmrf_rows(support, input.east[rows], input.north[rows], input.time[rows],
            input.channel[rows])
    mu = sigma .* (reads.weight1 .* z[reads.index1] .+ reads.weight2 .* z[reads.index2] .+
        reads.weight3 .* z[reads.index3])
    sum(logpdf.(Normal.(mu, 0.2), input.y[rows]); init=0.0), times
end

coordinate(names, name) = findfirst(==(Symbol(name)), names)
cell_coordinates(names, declaration, cell) =
    [i for (i, n) in enumerate(names) if startswith(string(n), "$declaration.field.$cell.")]

function gmrf_rk_build(model)
    artifact = BRM.emit_rk_artifact(model; case_id="gmrf-rk-cells")
    bound = BRM.rk_translate_artifact(artifact)
    built = build_kernel(bound)
    rk = RKBRMI(model, artifact.plan, built)
    rk, bound, built
end

function gmrf_rk_gradient_check(built, bound, oracle, u)
    sampler = prepare_sampler(built, bound, u; backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ oracle(u) atol=2e-8 rtol=2e-10
    for j in eachindex(u)
        plus, minus = copy(u), copy(u)
        h = 2e-6
        plus[j] += h
        minus[j] -= h
        @test gradient[j] ≈ (oracle(plus) - oracle(minus)) / (2h) atol=3e-5 rtol=3e-6
    end
end
