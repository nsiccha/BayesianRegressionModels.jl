# RUN: julia --project=test test/laplace_mode_overflow.jl
include(joinpath(@__DIR__, "fitting_packages.jl"))
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, SparseArrays, LogDensityProblems

# At the first Newton trial the inner exp overflows, while the mean and
# its first two derivatives are smaller than Float64 can represent.
function overflow_fixture(sumcell)
    a = exp(-10.0)
    sigma = sqrt(a * (1 - 2a) * exp(-2a) / 100.9)
    data = (; n=2, idx=[1,2], rows=[[1],[2]], baseline=[-10.0,0.0],
        y=[0.0,0.4], noise=[sigma,0.2])
    model = if sumcell
        @brm data begin
            shift ~ Normal(0,1)
            field[1:n] .~ Normal.(0,1)
            @plate for k in eachindex(rows)
                y[k] ~ Normal(exp(-exp(shift + baseline[k] + sum(field[rows[k]]))), noise[k])
            end
        end
    else
        @brm data begin
            shift ~ Normal(0,1)
            field[1:n] .~ Normal.(0,1)
            prediction = exp(-exp(shift + baseline + field[idx]))
            y ~ Normal(prediction,noise)
        end
    end
    (; rk=RKBRMI(model), data)
end

# Near x = 0 the mean is 2sqrt(1-x) - x. With y = 0, unit noise and a
# standard normal prior, the score and precision are both 4. The Newton
# trial x = 1 improves the log density from -2 to -1 (up to constants).
# There the mean is -1. Its slope of -8/min_denom - 1 makes both the
# log-density gradient and curvature overflow in Float64.
function steep_trial_fixture(sumcell)
    data = (; n=2, idx=[1,2], rows=[[1],[2]], y=[0.0,1.75], min_denom=floatmin(Float64))
    model = if sumcell
        @brm data begin
            shift ~ Normal(0,1)
            field[1:n] .~ Normal.(0,1)
            @plate for k in eachindex(rows)
                x = shift + sum(field[rows[k]])
                t = 1 - x
                y[k] ~ Normal(8 * t / max(4 * sqrt(abs(t)), min_denom) - x, 1)
            end
        end
    else
        @brm data begin
            shift ~ Normal(0,1)
            field[1:n] .~ Normal.(0,1)
            @plate for k in eachindex(idx)
                x = shift + field[idx[k]]
                t = 1 - x
                y[k] ~ Normal(8 * t / max(4 * sqrt(abs(t)), min_denom) - x, 1)
            end
        end
    end
    RKBRMI(model)
end

function overflow_problem(rk, solver)
    brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:field,), solver,
        matrixfree=MatrixFreeLaplace(; probes=2, preconditioner_memory=nothing), maxiter=200)
end

function overflow_assemble(p, pop, z)
    fc = zeros(length(p.components)); g = similar(z); h = BRM._brm_spl_precision(p)
    BRM._brm_spl_assemble!(fc,g,h,p,pop.A,pop.C,z; derivatives=true,uA=pop.uA)
    (; fc,g,h)
end
overflow_dense(h::SparseMatrixCSC) = Matrix(h)
overflow_dense(h::BRM._BRMSumPrecision) = Matrix(h.D + h.B' * h.W * h.B)

@stestset "sparse Newton steps reject a finite density with overflowing derivatives" begin
    for sumcell in (false,true)
        rk = steep_trial_fixture(sumcell)
        full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
        origin = zeros(LogDensityProblems.dimension(full))
        value, gradient = LogDensityProblems.logdensity_and_gradient(full,origin)
        @test isfinite(value) && all(isfinite,gradient)
        @testset "sumcell=$sumcell solver=$solver" for solver in (:exact,:matrixfree)
            p = overflow_problem(rk,solver)
            @test (p.algebra === nothing) == (solver === :exact)
            @test (only(p.cells) isa BRM._BRMSumCellGroup) == sumcell
            @test length(p.components) == length(p.sblocks) == 2
            phi = zeros(length(p.population))
            pop = BRM._brm_spl_population(p,phi)
            z = zeros(length(p.integrated))
            initial = overflow_assemble(p,pop,z)
            delta = BRM._brm_spl_direction(p,initial.h,initial.g)
            @test initial.g[1] == overflow_dense(initial.h)[1,1] == 4
            @test delta[1] == 1
            @test 0 < delta[2] < 1
            trial = overflow_assemble(p,pop,delta)
            @test all(isfinite,trial.fc)
            @test all(trial.fc .>= initial.fc .+ 1e-4 .* initial.g .* delta)
            @test !isfinite(trial.g[1]) && isfinite(trial.g[2])
            @test !BRM._brm_spl_finite(trial.h)
            @test BRM._brm_spl_finite_blocks(p,trial.g,trial.h) == [false,true]

            search = BRM._brm_spl_search(2, (1.0,))
            @test BRM._brm_spl_step!(p,pop,z,delta,initial.g,initial.h,initial.fc,falses(2); search)
            @test search.backtracks == [1,0]
            @test z == [0.5,delta[2]]
            accepted = overflow_assemble(p,pop,z)
            @test all(isfinite,accepted.g) && BRM._brm_spl_finite(accepted.h)
            @test initial.fc ≈ accepted.fc
            @test initial.g ≈ accepted.g
            @test overflow_dense(initial.h) ≈ overflow_dense(accepted.h)

            if solver === :exact
                @test p.blockwise == 1
                @test p.sweeps == 0
                sweep = zeros(length(z))
                @test BRM._brm_spl_sweep!(p,pop,sweep,p.component[1])
                @test sweep == [0.5,0.0]
                swept = overflow_assemble(p,pop,sweep)
                @test all(isfinite,swept.g) && BRM._brm_spl_finite(swept.h)
                # A bad curvature entry alone invalidates just its own block.
                for field in (sumcell ? (:D,:B,:W) : (:sparse,))
                    h = deepcopy(accepted.h)
                    buffer = field === :sparse ? h : getproperty(h,field)
                    nonzeros(buffer)[1] = NaN
                    @test BRM._brm_spl_finite_blocks(p,zeros(length(z)),h) == [false,true]
                end
            end

            result = LogDensityProblems.logdensity_and_gradient(p,phi)
            @test isfinite(result[1]) && all(isfinite,result[2])
            state = BRM._brm_spl_mode!(p,phi)
            @test state.finite
            joint, jointgradient = LogDensityProblems.logdensity_and_gradient(full,state.u)
            @test norm(jointgradient[p.integrated],Inf) < 1e-5
            @test result[1] ≈ joint + length(z)/2*log(2pi) -
                logdet(Symmetric(overflow_dense(state.h)))/2 atol=1e-8
            eps = 1e-4
            fd = (LogDensityProblems.logdensity(p,phi .+ eps) -
                  LogDensityProblems.logdensity(p,phi .- eps))/(2eps)
            @test only(result[2]) ≈ fd rtol=2e-5 atol=2e-5
            LogDensityProblems.logdensity_and_gradient(p,phi .+ 0.2)
            @test LogDensityProblems.logdensity_and_gradient(p,phi) == result
            @test p.nonfinite == p.unconverged == 0

            # At this shift the starting derivatives already overflow.
            @test LogDensityProblems.logdensity(p,[1.0]) == -Inf
            @test p.nonfinite == 1
        end
    end
end

@stestset "saturated exponential at a sparse Newton trial" begin
    for sumcell in (false,true)
        (; rk, data) = overflow_fixture(sumcell)
        @testset "sumcell=$sumcell solver=$solver" for solver in (:exact,:matrixfree)
            p = overflow_problem(rk,solver)
            @test (only(p.cells) isa BRM._BRMSumCellGroup) == sumcell
            pop = BRM._brm_spl_population(p,[0.0])
            z = zeros(length(p.integrated))
            initial = overflow_assemble(p,pop,z)
            delta = BRM._brm_spl_direction(p,initial.h,initial.g)
            @test delta[1] > 1000
            trial = overflow_assemble(p,pop,delta)
            @test all(isfinite,trial.fc)
            @test all(trial.fc .> initial.fc)
            finite = all(isfinite,trial.g) && BRM._brm_spl_finite(trial.h)
            # Ordinary cells differentiate the saturated mean to zero. A sum
            # cell can still encounter the indeterminate product 0 * Inf.
            @test finite || sumcell
            search = BRM._brm_spl_search(2, (1.0,))
            @test BRM._brm_spl_step!(p,pop,z,delta,initial.g,initial.h,initial.fc,falses(2); search)
            if finite
                # In the saturated cell only the standard normal prior contributes.
                @test trial.g[1] == -delta[1]
                @test overflow_dense(trial.h)[1,1] == 1
                @test search.backtracks == [0,0]
                @test z == delta
                value = LogDensityProblems.logdensity(p,[1000.0])
                expected = logpdf(Normal(),1000.0) + sum(logpdf.(Normal.(0,data.noise),data.y))
                @test value ≈ expected atol=1e-8 rtol=0
                @test p.nonfinite == 0
            else
                @test search.backtracks == [1,0]
                @test 0 < z[1] < delta[1]
                @test z[2] == delta[2]
            end
        end
    end
end
