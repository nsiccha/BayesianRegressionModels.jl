# RUN: julia --project=test test/laplace_mode_overflow.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, SparseArrays, LogDensityProblems

# The nested exponential has a finite primal even when exp's derivatives
# overflow. This noise makes the first shifted Newton direction exceed 1000.
# The second independent cell needs a small, ordinary Newton step.
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
    RKBRMI(model)
end

function overflow_assemble(p, pop, z)
    fc = zeros(length(p.components)); g = similar(z); h = BRM._brm_spl_precision(p)
    BRM._brm_spl_assemble!(fc,g,h,p,pop.A,pop.C,z; derivatives=true,uA=pop.uA)
    (; fc,g,h)
end
overflow_dense(h::SparseMatrixCSC) = Matrix(h)
overflow_dense(h::BRM._BRMSumPrecision) = Matrix(h.D + h.B' * h.W * h.B)

@stestset "finite density with overflowing derivatives backtracks to a valid mode" begin
    for sumcell in (false,true)
        rk = overflow_fixture(sumcell)
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:field,), maxiter=200)
        @test length(p.components) == 2
        @test length(p.sblocks) == 2
        @test (only(p.cells) isa BRM._BRMSumCellGroup) == sumcell
        phi = zeros(length(p.population))
        full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
        value, gradient = LogDensityProblems.logdensity_and_gradient(full, zeros(length(p.u)))
        @test isfinite(value) && all(isfinite,gradient)

        pop = BRM._brm_spl_population(p,phi)
        z = zeros(length(p.integrated))
        initial = overflow_assemble(p,pop,z)
        delta = BRM._brm_spl_direction(p,initial.h,initial.g)
        @test delta[1] > 1000
        trial = overflow_assemble(p,pop,z+delta)
        @test all(isfinite,trial.fc)
        @test all(trial.fc .> initial.fc)
        @test !all(isfinite,trial.g)
        @test !BRM._brm_spl_finite(trial.h)

        # Full-step rejection, local backtracking, and reuse of the accepted
        # derivatives. The healthy component keeps its exact full step.
        search = BRM._brm_spl_search(2, (1.0,))
        @test BRM._brm_spl_step!(p,pop,z,delta,initial.g,initial.h,initial.fc,falses(2); search)
        @test search.backtracks == [1, 0]
        @test p.blockwise == 1
        @test p.sweeps == 0
        @test 0 < z[1] < delta[1]
        @test z[2] == delta[2]
        accepted = overflow_assemble(p,pop,z)
        @test all(isfinite,accepted.g) && BRM._brm_spl_finite(accepted.h)
        @test initial.fc ≈ accepted.fc
        @test initial.g ≈ accepted.g
        @test overflow_dense(initial.h) ≈ overflow_dense(accepted.h)

        # The sequential fallback must reject the same overflowing trial.
        sweep = zeros(length(z))
        @test BRM._brm_spl_sweep!(p,pop,sweep,p.component[1])
        swept = overflow_assemble(p,pop,sweep)
        @test 0 < sweep[1] < delta[1]
        @test all(isfinite,swept.g) && BRM._brm_spl_finite(swept.h)

        # A finite gradient alone is insufficient: bad curvature in any of
        # the sparse/augmented buffers must invalidate its own step block.
        for field in (sumcell ? (:D,:B,:W) : (:sparse,))
            h = deepcopy(accepted.h)
            buffer = field === :sparse ? h : getproperty(h,field)
            nonzeros(buffer)[1] = NaN
            @test BRM._brm_spl_finite_blocks(p,zeros(length(z)),h) == [false,true]
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
        @test p.nonfinite == 0
        @test p.unconverged == 0

        # An outer point whose initial derivatives overflow is still invalid;
        # the safeguard does not clip the model or manufacture a finite value.
        @test LogDensityProblems.logdensity(p,fill(1000.0,length(phi))) == -Inf
        @test p.nonfinite == 1
    end
end

@stestset "matrix-free conditional steps reject overflowing derivatives" begin
    for sumcell in (false,true)
        rk = overflow_fixture(sumcell)
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:field,),
            solver=:matrixfree, matrixfree=MatrixFreeLaplace(; probes=2,
                preconditioner_memory=nothing), maxiter=200)
        phi = zeros(length(p.population))
        pop = BRM._brm_spl_population(p,phi)
        z = zeros(length(p.integrated))
        initial = overflow_assemble(p,pop,z)
        delta = BRM._brm_spl_direction(p,initial.h,initial.g)
        @test delta[1] > 1000
        trial = overflow_assemble(p,pop,z+delta)
        @test all(isfinite,trial.fc) && !all(isfinite,trial.g)
        @test BRM._brm_spl_step!(p,pop,z,delta,initial.g,initial.h,initial.fc,falses(2))
        @test 0 < z[1] < delta[1]
        @test z[2] == delta[2]
        accepted = overflow_assemble(p,pop,z)
        @test all(isfinite,initial.g) && BRM._brm_spl_finite(initial.h)
        @test initial.fc ≈ accepted.fc
        @test initial.g ≈ accepted.g
        @test overflow_dense(initial.h) ≈ overflow_dense(accepted.h)
        result = LogDensityProblems.logdensity_and_gradient(p,phi)
        @test isfinite(result[1]) && all(isfinite,result[2])
        eps = 1e-4
        fd = (LogDensityProblems.logdensity(p,phi .+ eps) -
              LogDensityProblems.logdensity(p,phi .- eps))/(2eps)
        @test only(result[2]) ≈ fd rtol=2e-5 atol=2e-5
        LogDensityProblems.logdensity_and_gradient(p,phi .+ 0.2)
        @test LogDensityProblems.logdensity_and_gradient(p,phi) == result
        @test p.nonfinite == p.unconverged == 0
    end
end
