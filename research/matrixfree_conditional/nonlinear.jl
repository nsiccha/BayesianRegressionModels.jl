using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
using LogDensityProblems, LinearAlgebra, Random, Printf
const BRM = BayesianRegressionModels
BLAS.set_num_threads(1)

function lattice_data(nx, ny, nt)
    n = nx * ny * nt
    from = Int[]; to = Int[]
    for t in 1:nt, y in 1:ny, x in 1:nx
        i = x + nx * (y - 1) + nx * ny * (t - 1)
        for j in (x < nx ? i+1 : 0, y < ny ? i+nx : 0, t < nt ? i+nx*ny : 0)
            j == 0 && continue
            push!(from,i); push!(to,j)
        end
    end
    # Fixed varied counts make the conditional curvature depend on the mode.
    (; area=collect(1:n), x=sin.(collect(1:n) ./ 17),
        y=[mod(i,5) for i in 1:n], from, to)
end

function lattice_problem(data; solver=:matrixfree)
    rk = RKBRMI(@brm data begin
        log(mu) ~ 1 + x + car(area; edges=(from,to), type=:icar)
        y ~ Poisson(mu)
    end)
    brm_laplace_problem(rk; integrate=:area,
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), solver)
end

function measure_nonlinear(nx, ny, nt; exact=false, finite_difference=false, solver=:matrixfree)
    data = lattice_data(nx,ny,nt)
    setup = @timed lattice_problem(data; solver)
    p = setup.value
    x = fill(0.1,length(p.population))
    cold = @timed LogDensityProblems.logdensity_and_gradient(p,x)
    warm = @timed LogDensityProblems.logdensity_and_gradient(p,x)
    value, gradient = warm.value
    operator = p.algebra !== nothing
    println("dimensions\t",length(p.integrated),'\t',length(x),'\t',length(data.y))
    println("population\t",p.names)
    println("solver\t",solver,"\toperator=",operator)
    @printf("setup\t%.9g\t%d\n",setup.time,setup.bytes)
    @printf("cold_value_gradient\t%.9g\t%d\n",cold.time,cold.bytes)
    @printf("warm_value_gradient\t%.9g\t%d\t%.12g\t%.12g\n",
        warm.time,warm.bytes,value,norm(gradient))
    operator && println("diagnostics\t",p.algebra.diagnostics)
    println("mode_failures\t",p.nonfinite,'\t',p.unconverged)
    @assert isfinite(value) && all(isfinite,gradient)
    @assert cold.value == warm.value
    operator && @assert p.factor[] === nothing
    if finite_difference
        h = 1e-4
        fd = map(eachindex(x)) do j
            d=zeros(length(x)); d[j]=h
            (LogDensityProblems.logdensity(p,x+d)-LogDensityProblems.logdensity(p,x-d))/(2h)
        end
        println("gradient\t",gradient)
        println("finite_difference\t",fd)
        println("fd_relative_error\t",norm(gradient-fd)/max(1.0,norm(fd)))
        @assert gradient ≈ fd rtol=2e-4 atol=2e-4
    end
    if exact
        q = deepcopy(p)
        BRM._brm_spl_algebra!(q,:exact,MatrixFreeLaplace())
        LogDensityProblems.logdensity_and_gradient(q,x)
        reference = @timed LogDensityProblems.logdensity_and_gradient(q,x)
        ve,ge = reference.value
        @printf("exact_warm_value_gradient\t%.9g\t%d\t%.12g\t%.12g\n",
            reference.time,reference.bytes,ve,norm(ge))
        println("approximation_error\t",value-ve,'\t',norm(gradient-ge))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS)>=3 || error("pass NX NY NT [exact] [fd] [auto]")
    println("julia\t",VERSION)
    measure_nonlinear(parse.(Int,ARGS[1:3])...; exact="exact" in ARGS,
        finite_difference="fd" in ARGS, solver="auto" in ARGS ? :auto : :matrixfree)
end
