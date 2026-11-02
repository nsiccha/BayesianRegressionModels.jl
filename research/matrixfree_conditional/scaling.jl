# Synthetic separable GMRF proxy: no consumer data or model is imported.
# RUN: julia --project=test research/matrixfree_conditional/scaling.jl NX NY HOURS [det]
# `det` also measures the fixed-probe determinant and its selected derivative.
using BayesianRegressionModels, LinearAlgebra, SparseArrays, Random, Printf
const BRM = BayesianRegressionModels
BLAS.set_num_threads(1)

function space_precision(nx, ny)
    n = nx * ny
    ii = Int[]; jj = Int[]; vv = Float64[]
    for y in 1:ny, x in 1:nx
        i = x + nx * (y - 1)
        for j in (x < nx ? i+1 : 0, y < ny ? i+nx : 0)
            j == 0 && continue
            append!(ii, (i,j,i,j)); append!(jj, (i,j,j,i)); append!(vv, (1,1,-1,-1))
        end
    end
    A = spdiagm(0 => ones(n)) + sparse(ii,jj,vv,n,n)
    A' * A
end
function proxy(nx, ny, nt; rho=0.9)
    qt = nt == 1 ? spdiagm(0 => [1.0]) :
        spdiagm(-1=>fill(-rho,nt-1), 0=>vcat(1.0,fill(1+rho^2,nt-2),1.0),
            1=>fill(-rho,nt-1)) / (1-rho^2)
    H = kron(qt, space_precision(nx, ny))
    H + spdiagm(0 => fill(0.1, size(H,1)))
end
function operator_problem(H; options=MatrixFreeLaplace())
    n = size(H,1)
    p = (; pattern=H, components=[collect(1:n)], component=ones(Int,n),
        wcols=Vector{Int}[], wλ=Float64[],
        algebra=(; options, large=[true], probes=[2.0 .* rand(Xoshiro(options.seed),Bool,n,options.probes) .- 1],
            references=Any[nothing],
            diagnostics=Dict{Symbol,Any}()))
    if options.preconditioner_memory !== nothing
        F = BRM._brm_operator_factor(p,H)
        p.algebra.references[1] = BRM._brm_fixed_reference(only(F.parts).H,
            options.preconditioner_memory;ridge=options.preconditioner_ridge)
    end
    p
end
function measure(nx,ny,nt; determinant=false, options=MatrixFreeLaplace())
    H = proxy(nx,ny,nt)
    n = size(H,1)
    preparation = @timed operator_problem(H; options)
    p = preparation.value
    println("preparation\t",(;n,seconds=preparation.time,bytes=preparation.bytes))
    b = randn(Xoshiro(72),n)
    F = BRM._brm_operator_factor(p,H)
    # Warm compilation, then keep preparation in the per-step measurement.
    x = F \ b
    relative_residual = norm(H*x-b)/norm(b)
    operator_step = @timed begin
        F = BRM._brm_operator_factor(p,H)
        F \ b
    end
    exact = cholesky(Symmetric(H))
    cholesky!(exact,Symmetric(H))
    exact_step = @timed begin
        cholesky!(exact,Symmetric(H))
        exact \ b
    end
    @printf("step\t%d\t%d\t%.9g\t%.9g\t%.9g\t%d\t%d\n", n,nnz(H),
        operator_step.time,exact_step.time,relative_residual,operator_step.bytes,Base.summarysize(F))
    if determinant
        logdet(BRM._brm_operator_factor(p,H)) # warm compilation, independent cache
        det = @timed logdet(F)
        @printf("det\t%d\t%.9g\t%.9g\t%.9g\t%d\t%d\n",n,det.time,det.value,
            logdet(exact),det.bytes,Base.summarysize(F))
        println("diagnostics\t",p.algebra.diagnostics)
    end
    nothing
end
function setting(name, default, type)
    args = filter(x -> startswith(x,name * "="),ARGS[4:end])
    isempty(args) ? default : parse(type,split(only(args),'=';limit=2)[2])
end
if abspath(PROGRAM_FILE) == @__FILE__
length(ARGS) >= 3 || error("pass NX NY HOURS [det]")
println("julia\t", VERSION)
println("step columns: n nnz operator_seconds exact_refactor_solve_seconds relative_residual operator_allocated_bytes retained_operator_bytes")
println("det columns: n seconds approximate_logdet exact_logdet allocated_bytes retained_operator_bytes")
options = MatrixFreeLaplace(; solve_maxiter=setting("iterations",512,Int),
    probes=setting("probes",8,Int), solve_tolerance=setting("tolerance",1e-10,Float64),
    logdet_step=setting("step",1.0,Float64), logdet_extent=setting("extent",32.0,Float64),
    preconditioner_memory="unpreconditioned" in ARGS ? nothing : setting("memory",32,Int),
    preconditioner_ridge=setting("ridge",0.01,Float64))
println("options\t",options)
measure(parse.(Int,ARGS[1:3])...; determinant="det" in ARGS, options)
end
