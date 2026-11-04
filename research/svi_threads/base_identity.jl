# Writes every number of short `brm_svi_fit` runs to a file, raw bytes, so `cmp`
# can compare two commits, or two thread counts, to the last bit: the ELBO trace,
# the variational factors, Adam's state, the optimizer's random stream and the draws.
# A run on another seed must differ, or the comparison shows nothing.
#
# RUN: julia --project=<checkout>/test --threads=4 research/svi_threads/base_identity.jl \
#          <out.bin> <seed> [T]
# with no T for a commit that has no `threads` option.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
using LinearAlgebra, Random

flat_data(rng, N, n) = let g = repeat(1:N; inner=n), x = randn(rng, N * n),
        a = randn(rng, N), b = randn(rng, N)
    (; subject=repeat(["s$i" for i in 1:N]; inner=n), x,
       y=0.5 .+ 0.3 .* a[g] .+ (0.2 .+ 0.1 .* b[g]) .* x .+ 0.5 .* randn(rng, N * n))
end
flat_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | g | subject)
    y ~ Normal(mu, sigma)
end

plate_data(rng, N, n) = (; subject=["s$i" for i in 1:N], t=[collect(1.0:n) for _ in 1:N],
    y=[0.5 .+ cumsum(0.2 .* randn(rng, n)) .+ 0.3 .* randn(rng, n) for _ in 1:N])
plate_model(data) = @brm data begin
    sigma ~ Exponential(1)
    a ~ 1 + (1 | g | subject)
    b ~ 1 + (1 | g | subject)
    @plate for i in eachindex(a)
        y[i] ~ Normal(a[i] .+ b[i] .* log.(t[i]), sigma)
    end
end

# Every number reachable from `x`, in field order; refuses a type it cannot walk,
# so nothing is left out of the comparison unseen.
numbers!(io, x::AbstractArray{<:Union{Integer,AbstractFloat}}) = write(io, Array(x))
numbers!(io, x::Union{Integer,AbstractFloat,Bool}) = write(io, x)
numbers!(io, ::Nothing) = nothing
numbers!(io, x::AbstractArray) = foreach(e -> numbers!(io, e), x)
numbers!(io, x::Union{Tuple,NamedTuple}) = foreach(e -> numbers!(io, e), x)
numbers!(io, x::Random.Xoshiro) = foreach(n -> write(io, getfield(x, n)), fieldnames(typeof(x)))
function numbers!(io, x)
    isstructtype(typeof(x)) && !(x isa Function) || error("cannot compare a $(typeof(x))")
    foreach(n -> numbers!(io, getfield(x, n)), fieldnames(typeof(x)))
end

function main(path, seed, T)
    BLAS.set_num_threads(1)
    threads = T === nothing ? (;) : (; threads=T)
    cases = (("flat", flat_model(flat_data(Xoshiro(11), 24, 6)), MeanFieldGlobal(), 3000),
        ("flat_lowrank", flat_model(flat_data(Xoshiro(12), 24, 6)), LowRankGlobal(2), 3000),
        ("plate", plate_model(plate_data(Xoshiro(13), 24, 12)), MeanFieldGlobal(), 3000))
    open(path, "w") do io
        for (name, builder, family, iterations) in cases
            # Enough iterations to cross into polishing, whose average is also written.
            m = brm_nlme_model(RKBRMI(builder); ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            fit = brm_svi_fit(m; rng=Xoshiro(seed), batch_size=8, family, window=100,
                min_windows=5, patience=2, polish_windows=3, max_iterations=iterations,
                ndraws=50, threads...)
            d, run = fit.diagnostics, fit.diagnostics.optimizer.run
            numbers!(io, (; d.elbo, d.elbo_windows, d.iterations, d.converged, d.q, fit.draws,
                run.adam, run.rng, run.average, run.polish_start))
            println(name, ": iterations=", d.iterations, " converged=", d.converged,
                " polish_start=", run.polish_start, " tasks=", get(d, :threads, "n/a"),
                " last_elbo=", last(d.elbo), " bytes_so_far=", position(io))
            flush(stdout)
        end
    end
end

main(ARGS[1], parse(Int, ARGS[2]), length(ARGS) > 2 ? parse(Int, ARGS[3]) : nothing)
