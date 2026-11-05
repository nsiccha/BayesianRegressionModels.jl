module SMCTuningFixture

using BayesianRegressionModels, Distributions
using Random
using WarmupHMC: MatrixFactorization, SuccessiveReflections

# A normalized Gaussian regression with a narrow sum direction and a wider
# difference direction. The oracle is scalar two-variable Gaussian algebra;
# it is never supplied to the sampler as a dense metric or Hessian.
model(data) = @brm data begin
    mu ~ 0 + x1 + x2
    effect(mu, :) ~ Normal(0, 1)
    y ~ Normal(mu, 0.1)
end

function data(; rows=32, separation=0.02)
    phase = 2pi .* (0:rows-1) ./ rows
    x, z = cos.(phase), sin.(phase)
    x1 = (x .+ separation .* z) ./ sqrt(2)
    x2 = (x .- separation .* z) ./ sqrt(2)
    y = 0.6 .* x .- 0.3separation .* z .+ 0.03 .* cos.(2 .* phase)
    (; x1, x2, y)
end

prefix(d, rows) = (; x1=d.x1[1:rows], x2=d.x2[1:rows], y=d.y[1:rows])

function posterior(d)
    precision = 100.0
    a = 1 + precision * sum(abs2, d.x1)
    b = precision * sum(d.x1 .* d.x2)
    c = 1 + precision * sum(abs2, d.x2)
    h1 = precision * sum(d.x1 .* d.y)
    h2 = precision * sum(d.x2 .* d.y)
    determinant = a*c - b*b
    m1, m2 = (c*h1 - b*h2)/determinant, (a*h2 - b*h1)/determinant
    means = [(m1 + m2)/sqrt(2), (m1 - m2)/sqrt(2)]
    variances = [(a + c - 2b)/(2determinant), (a + c + 2b)/(2determinant)]
    log_normalizer = -length(d.y)*log(2pi/precision)/2 - log(determinant)/2 -
        (precision*sum(abs2, d.y) - h1*m1 - h2*m2)/2
    (; mean=[m1, m2], means, variances, log_normalizer, a, b, c, determinant)
end

function logjoint(d, beta)
    residual = d.y .- beta[1] .* d.x1 .- beta[2] .* d.x2
    -log(2pi) - sum(abs2, beta)/2 - length(d.y)*log(2pi*0.01)/2 -
        sum(abs2, residual)/0.02
end

function logposterior(p, beta)
    u, v = beta .- p.mean
    -log(2pi) + log(p.determinant)/2 - (p.a*u*u + 2p.b*u*v + p.c*v*v)/2
end

function direction_summary(positions, weights, order)
    b1, b2 = positions[order[1], :], positions[order[2], :]
    directions = ((b1 .+ b2) ./ sqrt(2), (b1 .- b2) ./ sqrt(2))
    means = [sum(weights .* values) for values in directions]
    variances = [sum(weights .* (values .- mean).^2)
        for (values, mean) in zip(directions, means)]
    (; means, variances)
end

# Compare data values, not Serialization bytes (isbits records can have padding).
# This deliberately includes every schedule/history field and the continuation RNG.
function checkpoint_equal(a, b)
    typeof(a) == typeof(b) || return false
    if a isa Random.Xoshiro
        return a == b && rand(copy(a), 16) == rand(copy(b), 16)
    elseif a isa Union{MatrixFactorization,SuccessiveReflections}
        # These lazy operators store factors but do not support getindex.
        # Compare the complete stored state without materializing a matrix.
        return all(checkpoint_equal(getfield(a, i), getfield(b, i))
            for i in 1:fieldcount(typeof(a)))
    elseif a isa AbstractArray
        return axes(a) == axes(b) && all(checkpoint_equal(a[i], b[i]) for i in eachindex(a))
    elseif a isa AbstractDict
        return keys(a) == keys(b) && all(checkpoint_equal(a[k], b[k]) for k in keys(a))
    elseif a isa Expr || a isa QuoteNode || a isa GlobalRef || a isa LineNumberNode
        # Compare authored syntax with Julia's AST equality, rather than walking
        # runtime binding caches reachable from a GlobalRef.
        return isequal(a, b)
    elseif a isa Type || a isa Module || a isa Number ||
            a isa AbstractString || a isa Symbol || a === nothing
        return isequal(a, b)
    elseif isstructtype(typeof(a)) && fieldcount(typeof(a)) > 0
        return all(checkpoint_equal(getfield(a, i), getfield(b, i))
            for i in 1:fieldcount(typeof(a)))
    end
    isequal(a, b)
end

end
