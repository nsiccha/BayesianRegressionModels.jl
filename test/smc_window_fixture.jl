module SMCWindowFixture
using BayesianRegressionModels, Random, Serialization

normal(data) = @brm data begin
    mu ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end
scaled(data) = @brm data begin
    mu ~ 0 + zscale(x)
    effect(mu, :) ~ Normal(0, 1)
    y ~ Normal(mu, 0.7)
end
changed(data) = @brm data begin
    mu ~ Normal(1, 1)
    y ~ Normal(mu, 1)
end
weighted_mean(s) = vec(s.positions*s.weights)
weighted_variance(s) = vec((s.positions .- weighted_mean(s)).^2*s.weights)
function roundtrip(x)
    io = IOBuffer()
    serialize(io, x)
    seekstart(io)
    deserialize(io)
end
function serialized_size(x)
    io = IOBuffer()
    serialize(io, x)
    position(io)
end

# Semantic equality includes every nested checkpoint field, including the
# continuation RNG; Serialization bytes can contain isbits padding.
function equal(a, b)
    typeof(a) == typeof(b) || return false
    if a isa Random.Xoshiro
        return a == b && rand(copy(a), 16) == rand(copy(b), 16)
    elseif a isa AbstractArray
        return axes(a) == axes(b) && all(equal(a[i], b[i]) for i in eachindex(a))
    elseif a isa AbstractDict
        return keys(a) == keys(b) && all(equal(a[k], b[k]) for k in keys(a))
    elseif a isa Expr || a isa QuoteNode || a isa GlobalRef || a isa LineNumberNode
        return isequal(a, b)
    elseif a isa Type || a isa Module || a isa Number || a isa AbstractString ||
            a isa Symbol || a === nothing
        return isequal(a, b)
    elseif isstructtype(typeof(a)) && fieldcount(typeof(a)) > 0
        return all(equal(getfield(a, i), getfield(b, i)) for i in 1:fieldcount(typeof(a)))
    end
    isequal(a, b)
end
end
