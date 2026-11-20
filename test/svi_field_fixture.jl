# Stations at fixed positions read the field at several of a shared set of times.
function station_data(; stations=6, times=[0.0, 1.0, 2.0, 4.0], reads=3, rows=2, seed=1, channels=nothing)
    rng = Xoshiro(seed)
    px, py = rand(rng, stations), rand(rng, stations)
    east, north, time, site = Float64[], Float64[], Float64[], String[]
    for s in 1:stations, t in sort!(randperm(rng, length(times))[1:reads]), _ in 1:rows
        push!(east, px[s]); push!(north, py[s]); push!(time, times[t]); push!(site, "s$s")
    end
    n = length(east)
    x = randn(rng, n)
    y = sin.(3 .* east) .+ cos.(time) .+ 0.2 .* x .+ 0.3 .* randn(rng, n)
    (; east, north, time, site, x, y, state=collect(1:n), nb=3)
end

function field_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        field ~ 0 + gmrf((east, north), time; neighbors=nb)
        sd(field, gmrf((east, north), time)) ~ Exponential(1)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(0.3), 0.7)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(3.0), 0.7)
        mu ~ 1 + (0 + x | site)
        @plate for s in eachindex(state)
            y[s] ~ Normal(mu[s] .+ field[state[s]], sigma)
        end
    end
end
