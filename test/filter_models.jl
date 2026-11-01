module FilterModels
using BayesianRegressionModels, ReactiveKernels
const BRM = BayesianRegressionModels

function accumulated_input(x, gain)
    carry = 0.0
    map(x) do value
        carry = 0.7carry + gain * value
    end
end
function BRM._rk_callable_source!(definitions, bindings, entry, ::typeof(accumulated_input))
    push!(definitions, :(ReactiveKernels.@kernel $entry(x, gain) = begin
        values = ReactiveKernels.scan(x; init=0.0) do carry, value
            next = 0.7carry + gain * value
            (next, next)
        end
        return values
    end))
    :done
end
provider(d) = @brm d begin
    gain ~ Normal(0, 1)
    loc = accumulated_input(x, gain)
    y ~ Normal(loc, 1)
end
ar_model(d) = @brm d begin
    mu ~ 1 + ar(t; p=1)
    y ~ Normal(mu, 1)
end
scaled(d) = @brm d begin
    mu ~ 1 + zscale(x)
    y ~ Normal(mu, 1)
end
observed_groups(d) = @brm d begin
    @plate for i in eachindex(y)
        @scan begin
            y[i][1] ~ Normal(0, 1)
            for t in 2:T
                y[i][t] ~ Normal(0.5y[i][t-1], 1)
            end
        end
    end
end
end
