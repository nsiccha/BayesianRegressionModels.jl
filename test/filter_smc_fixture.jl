module FilterSMCFixture
using BayesianRegressionModels, Random, Serialization, ReactiveKernelsPPL

linear(data) = @brm data begin
    theta ~ Normal(0, 1)
    @plate for i in eachindex(y)
        @scan begin
            state[1] ~ Normal(0, 1)
            for t in 2:T
                state[t] ~ Normal(0.6state[t-1], 1)
            end
        end
        y[i] ~ Normal(theta .+ state, 1)
    end
end

changed(data) = @brm data begin
    theta ~ Normal(1, 1)
    @plate for i in eachindex(y)
        @scan begin
            state[1] ~ Normal(0, 1)
            for t in 2:T
                state[t] ~ Normal(0.6state[t-1], 1)
            end
        end
        y[i] ~ Normal(theta .+ state, 1)
    end
end

stub(; groups=1) = linear((; y=[zeros(1) for _ in 1:groups], T=1))
records(ys) = [(; observation=(; y=[y])) for y in ys]
options() = SMCOptions(; moves=1, stepsize=0.3, max_tree_depth=3, history_limit=4)
function calibration(; particles=256)
    data = (; y=[[0.2, 0.8, 0.5]], T=3)
    fit = brm_smc_fit(linear(data); rng=Xoshiro(184), particles,
        options=SMCOptions(; moves=3, stepsize=0.3, max_tree_depth=4, history_limit=4))
    fit, data
end
end
