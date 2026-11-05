using BayesianRegressionModels

function svi_owned_data(labels=collect(1:6); shared=false)
    n = length(labels)
    (; subject=["s$i" for i in labels],
        x=[[-0.7, 0.2, 0.9] for _ in labels],
        y=[[0.1i, -0.2, 0.4] for i in labels],
        c1=shared ? ones(Int, n) : 2 .* labels .- 1,
        c2=2 .* labels, active=falses(n), ncoeff=2maximum(labels))
end

function svi_owned_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        tau ~ Exponential(1)
        z[1:ncoeff] .~ Normal(0.4, tau)
        mu ~ 1 + (1 | unit | subject)
        @plate for i in eachindex(mu)
            y[i] ~ Normal(mu[i] .+ ifelse(active[i], z[c1[i]], z[c2[i]] .* x[i]), sigma)
        end
    end
end

function svi_owned_connected_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        tau ~ Exponential(1)
        z[1:ncoeff] .~ Normal(0.4, tau)
        mu ~ 1 + (1 | unit | subject)
        @plate for i in eachindex(left)
            y[i] ~ Normal(mu[left[i]] .+ weight[i] .* mu[right[i]] .+
                z[c1[i]] .+ z[c2[i]] .* x[i], sigma)
        end
    end
end
