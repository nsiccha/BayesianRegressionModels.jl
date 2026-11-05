# Generic Gaussian observations that may read several scientific subjects.
connected_model(data) = @brm data begin
    mu ~ 1 + (1 | effect | subject)
    @plate for i in eachindex(y)
        y[i] ~ Normal(mu[left[i]] + weight[i] * mu[right[i]], 0.5)
    end
end

function connected_data(n, left, right; weight=ones(length(left)))
    (; subject=["subject $i" for i in 1:n], left, right, weight,
       y=collect(range(-0.3, 0.7; length=length(left))))
end
