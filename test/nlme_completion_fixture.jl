# Shared gathered-response fixtures for NLME ownership and fitting checks.
const COMPLETION_AXES_DATA = (;
    subject=repeat([1, 2, 3]; inner=4), x=collect(range(-1., 1.; length=12)),
    y=fill(0.3, 12), pos=[8, 7, 2, 1, 12, 11, 5, 10],
    part=[1, 1, 2, 2, 3, 3, 4, 4], cells=[1, 2, 3], total=[5., 6., 7.],
    v=Union{Missing,Float64}[missing, missing, missing, missing, missing, missing, missing, 4.5],
    w=fill(1., 8), lo=fill(0.2, 8))

function gathered_completion(data)
    @brm data begin
        mu ~ 1 + x + (1 | subject)
        y ~ Normal(mu, 0.5)
        mu_shifted = mu + 0.2
        mu_gathered = mu_shifted[pos]
        mu_pub = log(exp(mu_gathered))
        rows = groups(part => cells)
        mi(v)::shares(total, rows) ~ LogNormal(mu_pub, 0.5)
        lv = log(v)
        w ~ Normal(lv, 0.5)
    end
end

function bounded_gathered_completion(data)
    @brm data begin
        mu ~ 1 + x + (1 | subject)
        y ~ Normal(mu, 0.5)
        mu_pub = mu[pos]
        rows = groups(part => cells)
        mi(v)::shares(total, rows) ~ restricted(LogNormal(mu_pub, 0.5); lower=lo)
        lv = log(v)
        w ~ Normal(lv, 0.5)
    end
end
