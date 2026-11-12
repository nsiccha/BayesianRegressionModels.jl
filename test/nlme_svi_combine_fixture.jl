using BayesianRegressionModels, Random, Statistics

# A public synthetic panel: subjects nested in regions, a class grouping crossed
# with them (`shared_groups`), a population intercept and two estimated scales.
# Under the log transform the scales' LogNormal priors are Normal, so the
# global prior is Gaussian and its projection has a closed form.
function svi_slice_model(data)
    @brm data begin
        a ~ Normal(0, 2)
        mu ~ 0 + (1 | subject) + (1 | class)
        sd(mu, subject) ~ LogNormal(log(0.7), 0.4)
        sd(mu, class) ~ LogNormal(log(0.5), 0.4)
        mean_y = a + mu
        y ~ Normal(mean_y, 0.5)
    end
end

# The same with a covariate standardized on the rows it is bound to.
function svi_slice_scaled_model(data)
    @brm data begin
        a ~ Normal(0, 2)
        mu ~ 0 + standardize(x) + (1 | subject) + (1 | class)
        sd(mu, subject) ~ LogNormal(log(0.7), 0.4)
        sd(mu, class) ~ LogNormal(log(0.5), 0.4)
        mean_y = a + mu
        y ~ Normal(mean_y, 0.5)
    end
end

const SVI_SLICE_GROUPS = (; subject_group=:subject, shared_groups=(:class,))
# The combination algebra below has packed-coordinate factors as its oracle.
# Automatic centering has its own transport and end-to-end regressions.
const SVI_SLICE_QUICK = (; max_iterations=0, ndraws=0, centering=nothing, SVI_SLICE_GROUPS...)

# `regions` maps a region to its subjects; each subject has `n` rows whose
# classes cycle through `classes[region]`.
function svi_slice_data(rng, regions, classes, n)
    subject, class, region = String[], String[], String[]
    for (r, members) in regions, s in members, k in 1:n
        push!(subject, s)
        push!(region, r)
        push!(class, classes[r][mod1(k, length(classes[r]))])
    end
    effect(s, scale) = scale * sin(sum(Int, codeunits(s)))
    y = [0.8 + effect(s, 0.6) + effect(c, 0.4) + 0.5randn(rng) for (s, c) in zip(subject, class)]
    x = randn(rng, length(y)) .+ [r == first(first(regions)) ? 3.0 : 0.0 for r in region]
    (; subject, class, region, x, y)
end

svi_slice_rows(data, keep) = map(v -> v[keep], data)
svi_slice_region(data, r) = svi_slice_rows(data, data.region .== r)

const SVI_SLICE_REGIONS = ["north" => ["n1", "n2", "n3"], "south" => ["s1", "s2"],
    "east" => ["e1", "e2", "e3", "e4"]]
const SVI_SLICE_CLASSES = Dict("north" => ["k1", "k2"], "south" => ["k2", "k3"],
    "east" => ["k1", "k3", "k4"])

# With a population intercept the subject block is mu-referenced: a subject's
# total intercept is `mu_Intercept + τ z`.
function svi_slice_intercept_model(data)
    @brm data begin
        mu ~ 1 + (1 | subject) + (1 | class)
        sd(mu, subject) ~ LogNormal(log(0.7), 0.4)
        sd(mu, class) ~ LogNormal(log(0.5), 0.4)
        y ~ Normal(mu, 0.5)
    end
end

# A correlated intercept and slope per subject, both mu-referenced.
function svi_slice_slope_model(data)
    @brm data begin
        mu ~ 1 + x + (1 + x | subject)
        y ~ Normal(mu, 0.5)
    end
end

# Subjects' effects on a log scale, so a draw read under another scale moves
# the mean by a factor.
function svi_slice_rate_model(data)
    @brm data begin
        mu ~ 1 + (1 | subject)
        sd(mu, subject) ~ LogNormal(log(0.3), 1.0)
        mean_y = exp(mu)
        y ~ Normal(mean_y, 0.05)
    end
end

# Two regions whose subjects differ in spread by a factor of 40: "calm"
# subjects' effects are about ±0.03, "wild" ones' about ±1.2.
function svi_slice_rate_data(rng, n)
    subject, region = String[], String[]
    for (r, members) in ("calm" => ["c$i" for i in 1:6], "wild" => ["w$i" for i in 1:3]),
            s in members, _ in 1:n
        push!(subject, s)
        push!(region, r)
    end
    effect(s) = (startswith(s, "c") ? 0.03 : 1.2) * sin(3 * sum(Int, codeunits(s)))
    y = [exp(1.0 + effect(s)) + 0.05randn(rng) for s in subject]
    (; subject, region, y, x=zeros(length(y)))
end

# The subject block stratified by a subject-level kind: one covariance family
# per kind, so the NLME view's units hold different families (ragged).
function svi_slice_stratified_model(data)
    @brm data begin
        mu ~ 1 + (1 | gr(subject, by = kind))
        y ~ Normal(mu, 0.5)
    end
end
svi_slice_kinds(data) = merge(data, (; kind=[isodd(sum(Int, codeunits(s))) ? "a" : "b"
    for s in data.subject]))

# Each subject reads its own device, a grouping declared shared, so a device's
# draw sits in its subject's unit; the units' own block has no intercept, so
# the device block absorbs `mu_Intercept`.
function svi_slice_device_model(data)
    @brm data begin
        mu ~ 1 + (1 | device) + (0 + x | subject)
        sd(mu, device) ~ LogNormal(log(0.5), 0.4)
        y ~ Normal(mu, 0.5)
    end
end
svi_slice_devices(data) = merge(data, (; device=["d" * s for s in data.subject]))

# Two globals that each region's data pins only in one combination: `x` is +1
# in "plus" and -1 in "minus", so "plus" pins a + b and "minus" pins a - b.
# Each slice's posterior is a ridge along the direction its data leaves to
# the prior, and only the union pins both.
function svi_slice_ridge_model(data)
    @brm data begin
        a ~ Normal(0, 3)
        b ~ Normal(0, 3)
        mu ~ 0 + (1 | subject)
        sd(mu, subject) ~ LogNormal(log(0.3), 0.3)
        mean_y = a + b * x + mu
        y ~ Normal(mean_y, 0.3)
    end
end

function svi_slice_ridge_data(rng; subjects=6, n=6, a=2.0, b=1.5)
    subject, region, x = String[], String[], Float64[]
    for (r, sign) in ("plus" => 1.0, "minus" => -1.0), i in 1:subjects, _ in 1:n
        push!(subject, string(r, i))
        push!(region, r)
        push!(x, sign)
    end
    effect(s) = 0.3 * sin(sum(Int, codeunits(s)))
    y = [a + b * xi + effect(s) + 0.3randn(rng) for (s, xi) in zip(subject, x)]
    (; subject, region, x, y)
end
