# test/nlme_grouped_hsgp_fixture.jl — public synthetic data and models with a
# grouped HSGP whose levels are the subjects, shared by
# test/nlme_grouped_hsgp.jl and test/nlme_grouped_hsgp_fit.jl. Include it after
# nlme_fixtures.jl.

# Four subjects with interleaved rows (flat rows).
const GH_FLAT = (;
    subject=["a", "b", "c", "a", "d", "b", "c", "a", "d", "b", "c", "d", "a", "b"],
    region=["n", "n", "s", "n", "s", "n", "s", "n", "s", "n", "s", "s", "n", "n"],
    t=[-0.8, -0.6, -0.9, -0.1, -0.5, 0.2, 0.0, 0.5, 0.4, 0.7, 0.6, 0.9, 0.9, -0.2],
    x=[0.3, -0.2, 0.5, 0.1, -0.7, 0.4, -0.1, 0.2, 0.6, -0.3, 0.0, 0.8, -0.5, 0.1],
    y=[0.4, -0.3, 0.9, 0.2, -0.6, 0.1, 0.5, 0.7, 0.0, -0.4, 0.8, 0.3, 0.6, -0.1])

flat_hsgp(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + hsgp(t; by=subject, k=3, domain=(-1.0, 1.0)) + (1 | subject)
    y ~ Normal(mu, sigma)
end

# Subjects on their own axis; ECG rows labelled by `ecg_subject`, read in each
# subject's plate cell through a `groups` binding. Subject "d" has no ECG row
# in `GH_PLATE_GAP`.
const GH_PLATE = (; subject=["a", "b", "c", "d"],
    ecg_subject=["a", "b", "a", "c", "d", "b", "c", "a", "d", "c", "b", "d"],
    ecg_time=[0.0, 0.5, 1.0, 0.2, 0.1, 1.5, 1.8, 2.5, 2.2, 3.0, 2.9, 3.4],
    qt=[0.1, 0.9, 0.4, -0.2, 1.4, 0.6, 0.0, 0.8, 1.7, -0.4, 1.1, 1.5])
const GH_PLATE_GAP = (; subject=["a", "b", "c", "d"],
    ecg_subject=["a", "b", "a", "c", "b", "c", "a", "c", "b"],
    ecg_time=[0.0, 0.5, 1.0, 0.2, 1.5, 1.8, 2.5, 3.0, 2.9],
    qt=[0.1, 0.9, 0.4, -0.2, 0.6, 0.0, 0.8, -0.4, 1.1])

plate_hsgp(data) = @brm data begin
    sigma ~ Exponential(1)
    base ~ 1 + (1 | subject)
    qt_time ~ 0 + hsgp(ecg_time; by=ecg_subject, k=4, orthogonal_to=:intercept)
    ecg_rows = groups(ecg_subject => subject)
    @plate for i in eachindex(base)
        qt[ecg_rows[i]] ~ Normal(base[i] .+ qt_time[ecg_rows[i]], sigma)
    end
end

plate_hyper(data) = @brm data begin
    sigma ~ Exponential(1)
    base ~ 1 + (1 | subject)
    qt_time ~ 0 + hsgp(ecg_time; by=ecg_subject, k=3)
    log(sd(hsgp(ecg_time))) ~ 1 + (1 | ecg_subject)
    ecg_rows = groups(ecg_subject => subject)
    @plate for i in eachindex(base)
        qt[ecg_rows[i]] ~ Normal(base[i] .+ qt_time[ecg_rows[i]], sigma)
    end
end

weight_family(v) = findfirst(f -> hasproperty(f.block, :hsgp), v.families)
