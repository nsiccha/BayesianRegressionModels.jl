# test/nlme_crossed_fixture.jl — public synthetic data and models with a
# grouping crossed with the NLME units (`shared_groups`) and a `car` field whose
# edges cross them, shared by test/nlme_crossed.jl and
# test/nlme_crossed_fit.jl. Include it after nlme_fixtures.jl.

using Random

const CR_RNG = Xoshiro(20261028)

# Six subjects, three size classes; every class occurs in several subjects.
const CR_SUBJECTS = ["a", "b", "c", "d", "e", "f"]
const CR_FLAT = let n = 18
    subject = repeat(CR_SUBJECTS, 3)
    class = ["k$(1 + (i + div(i - 1, 6)) % 3)" for i in 1:n]
    x = round.(randn(CR_RNG, n); digits=2)
    effect = Dict(s => 0.6 * randn(CR_RNG) for s in CR_SUBJECTS)
    level = Dict("k1" => -0.5, "k2" => 0.1, "k3" => 0.6)
    y = [round(0.3 + 0.8 * x[i] + effect[subject[i]] + level[class[i]] +
        0.3 * randn(CR_RNG); digits=2) for i in 1:n]
    (; subject, class, x, y)
end

crossed_flat(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject) + (1 + x | class)
    y ~ Normal(mu, sigma)
end

# Subjects on their own axis; observation rows labelled by `obs_subject`, read
# in each subject's plate cell through a `groups` binding. The rows' predictor
# carries a class intercept: with a class slope the RK model's own static
# Reverse gradient is rejected by Enzyme, NLME or not (BRM todo
# 2026-10-28T14-26-47-032-1ltq3fp).
const CR_PLATE = (; subject=CR_SUBJECTS, obs_subject=CR_FLAT.subject,
    class=CR_FLAT.class, x=CR_FLAT.x, y=CR_FLAT.y)

crossed_plate(data) = @brm data begin
    sigma ~ Exponential(1)
    base ~ 1 + (1 | subject)
    row ~ 0 + x + (1 | class)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(base)
        y[obs[i]] ~ Normal(base[i] .+ row[obs[i]], sigma)
    end
end

# Areas nested in the subjects (subject a's rows lie in r1 and r2, b's in r3
# and r4, …), on a chain r1 – r2 – … – r12 whose edges cross the subjects, as
# Gemeinden lie in Länder and border Gemeinden of neighbouring Länder.
const CR_CAR = merge(CR_FLAT, (;
    area=["r$(2 * findfirst(==(s), CR_SUBJECTS) - (i <= 6 ? 1 : 0))"
        for (i, s) in enumerate(CR_FLAT.subject)],
    from=["r$j" for j in 1:11], to=["r$j" for j in 2:12]))

crossed_car(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject) + car(area; edges=(from, to), type=:icar)
    y ~ Normal(mu, sigma)
end

class_record(rk) = only(r for r in BRM._brm_nlme_records(rk.plan)
    if r.kind === :ranef && r.group === :class)
owned(v, declaration) = findall(n -> BRM._brm_nlme_owned(n, declaration), v.coordinates)
# Every coordinate of the class blocks (an uncorrelated `(1 + x || class)` is
# one block per margin).
class_coordinates(v, rk) = sort!(reduce(vcat, (owned(v, BRM._brm_nlme_ranef_declaration(r))
    for r in BRM._brm_nlme_records(rk.plan) if r.kind === :ranef && r.group === :class)))

# Other block forms over crossed groupings: a stratified class block (classes
# k1 and k2 small, k3 large) and a Student-t region block; a block whose
# region levels are correlated by a known matrix, and a known class scale.
const CR_FORMS = merge(CR_FLAT, (;
    size=[c == "k3" ? "large" : "small" for c in CR_FLAT.class],
    region=["n", "s", "n", "s", "s", "n", "s", "n", "s", "n", "n", "s",
            "n", "n", "s", "s", "n", "s"],
    C=[1.0 0.4; 0.4 1.0]))

crossed_stratified(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject) + (1 | gr(class, by = size)) + (1 | region)
    ranef(mu, region) ~ TDist(4.0)
    y ~ Normal(mu, sigma)
end

crossed_known(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject) + (1 | gr(region; cor = C)) + (1 | class)
    sd(mu, class) = 0.4
    y ~ Normal(mu, sigma)
end
