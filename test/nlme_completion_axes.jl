include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems, LinearAlgebra, Random
include(joinpath(@__DIR__, "nlme_unit_checks.jl"))

include(joinpath(@__DIR__, "nlme_completion_fixture.jl"))

function ordinary_gathered_completion(data)
    @brm data begin
        mu ~ 1 + x + (1 | subject)
        y ~ Normal(mu, 0.5)
        mu_pub = mu[pos]
        mi(v) ~ LogNormal(mu_pub, 0.5)
        lv = log(v)
        w ~ Normal(lv, 0.5)
    end
end

@stestset "ordinary completions can repeat source rows on a longer response axis" begin
    data = merge(COMPLETION_AXES_DATA, (;
        pos=repeat(COMPLETION_AXES_DATA.pos, 2),
        v=repeat(COMPLETION_AXES_DATA.v, 2), w=fill(1., 16)))
    rk = RKBRMI(ordinary_gathered_completion(data); completions=:noncentered)
    v = brm_nlme_view(rk; subject_group=:subject)
    missing_rows = findall(ismissing, data.v)
    owned = [v.subject[only(findall(==(Symbol("v.z_mis.", k)), v.coordinates))]
        for k in eachindex(missing_rows)]
    @test owned == data.subject[data.pos[missing_rows]]
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:subject)
    check_lockstep(m, Xoshiro(49))
    check_unit_problems(m, Xoshiro(50))
end

@stestset "same-length completion gathers preserve row order and reject coupled groups" begin
    data = merge(COMPLETION_AXES_DATA, (;
        pos=[8, 7, 2, 1, 12, 11, 5, 10, 3, 4, 6, 9],
        v=vcat(COMPLETION_AXES_DATA.v, [2., 3., 2., 4.]),
        part=vcat(COMPLETION_AXES_DATA.part, [5, 5, 6, 6]),
        w=fill(1., 12), lo=fill(0.2, 12)))
    rk = RKBRMI(gathered_completion(data); completions=:noncentered)
    v = brm_nlme_view(rk; subject_group=:subject)
    @test v.subject[only(findall(==(Symbol("v.z_shares.1")), v.coordinates))] == 2
    @test v.subject[only(findall(==(Symbol("v.z_free.1")), v.coordinates))] == 2
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:subject)
    @test only(last(a) for a in m.attribution if first(a) === :v_obs) == data.subject[data.pos[8:12]]
    check_unit_problems(m, Xoshiro(47))
    crossed = merge(data, (; pos=[8, 1, data.pos[3:end]...]))
    bad = RKBRMI(gathered_completion(crossed); completions=:noncentered)
    @test_throws BRM.BRMNLMEViewError brm_nlme_view(bad; subject_group=:subject)
end

@stestset "gathered completion coordinates and terms belong to their source subjects" begin
    rk = RKBRMI(gathered_completion(COMPLETION_AXES_DATA); completions=:noncentered)
    v = brm_nlme_view(rk; subject_group=:subject)
    for (name, owner) in zip(["v.z_shares.1", "v.z_shares.2", "v.z_shares.3", "v.z_free.1"], [2, 1, 3, 2])
        k = only(findall(==(Symbol(name)), v.coordinates))
        @test v.role[k] === :subject_effect
        @test v.subject[k] == owner
    end
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:subject)
    @test only(last(a) for a in m.attribution if first(a) === :v_shares_weights) == [2, 1, 3]
    check_lockstep(m, Xoshiro(43))
    check_unit_problems(m, Xoshiro(44))
end

@stestset "bounded free entries retain their density as subject draws" begin
    rk = RKBRMI(bounded_gathered_completion(COMPLETION_AXES_DATA); completions=:noncentered)
    v = brm_nlme_view(rk; subject_group=:subject)
    k = only(findall(==(Symbol("v.z_free.1")), v.coordinates))
    @test v.role[k] === :subject_effect
    @test v.subject[k] == 2
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:subject)
    check_lockstep(m, Xoshiro(45))
    check_unit_problems(m, Xoshiro(46))
end
