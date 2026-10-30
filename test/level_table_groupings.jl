# test/level_table_groupings.jl — labels read from a level table through each
# part's positions (snag `groupings-over-a-cb504b6a`, design decision `0mtr2eb`).
# The data hold one table of areas (each area's label, unit and Kreis) and,
# per part, only its rows' positions into it. Each part names its labels as a
# data-only gather (`unit_b = reg_unit[pos_b]`), which every grouping and term
# site reads as a data column (user decision `1tb1jrg`). Each model is checked
# against the same model given the gathered labels as data columns (one copy
# per part): SBBRMI emits the same program and data and its descriptor the same
# outputs and random-effect blocks (snag `brm-descriptor-r-20f52245`), RKBRMI
# has the same coordinates, density and gradient and binds the level table, not
# the copies.
# Covered: a shared `|ID|` block over two parts, `car(...)` areas, `car_units`
# keyed by the gathered names, a nested grouping named in `nested_groups`,
# `(… || g)`, `cdar(...; by = g)` and a stratified `gr(g, by = b)` block (on
# RKBRMI from ReactiveKernelsPPL 0.1.15, snag `rkppl-submodel-l-00ff0f27`). A
# gather written inline is refused with the assignment to write.
#
# Run: julia --project=test test/level_table_groupings.jl [testset substrings...]
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LinearAlgebra, Random

message(f) = try f(); "" catch e; sprint(showerror, e) end
rk_names(rk) = Vector{Symbol}(BRM._rk_layout_coordinate_names(rk))
car_terms(rk) = [t for p in BRM._rk_replay_predictors(rk.plan) for t in p.terms
    if t.kind === :car]

# 22 areas on a chain r1 – … – r18, two more edges across units and one inside
# a; r19 – r20 is a component of unit f, r21 an island of c, and r22 lies on an
# edge only. Each unit holds two Kreise.
const LEVELS = let
    unit(j) = j <= 18 ? string("abcdef"[cld(j, 3)]) : j == 21 ? "c" : "f"
    kreis(j) = string(unit(j), j <= 18 && mod1(j, 3) == 3 ? "2" : "1")
    (; reg=["r$j" for j in 1:22], reg_unit=[unit(j) for j in 1:22],
        reg_kreis=[kreis(j) for j in 1:22],
        from=vcat(["r$j" for j in 1:17], ["r2", "r8", "r1", "r19", "r18"]),
        to=vcat(["r$j" for j in 2:18], ["r5", "r14", "r3", "r20", "r22"]))
end
# Part a on 32 rows, part b on 21 rows in reverse order: positions into LEVELS.
const PARTS = let rng = Xoshiro(20261029)
    pos_a = vcat(1:21, 1:2:21)
    pos_b = collect(21:-1:1)
    x = round.(randn(rng, length(pos_a)); digits=2)
    y = round.(0.3 .+ 0.8 .* x .+ 0.4 .* randn(rng, length(pos_a)); digits=2)
    x_b = round.(randn(rng, length(pos_b)); digits=2)
    y_b = round.(-0.2 .+ 0.5 .* x_b .+ 0.4 .* randn(rng, length(pos_b)); digits=2)
    (; pos_a, pos_b, x, y, x_b, y_b)
end
const TABLE = merge(LEVELS, PARTS)
# The same data with one copy of each label per part.
copies(parts, levels, names...) = merge(parts, (; from=levels.from, to=levels.to),
    (; (n => getproperty(levels, source)[getproperty(parts, pos)]
        for (n, source, pos) in names)...))
const COPIES = copies(PARTS, LEVELS, (:area, :reg, :pos_a), (:unit, :reg_unit, :pos_a),
    (:kreis, :reg_kreis, :pos_a), (:area_b, :reg, :pos_b),
    (:unit_b, :reg_unit, :pos_b), (:kreis_b, :reg_kreis, :pos_b))

# A symbolic size (`num_elements(t)`) is a StanBlocks expression that compares
# by identity, even between two builds of one model, so sizes compare printed.
output_fields(o) = (o.name, o.kind, o.type, repr(o.size), o.generative, o.role,
    o.logical, o.labels, o.component, o.dims, o.predictors, o.term)
block_fields(b) = map(f -> getfield(b, f), fieldnames(RanefBlock))

# The descriptor, and the prediction plan behind it, reads a gathered grouping
# as the copied model's data column: the same outputs and random-effect blocks,
# their levels included (snag `brm-descriptor-r-20f52245`).
function descriptor_same(sg, sc)
    dg, dc = brm_descriptor(sg), brm_descriptor(sc)
    @test map(output_fields, dg.outputs) == map(output_fields, dc.outputs)
    @test !isempty(ranef_blocks(dg))
    @test map(block_fields, ranef_blocks(dg)) == map(block_fields, ranef_blocks(dc))
    dg, dc
end

function sb_same(gathered, copied)
    sg, sc = SBBRMI(gathered; mod=@__MODULE__), SBBRMI(copied; mod=@__MODULE__)
    @test sprint(show, stan_code(sg)) == sprint(show, stan_code(sc))
    @test stan_data(sg) == stan_data(sc)
    descriptor_same(sg, sc)
    sg
end

function rk_same(gathered, copied)
    rg, pg = consumer_problem(gathered)
    rc, pc = consumer_problem(copied)
    names = rk_names(rg)
    @test names == rk_names(rc)
    n = length(names)
    for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
        vg, gg = LogDensityProblems.logdensity_and_gradient(pg, u)
        vc, gc = LogDensityProblems.logdensity_and_gradient(pc, u)
        @test vg ≈ vc atol=1e-10 rtol=1e-10
        @test gg ≈ gc atol=1e-9 rtol=1e-9
    end
    rg, rc
end

string_fields(rk) = Set(k for (k, v) in pairs(rk_bound_data(rk))
    if v isa AbstractVector{<:AbstractString})

two_parts(data) = @brm data begin
    sigma ~ Exponential(1)
    area = reg[pos_a]
    unit = reg_unit[pos_a]
    area_b = reg[pos_b]
    unit_b = reg_unit[pos_b]
    mu ~ 1 + x + (1 | u | unit) + car(area; edges=(from, to), type=:icar)
    mu_b ~ 1 + x_b + (1 | u | unit_b) + car(area_b; edges=(from, to), type=:icar)
    y ~ Normal(mu, sigma)
    @plate for i in eachindex(y_b)
        y_b[i] ~ Normal(mu_b[i], sigma)
    end
end
two_parts_copied(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | u | unit) + car(area; edges=(from, to), type=:icar)
    mu_b ~ 1 + x_b + (1 | u | unit_b) + car(area_b; edges=(from, to), type=:icar)
    y ~ Normal(mu, sigma)
    @plate for i in eachindex(y_b)
        y_b[i] ~ Normal(mu_b[i], sigma)
    end
end

@stestset "a shared |ID| block and car areas over parts read a level table" begin
    sg = sb_same(two_parts(TABLE), two_parts_copied(COPIES))
    @test StanBlocks.stan.compiles(sg.model)
    rg, rc = rk_same(two_parts(TABLE), two_parts_copied(COPIES))
    # RK computes the gathers in its graph: it binds the level table and the
    # positions, never a per-part copy of a label.
    @test isempty(intersect(string_fields(rg), (:area, :area_b, :unit, :unit_b)))
    @test (:reg, :reg_unit) ⊆ string_fields(rg)
    @test (:area, :area_b, :unit, :unit_b) ⊆ string_fields(rc)
    @test length(string_fields(rg)) < length(string_fields(rc))
    # A replay computes the gathers again from the new data's table and
    # positions: the same replay as one given the copies.
    shifted(d) = merge(d, (; x=reverse(d.x), x_b=reverse(d.x_b)))
    @test stan_data(reprocess(sg, shifted(TABLE))) ==
        stan_data(reprocess(SBBRMI(two_parts_copied(COPIES); mod=@__MODULE__),
            shifted(COPIES)))
    u = fill(0.1, length(rk_names(rg)))
    density(rk) = LogDensityProblems.logdensity(
        rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), u)
    @test density(reprocess(rg, shifted(TABLE))) ≈
        density(reprocess(rc, shifted(COPIES))) atol=1e-10 rtol=1e-10
end

@stestset "car_units keyed by gathered names separates each part's field" begin
    units = (; area=:unit, area_b=:unit_b)
    rk = RKBRMI(two_parts(TABLE); car_units=units)
    copied = RKBRMI(two_parts_copied(COPIES); car_units=units)
    @test count(t -> haskey(t.options, :separation), car_terms(rk)) == 2
    @test rk_names(rk) == rk_names(copied)
    v = brm_nlme_view(rk; subject_group=:unit)
    vc = brm_nlme_view(copied; subject_group=:unit)
    @test !any(==(:unclaimed), v.role)
    @test v.coordinates == vc.coordinates && v.role == vc.role
end

# Kreise nested in the units, joined across the parts by one `|k|` block.
nested(data) = @brm data begin
    sigma ~ Exponential(1)
    unit = reg_unit[pos_a]
    kreis = reg_kreis[pos_a]
    kreis_b = reg_kreis[pos_b]
    mu ~ 1 + x + (1 | unit) + (1 | k | kreis)
    mu_b ~ 1 + x_b + (1 | k | kreis_b)
    y ~ Normal(mu, sigma)
    @plate for i in eachindex(y_b)
        y_b[i] ~ Normal(mu_b[i], sigma)
    end
end
nested_copied(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | unit) + (1 | k | kreis)
    mu_b ~ 1 + x_b + (1 | k | kreis_b)
    y ~ Normal(mu, sigma)
    @plate for i in eachindex(y_b)
        y_b[i] ~ Normal(mu_b[i], sigma)
    end
end

@stestset "a nested grouping gathered from the table is named in nested_groups" begin
    sb_same(nested(TABLE), nested_copied(COPIES))
    rg, rc = rk_same(nested(TABLE), nested_copied(COPIES))
    groups = (; subject_group=:unit, nested_groups=(kreis=:unit,))
    v, vc = brm_nlme_view(rg; groups...), brm_nlme_view(rc; groups...)
    @test !any(==(:unclaimed), v.role)
    @test v.coordinates == vc.coordinates && v.role == vc.role && v.subject == vc.subject
end

# Three years per area, one part.
const YEARS = let rng = Xoshiro(7)
    pos = repeat(1:21; inner=3)
    t = repeat([0.0, 1.0, 2.0]; outer=21)
    x = round.(randn(rng, length(pos)); digits=2)
    y = round.(0.1 .+ 0.5 .* x .+ 0.4 .* randn(rng, length(pos)); digits=2)
    nk = length(unique(LEVELS.reg_kreis[1:21]))
    (; pos, t, x, y, I_k=Matrix{Float64}(I, nk, nk))
end
const YEARS_COPIES = merge(YEARS, (; kreis=LEVELS.reg_kreis[YEARS.pos],
    unit=LEVELS.reg_unit[YEARS.pos]))

walks(data) = @brm data begin
    sigma ~ Exponential(1)
    kreis = reg_kreis[pos]
    unit = reg_unit[pos]
    mu ~ 1 + (1 + x || unit) + cdar(t; by = kreis, cor = I_k)
    y ~ Normal(mu, sigma)
end
walks_copied(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 + x || unit) + cdar(t; by = kreis, cor = I_k)
    y ~ Normal(mu, sigma)
end

@stestset "|| blocks and cdar walks read a gathered grouping" begin
    sb_same(walks(merge(LEVELS, YEARS)), walks_copied(YEARS_COPIES))
    rk_same(walks(merge(LEVELS, YEARS)), walks_copied(YEARS_COPIES))
end

strata(data) = @brm data begin
    sigma ~ Exponential(1)
    kreis = reg_kreis[pos]
    unit = reg_unit[pos]
    mu ~ 1 + x + (1 | gr(kreis, by = unit))
    y ~ Normal(mu, sigma)
end
strata_copied(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | gr(kreis, by = unit))
    y ~ Normal(mu, sigma)
end

@stestset "a stratified block over gathered labels" begin
    # RKBRMI's stratified component reads its scales and draws at the labels
    # `kreis[i]` / `unit[i]`; RKPPL looks them up for a data-only definition
    # from ReactiveKernelsPPL 0.1.15 (snag `rkppl-submodel-l-00ff0f27`). Before
    # it the density raised `invalid index: "a" of type String`.
    sg = sb_same(strata(merge(LEVELS, YEARS)), strata_copied(YEARS_COPIES))
    rk_same(strata(merge(LEVELS, YEARS)), strata_copied(YEARS_COPIES))
    # Its draws per stratum read the strata off the gathered `by` labels. Both
    # models emit one program, so one compiled model serves both descriptors.
    stan = consumer_stan(strata(merge(LEVELS, YEARS)), "level_table_strata")
    names = BridgeStan.param_names(stan.model; include_tp=true)
    rng = Xoshiro(3)
    draws = reduce(vcat, permutedims(BridgeStan.param_constrain(stan.model,
        0.4 .* randn(rng, BridgeStan.param_unc_num(stan.model)); include_tp=true))
        for _ in 1:3)
    by_stratum(sb) = brm_ranef_draws(brm_descriptor(sb), draws, names; group=:kreis)
    gathered = by_stratum(sg)
    copied = by_stratum(SBBRMI(strata_copied(YEARS_COPIES); mod=@__MODULE__))
    @test [e.stratum for e in gathered] == [e.stratum for e in copied]
    @test Set(e.stratum for e in gathered) == Set(LEVELS.reg_unit[YEARS.pos])
    @test [(e.levels, e.sd, e.effects) for e in gathered] ==
        [(e.levels, e.sd, e.effects) for e in copied]
end

inline_group(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | reg_unit[pos_a])
    y ~ Normal(mu, sigma)
end
inline_stratum(data) = @brm data begin
    sigma ~ Exponential(1)
    unit = reg_unit[pos_a]
    mu ~ 1 + x + (1 | gr(unit, by = reg_kreis[pos_a]))
    y ~ Normal(mu, sigma)
end
inline_car(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + car(reg[pos_a]; edges=(from, to), type=:icar)
    y ~ Normal(mu, sigma)
end

@stestset "a gather written inline is refused with the assignment to write" begin
    # refused: an inline gather has no name for a prior, an NLME keyword or the
    # descriptor to address; the named gather is the spelling (principle
    # `1mwjios`, design decision `0mtr2eb`).
    for build in (data -> SBBRMI(data; mod=@__MODULE__), RKBRMI)
        @test occursin("write `reg_unit_pos_a = reg_unit[pos_a]` as its own statement",
            message(() -> build(inline_group(TABLE))))
        @test occursin("write `reg_kreis_pos_a = reg_kreis[pos_a]` as its own statement",
            message(() -> build(inline_stratum(TABLE))))
        @test occursin("write `reg_pos_a = reg[pos_a]` as its own statement",
            message(() -> build(inline_car(TABLE))))
    end
end

println("level_table_groupings.jl: all testsets passed")
