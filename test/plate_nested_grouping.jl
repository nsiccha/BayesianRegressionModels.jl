# test/plate_nested_grouping.jl — a per-subject linear predictor whose random
# effects span several groupings, read with the plate index (snag
# `plate-predictor-c996ec8d`).
#
# Districts nested in regions, one row per district:
#
#     alpha ~ 1 + x + (1 | region | region_g) + (1 | district | district_g)
#
# `district_g` lists one unique label per row, so it labels the plate's cells;
# the region effect is an ordinary term already added into each row's
# `alpha[d]`, as `(1 | region) + (1 | district)` is in brms. The program must
# equal the spelling it replaces: a separate region predictor on a region
# frame, added in the cell by index.
#
# A predictor none of whose groupings lists one label per row (a size class
# over the cells, snag `cell-lp-with-a-c-3d6db673`) labels no cell: its rows
# are the cells of the frame the plate's other inputs define, and each
# grouping is an ordinary random effect of its row.
#
# Run: julia --startup-file=no --project=test test/plate_nested_grouping.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const ND = 12
const REGIONS = ["R1", "R2", "R3", "R4"]
const NESTED = let
    region_g = [REGIONS[1 + (k % 4)] for k in 1:ND]
    # Unsorted labels: cells stay in row order.
    district_g = ["D$(lpad(k, 2, '0'))" for k in [5, 2, 11, 8, 1, 12, 3, 9, 6, 10, 4, 7]]
    (; region_g, district_g,
       x=[0.1 * ((k * 7) % 5) - 0.2 for k in 1:ND],
       log_exposure=[[0.2 + 0.01k, 0.25, 0.3 - 0.01k] for k in 1:ND],
       period=[[1, 2, 3] for _ in 1:ND],
       count_r=[[(k + t) % 4 + 1 for t in 1:3] for k in 1:ND],
       count_s=[[(2k + t) % 3 for t in 1:3] for k in 1:ND],
       period_g=[1, 2, 3],
       region_frame=REGIONS,
       region_pos=[findfirst(==(r), REGIONS) for r in region_g])
end

# Two predictors share the region and district blocks, as in a multi-margin
# small-area model; a period effect on its own frame is read by index.
nested_model(data) = @brm data begin
    eta ~ 0 + (1 | per | period_g)
    alpha_r ~ 1 + x + (1 | region | region_g) + (1 | district | district_g)
    alpha_s ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    inv_sqrt_phi ~ Exponential(1)
    @plate for d in eachindex(alpha_r)
        count_r[d] ~ NegativeBinomial2(
            exp.(log_exposure[d] .+ alpha_r[d] .+ eta[period[d]]),
            1 / (inv_sqrt_phi * inv_sqrt_phi))
    end
    @plate for d in eachindex(alpha_s)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha_s[d]))
    end
end

# The consumer shape the fix replaces: the region effect as separate
# predictors over a region frame, added in every cell by index.
indexed_model(data) = @brm data begin
    eta ~ 0 + (1 | per | period_g)
    nu_r ~ 0 + (1 | region | region_frame)
    nu_s ~ 0 + (1 | region | region_frame)
    alpha_r ~ 1 + x + (1 | district | district_g)
    alpha_s ~ 1 + (1 | district | district_g)
    inv_sqrt_phi ~ Exponential(1)
    @plate for d in eachindex(alpha_r)
        count_r[d] ~ NegativeBinomial2(
            exp.(log_exposure[d] .+ alpha_r[d] .+ nu_r[region_pos[d]] .+ eta[period[d]]),
            1 / (inv_sqrt_phi * inv_sqrt_phi))
    end
    @plate for d in eachindex(alpha_s)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha_s[d] .+ nu_s[region_pos[d]]))
    end
end

# RK coordinate name => Stan unconstrained name, by the BRM naming scheme:
# `<block>.z.<level>.<margin>` is level-major in `<block>_z_flat`. `rename`
# pairs apply to the Stan names (a linked `log(cv)`'s intercept is RK's
# `cv_Intercept` but Stan's `pop_log_cv_beta_pop.1`).
function stan_names_for(rk_names, stan_names; rename=())
    K = Dict{String,Int}()
    for n in string.(rk_names)
        m = match(r"^(.*)\.tau\.(\d+)$", n)
        m === nothing || (K[m[1]] = max(get(K, m[1], 0), parse(Int, m[2])))
    end
    map(rk_names) do n
        s = string(n)
        stan = if (m = match(r"^(.*)\.z\.(\d+)\.(\d+)$", s)) !== nothing
            "$(m[1])_z_flat.$((parse(Int, m[2]) - 1) * K[m[1]] + parse(Int, m[3]))"
        elseif (m = match(r"^(.*)\.(tau|L|beta_pop)\.(\d+)$", s)) !== nothing
            "$(m[1])_$(m[2]).$(m[3])"
        elseif (m = match(r"^(.*)_Intercept$", s)) !== nothing
            "pop_$(m[1])_beta_pop.1"
        else
            s
        end
        stan = foldl(replace, rename; init=stan)
        @test stan in stan_names
        n => stan
    end
end

@stestset "an outer grouping beside the per-row grouping labels cells by the per-row one" begin
    data = NESTED
    saved = deepcopy(data)
    brmi = nested_model(data)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_count_r_cell] == ND
    @test sb.data[:kernel_nsub_count_s_cell] == ND
    # Both predictors stay parameters, sliced per cell.
    @test !haskey(sb.data, :alpha_r) && !haskey(sb.data, :alpha_s)
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok

    stan = consumer_stan(brmi, "plate-nested-grouping")
    backend, problem = consumer_problem(brmi)
    rk_names = coordinate_names(backend.model.layout)
    mapping = stan_names_for(rk_names, BridgeStan.param_unc_names(stan.model))
    n = length(rk_names)
    for u in (zeros(n), collect(range(-0.3, 0.3; length=n)), [0.1 * sin(3k) for k in 1:n])
        check_consumer_stan(problem, stan, mapping, backend, u)
    end

    # The same density as the index-read spelling: only the region block's
    # grouping column is named differently.
    indexed = consumer_stan(indexed_model(data), "plate-nested-grouping-indexed")
    names = replace.(BridgeStan.param_unc_names(stan.model),
        "region_region_g" => "region_region_frame")
    indexed_names = BridgeStan.param_unc_names(indexed.model)
    @test sort(names) == sort(indexed_names)
    # `indexed_u[order[i]] = u[i]`
    order = [findfirst(==(name), indexed_names) for name in names]
    for u in (zeros(n), collect(range(-0.4, 0.4; length=n)), [0.2 * cos(2k) for k in 1:n])
        @test BridgeStan.log_density(stan.model, u; propto=false) ≈
            BridgeStan.log_density(indexed.model, u[invperm(order)]; propto=false) atol=1e-10
    end
    @test isequal(data, saved)
end

# Several `|ID|` blocks over the one per-row grouping are one grouping.
two_blocks_model(data) = @brm data begin
    alpha ~ 1 + (1 | p | district_g) + (0 + x | q | district_g)
    @plate for d in eachindex(alpha)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha[d]))
    end
end

@stestset "several |ID| blocks over one grouping" begin
    sb = SBBRMI(two_blocks_model(NESTED); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_count_s_cell] == ND
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
    @test RKBRMI(two_blocks_model(NESTED)) isa RKBRMI
end

# No grouping column lists one label per row: each grouping is an ordinary
# random effect added into its row's value, as `(1 | region) + (1 | half)` is
# in brms, and the rows are the cells (snag `cell-lp-with-a-c-3d6db673`; the
# old long-format refusal guarded a derived cell count that dropped rows,
# decision `1cvsc29`). Here the pre-grouped `count_s` rows are the frame.
const HALVES = let half_g = repeat(["H1", "H2", "H3", "H4", "H5", "H6"], 2)
    merge(NESTED, (; half_g, half_frame=sort(unique(half_g)),
        half_pos=[findfirst(==(h), sort(unique(half_g))) for h in half_g]))
end
coarse_rows_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | half | half_g)
    effect(alpha, Intercept) ~ Normal(0, 1)
    sd(:, region) ~ Exponential(2)
    sd(:, half) ~ Exponential(2)
    @plate for d in eachindex(alpha)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha[d]))
    end
end
# The same model with the effects as separate predictors on their level
# frames, added in every cell by index.
indexed_rows_model(data) = @brm data begin
    nu_region ~ 0 + (1 | region | region_frame)
    nu_half ~ 0 + (1 | half | half_frame)
    a0 ~ Normal(0, 1)
    sd(:, region) ~ Exponential(2)
    sd(:, half) ~ Exponential(2)
    @plate for d in eachindex(count_s)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ a0 .+ nu_region[region_pos[d]] .+
            nu_half[half_pos[d]]))
    end
end

# Stan names of `model` renamed by `pairs`, as the positions of `twin`'s
# names: `twin_u[order[i]] = u[i]`.
function twin_order(model, twin, pairs...)
    names = replace.(BridgeStan.param_unc_names(model), pairs...)
    twin_names = BridgeStan.param_unc_names(twin)
    @test sort(names) == sort(twin_names)
    [findfirst(==(name), twin_names) for name in names]
end
function check_twin_density(model, twin, order)
    n = length(order)
    for u in (zeros(n), collect(range(-0.4, 0.4; length=n)), [0.2 * cos(2k) for k in 1:n])
        @test BridgeStan.log_density(model, u; propto=false) ≈
            BridgeStan.log_density(twin, u[invperm(order)]; propto=false) atol=1e-10
    end
end
function check_rk_stan(brmi, stan; rename=())
    backend, problem = consumer_problem(brmi)
    rk_names = coordinate_names(backend.model.layout)
    mapping = stan_names_for(rk_names, BridgeStan.param_unc_names(stan.model); rename)
    n = length(rk_names)
    for u in (zeros(n), collect(range(-0.3, 0.3; length=n)), [0.1 * sin(3k) for k in 1:n])
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
end

@stestset "groupings that label no row are ordinary effects of each row's cell" begin
    data = HALVES
    saved = deepcopy(data)
    sb = SBBRMI(coarse_rows_model(data); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_count_s_cell] == ND
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
    stan = consumer_stan(coarse_rows_model(data), "plate-coarse-rows")
    indexed = consumer_stan(indexed_rows_model(data), "plate-coarse-rows-indexed")
    order = twin_order(stan.model, indexed.model, "pop_alpha_beta_pop.1" => "a0",
        "region_region_g" => "region_region_frame", "half_half_g" => "half_half_frame")
    check_twin_density(stan.model, indexed.model, order)
    check_rk_stan(coarse_rows_model(data), stan)
    @test isequal(data, saved)
end

# Cells labelled by a binding's label vector; the per-cell predictor's only
# random effect groups the cells by a coarser class (snag
# `cell-lp-with-a-c-3d6db673`: a class's dispersion partially pooled over
# Kreis-year x class cells). Units join their cells by label.
const NCELL = 12
const CLASSES = ["A", "B", "C"]
const COARSE = let
    cells = ["K$(lpad(k, 2, '0'))" for k in [7, 3, 10, 1, 5, 12, 2, 9, 4, 11, 6, 8]]
    cell_class = [CLASSES[1 + (k % 3)] for k in 1:NCELL]
    unit_cell = cells[[1, 2, 3, 1, 4, 5, 6, 2, 7, 8, 9, 10, 11, 12, 3, 5, 7, 9, 11, 12,
        1, 6, 8, 10]]
    nunit = length(unit_cell)
    (; cells, cell_class, unit_cell,
       unit_site=["S$(1 + (u % 5))" for u in 1:nunit],
       x_u=[0.1 * ((u * 7) % 9) - 0.4 for u in 1:nunit],
       y=[(3k) % 7 + 2 for k in 1:NCELL],
       class_frame=CLASSES,
       class_pos=[findfirst(==(c), CLASSES) for c in cell_class])
end
coarse_cells_model(data) = @brm data begin
    log_mu ~ 1 + x_u + (1 | site | unit_site)
    log(cv) ~ 1 + (1 | cls | cell_class)
    effect(log_mu, Intercept) ~ Normal(0, 1)
    effect(log_mu, x_u) ~ Normal(0, 1)
    effect(cv, Intercept) ~ Normal(-1, 0.5)
    sd(:, site) ~ Exponential(2)
    sd(:, cls) ~ Exponential(2)
    rows = groups(unit_cell => cells)
    @plate for k in eachindex(y)
        y[k] ~ NegativeBinomial2(sum(exp.(log_mu[rows[k]])), 1.0 / cv[k]^2)
    end
end
# The class effect as a separate predictor on the class frame, added in every
# cell by index.
indexed_cells_model(data) = @brm data begin
    log_mu ~ 1 + x_u + (1 | site | unit_site)
    nu ~ 0 + (1 | cls | class_frame)
    effect(log_mu, Intercept) ~ Normal(0, 1)
    effect(log_mu, x_u) ~ Normal(0, 1)
    b0 ~ Normal(-1, 0.5)
    sd(:, site) ~ Exponential(2)
    sd(:, cls) ~ Exponential(2)
    rows = groups(unit_cell => cells)
    @plate for k in eachindex(y)
        y[k] ~ NegativeBinomial2(sum(exp.(log_mu[rows[k]])),
            1.0 / exp(b0 + nu[class_pos[k]])^2)
    end
end

@stestset "a per-cell predictor with a coarser grouping reads in cells its binding labels" begin
    data = COARSE
    saved = deepcopy(data)
    sb = SBBRMI(coarse_cells_model(data); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_y_cell] == NCELL
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
    stan = consumer_stan(coarse_cells_model(data), "plate-coarse-cells")
    indexed = consumer_stan(indexed_cells_model(data), "plate-coarse-cells-indexed")
    order = twin_order(stan.model, indexed.model, "pop_log_cv_beta_pop.1" => "b0",
        "cls_cell_class" => "cls_class_frame")
    check_twin_density(stan.model, indexed.model, order)
    check_rk_stan(coarse_cells_model(data), stan; rename=("pop_cv_" => "pop_log_cv_",))
    @test isequal(data, saved)

    # A forecast replays onto new cells of the trained classes and sites.
    future = (; cells=["K20", "K21", "K22", "K23", "K24"],
        cell_class=["B", "A", "C", "A", "B"],
        unit_cell=["K20", "K21", "K21", "K22", "K23", "K24", "K24", "K20"],
        unit_site=["S1", "S2", "S3", "S4", "S5", "S1", "S2", "S3"],
        x_u=[0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.4, -0.3],
        y=[3, 5, 2, 4, 6],
        class_frame=CLASSES, class_pos=[2, 1, 3, 1, 2])
    replay = reprocess(sb, future)
    @test replay.data[:kernel_nsub_y_cell] == 5
    @test BRM.stan_code(replay) == BRM.stan_code(sb)
    direct = SBBRMI(coarse_cells_model(future); mod=@__MODULE__, total_groups=())
    @test BRM.stan_code(direct) == BRM.stan_code(sb)
    for key in keys(direct.data)
        @test isequal(replay.data[key], direct.data[key])
    end
end

# Crossed coarse groupings in cells a binding labels, one block shared with
# the unit predictor (co-report of snag `cell-lp-with-a-c-3d6db673`: a cell
# predictor with Kreis and kind effects beside a Gemeinde predictor sharing the
# Kreis block).
const CROSSED = let regions = ["R1", "R2"]
    cell_region = [regions[1 + (k ÷ 3) % 2] for k in 1:NCELL]
    merge(COARSE, (; cell_region,
        unit_region=[cell_region[findfirst(==(c), COARSE.cells)] for c in COARSE.unit_cell]))
end
crossed_cells_model(data) = @brm data begin
    log_mu ~ 1 + x_u + (1 | r | unit_region)
    log_p ~ 1 + (1 | cls | cell_class) + (1 | r | cell_region)
    rows = groups(unit_cell => cells)
    @plate for k in eachindex(y)
        y[k] ~ Poisson(exp(log_p[k]) * sum(exp.(log_mu[rows[k]])))
    end
end

@stestset "crossed coarse groupings of a cell predictor, one block shared with the units" begin
    sb = SBBRMI(crossed_cells_model(CROSSED); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_y_cell] == NCELL
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
    check_rk_stan(crossed_cells_model(CROSSED),
        consumer_stan(crossed_cells_model(CROSSED), "plate-crossed-cells"))
end

# A per-row predictor beside one with an outer grouping only: the per-row
# grouping labels the cells, and the outer predictor rides on them.
outer_beside_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    beta ~ 1 + (1 | region | region_g)
    @plate for d in eachindex(alpha)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha[d] .+ beta[d]))
    end
end

@stestset "an outer-grouped predictor rides on the cells a per-row one labels" begin
    sb = SBBRMI(outer_beside_model(NESTED); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_count_s_cell] == ND
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
    check_rk_stan(outer_beside_model(NESTED),
        consumer_stan(outer_beside_model(NESTED), "plate-outer-beside"))
end

# refused: one cell's predictors describe different subjects, each grouping
# listing its own labels one per row.
disagreeing_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    beta ~ 1 + (1 | site | site_g)
    @plate for d in eachindex(alpha)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha[d] .+ beta[d]))
    end
end

@stestset "predictors labelling different cells are refused" begin
    data = merge(NESTED, (; site_g=["S$k" for k in 1:ND]))
    @test_throws "disagree on their grouping" SBBRMI(disagreeing_model(data);
        mod=@__MODULE__, total_groups=())
    @test_throws "disagree on subject grouping" RKBRMI(disagreeing_model(data))
end

# One district per region: both columns list one label per row and label the
# same rows in the same order.
const ONE_PER_REGION = let rows = [1, 2, 3, 4]
    (; region_g=NESTED.region_g[rows], district_g=NESTED.district_g[rows],
       log_exposure=NESTED.log_exposure[rows], count_s=NESTED.count_s[rows],
       ev_district=NESTED.district_g[[1, 1, 2, 3, 3, 3, 4]],
       ev_x=[0.1, 0.2, -0.1, 0.3, 0.0, 0.2, -0.2],
       ev_y=[0.5, 0.4, 0.1, 0.7, 0.2, 0.3, -0.1])
end

tie_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    @plate for d in eachindex(alpha)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha[d]))
    end
end

# Event entries join the cells by the label vector their binding names: with
# both groupings listing one label per cell, nothing is guessed.
tie_named_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    sigma ~ Exponential(1)
    events = groups(ev_district => district_g)
    @plate for d in eachindex(alpha)
        ev_y[events[d]] ~ Normal(alpha[d] .+ ev_x[events[d]], sigma)
    end
end

@stestset "several per-row groupings: labels decide only where rows join by label" begin
    data = ONE_PER_REGION
    # No label join: either column gives the same cells, count and order.
    sb = SBBRMI(tie_model(data); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_count_s_cell] == 4
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
    @test RKBRMI(tie_model(data)) isa RKBRMI
    named = SBBRMI(tie_named_model(data); mod=@__MODULE__, total_groups=())
    @test named.data[:kernel_nsub_ev_y_cell] == 4
    @test StanBlocks.stanc_check(StanBlocks.stan_code(named.model); warn_pedantic=false).ok
    @test RKBRMI(tie_named_model(data)) isa RKBRMI
end

# A response on the event axis observed against the plate result joins it by
# the per-row grouping's labels.
const NESTED_EVENTS = merge(NESTED, (;
    ev_district=NESTED.district_g[[1, 1, 2, 5, 5, 5, 9, 12]],
    ev_x=[0.1, 0.2, -0.1, 0.3, 0.0, 0.2, -0.2, 0.4],
    ev_y=[0.5, 0.4, 0.1, 0.7, 0.2, 0.3, -0.1, 0.6]))
flat_response_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    sigma ~ Exponential(1)
    events = groups(ev_district => district_g)
    @plate for d in eachindex(alpha)
        loc[d] = alpha[d] .+ ev_x[events[d]]
        ev_y[events[d]] ~ Normal(loc[d], sigma)
    end
end

@stestset "an event-axis response joins the plate result by the per-row grouping" begin
    sb = SBBRMI(flat_response_model(NESTED_EVENTS); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_loc] == ND
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
end
