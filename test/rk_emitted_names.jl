# Names in the emitted RK main block say what each value is: a design matrix
# shared by several predictors is named for its columns and read through one
# joint population block, `pop_<columns>`, whose columns the predictors read
# (decision `1pj2p5j`), a grouping's level
# positions are computed once under that grouping's name, a monotonic column
# repeated across predictors is one joint block that each predictor reads by
# column (decision `1pj2p5j`), and the linear predictor of `log(v) ~ …` is
# `log_v`, with `v = exp.(log_v)` its response-scale value on both routes.
#
#     julia --project=test test/rk_emitted_names.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LogExpFunctions: logit

const NAMES_DATA = (;
    subject=repeat(["s1", "s2", "s3", "s4", "s5"]; inner=3),
    x1=[sin(1.3j) for j in 1:15], x2=[cos(0.7j) for j in 1:15],
    rank=[1, 2, 3, 2, 1, 3, 3, 2, 1, 1, 2, 3, 2, 3, 1],
    g1=repeat(["a", "b", "c"], 5), g2=repeat(["c", "a", "b"]; inner=5),
    y=[0.4 * sin(2.1j) + 1.5 for j in 1:15],
    y2=[0.3 * cos(1.7j) - 0.5 for j in 1:15])

emitted_main(brmi) = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi)).main
main_statements(brmi) = filter(s -> !(s isa LineNumberNode), emitted_main(brmi).args)

@stestset "value route: designs, level positions, monotonic effects, linked predictors" begin
    before = deepcopy(NAMES_DATA)
    brmi = @brm NAMES_DATA begin
        log(a) ~ 1 + x1 + x2 + (1 | p | subject)
        log(b) ~ 1 + x1 + x2 + (1 | p | subject)
        c ~ 0 + x1 + mo(rank) + (1 | subject)
        d ~ 0 + x2 + mo(rank)
        e ~ 0 + x2 + mo(rank) + (1 | mm(g1, g2))
        mu = a * b + c + d + e
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    stmts = main_statements(brmi)
    main = string(emitted_main(brmi))
    # One design per column set. `a` and `b` share theirs, so it belongs to
    # neither and is named for its columns; one joint block computes
    # `X * beta_pop` once and each predictor reads its column. `d` and `e`
    # share `hcat(x2)`. `c` alone reads `hcat(x1)` and keeps its own component.
    @test :(X_Intercept_x1_x2 = hcat(ones(length(subject)), x1, x2)) in stmts
    @test occursin("pop_Intercept_x1_x2 ~ brm_joint_population_effects(X_Intercept_x1_x2, 3, 2, 0.0, 1.0)", main)
    @test :(pop_log_a = pop_Intercept_x1_x2[:, 1]) in stmts
    @test :(pop_log_b = pop_Intercept_x1_x2[:, 2]) in stmts
    @test :(X_x2 = hcat(x2)) in stmts
    @test occursin("pop_x2 ~ brm_joint_population_effects(X_x2, 1, 2, 0.0, 1.0)", main)
    @test :(pop_d = pop_x2[:, 1]) in stmts
    @test :(pop_e = pop_x2[:, 2]) in stmts
    @test :(X_c = hcat(x1)) in stmts
    @test occursin("pop_c ~ brm_population_effects(X_c, 1, 0.0, 1.0)", main)
    @test !occursin(r"X_log_a\b|X_log_b\b|X_d\b|X_e\b", main)
    # The `p` and id-less buckets over `subject` read one level position,
    # among the fitted levels of `subject`'s fit-role copy (the training groups
    # on a frozen `reprocess`), a named fitted value of their own.
    @test count("brm_fit_levels(subject_fit)", main) == 1
    @test :(subject_fit_levels = brm_fit_levels(subject_fit)) in stmts
    @test count("brm_level_indices(subject, subject_fit_levels)", main) == 1
    @test :(subject_level = brm_level_indices(subject, subject_fit_levels)) in stmts
    # Each bucket gathers its draws inline, so RK sees the per-row index.
    @test occursin("b_p_subject[subject_level, 1]", main)
    @test occursin("b_subject[subject_level, 1]", main)
    @test !occursin("brm_ranef_column", main)
    @test !occursin("_index_", main)
    # A multi-membership column's positions are into its bucket's joint levels.
    groups = :b_mm__g1__g2_groups
    @test :(b_mm__g1__g2_groups_g1_level = brm_level_indices(g1, $groups)) in stmts
    @test :(b_mm__g1__g2_groups_g2_level = brm_level_indices(g2, $groups)) in stmts
    # `mo(rank)` in three predictors is one joint block, `mo_rank`: a simplex
    # column and a coefficient per predictor, one small level table. Each
    # predictor gathers its own column under its own name.
    @test occursin("mo_rank ~ brm_joint_monotonic_effects([1.0, 1.0], 2, 3, 0.0, 1.0)", main)
    @test :(mo_c_rank = mo_rank[rank_idx, 1]) in stmts
    @test :(mo_d_rank = mo_rank[rank_idx, 2]) in stmts
    @test :(mo_e_rank = mo_rank[rank_idx, 3]) in stmts
    @test !occursin("brm_monotonic_effect(", main)
    @test !occursin(r"mo_rank_\d", main)
    # `log(a) ~ …` sums into `log_a`, and `a` is its value.
    @test :(log_a = pop_log_a .+ ranef_a_p_subject) in stmts
    @test :(a = exp.(log_a)) in stmts
    @test :(b = exp.(log_b)) in stmts
    @test !occursin(r"\b[ab]_\b", main)

    # Coordinates follow the component names.
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    for name in ("pop_Intercept_x1_x2.beta_pop.1.1", "pop_Intercept_x1_x2.beta_pop.3.2",
            "pop_x2.beta_pop.1.2", "pop_c.beta_pop.1",
            "mo_rank.beta.1", "mo_rank.beta.2", "mo_rank.beta.3",
            "mo_rank.simplex_incr.1.1", "mo_rank.simplex_incr.1.2", "mo_rank.simplex_incr.1.3")
        @test Symbol(name) in names
    end
    for u in (zeros(length(names)), [0.3 * sin(3j) for j in eachindex(names)])
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test isfinite(value)
        @test all(isfinite, gradient)
    end
    @test isequal(NAMES_DATA, before)
end

@stestset "monotonic carriers match SBBRMI's" begin
    brmi = @brm NAMES_DATA begin
        mu ~ 1 + x1 + mo(rank) + mo1(rank)
        log(sigma) ~ 1 + x2 + mo(rank)
        y ~ Normal(mu, sigma)
    end
    stmts = main_statements(brmi)
    main = string(emitted_main(brmi))
    # The two `mo(rank)` effects are one block; a linked predictor's column
    # carries its link, as `pop_log_sigma` does. The single `mo1(rank)` keeps
    # its own component and SBBRMI's carrier name.
    @test occursin("mo_rank ~ brm_joint_monotonic_effects([1.0, 1.0], 2, 2, 0.0, 1.0)", main)
    @test :(mo_mu_rank = mo_rank[rank_idx, 1]) in stmts
    @test :(mo_log_sigma_rank = mo_rank[rank_idx, 2]) in stmts
    @test occursin("mo1_rank ~ brm_monotonic_value(rank_idx", main)
    code = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, total_groups=()))
    for carrier in ("mo_rank", "mo1_rank", "mo_log_sigma_rank")
        @test occursin("$(carrier)_simplex_incr ~ dirichlet", code)
    end
end

@stestset "structural route: shared designs and one level position" begin
    brmi = @brm NAMES_DATA begin
        mu ~ 1 + x1 + x2 + (1 | p | subject)
        nu ~ 1 + x1 + x2 + (1 | p | subject)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
        y2 ~ Normal(nu, sigma)
    end
    stmts = main_statements(brmi)
    main = string(emitted_main(brmi))
    @test :(X_Intercept_x1_x2 = hcat(ones(length(subject)), x1, x2)) in stmts
    @test count("hcat(", main) == 1
    @test occursin("pop_Intercept_x1_x2 ~ brm_joint_population_effects(X_Intercept_x1_x2, 3, 2, 0.0, 1.0)", main)
    @test :(pop_mu = pop_Intercept_x1_x2[:, 1]) in stmts
    @test :(pop_nu = pop_Intercept_x1_x2[:, 2]) in stmts
    @test :(subject_level = brm_level_indices(subject, subject_fit_levels)) in stmts
end

@stestset "structural route: a linked predictor is its response-scale value" begin
    data = (; subject=NAMES_DATA.subject, x1=NAMES_DATA.x1, x2=NAMES_DATA.x2,
        count=[mod(3j, 7) for j in 1:15], hit=[isodd(j) ? 1 : 0 for j in 1:15],
        w=[mod(2j, 5) for j in 1:15])
    before = deepcopy(data)
    brmi = @brm data begin
        log(mu) ~ 1 + x1 + (1 + x1 | subject)
        log(shape) ~ 1 + x2
        count ~ NegativeBinomial2(mu, shape)
        logit(p) ~ 1 + x2
        hit ~ Bernoulli(p)
        log(lam) ~ 1
        w ~ Poisson(lam)
    end
    stmts = main_statements(brmi)
    main = string(emitted_main(brmi))
    # The linear predictors carry their link, as authored; the responses read
    # them through their link-scale families, and the authored names are the
    # response-scale values (snag `rk-node-of-a-lin-d8216a61`).
    @test :(log_mu = pop_log_mu .+ ranef_mu_subject) in stmts
    @test :(mu = exp.(log_mu)) in stmts
    @test :(shape = exp.(log_shape)) in stmts
    @test :(p = logistic.(logit_p)) in stmts
    @test :(log_lam = lam_Intercept) in stmts
    @test :(lam = exp.(log_lam)) in stmts
    @test occursin("count .~ NegativeBinomial2Log.(log_mu, exp.(log_shape))", main)
    @test occursin("hit .~ BernoulliLogit.(logit_p)", main)
    @test occursin("w .~ PoissonLog.(log_lam)", main)

    # Queried by the authored names, the values are on the response scale, and
    # they are the values the likelihood reads.
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    likelihood = BRM._rk_likelihood_reader(backend)
    for u in ([0.3 * sin(3j) for j in eachindex(names)],
            collect(range(-0.6, 0.7; length=length(names))))
        v = map(only, rk_draw_values(backend, reshape(u, :, 1),
            (:mu, :log_mu, :shape, :log_shape, :p, :logit_p, :lam, :log_lam)))
        @test v.mu ≈ exp.(v.log_mu)
        @test v.shape ≈ exp.(v.log_shape)
        @test v.p ≈ LogExpFunctions.logistic.(v.logit_p)
        @test v.lam ≈ exp(v.log_lam)
        @test all(>(0), v.mu)
        oracle = sum(logpdf.(NegativeBinomial.(v.shape, v.shape ./ (v.mu .+ v.shape)),
                data.count)) +
            sum(logpdf.(Bernoulli.(v.p), data.hit)) + sum(logpdf.(Poisson(v.lam), data.w))
        @test likelihood(u) ≈ oracle atol=1e-10 rtol=1e-12
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test isfinite(value)
        @test all(isfinite, gradient)
    end
    @test isequal(data, before)
end
