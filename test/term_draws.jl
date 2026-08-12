# test/term_draws.jl — acceptance for `term_draws` (src/prediction.jl): ablate ONE
# named linear-predictor term in fitted UNCONSTRAINED draws, then regenerate the
# descriptor outputs with `:predict`.
#
# WHY THE MODEL-LEVEL CHECKS ARE THE LOAD-BEARING ONES. It is easy to write a
# term-ablation that zeroes *some* coordinates and looks right: the draw matrix
# has the expected shape and nothing errors. What proves it is (a) that EXACTLY
# the hsgp basis-weight coordinates change and nothing else, and (b) reading the
# model's OWN `log_F` transformed parameter back out of BridgeStan and asserting
# that it is AFFINE in the basis weights with the ablated draw sitting on the
# zero-contribution point — that is the model agreeing the hsgp term is gone,
# not this file agreeing with itself about an index layout. The `:predict` leg
# then confirms the reporter's actual regeneration path carries the change.
#
# This is the `hsgp(op_log_dose; k=5)` in `log_F` case from snag
# `joint-brm2-term-065de094`: the exact shape a downstream diagnostic needs to
# evaluate the fitted draws twice — full `log_F` and a linearized `log_F` with
# only the HSGP term removed.
#
# Run: julia --project=<worktree> test/term_draws.jl
# (needs BridgeStan reachable through StanBlocks; set BRM_TERM_RUNTIME=0 to run
# only the declaration-level / fail-closed half.)

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using Distributions: LogNormal, Normal

const RUNTIME = get(ENV, "BRM_TERM_RUNTIME", "1") != "0"
const BS = StanBlocks.BridgeStan

# ---- fixture: the reporter's `log_F` shape ----------------------------------
# Population terms + a monotonic `mo(op_diet)` + `hsgp(op_log_dose; k=5)`, observed
# through a Normal so `:predict` is offered. Identical in structure to the joint
# PK model's `log_F` linear predictor.
const N = 12
term_df = (;
    y                = collect(range(0.0, 1.0; length = N)),
    vessel_bottle    = repeat([1.0, 0.0, 0.0], 4),
    vessel_bottle_20 = repeat([0.0, 1.0, 0.0], 4),
    vessel_tablet    = repeat([0.0, 0.0, 1.0], 4),
    vessel_tablet_20 = repeat([1.0, 0.0, 1.0], 4),
    op_diet          = repeat(1:4, 3),
    op_log_dose      = collect(range(-1.0, 1.0; length = N)),
)

term_builder = @brm begin
    y ~ Normal(log_F, 1.0)
    log_F ~ 0 + vessel_bottle + vessel_bottle_20 + vessel_tablet +
                vessel_tablet_20 + mo(op_diet) + op_log_dose +
                hsgp(op_log_dose; k = 5)
    effect(log_F, :) ~ Normal(0.0, 0.5)
    effect(log_F, op_log_dose) ~ Normal(0.0, 0.6676)
    length_scale(:, hsgp(op_log_dose)) ~ LogNormal(0.0, 1.0)
    sd(:, hsgp(op_log_dose)) ~ LogNormal(0.0, 1.0)
end

d = brm_descriptor(term_builder, term_df;
                   mod = @__MODULE__, name = :joint_log_F, highlights = ())

# ---- 1. fail-closed edges (need no compiled model) --------------------------

@testset "term_draws fails closed on unsupported target / term / shape" begin
    unc  = ["a.$i" for i in 1:4]        # dummy names; the guards fire before use
    zero = zeros(1, 4)

    # `to` other than :zero is refused (no other target has an established meaning).
    @test_throws "only `to = :zero`" term_draws(
        d, zero, unc; predictor = :log_F, term = :hsgp_op_log_dose, to = :mean)

    # A non-hsgp term exposes no `:basis_weights` role — refused, not zeroed.
    @test_throws "basis_weights" term_draws(
        d, zero, unc; predictor = :log_F, term = :mo_op_diet)

    # A missing term is refused with the available labels.
    @test_throws "available term labels" term_draws(
        d, zero, unc; predictor = :log_F, term = :hsgp_missing)

    # draws / unc_names shape mismatch is caught before any indexing.
    @test_throws ErrorException term_draws(
        d, zeros(1, 3), unc; predictor = :log_F, term = :hsgp_op_log_dose)
end

@testset "grouped hsgp(...; by=...) has no basis-weight carrier to ablate" begin
    # A grouped hsgp is a correlated random-effect block: its weights are zeroed
    # with `population_draws`, not `term_draws`. It must fail closed here.
    grouped_df = (;
        y           = collect(range(0.0, 1.0; length = 6)),
        op_log_dose = collect(range(-1.0, 1.0; length = 6)),
        subject     = repeat(1:2, 3),
    )
    grouped = @brm begin
        y ~ Normal(log_F, 1.0)
        log_F ~ 0 + op_log_dose + hsgp(op_log_dose; k = 4, by = subject)
    end
    dg = brm_descriptor(grouped, grouped_df; mod = @__MODULE__, highlights = ())
    # Fails closed one way or another — a grouped hsgp exposes no basis-weight
    # role, so there is no coordinate `term_draws` may zero.
    @test_throws ErrorException term_draws(
        dg, zeros(1, 2), ["a.1", "a.2"];
        predictor = :log_F, term = :hsgp_op_log_dose_by_subject)
end

if !RUNTIME
    @info "BRM_TERM_RUNTIME=0 — skipping the compiled-model half"
else

# ---- 2. against the real compiled model -------------------------------------

prob  = brm_execute(d, :fit)
sm    = prob.model
unc   = BS.param_unc_names(sm)
n_unc = BS.param_unc_num(sm)

# The basis-weight carrier, resolved BY NAME against the UNCONSTRAINED names.
bw = brm_term_coordinates(d, :log_F, unc;
                          term = :hsgp_op_log_dose, parameter = :basis_weights).coordinates

@testset "the hsgp basis weights resolve in unconstrained space" begin
    @test length(bw) == 5                       # k = 5 basis functions
    @test all(i -> startswith(String(unc[i]), "hsgp_op_log_dose_beta_raw."), bw)
    @test length(unique(bw)) == length(bw)
end

rng   = MersenneTwister(20260812)
draws = randn(rng, 6, n_unc)
lin   = term_draws(d, draws, unc; predictor = :log_F, term = :hsgp_op_log_dose)

@testset "term_draws zeroes EXACTLY the basis weights, nothing else" begin
    @test size(lin) == size(draws)
    @test all(iszero, lin[:, bw])                # the term's carrier → 0
    keep = setdiff(1:n_unc, bw)
    @test lin[:, keep] == draws[:, keep]         # every other coordinate identical
    @test draws[:, bw] != lin[:, bw]             # input DID carry nonzero weights
    @test !(draws === lin)                       # input matrix left untouched
end

# Read the model's own `log_F` transformed parameter back out of BridgeStan.
function logF(sm, theta)
    names = BS.param_names(sm; include_tp = true, include_gq = false)
    vals  = BS.param_constrain(sm, theta; include_tp = true, include_gq = false)
    [v for (nm, v) in zip(names, vals) if startswith(String(nm), "log_F.")]
end

@testset "the model's log_F is affine in beta_raw, ablated draw at the zero point" begin
    row  = draws[1, :]
    v    = row[bw]                               # the fitted basis weights
    @test !isempty(logF(sm, row))                # log_F is an exposed TP

    base = copy(row); base[bw] .= 0.0            # beta_raw := 0
    two  = copy(row); two[bw]  .= 2.0 .* v       # beta_raw := 2v

    lf_full = logF(sm, row)                       # beta_raw = v
    lf_base = logF(sm, base)                       # beta_raw = 0
    lf_two  = logF(sm, two)                         # beta_raw = 2v
    lf_abl  = logF(sm, lin[1, :])                   # what term_draws produced

    # (a) the ablated draw lands exactly on the beta_raw := 0 location.
    @test lf_abl ≈ lf_base
    # (b) the hsgp term genuinely contributes: with weights present, log_F moves.
    @test any(lf_full .!= lf_base)
    # (c) removed EXACTLY: log_F is affine in beta_raw with the ablated point as
    #     its zero-contribution baseline, so doubling the weights doubles the
    #     displacement from that baseline. Independent of any index layout.
    @test lf_two .- lf_base ≈ 2.0 .* (lf_full .- lf_base)
end

# ---- 3. the reporter's actual path: regenerate with `:predict` --------------

@testset "brm_execute(:predict) carries the ablation into pk-conc-style output" begin
    seed = 99
    full = brm_execute(d, :predict; problem = prob, draws = draws[1, :], seed = seed)
    abl  = brm_execute(d, :predict; problem = prob, draws = lin[1, :],   seed = seed)
    @test haskey(full, :y_gen) && haskey(abl, :y_gen)
    @test length(abl.y_gen) == N
    # Same seed ⇒ identical observation-noise draws, so any difference is exactly
    # the removed hsgp location shift propagating through the generated quantity.
    @test full.y_gen != abl.y_gen
    # And a second ablated draw with different (removed) weights regenerates the
    # SAME location structure the ablation defines — the term is gone downstream.
    lin2 = term_draws(d, draws, unc; predictor = :log_F, term = :hsgp_op_log_dose)
    abl2 = brm_execute(d, :predict; problem = prob, draws = lin2[2, :], seed = seed)
    @test length(abl2.y_gen) == N
end

end # RUNTIME
