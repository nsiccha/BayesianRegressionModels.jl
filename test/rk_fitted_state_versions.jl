# test/rk_fitted_state_versions.jl — a fitted state exported by an earlier BRM
# replays on this one when an emission change only rewrote how the program reads
# (snag `simulator-bundle-fa889214`: a downstream simulator's saved fits stopped loading).
#
# A data-free replay (`RKBRMI(builder(design), state)`) compares the program the
# state stored with the program this BRM emits for the design, statement by
# statement: the fit's draws apply only where the program is the fit's. The
# stored text is the exporting BRM's, so an emission change that only rewrote a
# statement's spelling refused every state exported before it, with an error
# that blamed the design. 0.2.270 passes a ragged join's columns as one tuple
# (`brm_ragged_rows(names, group, cell, (a,))` for `…, cell, a)`), which refused
# every state of a model with a grouped read exported by 0.2.269 or earlier.
# `_rk_stored_statement` (src/rk_reprocess.jl) writes a stored statement in this
# BRM's form, one rewrite per such change.
#
# `test/fitted_state_programs/<version>.txt` holds the fitted-state program of
# each model of `fitted_state_version_models.jl` as that released version
# exported it: 0.2.264 (before the tuple) and 0.2.272. Oracle: each, rewritten,
# is this BRM's program, and a state carrying the 0.2.264 program replays as
# this BRM's own export does (bound inputs, coordinates, binder inputs, re-export).
#
# When an emission change fails this test: if it changes only how a statement
# reads, not what the program computes, add its rewrite to
# `_rk_stored_statement` and keep the stored files, so states exported before it
# keep replaying. If it changes what the program computes, states exported
# before it must be exported again, and the replay refuses them by their first
# differing statement, naming this version: write the new version's file
# (`BRM_WRITE_FITTED_PROGRAMS=1`), remove the files that no longer read as this
# BRM's program, and say so in CHANGELOG.md. A change of what a fitted value
# means that leaves the program text alone bumps `_BRM_FITTED_STATE_FORMAT`.
#
# Run: julia --startup-file=no --project=test test/rk_fitted_state_versions.jl

using Test, BayesianRegressionModels, Distributions, Statistics
using ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))
include(joinpath(@__DIR__, "fitted_state_version_models.jl"))
const VM = FittedStateVersionModels
const PROGRAMS = joinpath(@__DIR__, "fitted_state_programs")

get(ENV, "BRM_WRITE_FITTED_PROGRAMS", "") == "1" &&
    println("wrote ", write_fitted_programs(joinpath(PROGRAMS, "$(pkgversion(BRM)).txt")))

const STORED = Dict(VersionNumber(first(splitext(file))) => read_fitted_programs(joinpath(PROGRAMS, file))
    for file in readdir(PROGRAMS) if endswith(file, ".txt"))
const RAGGED = (:flat_pk, :flat_pk_two, :flat_pk_ordered)

fitted(name) = let (build, data) = VM.MODELS[name]
    brm_fitted_state(RKBRMI(build(data)))
end
# A state as an earlier version exported it: this BRM's export with that
# version's program.
with_program(state, program) = BRM.BRMFittedState(map(f -> f === :program ? program :
    getfield(state, f), fieldnames(BRM.BRMFittedState))...)
# The rows of the subjects `labels` (renamed to `as`) of a grouped-read design.
function subjects(data, labels; as=labels)
    rename = Dict(zip(labels, as))
    unit, dose, obs = (findall(in(labels), data[by]) for by in (:subject, :dose_subject, :obs_subject))
    picked = Dict{Symbol,Any}()
    for (name, value) in pairs(data)
        picked[name] = name === :subject ? [rename[s] for s in value[unit]] :
            name === :dose_subject ? [rename[s] for s in value[dose]] :
            name === :obs_subject ? [rename[s] for s in value[obs]] :
            length(value) == length(data.subject) ? value[unit] :
            length(value) == length(data.dose_subject) ? value[dose] : value[obs]
    end
    NamedTuple{keys(data)}(Tuple(picked[name] for name in keys(data)))
end
function refusal(f)
    try
        f(); nothing
    catch e
        e
    end
end

@stestset "a stored program reads as this BRM's" begin
    # Positive control: both released versions are stored, for every model.
    @test issubset((v"0.2.264", v"0.2.272"), keys(STORED))
    for (version, programs) in STORED
        @test sort!(collect(keys(programs))) == sort!(collect(keys(VM.MODELS)))
        for (name, program) in programs
            current = fitted(name).program
            migrated = map(BRM._rk_stored_statement, program)
            same = migrated == current
            same || @info "the program $version stored for $name differs" first_difference=something(
                findfirst(i -> get(migrated, i, nothing) != get(current, i, nothing),
                    1:max(length(migrated), length(current))), 0)
            @test same
            # This BRM's own statements are already in its form.
            @test map(BRM._rk_stored_statement, current) == current
        end
    end
end

@stestset "0.2.264 differs from this BRM only in its ragged joins" begin
    old = STORED[v"0.2.264"]
    for (name, program) in old
        current = fitted(name).program
        @test length(program) == length(current)
        differing = [stored for (stored, now) in zip(program, current) if stored != now]
        if name in RAGGED
            # Negative control: the stored text is not this BRM's, and every
            # difference is a join's partition, as the earlier helper read it.
            @test !isempty(differing)
            @test all(s -> occursin(r"^statement: \w+ = brm_ragged_rows(_ordered|_shared)?\(", s),
                differing)
            @test !any(s -> occursin(r", \([^()]*\)\)$", s), differing)
        else
            @test isempty(differing)
        end
    end
    @test any(s -> occursin("brm_ragged_rows_shared(", s), old[:flat_pk_two])
    @test any(s -> occursin("brm_ragged_rows_ordered(", s), old[:flat_pk_ordered])
end

@stestset "a state exported by 0.2.264 replays as this BRM's own export" begin
    for name in RAGGED
        build, data = VM.MODELS[name]
        state = fitted(name)
        old = with_program(state, STORED[v"0.2.264"][name])
        @test old.program != state.program
        for design in (data, subjects(data, ["s1", "s2"]))
            a, b = RKBRMI(build(design), old), RKBRMI(build(design), state)
            @test isequal(rk_bound_data(a), rk_bound_data(b))
            @test BRM._rk_layout_coordinate_names(a) == state.coordinates
            # Its re-export is this BRM's state.
            @test brm_fitted_state(a).program == state.program
        end
        # New subjects, through the binder of the replayed backend.
        newcomers = subjects(data, ["s1", "s3"]; as=["n1", "n2"])
        a = rk_design_binder(RKBRMI(build(data), old); resample_groups=[:subject])
        b = rk_design_binder(RKBRMI(build(data), state); resample_groups=[:subject])
        @test isequal(a(newcomers), b(newcomers))
        # Without the rewrite it is refused, and the refusal names this BRM.
        e = refusal(() -> BRM._rk_check_replay_program(map(BRM._rk_unlocated_statement, old.program),
            state.program; prefix="RK data-free replay", exported=true))
        @test e isa ErrorException
        @test occursin("changes the fitted RK program, or the fitted state was exported by", e.msg)
        @test occursin("($(pkgversion(BRM)))", e.msg)
        @test occursin("export the state again from the fit", e.msg)
    end
end

@stestset "a ragged join's columns become one tuple" begin
    rewrite = BRM._rk_stored_statement
    @test rewrite("statement: r_2 = brm_ragged_rows([:a, :g, :c, :a], g, c, a)") ==
        "statement: r_2 = brm_ragged_rows([:a, :g, :c, :a], g, c, (a,))"
    @test rewrite("statement: r_3 = brm_ragged_rows_ordered([:a, :g, :c, :o, :a, :b], g, c, o, a, b)") ==
        "statement: r_3 = brm_ragged_rows_ordered([:a, :g, :c, :o, :a, :b], g, c, o, (a, b))"
    @test rewrite("statement: r_4 = brm_ragged_rows_shared(r_2, [:b, :g, :b], g, b)") ==
        "statement: r_4 = brm_ragged_rows_shared(r_2, [:b, :g, :b], g, (b,))"
    @test rewrite("statement: r_5 = brm_ragged_rows([:a, :g, :c], g, c)") ==
        "statement: r_5 = brm_ragged_rows([:a, :g, :c], g, c, ())"
    # This BRM's form, other statements and text that does not parse are kept.
    for kept in ("statement: r_2 = brm_ragged_rows([:a, :g, :c, :a], g, c, (a,))",
            "statement: r_2 = brm_ragged_rows_shared(r_1, [:a, :g, :a], g, (a,))",
            "statement: y_mean = brm_covariate_mean(y_fit)",
            "definition: r = brm_ragged_rows([:a, :g, :c, :a], g, c, a)",
            "statement: r = brm_ragged_rows(")
        @test rewrite(kept) == kept
    end
    # The 0.2.228 rewrite still applies.
    @test rewrite("definition: #= /fleet/job-a/checkout/src/kernels.jl:12 =# ReactiveKernels.@kernel(f(x) = x)") ==
        "definition: #= line 0 =# ReactiveKernels.@kernel(f(x) = x)"
end

@stestset "a stored program that differs in what it computes is still refused" begin
    build, data = VM.MODELS[:zscaled]
    state = fitted(:zscaled)
    program = copy(state.program)
    k = findfirst(startswith("statement: "), program)
    program[k] *= " # another law"
    changed = with_program(state, program)
    e = refusal(() -> RKBRMI(build(data), changed))
    @test e isa ErrorException
    @test occursin("RK data-free replay: the new data changes the fitted RK program", e.msg)
    @test occursin("# another law", e.msg)
    @test occursin("BayesianRegressionModels version that emits this model differently from this one ($(pkgversion(BRM)))", e.msg)
    # `reprocess` on the training fit keeps its own advice.
    e = refusal(() -> BRM._rk_check_replay_program(["statement: a"], ["statement: b"]; prefix="RK reprocess"))
    @test occursin("rebuild with `freeze_constants=false`", e.msg)
    @test !occursin("exported by", e.msg)
end
