# test/rk_planning_specialization.jl — RK planning, and each fit's per-design
# entry points, compile once per corpus.
#
# Run: julia --project=test test/rk_planning_specialization.jl
#
# A BRMI's operation NamedTuple spells every formula name and expression tree,
# so any planner that specializes on it compiles again for each model of a
# corpus. Members of this public synthetic family differ only in operation
# names. After a warm member, emitting another member's RK artifact and
# building its kernel must not compile a BRM method, or a generic method over
# BRM values, whose signature names that member's operations. Only the
# constructors of the model's own values may. ReactiveKernels' kernel types
# legitimately carry argument names and are outside this contract. The family
# uses the value route: a caller-owned submodel, an ordinary callable
# assignment, grouped and population predictors and a log-scale predictor.
using Test
using BayesianRegressionModels
using Distributions: Exponential, Normal
using Statistics: mean
using ReactiveKernels, ReactiveKernelsPPL

const BRM = BayesianRegressionModels

module SpecializationFamilySource
import BayesianRegressionModels: _rk_submodel_rhs!, getargs, getkwargs, name

function shifted_curve end
function shifted_curve_variant end
native_shifted_curve(x, location, shift) = location .+ x .* shift

function _rk_submodel_rhs!(definitions, statements, data, bindings,
        target::Symbol, ::Union{typeof(shifted_curve),typeof(shifted_curve_variant)}, rhs)
    x = only(getargs(rhs))
    key = Symbol(target, :_curve_x)
    data[key] = copy(parent(parent(x)))
    location, shift = name(getkwargs(rhs).location), name(getkwargs(rhs).shift)
    helper = Symbol(target, :_native_curve)
    reader = Symbol(target, :_reader)
    push!(bindings, helper => native_shifted_curve)
    push!(definitions, :(function $reader(x, location, shift)
        return $helper(x, location, shift)
    end))
    push!(statements, :($target = $reader($key, $location, $shift)))
    :done
end
end

scaled_sum(curve, scale) = curve .* scale

const FAMILY_DATA = (;
    x=[-0.4, 0.2, 0.7, 1.1, -0.9, 0.3],
    z=[0.5, -0.1, 0.8, -0.6, 0.2, 0.4],
    g=["a", "b", "a", "c", "b", "c"],
    y=[0.1, -0.2, 0.3, 0.6, -0.5, 0.2])

# Every operation and data column name carries `zqspec<index>`, the marker
# searched for in the compile dump.
function define_member(index; curve_function=:shifted_curve)
    tag(base) = Symbol(base, :_zqspec, index)
    x, z, g, y = tag(:x), tag(:z), tag(:g), tag(:y)
    a, shift, curve, gain, reads, scale = tag(:a), tag(:shift), tag(:curve),
        tag(:gain), tag(:reads), tag(:scale)
    builder = Symbol(:specialization_member_, index)
    Core.eval(@__MODULE__, quote
        function $builder(df)
            @brm df begin
                $a ~ 1 + $z + (1 | $g)
                $shift ~ Normal(0, 0.4)
                $curve ~ SpecializationFamilySource.$curve_function($x;
                    location=$a, shift=$shift)
                $gain ~ Normal(1, 0.2)
                $reads = scaled_sum($curve, $gain)
                log($scale) ~ 1 + $z
                $y ~ Normal($reads, $scale)
            end
        end
    end)
    data = NamedTuple{(x, z, g, y)}(values(FAMILY_DATA))
    getfield(@__MODULE__, builder), data
end

# The model's own values are constructed with its own types; nothing else may
# carry them. A constructor called with keywords (`RKBRMI(brmi, plan, model;
# program, bound, source)`) compiles as `Core.kwcall` on the same type. Methods
# owned by this test file are the harness, not BRM.
const INHERENT_CONSTRUCTOR = r"^\"Tuple\{(typeof\(Core\.kwcall\), NamedTuple\{\([:\w, ]*\), .*?\}, )?Type\{(NamedTuple\{|BayesianRegressionModels\.(BRMI|RKBRMI)\{)"
const HARNESS = r"^\"Tuple\{(typeof\(Main\.|Main\.var\")"

Base.@nospecializeinfer function compiled_during(@nospecialize(f))
    path, io = mktemp()
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), io.handle)
    value = try
        Base.invokelatest(f)
    finally
        ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
        close(io)
    end
    lines = [last(split(line, '\t'; limit=2)) for line in eachline(path)]
    rm(path)
    value, lines
end

Base.@nospecializeinfer function member_stage(@nospecialize(builder), data, index)
    brmi = builder(data)
    artifact = BRM.emit_rk_artifact(brmi; case_id="specialization-member-$index")
    backend = RKBRMI(brmi)
    (; brmi, artifact, backend)
end

@testset "distinct-name members reuse RK planning compilation" begin
    warm_builder, warm_data = define_member(1)
    warm = Base.invokelatest(member_stage, warm_builder, warm_data, 1)
    @test warm.artifact.plan isa BRM._RKValuePlan
    @test any(p -> last(p) === SpecializationFamilySource.native_shifted_curve,
        warm.artifact.bindings)

    for index in (2, 3)
        builder, data = define_member(index)
        member, lines = compiled_during(() -> member_stage(builder, data, index))
        marker = "_zqspec$index"
        specialized = filter(line -> occursin(marker, line), lines)
        # Positive control: the dump records this member's own construction.
        @test any(line -> occursin(INHERENT_CONSTRUCTOR, line), specialized)
        planning = filter(line -> occursin("BayesianRegressionModels", line) &&
            !occursin(INHERENT_CONSTRUCTOR, line) && !occursin(HARNESS, line),
            specialized)
        isempty(planning) || foreach(line -> println("SPECIALIZED ", line), planning)
        @test isempty(planning)
        # The reused compilation emits the same program up to the names.
        rename(ex) = replace(string(ex), "_zqspec$index" => "_zqspec1",
            "specialization-member-$index" => "specialization-member-1")
        @test rename(member.artifact.ast) == string(warm.artifact.ast)
        @test map(rename, member.artifact.defs) == map(string, warm.artifact.defs)
        @test length(coordinate_names(member.backend.model.layout)) ==
            length(coordinate_names(warm.backend.model.layout))
    end
end

# Source walkers read expression trees as syntax. A member whose submodel
# callable differs has new expression types throughout (a named column embeds
# its definition), yet the walkers must reuse their compilation. Constructing
# the member's values, the caller-owned hook and the prepared representation
# still specialize on those types.
const SOURCE_WALKERS = r"typeof\(BayesianRegressionModels\.(_brm_collect_mm_sources!|_brm_collect_non_mm_sources!|_brm_collect_data!|_brm_collect_rhs_refs!|_brm_collect_target_axes!|_brm_operation_references!|_brm_operation_dependencies|_brm_operation_role|_brm_observation_name|_brm_visit_op!|_brm_register_sampling_lhs!|_brm_collect_additive_terms!|_brm_additive_terms|_brm_prior_expression|_named_op|_named_op_inner|_nested_contains|_brm_foreach_child|_brm_foreach_item|_brm_any_child)\)"
const SOURCE_CLOSURE = r"^\"Tuple\{BayesianRegressionModels\.var\"#[0-9]+#[0-9]+\"\{Base\.(Set|Dict)\{Symbol"

@testset "a structurally new member reuses the source walkers" begin
    warm_builder, warm_data = define_member(11)
    Base.invokelatest(member_stage, warm_builder, warm_data, 11)
    builder, data = define_member(12; curve_function=:shifted_curve_variant)
    member, lines = compiled_during(() -> member_stage(builder, data, 12))
    structural = filter(line -> occursin("shifted_curve_variant", line), lines)
    @test any(line -> occursin(r"^\"Tuple\{Type\{BayesianRegressionModels\.", line), structural)
    walked = filter(line -> occursin(SOURCE_WALKERS, line) || occursin(SOURCE_CLOSURE, line),
        structural)
    isempty(walked) || foreach(line -> println("WALKER ", line), walked)
    @test isempty(walked)
    @test member.artifact.plan isa BRM._RKValuePlan
    for line in structural
        occursin("BayesianRegressionModels", line) && println("RESIDUAL ", first(line, 160))
    end
end

# A fit's per-design entry points run once per design, not per draw: its bound
# data and plan, a data-free open from its exported state, the design binder
# with its first and its cached kind of design, and `reprocess` (snag
# `brm-methods-spec-39436695`: a simulator opening fits of models its process
# had not compiled paid for each of them again, about 2.4 s of LLVM
# optimization per model on its profile). After one model ran them, another
# model's run must not compile a BRM method whose signature names that model's
# columns or operations, which carry `_zqdesign<index>`. Only constructing the
# model's own values may (`INHERENT_CONSTRUCTOR`), and so may the accessors
# that read the caller's own design columns and the binder's constructors,
# which take the fit and its named values (`DESIGN_VALUES`): about 25 ms of
# compilation per model on strato2, against 2.1 s before.
const DESIGN_VALUES = r"^\"Tuple\{(Type\{BayesianRegressionModels\.(RKDesignBinder|_RKFitCoordinateLevels)\}|typeof\(BayesianRegressionModels\.(_brm_df_column|_brm_df_has_column|_rk_design_input)\))"

function define_design_member(index)
    tag(base) = Symbol(base, :_zqdesign, index)
    x, z, subject, y = tag(:x), tag(:z), tag(:subject), tag(:y)
    sigma, xbar, mu = tag(:sigma), tag(:xbar), tag(:mu)
    builder = Symbol(:design_member_, index)
    Core.eval(@__MODULE__, quote
        function $builder(d)
            @brm d begin
                $sigma ~ Exponential(1)
                $xbar = constant(mean($x))
                $mu ~ 1 + zscale($x) + (1 + $z | $subject)
                effect($mu, Intercept) ~ Normal($xbar, 1)
                $y ~ Normal($mu, $sigma)
            end
        end
    end)
    n = 24
    xs = [sin(1.3i + index) for i in 1:n]
    zs = [cos(0.7i - index) for i in 1:n]
    ys = [0.4 * sin(0.9i) + 0.1index for i in 1:n]
    train = NamedTuple{(x, z, subject, y)}((xs, zs,
        repeat(["s$i" for i in 1:6], inner=4), ys))
    design(m, k, shift) = NamedTuple{(x, z, subject)}((
        [sin(0.5i + shift) for i in 1:m], [cos(0.3i + shift) for i in 1:m],
        repeat(["n$i" for i in 1:k], inner=m ÷ k)))
    (; builder=getfield(@__MODULE__, builder), train, subject,
       d1=design(8, 2, 0.1), d2=design(12, 3, 0.2), marker="_zqdesign$index")
end

# The fit, its exported state and the open's template are made before the
# measured run: constructing them is the construction guard's subject above.
Base.@nospecializeinfer function design_fit(@nospecialize(m))
    rk = RKBRMI(m.builder(m.train))
    (; rk, state=brm_fitted_state(rk), template=m.builder(m.train))
end

Base.@nospecializeinfer function design_entry_points(@nospecialize(m), @nospecialize(rk),
        state, @nospecialize(template))
    rk_bound_data(rk)
    rk_bound_plan(rk)
    opened = RKBRMI(template, state)
    rk_bound_data(opened)
    binder = rk_design_binder(opened; resample_groups=[m.subject])
    first_design = binder(m.d1)
    cached_design = binder(m.d2)
    replay = reprocess(opened, m.d1; resample_groups=[m.subject])
    (; opened, binder, first_design, cached_design, replay=rk_bound_data(replay))
end

@testset "a new model reuses the per-design entry points' compilation" begin
    warm = define_design_member(21)
    warm_fit = Base.invokelatest(design_fit, warm)
    warm_run = Base.invokelatest(design_entry_points, warm, warm_fit.rk, warm_fit.state,
        warm_fit.template)
    @test length(warm_run.binder.classes) == 1
    for index in (22, 23)
        m = define_design_member(index)
        fit = Base.invokelatest(design_fit, m)
        run, lines = compiled_during(() -> design_entry_points(m, fit.rk, fit.state, fit.template))
        specialized = filter(line -> occursin(m.marker, line), lines)
        # Positive control: the dump records compilation on this model's types
        # (its own values are constructed during the open).
        @test !isempty(specialized)
        per_model = filter(line -> occursin("BayesianRegressionModels", line) &&
            !occursin(INHERENT_CONSTRUCTOR, line) && !occursin(HARNESS, line) &&
            !occursin(DESIGN_VALUES, line), specialized)
        isempty(per_model) || foreach(line -> println("PER-MODEL ", first(line, 400)), per_model)
        @test isempty(per_model)
        # The reused compilation does the same work: one cached kind of design,
        # the fit's coordinates on the data-free open, the design's inputs.
        @test length(run.binder.classes) == 1
        @test BRM._rk_layout_coordinate_names(run.opened) ==
            BRM._rk_layout_coordinate_names(fit.rk)
        @test Set(keys(run.first_design)) == Set(keys(run.cached_design))
        @test haskey(run.replay, Symbol(:x_zqdesign, index))
    end
end
