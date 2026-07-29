module HistoricalInventoryGallery

using DynamicObjects
using HTMXObjects
import HTMXObjects: h, Raw

export InventoryRow, HistoricalInventory, HistoricalInventoryGalleryApp,
       DEFAULT_MATRIX_PATH, build_gallery, load_matrix,
       row_card, semantic_inventory, filtered_rows

const DEFAULT_MATRIX_PATH = normpath(joinpath(@__DIR__, "..", "model_matrix.tsv"))
const SBPMX_REFERENCE = "SbPMX origin/main@a8fd3c02e1e2305683a56c91384701bfeba6b528"

"""One source row, kept losslessly as the strings recorded in `model_matrix.tsv`."""
struct InventoryRow
    line::Int
    values::Dict{String,String}
end

Base.getindex(row::InventoryRow, key::AbstractString) = get(row.values, String(key), "")

"""
Read the authoritative inventory matrix at runtime.

The inventory is deliberately not compiled into this module. Empty fields and
all 73 recorded columns remain distinguishable, and malformed row widths fail
closed instead of shifting evidence into the wrong fields.
"""
function load_matrix(path::AbstractString=DEFAULT_MATRIX_PATH)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("inventory matrix is empty: $path"))
    header = split(first(lines), '\t'; keepempty=true)
    length(unique(header)) == length(header) ||
        throw(ArgumentError("inventory matrix has duplicate column names: $path"))

    rows = InventoryRow[]
    for (offset, line) in enumerate(Iterators.drop(lines, 1))
        line_number = offset + 1
        values = split(line, '\t'; keepempty=true)
        length(values) == length(header) || throw(ArgumentError(
            "inventory matrix row $line_number has $(length(values)) fields; " *
            "expected $(length(header))",
        ))
        push!(rows, InventoryRow(line_number, Dict(zip(header, values))))
    end
    rows
end

function _label(value::AbstractString)
    labels = Dict(
        "all" => "All",
        "ordinary_brm" => "Ordinary BRM",
        "stanblocks_plate" => "StanBlocks plate",
        "brm_kernel" => "BRM kernel",
        "unresolved-family" => "Unresolved family",
        "bridgestan-finite-density-gradient" => "BridgeStan finite density + gradient",
        "dead-source" => "Dead source",
        "adapted-but-defensible" => "Adapted but defensible",
    )
    get(labels, value, uppercasefirst(replace(value, '_' => ' ', '-' => ' ')))
end

"""A finite labelled domain whose values remain the exact matrix strings."""
function filter_domain(rows::AbstractVector{InventoryRow}, column::AbstractString)
    counts = Dict{String,Int}()
    for row in rows
        value = row[column]
        counts[value] = get(counts, value, 0) + 1
    end
    records = Pair{String,String}["all" => "All ($(length(rows)))"]
    for value in sort!(collect(keys(counts)); by=v -> (_label(v), v))
        push!(records, value => "$(_label(value)) ($(counts[value]))")
    end
    option_domain(records)
end

_selected(row::InventoryRow, column::AbstractString, selected::AbstractString) =
    selected == "all" || row[column] == selected

function filter_inventory(
    rows::AbstractVector{InventoryRow};
    source_fidelity::AbstractString="all",
    family_provenance::AbstractString="all",
    translation_route::AbstractString="all",
    validation_tier::AbstractString="all",
)
    InventoryRow[
        row for row in rows
        if _selected(row, "source_fidelity_verdict", source_fidelity) &&
           _selected(row, "inferred_family_provenance", family_provenance) &&
           _selected(row, "semantic_route", translation_route) &&
           _selected(row, "inferred_capability_tier", validation_tier)
    ]
end

function _row_title(row::InventoryRow)
    name = strip(row["name_claim"])
    key = strip(row["key"])
    isempty(name) ? (isempty(key) ? row["row_key"] : key) : name
end

_anchor(row::InventoryRow) = "row-" * strip(
    replace(lowercase(row["row_key"]), r"[^a-z0-9]+" => "-"), '-'
)

function _recorded_code(row::InventoryRow)
    exact = strip(row["exact_current_brm_body"])
    !isempty(exact) && return (exact, "exact current BRM body")
    inferred = strip(row["inferred_current_brm_body"])
    !isempty(inferred) && return (inferred, "inferred current BRM body")
    (row["formula_return"], "recorded formula return")
end

function _source_status(row::InventoryRow)
    verdict = row["source_fidelity_verdict"]
    state = verdict in ("confirmed", "adapted-but-defensible") ? :available :
            verdict in ("mismatch", "dead-source") ? :blocked :
            Symbol(replace(verdict, '-' => '_'))
    SemanticStatus(state; detail="Source fidelity: $verdict. $(row["source_fidelity_reason"])")
end

function _validation_status(row::InventoryRow)
    tier = row["inferred_capability_tier"]
    state = tier in ("warmuphmc-sampled", "bridgestan-finite-density-gradient", "stanc-accepted") ? :available :
            startswith(tier, "failed-") || tier == "unsupported" ? :blocked : :unresolved
    SemanticStatus(state; detail="Current audited translation validation tier: $tier.")
end

function _validation_unavailability(row::InventoryRow)
    status = row["inferred_translation_status"]
    status == "ready" && isempty(row["failure_stage"]) && return nothing
    detail = strip(row["inferred_translation_note"])
    isempty(detail) && (detail = strip(row["failure_evidence"]))
    isempty(detail) && (detail = "the row has no exact executable validation evidence")
    SemanticUnavailable("Exact validation unavailable ($status): $detail")
end

function _descriptor_unavailability(row::InventoryRow)
    contract = row["descriptor_mount_contract"]
    startswith(contract, "no executable descriptor") || return nothing
    SemanticUnavailable("Executable descriptor unavailable: $contract")
end

"""
Project one matrix row into reusable semantic elements.

The adapter field is intentionally a claim about presentation only. The matrix
row and its exact/inferred evidence remain authoritative; the semantic mount
adapter is explicitly experimental and cannot upgrade a validation tier.
"""
function row_card(row::InventoryRow)
    code, code_provenance = _recorded_code(row)
    unavailable = Any[
        item for item in (
            _validation_unavailability(row),
            _descriptor_unavailability(row),
        ) if item !== nothing
    ]

    SemanticCard(
        _row_title(row),
        SemanticFields(;
            row_key=row["row_key"],
            source=row["source"],
            source_path=row["source_path"],
            source_line=row["source_line"],
            dataset=row["dataset_claim"],
            formula=row["formula_return"],
            source_fidelity=row["source_fidelity_verdict"],
            family_claim=row["family_description_claim"],
            inferred_family=row["inferred_family"],
            family_provenance=row["inferred_family_provenance"],
            renderer_family_claim=row["renderer_family_claim"],
            family_discrepancy=row["family_discrepancy"],
            translation_route=row["semantic_route"],
            surface_support=row["inferred_surface_support_class"],
            surface_secondary_gap=row["inferred_surface_secondary_gap"],
            family_audit_class=row["family_audit_evidence_class"],
            family_adapter_control=row["family_adapter_control_tier"],
            family_adapter_control_scope=row["family_adapter_control_scope"],
            current_validation_tier=row["inferred_capability_tier"],
            exact_validation_tier=row["exact_capability_tier"],
            inferred_translation_status=row["inferred_translation_status"],
            exact_evidence=row["exact_evidence_kind"],
            descriptor=row["descriptor"],
            descriptor_contract=row["descriptor_mount_contract"],
            descriptor_adapter="experimental; evidence remains authoritative",
            code_provenance,
        ),
        _source_status(row),
        _validation_status(row),
        unavailable...,
        SemanticDisclosure(
            "Recorded code and translation evidence",
            SemanticCode(:julia, code; anchor=_anchor(row) * "-code"),
            SemanticFields(;
                exact_translation_note=row["exact_translation_note"],
                inferred_translation_note=row["inferred_translation_note"],
                semantic_route_note=row["semantic_route_note"],
                failure_stage=row["failure_stage"],
                failure_evidence=row["failure_evidence"],
            ),
        );
        anchor=_anchor(row),
    )
end

function semantic_inventory(rows::AbstractVector{InventoryRow}, total::Integer)
    SemanticSection(
        "Historical model inventory",
        SemanticFields(;
            matching_rows=length(rows),
            total_rows=total,
            matrix="model_matrix.tsv (runtime)",
            semantic_reference=SBPMX_REFERENCE,
            descriptor_adapter="experimental",
        ),
        SemanticGroup(Any[row_card(row) for row in rows]);
        anchor="historical-model-inventory",
    )
end

const GALLERY_CSS = read(joinpath(@__DIR__, "gallery.css"), String)

function _gallery_shell(content)
    htmx(
        h.main(content; class="inventory-shell");
        hyperscript_version=nothing,
        pico_version=nothing,
        feedback=false,
        compose=false,
        overlay=false,
        extra_head=(
            h.title("Historical BRM inventory"),
            h.style(Raw(GALLERY_CSS)),
        ),
    )
end

"""
The authoritative inventory surface. It owns data, typed filter context,
evaluated domains, selected rows, semantic output, and its one executable GET
operation in one inferred graph.
"""
@htmx struct HistoricalInventory
    matrix_path::String = DEFAULT_MATRIX_PATH
    rows = load_matrix(matrix_path)

    @param source_fidelity::String = "all"
    @param family_provenance::String = "all"
    @param translation_route::String = "all"
    @param validation_tier::String = "all"

    @options(source_fidelity) = filter_domain(rows, "source_fidelity_verdict")
    @options(family_provenance) = filter_domain(rows, "inferred_family_provenance")
    @options(translation_route) = filter_domain(rows, "semantic_route")
    @options(validation_tier) = filter_domain(rows, "inferred_capability_tier")

    filtered_rows = filter_inventory(
        rows;
        source_fidelity,
        family_provenance,
        translation_route,
        validation_tier,
    )
    inventory = semantic_inventory(filtered_rows, length(rows))

    __page__(content) = _gallery_shell(content)

    """Show every historical model row matching the selected evidence filters."""
    @get index() = inventory
end

function _gallery_semantic_operation(entry)
    initial = entry.name === :index ? entry.object.inventory : nothing
    result = isnothing(initial) ? entry.result : h.div(
        initial;
        id=entry.target_id,
        class="htmxo-semantic-operation-result inventory-results",
        aria_live="polite",
    )
    h.article(
        h.h2(entry.title),
        entry.form,
        h.p(
            "The descriptor-to-card adapter is experimental. It does not replace or upgrade matrix evidence.";
            class="adapter-note",
        ),
        result;
        class="htmxo-semantic-operation inventory-layout",
    )
end

"""
The served host mounts the authoritative inventory surface through
`semantic_app`. There is no application-owned store, manual route/form mirror,
or AppData/AppContext shadow graph.
"""
@htmx struct HistoricalInventoryGalleryApp
    matrix_path::String = DEFAULT_MATRIX_PATH
    @include surface = HistoricalInventory(; matrix_path)

    __page__(content) = _gallery_shell(content)

    """Mount the complete inferred inventory graph and its shared option context."""
    @get index() = semantic_app(
        surface;
        title="Historical Bayesian regression models",
        submit="Apply filters",
        render_operation=_gallery_semantic_operation,
    )
end

build_gallery(path::AbstractString=DEFAULT_MATRIX_PATH; cache_base::AbstractString=mktempdir()) = HistoricalInventoryGalleryApp(
    ; matrix_path=String(abspath(path)), __cache_base__=String(cache_base),
)

filtered_rows(graph::HistoricalInventory) = graph.filtered_rows
filtered_rows(graph::HistoricalInventoryGalleryApp) = graph.surface.filtered_rows

end # module
