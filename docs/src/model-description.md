# Scientific model descriptions

## Inspect roles and ranks

[`brm_model_view`](@ref) returns a reusable HTMXObjects semantic tree with the
declaration, data/parameter roles, value ranks and dimensions. Load HTMXObjects
0.1.60 or later; no web demo app is needed:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
model_view_demo = (@brm begin
    sigma ~ Exponential(1)
    beta ~ Normal(0, 1)
    mu = beta * x
    y ~ Normal(mu, sigma)
end)((; x=[-1., 0., 1.], y=[0.1, 0.2, 0.3]))
""", :model_view_demo; title="Regression model")
```

Pass the bound model to the view:

```julia
using HTMXObjects

view = brm_model_view(model_view_demo; title="Regression model")
repr("text/html", view)       # also text/markdown and text/plain
```

Use the result inside another semantic page or as an HTMXObjects route result.
Source, roles/ranks and callable documentation have separate tabs. Callable
documentation and carrier details use clickable, keyboard-accessible disclosures.
Documentation comes from the actual callable bindings in the model, when those
bindings have docstrings; unrelated overloads of Base arithmetic operators are
omitted.

Authored declaration docstrings appear as formatted prose in the Documentation
tab, using [`brm_documentation`](@ref). Each record includes its exact logical
path and a disclosure with the attached statement, source file and line. The
same records supply the source text and the PDF; the view stores no separate
copy of documentation metadata.
Headings use readable paths such as `left › inner`; the exact tuple path remains
in provenance, preserving the distinction between nested names and dotted names.

For a bound model containing reusable [components](components.md), the Source
tab includes the actual reachable definitions from [`brm_components`](@ref).
Each shared definition gets one source tab, with all its call-instance paths;
the model source also lists how each instance is called. Nested definitions
are included recursively. Paths identify authored call sites, so repeated
plate iterations share a site. The defining module and authored docstring
provenance remain available without compiling a backend.

`brm_model_view(component)` also inspects a component definition and its
reachable definitions without data. Its Component inputs tab distinguishes
positional and keyword arguments, required inputs and optional defaults.
Default expressions are shown as source without evaluation. An unbound builder
shows its own source; bind it to data to discover its component instances.

The view only inspects what it receives. It never prepares or fits a model and
never evaluates a custom helper to discover its result shape. A bound `BRMI`
provides concrete data shapes and distribution ranks where known. A `BRMBuilder`
provides its source and potential data inputs; observation roles depend on
binding, and a sampled name can remain latent when no data is supplied.
Joint observations label each authored response column as `Data · response`;
their synthetic operation is not a separate logical quantity. Empty input
inventories explicitly say when there are no potential or bound data inputs.

To request RK preparation explicitly, load `ReactiveKernels` and
`ReactiveKernelsPPL` 0.1.93 or later, then call `brm_model_view(RKBRMI(model))`. For an existing
Stan-backed descriptor, call `brm_model_view(descriptor)`. Preparation can enrich
parameter dimensions; arbitrary derived values can still have unknown shapes.
Unknown means no shape metadata is available, not that a value is scalar.
`brm_descriptor` is Stan-backed, so it is not a generic inspection step for
RK models with Julia helpers.

A frozen RK replay can share an earlier build with a different binding. Its
view leaves current parameter dimensions unresolved, because that build's
layout still belongs to the earlier binding. The RK data table is explicitly
labelled as declaration data; a replay can retain the original source data.
No layout is assigned during inspection to fill in missing current dimensions.

Rank is the number of axes: scalar 0, vector 1, matrix 2. Displayed dimensions
describe constrained values, including zero-length axes. Free-coordinate counts
are a separate fact: a length-three simplex is a vector with two free coordinates.
Generated carriers are identified separately from logical model quantities.
RK parameter paths retain their authored namespace segments. Ragged parameters
show their outer rank and shape, with each element's shape in a disclosure.

### Export the view

Use the same view for an accessible PDF. HTMXObjects projects every disclosure
and eager tab into Markdown, so the report includes their contents even when
they are closed in the browser. This includes all reachable component source,
not just the selected definition:

```julia
pdf_report(view, "model.pdf"; title="Regression model", lang="en-GB")
```

This uses HTMXObjects' Quarto/Typst exporter; Quarto must be on `PATH`.
To bundle an app containing the view, give its page shell vendored assets,
register the vendor routes, and call `export_site` on the app's existing routes:

```julia
# In the app's @htmx root:
# __page__(content) = HTMXObjects.pico_page(content; assets=:vendor, runtime=:linked)
vendorfiles()
staticfiles("reports", "files")  # if the page links /files/model.pdf
export_site("model-snapshot"; pages=["/model"])
```

`export_site` requires a new or empty destination. It bundles linked files,
syntax highlighting and the framework's interaction assets. Keep source and
documentation tabs eager to include their bodies in both the PDF and snapshot.
The app controls where it mounts the view and links the PDF; the reusable view
does not install routes or generate files when a reader opens it.

## Describe a prepared model

[`brm_description`](@ref) describes a selected, prepared StanBlocks model.
Reuse its [`BRMDescriptor`](@ref); the description reads its parsed formula,
prepared design, fitted preprocessing and generative declarations. It returns
structured prose, LaTeX, notation, effective priors and explicit coverage.
It does not fit, sample or instantiate an executable model.

This generic hierarchical model has a Gaussian observation distribution, a
standardized covariate and a shared intercept/slope covariance block:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
description_model = (@brm begin
    "Observation standard deviation."
    sigma ~ Exponential(2)
    "Expected response with a shared intercept and slope covariance block."
    mu ~ 1 + zscale(x) + (1 + x | shared | group)
    effect(mu, zscale_x) ~ Normal(0, 0.4)
    y ~ Normal(mu, sigma)
end)((;
    x=[-2.0, -0.5, 0.7, 2.5, 3.0, 4.2],
    group=[1, 1, 2, 2, 3, 3],
    y=[0.4, 0.1, 1.1, 1.7, 2.4, 2.8],
))
""", :description_model; title="Model to describe", require_stan=true, total_groups=())
```

```@eval
let d = brm_descriptor(SBBRMI(description_model; total_groups=())),
    anchors = Dict((:population, :mu, :zscale_x) => "#the-description-result")
    result = brm_description(d; prior_anchors=anchors)
    result.complete || error(join(result.diagnostics, "\n"))
    Markdown.parse(brm_description_markdown(result))
end
```

## Authored declaration documentation

Ordinary Julia docstrings keep Markdown beside the declaration it explains,
as in the model above. Read an exact declaration's documentation by its
logical path:

```julia
brm_documentation(description_model; path=:mu)
```

[`brm_documentation`](@ref) returns copied records containing the logical
declaration path, Markdown, authored statement and source location. The
records follow tuple expansion, prior/structure partition, merges and replay.
[`brm_documentation_markdown`](@ref) emits the authored Markdown for HTML and
PDF pipelines; [`brm_description_markdown`](@ref) includes it alongside the
generated scientific description. Authored prose does not change the
description's coverage assessment or the model's numerical identity.

## The description result

```julia
d = brm_descriptor(prepared)
result = brm_description(d;
    labels=Dict(:mu => (; meaning="conditional mean", symbol="\\mu")),
    prior_anchors=Dict((:population, :mu, :x) => "#slope-prior"),
)
markdown = brm_description_markdown(result)
```

`result.model_id` is the selected descriptor's inference identity. Report labels,
units, symbols, hooks and links are presentation metadata. They do not alter
source, data, draws or inference caches. Units and biological meanings must be
supplied; a name such as `log_CL` does not establish a log transformation.

`prose` and `equations` hold every sentence and equation in inventory order.
`sections` holds the same sentences and equations per described component, as
`(; id, prose, equations)`; the Markdown writes each part's sentences followed
by its own equations, in model order.

`components` is an ordered recursive inventory. Each component has an exact
logical tuple `id`, `kind`, actual `callable`, `arguments`, `keywords`, `children`,
`axes`, `outputs`, `priors`, `fitted_constants`, `bindings`, `notation` and
`provenance`. Use `brm_description_components(result)` to traverse it.
Arguments contain literals, [`BRMDescriptionReference`](@ref) values, tuples
and recursive components. Keyword arguments are a named tuple. Collections
and fitted values are read-only snapshots.
Vectors use tuples; matrices and higher-dimensional arrays use
`(size=..., values=...)`, with values in Julia's column-major order.

Output records expose `name`, `logical`, `role`, `kind` and `segments`.
`logical` is a symbol or `nothing`; component and prior IDs are tuples.
Ragged `segments` preserve the descriptor's inclusive grouping ends. Axes
and fitted constants retain their declared source and fitted level/basis data.
Conditioned, held-out and unconditioned observations remain distinct.

`priors` contains [`BRMPriorDescription`](@ref) records with `id`, `distribution`,
`support`, `source`, `anchor` and `provenance`. The distribution retains its actual
callable, arguments and keywords. Winning selector metadata comes from the
producer's precedence resolver. Generated latent priors, scalar hyperpriors,
positive scales and improper unpenalized coefficients are inventoried too.
Gaussian equations square the SD argument; Exponential scale and rate
parameterizations are stated explicitly.
Affine Student-t observations retain their location and scale: for degrees of
freedom ν > 2, their SD is scale × √(ν/(ν−2)). Scale is not their SD.
Declared support bounds describe the prepared parameter constraints; the
inventory retains the actual density factor on that support.

The Markdown renderer creates a stable target for each row of its complete
effective-prior table and links component prior references to those targets.
CommonMark has no Markdown syntax for an element ID, so each target is an empty
raw-HTML element in the row's first cell:
`<span id="brm-prior-5a1f36447d850e2a"></span>P1`. Pandoc keeps it in HTML
output and turns it into a LaTeX label in Quarto PDF output; CommonMark.jl's
`html` and VitePress pass it through. A renderer that escapes all raw HTML
shows it as text unless it keeps empty ID targets.
`brm_description_prior_anchor(result_or_context, prior.id)` returns the target ID;
`brm_description_prior_references(context)` returns the related logical prior IDs.
Targets are scoped to `model_id`. For repeated instances of one model artifact,
pass a unique `prefix` to both `brm_description_markdown` and the anchor helper.
This also applies when separate data or fitted constants share one source
identity; the adapter should give each model mount its own prefix.
The optional `prior_anchors` map adds outbound links to targets owned by the
calling report; it does not create those external targets.
Unobserved declarations that remain sampled parameters retain their actual
conditional density in this inventory, including parameter-dependent families.
Their contexts use `observation_role=:latent_parameter`; the prose identifies
the sampled quantity and its contribution to the joint model.

## Language

A description is written in one language, chosen when it is built:

```julia
const DE = description_texts(:de)
result = brm_description(d; labels, texts=DE)
markdown = brm_description_markdown(result)   # German throughout
```

[`description_texts`](@ref) returns the built-in `:en` (the default) or `:de`
texts. They cover everything BRM writes itself: the prose sentences, the
meanings of the notation it introduces, the coverage-gap messages, the Markdown
headings and table headers, and the descriptive words inside its LaTeX, such as
`\operatorname{Exponential}_{\mathrm{rate}}` or `\text{(continuous)}`. The
result records its texts, so [`brm_description_markdown`](@ref) needs no
language argument and the page cannot mix languages. `model_id`, the component
inventory and the equations' structure do not depend on the language.

What the caller supplies is never translated: `labels` meanings, symbols and
units, hook prose, and the model's own identifiers. Code-level names stay as
they are in every language: logical IDs and axis names (`random_effect`,
`declared`), distribution families (`Gamma`, `LogNormal`), standard operators
(`\operatorname{diag}`, `\operatorname{logit}`) and function names
(`\operatorname{range}`, `\operatorname{constant}`, `rep_matrix`).

Replace any text by keyword, e.g.
`description_texts(:de; effective_priors="A-priori-Annahmen:")`. A word in
braces such as `{owner}` is a value the description fills in; a replacement
must keep exactly the placeholders of the English text. Braces directly after a
letter, `\`, `_` or `^` are LaTeX groups, not placeholders. An unknown locale or
text name, or a placeholder mismatch, throws an `ArgumentError`.

A hook writes its own prose in the description's language by reading
[`brm_description_texts`](@ref)`(context).locale`. A page that needs two
languages calls `brm_description` once per language; it does not fit or sample.

The example model's prose in German:

```@eval
let d = brm_descriptor(SBBRMI(description_model; total_groups=()))
    result = brm_description(d; texts=description_texts(:de))
    result.complete || error(join(result.diagnostics, "\n"))
    Markdown.parse(join(result.prose, "\n\n"))
end
```

## Scientific extensions

An unknown callable leaves a coverage gap. Add a reusable
[`brm_describe_component`](@ref) method, or pass request-local hooks keyed by
actual callable identity:

```julia
hook = context -> BRMDescriptionFragment(
    prose=("The response is a quadratic function of its input.",),
    equations=("q=" * brm_description_math(context, only(context.arguments)) *
               "^2+0.6",),
    covers=(context.id,),
    notation=((; name=:q, meaning="quadratic response", axis=:observation),),
)
result = brm_description(d; hooks=(response_curve => hook,))
```

For this hook, the scientific callable's body is `response_curve(x) = x^2 + 0.6`.
Its actual model registration and units belong to the consumer. Equations are
LaTeX without display delimiters. Notation records have `name`, `meaning` and
optional `symbol`, `axis`, `unit`. Explicit symbols are trusted LaTeX; default
identifiers are escaped by [`brm_description_symbol`](@ref) and
[`brm_description_math`](@ref).

[`brm_description_binding`](@ref)`(context, :offset)` resolves a submodel's
authored internal parameter. Its record contains `name`, `role`, `value`,
`prior_ids` and, in an included submodel, `path`. Render `value` with
`brm_description_math`; use [`brm_description_prior`](@ref) on each exact prior
ID. Repeated nested names can be addressed by an authored tuple path.
Kernel lambda aliases point to actual model arguments. No generated prefix or
declaration-order assumption is needed.
Reference-valued keywords use the same math API, including identity aliases.

A `@plate for` cell is described as written, in its own index. A read `x[i]`
renders as `x` subscripted by `i`. A read `x[rows[i]]` through a model-scope
binding `rows = groups(g => labels)` renders as `x` subscripted by `rows_i`,
and `rows_i = {j : g_j = labels_i}` is stated once per binding. In a
`groups(g)` plate, `x[p]` renders as `x` subscripted by `J_p`, group `p`'s rows.
`indexin(…)` positions render as written. A sum of elementwise reads of one row
set, `sum(exp.(x[rows[i]]))`, renders as the sum over `j` in `rows_i` of
`exp(x_j)`. An output `y[i] = …` renders as `y_i`. A cell latent's law is one
of the cell's statements, and its prior row reads in the cell's names. The
cell's sentence states its index and its row sets; its statements follow as
their own parts. A deprecated `kernel(...) do` cell has no authored plate and
keeps the grouped input mapping below.

Kernel equations show the grouped input mapping; scientific child hooks provide
their recurrences without inserting lambda or block syntax into the equation.
Cell assignments emit separate indexing and arithmetic relations, and the
mapping connects to the cell's returned quantity. Cell-local names are scoped
to their kernel owner (for example, `pred.state`) unless notation overrides them.
The compact grouped map symbol refers to separately defined exact input
bindings. Each authored cell alias has a short symbol and its own binding
relation. Long record arguments receive short symbols with aligned definitions
of all their named fields. Long call assignments and additive population
relations use aligned lines. Wide transformed designs use named prepared coordinates, with their
fitted centering/scaling definitions and logical column mappings listed
separately; values and scientific child coverage are preserved.
Long likelihood arguments use compact intermediate symbols with separate exact
definitions and notation tied to their source component IDs. Distribution
conventions, input bindings and child coverage are preserved.
Long opaque observation-family relations place each actual argument on its own
aligned row, in authored order, including keyword bindings. The family still
needs independent scientific hook coverage.
Inside a cell, `censored`, `truncated`, `weighted` and `interval_censored` are
StanBlocks' distribution combinators. They are built-in descriptions with the
formula wrappers' laws, and their bounds are `lower=`/`upper=` keywords. A
family call, as in `censored(Normal(mu, s); lower=l)`, is an argument child. A
family token followed by its arguments, as in `censored(normal, mu, s; lower=l)`,
becomes a `(..., :family)` child with that token as its callable, so a custom
family token needs its own hook exactly as it does at top level. A cell
`interval_censored(family, L, U, args...)` observes a latent value in `(L, U]`
and contributes `F(U) - F(L)`.
Authored literal assignments have `role=:constant` and expose their value;
composed assignments have `role=:deterministic` and expose a public expression.
This includes assignments preceded by documentation strings in an included
SlicModel. Supplied data arguments retain their prepared bindings. A supplied
keyword wins over an authored fixed default in that same included scope,
including documented defaults; nested scopes remain distinct.
References to deterministic quantities render as symbols; render the binding's
`value` explicitly to expand its defining expression.
Actual higher-order callable arguments retain their identity and render through
the same escaped math helper.
Statically named records such as `(; lower, upper, weights=exp(offset))`
retain public named field values. Use
[`brm_description_record`](@ref)`(context, argument)` to resolve a record
argument through its authoritative local bindings. The returned NamedTuple
contains literal values, parameter/data references and expression components;
render individual fields with `brm_description_math`. Unresolved, cyclic or
non-record arguments fail rather than guessing a scope or evaluating code.

Included-model inputs backed by known prepared arrays retain
`BRMDescriptionReference` values rather than anonymous numeric tuples. Generated
prepared inputs have logical IDs `(:prepared_data, key)`; their math uses the
bound data symbol. This preserves observation, mask and transformed-input
identities without printing the dataset into an equation. Actual literal prior
shapes and limits still retain their numeric values. Static global callable
bindings retain their actual identity without traversing Julia's binding objects.
The lowered `dims` call describes the array shape; indexing selects an axis
extent. Lowered `max` and `min` describe vector reductions or scalar extrema.
Categorical fitted values retain their labels without pool or pointer state.
Embedded likelihood contexts include the response's actual descriptor outputs,
including its posterior-predictive quantity and selected conditioning role.
Conditioning follows prepared response provenance through ragged input aliases
and joint-completion carriers. A partially observed vector has
`observation_role=:partially_observed`, `observation_sources`, and exact
`observed_entries`/`missing_entries` counts; observed entries stay fixed.
Fresh modeled covariate replay uses `observation_role=:covariate_draw` with
`generated_entries` counting the newly drawn values. Every selected prediction
row is drawn conditional on retained model parameters; it adds no sampled
missing coordinates. Multivariate Gaussian Cholesky families retain the actual
lower covariance factor and describe covariance as its product with its transpose.
Joint covariance factors retain their actual marginal-scale and LKJ prior IDs.
The joint completion's allocation-only declaration has constant log density;
its substantive density comes from the declared multivariate model.
Array conversions, matrix replication, ranges, transposition and joint-column
selection retain their actual callable identities and explicit semantics.
Vector/matrix conversion order follows the
[Stan mixed operations reference](https://mc-stan.org/docs/functions-reference/mixed_operations.html).

Covariance blocks use compact, distinct numbers within each result. The notation
entries map each number to its full logical ID and ordered margins; predictor
designs and covariance factors use that same number. Logical IDs and prior
anchors keep their exact identities. The prior table uses short P1, P2 row keys
and names the distribution family. Complete logical IDs, distribution equations
and individual support facts follow in full-width definitions. Component prior
links use those same P keys; their C keys have separate complete logical paths.
Notation uses individual metadata entries, preserving supplied meanings and
units without repeating an unsupplied identifier as its meaning. The original
prior anchors remain in their table cells; exact logical IDs remain in the
public prior records. These presentation keys do not change logical identity.
GP axis and multiple-membership weight references use inline math in prose.

Included scientific SlicModels expose their parameter declarations,
deterministic expressions and nested calls as children. A top-level caption
does not cover those children. `covers` lists exact IDs in the hook's own
subtree; a hook must explicitly explain every scientific subtree it claims.
Claims outside that subtree or for nonexistent IDs raise an error.

`result.complete` certifies inventory coverage. The consumer remains responsible
for independent scientific review of its handwritten hooks. Unsupported calls
and syntax retain precise `coverage` records and `diagnostics`; the Markdown
renderer labels such a result **incomplete**, with the number of uncovered
components and their callables (a component without one by its kind), and
lists them under "Coverage gaps".
Population Horseshoe allocation nodes currently retain explicit gaps. Their
Gaussian conditionals do not establish coverage of the allocation.

## Exact totals and generated vector priors

Exact totals (`total_groups=:auto`; not the default) describe the
authored hierarchy of the conventional program they replace. Each total column
keeps its conventional random-effect identity: `(:random_effect, id, group)`
for a declared `|ID|`, otherwise `(:independent, k)` in emission order. Its SD
prior is listed under `(…, :sd, 1)`, and the absorbed population coefficients
keep their declared priors under `(:population, predictor, coefficient)`, with
`source.integrated == (:total_effect, predictor)`. The sampled totals density
is `(:total_effect, predictor, :totals)`; Student-t population priors add their
exact Gaussian scale mixture as `(:total_effect, predictor, :scale_mixture)`.
The `(:total_effect, predictor)` component gives the design map A, the prior
locations m and precisions q, the integrated density and the exact conditional
recovery of β. When the block integrates group-level predictors (columns
constant within each group), it names them (`group_columns`) and each group's
mean is A(wᵢ∘β), with wᵢ that group's column values. Coordinates that only the
conventional program samples, such as standardized deviations, are not
inventoried.

A vector prior whose coordinates carry different priors, such as tensor
smoothing SDs configured per block, is a product of per-coordinate priors.
That holds whether BRM emits one native vectorised statement (every coordinate
in one family), native per-family blocks laid end to end with `vcat` (builtin
families that differ), or a generated family (custom or composed families).
Each coordinate is a separate component bound to its actual scalar family, so a
consumer-defined family remains an explicit gap until a hook covers that
callable. One prior
shared by every coordinate is described as that family. Sum-to-zero blocks
still retain an explicit `brm_s2z_contrast` gap.

## R2D2 allocation conventions

Whole-predictor `effect(mu, :) ~ r2d2(...)` and shared-block
`sd(:, id) ~ r2d2(...)` use different scale conventions. Descriptions bind their
actual R² prior, simplex concentration, share indices and reference scales from
the prepared emitter; these quantities also appear in the effective prior
inventory.

For the whole-predictor form, T is the total latent scale, Vⱼ is the sample
variance of prepared design column j, and k(j) is its selected simplex share:

$$
s_j=T\sqrt{\phi_{k(j)}R^2/V_j},\qquad
\operatorname{SD}_{\mathrm{group}}=T\sqrt{1-R^2}.
$$

The intercept and population columns with their own explicit priors stay
outside the allocation. With no non-intercept columns, no R² or simplex is
introduced and the group SD is T. A one-component simplex equals [1].

The shared-block R2D2M2/ICC form uses each margin's reference scale rₘ:

$$
\operatorname{SD}_m=r_m\sqrt{\phi_{k(m)}R^2/(1-R^2)}.
$$

With `include=(:population, :contrasts)`, the same budget additionally allocates
population and contrast scales. Each predictor uses its own selected margin
reference; its total-scale expression is r/√(1−R²). Its population and contrast
scales therefore also contain R²/(1−R²), divided by their prepared design-column
variance. Treatment indicators use V=m(n−m)/(n(n−1)), for m selected rows among
n rows. Completed covariates stay inside their model-dependent variance.
Margins outside a selected budget retain their separately inventoried free
scale priors. Correlations keep their declared priors; allocated marginal SDs
do not receive an invented independent SD prior.
