# Public example data

These tables are the published simulated example of CDC's
[ww-inference-model](https://github.com/CDCgov/ww-inference-model/tree/3e92fea29a0b96212e1690dbdcaeb8abfe19cc4f/data),
commit `3e92fea29a0b96212e1690dbdcaeb8abfe19cc4f`, downloaded 2026-11-11.
They are simulated observations, not records of patients or wastewater plants.
The pinned repository includes the [Apache-2.0 license](LICENSE), copied here
with the exported tables. Its `data-raw/vignette_data.R` generates the example
with seed 1. The TSV format conversion is the only change to these input values;
the selection of the fitting window happens in `../data.jl`.

The R `*.rda` objects were exported with base R `write.table(..., sep="\t",
row.names=FALSE, quote=FALSE)`. For the two probability vectors the columns
are `lag` and `probability`; generation intervals begin at lag **1**, while
hospital delays begin at lag **0**. The original numeric precision is retained.

`ww_data.tsv` contains log genome copies per mL, log detection limits and the
catchment population for each sample. `hosp_data.tsv` contains daily state
admissions and the state population. `true_global_rt.tsv` is CDC's generating
trajectory. The two PMF tables are CDC's packaged COVID generation-interval
and infection-to-hospitalization distributions.

`../data.jl` selects site 1 and 35 calendar days, 2023-09-01 through 2023-10-05.
The selected catchment has 400,000 people inside a state of 3,000,000; the
remaining population is the unobserved reference. It retains 50 days for the
seeding history and a seven-day forecast. It uses 757,000 mL per person per day
for dilution so concentration stays in genome copies per mL.
