# CHANGELOG.md's newest heading is Project.toml's version.
# RUN: julia --project=test test/changelog.jl
#
# A landing that adds or changes public API bumps the patch version and adds its
# entry under a new heading at the top of CHANGELOG.md (dev §7.5, user decision
# DynamicObjects:snag.dynamicobjects-h-c580a9f3/decisions/2026-10-27T14-24-51-154-0uokyy9),
# so a dependant can state that version as its [compat] floor. Two landings that
# bump at once then collide on that heading, never on a shared number.
using Test
import TOML

@testset "CHANGELOG.md's newest version is Project.toml's version" begin
    root = dirname(@__DIR__)
    version = VersionNumber(TOML.parsefile(joinpath(root, "Project.toml"))["version"])
    # eachline strips "\n" and "\r\n" alike: a Windows checkout writes CRLF.
    matches = (match(r"^## (\d+\.\d+\.\d+)\s*$", line)
               for line in eachline(joinpath(root, "CHANGELOG.md")))
    headings = [VersionNumber(m[1]) for m in matches if m !== nothing]
    @test !isempty(headings)
    @test first(headings) == version
    @test allunique(headings)
    @test issorted(headings; rev=true)
end
