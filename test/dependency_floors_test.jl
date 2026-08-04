using Test

include("dependency_floors.jl")

function git!(arguments...; dir=nothing)
    command = Cmd(["git", String.(arguments)...])
    isnothing(dir) || (command = Cmd(command; dir=dir))
    run(pipeline(command; stdout=devnull, stderr=devnull))
end

function make_package_checkout(path; name="WarmupHMC")
    mkpath(path)
    git!("init", "--quiet", path)
    git!("config", "user.name", "BRM bootstrap test"; dir=path)
    git!("config", "user.email", "brm-bootstrap@example.invalid"; dir=path)
    write(joinpath(path, "Project.toml"), """
        name = "$name"
        uuid = "a2f4d5b1-1111-4444-8888-123456789abc"
        version = "0.2.1"
        """)
    git!("add", "Project.toml"; dir=path)
    git!("commit", "--quiet", "-m", "test package"; dir=path)
    readchomp(`git -C $path rev-parse HEAD`)
end

@testset "dependency-floor checkout resolution" begin
    mktempdir() do root
        valid = joinpath(root, "valid")
        minimum = make_package_checkout(valid)
        origin = joinpath(root, "origin.git")
        git!("init", "--quiet", "--bare", origin)
        git!("remote", "add", "origin", origin; dir=valid)
        git!("push", "--quiet", "origin", "HEAD:refs/heads/dev"; dir=valid)
        git!("--git-dir=$origin", "update-ref", "refs/kb-pins/$minimum", minimum)

        direct = resolve_git_floor_checkout(
            "WarmupHMC", valid, minimum;
            cache_root=joinpath(root, "unused-cache"),
            mirror="", origin="unused", branch="dev", reason="test floor")
        @test direct == abspath(valid)

        stale = joinpath(root, "stale")
        make_package_checkout(stale; name="StaleWarmupHMC")
        mirror_cache = joinpath(root, "mirror-cache")
        resolved_ref = Ref{String}()
        @test_logs (:warn, r"develop path is stale") (:info, r"Materialized WarmupHMC test dependency") begin
            resolved_ref[] = resolve_git_floor_checkout(
                "WarmupHMC", stale, minimum;
                cache_root=mirror_cache,
                mirror=origin, origin="unused", branch="dev",
                reason="test floor")
        end
        resolved = resolved_ref[]
        @test resolved != abspath(stale)
        @test _git_checkout_contains(resolved, minimum)
        @test isfile(joinpath(resolved, "Project.toml"))

        # A valid cache is sufficient even when neither remote source is usable.
        @test resolve_git_floor_checkout(
            "WarmupHMC", nothing, minimum;
            cache_root=mirror_cache,
            mirror=joinpath(root, "missing-mirror"),
            origin=joinpath(root, "missing-origin"), branch="dev",
            reason="test floor") == resolved

        origin_resolved = resolve_git_floor_checkout(
            "WarmupHMC", nothing, minimum;
            cache_root=joinpath(root, "origin-cache"),
            mirror="", origin=origin, branch="dev", reason="test floor")
        @test _git_checkout_contains(origin_resolved, minimum)
    end
end
