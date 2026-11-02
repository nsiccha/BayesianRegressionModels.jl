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

# Loading the bootstrap's actual pin/mirror configuration must not instantiate
# the test environment or clone any dependencies.
module BootstrapConfig
include("setup_env.jl")
end

@testset "every bootstrap pin can use its host mirror" begin
    mktempdir() do root
        package = joinpath(root, "package")
        revision = String(make_package_checkout(package))
        host_mirrors = joinpath(homedir(), ".local", "state", "kb-agents", "git-mirrors")
        for (name, _, _) in BootstrapConfig.PINS
            configured = get(BootstrapConfig.MIRRORS, name, "")
            @test configured == joinpath(host_mirrors, "$name.jl.git")
            isempty(configured) && continue
            mirror = joinpath(root, basename(configured))
            git!("clone", "--quiet", "--bare", package, mirror)
            cache_root = joinpath(root, "cache")
            resolved = resolve_git_revision_checkout(
                name, revision; cache_root, mirror,
                origin=joinpath(root, "unavailable-origin"))
            @test readchomp(`git -C $resolved rev-parse HEAD`) == revision
            @test read(joinpath(resolved, "Project.toml")) ==
                read(joinpath(package, "Project.toml"))
            @test readchomp(`git -C $resolved remote get-url origin`) == mirror

            # Reuse must not contact either source after the exact cache exists.
            @test resolve_git_revision_checkout(
                name, revision; cache_root,
                mirror=joinpath(root, "unavailable-mirror"),
                origin=joinpath(root, "unavailable-origin")) == resolved
        end
    end
end

@testset "exact pin falls back only when absent from the mirror" begin
    mktempdir() do root
        package = joinpath(root, "package")
        older = make_package_checkout(package)
        mirror = joinpath(root, "mirror.git")
        git!("clone", "--quiet", "--bare", package, mirror)
        write(joinpath(package, "payload.txt"), "new pinned tree\n")
        git!("add", "payload.txt"; dir=package)
        git!("commit", "--quiet", "-m", "new pinned revision"; dir=package)
        revision = String(readchomp(`git -C $package rev-parse HEAD`))
        @test _mirror_contains(mirror, older)
        @test !_mirror_contains(mirror, revision)

        for (case, source) in (("absent", joinpath(root, "absent.git")),
                ("stale", mirror))
            resolved = resolve_git_revision_checkout(
                "WarmupHMC", revision; cache_root=joinpath(root, case),
                mirror=source, origin=package)
            @test readchomp(`git -C $resolved rev-parse HEAD`) == revision
            @test read(joinpath(resolved, "payload.txt"), String) == "new pinned tree\n"
            @test readchomp(`git -C $resolved remote get-url origin`) == package
        end
    end
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
