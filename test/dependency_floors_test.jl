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

# A package with two commits, a dangling symlink and a nested directory, the
# shapes the store's single-commit fetch and read-only walk have to get right.
function make_pinned_package(path)
    make_package_checkout(path)
    mkpath(joinpath(path, "src"))
    write(joinpath(path, "src", "payload.jl"), "payload\n")
    symlink("missing-target", joinpath(path, "dangling"))
    git!("add", "--all"; dir=path)
    git!("commit", "--quiet", "-m", "pinned revision"; dir=path)
    String(readchomp(`git -C $path rev-parse HEAD`))
end

function set_mtime!(path, time)
    file = Base.Filesystem.open(path, Base.Filesystem.JL_O_WRONLY)
    try
        Base.Filesystem.futime(file, time, time)
    finally
        close(file)
    end
end

function writable_entries(root)
    entries = String[]
    for (dir, _, files) in walkdir(root)
        filemode(dir) & 0o222 == 0 || push!(entries, dir)
        for file in files
            entry = joinpath(dir, file)
            islink(entry) || filemode(entry) & 0o222 == 0 || push!(entries, entry)
        end
    end
    entries
end

@testset "the store's default location is outside every checkout" begin
    withenv("BRM_TEST_PACKAGE_STORE" => nothing) do
        @test test_package_store() == joinpath(homedir(), ".local", "share",
            "BayesianRegressionModels", "package-sets")
    end
    withenv("BRM_TEST_PACKAGE_STORE" => "relative/store") do
        @test test_package_store() == abspath("relative/store")
    end
end

@testset "a pin is one read-only commit, materialized once" begin
    mktempdir() do root
        package = joinpath(root, "package")
        revision = make_pinned_package(package)
        store = joinpath(root, "store")
        checkout = resolve_git_revision_checkout(
            "WarmupHMC", revision; cache_root=store, origin=package)
        @test dirname(checkout) == store
        @test readchomp(`git -C $checkout rev-parse HEAD`) == revision
        @test readchomp(`git -C $checkout rev-list --count HEAD`) == "1"
        @test read(joinpath(checkout, "src", "payload.jl"), String) == "payload\n"
        @test islink(joinpath(checkout, "dangling"))
        @test isempty(writable_entries(checkout))

        # Julia's cache validation reads these mtimes, so a later run on the same
        # pin must leave them alone, and needs neither source to do it.
        source = joinpath(checkout, "src", "payload.jl")
        materialized = mtime(source)
        @test resolve_git_revision_checkout(
            "WarmupHMC", revision; cache_root=store,
            mirror=joinpath(root, "unavailable-mirror"),
            origin=joinpath(root, "unavailable-origin")) == checkout
        @test mtime(source) == materialized

        # Git still reads the identity from the read-only tree, as
        # test/rk_append_worker.jl does for a developed dependency.
        @test success(`git -C $checkout status --porcelain`)
        @test isempty(readchomp(`git -C $checkout status --porcelain`))
    end
end

@testset "a mirror may hold the pin under no branch" begin
    mktempdir() do root
        package = joinpath(root, "package")
        revision = make_pinned_package(package)
        mirror = joinpath(root, "mirror.git")
        git!("init", "--quiet", "--bare", mirror)
        git!("--git-dir=$mirror", "fetch", "--quiet", package,
            "$revision:refs/kb-pins/$revision")
        @test _mirror_contains(mirror, revision)
        checkout = resolve_git_revision_checkout(
            "WarmupHMC", revision; cache_root=joinpath(root, "pinned"), mirror,
            origin=joinpath(root, "unavailable-origin"))
        @test readchomp(`git -C $checkout rev-parse HEAD`) == revision

        # Dropping the ref leaves an object only `cat-file` can vouch for.
        git!("--git-dir=$mirror", "update-ref", "-d", "refs/kb-pins/$revision")
        @test _mirror_contains(mirror, revision)
        checkout = resolve_git_revision_checkout(
            "WarmupHMC", revision; cache_root=joinpath(root, "unreferenced"), mirror,
            origin=joinpath(root, "unavailable-origin"))
        @test readchomp(`git -C $checkout rev-parse HEAD`) == revision
    end
end

@testset "runs materializing one pin at once all get its one checkout" begin
    mktempdir() do root
        package = joinpath(root, "package")
        revision = make_pinned_package(package)
        store = joinpath(root, "store")

        # Every task passes the first `ispath` before any finishes its clone, so
        # all but one lose at the rename.
        checkouts = fetch.([@async resolve_git_revision_checkout(
            "WarmupHMC", revision; cache_root=store, origin=package) for _ in 1:8])
        @test only(unique(checkouts)) == joinpath(store, "warmuphmc-$(revision[1:12])")
        @test readchomp(`git -C $(checkouts[1]) rev-parse HEAD`) == revision
        @test readdir(store; sort=true) ==
            sort!([LAST_USED_DIR, basename(checkouts[1])])

        # The losing path, deterministically: the winner's tree stays untouched,
        # and a read-only scratch directory is still cleaned up.
        source = joinpath(checkouts[1], "src", "payload.jl")
        published = mtime(source)
        scratch_parent = mktempdir(root)
        mktempdir(scratch_parent) do scratch
            _clone_revision_checkout!(scratch, revision; mirror="", origin=package)
            @test _publish_checkout!("WarmupHMC", scratch, checkouts[1], revision) ==
                checkouts[1]
            @test ispath(scratch)
        end
        @test isempty(readdir(scratch_parent))
        @test mtime(source) == published

        # A failure that is not the race surfaces as itself, not as a later
        # complaint about a missing checkout.
        @test_throws Base.IOError mktempdir(root) do scratch
            _clone_revision_checkout!(scratch, revision; mirror="", origin=package)
            _publish_checkout!("WarmupHMC", scratch,
                joinpath(root, "absent-parent", "checkout"), revision)
        end
    end
end

@testset "eviction removes only what the store wrote and nobody used" begin
    mktempdir() do root
        package = joinpath(root, "package")
        revision = make_pinned_package(package)
        store = joinpath(root, "store")
        resolve(name) = resolve_git_revision_checkout(
            name, revision; cache_root=store, origin=package)
        stale = resolve("WarmupHMC")
        recent = resolve("StanBlocks")
        stamp(checkout) = joinpath(store, LAST_USED_DIR, basename(checkout))

        # Using a checkout refreshes its stamp without touching the tree.
        set_mtime!(stamp(stale), time() - 86400)
        resolve("WarmupHMC")
        @test time() - mtime(stamp(stale)) < 3600

        # What a killed run leaves, and names the store never writes.
        unfinished = mkpath(joinpath(store, ".warmuphmc-bootstrap-abc123"))
        interrupted = mkpath(joinpath(store, ".evicting-warmuphmc-0123456789ab-42"))
        foreign_file = joinpath(store, "notes.txt")
        foreign_dir = mkpath(joinpath(store, "keep-me"))
        foreign_sha_dir = mkpath(joinpath(store, "0123456789abcdef0123456789abcdef01234567"))
        foreign_hidden = mkpath(joinpath(store, ".cache"))
        write(foreign_file, "not the store's\n")
        foreigners = (foreign_file, foreign_dir, foreign_sha_dir, foreign_hidden)

        # Young enough: nothing goes.
        @test isempty(evict_unused_checkouts!(store))
        @test isdir(stale) && isdir(recent) && isdir(unfinished) && isdir(interrupted)

        # Two days on, only the unfinished entries are stale.
        @test sort!(evict_unused_checkouts!(store; now=time() + 2 * 86400)) ==
            sort!(basename.([unfinished, interrupted]))
        @test isdir(stale) && isdir(recent)

        # A used checkout outlives an unused one, read-only tree notwithstanding.
        set_mtime!(stamp(stale), time() - 20 * 86400)
        @test evict_unused_checkouts!(store) == [basename(stale)]
        @test !ispath(stale) && !ispath(stamp(stale))
        @test isdir(recent) && isfile(stamp(recent))

        # A checkout with no stamp ages by its own mtime, and nothing here is left
        # half removed.
        rm(stamp(recent))
        @test evict_unused_checkouts!(store; now=time() + 20 * 86400) == [basename(recent)]
        @test !ispath(recent)
        @test sort!(filter(!=(LAST_USED_DIR), readdir(store))) ==
            sort!(basename.(collect(foreigners)))
        @test all(ispath, foreigners)

        # Eviction is recoverable: the next setup materializes the pin again.
        @test readchomp(`git -C $(resolve("WarmupHMC")) rev-parse HEAD`) == revision
        @test isempty(evict_unused_checkouts!(joinpath(root, "absent-store")))
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
