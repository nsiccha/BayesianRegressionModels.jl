function _git_succeeds(arguments)
    try
        success(pipeline(Cmd(String.(arguments)); stdout=devnull, stderr=devnull))
    catch
        false
    end
end

function _git_output(arguments)
    readchomp(Cmd(String.(arguments)))
end

function _git_checkout_contains(path, minimum)
    path = abspath(path)
    _git_succeeds(["git", "-C", path, "rev-parse", "--git-dir"]) &&
        _git_succeeds(["git", "-C", path, "cat-file", "-e", "$minimum^{commit}"]) &&
        _git_succeeds([
            "git", "-C", path, "merge-base", "--is-ancestor", minimum, "HEAD",
        ])
end

function require_git_ancestor(name, path, minimum; reason)
    path = abspath(path)
    _git_succeeds(["git", "-C", path, "rev-parse", "--git-dir"]) || error("""
        $name at $path is not a readable Git checkout.
        $reason
        Point the test bootstrap at a checkout containing $minimum or later.
        """)

    head = readchomp(`git -C $path rev-parse HEAD`)
    _git_succeeds([
        "git", "-C", path, "merge-base", "--is-ancestor", minimum, head,
    ]) || error("""
        $name checkout $path is too old: HEAD is $head.
        $reason
        Required minimum: $minimum. Materialize that commit or a descendant
        with test/setup_env.jl.
        """)
    head
end

function _require_julia_package_checkout(name, path)
    isdir(path) || error("$name checkout does not exist: $path")
    isfile(joinpath(path, "Project.toml")) ||
        error("$name checkout at $path has no Project.toml")
    path
end

function _mirror_contains(mirror, revision)
    !isempty(mirror) && isdir(mirror) && _git_succeeds([
        "git", "--git-dir=$mirror", "cat-file", "-e", "$revision^{commit}",
    ])
end

function _clone_floor_checkout!(destination, minimum;
        mirror, origin, branch)
    branch_ref = "refs/heads/$branch"
    if _mirror_contains(mirror, minimum)
        if _mirror_contains(mirror, branch_ref) && _git_succeeds([
                "git", "--git-dir=$mirror", "merge-base", "--is-ancestor",
                minimum, branch_ref,
            ])
            run(Cmd([
                "git", "clone", "--quiet", "--branch", branch,
                "--single-branch", mirror, destination,
            ]))
            return "host mirror $branch_ref"
        end
        run(Cmd(["git", "clone", "--quiet", "--no-checkout", mirror, destination]))
        run(Cmd(["git", "-C", destination, "checkout", "--quiet", "--detach", minimum]))
        return "host mirror refs/kb-pins/$minimum"
    end

    run(Cmd([
        "git", "clone", "--quiet", "--branch", branch,
        "--single-branch", origin, destination,
    ]))
    "public $origin $branch"
end

# A pin needs one commit, and a checkout in the shared store is kept for weeks,
# so it holds that commit alone: 5 MB of `.git` for a ReactiveKernels pin against
# 148 MB for a full clone wherever git cannot hardlink from the mirror, and no
# hardlink into the mirror for `_make_read_only!` to reach.
function _clone_revision_checkout!(destination, revision; mirror, origin)
    remote, source = if _mirror_contains(mirror, revision)
        mirror, "host mirror revision $revision"
    else
        origin, "public $origin revision $revision"
    end
    git(arguments...) = run(Cmd(["git", "-C", destination, arguments...]))
    git("init", "--quiet")
    git("remote", "add", "origin", remote)
    git("fetch", "--quiet", "--depth=1", "origin", revision)
    git("checkout", "--quiet", "--detach", revision)
    source
end

function require_git_revision(name, path, revision)
    path = abspath(path)
    _require_julia_package_checkout(name, path)
    _git_succeeds(["git", "-C", path, "rev-parse", "--git-dir"]) ||
        error("$name at $path is not a readable Git checkout")
    head = _git_output(["git", "-C", path, "rev-parse", "HEAD"])
    head == revision || error("""
        $name checkout $path is at $head, not the pinned revision $revision.
        A checkout never moves, so this one is damaged. Other runs on this host
        may share it: remove it (`chmod -R u+w`, then `rm -rf`) and run
        test/setup_env.jl again.
        """)
    head
end

"""
    test_package_store()

The directory holding this host's pinned test dependencies, one read-only
checkout per revision. It sits outside every checkout of this repository because
Julia validates a compiled image against the absolute path of its source: a
dependency at one stable path keeps its images, and its dependents', across
worktrees and jobs. `BRM_TEST_PACKAGE_STORE` overrides it.
"""
test_package_store() = abspath(get(ENV, "BRM_TEST_PACKAGE_STORE",
    joinpath(homedir(), ".local", "share", "BayesianRegressionModels", "package-sets")))

# Nothing may write inside a shared checkout: a changed source file would reject
# every image built from it on this host, and any other file would reach every
# run. So files and directories lose their write bits, as `Pkg` does for the
# depot's packages. `chmod` leaves mtimes alone, and those are what Julia's cache
# validation reads. A symlink is skipped because `chmod` would follow it.
function _make_read_only!(path)
    read_only(entry) = chmod(entry, filemode(entry) & (typemax(filemode(entry)) ⊻ 0o222))
    for (root, _, files) in walkdir(path; topdown=false)
        for file in files
            entry = joinpath(root, file)
            islink(entry) || read_only(entry)
        end
        read_only(root)
    end
end

# Publishing renames a complete, read-only tree into place, so no run ever sees
# part of one. Two runs materializing one pin both get here. The rename never
# replaces a checkout, so the later run's `mv` throws, and it validates the
# earlier run's checkout instead; its own scratch directory is then discarded.
function _publish_checkout!(name, scratch, cache, revision)
    _make_read_only!(scratch)
    try
        mv(scratch, cache)
    catch
        ispath(cache) || rethrow()
    end
    require_git_revision(name, cache, revision)
    cache
end

const CHECKOUT_RETENTION_DAYS = 14
const UNFINISHED_ENTRY_RETENTION_DAYS = 1
const LAST_USED_DIR = ".last-used"
# The only names the store writes. Eviction touches nothing else, whatever
# `BRM_TEST_PACKAGE_STORE` points at.
const _CHECKOUT_ENTRY = r"^[a-z0-9_]+-[0-9a-f]{12}$"
const _UNFINISHED_ENTRY = r"^\.(?:[a-z0-9_]+-bootstrap-|evicting-)"

# Use is recorded beside the checkout, since nothing may write inside it.
function _mark_checkout_used(cache)
    stamps = joinpath(dirname(cache), LAST_USED_DIR)
    mkpath(stamps)
    touch(joinpath(stamps, basename(cache)))
end

# The rename comes first so that no run can take a half-removed tree for a
# checkout. Removing a read-only tree needs its directories writable again.
function _remove_checkout!(path)
    doomed = path
    if !startswith(basename(path), ".")
        doomed = joinpath(dirname(path), ".evicting-$(basename(path))-$(getpid())")
        mv(path, doomed)
    end
    for (root, _, _) in walkdir(doomed)
        chmod(root, filemode(root) | 0o700)
    end
    rm(doomed; recursive=true)
end

"""
    evict_unused_checkouts!(cache_root; retention_days, now)

Remove the checkouts below `cache_root` that no setup has used for
`retention_days`, and what a killed run left behind. A worktree's reap used to
free its pins; a shared store has to free them itself, because pins move daily.
A worktree whose pinned checkout was evicted recovers by running
`test/setup_env.jl` again. Returns the entries removed.
"""
function evict_unused_checkouts!(cache_root;
        retention_days=CHECKOUT_RETENTION_DAYS,
        now=time())
    cache_root = abspath(cache_root)
    evicted = String[]
    isdir(cache_root) || return evicted
    stamps = joinpath(cache_root, LAST_USED_DIR)
    for entry in readdir(cache_root)
        unfinished = occursin(_UNFINISHED_ENTRY, entry)
        unfinished || occursin(_CHECKOUT_ENTRY, entry) || continue
        path = joinpath(cache_root, entry)
        stamp = joinpath(stamps, entry)
        last_used = mtime(!unfinished && isfile(stamp) ? stamp : path)
        limit = unfinished ? UNFINISHED_ENTRY_RETENTION_DAYS : retention_days
        now - last_used > limit * 86400 || continue
        try
            _remove_checkout!(path)
            rm(stamp; force=true)
            push!(evicted, entry)
        catch exception
            # Gone means another run evicted it first. Otherwise this setup can
            # still finish, so the failure is reported rather than raised.
            ispath(path) && @warn "Could not evict a pinned test dependency" path exception
        end
    end
    isempty(evicted) || @info "Evicted unused pinned test dependencies" cache_root evicted
    evicted
end

"""
    resolve_git_revision_checkout(name, revision; kwargs...)

Materialize and return an exact, read-only Git checkout below `cache_root`, and
record that it was used. An existing checkout is validated and never rewritten,
because a fresh mtime rejects every image built from it, identical bytes
included. Prefer a host mirror containing `revision`, otherwise fetch it from the
public `origin`. Unlike [`resolve_git_floor_checkout`](@ref), descendants do not
satisfy an exact source pin.
"""
function resolve_git_revision_checkout(name, revision;
        cache_root,
        mirror="",
        origin)
    cache_root = abspath(cache_root)
    cache = joinpath(cache_root, "$(lowercase(name))-$(revision[1:12])")
    # Before the checkout is read, so a concurrent eviction sees this use.
    _mark_checkout_used(cache)
    if ispath(cache)
        require_git_revision(name, cache, revision)
        return cache
    end

    source = mktempdir(cache_root; prefix=".$(lowercase(name))-bootstrap-") do scratch
        source = _clone_revision_checkout!(
            scratch, revision; mirror=mirror, origin=origin)
        require_git_revision(name, scratch, revision)
        _publish_checkout!(name, scratch, cache, revision)
        source
    end
    @info "Materialized $name test dependency" path=cache source revision
    cache
end

"""
    resolve_git_floor_checkout(name, configured_path, minimum; kwargs...)

Return a Julia-package checkout whose `HEAD` contains `minimum`. A valid
caller-provided checkout wins. Otherwise materialize one persistent,
versioned checkout below `cache_root`, preferring the host mirror's immutable
pin/`branch` and falling back to the public `origin` branch. The shared
`~/github` checkout is deliberately never treated as evidence of what has
landed or been published: it may be stale when its worktree is dirty.
"""
function resolve_git_floor_checkout(name, configured_path, minimum;
        cache_root,
        mirror="",
        origin,
        branch="dev",
        reason)
    if !isnothing(configured_path)
        configured_path = abspath(configured_path)
        _require_julia_package_checkout(name, configured_path)
        if _git_checkout_contains(configured_path, minimum)
            return configured_path
        end
        head = _git_succeeds(["git", "-C", configured_path, "rev-parse", "HEAD"]) ?
            _git_output(["git", "-C", configured_path, "rev-parse", "HEAD"]) :
            "not a Git checkout"
        @warn "$name develop path is stale; materializing the required floor instead" path=configured_path head=head minimum=minimum
    end

    cache_root = abspath(cache_root)
    cache = joinpath(cache_root, "$(lowercase(name))-$(minimum[1:12])")
    if ispath(cache)
        _require_julia_package_checkout(name, cache)
        require_git_ancestor(name, cache, minimum; reason)
        return cache
    end

    mkpath(cache_root)
    source = mktempdir(cache_root; prefix=".$(lowercase(name))-bootstrap-") do scratch
        source = _clone_floor_checkout!(
            scratch, minimum; mirror=mirror, origin=origin, branch=branch)
        _require_julia_package_checkout(name, scratch)
        require_git_ancestor(name, scratch, minimum; reason)
        if ispath(cache)
            _require_julia_package_checkout(name, cache)
            require_git_ancestor(name, cache, minimum; reason)
        else
            mv(scratch, cache)
        end
        source
    end
    @info "Materialized $name test dependency" path=cache source minimum
    cache
end
