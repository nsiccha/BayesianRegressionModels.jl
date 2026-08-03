const WARMUPHMC_NATIVE_PPL_MINIMUM =
    "6e6be51a016ab1e3cae9c7478f5c885788c65155"

function _git_succeeds(arguments)
    try
        success(pipeline(Cmd(String.(arguments)); stdout=devnull, stderr=devnull))
    catch
        false
    end
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
        Required minimum: $minimum. Re-run test/bootstrap.jl with
        BRM_TEST_WARMUPHMC pointing at that commit or a descendant.
        """)
    head
end
