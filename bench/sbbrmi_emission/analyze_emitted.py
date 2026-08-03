import re, os, sys, glob, collections
D = sys.argv[1]
BLOCKS = ["functions","data","transformed data","parameters","transformed parameters","model","generated quantities"]

def split_blocks(src):
    out, i, n = {}, 0, len(src)
    while i < n:
        m = None
        for b in sorted(BLOCKS, key=len, reverse=True):
            mm = re.compile(r'^'+re.escape(b)+r'\s*\{', re.M).search(src, i)
            if mm and (m is None or mm.start() < m[1].start()): m = (b, mm)
        if m is None: break
        b, mm = m
        depth, j = 1, mm.end()
        while j < n and depth:
            if src[j]=='{': depth+=1
            elif src[j]=='}': depth-=1
            j+=1
        out[b] = src[mm.end():j-1]
        i = j
    return out

def udfs(fsrc):
    # top-level function definitions in the functions block
    res, i, n = [], 0, len(fsrc)
    pat = re.compile(r'^\s*(real|vector|matrix|row_vector|void|int|array\[[^\]]*\]\s*\w+|tuple\([^)]*\))\s+(\w+)\s*\(', re.M)
    for m in pat.finditer(fsrc):
        depth, j, started = 0, m.end(), False
        while j < n:
            if fsrc[j]=='{': depth+=1; started=True
            elif fsrc[j]=='}':
                depth-=1
                if started and depth==0: break
            j+=1
        res.append((m.group(2), fsrc[m.start():j+1]))
    return res

rows = []
for path in sorted(glob.glob(os.path.join(D, "*.stan"))):
    name = os.path.basename(path)[:-5]
    src = open(path).read()
    B = split_blocks(src)
    fns = udfs(B.get("functions",""))
    bodies = {}
    for f, b in fns: bodies[f] = bodies.get(f, "") + "\n" + b
    fnames = sorted(bodies)
    body_wo_fns = "".join(v for k,v in B.items() if k!="functions")
    gq = B.get("generated quantities","")
    non_gq = "".join(v for k,v in B.items() if k not in ("functions","generated quantities"))

    def closure(seed):
        reach, frontier = set(), list(seed)
        while frontier:
            f = frontier.pop()
            if f in reach: continue
            reach.add(f)
            for g in fnames:
                if g != f and re.search(r'\b'+g+r'\s*\(', bodies[f]): frontier.append(g)
        return reach
    def direct(text):
        hits = []
        for w in fnames:
            base = re.sub(r'_(lpdf|lpmf|rng)$','',w)
            if re.search(r'\b'+w+r'\s*\(', text) or re.search(r'\b'+base+r'\s*\(', text) \
               or re.search(r'~\s*'+base+r'\s*\(', text): hits.append(w)
        return hits
    reach   = closure(direct(non_gq))                 # needed for log_density/gradient
    reach_gq= closure(direct(gq)) - reach             # needed only for generated quantities
    dead    = [f for f in fnames if f not in reach and f not in reach_gq]
    gq_only = sorted(reach_gq)
    fnlines = {f: len([l for l in b.splitlines() if l.strip()]) for f,b in bodies.items()}
    ln_reach = sum(fnlines[f] for f in reach if f in fnlines)
    ln_gq    = sum(fnlines[f] for f in gq_only if f in fnlines)
    ln_dead  = sum(fnlines[f] for f in dead if f in fnlines)

    def nlines(k): 
        b = B.get(k,"")
        return len([l for l in b.splitlines() if l.strip()])
    tp_decls = len(re.findall(r'^\s*(vector|matrix|real|row_vector|array)\S*\s+\w+\s*=', B.get("transformed parameters",""), re.M))
    rows.append(dict(
        name=name, total=len([l for l in src.splitlines() if l.strip()]),
        fn=nlines("functions"), td=nlines("transformed data"), par=nlines("parameters"),
        tp=nlines("transformed parameters"), mdl=nlines("model"), gqn=nlines("generated quantities"),
        nudf=len(fns), n_reach=len(reach), n_gqonly=len(gq_only), n_dead=len(dead),
        gq_only=gq_only, dead=dead, tp_decls=tp_decls, ln_reach=ln_reach, ln_gq=ln_gq, ln_dead=ln_dead,
        lkj1 = len(re.findall(r'cholesky_factor_corr\[n_terms_\w+\]', B.get("parameters",""))),
        reshape = src.count("reshape("), hcat = src.count("hcat("),
        diagpre = src.count("diag_pre_multiply("),
        ranefsd = 1 if "brm_ranef_sd" in src else 0,
        scalarized = len(re.findall(r'jbroadcasted_\w+', src)),
        constfold = len(re.findall(r'\(1\.0 \./ \d', src)),
        matmul1 = len(re.findall(r'matrix\[\w+, 1\]', src)),
    ))

W = "{name:24} {total:>5} {fn:>5} {td:>4} {par:>4} {tp:>4} {mdl:>4} {gqn:>4} {nudf:>5} {n_reach:>6} {n_gqonly:>7} {n_dead:>5}"
print("### Per-model emitted-code anatomy (non-blank lines)\n")
print(W.format(name="model", total="TOT", fn="fns", td="tdat", par="par", tp="tpar", mdl="mdl", gqn="gq",
               nudf="nUDF", n_reach="reach", n_gqonly="gqOnly", n_dead="dead"))
print("-"*104)
for r in rows: print(W.format(**r))

tot=lambda k: sum(r[k] for r in rows)
print(f"\nTOTALS over {len(rows)} models: lines={tot('total')}  functions-block={tot('fn')} ({100*tot('fn')/tot('total'):.0f}%)"
      f"  GQ={tot('gqn')} ({100*tot('gqn')/tot('total'):.0f}%)")
print(f"functions-block LINES: needed-by-density={tot('ln_reach')}  GQ-only={tot('ln_gq')}  DEAD={tot('ln_dead')}")
print(f"UDFs emitted={tot('nudf')}  reachable-from-density={tot('n_reach')}  GQ-only={tot('n_gqonly')}  DEAD={tot('n_dead')}")

print("\n### Cross-cutting markers (count per model)\n")
M = "{name:24} {lkj1:>5} {reshape:>8} {hcat:>5} {diagpre:>8} {ranefsd:>8} {scalarized:>11} {constfold:>10} {matmul1:>8} {tp_decls:>8}"
print(M.format(name="model", lkj1="LKJ", reshape="reshape", hcat="hcat", diagpre="diagpre", ranefsd="ranef_sd",
               scalarized="scalarized", constfold="constfold", matmul1="Nx1 mat", tp_decls="tp decls"))
print("-"*104)
for r in rows: print(M.format(**r))

print("\n### Dead / GQ-only UDFs by model\n")
for r in rows:
    if r["gq_only"] or r["dead"]:
        print(f"  {r['name']:24} gq-only={sorted(r['gq_only'])}  dead={sorted(r['dead'])}")

# helper duplication across the corpus
allfn = collections.Counter()
for path in sorted(glob.glob(os.path.join(D,"*.stan"))):
    for f,_ in udfs(split_blocks(open(path).read()).get("functions","")): allfn[f]+=1
print("\n### Helper emission frequency across the 27-model corpus (top 20)\n")
for f,c in allfn.most_common(20): print(f"  {c:3}x  {f}")
