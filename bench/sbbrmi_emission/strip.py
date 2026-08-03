import re
def _fnblock(src):
    m = re.match(r'\s*functions\s*\{', src)
    if not m: return None
    depth, k = 1, m.end()
    while k < len(src) and depth:
        if src[k]=='{': depth+=1
        elif src[k]=='}': depth-=1
        k+=1
    return m.start(), m.end(), k

def _defs(fsrc):
    out, n = [], len(fsrc)
    pat = re.compile(r'^\s*(real|vector|matrix|row_vector|void|int|array\[[^\]]*\]\s*\w+)\s+(\w+)\s*\(', re.M)
    for mm in pat.finditer(fsrc):
        depth, j, started = 0, mm.end(), False
        while j < n:
            if fsrc[j]=='{': depth+=1; started=True
            elif fsrc[j]=='}':
                depth-=1
                if started and depth==0: break
            j+=1
        out.append((mm.group(2), fsrc[mm.start():j+1]))
    return out

def _calls(text, names):
    hit = set()
    for w in names:
        base = re.sub(r'_(lpdf|lpmf|lupdf|rng)$','',w)
        if re.search(r'\b'+re.escape(w)+r'\s*\(', text) or re.search(r'\b'+re.escape(base)+r'\w*\s*\(', text):
            hit.add(w)
    return hit

def strip_gq(src):
    m = re.search(r'^generated quantities\s*\{', src, re.M)
    if m:
        depth, j = 1, m.end()
        while j < len(src) and depth:
            if src[j]=='{': depth+=1
            elif src[j]=='}': depth-=1
            j+=1
        src = src[:m.start()] + src[j:]
    fb = _fnblock(src)
    if not fb: return src
    s0, s1, s2 = fb
    fsrc, rest = src[s1:s2-1], src[s2:]
    defs = _defs(fsrc)
    bodies = {}
    for nm, b in defs: bodies[nm] = bodies.get(nm, "") + "\n" + b
    names = list(bodies)
    reach, frontier = set(), list(_calls(rest, names))       # seed: ONLY the non-functions body
    while frontier:                                          # transitive closure over kept bodies
        f = frontier.pop()
        if f in reach: continue
        reach.add(f)
        frontier.extend(g for g in _calls(bodies[f], names) if g != f and g not in reach)
    kept = [b for nm, b in defs if nm in reach]
    if not kept: return src[:s0] + rest
    return src[:s0] + "functions {\n" + "\n".join(kept) + "\n}\n" + rest
