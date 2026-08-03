# Quantify the compile-time cost of the GQ-only emission surface.
import re, os, sys, glob, subprocess, time, shutil, json
D, W = sys.argv[1], sys.argv[2]
BS = os.path.expanduser("~/.bridgestan/bridgestan-2.9.0")
os.makedirs(W, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strip import strip_gq

def compile_time(stan_path):
    so = stan_path[:-5] + "_model.so"
    for f in glob.glob(stan_path[:-5] + "*.hpp") + glob.glob(so): os.remove(f)
    t = time.time()
    r = subprocess.run(["make", "-C", BS, os.path.abspath(so)],
                       capture_output=True, text=True)
    return (time.time() - t, r.returncode, os.path.getsize(so) if os.path.exists(so) else 0, r.stderr[-800:])

rows = []
for path in sorted(glob.glob(os.path.join(D, "*.stan"))):
    name = os.path.basename(path)[:-5]
    if name not in ("A1_gauss_intercept","A2_gauss_glm3","B2_poisson_log","D3_corr_K1","D5_corr_K3","H1_truncated"): continue
    src = open(path).read()
    full = os.path.join(W, name + "_full.stan");  open(full,"w").write(src)
    lean = os.path.join(W, name + "_lean.stan");  open(lean,"w").write(strip_gq(src))
    tf, rcf, szf, ef = compile_time(full)
    tl, rcl, szl, el = compile_time(lean)
    nf = len([l for l in src.splitlines() if l.strip()])
    nl = len([l for l in strip_gq(src).splitlines() if l.strip()])
    rows.append((name, nf, nl, tf, tl, szf, szl, rcf, rcl))
    print(f"{name:22} lines {nf:4}->{nl:4}   compile {tf:6.1f}s -> {tl:6.1f}s "
          f"({100*(tf-tl)/tf:5.1f}% faster)   .so {szf/1e6:5.2f}MB -> {szl/1e6:5.2f}MB "
          f"({100*(szf-szl)/szf:5.1f}% smaller)  rc={rcf},{rcl}")
    if rcf or rcl: print("   ERR:", (ef or el)[-400:])
if rows:
    print(f"\nMEAN over {len(rows)}: compile "
          f"{sum(r[3] for r in rows)/len(rows):.1f}s -> {sum(r[4] for r in rows)/len(rows):.1f}s ; "
          f".so {sum(r[5] for r in rows)/len(rows)/1e6:.2f}MB -> {sum(r[6] for r in rows)/len(rows)/1e6:.2f}MB")
