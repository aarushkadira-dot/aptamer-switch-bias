import numpy as np, pandas as pd, warnings, collections
warnings.filterwarnings("ignore")
from scipy import stats
cm={'A':'T','C':'G','G':'C','T':'A'}; rc=lambda s:"".join(cm[c] for c in reversed(s))
P7="CAAGCAGAAGACGGCATACGAGAT"; P5="AATGATACGGCGACCACCGAGATCTACAC"
def maxmatch(w,t):
    best=0
    for L in range(3,len(w)+1):
        if any(rc(w[i:i+L]) in t for i in range(len(w)-L+1)): best=L
        else: break
    return best
d=pd.read_csv("/tmp/atp_fixB.csv"); d=d[np.isfinite(d.ratio)]
obs=dict(zip(d.seq.astype(str), d.B))
rng=np.random.default_rng(1)
S=set()
while len(S)<120000: S.add("".join(rng.choice(list("ACGT"),10)))
rows=[{"seq":w,"observed":int(w in obs),"abund":obs.get(w,np.nan),
       "comp":(w.count("A"),w.count("C"),w.count("G")),
       "p7":maxmatch(w,P7)} for w in S]
X=pd.DataFrame(rows)
print(f"sampled {len(X):,} 10-mers, {100*X.observed.mean():.1f}% observed\n")

print("=== STRESS 1: exact-composition matching (identical A/C/G/T counts) ===")
print("  Within each exact composition class, compare observation rate by P7 complementarity.")
res=[]
for comp,sub in X.groupby("comp"):
    if len(sub)<400: continue
    lo=sub[sub.p7<=3]; hi=sub[sub.p7>=5]
    if len(lo)<50 or len(hi)<50: continue
    res.append((comp,len(sub),100*lo.observed.mean(),100*hi.observed.mean(),
                100*(hi.observed.mean()-lo.observed.mean())))
res.sort(key=lambda r:-r[1])
print(f"  {'composition (A,C,G)':>20} {'n':>7} {'obs% p7<=3':>11} {'obs% p7>=5':>11} {'diff':>8}")
for c,n,a,b,dd in res[:14]:
    print(f"  {str(c):>20} {n:>7,} {a:>11.2f} {b:>11.2f} {dd:>+8.2f}")
diffs=np.array([r[4] for r in res]); ws=np.array([r[1] for r in res])
print(f"\n  {len(res)} composition classes with both arms")
print(f"  weighted mean difference = {np.average(diffs,weights=ws):+.2f} pp")
print(f"  classes where high-P7 is LOWER: {(diffs<0).sum()}/{len(diffs)}")
t=stats.wilcoxon(diffs) if len(diffs)>5 else None
if t: print(f"  Wilcoxon signed-rank on per-class differences: statistic={t.statistic:.0f}, p={t.pvalue:.4g}")

print("\n=== STRESS 2: is it P7 specifically, or any 24-mer? ===")
print("  Compare against 20 random 24-mers as sham 'primers'.")
sham=[]
for _ in range(20):
    t_="".join(rng.choice(list("ACGT"),24))
    v=np.array([maxmatch(w,t_) for w in X.seq.values[:20000]])
    o=X.observed.values[:20000]
    r=stats.spearmanr(v,o)
    sham.append(r.statistic)
v7=np.array([maxmatch(w,P7) for w in X.seq.values[:20000]])
r7=stats.spearmanr(v7,X.observed.values[:20000]).statistic
sham=np.array(sham)
print(f"  real P7   rho = {r7:+.5f}")
print(f"  sham mean rho = {sham.mean():+.5f}   sd = {sham.std():.5f}   range [{sham.min():+.5f}, {sham.max():+.5f}]")
z=(r7-sham.mean())/sham.std()
print(f"  z vs sham distribution = {z:+.2f}   -> {'P7-specific' if abs(z)>2 else 'NOT P7-specific, generic k-mer effect'}")

print("\n=== STRESS 3: does it also depress ABUNDANCE among observed sequences? ===")
o=X[X.observed==1].copy()
for comp,sub in list(o.groupby("comp"))[:0]: pass
g=o.groupby("p7").abund.agg(["size","median"])
print(f"  {'p7':>3} {'n':>8} {'median abundance':>18}")
for k,r in g.iterrows():
    if r['size']<200: continue
    print(f"  {k:>3} {int(r['size']):>8,} {r['median']:>18.1f}")
rr=stats.spearmanr(o.p7, o.abund)
print(f"  Spearman(p7, abundance | observed) = {rr.statistic:+.4f}  p={rr.pvalue:.3g}")
