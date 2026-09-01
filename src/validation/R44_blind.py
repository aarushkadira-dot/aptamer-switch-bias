import numpy as np, pandas as pd, gc
from scipy import stats
L="ACGT"
pred=np.load("R_predfull.npy")
a=pd.read_csv("atp_fixB.csv",usecols=["seq","ratio"]); a=a[a.seq.str.len()==10]
obs=set(a.seq.unique())
pw=(4**np.arange(10)).astype(np.int64)
s2=lambda c:"".join(L[(c//(4**i))%4] for i in range(10))
print("=== Does the screen systematically MISS the best sequence space? ===")
order=np.argsort(-pred)
for k in [100,1000,10000,100000]:
    top=[s2(c) for c in order[:k]]
    f=np.mean([q in obs for q in top])
    print(f"  top {k:>7,} model-predicted: {f*100:5.1f}% were observed in the screen")
print(f"  baseline: {len(obs)/1048576*100:5.1f}% of all sequence space was observed")
bot=[s2(c) for c in order[-100000:]]
print(f"  bottom 100,000 predicted:      {np.mean([q in obs for q in bot])*100:5.1f}% observed")
print("\n  composition of the model's top 1000 (whole space):")
j="".join([s2(c) for c in order[:1000]]); n=len(j)
print("   "+"  ".join(f"{b} {j.count(b)/n*100:4.1f}%" for b in "ACGT"))
print("  composition of what the screen observed:")
j2="".join(list(obs)[:20000]); n2=len(j2)
print("   "+"  ".join(f"{b} {j2.count(b)/n2*100:4.1f}%" for b in "ACGT"))
print("\n=== CLAIM 2: mutational robustness of good vs random switches ===")
allc=np.arange(1048576,dtype=np.int64)
rng=np.random.default_rng(0)
def sens(codes):
    drops=[]
    for c in codes:
        base=pred[c]; d=[]
        for p in range(10):
            cur=(c//pw[p])%4
            for b in range(4):
                if b==cur: continue
                d.append(base-pred[c+(b-cur)*pw[p]])
        drops.append(np.mean(d))
    return np.array(drops)
top=order[:500]; mid=rng.choice(1048576,500,replace=False)
st=sens(top); sm=sens(mid)
print(f"  top-500 predicted : mean value {pred[top].mean():.3f}   mean drop per mutation {st.mean():.4f}")
print(f"  random 500        : mean value {pred[mid].mean():.3f}   mean drop per mutation {sm.mean():.4f}")
print(f"  relative fragility (drop / value): top {st.mean()/pred[top].mean():.4f}   random {sm.mean()/pred[mid].mean():.4f}")
print(f"  ratio = {(st.mean()/pred[top].mean())/(sm.mean()/pred[mid].mean()):.2f}x")
print(f"  Mann-Whitney p = {stats.mannwhitneyu(st,sm).pvalue:.2e}")
print("\n  COOPERATIVITY PREDICTS high fragility for good switches (they need a contiguous block).")
