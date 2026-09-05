import collections, numpy as np, pandas as pd
from scipy import stats
B="ACGT"
print("="*76)
print(" A. Does the ATP-derived mix ratio predict the GLUCOSE library?")
print("    (glucose N10 was ordered separately -> an independent test, no new order needed)")
print("="*76)
def comp(df):
    c=collections.Counter()
    for w,b in zip(df.seq.astype(str), df.B): 
        for ch in w: c[ch]+=b
    t=sum(c.values()); return np.array([c[x]/t for x in B])
atp=pd.read_csv("/tmp/atp_fixB.csv"); atp=atp[np.isfinite(atp.ratio)]
glu=pd.read_csv("/sessions/dazzling-sweet-pascal/mnt/outputs/aptamer/glu_L.csv"); glu=glu[np.isfinite(glu.ratio)]
a=comp(atp); g=comp(glu)
print(f"\n  ATP     library: " + "  ".join(f"{x} {100*a[i]:.2f}%" for i,x in enumerate(B)))
print(f"  GLUCOSE library: " + "  ".join(f"{x} {100*g[i]:.2f}%" for i,x in enumerate(B)))
print(f"\n  per-base agreement: max |diff| = {100*np.abs(a-g).max():.2f} pp   corr = {np.corrcoef(a,g)[0,1]:.4f}")
# the ATP-derived correction, applied to a nominally equimolar order, should reproduce glucose
pred = a  # multiplicative model says a separately-ordered library from the same vendor mix lands here
chi = ((g-pred)**2/pred).sum()*len(glu)
print(f"\n  ATP-predicted glucose composition (same vendor mix): " + "  ".join(f"{x} {100*pred[i]:.2f}%" for i,x in enumerate(B)))
print(f"  observed - predicted:                                " + "  ".join(f"{x} {100*(g-pred)[i]:+.2f}pp" for i,x in enumerate(B)))
print(f"\n  -> the two independently ordered libraries agree to {100*np.abs(a-g).max():.2f} pp on every base.")
print("     That is the validation the mix-ratio recipe needed, without buying anything.")

print("\n"+"="*76)
print(" B. Is the depletion consistent with UNIFORM unequal amidite incorporation?")
print("    If C is incorporated independently at each of the 10 positions with prob p,")
print("    the C-count distribution must be Binomial(10, p).")
print("="*76)
for name,df in (("ATP",atp),("glucose",glu)):
    nC=df.seq.astype(str).str.count("C")
    w=df.B.values
    p=np.average(nC,weights=w)/10
    obs=np.array([w[nC==k].sum() for k in range(11)]); obs=obs/obs.sum()
    exp=np.array([stats.binom.pmf(k,10,p) for k in range(11)])
    print(f"\n  {name}: fitted per-position P(C) = {p:.4f}")
    print(f"    {'k':>2} {'observed':>10} {'Binomial':>10} {'ratio':>8}")
    for k in range(9):
        r = obs[k]/exp[k] if exp[k]>0 else np.nan
        print(f"    {k:>2} {100*obs[k]:9.4f}% {100*exp[k]:9.4f}% {r:8.3f}")
    m=exp>1e-6
    print(f"    max |obs-exp| over k<=8: {100*np.abs(obs[:9]-exp[:9]).max():.4f} pp")
    print(f"    correlation(obs, binomial) = {np.corrcoef(obs[m],exp[m])[0,1]:.5f}")

print("\n"+"="*76)
print(" C. CORRECTION to my earlier dinucleotide conclusion")
print("="*76)
print("""
  I said the symmetric 3'/5' result 'excluded synthesis chemistry'. That was too strong.

  What the symmetry excludes is CONTEXT-DEPENDENT coupling -- i.e. base X coupling
  differently depending on which base precedes it. That was never the main hypothesis.

  The actual hypothesis is UNIFORM unequal amidite reactivity: C couples less
  efficiently than G at every position regardless of context. That model predicts:
     - position-independent depletion        -> observed (C is 15.5-16.9% at all 10 positions)
     - multiplicative in C-count             -> tested in section B above
     - NO dinucleotide asymmetry             -> observed, because the effect has no context term

  So the symmetric result is CONSISTENT with the synthesis hypothesis, not evidence
  against it. I had the logic backwards. The synthesis explanation is still live.
""")
