import pandas as pd, numpy as np
from scipy import stats
import statsmodels.api as sm

def perbase(seqs,y,L,label,libname):
    df=pd.DataFrame({'s':seqs,'y':y})
    for b in 'ACGT': df['n'+b]=df['s'].str.count(b)
    df['pur']=(df['nA']+df['nG'])/L
    print(f"\n--- {libname} ({label}), n={len(df):,}, variable region {L} nt ---")
    print(f"  {'base':>5} {'lib freq':>9} {'rho vs function':>16} {'p':>11}")
    for b in 'ACGT':
        r=stats.spearmanr(df['n'+b],df['y'])
        print(f"  {b:>5} {df['n'+b].mean()/L:>9.3f} {r.statistic:>+16.4f} {r.pvalue:>11.3g}")
    r=stats.spearmanr(df['pur'],df['y'])
    print(f"  PURINE (A+G) frac: lib {df['pur'].mean():.3f}   rho vs function = {r.statistic:+.4f}  p={r.pvalue:.3g}")
    # optimum vs library location
    g=df.groupby(df['pur'].round(2)).agg(n=('y','size'),perf=('y','mean')); g=g[g['n']>=100]
    opt=g['perf'].idxmax()
    print(f"  library mean purine frac = {df['pur'].mean():.3f}   best-performing purine bin = {opt:.2f}")
    print(f"  fraction of library at/below optimum purine frac: {(df['pur']<=opt).mean():.1%}")
    return df

print("="*74)
print(" CROSS-LIBRARY TEST: two independent screens, different labs/platforms/molecules")
print("="*74)

# --- ATP aptamer switch library (Soh lab, Nat Commun 2023), degenerate N10 ---
a=pd.read_csv('/tmp/atp_fixB.csv')
a=a[np.isfinite(a['ratio'])]
A=perbase(a['seq'].astype(str),a['ratio'].values,10,
          "ratio = target/buffer signal","ATP DNA aptamer switch, N10 degenerate")

# --- Toehold switch library (Collins lab, Nat Commun 2020), genome-tiled ---
F="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/GSE149225_toehold_processed_datafile.csv"
d=pd.read_csv(F,low_memory=False); d=d.dropna(subset=['ON_OFF']); d=d[d['QC_ON_OFF']>=3].copy()
T=perbase(d['off_id'].astype(str).str[20:50],d['ON_OFF'].values,30,
          "ON_OFF ratio","RNA toehold switch, genome-tiled")

print("\n"+"="*74)
print(" SUMMARY")
print("="*74)
print(f"  {'':38} {'ATP (DNA)':>16} {'toehold (RNA)':>16}")
print(f"  {'library purine fraction':38} {A['pur'].mean():>16.3f} {T['pur'].mean():>16.3f}")
print(f"  {'purine fraction if unbiased':38} {0.5:>16.3f} {'(genomic)':>16}")
ra=stats.spearmanr(A['pur'],A['y']); rt=stats.spearmanr(T['pur'],T['y'])
print(f"  {'rho(purine, function)':38} {ra.statistic:>+16.4f} {rt.statistic:>+16.4f}")
print(f"  {'p':38} {ra.pvalue:>16.3g} {rt.pvalue:>16.3g}")
print(f"\n  Same sign in both: {'YES' if np.sign(ra.statistic)==np.sign(rt.statistic) else 'NO'}")
