import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy import stats
cm={'A':'T','C':'G','G':'C','T':'A'}; rc=lambda s:"".join(cm[c] for c in reversed(s))
GLU="CCCGTGTTAGATAGTAAGTGCAATCTCGGC"; ATP="ACCTGGGGGAGTATTGCGGAGGAAGGT"
print("DECISIVE CONTROL: does the top-1000 positional profile differ from the profile")
print("of 1000 RANDOMLY CHOSEN sequences from the same library?")
print("If not, Fig. 4b reflects which aptamer regions have matchable k-mers -- not function.\n")
def prof(seqs, apt):
    p=np.zeros(len(apt))
    for w in seqs:
        r=rc(w)
        for L in (4,5,6):
            for i in range(len(r)-L+1):
                k=apt.find(r[i:i+L])
                while k!=-1:
                    p[k:k+L]+=1; k=apt.find(r[i:i+L],k+1)
    return p
def norm(p): return p/p.sum() if p.sum()>0 else p
rng=np.random.default_rng(0)
def run(label, path, apt, take_smallest):
    d=pd.read_csv(path); d=d[np.isfinite(d.ratio)]
    top = d.nsmallest(1000,"ratio") if take_smallest else d.nlargest(1000,"ratio")
    ptop=norm(prof(top.seq.astype(str).values, apt))
    rs=[]; profs=[]
    for _ in range(20):
        samp=d.sample(1000, random_state=int(rng.integers(1e9)))
        pr=norm(prof(samp.seq.astype(str).values, apt))
        profs.append(pr); rs.append(stats.pearsonr(ptop,pr).statistic)
    prand=np.mean(profs,axis=0)
    # how far is top from the random-sample cloud, per position, in sd units
    sd=np.std(profs,axis=0)
    z=(ptop-prand)/np.where(sd>0,sd,np.nan)
    print(f"=== {label} ===")
    print(f"  corr(top-1000 profile, random-1000 profile) = {np.mean(rs):+.4f}  (20 draws, sd {np.std(rs):.4f})")
    print(f"  max |z| of any position vs the random cloud = {np.nanmax(np.abs(z)):.1f}")
    print(f"  positions with |z| > 5: {[(i+1,round(float(z[i]),1)) for i in range(len(apt)) if abs(z[i])>5][:8]}")
    print(f"  top-1000 peak position: nt {int(np.argmax(ptop))+1}    random peak: nt {int(np.argmax(prand))+1}")
    print()
    return ptop, prand
run("GLUCOSE 10 mM (signal-off: 1000 largest decreases)",
    "/sessions/dazzling-sweet-pascal/mnt/outputs/aptamer/glu_L.csv", GLU, True)
run("GLUCOSE 100 mM",
    "/sessions/dazzling-sweet-pascal/mnt/outputs/aptamer/glu_H.csv", GLU, True)
run("ATP (benchmark, reliability 0.699)",
    "/tmp/atp_fixB.csv", ATP, False)
