import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy import stats
cm={'A':'T','C':'G','G':'C','T':'A'}; rc=lambda s:"".join(cm[c] for c in reversed(s))
GLU="CCCGTGTTAGATAGTAAGTGCAATCTCGGC"   # NNGmin, Z->T
print("The paper's Fig. 4b is a histogram of how often each aptamer base-position is")
print("complementary to one of the TOP 1000 SD sequences from the glucose screen.")
print("MEME motif m1 (bases 2-5) and the whole Fig.4 analysis rest on that top-1000 list.")
print("\nTEST: split the glucose clusters in half at random, build the top-1000 list")
print("independently in each half, and compare the two positional profiles.")
print("If the ranking is noise, the two halves will not agree.\n")

def profile(seqs):
    """for each aptamer position, count how many SDs contain a >=4nt match to the revcomp there"""
    prof=np.zeros(len(GLU))
    for w in seqs:
        r=rc(w)
        for L in (4,5,6):
            for i in range(len(r)-L+1):
                k=GLU.find(r[i:i+L])
                while k!=-1:
                    prof[k:k+L]+=1
                    k=GLU.find(r[i:i+L],k+1)
    return prof

for label,f in (("10 mM","/sessions/dazzling-sweet-pascal/mnt/outputs/aptamer/glu_L.csv"),
                ("100 mM","/sessions/dazzling-sweet-pascal/mnt/outputs/aptamer/glu_H.csv")):
    d=pd.read_csv(f); d=d[np.isfinite(d.ratio)]
    print(f"=== {label} glucose, n={len(d):,} clusters ===")
    rng=np.random.default_rng(0)
    cors=[]
    for rep in range(5):
        idx=rng.permutation(len(d)); h1=d.iloc[idx[:len(d)//2]]; h2=d.iloc[idx[len(d)//2:]]
        # signal-OFF screen: the paper's hits are the LARGEST DECREASES
        t1=h1.nsmallest(1000,"ratio").seq.astype(str).values
        t2=h2.nsmallest(1000,"ratio").seq.astype(str).values
        overlap=len(set(t1)&set(t2))
        p1,p2=profile(t1),profile(t2)
        r=stats.pearsonr(p1,p2)
        cors.append(r.statistic)
        if rep==0:
            print(f"  top-1000 sequence overlap between halves: {overlap}/1000")
            print(f"  half-1 peak position: nt {int(np.argmax(p1))+1}   half-2 peak position: nt {int(np.argmax(p2))+1}")
            print(f"  (paper reports the peak at bases 2-5, 5' end)")
    print(f"  profile correlation between independent halves, 5 reps: "
          f"mean r = {np.mean(cors):+.3f}  range [{min(cors):+.3f}, {max(cors):+.3f}]")
    # benchmark: same test on ATP, where reliability is 0.699
    print()

d=pd.read_csv("/tmp/atp_fixB.csv"); d=d[np.isfinite(d.ratio)]
print(f"=== BENCHMARK: same test on ATP (split-half reliability 0.699), n={len(d):,} ===")
rng=np.random.default_rng(0); cors=[]; ov=[]
ATP="ACCTGGGGGAGTATTGCGGAGGAAGGT"
def profA(seqs):
    prof=np.zeros(len(ATP))
    for w in seqs:
        r=rc(w)
        for L in (4,5,6):
            for i in range(len(r)-L+1):
                k=ATP.find(r[i:i+L])
                while k!=-1:
                    prof[k:k+L]+=1; k=ATP.find(r[i:i+L],k+1)
    return prof
for rep in range(5):
    idx=rng.permutation(len(d)); h1=d.iloc[idx[:len(d)//2]]; h2=d.iloc[idx[len(d)//2:]]
    t1=h1.nlargest(1000,"ratio").seq.astype(str).values
    t2=h2.nlargest(1000,"ratio").seq.astype(str).values
    ov.append(len(set(t1)&set(t2)))
    cors.append(stats.pearsonr(profA(t1),profA(t2)).statistic)
print(f"  top-1000 overlap between halves: mean {np.mean(ov):.0f}/1000")
print(f"  profile correlation: mean r = {np.mean(cors):+.3f}  range [{min(cors):+.3f}, {max(cors):+.3f}]")
