import numpy as np
from scipy import stats
cm={'A':'T','C':'G','G':'C','T':'A'}; rc=lambda s:"".join(cm[c] for c in reversed(s))
# aptamer sequences (ATP from Fig 2c, glucose NNGmin from Fig 4a / Supp Table 1 with Z->T)
ATP="ACCTGGGGGAGTATTGCGGAGGAAGGT"
GLU="CCCGTGTTAGATAGTAAGTGCAATCTCGGC"
# published targeting profiles (authors' own analyses)
P_ATP=[34,63,77,83,85,72,52,43,31,37,43,67,127,162,207,229,248,193,142,98,70,45,33,25,21,9]      # Fig 2a, pos 1-26
P_GLU=[40,52,52,53,54,24,11,8,3,4,3,5,17,20,18,20,21,17,15,20,24,23,28,29,28,25,16,7,6,4]         # Fig 4b, pos 1-30
NN={'AA':-1.00,'TT':-1.00,'AT':-0.88,'TA':-0.58,'CA':-1.45,'TG':-1.45,'GT':-1.44,'AC':-1.44,
    'CT':-1.28,'AG':-1.28,'GA':-1.30,'TC':-1.30,'CG':-2.17,'GC':-2.24,'GG':-1.84,'CC':-1.84}
def feats(apt,W=5):
    n=len(apt); out={}
    def win(i):
        a=max(0,i-W//2); b=min(n,i+W//2+1); return apt[a:b]
    out['G_content']   =[win(i).count('G')/len(win(i)) for i in range(n)]
    out['C_content']   =[win(i).count('C')/len(win(i)) for i in range(n)]
    out['GC_content']  =[(win(i).count('G')+win(i).count('C'))/len(win(i)) for i in range(n)]
    out['purine']      =[(win(i).count('A')+win(i).count('G'))/len(win(i)) for i in range(n)]
    # complement C content: how C-rich is the SD that targets this region
    out['compC']       =[rc(win(i)).count('C')/len(win(i)) for i in range(n)]
    # duplex stability of the local window with its perfect complement
    def dg(s):
        return -sum(NN.get(s[k:k+2],0) for k in range(len(s)-1))
    out['duplex_dG']   =[dg(win(i)) for i in range(n)]
    # self-structure: can this window pair elsewhere in the aptamer?
    def selfpair(i):
        w=win(i); r=rc(w); best=0
        for L in range(3,len(w)+1):
            for a in range(len(w)-L+1):
                if w[a:a+L] in apt.replace(w,'',1) or r[a:a+L] in apt: best=max(best,L)
        return best
    out['self_pair']   =[selfpair(i) for i in range(n)]
    return out
print("CROSS-APTAMER TEST: does one rule predict BOTH published targeting profiles?\n")
print(f"  ATP aptamer     {ATP}  ({len(ATP)} nt), profile peaks at nt {int(np.argmax(P_ATP))+1}")
print(f"  glucose aptamer {GLU}  ({len(GLU)} nt), profile peaks at nt {int(np.argmax(P_GLU))+1}\n")
FA=feats(ATP); FG=feats(GLU)
pa=np.array(P_ATP,dtype=float); pg=np.array(P_GLU,dtype=float)
print(f"  {'feature':14s} {'r (ATP)':>10} {'p':>10} | {'r (glucose)':>12} {'p':>10} | {'both same sign?':>16}")
print("  "+"-"*80)
for k in FA:
    a=np.array(FA[k][:len(pa)],dtype=float); g=np.array(FG[k][:len(pg)],dtype=float)
    ra=stats.pearsonr(a,pa); rg=stats.pearsonr(g,pg)
    same = "YES" if np.sign(ra.statistic)==np.sign(rg.statistic) else "no"
    star = " <<<" if (same=="YES" and ra.pvalue<0.05 and rg.pvalue<0.05) else ""
    print(f"  {k:14s} {ra.statistic:+10.3f} {ra.pvalue:10.4f} | {rg.statistic:+12.3f} {rg.pvalue:10.4f} | {same:>16}{star}")
