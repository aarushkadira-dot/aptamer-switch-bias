import gzip, collections, numpy as np
F="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/SRR24093234_1.fastq.gz"
def comp(it):
    c=collections.Counter()
    for q in it: c.update(q)
    t=sum(c.values()); return {b:c[b]/t*100 for b in "ACGT"}
matched=collections.Counter(); unmatched=[]; nomatch_n=0; n=0
qual_m=[]; qual_u=[]
with gzip.open(F,"rt") as fh:
    lines=[]
    for i,line in enumerate(fh):
        lines.append(line.strip())
        if len(lines)<4: continue
        s,q=lines[1],lines[3]; lines=[]
        n+=1
        if n>3000000: break
        j=s.find("TCTCAG")
        ok = j>=10 and len(s[j-10:j])==10 and set(s[j-10:j])<=set("ACGT") and s[max(0,j-22):j-10].count("T")>=10
        if ok:
            matched[s[j-10:j]]+=1; qual_m.append(np.mean([ord(c)-33 for c in q]))
        else:
            nomatch_n+=1
            if len(unmatched)<200000:
                # take the 10 nt after the longest polyT run, as a best-effort N10
                import re
                m=re.search(r"T{15,32}",s[:50])
                if m and len(s[m.end():m.end()+10])==10: unmatched.append(s[m.end():m.end()+10])
            qual_u.append(np.mean([ord(c)-33 for c in q]))
print("=== CHECK 1: are the EXCLUDED reads compositionally different? ===")
print(f"  reads {n:,}   framed {sum(matched.values()):,} ({sum(matched.values())/n*100:.1f}%)   excluded {nomatch_n:,}")
print("  framed reads N10   : "+"  ".join(f"{b} {comp(list(matched.elements()))[b]:5.2f}%" for b in "ACGT"))
if unmatched:
    print(f"  EXCLUDED reads N10 : "+"  ".join(f"{b} {comp(unmatched)[b]:5.2f}%" for b in "ACGT")+f"   (n={len(unmatched):,})")
print(f"  mean read quality: framed {np.mean(qual_m):.1f}   excluded {np.mean(qual_u):.1f}")
print("\n=== CHECK 2: do SINGLETON (likely error) sequences inflate the unique-composition estimate? ===")
byn=collections.defaultdict(list)
for s_,c_ in matched.items(): byn[min(c_,6)].append(s_)
print(f"  {'read count':>11} {'n unique':>10}   composition")
for k in sorted(byn):
    lab=f"{k}" if k<6 else "6+"
    print(f"  {lab:>11} {len(byn[k]):>10}   "+"  ".join(f"{b} {comp(byn[k])[b]:5.2f}%" for b in "ACGT"))
allu=list(matched.keys())
rep=[s_ for s_,c_ in matched.items() if c_>=3]
print(f"\n  ALL unique      (n={len(allu):>7,}): "+"  ".join(f"{b} {comp(allu)[b]:5.2f}%" for b in "ACGT"))
print(f"  unique, >=3 reads (n={len(rep):>7,}): "+"  ".join(f"{b} {comp(rep)[b]:5.2f}%" for b in "ACGT"))
print("  -> singletons are error-prone and drift toward 25% each, inflating the C estimate.")
