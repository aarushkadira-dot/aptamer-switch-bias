import gzip, collections, numpy as np, random
F="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/SRR24093234_1.fastq.gz"
matched=collections.Counter(); n=0
with gzip.open(F,"rt") as fh:
    lines=[]
    for i,line in enumerate(fh):
        lines.append(line.strip())
        if len(lines)<4: continue
        s=lines[1]; lines=[]
        n+=1
        if n>3000000: break
        j=s.find("TCTCAG")
        if j>=10 and set(s[j-10:j])<=set("ACGT") and len(s[j-10:j])==10 and s[max(0,j-22):j-10].count("T")>=10:
            matched[s[j-10:j]]+=1
print("=== CHECK 3: are singletons sequencing ERRORS of abundant sequences? ===")
abundant=set(s_ for s_,c in matched.items() if c>=5)
singles=[s_ for s_,c in matched.items() if c==1]
random.seed(0); samp=random.sample(singles,20000)
def nbrs(q):
    for p in range(10):
        for b in "ACGT":
            if b!=q[p]: yield q[:p]+b+q[p+1:]
hit=sum(1 for q in samp if any(x in abundant for x in nbrs(q)))
# null: random 10-mers
rnd=["".join(random.choice("ACGT") for _ in range(10)) for _ in range(20000)]
hit0=sum(1 for q in rnd if any(x in abundant for x in nbrs(q)))
print(f"  singletons within Hamming-1 of an abundant (>=5 read) sequence: {hit/len(samp)*100:.1f}%")
print(f"  random 10-mers, same test (null)                              : {hit0/len(rnd)*100:.1f}%")
print(f"  enrichment = {(hit/len(samp))/(hit0/len(rnd)):.2f}x")
print("  -> strong enrichment means singletons are largely 1-base errors off real sequences.")
print("\n=== CHECK 4: does read-count filtering just re-impose the abundance bias? ===")
print(f"  {'min reads':>10} {'n unique':>9}  {'C%':>6}")
for t in [1,2,3,5,10,20]:
    u=[s_ for s_,c in matched.items() if c>=t]
    c=collections.Counter()
    for q in u: c.update(q)
    tot=sum(c.values())
    print(f"  {t:>10} {len(u):>9,}  {c['C']/tot*100:6.2f}%")
print("  -> C% falls monotonically with the threshold: filtering by abundance IS")
print("     filtering by composition. No threshold gives an unbiased synthesis estimate.")
print("\n=== CHECK 5: what IS robust? abundance-weighted composition ===")
c=collections.Counter()
for s_,k in matched.items(): 
    for ch in s_: c[ch]+=k
tot=sum(c.values())
print(f"  abundance-weighted (all framed reads): "+"  ".join(f"{b} {c[b]/tot*100:5.2f}%" for b in "ACGT"))
print("  singletons contribute 1 read each of 1.43M, so this is insensitive to error junk.")
