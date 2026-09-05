import gzip, collections, numpy as np
F="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/SRR24093234_1.fastq.gz"
LIM=4_000_000
B="ACGT"
# Anchor on TCTCAG (re-establishes register per read, robust to homopolymer indels).
# The 12 bases immediately 5' of the N10 are known truth = all T.
tcount=collections.Counter(); n=0; nread=0
c10=collections.Counter()
with gzip.open(F,"rt") as fh:
    lines=[]
    for line in fh:
        lines.append(line.rstrip("\n"))
        if len(lines)<4: continue
        s=lines[1]; lines=[]
        nread+=1
        if nread>LIM: break
        j=s.find("TCTCAG")
        if j<22: continue
        w=s[j-10:j]
        if len(w)!=10 or not set(w)<=set("ACGT"): continue
        polyT=s[j-22:j-10]           # 12 nt, known truth = TTTTTTTTTTTT
        tcount.update(ch for ch in polyT if ch in B)
        c10.update(w)
        n+=1
t=sum(tcount.values())
print(f"TCTCAG-anchored reads (no polyT purity filter applied): {n:,}")
print(f"\nknown-truth polyT window (12 nt immediately 5' of N10), n={t:,} bases:")
for b in B: print(f"   {b}: {100*tcount[b]/t:7.4f}%")
e = 1 - tcount['T']/t
print(f"\n   measured miscall rate on known-T positions: {100*e:.3f}%")
mis = {b: tcount[b]/(t*e) for b in "ACG"}
print( "   miscalls distribute as: " + "  ".join(f"{b} {100*mis[b]:.1f}%" for b in "ACG"))
print(f"   symmetric would be 33.3% each")

tt=sum(c10.values())
obs={b:c10[b]/tt for b in B}
print(f"\nobserved N10 composition (same reads): " + "  ".join(f"{b} {100*obs[b]:.2f}%" for b in B))

print("\n=== what symmetric error does to an observed composition ===")
print("  If true composition is p and error rate e distributes uniformly to the other 3 bases:")
print("     observed = p(1-e) + (1-p)(e/3)")
print("  This pulls every base TOWARD 25%. So a symmetric error CANNOT create a deficit —")
print("  it can only shrink one. Inverting it makes the true deficit LARGER, not smaller.\n")
for b in B:
    o=obs[b]
    true=(o - e/3)/(1-e-e/3)
    print(f"   {b}: observed {100*o:6.2f}%  ->  error-inverted true {100*true:6.2f}%   (deficit vs 25%: {25-100*true:+.2f} pp)")

print("\n=== the two estimates disagree, and here is why ===")
print("  quality filtering  -> C rises to ~19.6%  (deficit ~5 pp)")
print("  error inversion    -> C falls below observed (deficit LARGER than ~9 pp)")
print("  Quality filtering does not just remove errors: it SELECTS reads that sequenced")
print("  cleanly, and cleanliness correlates with composition. So it is confounded.")
print("  Error inversion assumes symmetry, which the miscall distribution above tests.")
