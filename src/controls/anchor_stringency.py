import gzip, collections, numpy as np, regex
F="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/SRR24093234_1.fastq.gz"
LIM=3_000_000
B="ACGT"
def run(maxmis):
    pat=regex.compile("(TCTCAG){s<=%d}"%maxmis)
    tc=collections.Counter(); c10=collections.Counter(); n=0; seen=0
    with gzip.open(F,"rt") as fh:
        lines=[]
        for line in fh:
            lines.append(line.rstrip("\n"))
            if len(lines)<4: continue
            s=lines[1]; lines=[]
            seen+=1
            if seen>LIM: break
            m=pat.search(s,25)          # search after the polyT
            if not m: continue
            j=m.start()
            if j<22: continue
            w=s[j-10:j]
            if len(w)!=10 or not set(w)<=set("ACGT"): continue
            tc.update(ch for ch in s[j-22:j-10] if ch in B)
            c10.update(w); n+=1
    t=sum(tc.values()); tt=sum(c10.values())
    err=1-tc['T']/t
    return n, seen, err, {b:100*c10[b]/tt for b in B}

print("Is the 0.332% error rate an artifact of anchoring on a PERFECT TCTCAG match?")
print("Relax the anchor to allow substitutions and see whether the error rate and")
print("the N10 composition move.\n")
print(f"  {'mismatches':>11} {'reads kept':>11} {'% of scanned':>13} {'polyT err':>10} " + "".join(f"{b:>8}" for b in B))
prev=None
for mm in (0,1,2):
    n,seen,err,comp = run(mm)
    print(f"  {mm:>11} {n:>11,} {100*n/seen:>12.1f}% {100*err:>9.3f}% " + "".join(f"{comp[b]:8.2f}" for b in B))
    if prev: 
        print(f"              -> C shift vs stricter anchor: {comp['C']-prev['C']:+.2f} pp")
    prev=comp

print("\n=== TEST 2: what amidite ratio should be ordered to get an equimolar N region ===")
n,seen,err,obs = run(0)
print(f"  observed output when ordering 'equimolar': " + "  ".join(f"{b} {obs[b]:.2f}%" for b in B))
corr={b: 25.0/obs[b] for b in B}
z=sum(corr.values())
norm={b: 100*corr[b]/z for b in B}
print(f"\n  required input ratio (normalised to 100):")
for b in B:
    print(f"    {b}: {norm[b]:6.2f}%   (relative factor {corr[b]:.3f}x)")
print(f"\n  i.e. order roughly  A:C:G:T = " + " : ".join(f"{norm[b]/norm['G']:.2f}" for b in B) + "   (G normalised to 1)")
print("  Caveat: assumes the deviation is multiplicative and stable across orders and vendors.")
print("  It is a prediction, not a validated recipe, and needs one test order to confirm.")
