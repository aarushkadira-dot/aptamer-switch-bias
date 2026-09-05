import gzip, collections, numpy as np
F1="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/SRR24093234_1.fastq.gz"
F2="/sessions/dazzling-sweet-pascal/mnt/Figure 2 and 3/SRR24093233_1.fastq.gz"
LIM=4_000_000
B="ACGT"
def scan(F,minq):
    c=collections.Counter(); kept=0; seen=0
    with gzip.open(F,"rt") as fh:
        lines=[]
        for line in fh:
            lines.append(line.rstrip("\n"))
            if len(lines)<4: continue
            s,q=lines[1],lines[3]; lines=[]
            seen+=1
            if seen>LIM: break
            j=s.find("TCTCAG")
            if j<10: continue
            w=s[j-10:j]
            if len(w)!=10 or not set(w)<=set("ACGT"): continue
            if s[max(0,j-22):j-10].count("T")<10: continue
            qw=[ord(x)-33 for x in q[j-10:j]]
            if min(qw)<minq: continue          # EVERY base in N10 above threshold
            c.update(w); kept+=1
    t=sum(c.values())
    return {b:100*c[b]/t for b in B}, kept

print("Composition of N10 under increasingly strict PER-BASE quality filters")
print("(every one of the 10 bases must exceed the threshold)\n")
print(f"  {'minQ':>5} {'reads kept':>12} " + "".join(f"{b:>8}" for b in B) + "   C deficit vs 25%")
prev=None
for mq in (0,20,25,30,35,37):
    comp,kept = scan(F1,mq)
    if kept<20000:
        print(f"  {mq:>5} {kept:>12,}   -- too few reads, stopping")
        break
    print(f"  {mq:>5} {kept:>12,} " + "".join(f"{comp[b]:8.2f}" for b in B) + f"   {25-comp['C']:+8.2f} pp")
    prev=comp

print("\nSame, second independent run (SRR24093233):")
print(f"  {'minQ':>5} {'reads kept':>12} " + "".join(f"{b:>8}" for b in B) + "   C deficit vs 25%")
for mq in (0,30,35):
    comp,kept = scan(F2,mq)
    if kept<20000:
        print(f"  {mq:>5} {kept:>12,}   -- too few reads")
        continue
    print(f"  {mq:>5} {kept:>12,} " + "".join(f"{comp[b]:8.2f}" for b in B) + f"   {25-comp['C']:+8.2f} pp")

print("\n=== interior-only, high quality (drop positions 1 and 10, which show edge bleed) ===")
def interior(F,minq):
    c=collections.Counter(); seen=0; kept=0
    with gzip.open(F,"rt") as fh:
        lines=[]
        for line in fh:
            lines.append(line.rstrip("\n"))
            if len(lines)<4: continue
            s,q=lines[1],lines[3]; lines=[]
            seen+=1
            if seen>LIM: break
            j=s.find("TCTCAG")
            if j<10: continue
            w=s[j-10:j]
            if len(w)!=10 or not set(w)<=set("ACGT"): continue
            if s[max(0,j-22):j-10].count("T")<10: continue
            qw=[ord(x)-33 for x in q[j-10:j]]
            if min(qw)<minq: continue
            c.update(w[1:9]); kept+=1
    t=sum(c.values())
    return {b:100*c[b]/t for b in B}, kept
for mq in (0,30,35):
    comp,kept=interior(F1,mq)
    if kept<20000: continue
    print(f"  minQ {mq:>2}  n={kept:>10,}  " + "  ".join(f"{b} {comp[b]:5.2f}%" for b in B) + f"   C deficit {25-comp['C']:+.2f} pp")
