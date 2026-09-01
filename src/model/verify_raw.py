import openpyxl, numpy as np, pandas as pd
p='/sessions/dazzling-sweet-pascal/mnt/uploads/41467_2023_38105_MOESM4_ESM.xlsx'
wb=openpyxl.load_workbook(p,read_only=True); ws=wb['Soh Supp Data 1']
it=ws.iter_rows(values_only=True); hdr=next(it)
print("measurement columns (local index -> header):")
for i,h in enumerate(hdr[4:]): print(f"   {i:>2}  {h}")
seqs=[];V=[]
for r in it:
    if r[0] is None: continue
    seqs.append(r[0]); V.append(r[4:16])
V=np.array(V,dtype=np.float64); seqs=np.array(seqs)
print(f"\nrows read: {len(V):,}")
np.save("V_raw.npy",V); np.save("seq_raw.npy",seqs)
fm=[0,11]; bufs=[1,3,5,7,9]; atps=[2,4,6,8,10]
print(f"\nmean FM  (local 0, 11): {V[:,0].mean():7.1f}  {V[:,11].mean():7.1f}")
print(f"mean buffers          : "+"  ".join(f"{V[:,b].mean():7.1f}" for b in bufs))
print(f"mean ATP              : "+"  ".join(f"{V[:,a].mean():7.1f}" for a in atps))
print(f"\nFM(11) vs buffer(9): FM is {(V[:,11].mean()/V[:,9].mean()-1)*100:+.1f}% relative to buf9")
