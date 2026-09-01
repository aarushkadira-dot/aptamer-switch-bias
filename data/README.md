# Data

## `atp_corrected.csv.gz`

The ATP switching-domain screen after correcting the photobleaching normalization error
described in the top-level README.

- 494,065 clusters, 339,373 unique 10-mers
- split-half reliability r = 0.699 (original: 0.660)

| column | meaning |
|---|---|
| `seq`   | 10-nt switching domain sequence |
| `B`     | read depth for that cluster |
| `ratio` | target / buffer fluorescence ratio (the switching readout) |

```python
import pandas as pd
d = pd.read_csv("data/atp_corrected.csv.gz")
```

Derived from Supplementary Data 1 of Yoshikawa et al., *Nature Communications* 14:2336 (2023).
Please cite the original authors alongside this repository.

## Not included

Raw reads (SRA PRJNA952942) and the toehold dataset (GSE149225) are large and are not
redistributed here. Download them from the sources listed in the top-level README.
