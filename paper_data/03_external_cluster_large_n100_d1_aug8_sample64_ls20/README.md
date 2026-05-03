# Enhanced External Benchmark: Cluster_large n100 d1

This folder contains a stronger inference stress test on the same 10 external
`Cluster_large` n100/d1 CTSP-d benchmark instances used by the thesis LKH
comparison.

Inference setting:
- reconstructed feature modes: `anchor,mds`
- 8-fold geometric augmentation
- greedy/POMO decoding plus 64 stochastic sampling runs per feature mode
- best candidate selected by the original `.ctspd` distance matrix
- same-priority swap local search with up to 20 passes
- final tours saved under each model's `evaluations/<model>/tours/` folder
- aligned per-instance costs are in `per_instance_costs.csv`
- per-instance winner counts are in `per_instance_winner_counts.csv`
- pairwise model win/loss counts are in `pairwise_win_table.csv`

This is an external/generalization benchmark, not the same-distribution thesis
ablation test. The final thesis checkpoints were trained on synthetic
n100/g8/d1 data, while these instances are TSPLIB-derived and use different
group counts.

| Model | Average cost | Average gap to LKH | Avg. local-search gain |
|---|---:|---:|---:|
| `w/o group embedding` | 24272.4 | 6.88054973950998% | 1128.2 |
| Scheduled/fixed bias | 25305.0 | 11.509764257512131% | 1065.6 |
| New full learnable bias | 25761.3 | 13.571954288993018% | 1077.8 |
| True `w/o all bias` | 25787.2 | 13.734386120421016% | 1158.6 |
| `w/o fusion gate` | 26000.9 | 14.592376703089329% | 1597.8 |
| Baseline | 27038.5 | 19.107016748752358% | 1826.7 |

Interpretation:
The stronger inference budget improves every model, but it does not make the
new full learnable-bias model best on this external benchmark. The main thesis
ablation evidence remains the fixed 1000-instance synthetic n100/g8/d1 test,
where the full learnable-bias model has the best average cost.
