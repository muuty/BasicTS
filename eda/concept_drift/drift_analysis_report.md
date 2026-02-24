# Concept Drift Detailed Analysis (Q1, Unmasked MAE)

## Cross-Year Evaluation Summary

| Train \\ Test | 2022 Q1 | 2023 Q1 | 2024 Q1 |
|---|---|---|---|
| 2022 Q1 | **10.90** | 22.43 | 12.83 |
| 2023 Q1 | 25.12 | **10.47** | 24.20 |
| 2024 Q1 | 13.15 | 23.33 | **9.85** |


## Q1: Is drift concentrated in specific nodes or gradual?

### 2023→2022 (functional nodes only, n=745)
  Overall MAE: 10.90 → 25.12 (+14.23)
  Per-node degradation stats:
    Mean: 14.1567
    Median: 8.7103
    Std: 17.9432
    Max: 117.4099 (node 38)
    Min: -6.4266 (node 834)
    Nodes worse: 743/745 (99.7%)
    Nodes better: 2/745 (0.3%)
    Worst 10% nodes account for 41.0% of total degradation
    Top 10 worst nodes: [38, 230, 865, 148, 192, 328, 406, 433, 453, 196]
      Node 38: self=23.63 → cross=141.04 (Δ=117.41, mean_flow=220.2)
      Node 230: self=44.10 → cross=155.37 (Δ=111.27, mean_flow=590.7)
      Node 865: self=12.21 → cross=118.54 (Δ=106.33, mean_flow=511.5)
      Node 148: self=3.38 → cross=107.15 (Δ=103.77, mean_flow=368.7)
      Node 192: self=3.36 → cross=103.47 (Δ=100.11, mean_flow=368.6)

### 2024→2022 (functional nodes only, n=745)
  Overall MAE: 10.90 → 13.15 (+2.26)
  Per-node degradation stats:
    Mean: 2.0544
    Median: 0.3049
    Std: 6.7762
    Max: 63.2416 (node 731)
    Min: -28.2537 (node 465)
    Nodes worse: 510/745 (68.5%)
    Nodes better: 235/745 (31.5%)
    Worst 10% nodes account for 69.4% of total degradation
    Top 10 worst nodes: [731, 728, 473, 401, 807, 156, 578, 489, 371, 848]
      Node 731: self=32.58 → cross=95.82 (Δ=63.24, mean_flow=379.6)
      Node 728: self=21.81 → cross=77.59 (Δ=55.78, mean_flow=293.9)
      Node 473: self=4.35 → cross=59.19 (Δ=54.84, mean_flow=180.0)
      Node 401: self=5.72 → cross=59.08 (Δ=53.35, mean_flow=270.0)
      Node 807: self=5.84 → cross=47.73 (Δ=41.90, mean_flow=184.8)

### 2022→2023 (functional nodes only, n=745)
  Overall MAE: 10.47 → 22.43 (+11.97)
  Per-node degradation stats:
    Mean: 14.3786
    Median: 9.9581
    Std: 15.0187
    Max: 107.4258 (node 502)
    Min: -7.6671 (node 832)
    Nodes worse: 701/745 (94.1%)
    Nodes better: 44/745 (5.9%)
    Worst 10% nodes account for 34.7% of total degradation
    Top 10 worst nodes: [502, 88, 525, 527, 739, 481, 662, 323, 149, 51]
      Node 502: self=38.10 → cross=145.52 (Δ=107.43, mean_flow=418.6)
      Node 88: self=3.40 → cross=108.71 (Δ=105.31, mean_flow=370.1)
      Node 525: self=7.60 → cross=97.07 (Δ=89.47, mean_flow=470.2)
      Node 527: self=31.69 → cross=117.17 (Δ=85.48, mean_flow=416.5)
      Node 739: self=17.49 → cross=102.77 (Δ=85.28, mean_flow=288.3)

### 2024→2023 (functional nodes only, n=745)
  Overall MAE: 10.47 → 23.33 (+12.86)
  Per-node degradation stats:
    Mean: 15.3358
    Median: 8.7079
    Std: 20.1932
    Max: 157.6208 (node 728)
    Min: -5.3884 (node 426)
    Nodes worse: 707/745 (94.9%)
    Nodes better: 38/745 (5.1%)
    Worst 10% nodes account for 41.4% of total degradation
    Top 10 worst nodes: [728, 662, 473, 583, 88, 727, 395, 412, 323, 344]
      Node 728: self=7.64 → cross=165.26 (Δ=157.62, mean_flow=470.2)
      Node 662: self=4.37 → cross=136.58 (Δ=132.21, mean_flow=352.3)
      Node 473: self=3.69 → cross=134.10 (Δ=130.41, mean_flow=370.1)
      Node 583: self=16.86 → cross=145.73 (Δ=128.87, mean_flow=326.3)
      Node 88: self=3.40 → cross=125.02 (Δ=121.62, mean_flow=370.1)

### 2022→2024 (functional nodes only, n=745)
  Overall MAE: 9.85 → 12.83 (+2.98)
  Per-node degradation stats:
    Mean: 3.0308
    Median: 1.6366
    Std: 5.3294
    Max: 41.8459 (node 741)
    Min: -15.9772 (node 115)
    Nodes worse: 692/745 (92.9%)
    Nodes better: 53/745 (7.1%)
    Worst 10% nodes account for 49.6% of total degradation
    Top 10 worst nodes: [741, 230, 456, 256, 494, 865, 270, 472, 538, 728]
      Node 741: self=2.60 → cross=44.45 (Δ=41.85, mean_flow=366.2)
      Node 230: self=4.63 → cross=43.58 (Δ=38.94, mean_flow=465.4)
      Node 456: self=22.21 → cross=59.64 (Δ=37.43, mean_flow=285.4)
      Node 256: self=11.25 → cross=47.79 (Δ=36.54, mean_flow=636.8)
      Node 494: self=5.35 → cross=38.65 (Δ=33.30, mean_flow=268.2)

### 2023→2024 (functional nodes only, n=745)
  Overall MAE: 9.85 → 24.20 (+14.35)
  Per-node degradation stats:
    Mean: 14.4228
    Median: 9.7181
    Std: 17.7060
    Max: 132.5274 (node 256)
    Min: -8.7241 (node 138)
    Nodes worse: 743/745 (99.7%)
    Nodes better: 2/745 (0.3%)
    Worst 10% nodes account for 39.1% of total degradation
    Top 10 worst nodes: [256, 38, 538, 230, 433, 328, 192, 823, 406, 196]
      Node 256: self=11.25 → cross=143.77 (Δ=132.53, mean_flow=636.8)
      Node 38: self=29.25 → cross=143.57 (Δ=114.31, mean_flow=221.0)
      Node 538: self=4.62 → cross=116.78 (Δ=112.16, mean_flow=465.4)
      Node 230: self=4.63 → cross=114.96 (Δ=110.32, mean_flow=465.4)
      Node 433: self=11.65 → cross=113.89 (Δ=102.24, mean_flow=636.8)


## Q2: Degradation by Time-of-Day and Day-of-Week

### 2023→2022
  ToD (worst degradation): 7:00 (Δ=22.29, self=14.12→cross=36.42)
  ToD (least degradation): 1:00 (Δ=4.49)
  Peak hours (7-9,16-19) avg degradation: 18.03
  Off-peak avg degradation: 12.62
  DoW (worst): Fri (Δ=15.22)
  DoW (best): Mon (Δ=12.28)
  Weekday avg degradation: 13.96
  Weekend avg degradation: 14.39

### 2024→2022
  ToD (worst degradation): 6:00 (Δ=3.60, self=13.19→cross=16.78)
  ToD (least degradation): 1:00 (Δ=0.85)
  Peak hours (7-9,16-19) avg degradation: 2.63
  Off-peak avg degradation: 2.09
  DoW (worst): Sat (Δ=2.55)
  DoW (best): Mon (Δ=1.66)
  Weekday avg degradation: 2.10
  Weekend avg degradation: 2.49

### 2022→2023
  ToD (worst degradation): 5:00 (Δ=24.04, self=10.81→cross=34.85)
  ToD (least degradation): 1:00 (Δ=2.61)
  Peak hours (7-9,16-19) avg degradation: 13.45
  Off-peak avg degradation: 11.30
  DoW (worst): Sat (Δ=13.73)
  DoW (best): Tue (Δ=8.94)
  Weekday avg degradation: 11.28
  Weekend avg degradation: 12.83

### 2024→2023
  ToD (worst degradation): 5:00 (Δ=19.78, self=10.81→cross=30.59)
  ToD (least degradation): 1:00 (Δ=3.47)
  Peak hours (7-9,16-19) avg degradation: 15.80
  Off-peak avg degradation: 11.60
  DoW (worst): Sat (Δ=14.59)
  DoW (best): Tue (Δ=9.27)
  Weekday avg degradation: 11.88
  Weekend avg degradation: 14.29

### 2022→2024
  ToD (worst degradation): 7:00 (Δ=4.77, self=11.90→cross=16.67)
  ToD (least degradation): 1:00 (Δ=0.86)
  Peak hours (7-9,16-19) avg degradation: 3.57
  Off-peak avg degradation: 2.74
  DoW (worst): Sat (Δ=3.50)
  DoW (best): Tue (Δ=2.20)
  Weekday avg degradation: 2.88
  Weekend avg degradation: 3.32

### 2023→2024
  ToD (worst degradation): 7:00 (Δ=22.24, self=11.90→cross=34.14)
  ToD (least degradation): 1:00 (Δ=4.94)
  Peak hours (7-9,16-19) avg degradation: 17.53
  Off-peak avg degradation: 13.03
  DoW (worst): Sat (Δ=16.16)
  DoW (best): Mon (Δ=12.17)
  Weekday avg degradation: 14.32
  Weekend avg degradation: 15.06


## Q3: Behavioral Insights

### 3a. Sensor health changes across years
  2022 Q1 test set: dead=142, major_fail=18, partial=136, functional=597
  2023 Q1 test set: dead=131, major_fail=22, partial=194, functional=546
  2024 Q1 test set: dead=141, major_fail=11, partial=131, functional=610

### 3b. Sensor state transitions (test set zero_rate)
  2022→2023: revived=123, died=112, both_dead=13
  2023→2024: revived=107, died=113, both_dead=23
  2022→2024: revived=47, died=48, both_dead=90

### 3c. Degradation correlates
  2022→2023 (functional nodes):
    Degradation vs mean_flow: r=0.682
    Degradation vs zero_rate: r=-0.294
    Degradation vs self_MAE:  r=0.409
  2024→2023 (functional nodes):
    Degradation vs mean_flow: r=0.652
    Degradation vs zero_rate: r=-0.245
    Degradation vs self_MAE:  r=0.390
  2022→2024 (functional nodes):
    Degradation vs mean_flow: r=0.351
    Degradation vs zero_rate: r=-0.264
    Degradation vs self_MAE:  r=0.073

### 3d. Flow distribution shift across years
  2022 Q1 (functional): mean=132.65, std=163.83, median=60.00, p95=485.00, p99=622.00
  2023 Q1 (functional): mean=148.69, std=162.80, median=84.00, p95=495.00, p99=621.00
  2024 Q1 (functional): mean=132.59, std=161.75, median=63.00, p95=483.00, p99=612.00

### 3e. Nodes with largest flow level change
  2022→2023:
    Biggest flow increase:
      Node 630: 0.0 → 635.8 (Δ=+635.8)
      Node 412: 0.0 → 627.8 (Δ=+627.8)
      Node 126: 98.1 → 653.3 (Δ=+555.2)
      Node 209: 0.0 → 507.2 (Δ=+507.2)
      Node 727: 0.0 → 470.2 (Δ=+470.2)
    Biggest flow decrease:
      Node 230: 590.7 → 0.0 (Δ=-590.7)
      Node 805: 567.9 → 24.6 (Δ=-543.3)
      Node 256: 501.0 → 25.2 (Δ=-475.9)
      Node 865: 511.5 → 55.2 (Δ=-456.3)
      Node 788: 433.7 → 2.7 (Δ=-430.9)
  2023→2024:
    Biggest flow increase:
      Node 256: 25.2 → 636.8 (Δ=+611.6)
      Node 805: 24.6 → 537.9 (Δ=+513.3)
      Node 230: 0.0 → 465.4 (Δ=+465.4)
      Node 788: 2.7 → 437.4 (Δ=+434.7)
      Node 270: 3.9 → 434.0 (Δ=+430.2)
    Biggest flow decrease:
      Node 412: 627.8 → 0.0 (Δ=-627.8)
      Node 630: 635.8 → 12.2 (Δ=-623.6)
      Node 126: 653.3 → 104.3 (Δ=-549.1)
      Node 209: 507.2 → 0.0 (Δ=-507.2)
      Node 727: 470.2 → 9.5 (Δ=-460.7)
  2022→2024:
    Biggest flow increase:
      Node 371: 83.6 → 306.3 (Δ=+222.7)
      Node 345: 15.3 → 225.0 (Δ=+209.6)
      Node 741: 162.6 → 366.2 (Δ=+203.7)
      Node 456: 83.6 → 285.4 (Δ=+201.8)
      Node 730: 24.4 → 221.3 (Δ=+196.9)
    Biggest flow decrease:
      Node 731: 379.6 → 83.0 (Δ=-296.5)
      Node 465: 451.2 → 155.5 (Δ=-295.7)
      Node 865: 511.5 → 306.8 (Δ=-204.7)
      Node 848: 278.8 → 83.0 (Δ=-195.7)
      Node 834: 450.6 → 256.0 (Δ=-194.5)
