#!/usr/bin/env python3
"""
Known:
  81 cells wt[1,25] = ~170s max
  90 cells wt[1,30] = 1331s
  
Need 240-600s. Target 85-90 cells with wt[1,30]-[1,40].
"""
import random, os

OUT_DIR = "./mapa_test2"
os.makedirs(OUT_DIR, exist_ok=True)

configs = [
    # 85 cells, wt[1,30] — between 81-cell 170s and 90-cell 1331s
    ("m_5x17_w30_a.txt",  5,17, 1, 30, 11001),
    ("m_5x17_w30_b.txt",  5,17, 1, 30, 11002),
    ("m_17x5_w30_a.txt", 17, 5, 1, 30, 11003),
    ("m_17x5_w30_b.txt", 17, 5, 1, 30, 11004),

    # 84 cells, wt[1,25] — more cells than 81 + very low weights
    ("m_7x12_w25_a.txt",  7,12, 1, 25, 12001),
    ("m_7x12_w25_b.txt",  7,12, 1, 25, 12002),
    ("m_12x7_w25_a.txt", 12, 7, 1, 25, 12003),
    ("m_12x7_w25_b.txt", 12, 7, 1, 25, 12004),

    # 84 cells, wt[1,30]
    ("m_7x12_w30_a.txt",  7,12, 1, 30, 13001),
    ("m_7x12_w30_b.txt",  7,12, 1, 30, 13002),
    ("m_12x7_w30_a.txt", 12, 7, 1, 30, 13003),
    ("m_12x7_w30_b.txt", 12, 7, 1, 30, 13004),

    # 88 cells, wt[1,30] — close to 90, should be long
    ("m_8x11_w30_a.txt",  8,11, 1, 30, 14001),
    ("m_8x11_w30_b.txt",  8,11, 1, 30, 14002),
    ("m_8x11_w30_c.txt",  8,11, 1, 30, 14003),
    ("m_11x8_w30_a.txt", 11, 8, 1, 30, 14004),
    ("m_11x8_w30_b.txt", 11, 8, 1, 30, 14005),

    # 88 cells, wt[1,35]
    ("m_8x11_w35_a.txt",  8,11, 1, 35, 15001),
    ("m_8x11_w35_b.txt",  8,11, 1, 35, 15002),
    ("m_11x8_w35_a.txt", 11, 8, 1, 35, 15003),
    ("m_11x8_w35_b.txt", 11, 8, 1, 35, 15004),

    # 90 cells, wt[1,35] — same as your 1331s board but higher weights
    ("m_9x10_w35_a.txt",  9,10, 1, 35, 16001),
    ("m_9x10_w35_b.txt",  9,10, 1, 35, 16002),
    ("m_10x9_w35_a.txt", 10, 9, 1, 35, 16003),
    ("m_10x9_w35_b.txt", 10, 9, 1, 35, 16004),

    # 90 cells, wt[1,40] — even more pruning friendly
    ("m_9x10_w40_a.txt",  9,10, 1, 40, 17001),
    ("m_9x10_w40_b.txt",  9,10, 1, 40, 17002),
    ("m_10x9_w40_a.txt", 10, 9, 1, 40, 17003),
    ("m_10x9_w40_b.txt", 10, 9, 1, 40, 17004),

    # 90 cells, wt[1,50] — might land in the sweet spot
    ("m_9x10_w50_a.txt",  9,10, 1, 50, 18001),
    ("m_10x9_w50_a.txt", 10, 9, 1, 50, 18002),
]

print(f"{'File':<25s}  {'Size':>6s}  {'Cells':>5s}  {'Weights':>8s}")
print("-" * 52)

for name, rows, cols, lo, hi, seed in configs:
    random.seed(seed)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        f.write(f"{rows} {cols}\n")
        for r in range(rows):
            f.write(" ".join(str(random.randint(lo, hi)) for _ in range(cols)) + "\n")
    print(f"{name:<25s}  {rows}x{cols:>2d}  {rows*cols:>5d}  [{lo},{hi}]")

print(f"\nTotal: {len(configs)} maps in {OUT_DIR}/")