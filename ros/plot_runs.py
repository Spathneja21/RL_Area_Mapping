#!/usr/bin/env python3
"""Plot SLAM mapped area vs time from sim_test.sh run folders.
usage: python plot_runs.py out.png LABEL=runs/sim_xxx [LABEL=runs/sim_yyy ...]"""
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[1]
for arg in sys.argv[2:]:
    label, folder = arg.split('=', 1)
    stamps, areas = [], []
    for line in open(f'{folder}/explorer.log'):
        m = re.search(r'\[(\d+\.\d+)\].*mapped area: ([\d.]+)', line)
        if m:
            stamps.append(float(m.group(1)))
            areas.append(float(m.group(2)))
    plt.plot([t - stamps[0] for t in stamps], areas, label=f'{label} ({areas[-1]:.0f} m²)')
plt.xlabel('time since explorer start (s)')
plt.ylabel('SLAM mapped area (m²)')
plt.title('MuJoCo office, 900 s')
plt.grid(alpha=.3)
plt.legend()
plt.savefig(out, dpi=90, bbox_inches='tight')
