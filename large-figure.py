"""
Electric Truck Charging — Large-Scale Figure 3
Loads experiment_data_large.pkl and produces Figure 3: Cost Breakdown.
No Exact Optimal — reference line is best Rollout cost.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────
# LOAD DATA
# ───────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
pkl_path   = os.path.join(OUTPUT_DIR, "experiment_data_large.pkl")

with open(pkl_path, 'rb') as f:
    d = pickle.load(f)

ro_vis       = d['ro_vis']
sched_fcfs_v = d['sched_fcfs_v']
sched_edf_v  = d['sched_edf_v']
sched_scdf_v = d['sched_scdf_v']

# ───────────────────────────────────────────────────────
# COLOR PALETTE
# ───────────────────────────────────────────────────────
RO_COLORS = {'FCFS': '#1abc9c', 'EDF': '#2980b9', 'SCDF': '#8e44ad'}
H_COLORS  = {'FCFS': '#e74c3c', 'EDF': '#e67e22', 'SCDF': '#f39c12'}

def cost_breakdown(sched):
    ec = sum(v['energy_cost'] for v in sched.values())
    wc = sum(v['wait_cost']   for v in sched.values())
    pc = sum(v['tard_cost']   for v in sched.values())
    return ec, wc, pc

# ═══════════════════════════════════════════════════════
# FIGURE 3 — Stacked Cost Breakdown (no Optimal)
# ═══════════════════════════════════════════════════════
br_labels = ['RO(FCFS)', 'RO(EDF)', 'RO(SCDF)', 'FCFS', 'EDF', 'SCDF']
br_scheds  = [
    ro_vis['FCFS']['sched'], ro_vis['EDF']['sched'], ro_vis['SCDF']['sched'],
    sched_fcfs_v, sched_edf_v, sched_scdf_v,
]
br_cols = [
    RO_COLORS['FCFS'], RO_COLORS['EDF'], RO_COLORS['SCDF'],
    H_COLORS['FCFS'],  H_COLORS['EDF'],  H_COLORS['SCDF'],
]

ecs, wcs, pcs, tots = [], [], [], []
for sc in br_scheds:
    ec, wc, pc = cost_breakdown(sc)
    ecs.append(ec); wcs.append(wc); pcs.append(pc); tots.append(ec+wc+pc)

fig3, ax_br = plt.subplots(figsize=(8, 5.5))
fig3.patch.set_facecolor('white')
ax_br.set_facecolor('white')
xc = np.arange(len(br_labels))

ax_br.bar(xc, ecs, 0.55, label='Energy Cost',      color='#5dade2', alpha=0.9)
ax_br.bar(xc, wcs, 0.55, bottom=ecs,               label='Waiting Cost',     color='#e67e22', alpha=0.9)
ax_br.bar(xc, pcs, 0.55,
          bottom=[e+w for e, w in zip(ecs, wcs)],
          label='Tardiness Penalty', color='#c0392b', alpha=0.9)

# Annotate cost on top of each bar
for xi, (tot, col) in enumerate(zip(tots, br_cols)):
    ax_br.text(xi, tot + max(tots)*0.012, f"€{tot:.0f}",
               ha='center', va='bottom', fontsize=10,
               fontweight='bold', color=col)

# Divider between Rollout and Heuristic
ax_br.axvline(2.5, color='#aaa', lw=1, ls=':', alpha=0.7)

ax_br.set_xticks(xc)
ax_br.set_xticklabels(br_labels, fontsize=13)
ax_br.set_ylabel("Cost [€]", fontsize=13)
ax_br.tick_params(axis='y', labelsize=13)
ax_br.legend(fontsize=11, loc='upper left')
ax_br.grid(axis='y', alpha=0.3)
ax_br.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax_br.set_ylim(0, max(tots) * 1.35)
fig3.tight_layout()

out3 = os.path.join(OUTPUT_DIR, "fig3_large_cost_breakdown.png")
fig3.savefig(out3, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print(f"Figure 3 saved → {out3}")