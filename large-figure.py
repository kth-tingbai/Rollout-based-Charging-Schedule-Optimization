"""
Electric Truck Charging — Large-Scale Figures
Loads experiment_data_large.pkl and produces:
  Fig 3 — Stacked Cost Breakdown  (no Exact Optimal; reference = best Rollout)
  Fig 4 — Total Power Profile     (absolute clock time; P_STATION=3350 kW hard limit)
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

P_STATION    = d['P_STATION']           # 3350 kW
P_OPTIONS    = d.get('P_OPTIONS', [350.0])
TOU_SCHEDULE = d['TOU_SCHEDULE']
N_VIS        = d['N_VIS']
trucks_vis   = d['trucks_vis']
ro_vis       = d['ro_vis']              # {name: {cost, sched, p_assign}}
sched_fcfs_v = d['sched_fcfs_v']
sched_edf_v  = d['sched_edf_v']
sched_scdf_v = d['sched_scdf_v']
cost_fcfs_v  = d['cost_fcfs_v']
cost_edf_v   = d['cost_edf_v']
cost_scdf_v  = d['cost_scdf_v']

# ───────────────────────────────────────────────────────
# COLOR PALETTE
# ───────────────────────────────────────────────────────
RO_COLORS = {'FCFS': '#1abc9c', 'EDF': '#2980b9', 'SCDF': '#8e44ad'}
H_COLORS  = {'FCFS': '#e74c3c', 'EDF': '#e67e22', 'SCDF': '#f39c12'}

TOU_BAND_COLORS = {
    0.101: '#d5f5e3',
    0.174: '#fadbd8',
    0.128: '#fef9e7',
    0.110: '#eaf4fb',
    0.202: '#fdebd0',
}

LINE_STYLES = {
    'RO(FCFS)': (RO_COLORS['FCFS'], '-',  2.0),
    'RO(EDF)':  (RO_COLORS['EDF'],  '-',  2.0),
    'RO(SCDF)': (RO_COLORS['SCDF'], '-',  2.0),
    'FCFS':     (H_COLORS['FCFS'],  '--', 1.6),
    'EDF':      (H_COLORS['EDF'],   '--', 1.6),
    'SCDF':     (H_COLORS['SCDF'],  '--', 1.6),
}

# ───────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────
def cost_breakdown(sched):
    ec = sum(v['energy_cost'] for v in sched.values())
    wc = sum(v['wait_cost']   for v in sched.values())
    pc = sum(v['tard_cost']   for v in sched.values())
    return ec, wc, pc


def total_power(sched, tg):
    """Total station power at each grid point using info['P'] per truck."""
    pw = np.zeros(len(tg))
    for info in sched.values():
        mask = (tg >= info['t_s']) & (tg < info['t_d'])
        pw[mask] += info['P']
    return pw


def add_tou_bands(ax, x_lo, x_hi, alpha=0.18, zorder=0):
    for s, e, p in TOU_SCHEDULE:
        ov_s = max(x_lo, float(s))
        ov_e = min(x_hi, float(e))
        if ov_e > ov_s:
            ax.axvspan(ov_s, ov_e,
                       alpha=alpha,
                       color=TOU_BAND_COLORS.get(p, '#eeeeee'),
                       zorder=zorder)


def h_to_hhmm(h):
    """Fractional hours → 'HH:MM' string."""
    h_mod = h % 24
    hh    = int(h_mod)
    mm    = int(round((h_mod - hh) * 60))
    if mm == 60:
        hh += 1; mm = 0
    return f"{hh:02d}:{mm:02d}"


# ═══════════════════════════════════════════════════════
# FIGURE 3 — Stacked Cost Breakdown
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

ax_br.bar(xc, ecs, 0.55, label='Energy Cost',       color='#5dade2', alpha=0.9)
ax_br.bar(xc, wcs, 0.55, bottom=ecs,                label='Waiting Cost',      color='#e67e22', alpha=0.9)
ax_br.bar(xc, pcs, 0.55,
          bottom=[e+w for e, w in zip(ecs, wcs)],
          label='Tardiness Penalty', color='#c0392b', alpha=0.9)

for xi, tot in enumerate(tots):
    ax_br.text(xi, tot + max(tots)*0.012,
               f"€{tot:,.0f}",
               ha='center', va='bottom', fontsize=11,
               color='black')

ax_br.axvline(2.5, color='#aaa', lw=1, ls=':', alpha=0.7)
ax_br.set_xticks(xc)
ax_br.set_xticklabels(br_labels, fontsize=13)
ax_br.set_ylabel("Cost [€]", fontsize=13)
ax_br.tick_params(axis='y', labelsize=13)
ax_br.legend(fontsize=11, loc='upper left')
ax_br.grid(axis='y', alpha=0.3)
ax_br.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax_br.set_ylim(0, max(tots) * 1.40)
fig3.tight_layout()

out3 = os.path.join(OUTPUT_DIR, "fig3_large_cost_breakdown.png")
fig3.savefig(out3, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print(f"Figure 3 saved → {out3}")


# ═══════════════════════════════════════════════════════
# FIGURE 4 — Total Power Profile  (absolute clock time)
# ═══════════════════════════════════════════════════════
PANEL_ITEMS = [
    ('RO(FCFS)', ro_vis['FCFS']['sched'], ro_vis['FCFS']['cost']),
    ('RO(EDF)',  ro_vis['EDF']['sched'],  ro_vis['EDF']['cost']),
    ('RO(SCDF)', ro_vis['SCDF']['sched'], ro_vis['SCDF']['cost']),
]

# Time axis bounds from all schedules
all_t = []
for _, sched, _ in PANEL_ITEMS:
    for info in sched.values():
        all_t += [info['t_s'], info['t_d']]
t_lo   = min(all_t) - 0.05
t_hi   = max(all_t) + 0.10
t_grid = np.linspace(t_lo, t_hi, 8000)

fig4, ax4 = plt.subplots(figsize=(8, 6))
fig4.patch.set_facecolor('white')
ax4.set_facecolor('white')

add_tou_bands(ax4, t_lo, t_hi, alpha=0.20, zorder=0)

for name, sched, total_cost in PANEL_ITEMS:
    ec  = sum(v['energy_cost'] for v in sched.values())
    pw  = total_power(sched, t_grid)
    col, ls, lw = LINE_STYLES[name]

    # Sanity check
    if pw.max() > P_STATION + 1.0:
        print(f"  ⚠ WARNING: {name} exceeds {P_STATION} kW "
              f"(max={pw.max():.1f} kW) — check solver!")

    ax4.step(t_grid, pw / 1000, where='post',
             color=col, lw=lw, ls=ls,
             label=f"{name}  (Cost: €{ec:,.1f})", zorder=3)

# Hard constraint line — from P_STATION variable, never hardcoded
ax4.axhline(P_STATION / 1000, color='#e74c3c', lw=2.2, ls=':',
            zorder=7, label=f'Station limit {P_STATION/1000:.2f} MW')

# x-axis: absolute clock time HH:MM
xticks = np.linspace(t_lo, t_hi, 10)
ax4.set_xticks(xticks)
ax4.set_xticklabels([h_to_hhmm(t) for t in xticks], fontsize=12)
ax4.set_xlabel("Time of Day", fontsize=12)
ax4.set_ylabel("Total Charging Power [MW]", fontsize=12)
ax4.tick_params(axis='y', labelsize=12)
ax4.set_xlim(t_lo, t_hi)

# y-axis ceiling slightly above limit to make any violation immediately visible
ax4.set_ylim(0, P_STATION / 1000 * 1.20)

ax4.legend(fontsize=11, loc='upper right', framealpha=0.92, ncol=2)
ax4.grid(alpha=0.30, zorder=0)
fig4.tight_layout()

out4 = os.path.join(OUTPUT_DIR, "fig4_large_power_profiles.png")
fig4.savefig(out4, dpi=500, bbox_inches='tight', facecolor='white')
plt.close(fig4)
print(f"Figure 4 saved → {out4}")

print("\nAll figures saved to:", OUTPUT_DIR)