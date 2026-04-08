"""
Electric Truck Charging — Figure Generation
Loads experiment_data.pkl and produces 4 publication figures.

Compatible with new data format:
  - P_OPTIONS = [300, 350] kW per truck (discrete choice, fixed per session)
  - P_STATION = 1000 kW hard constraint
  - p_assign stored per schedule: {tid: power_kW}
  - Gantt shows chosen power level per truck bar
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ───────────────────────────────────────────────────────
# LOAD DATA
# ───────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
pkl_path   = os.path.join(OUTPUT_DIR, "experiment_data.pkl")

with open(pkl_path, 'rb') as f:
    d = pickle.load(f)

C            = d['C']
P_STATION    = d['P_STATION']          # 1000 kW
P_OPTIONS    = d.get('P_OPTIONS', [350.0])   # [300, 350] kW
E_MAX        = d['E_MAX']
P_MAX        = d['P_MAX']
EPSILON      = d['EPSILON']
GAMMA        = d['GAMMA']
N_LIST       = d['N_LIST']
K            = d['K']
TOU_SCHEDULE = d['TOU_SCHEDULE']
results      = d['results']
results_opt  = d['results_opt']
METHODS      = d['METHODS']
N_VIS        = d['N_VIS']
trucks_vis   = d['trucks_vis']
J_opt_v      = d['J_opt_v']
sched_opt_v  = d['sched_opt_v']
p_opt_v      = d.get('p_opt_v', {tr['id']: P_MAX for tr in trucks_vis})
ro_vis       = d['ro_vis']             # {name: {cost, sched, p_assign}}
cost_fcfs_v  = d['cost_fcfs_v']; sched_fcfs_v = d['sched_fcfs_v']
cost_edf_v   = d['cost_edf_v'];  sched_edf_v  = d['sched_edf_v']
cost_scdf_v  = d['cost_scdf_v']; sched_scdf_v = d['sched_scdf_v']
p_fcfs_v     = d.get('p_fcfs_v', {})
p_edf_v      = d.get('p_edf_v',  {})
p_scdf_v     = d.get('p_scdf_v', {})

TOU_EXT = TOU_SCHEDULE + [(s+24, e+24, p) for s, e, p in TOU_SCHEDULE]

# ───────────────────────────────────────────────────────
# COLOR PALETTE
# ───────────────────────────────────────────────────────
PORT_COLORS = ['#3498db', '#bdc3c7', '#2ecc71']  # Port 1 → light grey
RO_COLORS   = {'FCFS': '#1abc9c', 'EDF': '#2980b9', 'SCDF': '#8e44ad'}
H_COLORS    = {'FCFS': '#e74c3c', 'EDF': '#e67e22', 'SCDF': '#f39c12'}
OPT_COLOR   = '#2c3e50'

TOU_BAND_COLORS = {
    0.101: '#d5f5e3',
    0.174: '#fadbd8',
    0.128: '#fef9e7',
    0.110: '#eaf4fb',
    0.202: '#fdebd0',
}

TRUCK_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#9b59b6',
    '#f39c12', '#1abc9c', '#e67e22', '#34495e',
]

M_STYLES = {
    'ro_fcfs': (RO_COLORS['FCFS'], 'o-',  'RO(FCFS)'),
    'ro_edf':  (RO_COLORS['EDF'],  's-',  'RO(EDF)'),
    'ro_scdf': (RO_COLORS['SCDF'], '^-',  'RO(SCDF)'),
    'fcfs':    (H_COLORS['FCFS'],  'o--', 'FCFS'),
    'edf':     (H_COLORS['EDF'],   's--', 'EDF'),
    'scdf':    (H_COLORS['SCDF'],  '^--', 'SCDF'),
}


def add_tou_bands(ax, x_lo, x_hi, alpha=0.18, zorder=0):
    for s, e, p in TOU_SCHEDULE:
        ov_s = max(x_lo, float(s))
        ov_e = min(x_hi, float(e))
        if ov_e > ov_s:
            ax.axvspan(ov_s, ov_e,
                       alpha=alpha,
                       color=TOU_BAND_COLORS.get(p, '#eeeeee'),
                       zorder=zorder)


def cost_breakdown(sched):
    ec = sum(v['energy_cost'] for v in sched.values())
    wc = sum(v['wait_cost']   for v in sched.values())
    pc = sum(v['tard_cost']   for v in sched.values())
    return ec, wc, pc


def total_power(sched, tg):
    """Compute total station power at each time point.
    Uses info['P'] which stores the actual chosen power level per truck."""
    pw = np.zeros(len(tg))
    for tid, info in sched.items():
        mask = (tg >= info['t_s']) & (tg < info['t_d'])
        pw[mask] += info['P']   # info['P'] = 300 or 350 kW (chosen level)
    return pw


# ═══════════════════════════════════════════════════════
# FIGURE 1 — Gantt chart  (a) best Rollout  (b) Optimal
# Shows per-truck power level in bar label
# ═══════════════════════════════════════════════════════
best_ro    = min(ro_vis, key=lambda k: ro_vis[k]['cost'])
sched_best = ro_vis[best_ro]['sched']
print(f"Fig 1(a) uses: RO({best_ro}),  cost = {ro_vis[best_ro]['cost']:.4f} €")
print(f"Optimal cost = {J_opt_v:.4f} €")


def draw_gantt(ax, sched, trucks_vis, C, x_lo=None, x_hi=None):
    ax.set_facecolor('white')
    all_ts = [info['t_s'] for info in sched.values()]
    all_td = [info['t_d'] for info in sched.values()]
    if x_lo is None: x_lo = max(0, min(all_ts) - 0.4)
    if x_hi is None: x_hi = max(all_td) + 0.6

    # No TOU background bands — plain white

    bar_height = 0.50

    # Track vertical rank per port to avoid annotation overlap
    port_order = {}
    for port in range(C):
        tids_in_port = sorted(
            [tid for tid, info in sched.items() if info['port'] == port],
            key=lambda tid: sched[tid]['t_s']
        )
        for rank, tid in enumerate(tids_in_port):
            port_order[tid] = rank

    for tid, info in sched.items():
        port  = info['port']
        t_s   = info['t_s']
        t_d   = info['t_d']
        P_i   = info['P']
        tr    = trucks_vis[tid]
        a_i   = tr['a']
        d_i   = tr['d']
        tardy = t_d > d_i + 1e-6
        tc    = TRUCK_COLORS[tid % len(TRUCK_COLORS)]

        # Main charging bar (full duration, solid color)
        ax.barh(port, t_d - t_s, left=t_s, height=bar_height,
                color=PORT_COLORS[port], alpha=0.85,
                edgecolor='white', lw=0.8, zorder=2)

        # Hatch ONLY the tardy portion [d_i, t_d]
        if tardy:
            tard_start = max(t_s, d_i)   # should equal d_i when tardy
            tard_dur   = t_d - tard_start
            ax.barh(port, tard_dur, left=tard_start, height=bar_height,
                    color='none', edgecolor='#c0392b',
                    hatch='///', linewidth=0, alpha=0.70, zorder=3)
            ax.annotate('', xy=(t_d, port), xytext=(d_i, port),
                        arrowprops=dict(arrowstyle='->', color='#c0392b',
                                        lw=1.8, shrinkA=0, shrinkB=0),
                        zorder=5)

        # Bar label: truck id + chosen power level
        p_label = f"T{tid}\n{int(P_i)}kW"
        ax.text((t_s + t_d) / 2, port, p_label,
                ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', zorder=4,
                path_effects=[pe.withStroke(linewidth=2, foreground='#00000066')])

        # Arrival ^ and deadline v brackets
        rank  = port_order[tid]
        y_ann = port + bar_height / 2 + 0.10 + rank * 0.18
        ax.plot(a_i, y_ann, marker='^', ms=10, color=tc,
                markeredgecolor='white', markeredgewidth=0.7, zorder=6)
        ax.plot(d_i, y_ann, marker='v', ms=10, color=tc,
                markeredgecolor='white', markeredgewidth=0.7, zorder=6)
        ax.plot([a_i, d_i], [y_ann, y_ann], color=tc, lw=1.4, alpha=0.7, zorder=5)

    ax.set_yticks(range(C))
    ax.set_yticklabels([f"Port {c}" for c in range(C)], fontsize=14)
    ax.set_xlabel("Time [hours]", fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.6, C + 0.8)
    ax.grid(axis='x', alpha=0.30, ls='--', zorder=1)

    legend_handles = [
        mpatches.Patch(color=PORT_COLORS[c], alpha=0.85, label=f'Port {c}')
        for c in range(C)
    ] + [
        Line2D([0], [0], color='gray', marker='^', ls='-', ms=11,
               label='Arrival → Deadline (per-truck color)'),
        mpatches.Patch(facecolor='none', edgecolor='#c0392b',
                       hatch='///', label='Tardy (miss deadline)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=12,
              ncol=2, framealpha=0.9)


# Shared x-axis range across both Gantt panels
all_ts_both = ([info['t_s'] for info in sched_best.values()] +
               [info['t_s'] for info in sched_opt_v.values()])
all_td_both = ([info['t_d'] for info in sched_best.values()] +
               [info['t_d'] for info in sched_opt_v.values()])
x_lo_shared = max(0, min(all_ts_both) - 0.4)
x_hi_shared = max(all_td_both) + 0.6

# Fig 1(a) — best Rollout
fig1a, ax_a = plt.subplots(figsize=(8, 5.5))
fig1a.patch.set_facecolor('white')
draw_gantt(ax_a, sched_best, trucks_vis, C, x_lo_shared, x_hi_shared)
fig1a.tight_layout()
out1a = os.path.join(OUTPUT_DIR, "fig1a_gantt_rollout.png")
fig1a.savefig(out1a, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig1a)
print(f"Figure 1(a) saved → {out1a}")

# Fig 1(b) — Optimal
fig1b, ax_b = plt.subplots(figsize=(8, 5.5))
fig1b.patch.set_facecolor('white')
draw_gantt(ax_b, sched_opt_v, trucks_vis, C, x_lo_shared, x_hi_shared)
fig1b.tight_layout()
out1b = os.path.join(OUTPUT_DIR, "fig1b_gantt_optimal.png")
fig1b.savefig(out1b, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig1b)
print(f"Figure 1(b) saved → {out1b}")


# ═══════════════════════════════════════════════════════
# FIGURE 2 — Optimality Gap vs N
# N=4..7 from pkl (K=3, mean±std shading)
# N=8    from CSV (K=1, single point, no shading)
# ═══════════════════════════════════════════════════════

# ── Load N=8 gap from existing CSV ──────────────────────
import glob, csv as _csv

N8_GAPS = {}   # {mkey: gap_pct}
# Map CSV column names to METHODS keys
_col_map = {
    'gap_ro_fcfs_pct': 'ro_fcfs',
    'gap_ro_edf_pct':  'ro_edf',
    'gap_ro_scdf_pct': 'ro_scdf',
    'gap_fcfs_pct':    'fcfs',
    'gap_edf_pct':     'edf',
    'gap_scdf_pct':    'scdf',
}
_csv_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'charging_results_*.csv')))
if _csv_files:
    _latest = _csv_files[-1]   # most recent CSV
    with open(_latest, newline='') as _f:
        reader = _csv.DictReader(_f)
        for row in reader:
            if int(row['N']) == 8:
                for col, mkey in _col_map.items():
                    N8_GAPS[mkey] = float(row[col])
                break
    print(f"N=8 gaps loaded from: {_latest}")
else:
    # Fallback: hard-coded from the provided CSV
    N8_GAPS = {
        'ro_fcfs': 80.313, 'ro_edf':  0.000, 'ro_scdf': 16.587,
        'fcfs':    81.636, 'edf':     46.246, 'scdf':   131.547,
    }
    print("No CSV found — using hard-coded N=8 gaps.")

# ── Combined N axis ──────────────────────────────────────
N_ALL = N_LIST + [8]   # e.g. [4,5,6,7] + [8]

fig2, ax_gap = plt.subplots(figsize=(8, 5.7))
fig2.patch.set_facecolor('white')
ax_gap.set_facecolor('white')

for mkey, (col, ls, lbl) in M_STYLES.items():
    # N=4..7: mean ± std from pkl (K=3)
    means = [np.mean(results[n][mkey]['gap']) for n in N_LIST]
    stds  = [np.std( results[n][mkey]['gap']) for n in N_LIST]

    # N=8: single value from CSV (std=0, no shading)
    means.append(N8_GAPS.get(mkey, 0.0))
    stds.append(0.0)

    means = np.array(means)
    stds  = np.array(stds)

    ax_gap.plot(N_ALL, means, ls, color=col, lw=2.2, ms=10,
                label=lbl, zorder=2)

ax_gap.axhline(0, color='k', lw=1.2, ls='--', alpha=0.5,
               label='Optimal (0%)', zorder=3)
ax_gap.set_xticks(N_ALL)
ax_gap.set_xticklabels([f"N={n}" for n in N_ALL], fontsize=13)
ax_gap.set_xlabel("Fleet Size N", fontsize=13)
ax_gap.set_ylabel("Optimality Gap [%]", fontsize=13)
ax_gap.tick_params(axis='y', labelsize=13)
ax_gap.legend(fontsize=11, ncol=4, loc='upper left')
ax_gap.grid(alpha=0.3, zorder=0)
ax_gap.set_ylim(bottom=0)
ax_gap.set_xlim(N_ALL[0] - 0.3, N_ALL[-1] + 0.3)
fig2.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "fig2_gap_vs_N.png")
fig2.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"Figure 2 saved → {out2}")


# ═══════════════════════════════════════════════════════
# FIGURE 3 — Stacked Cost Breakdown
# ═══════════════════════════════════════════════════════
br_labels = ['Optimal',
             'RO(FCFS)', 'RO(EDF)', 'RO(SCDF)',
             'FCFS', 'EDF', 'SCDF']
br_scheds  = [sched_opt_v,
              ro_vis['FCFS']['sched'], ro_vis['EDF']['sched'],
              ro_vis['SCDF']['sched'],
              sched_fcfs_v, sched_edf_v, sched_scdf_v]
br_cols    = [OPT_COLOR,
              RO_COLORS['FCFS'], RO_COLORS['EDF'], RO_COLORS['SCDF'],
              H_COLORS['FCFS'],  H_COLORS['EDF'],  H_COLORS['SCDF']]

ecs, wcs, pcs, tots = [], [], [], []
for sc in br_scheds:
    ec, wc, pc = cost_breakdown(sc)
    ecs.append(ec); wcs.append(wc); pcs.append(pc); tots.append(ec+wc+pc)

fig3, ax_br = plt.subplots(figsize=(8, 5.7))
fig3.patch.set_facecolor('white')
ax_br.set_facecolor('white')
xc = np.arange(len(br_labels))

ax_br.bar(xc, ecs, 0.55, label='Energy Cost',        color='#5dade2', alpha=0.9)
ax_br.bar(xc, wcs, 0.55, bottom=ecs,                 label='Waiting Cost',     color='#e67e22', alpha=0.9)
ax_br.bar(xc, pcs, 0.55,
          bottom=[e+w for e, w in zip(ecs, wcs)],
          label='Tardiness Penalty', color='#c0392b', alpha=0.9)
ax_br.axhline(J_opt_v, color=OPT_COLOR, lw=2, ls='--', alpha=0.8, label='Optimal')

for xi, (tot, col) in enumerate(zip(tots, br_cols)):
    lbl = (f"€{tot:.0f}" if xi == 0
           else f"€{tot:.0f}\n(+{(tot-J_opt_v)/J_opt_v*100:.1f}%)")
    ax_br.text(xi, tot + max(tots)*0.012, lbl,
               ha='center', va='bottom', fontsize=10, fontweight='bold', color=col)

ax_br.axvline(0.5, color='#aaa', lw=1, ls=':', alpha=0.7)
ax_br.axvline(3.5, color='#aaa', lw=1, ls=':', alpha=0.7)
ax_br.set_xticks(xc)
ax_br.set_xticklabels(br_labels, fontsize=13)
ax_br.set_ylabel("Cost [€]", fontsize=13)
ax_br.tick_params(axis='y', labelsize=13)
ax_br.legend(fontsize=11, loc='upper left')
ax_br.grid(axis='y', alpha=0.3)
ax_br.set_ylim(0, max(tots) * 1.35)
fig3.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "fig3_cost_breakdown.png")
fig3.savefig(out3, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print(f"Figure 3 saved → {out3}")


# ═══════════════════════════════════════════════════════
# FIGURE 4 — Power Profiles
# Uses info['P'] (actual chosen power) → hard cap at P_STATION=1000 kW
# ═══════════════════════════════════════════════════════
PANEL_ITEMS = [
    ('Optimal',  sched_opt_v,             OPT_COLOR),
    ('RO(FCFS)', ro_vis['FCFS']['sched'], RO_COLORS['FCFS']),
    ('RO(EDF)',  ro_vis['EDF']['sched'],  RO_COLORS['EDF']),
    ('RO(SCDF)', ro_vis['SCDF']['sched'], RO_COLORS['SCDF']),
    ('FCFS',     sched_fcfs_v,            H_COLORS['FCFS']),
    ('EDF',      sched_edf_v,             H_COLORS['EDF']),
    ('SCDF',     sched_scdf_v,            H_COLORS['SCDF']),
]

all_t = []
for _, sched, _ in PANEL_ITEMS:
    for info in sched.values():
        all_t += [info['t_s'], info['t_d']]
t_lo   = min(all_t) - 0.02
t_hi   = max(all_t) + 0.05
t_grid = np.linspace(t_lo, t_hi, 5000)
t_ref  = min(trucks_vis[i]['a'] for i in range(N_VIS))

LINE_STYLES = {
    'Optimal':  (OPT_COLOR,           '-',  3.0),
    'RO(FCFS)': (RO_COLORS['FCFS'],   '-',  1.8),
    'RO(EDF)':  (RO_COLORS['EDF'],    '-',  1.8),
    'RO(SCDF)': (RO_COLORS['SCDF'],   '-',  1.8),
    'FCFS':     (H_COLORS['FCFS'],    '--', 1.6),
    'EDF':      (H_COLORS['EDF'],     '--', 1.6),
    'SCDF':     (H_COLORS['SCDF'],    '--', 1.6),
}

fig4, ax4 = plt.subplots(figsize=(12, 5))
fig4.patch.set_facecolor('#f8f9fa')
ax4.set_facecolor('#fdfdfd')

add_tou_bands(ax4, t_lo, t_hi, alpha=0.20, zorder=0)

for name, sched, _ in PANEL_ITEMS:
    pw  = total_power(sched, t_grid)    # uses info['P'] per truck
    col, ls, lw = LINE_STYLES[name]
    total_cost = sum(v['energy_cost']+v['wait_cost']+v['tard_cost']
                     for v in sched.values())
    gap_str = (f" (+{(total_cost-J_opt_v)/J_opt_v*100:.1f}%)"
               if name != 'Optimal' else " (optimal)")

    # Sanity check: warn if any point exceeds P_STATION
    violation = pw[pw > P_STATION + 1.0]
    if len(violation) > 0:
        print(f"  ⚠ WARNING: {name} exceeds {P_STATION} kW "
              f"(max={pw.max():.1f} kW) — check solver!")

    ax4.step(t_grid, pw/1000, where='post', color=col, lw=lw, ls=ls,
             label=f"{name}{gap_str}",
             zorder=6 if name == 'Optimal' else 3)

# Hard constraint line — always reads from P_STATION (no hardcoding)
ax4.axhline(P_STATION/1000, color='#e74c3c', lw=2.2, ls=':',
            zorder=7, label=f'Station limit {P_STATION/1000:.1f} MW')

# x-axis: absolute clock time (HH:MM)
def h_to_hhmm(h):
    """Convert fractional hours to 'HH:MM' string."""
    h_mod = h % 24
    hh = int(h_mod)
    mm = int(round((h_mod - hh) * 60))
    if mm == 60:
        hh += 1; mm = 0
    return f"{hh:02d}:{mm:02d}"

xticks = np.linspace(t_lo, t_hi, 9)
ax4.set_xticks(xticks)
ax4.set_xticklabels([h_to_hhmm(t) for t in xticks], fontsize=10)
ax4.set_xlabel("Time of Day", fontsize=11)
ax4.set_ylabel("Total Charging Power [MW]", fontsize=11)
ax4.set_xlim(t_lo, t_hi)

# y-axis: ceiling just above P_STATION to make violations obvious if they occur
ax4.set_ylim(0, P_STATION/1000 * 1.20)

ax4.legend(fontsize=9, loc='upper right', framealpha=0.92, ncol=2)
ax4.grid(alpha=0.30, zorder=0)
fig4.tight_layout()
out4 = os.path.join(OUTPUT_DIR, "fig4_power_profiles.png")
fig4.savefig(out4, dpi=150, bbox_inches='tight', facecolor=fig4.get_facecolor())
plt.close(fig4)
print(f"Figure 4 saved → {out4}")

print("\nAll figures saved to:", OUTPUT_DIR)