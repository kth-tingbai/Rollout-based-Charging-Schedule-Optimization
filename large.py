"""
Electric Truck Charging Scheduling — Large-Scale Experiment
  C=10, N=25/50/75/100/125, K=1
  6 methods: Rollout(FCFS/EDF/SCDF) + Heuristic(FCFS/EDF/SCDF)
  No Exact Optimal
  Per-truck power: chosen from P_OPTIONS={300,350} kW, fixed for entire session
  Hard power constraint: P_STATION = 3350 kW
  Arrivals: uniform [06:00, 12:00), each truck independent
  Deadline: d_i = a_i + 2.0 * delta_E_i / P_max_i
  Saves all data to outputs/experiment_data_large.pkl

Power selection strategy:
  - Rollout : decision space = (truck, port, power_level), greedy lookahead
  - Heuristic: after fixing port assignment (omega), greedy per-truck power
               selection in scheduling order — O(N) extra solves, no 2^N enum
"""

import os
import datetime
import numpy as np
import pickle
import csv
from copy import deepcopy
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# ═══════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════
C              = 10
P_OPTIONS      = [300.0, 350.0]   # kW — two discrete power levels per truck
P_STATION      = 3350.0           # kW hard constraint (10×350=3500 > 3350)
E_MAX          = 468.0
P_MAX          = 350.0            # used for deadline computation & lookahead default
SOC_LO         = 0.20
SOC_HI         = 0.80
EPSILON        = 120.0
GAMMA          = 600.0
DEADLINE_SLACK = 2.0
ARRIVAL_LO     = 6.0
ARRIVAL_HI     = 12.0

N_LIST = [25]
K      = 1

TOU_SCHEDULE = [
    (0,  6,  0.101),
    (6,  9,  0.174),
    (9,  12, 0.128),
    (12, 17, 0.110),
    (17, 21, 0.202),
    (21, 24, 0.101),
]
TOU_EXT = TOU_SCHEDULE + [(s+24, e+24, p) for s, e, p in TOU_SCHEDULE]


# ═══════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════
def energy_cost_integral(t_s, t_d, P_i):
    cost = 0.0
    for s, e, p in TOU_EXT:
        ov_s = max(t_s, float(s))
        ov_e = min(t_d, float(e))
        if ov_e > ov_s:
            cost += p * P_i * (ov_e - ov_s)
    return cost


def generate_trucks(N, seed=42):
    rng = np.random.default_rng(seed)
    trucks = []
    for i in range(N):
        a_i     = rng.uniform(ARRIVAL_LO, ARRIVAL_HI)
        E0_i    = rng.uniform(SOC_LO, SOC_HI) * E_MAX
        delta_E = E_MAX - E0_i
        d_i     = a_i + DEADLINE_SLACK * delta_E / P_MAX
        trucks.append(dict(
            id=i, a=a_i, E0=E0_i, E_max=E_MAX,
            delta_E=delta_E, d=d_i, P_max=P_MAX,
            epsilon=EPSILON, gamma=GAMMA,
        ))
    return trucks


# ═══════════════════════════════════════════════════════
# INNER LAYER SOLVER  (hard P_STATION constraint)
# ═══════════════════════════════════════════════════════
def solve_inner_layer(omega, trucks, p_assign=None):
    """
    Schedule trucks given:
      omega    – port sequences (list of C lists of truck-ids)
      trucks   – truck data list
      p_assign – dict {tid: power_kW} from P_OPTIONS.
                 Defaults to P_MAX for all trucks when None.
    Hard constraint: sum of concurrent powers <= P_STATION at all times.
    """
    if p_assign is None:
        p_assign = {tr['id']: P_MAX for tr in trucks}

    tentative = {}
    port_rel  = [0.0] * len(omega)

    # Pass 1: tentative start times per port
    for port_idx, port_seq in enumerate(omega):
        for tid in port_seq:
            tr  = trucks[tid]
            P_i = p_assign.get(tid, P_MAX)
            t_s = max(tr['a'], port_rel[port_idx])
            t_d = t_s + tr['delta_E'] / P_i
            port_rel[port_idx] = t_d
            tentative[tid] = dict(port=port_idx, t_s=t_s)

    order = sorted(tentative, key=lambda tid: tentative[tid]['t_s'])

    schedule  = {}
    port_rel2 = [0.0] * len(omega)
    intervals = []   # committed: (t_s, t_d, P)
    total     = 0.0

    def is_feasible_interval(t_s, t_d, P_i):
        pts = sorted(set(
            [t_s, t_d] +
            [ts for ts, td, _ in intervals if t_s < ts < t_d] +
            [td for ts, td, _ in intervals if t_s < td < t_d]
        ))
        for i in range(len(pts) - 1):
            mid   = 0.5 * (pts[i] + pts[i + 1])
            power = sum(P for ts, td, P in intervals if ts <= mid < td)
            if power + P_i > P_STATION + 1e-6:
                return False
        return True

    def find_earliest_start(t_ready, dur, P_i):
        t = t_ready
        while True:
            if is_feasible_interval(t, t + dur, P_i):
                return t
            future = [td for ts, td, _ in intervals if td > t]
            if not future:
                return t
            t = min(future)

    for tid in order:
        tr       = trucks[tid]
        P_i      = p_assign.get(tid, P_MAX)
        port_idx = tentative[tid]['port']
        dur      = tr['delta_E'] / P_i
        t_ready  = max(tr['a'], port_rel2[port_idx])
        t_s      = find_earliest_start(t_ready, dur, P_i)
        t_d      = t_s + dur
        port_rel2[port_idx] = t_d

        ec = energy_cost_integral(t_s, t_d, P_i)
        wc = tr['epsilon'] * (t_s - tr['a'])
        pc = tr['gamma']   * max(t_d - tr['d'], 0.0)
        total += ec + wc + pc

        schedule[tid] = dict(
            t_s=t_s, t_d=t_d, P=P_i, port=port_idx,
            energy_cost=ec, wait_cost=wc, tard_cost=pc
        )
        intervals.append((t_s, t_d, P_i))

    return total, schedule


# ═══════════════════════════════════════════════════════
# PORT RELEASE HELPER
# ═══════════════════════════════════════════════════════
def _port_release(omega, trucks, p_assign=None):
    if p_assign is None:
        p_assign = {tr['id']: P_MAX for tr in trucks}
    release = []
    for seq in omega:
        t = 0.0
        for tid in seq:
            tr  = trucks[tid]
            P_i = p_assign.get(tid, P_MAX)
            t   = max(tr['a'], t) + tr['delta_E'] / P_i
        release.append(t)
    return release


# ═══════════════════════════════════════════════════════
# BASE POLICIES
# Unassigned trucks use P_MAX as default in lookahead
# ═══════════════════════════════════════════════════════
def base_policy_fcfs(partial_omega, unassigned, trucks, p_assign=None):
    omega = deepcopy(partial_omega)
    for tid in sorted(unassigned, key=lambda i: trucks[i]['a']):
        omega[int(np.argmin(_port_release(omega, trucks, p_assign)))].append(tid)
    return omega


def base_policy_edf(partial_omega, unassigned, trucks, p_assign=None):
    omega = deepcopy(partial_omega)
    for tid in sorted(unassigned, key=lambda i: trucks[i]['d']):
        omega[int(np.argmin(_port_release(omega, trucks, p_assign)))].append(tid)
    return omega


def base_policy_scdf(partial_omega, unassigned, trucks, p_assign=None):
    omega = deepcopy(partial_omega)
    for tid in sorted(unassigned, key=lambda i: trucks[i]['delta_E']):
        omega[int(np.argmin(_port_release(omega, trucks, p_assign)))].append(tid)
    return omega


BASE_POLICIES = {
    'FCFS': base_policy_fcfs,
    'EDF':  base_policy_edf,
    'SCDF': base_policy_scdf,
}


# ═══════════════════════════════════════════════════════
# ROLLOUT SCHEDULER
# Decision per step: (truck, port, power_level)
# ═══════════════════════════════════════════════════════
def rollout_scheduler(trucks, C, base_policy_fn, desc=""):
    N_trucks = len(trucks)
    x_k      = [[] for _ in range(C)]
    p_curr   = {}        # power assignments committed so far
    assigned = set()

    iter_obj = range(N_trucks)
    if HAS_TQDM:
        iter_obj = tqdm(iter_obj,
                        desc=desc if desc else "Rollout",
                        leave=False, dynamic_ncols=True)

    for _ in iter_obj:
        unassigned = [i for i in range(N_trucks) if i not in assigned]
        J_star, u_star = np.inf, None

        for tid in unassigned:
            for port in range(C):
                for P_i in P_OPTIONS:           # try both power levels
                    x_next      = deepcopy(x_k)
                    x_next[port].append(tid)

                    p_next      = deepcopy(p_curr)
                    p_next[tid] = P_i

                    rest   = [i for i in unassigned if i != tid]
                    # unassigned remaining default to P_MAX in lookahead
                    p_full = {**p_next, **{r: P_MAX for r in rest}}

                    omega_full = base_policy_fn(x_next, rest, trucks, p_full)
                    J_hat, _   = solve_inner_layer(omega_full, trucks, p_full)

                    if J_hat < J_star:
                        J_star = J_hat
                        u_star = (tid, port, P_i)

        tid_s, port_s, P_s = u_star
        x_k[port_s].append(tid_s)
        p_curr[tid_s] = P_s
        assigned.add(tid_s)

    cost, sched = solve_inner_layer(x_k, trucks, p_curr)
    return x_k, cost, sched, p_curr


# ═══════════════════════════════════════════════════════
# HEURISTICS
# Fix port assignment by sorting, then greedy power selection:
#   process trucks in tentative start-time order;
#   for each truck try both P_OPTIONS, keep the one with lower
#   incremental cost given already-committed intervals.
#   O(N) solves — no 2^N enumeration needed.
# ═══════════════════════════════════════════════════════
def _sort_assign_with_greedy_power(trucks, C, key_fn):
    """
    1. Assign ports by round-robin on sorted order.
    2. Sort each port queue by arrival time (stabilises scheduling).
    3. Greedy power selection: schedule trucks one by one in start-time
       order; at each step choose the power level that minimises the
       total cost accumulated so far (committed intervals are frozen).
    """
    N = len(trucks)
    order = sorted(range(N), key=key_fn)
    omega = [[] for _ in range(C)]
    for rank, tid in enumerate(order):
        omega[rank % C].append(tid)
    for c in range(C):
        omega[c].sort(key=lambda tid: trucks[tid]['a'])

    # --- Greedy power selection ---
    # Determine scheduling order (tentative, using P_MAX)
    p_default = {tr['id']: P_MAX for tr in trucks}
    tentative = {}
    port_rel  = [0.0] * C
    for port_idx, port_seq in enumerate(omega):
        for tid in port_seq:
            tr  = trucks[tid]
            t_s = max(tr['a'], port_rel[port_idx])
            t_d = t_s + tr['delta_E'] / P_MAX
            port_rel[port_idx] = t_d
            tentative[tid] = dict(port=port_idx, t_s=t_s)
    sched_order = sorted(tentative, key=lambda tid: tentative[tid]['t_s'])

    p_assign  = {}
    intervals = []   # committed (t_s, t_d, P) — same feasibility logic
    port_rel2 = [0.0] * C

    def is_feasible(t_s, t_d, P_i):
        pts = sorted(set(
            [t_s, t_d] +
            [ts for ts, td, _ in intervals if t_s < ts < t_d] +
            [td for ts, td, _ in intervals if t_s < td < t_d]
        ))
        for i in range(len(pts) - 1):
            mid   = 0.5 * (pts[i] + pts[i + 1])
            power = sum(P for ts, td, P in intervals if ts <= mid < td)
            if power + P_i > P_STATION + 1e-6:
                return False
        return True

    def earliest_start(t_ready, dur, P_i):
        t = t_ready
        while True:
            if is_feasible(t, t + dur, P_i):
                return t
            future = [td for ts, td, _ in intervals if td > t]
            if not future:
                return t
            t = min(future)

    for tid in sched_order:
        tr       = trucks[tid]
        port_idx = tentative[tid]['port']
        t_ready  = max(tr['a'], port_rel2[port_idx])

        best_cost_local = np.inf
        best_P          = P_OPTIONS[0]
        best_ts         = None
        best_td         = None

        for P_i in P_OPTIONS:
            dur  = tr['delta_E'] / P_i
            t_s  = earliest_start(t_ready, dur, P_i)
            t_d  = t_s + dur
            ec   = energy_cost_integral(t_s, t_d, P_i)
            wc   = tr['epsilon'] * (t_s - tr['a'])
            pc   = tr['gamma']   * max(t_d - tr['d'], 0.0)
            cost = ec + wc + pc
            if cost < best_cost_local:
                best_cost_local = cost
                best_P          = P_i
                best_ts         = t_s
                best_td         = t_d

        p_assign[tid]        = best_P
        port_rel2[port_idx]  = best_td
        intervals.append((best_ts, best_td, best_P))

    # Final full solve with chosen p_assign
    cost, sched = solve_inner_layer(omega, trucks, p_assign)
    return omega, cost, sched, p_assign


def heuristic_fcfs(trucks, C):
    return _sort_assign_with_greedy_power(trucks, C, lambda i: trucks[i]['a'])


def heuristic_edf(trucks, C):
    return _sort_assign_with_greedy_power(trucks, C, lambda i: trucks[i]['d'])


def heuristic_scdf(trucks, C):
    return _sort_assign_with_greedy_power(trucks, C, lambda i: trucks[i]['delta_E'])


def cost_breakdown(sched):
    ec = sum(v['energy_cost'] for v in sched.values())
    wc = sum(v['wait_cost']   for v in sched.values())
    pc = sum(v['tard_cost']   for v in sched.values())
    return ec, wc, pc


# ═══════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════
METHODS = ['ro_fcfs', 'ro_edf', 'ro_scdf', 'fcfs', 'edf', 'scdf']

results = {n: {m: {'cost': [], 'time': [], 'ec': [], 'wc': [], 'pc': []}
               for m in METHODS}
           for n in N_LIST}

print("=" * 70)
print("  Electric Truck Charging — Large-Scale (No Exact Optimal)")
print(f"  C={C} | P∈{P_OPTIONS} kW | N={N_LIST} | K={K}")
print(f"  P_STATION={P_STATION} kW (hard) | Arrivals: "
      f"[{ARRIVAL_LO:.0f}:00, {ARRIVAL_HI:.0f}:00)")
print("=" * 70)

for n_s in N_LIST:
    print(f"\n══ N={n_s} ══════════════════════════════════")
    for k in range(K):
        seed = 1000 * n_s + k
        trs  = generate_trucks(n_s, seed=seed)
        print(f"  k={k} (seed={seed})")

        for ro_key, bp_fn, ro_name in [
            ('ro_fcfs', base_policy_fcfs, 'RO(FCFS)'),
            ('ro_edf',  base_policy_edf,  'RO(EDF) '),
            ('ro_scdf', base_policy_scdf, 'RO(SCDF)'),
        ]:
            t0 = time.perf_counter()
            _, J_ro, sched_ro, _ = rollout_scheduler(
                trs, C, bp_fn, desc=f"{ro_name} N={n_s}")
            t_ro = time.perf_counter() - t0
            ec, wc, pc = cost_breakdown(sched_ro)
            results[n_s][ro_key]['cost'].append(J_ro)
            results[n_s][ro_key]['time'].append(t_ro)
            results[n_s][ro_key]['ec'].append(ec)
            results[n_s][ro_key]['wc'].append(wc)
            results[n_s][ro_key]['pc'].append(pc)
            print(f"  ✓ {ro_name}: {t_ro:.2f}s  cost={J_ro:.2f} €")

        for h_key, h_fn, h_name in [
            ('fcfs', heuristic_fcfs, 'FCFS'),
            ('edf',  heuristic_edf,  'EDF '),
            ('scdf', heuristic_scdf, 'SCDF'),
        ]:
            t0 = time.perf_counter()
            _, J_h, sched_h, _ = h_fn(trs, C)
            t_h = time.perf_counter() - t0
            ec, wc, pc = cost_breakdown(sched_h)
            results[n_s][h_key]['cost'].append(J_h)
            results[n_s][h_key]['time'].append(t_h)
            results[n_s][h_key]['ec'].append(ec)
            results[n_s][h_key]['wc'].append(wc)
            results[n_s][h_key]['pc'].append(pc)
            print(f"  ✓ Heur {h_name}: {t_h*1000:.2f}ms  cost={J_h:.2f} €")

# Gap relative to best rollout per (N, k)
results_gap = {n: {m: [] for m in METHODS} for n in N_LIST}
for n_s in N_LIST:
    for k in range(K):
        ref = min(results[n_s][m]['cost'][k]
                  for m in ['ro_fcfs', 'ro_edf', 'ro_scdf'])
        for m in METHODS:
            gap = 100.0 * (results[n_s][m]['cost'][k] - ref) / (ref + 1e-9)
            results_gap[n_s][m].append(gap)

# ═══════════════════════════════════════════════════════
# VISUALISATION INSTANCE  (N=25, k=0)
# ═══════════════════════════════════════════════════════
N_VIS      = 100
trucks_vis = generate_trucks(N_VIS, seed=1000 * N_VIS)
print(f"\nComputing vis instance (N={N_VIS}, k=0) ...")

ro_vis = {}
for name, bp_fn in BASE_POLICIES.items():
    _, cost, sched, p_ro = rollout_scheduler(
        trucks_vis, C, bp_fn, desc=f"Vis RO({name})")
    ro_vis[name] = dict(cost=cost, sched=sched, p_assign=p_ro)
    print(f"  RO({name}): {cost:.2f} €")

_, cost_fcfs_v, sched_fcfs_v, p_fcfs_v = heuristic_fcfs(trucks_vis, C)
_, cost_edf_v,  sched_edf_v,  p_edf_v  = heuristic_edf(trucks_vis,  C)
_, cost_scdf_v, sched_scdf_v, p_scdf_v = heuristic_scdf(trucks_vis, C)

# ═══════════════════════════════════════════════════════
# SAVE ALL DATA
# ═══════════════════════════════════════════════════════
data = dict(
    C=C, P_OPTIONS=P_OPTIONS, P_STATION=P_STATION,
    E_MAX=E_MAX, P_MAX=P_MAX,
    EPSILON=EPSILON, GAMMA=GAMMA,
    N_LIST=N_LIST, K=K,
    ARRIVAL_LO=ARRIVAL_LO, ARRIVAL_HI=ARRIVAL_HI,
    DEADLINE_SLACK=DEADLINE_SLACK,
    TOU_SCHEDULE=TOU_SCHEDULE,
    METHODS=METHODS,
    results=results,
    results_gap=results_gap,
    N_VIS=N_VIS,
    trucks_vis=trucks_vis,
    ro_vis=ro_vis,
    cost_fcfs_v=cost_fcfs_v, sched_fcfs_v=sched_fcfs_v, p_fcfs_v=p_fcfs_v,
    cost_edf_v=cost_edf_v,   sched_edf_v=sched_edf_v,   p_edf_v=p_edf_v,
    cost_scdf_v=cost_scdf_v, sched_scdf_v=sched_scdf_v, p_scdf_v=p_scdf_v,
)

pkl_path = os.path.join(OUTPUT_DIR, "experiment_data_large.pkl")
with open(pkl_path, 'wb') as f:
    pickle.dump(data, f)
print(f"\nData saved → {pkl_path}")

# ═══════════════════════════════════════════════════════
# SAVE CSV
# ═══════════════════════════════════════════════════════
_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

csv_path = os.path.join(OUTPUT_DIR, f"large_results_{_ts}.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ['N', 'k', 'seed'] +
        [f'cost_{m}'    for m in METHODS] +
        [f'time_{m}_s'  for m in METHODS] +
        [f'ec_{m}'      for m in METHODS] +
        [f'wc_{m}'      for m in METHODS] +
        [f'pc_{m}'      for m in METHODS] +
        [f'gap_{m}_pct' for m in METHODS]
    )
    for n_s in N_LIST:
        for k in range(K):
            row = [n_s, k, 1000*n_s+k]
            row += [results[n_s][m]['cost'][k] for m in METHODS]
            row += [results[n_s][m]['time'][k] for m in METHODS]
            row += [results[n_s][m]['ec'][k]   for m in METHODS]
            row += [results[n_s][m]['wc'][k]   for m in METHODS]
            row += [results[n_s][m]['pc'][k]   for m in METHODS]
            row += [results_gap[n_s][m][k]     for m in METHODS]
            writer.writerow(row)
print(f"Full CSV saved → {csv_path}")

csv_time = os.path.join(OUTPUT_DIR, f"large_comp_time_{_ts}.csv")
with open(csv_time, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['N'] + [f't_{m}_mean_s' for m in METHODS])
    for n_s in N_LIST:
        row = [n_s] + [np.mean(results[n_s][m]['time']) for m in METHODS]
        writer.writerow(row)
print(f"Time CSV saved  → {csv_time}")

csv_table = os.path.join(OUTPUT_DIR, f"large_summary_table_{_ts}.csv")
with open(csv_table, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric'] + [f'N={n}' for n in N_LIST])

    best_ro_costs = []
    best_ro_names = []
    best_ro_times = []
    base_costs    = []

    for n_s in N_LIST:
        best_m = min(['ro_fcfs', 'ro_edf', 'ro_scdf'],
                     key=lambda m: results[n_s][m]['cost'][0])
        best_ro_costs.append(results[n_s][best_m]['cost'][0])
        best_ro_times.append(results[n_s][best_m]['time'][0])
        best_ro_names.append(best_m)
        base_key = best_m.replace('ro_', '')
        base_costs.append(results[n_s][base_key]['cost'][0])

    writer.writerow(['Best Rollout Cost (EUR)'] +
                    [f'{c:.2f}' for c in best_ro_costs])
    reductions = [100.0 * (base - ro) / base
                  for base, ro in zip(base_costs, best_ro_costs)]
    writer.writerow(['Cost Reduction vs Base Policy (%)'] +
                    [f'{r:.2f}' for r in reductions])
    writer.writerow(['Rollout Time (s)'] +
                    [f'{t:.2f}' for t in best_ro_times])
    writer.writerow(['Best Rollout Method'] + best_ro_names)

print(f"Summary table saved → {csv_table}")
print("\nAll done!")