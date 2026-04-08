"""
Electric Truck Charging Scheduling — Data Generation + Save
  C=3, N=5..6, K=3
  6 methods: Rollout(FCFS/EDF/SCDF) + Heuristic(FCFS/EDF/SCDF)
  vs Exact Optimal (brute-force enumeration)
  Hard power constraint: total station power <= P_STATION at all times
  Per-truck power: chosen from P_OPTIONS={300,350} kW, fixed for entire session
  Saves all data to outputs/experiment_data.pkl
"""

import os
import datetime
import numpy as np
import pickle
import csv
from copy import deepcopy
from itertools import permutations, combinations_with_replacement, product as iproduct
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
C          = 3
P_OPTIONS  = [300.0, 350.0]   # kW — two discrete charging power levels per truck
P_STATION  = 1000.0           # kW  hard constraint (3×350=1050 > 1000, so binding)
E_MAX      = 468.0            # kWh
P_MAX      = 350.0            # kW  (used for deadline & default lookahead power)
SOC_LO     = 0.20
SOC_HI     = 0.80
EPSILON    = 120.0            # EUR/h  waiting cost coefficient
GAMMA      = 600.0            # EUR/h  tardiness penalty coefficient
DEADLINE_SLACK = 1.5
ARRIVAL_LO     = 0.0
ARRIVAL_HI     = 24.0
ARRIVAL_WINDOW = 30 / 60.0   # h  (5 min)

N_LIST = [4,5, 6,7]
K  = 3

TOU_SCHEDULE = [
    (0,  6,  0.101),
    (6,  9,  0.174),
    (9,  12, 0.128),
    (12, 17, 0.110),
    (17, 21, 0.202),
    (21, 24, 0.101),
]
TOU_EXT = TOU_SCHEDULE + [(s+24, e+24, p) for s, e, p in TOU_SCHEDULE]


def energy_cost_integral(t_s, t_d, P_i):
    cost = 0.0
    for s, e, p in TOU_EXT:
        ov_s = max(t_s, float(s))
        ov_e = min(t_d, float(e))
        if ov_e > ov_s:
            cost += p * P_i * (ov_e - ov_s)
    return cost


def generate_trucks(N, seed=42):
    rng    = np.random.default_rng(seed)
    a_base = rng.uniform(ARRIVAL_LO, ARRIVAL_HI - ARRIVAL_WINDOW)
    trucks = []
    for i in range(N):
        a_i     = rng.uniform(a_base, a_base + ARRIVAL_WINDOW)
        E0_i    = rng.uniform(SOC_LO, SOC_HI) * E_MAX
        delta_E = E_MAX - E0_i
        d_i     = a_i + DEADLINE_SLACK * delta_E / P_MAX   # deadline at max power
        trucks.append(dict(
            id=i, a=a_i, E0=E0_i, E_max=E_MAX,
            delta_E=delta_E, d=d_i, P_max=P_MAX,
            epsilon=EPSILON, gamma=GAMMA,
        ))
    return trucks


# ═══════════════════════════════════════════════════════
# INNER LAYER SOLVER
# ═══════════════════════════════════════════════════════
def solve_inner_layer(omega, trucks, p_assign=None):
    """
    Schedule trucks given:
      omega    – port sequences (list of C lists of truck-ids)
      trucks   – truck data list
      p_assign – dict {tid: power_kW}  chosen from P_OPTIONS, fixed per session.
                 Defaults to P_MAX for all trucks when None.

    Hard constraint: sum of all concurrently charging powers <= P_STATION.
    """
    if p_assign is None:
        p_assign = {tr['id']: P_MAX for tr in trucks}

    tentative = {}
    port_rel  = [0.0] * len(omega)

    # -------- Pass 1: tentative start times per port --------
    for port_idx, port_seq in enumerate(omega):
        for tid in port_seq:
            tr  = trucks[tid]
            P_i = p_assign.get(tid, P_MAX)
            t_s = max(tr['a'], port_rel[port_idx])
            t_d = t_s + tr['delta_E'] / P_i
            port_rel[port_idx] = t_d
            tentative[tid] = dict(port=port_idx, t_s=t_s)

    # -------- Sort globally by tentative start --------
    order = sorted(tentative, key=lambda tid: tentative[tid]['t_s'])

    schedule  = {}
    port_rel2 = [0.0] * len(omega)
    intervals = []          # committed intervals: (t_s, t_d, P)

    # -------- Interval feasibility check --------
    def is_feasible_interval(t_s, t_d, P_i):
        pts = sorted(set(
            [t_s, t_d]
            + [ts for ts, td, _ in intervals if t_s < ts < t_d]
            + [td for ts, td, _ in intervals if t_s < td < t_d]
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

    total = 0.0

    # -------- Pass 2: global scheduling with feasibility --------
    for tid in order:
        tr       = trucks[tid]
        P_i      = p_assign.get(tid, P_MAX)
        port_idx = tentative[tid]['port']

        dur     = tr['delta_E'] / P_i
        t_ready = max(tr['a'], port_rel2[port_idx])
        t_s     = find_earliest_start(t_ready, dur, P_i)
        t_d     = t_s + dur

        port_rel2[port_idx] = t_d

        ec = energy_cost_integral(t_s, t_d, P_i)
        wc = tr['epsilon'] * (t_s - tr['a'])
        pc = tr['gamma'] * max(t_d - tr['d'], 0.0)

        total += ec + wc + pc
        schedule[tid] = dict(
            t_s=t_s, t_d=t_d, P=P_i, port=port_idx,
            energy_cost=ec, wait_cost=wc, tard_cost=pc
        )
        intervals.append((t_s, t_d, P_i))

    return total, schedule


# ═══════════════════════════════════════════════════════
# EXACT OPTIMAL
# Enumerate: all power combos × all permutations × all port splits
# ═══════════════════════════════════════════════════════
def exact_optimal(trucks, C, desc=""):
    N = len(trucks)
    best_cost  = np.inf
    best_omega = None
    best_p     = None
    t0 = time.perf_counter()

    all_power_combos = list(iproduct(P_OPTIONS, repeat=N))   # 2^N combos
    all_perms        = list(permutations(range(N)))
    all_cuts         = list(combinations_with_replacement(range(N + 1), C - 1))

    total_iters = len(all_power_combos) * len(all_perms)
    label = desc if desc else f"Exact N={N},C={C}"

    outer = ((pc, perm) for pc in all_power_combos for perm in all_perms)
    if HAS_TQDM:
        outer = tqdm(outer, total=total_iters, desc=label,
                     unit="combo", dynamic_ncols=True)

    for p_combo, perm in outer:
        p_assign = {i: p_combo[i] for i in range(N)}
        for cuts in all_cuts:
            omega, prev = [], 0
            for cut in cuts:
                omega.append(list(perm[prev:cut]))
                prev = cut
            omega.append(list(perm[prev:]))
            cost, _ = solve_inner_layer(omega, trucks, p_assign)
            if cost < best_cost:
                best_cost  = cost
                best_omega = deepcopy(omega)
                best_p     = deepcopy(p_assign)

    elapsed = time.perf_counter() - t0
    _, best_sched = solve_inner_layer(best_omega, trucks, best_p)
    return best_omega, best_cost, best_sched, elapsed, best_p


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
# BASE POLICIES  (unassigned trucks default to P_MAX in lookahead)
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
# Decision per step: (truck, port, power_level)  ← NEW: power_level added
# ═══════════════════════════════════════════════════════
def rollout_scheduler(trucks, C, base_policy_fn):
    N        = len(trucks)
    x_k      = [[] for _ in range(C)]
    p_curr   = {}          # power assignments committed so far
    assigned = set()

    while len(assigned) < N:
        unassigned = [i for i in range(N) if i not in assigned]
        J_star, u_star = np.inf, None

        for tid in unassigned:
            for port in range(C):
                for P_i in P_OPTIONS:          # ← try both power levels
                    x_next      = deepcopy(x_k)
                    x_next[port].append(tid)

                    p_next      = deepcopy(p_curr)
                    p_next[tid] = P_i

                    rest   = [i for i in unassigned if i != tid]
                    p_full = {**p_next, **{r: P_MAX for r in rest}}   # remaining → P_MAX

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
# Fixed ordering; enumerate all 2^N power combos → pick best
# ═══════════════════════════════════════════════════════
def _sort_assign(trucks, C, key_fn):
    order = sorted(range(len(trucks)), key=key_fn)
    omega = [[] for _ in range(C)]
    for rank, tid in enumerate(order):
        omega[rank % C].append(tid)

    N          = len(trucks)
    best_cost  = np.inf
    best_sched = None
    best_p     = None

    for p_combo in iproduct(P_OPTIONS, repeat=N):      # 2^N = 32 or 64, very fast
        p_assign = {i: p_combo[i] for i in range(N)}
        cost, sched = solve_inner_layer(omega, trucks, p_assign)
        if cost < best_cost:
            best_cost  = cost
            best_sched = sched
            best_p     = deepcopy(p_assign)

    return omega, best_cost, best_sched, best_p


def heuristic_fcfs(trucks, C):
    return _sort_assign(trucks, C, lambda i: trucks[i]['a'])

def heuristic_edf(trucks, C):
    return _sort_assign(trucks, C, lambda i: trucks[i]['d'])

def heuristic_scdf(trucks, C):
    return _sort_assign(trucks, C, lambda i: trucks[i]['delta_E'])


def cost_breakdown(sched):
    ec = sum(v['energy_cost'] for v in sched.values())
    wc = sum(v['wait_cost']   for v in sched.values())
    pc = sum(v['tard_cost']   for v in sched.values())
    return ec, wc, pc


# ═══════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════
METHODS = ['ro_fcfs', 'ro_edf', 'ro_scdf', 'fcfs', 'edf', 'scdf']
results     = {n: {m: {'cost': [], 'time': [], 'gap': []} for m in METHODS}
               for n in N_LIST}
results_opt = {n: {'cost': [], 'time': []} for n in N_LIST}

print("=" * 70)
print("  Electric Truck Charging — 6 Methods vs Exact Optimal")
print(f"  C={C} ports | P∈{P_OPTIONS} kW | N={N_LIST} | K={K} | P_STATION={P_STATION} kW")
print("=" * 70)

for n_s in N_LIST:
    print(f"\n── N={n_s} ─────────────────────")
    for k in range(K):
        seed = 1000 * n_s + k
        trs  = generate_trucks(n_s, seed=seed)
        print(f"  k={k} (seed={seed})")

        _, J_opt, _, t_opt, p_opt = exact_optimal(
            trs, C, desc=f"  Exact N={n_s},k={k}")
        results_opt[n_s]['cost'].append(J_opt)
        results_opt[n_s]['time'].append(t_opt)
        print(f"  ✓ Exact:  {t_opt:.2f}s  J*={J_opt:.4f} €")

        for ro_key, bp_fn, ro_name in [
            ('ro_fcfs', base_policy_fcfs, 'RO(FCFS)'),
            ('ro_edf',  base_policy_edf,  'RO(EDF) '),
            ('ro_scdf', base_policy_scdf, 'RO(SCDF)'),
        ]:
            t0 = time.perf_counter()
            _, J_ro, _, _ = rollout_scheduler(trs, C, bp_fn)
            t_ro = time.perf_counter() - t0
            gap  = 100.0 * (J_ro - J_opt) / J_opt
            results[n_s][ro_key]['cost'].append(J_ro)
            results[n_s][ro_key]['time'].append(t_ro)
            results[n_s][ro_key]['gap'].append(gap)
            print(f"  ✓ {ro_name}: {t_ro:.4f}s  gap={gap:.3f}%")

        for h_key, h_fn, h_name in [
            ('fcfs', heuristic_fcfs, 'FCFS'),
            ('edf',  heuristic_edf,  'EDF '),
            ('scdf', heuristic_scdf, 'SCDF'),
        ]:
            t0 = time.perf_counter()
            _, J_h, _, _ = h_fn(trs, C)
            t_h = time.perf_counter() - t0
            gap = 100.0 * (J_h - J_opt) / J_opt
            results[n_s][h_key]['cost'].append(J_h)
            results[n_s][h_key]['time'].append(t_h)
            results[n_s][h_key]['gap'].append(gap)
            print(f"  ✓ Heur {h_name}: {t_h*1000:.3f}ms  gap={gap:.3f}%")


# ═══════════════════════════════════════════════════════
# VISUALISATION INSTANCE  (N=6, k=0)
# ═══════════════════════════════════════════════════════
N_VIS      = 4
trucks_vis = generate_trucks(N_VIS, seed=1000 * N_VIS)
print(f"\nComputing exact optimal for visualisation instance (N={N_VIS}) ...")
_, J_opt_v, sched_opt_v, _, p_opt_v = exact_optimal(
    trucks_vis, C, desc=f"Exact N={N_VIS} (vis)")

ro_vis = {}
for name, bp_fn in BASE_POLICIES.items():
    _, cost, sched, p_ro = rollout_scheduler(trucks_vis, C, bp_fn)
    ro_vis[name] = dict(cost=cost, sched=sched, p_assign=p_ro)

_, cost_fcfs_v,  sched_fcfs_v,  p_fcfs_v  = heuristic_fcfs(trucks_vis, C)
_, cost_edf_v,   sched_edf_v,   p_edf_v   = heuristic_edf(trucks_vis,  C)
_, cost_scdf_v,  sched_scdf_v,  p_scdf_v  = heuristic_scdf(trucks_vis, C)


# ═══════════════════════════════════════════════════════
# SAVE ALL DATA
# ═══════════════════════════════════════════════════════
data = dict(
    C=C, P_OPTIONS=P_OPTIONS, P_STATION=P_STATION,
    E_MAX=E_MAX, P_MAX=P_MAX,
    EPSILON=EPSILON, GAMMA=GAMMA,
    N_LIST=N_LIST, K=K,
    TOU_SCHEDULE=TOU_SCHEDULE,
    results=results,
    results_opt=results_opt,
    METHODS=METHODS,
    N_VIS=N_VIS,
    trucks_vis=trucks_vis,
    J_opt_v=J_opt_v,
    sched_opt_v=sched_opt_v,
    p_opt_v=p_opt_v,
    ro_vis=ro_vis,
    cost_fcfs_v=cost_fcfs_v, sched_fcfs_v=sched_fcfs_v, p_fcfs_v=p_fcfs_v,
    cost_edf_v=cost_edf_v,   sched_edf_v=sched_edf_v,   p_edf_v=p_edf_v,
    cost_scdf_v=cost_scdf_v, sched_scdf_v=sched_scdf_v, p_scdf_v=p_scdf_v,
)

pkl_path = os.path.join(OUTPUT_DIR, "experiment_data.pkl")
with open(pkl_path, 'wb') as f:
    pickle.dump(data, f)
print(f"\nData saved → {pkl_path}")


# ═══════════════════════════════════════════════════════
# SAVE CSV
# ═══════════════════════════════════════════════════════
_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

csv_path = os.path.join(OUTPUT_DIR, f"charging_results_{_ts}.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'N', 'k', 'seed',
        'J_opt', 't_opt_s',
        'J_ro_fcfs', 't_ro_fcfs_s', 'gap_ro_fcfs_pct',
        'J_ro_edf',  't_ro_edf_s',  'gap_ro_edf_pct',
        'J_ro_scdf', 't_ro_scdf_s', 'gap_ro_scdf_pct',
        'J_fcfs',    't_fcfs_s',    'gap_fcfs_pct',
        'J_edf',     't_edf_s',     'gap_edf_pct',
        'J_scdf',    't_scdf_s',    'gap_scdf_pct',
    ])
    for n_s in N_LIST:
        for k in range(K):
            row = [n_s, k, 1000 * n_s + k,
                   results_opt[n_s]['cost'][k],
                   results_opt[n_s]['time'][k]]
            for mkey in METHODS:
                row += [
                    results[n_s][mkey]['cost'][k],
                    results[n_s][mkey]['time'][k],
                    results[n_s][mkey]['gap'][k],
                ]
            writer.writerow(row)

csv_summary = os.path.join(OUTPUT_DIR, f"charging_results_summary_{_ts}.csv")
with open(csv_summary, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'N',
        't_exact_mean_s',
        't_ro_fcfs_mean_s', 't_ro_edf_mean_s', 't_ro_scdf_mean_s',
        't_fcfs_mean_s',    't_edf_mean_s',     't_scdf_mean_s',
        'gap_ro_fcfs_mean_pct', 'gap_ro_edf_mean_pct', 'gap_ro_scdf_mean_pct',
        'gap_fcfs_mean_pct',    'gap_edf_mean_pct',     'gap_scdf_mean_pct',
        'J_opt_mean',
    ])
    for n_s in N_LIST:
        row = [n_s, np.mean(results_opt[n_s]['time'])]
        for mkey in METHODS:
            row.append(np.mean(results[n_s][mkey]['time']))
        for mkey in METHODS:
            row.append(np.mean(results[n_s][mkey]['gap']))
        row.append(np.mean(results_opt[n_s]['cost']))
        writer.writerow(row)

print(f"CSV saved         → {csv_path}")
print(f"Summary CSV saved → {csv_summary}")
print("\nAll done! Run plot_figures.py to generate figures.")