#!/usr/bin/env python3
"""Reproduce the N=100 experiment in GPG-Bregman_v27.tex.

Dependencies: Python 3.10+, NumPy, Matplotlib.
Install: python -m pip install numpy matplotlib
Run: python gpg_numerical_experiment.py --output-dir .

The model and algorithms are deterministic. No random seed or Monte Carlo
sampling is involved. Outputs are three PNG figures, a CSV of the plotted
distances, and a JSON file containing the parameters, complete partitions,
block masses, objective values, and first threshold-crossing times.

State space: {0,...,99}, a cycle. H(x)=-cos(8*pi*x/100)-.15*cos(2*pi*x/100).
Target: pi(x) proportional to exp(-2*H(x)). The baseline is lazy nearest-
neighbor Metropolis: P(x,x+-1)=.25*min(1,pi(x+-1)/pi(x)); the remaining
probability is assigned to x. Indices are taken modulo 100.

G1 isolates the three least probable states and groups the other 97 states.
G2 sorts masses in decreasing order, seeds four singleton blocks, then assigns
each remaining state to the lightest block. Exact ties use the lowest state
index, then the lowest block index. The energy is averaged with its reflection
to enforce its mathematical symmetry exactly despite floating-point cosines.

For Q in {P,G1 P G1,G2 P G2}, the plotted quantities at integer t are
  max_x (1/2)*sum_y |Q^t(x,y)-pi(y)|,
  sum_xy pi(x)/pi(y)*(Q^t(x,y)-pi(y))**2.
Distances use the reversible spectral decomposition of each finite matrix,
checked against direct powers and the four-state projection chains. Integer
threshold crossings are found by doubling and binary search and checked at
both t and t-1. Floating-point zero eigenvalues of GPG are removed; t=0 is
handled separately, where Q^0=I and the squared-Frobenius distance is 99.

Plots use logarithmic axes and integer evaluation times. Curves stop at the
display floor 1e-10; a downward triangle marks the first value below this
floor. This avoids depicting a spurious numerical plateau. All original
computed values, including those below the display floor, are saved to CSV.
The comparison is by transitions: one projected transition includes two
additional exact conditional draws. It is not a runtime benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

N = 100
K = 4
BETA = 2.0
EPSILON = 0.01
DISPLAY_FLOOR = 1e-10
CHAIN_NAMES = ("P", "G1PG1", "G2PG2")
CHAIN_LABELS = (r"$P$", r"$G_1PG_1$", r"$G_2PG_2$")
CHAIN_COLORS = ("#525E70", "#2166AC", "#C44E18")
BLOCK_COLORS = ("#D55E00", "#009E73", "#CC79A7", "#0072B2")


def build_model():
    states = np.arange(N)
    theta = 2*np.pi*states/N
    energy = -np.cos(4*theta)-0.15*np.cos(theta)
    # The true energy is reflection-symmetric. Make its ties reproducible.
    energy = (energy+energy[(-states) % N])/2
    weights = np.exp(-BETA*(energy-energy.min()))
    pi = weights/weights.sum()
    P = np.zeros((N, N))
    for x in states:
        for y in ((x-1) % N, (x+1) % N):
            P[x, y] = 0.25*min(1.0, pi[y]/pi[x])
        P[x, x] = 1-P[x].sum()
    return states, energy, pi, P


def exact_partition(pi):
    order = np.argsort(pi, kind="stable")
    return [order[i:i+1] for i in range(K-1)]+[order[K-1:]]


def greedy_partition(pi):
    order = np.argsort(-pi, kind="stable")
    blocks = [[int(x)] for x in order[:K]]
    masses = pi[order[:K]].copy()
    for x in order[K:]:
        block = int(np.argmin(masses))
        blocks[block].append(int(x))
        masses[block] += pi[x]
    return [np.array(block, dtype=int) for block in blocks]


def gibbs(pi, blocks):
    G = np.zeros((len(pi), len(pi)))
    labels = np.empty(len(pi), dtype=int)
    masses = np.array([pi[b].sum() for b in blocks])
    for i, block in enumerate(blocks):
        G[np.ix_(block, block)] = pi[block][None, :]/masses[i]
        labels[block] = i
    return G, labels, masses


def direct_metrics(power, pi):
    centered = power-pi[None, :]
    tv = float(np.max(np.abs(centered).sum(axis=1)/2))
    f2 = float(np.sum((pi[:, None]/pi[None, :])*centered**2))
    return tv, f2


class Chain:
    """A reversible kernel and its centered spectral representation."""

    def __init__(self, matrix, pi):
        self.matrix, self.pi = matrix, pi
        root_pi = np.sqrt(pi)
        weighted = root_pi[:, None]*matrix/root_pi[None, :]
        np.testing.assert_allclose(weighted, weighted.T, atol=2e-14)
        values, vectors = np.linalg.eigh((weighted+weighted.T)/2)
        stationary = int(np.argmax(values))
        np.testing.assert_allclose(values[stationary], 1, atol=2e-14)
        # Laziness of P makes P and both GPG kernels positive semidefinite.
        assert values.min() >= -2e-14
        keep = np.arange(len(values)) != stationary
        keep &= np.abs(values) > 1e-13
        self.values = values[keep]
        self.vectors = vectors[:, keep]
        self.density_scale = root_pi[None, :]/root_pi[:, None]
        self.cache = {0: (float(1-pi.min()), float(len(pi)-1))}

    def metrics(self, t):
        t = int(t)
        if t < 0:
            raise ValueError("Time must be a nonnegative integer.")
        if t not in self.cache:
            powers = self.values**t
            centered = ((self.vectors*powers) @ self.vectors.T)*self.density_scale
            self.cache[t] = (
                float(np.abs(centered).sum(axis=1).max()/2),
                float(np.sum(powers**2)),
            )
        return self.cache[t]

    def first_below(self, metric, epsilon=EPSILON):
        if self.metrics(0)[metric] <= epsilon:
            return 0
        lo, hi = 0, 1
        while self.metrics(hi)[metric] > epsilon:
            lo, hi = hi, 2*hi
            if hi > 10**8:
                raise RuntimeError("Threshold search exceeded its limit.")
        while hi-lo > 1:
            mid = (lo+hi)//2
            if self.metrics(mid)[metric] <= epsilon:
                hi = mid
            else:
                lo = mid
        assert self.metrics(hi)[metric] <= epsilon
        assert self.metrics(hi-1)[metric] > epsilon
        return hi


def validate(pi, P, partitions, matrices, chains):
    for Q, chain in zip(matrices, chains):
        assert Q.min() >= -2e-14
        np.testing.assert_allclose(Q.sum(axis=1), 1, atol=2e-14)
        np.testing.assert_allclose(pi @ Q, pi, atol=2e-14)
        for t in (0, 1, 2, 5, 10, 30):
            np.testing.assert_allclose(chain.metrics(t),
                direct_metrics(np.linalg.matrix_power(Q, t), pi), atol=2e-11)
    for blocks, Q, chain in zip(partitions, matrices[1:], chains[1:]):
        assert len(blocks) == K and all(len(b) > 0 for b in blocks)
        np.testing.assert_array_equal(np.sort(np.concatenate(blocks)), np.arange(N))
        G, labels, masses = gibbs(pi, blocks)
        np.testing.assert_allclose(G @ G, G, atol=2e-14)
        np.testing.assert_allclose(G @ P @ G, Q, atol=2e-14)
        reduced = np.array([[np.sum(pi[a, None]*P[np.ix_(a, b)])/pi[a].sum()
                             for b in blocks] for a in blocks])
        for t in (1, 2, 5, 10, 30):
            power = np.linalg.matrix_power(reduced, t)
            lifted = power[np.ix_(labels, labels)]*pi[None, :]/masses[labels][None, :]
            np.testing.assert_allclose(chain.metrics(t), direct_metrics(lifted, pi),
                                       atol=2e-11)


def configure_plots():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#243247", "text.color": "#243247",
        "axes.edgecolor": "#ABB5C2", "xtick.color": "#526174",
        "ytick.color": "#526174", "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    })


def distance_plot(times, values, mixing, metric, path):
    fig, ax = plt.subplots(figsize=(7.4, 5.3), layout="constrained")
    title = ("Worst-case total variation", "Squared weighted Frobenius distance")[metric]
    fig.suptitle(title, fontsize=15, weight="bold", x=.08, ha="left")
    ax.set_title(r"$N=100$, $k=4$, $\beta=2$",
                 loc="left", fontsize=10, color="#607087", pad=12)
    for i in range(3):
        ys = values[i, :, metric]
        valid = ys >= DISPLAY_FLOOR
        ax.plot(times[valid], ys[valid], color=CHAIN_COLORS[i], lw=2.2,
                label=CHAIN_LABELS[i], zorder=3)
        below = np.flatnonzero(~valid)
        if len(below):
            first = int(below[0])
            if first:
                ax.plot(times[first-1:first+1], [ys[first-1], DISPLAY_FLOOR],
                        color=CHAIN_COLORS[i], lw=2.2)
            ax.plot(times[first], DISPLAY_FLOOR, "v", color=CHAIN_COLORS[i],
                    ms=6, clip_on=False)
        crossing = mixing[i][metric]
        at_crossing = ys[np.searchsorted(times, crossing)]
        ax.scatter([crossing], [at_crossing], s=30, color=CHAIN_COLORS[i],
                   edgecolors="white", linewidths=.7, zorder=5)
    ax.axhline(EPSILON, color="#7B8795", ls=(0, (4, 4)), lw=1)
    fig.supxlabel(
        "First integer t with distance ≤ 0.01:  "
        + "    ".join(f"{label}: {mixing[i][metric]:,}" for i, label in enumerate(CHAIN_LABELS)),
        fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, times[-1])
    ax.set_ylim(DISPLAY_FLOOR*.65, 1.3 if metric == 0 else 150)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#E6EAF0", lw=.8)
    ax.set_xlabel("Number of transitions t  (log scale)")
    ax.set_ylabel((r"$\max_x\,\|Q^t(x,\cdot)-\pi\|_{\rm TV}$",
                   r"$\|Q^t-\Pi\|_{F,\pi}^{\,2}$")[metric])
    ax.legend(loc="lower left", frameon=True, framealpha=.97, edgecolor="#DEE3EA")
    fig.savefig(path, dpi=250)
    plt.close(fig)


def partition_plot(states, energy, pi, partitions, path):
    fig = plt.figure(figsize=(11.5, 6.1), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(2.35, 1))
    upper = [fig.add_subplot(grid[0, i]) for i in range(2)]
    lower = [fig.add_subplot(grid[1, i]) for i in range(2)]
    fig.suptitle("Two four-block partitions of the same Gibbs target", fontsize=15,
                 weight="bold")
    captions = (r"$G_1$: exact for $\phi(u)=u^{1.5}$",
                r"$G_2$: greedy for $\phi(u)=u^3$")
    for i, blocks in enumerate(partitions):
        G, labels, masses = gibbs(pi, blocks)
        ax, bar = upper[i], lower[i]
        ax.plot(states, pi, color="#AEB8C6", lw=1, zorder=1)
        for j in range(K):
            indices = np.sort(blocks[j])
            ax.scatter(indices, pi[indices], s=26, color=BLOCK_COLORS[j],
                       edgecolors="white", linewidths=.35, label=f"Block {j+1}", zorder=3)
        if i == 0:
            rare = np.concatenate(blocks[:K-1])
            for j, x in enumerate(rare):
                ax.annotate(str(x), (x, pi[x]), xytext=(-13+13*j, 25+12*(j % 2)),
                            textcoords="offset points", fontsize=8.5,
                            color=BLOCK_COLORS[j], ha="center",
                            arrowprops=dict(arrowstyle="-", lw=.8, color=BLOCK_COLORS[j]))
        ax.set_title(captions[i], fontsize=12.2, pad=9)
        ax.set_xlim(-2, 101)
        ax.set_ylim(0, 1.19*pi.max())
        ax.set_xlabel("State x on the cycle")
        if i == 0:
            ax.set_ylabel(r"Target probability $\pi(x)$")
        ax.grid(axis="y", color="#E8ECF1", lw=.7)
        ax.legend(loc="upper right", ncol=2, fontsize=8.5, framealpha=.96,
                  columnspacing=.8, handlelength=1, borderpad=.4)
        bar.bar(np.arange(K), masses, color=BLOCK_COLORS, width=.64)
        bar.set_ylim(0, 1.19)
        bar.set_yticks([0, .25, .5, .75, 1])
        bar.set_xticks(np.arange(K), [f"Block {j+1}\n{len(b)} state" + ("s" if len(b)>1 else "")
                                    for j,b in enumerate(blocks)])
        bar.tick_params(axis="x", labelsize=9)
        bar.grid(axis="y", color="#E8ECF1", lw=.7)
        bar.set_axisbelow(True)
        if i == 0:
            bar.set_ylabel("Block mass")
        for j,mass in enumerate(masses):
            bar.text(j, mass+.045, f"{100*mass:.4f}%", ha="center", fontsize=8.9)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states, energy, pi, P = build_model()
    partitions = [exact_partition(pi), greedy_partition(pi)]
    projections = [gibbs(pi, b)[0] for b in partitions]
    matrices = [P]+[G @ P @ G for G in projections]
    chains = [Chain(Q, pi) for Q in matrices]
    validate(pi, P, partitions, matrices, chains)
    mixing = [[chain.first_below(metric) for metric in range(2)] for chain in chains]
    times = np.unique(np.r_[np.arange(1, 101),
                            np.rint(np.geomspace(101, 100000, 350)).astype(int),
                            np.array(mixing).ravel()])
    values = np.array([[chain.metrics(t) for t in times] for chain in chains])
    assert np.all(values[1:, :, 1] <= values[0, :, 1]+2e-11)
    for i in range(3):
        assert np.all(np.diff(values[i], axis=0) <= 2e-11)
    configure_plots()
    distance_plot(times, values, mixing, 0, args.output_dir/"gpg_worst_case_tv.png")
    distance_plot(times, values, mixing, 1, args.output_dir/"gpg_squared_frobenius.png")
    partition_plot(states, energy, pi, partitions, args.output_dir/"gpg_partitions.png")
    report = {
        "N": N, "k": K, "beta": BETA, "epsilon": EPSILON,
        "energy": "-cos(8*pi*x/100)-0.15*cos(2*pi*x/100)",
        "state_indices": states.tolist(), "pi": pi.tolist(),
        "tie_rule": "lowest state index; lowest block index",
        "partitions": {}, "chains": {},
    }
    for i,blocks in enumerate(partitions):
        masses = np.array([pi[b].sum() for b in blocks])
        p = (1.5, 3)[i]
        report["partitions"][f"G{i+1}"] = {
            "blocks": [np.sort(b).tolist() for b in blocks],
            "counts": [len(b) for b in blocks], "masses": masses.tolist(),
            "power": p, "design_objective": float(np.sum(masses**(2-p))-1),
        }
    for i,chain in enumerate(chains):
        report["chains"][CHAIN_NAMES[i]] = {
            "worst_case_tv_mixing_time": mixing[i][0],
            "squared_frobenius_mixing_time": mixing[i][1],
            "nonzero_centered_eigenvalues": chain.values.tolist(),
            "metrics_at_selected_times": {str(t): {"worst_case_tv": chain.metrics(t)[0],
                "squared_frobenius": chain.metrics(t)[1]} for t in (0,1,2,5,10,20,30,50)},
        }
    (args.output_dir/"gpg_numerical_results.json").write_text(json.dumps(report, indent=2)+"\n")
    with (args.output_dir/"gpg_numerical_curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t"]+[f"{name}_{metric}" for name in CHAIN_NAMES
                                for metric in ("worst_case_tv", "squared_frobenius")])
        writer.writerow([0]+[v for chain in chains for v in chain.metrics(0)])
        for j,t in enumerate(times):
            writer.writerow([int(t)]+values[:,j,:].ravel().tolist())
    print("Kernel    worst-case TV t_mix(.01)    squared-Frobenius t_mix(.01)")
    for name,row in zip(CHAIN_NAMES,mixing):
        print(f"{name:8s}  {row[0]:>12,d}                    {row[1]:>12,d}")
    print("All stationarity, reversibility, direct-power, and block-reduction checks passed.")
    print(f"Saved three figures, numerical curves, and results in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
