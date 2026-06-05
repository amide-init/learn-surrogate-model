"""Regenerate COCO comparison figures with custom x-axis style.

Uses only documented public API from cocopp:
  - genericsettings.xlimit_pprldmany  — documented x-axis config variable
  - pprldmany.save_figure / close_figure — public booleans to suppress auto-save
  - pprldmany.x_limit                 — per-call override of the x-axis cap
  - ppfig.save_figure()               — called manually after axis fix-up

Customisations:
  - x-axis capped at 250 / dimension
  - tick labels show actual values (1, 10, 100, …) not log10 exponents
  - log scale preserved
  - xlabel "FEvals / DIM" shown only on the bottom (highest) dimension subplot

Usage:
    cd /path/to/learn-surrogate
    source .venv/bin/activate
    python compare/run_compare.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cocopp
from cocopp import ppfig, genericsettings, pproc as pp
from cocopp.compall import pprldmany
from cocopp import rungenericmany

BUDGET = 250  # x-axis cap = BUDGET / dim

_HERE = os.path.dirname(os.path.abspath(__file__))
ALGORITHMS = [
    os.path.join(_HERE, "AFN-CMA-ES"),
    os.path.join(_HERE, "DTS-CMA-ES"),
    os.path.join(_HERE, "LMM-CMA-ES"),
    os.path.join(_HERE, "LQ-CMA-ES")
]
OUTPUT_DIR = os.path.join(_HERE, "ppdata")

# ── axis fix-up applied after pprldmany draws but before we save ──────────────

def _fix_axes(ax, show_xlabel):
    """Linear x-axis 0–250 (budget = 250×DIM, axis = FEvals/DIM)."""
    ax.set_xscale("linear")
    ax.set_xlim(0, BUDGET)
    ticks = [0, 50, 100, 150, 200, 250]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("FEvals / DIM" if show_xlabel else "")


# ── monkey-patch pprldmany.main ───────────────────────────────────────────────
# Uses the documented public booleans save_figure / close_figure so that we
# get control of the figure AFTER pprldmany has drawn but BEFORE it is saved.

_orig_main = pprldmany.main
_show_xlabel = True  # controlled by the grouped-loop wrapper below


def _custom_main(dictAlg, order=None, outputdir=".", info="default",
                 dimension=None, parentHtmlFileName=None,
                 plotType=pprldmany.PlotType.ALG, settings=genericsettings):
    global _show_xlabel

    # ── determine dimension and effective x-axis limit ────────────────────────
    tmp = pp.dictAlgByDim(dictAlg)
    dims = sorted(tmp.keys())
    dim = dims[0] if dims else 2
    xlimit = BUDGET  # always 250: budget = 250×DIM, axis = FEvals/DIM
    print(f"Custom plotting active: dim={dim}, xlimit={xlimit}")

    # pprldmany.x_limit is the module-level variable that controls the x-axis;
    # genericsettings.xlimit_pprldmany is its documented source of truth.
    orig_xlimit = pprldmany.x_limit
    pprldmany.x_limit = xlimit

    # ── use documented flags to suppress auto-save/close ─────────────────────
    pprldmany.save_figure = False
    pprldmany.close_figure = False

    _orig_main(dictAlg, order=order, outputdir=outputdir, info=info,
               dimension=dimension, parentHtmlFileName=parentHtmlFileName,
               plotType=plotType, settings=settings)

    # ── figure is still open: fix up the axes ─────────────────────────────────
    # cocopp's right-side annotation legend is drawn in log-scale coordinates;
    # switching to linear invalidates its positions, so we remove those artists
    # and replace with a proper matplotlib figure legend.
    fig = plt.gcf()
    ax_main = fig.axes[0] if fig.axes else None

    # collect all non-hidden line handles/labels from the axes
    handles, labels = [], []
    if ax_main:
        for line in ax_main.get_lines():
            lbl = line.get_label()
            if lbl and not lbl.startswith("_"):
                clean = os.path.basename(lbl)
                if clean not in labels:
                    handles.append(line)
                    labels.append(clean)

    # switch to linear scale and fix ticks
    for ax in fig.axes:
        _fix_axes(ax, _show_xlabel)

    # add a compact legend inside the main axes (top-left keeps it away from data)
    if handles:
        ax_main.legend(handles, labels, loc="upper left",
                       fontsize=7, framealpha=0.8, ncol=1)

    # ── save using ppfig.save_figure (documented public API) ─────────────────
    pfile = genericsettings.pprldmany_file_name
    fig_name = (os.path.join(outputdir, "%s_%s" % (pfile, info))
                if info != "default"
                else os.path.join(outputdir, pfile))

    algs_with_data = [a for a in dictAlg if dictAlg[a] != []]
    alg_id = dictAlg[algs_with_data[0]][0].algId

    n_funcs = len(pp.dictAlgByFun(dictAlg))
    ppfig.save_figure(
        fig_name,
        alg_id,
        layout_rect=(0, 0, 1, 1),
        subplots_adjust=dict(
            bottom=0.135,
            right=0.97,
            top=0.92 if n_funcs == 1 else 0.98,
        ),
    )
    plt.close()

    # ── restore module state ──────────────────────────────────────────────────
    pprldmany.save_figure = True
    pprldmany.close_figure = True
    pprldmany.x_limit = orig_xlimit


pprldmany.main = _custom_main


# ── monkey-patch grouped_ecdf_graphs ─────────────────────────────────────────
# Wraps the dimension loop so _show_xlabel is True only for the last (bottom)
# dimension in each function-group panel.

_orig_grouped = rungenericmany.grouped_ecdf_graphs


def _custom_grouped(alg_dict, order, output_dir, function_groups,
                    settings, parent_file_name):
    global _show_xlabel
    for gr, tmpdictAlg in alg_dict.items():
        dictDim = pp.dictAlgByDim(tmpdictAlg)
        dims = sorted(dictDim)
        last_dim = dims[-1] if dims else None
        for d in dims:
            _show_xlabel = (d == last_dim)
            # Pass a single-dimension slice so _custom_main sees one dim
            _orig_grouped(
                {gr: dictDim[d]},
                order, output_dir, function_groups, settings, parent_file_name,
            )
    _show_xlabel = True  # restore default


rungenericmany.grouped_ecdf_graphs = _custom_grouped


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    genericsettings.interactive_mode = False  # suppress auto browser open
    cocopp.main("-o " + OUTPUT_DIR + " " + " ".join(ALGORITHMS))
    print("\nFigures saved to", OUTPUT_DIR)
