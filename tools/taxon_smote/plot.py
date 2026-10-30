"""Render the complete aggregate comparison; no fitting or result selection."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(summary, output):
    with summary.open() as handle:
        rows = list(csv.DictReader(handle))
    taxa = list(dict.fromkeys(r['taxon'] for r in rows))
    lookup = {(r['taxon'], int(r['block_degrees']), r['method']): r for r in rows}
    methods = ['pr28_quadratic', 'nnet_original', 'nnet_smote', 'nnet_smote_prior_adjusted']
    labels = ['Regression #28', 'Netz ohne SMOTE', 'Netz mit SMOTE', 'SMOTE + Prior-Diagnose']
    colors = ['#7b8593', '#2166ac', '#e68a17', '#38816b']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout='constrained')
    positions = np.arange(len(taxa))
    for col, block in enumerate([5, 10]):
        for row, metric in enumerate(['balanced_accuracy', 'brier']):
            ax = axes[row, col]
            for i, method in enumerate(methods):
                values = [float(lookup[t, block, method][metric]) for t in taxa]
                ax.bar(positions+(i-1.5)*.18, values, .17, color=colors[i], label=labels[i])
            ax.set_xticks(positions, [t.replace('_', ' ').capitalize() for t in taxa], rotation=28, ha='right', fontsize=8)
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'{block}°-Blöcke · 5 Folds · 100-km-Puffer')
            ax.set_ylabel('Balanced Accuracy (höher besser)' if row == 0 else 'Flächengewichteter Brier (niedriger besser)')
            if row == 0:
                ax.set_ylim(.45, 1.)
                ax.axhline(.5, color='#444444', linewidth=.6)
    fig.suptitle('Sieben Artenfunktionen: identische räumliche Original-Testzellen', fontsize=15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False)
    fig.savefig(output, dpi=150, metadata={'Software': 'Matplotlib; source metrics_summary.csv'})
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    plot(a.summary, a.output)
