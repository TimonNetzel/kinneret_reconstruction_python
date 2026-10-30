"""Render the fitted forward responses; contours are not uncertainty intervals."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model import probabilities


def plot(models_path, scores_path, target):
    models = json.loads(models_path.read_text())
    with scores_path.open() as handle:
        scores = {(r['taxon'], int(r['block_degrees'])): float(r['brier_skill'])
                  for r in csv.DictReader(handle) if r['scope'] == 'all'}
    t = np.linspace(-30, 30, 241)
    p = np.linspace(0, 4000, 181)
    tt, pp = np.meshgrid(t, p)
    x = np.column_stack([tt.ravel(), pp.ravel()])
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.4), constrained_layout=True)
    for ax, (name, model) in zip(axes.ravel(), models.items()):
        values, inside = probabilities(model, x, reject_extrapolation=False)
        field = np.ma.masked_where(~inside.reshape(tt.shape), values.reshape(tt.shape))
        ax.set_facecolor('#e6e7e9')
        picture = ax.pcolormesh(tt, pp, field, shading='nearest', cmap='viridis', vmin=0, vmax=1, rasterized=True)
        levels = [v for v in [.1, .5, .9] if field.min() < v < field.max()]
        if levels:
            contours = ax.contour(tt, pp, field, levels=levels, colors='white', linewidths=.65)
            ax.clabel(contours, fmt='%g', fontsize=7)
        ax.set_title(model['taxon']['name'], fontsize=12, fontstyle='italic', loc='left')
        ax.text(.02, .98, f"Brier-Skill 5° / 10°: {scores[name,5]:.3f} / {scores[name,10]:.3f}",
                va='top', transform=ax.transAxes, fontsize=8,
                bbox={'facecolor':'white', 'alpha':.85, 'edgecolor':'none', 'pad':3})
        ax.set_xlabel('TANN (°C)')
        ax.set_ylabel('PANN (mm/Jahr)')
        ax.set_xlim(-30, 30)
        ax.set_ylim(0, 4000)
    axes.ravel()[-1].axis('off')
    axes.ravel()[-1].text(0, .93, 'Sieben neue Artenfunktionen', fontsize=14, weight='bold', va='top')
    axes.ravel()[-1].text(0, .79,
        'Farbe: Wahrscheinlichkeit des\nkartierten nativen Arealindikators.\n\nGrau: außerhalb der Trainingshülle.\nKonturen: gleiche Wahrscheinlichkeit,\nkeine Unsicherheitsintervalle.\n\nMeusel-Areale / BfN-GeoJSON;\nWorldClim 2.1, 1970–2000.\n\nExperimentelle Kalibrierung.\nKeine Pollen- oder Fossilvorhersage.',
        fontsize=10, va='top', linespacing=1.4)
    fig.colorbar(picture, ax=list(axes.ravel()[:-1]), label='P(kartierter nativer Arealindikator | TANN, PANN)',
                 fraction=.025, pad=.02, shrink=.85)
    fig.suptitle('Botanische Transferfunktionen: erster geprüfter Trainingsstand', fontsize=17, weight='bold')
    fig.savefig(target, dpi=120, metadata={'Software':'Matplotlib'})
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.models, args.scores, args.output)
