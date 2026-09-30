"""Recreate the review figure from recorded CSVs; run before sealing the review."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent
if (folder / 'artifacts.json').exists():
    raise SystemExit('Review is sealed; copy the CSVs and this script to a new directory before rendering.')
df = pd.read_csv(folder / 'comparison.csv').sort_values('macro_f1')
classes = pd.read_csv(folder / 'per_class_comparison.csv')
best = 'no_smoothing_flips4'
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]}, layout='constrained')
colors = ['#d77e15' if name == best else '#38789c' for name in df.candidate]
axes[0].barh(df.candidate, df.macro_f1, color=colors)
axes[0].axvline(.7300203789835792, color='#b62a2a', linestyle='--', label='Melhor CNN individual: 0,7300')
axes[0].set(xlim=(.68, .798), xlabel='Macro-F1 em validation (eixo truncado)',
            title='Nove combinações fixas; 1.502 imagens / 1.121 grupos')
axes[0].legend(loc='lower right', fontsize=8)
for i, (_, row) in enumerate(df.iterrows()):
    axes[0].text(row.macro_f1 + .001, i, f'{row.macro_f1:.3f} | {row.forward_passes} forwards', va='center', fontsize=8)
x = np.arange(len(classes)); width = .38
axes[1].bar(x-width/2, classes.baseline_f1, width, label='CNN individual (referência)', color='#38789c')
axes[1].bar(x+width/2, classes.candidate_f1, width, label='3 CNNs sem smoothing + flips4', color='#d77e15')
axes[1].set(xticks=x, xticklabels=classes['class'], ylim=(0, 1), ylabel='F1 por classe', xlabel='Classe',
            title='Ganho concentrado principalmente nas classes raras')
axes[1].legend(fontsize=8)
fig.suptitle('Comparação de desenvolvimento — validation utilizada para seleção', fontsize=13)
fig.savefig(folder / 'comparison.png', dpi=160)
fig.savefig(folder / 'comparison.svg')
plt.close(fig)
