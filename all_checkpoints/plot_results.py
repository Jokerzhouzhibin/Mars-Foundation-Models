"""
Mars Remote Sensing Foundation Model Benchmark — Publication Figures
====================================================================
  Fig 1: KNN best F1-Micro — horizontal bar (2×3 grid)
  Fig 2: Linear Probe — vertical bar (2×3 grid)
  Fig 3: Full Fine-tuning — vertical bar (2×3 grid)
  Fig 4: Data efficiency — line plots (2×3 grid)
  Fig 5: Heatmap summary — 3 methods side by side

Output: PDF (vector) + PNG (raster)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Academic Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'axes.grid': False,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.4,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Naming ────────────────────────────────────────────────────────────
DATASET_PRETTY = {
    'mb-atmospheric_dust_cls_rdr': 'Atmos. Dust',
    'mb-domars16k': 'DoMars16k',
    'mb-frost_cls': 'Frost',
    'mb-change_cls_hirise': 'Change-HiRISE',
    'mb-change_cls_ctx': 'Change-CTX',
    'mb-landmark_cls': 'Landmark',
}

MODEL_PRETTY = {
    'dinov3_sat': 'DINOv3-SAT',
    'dinov3_lvd': 'DINOv3-LVD',
    'dinov3_sat_1cha': 'DINOv3-SAT$_{1ch}$',
    'dinov3_lvd_1cha': 'DINOv3-LVD$_{1ch}$',
    'dinov2_l14': 'DINOv2-L/14',
    'dinov1_b8': 'DINOv1-B/8',
    'mars_mae': 'Mars-MAE',
    'satmae': 'SatMAE++',
    'openclip': 'OpenCLIP',
    'siglip2': 'SigLIP2',
    'vit_l': 'ViT-L/16',
    'swin_v2': 'Swin-V2-B',
}

MODEL_ORDER = [
    'dinov3_sat', 'dinov3_lvd', 'dinov3_sat_1cha', 'dinov3_lvd_1cha',
    'dinov2_l14', 'dinov1_b8',
    'mars_mae', 'satmae',
    'openclip', 'siglip2', 'vit_l', 'swin_v2',
]

DATASET_ORDER = [
    'mb-atmospheric_dust_cls_rdr', 'mb-domars16k', 'mb-frost_cls',
    'mb-change_cls_hirise', 'mb-change_cls_ctx', 'mb-landmark_cls',
]

PAL = {
    'dinov3_sat':      '#c0392b',
    'dinov3_lvd':      '#e67e22',
    'dinov3_sat_1cha': '#e74c3c',
    'dinov3_lvd_1cha': '#f39c12',
    'dinov2_l14':      '#2980b9',
    'dinov1_b8':       '#3498db',
    'mars_mae':        '#27ae60',
    'satmae':          '#8e44ad',
    'openclip':        '#7f8c8d',
    'siglip2':         '#95a5a6',
    'vit_l':           '#34495e',
    'swin_v2':         '#bdc3c7',
}

MARKERS = {
    'dinov3_sat': 'o', 'dinov3_lvd': 's', 'dinov3_sat_1cha': '^', 'dinov3_lvd_1cha': 'v',
    'dinov2_l14': 'D', 'dinov1_b8': 'p',
    'mars_mae': '*', 'satmae': 'h',
    'openclip': 'X', 'siglip2': 'P', 'vit_l': 'd', 'swin_v2': '8',
}

LSTYLES = {
    'dinov3_sat': '-', 'dinov3_lvd': '-', 'dinov3_sat_1cha': '--', 'dinov3_lvd_1cha': '--',
    'dinov2_l14': '-', 'dinov1_b8': '-.',
    'mars_mae': '-', 'satmae': ':',
    'openclip': '-.', 'siglip2': '-.', 'vit_l': ':', 'swin_v2': ':',
}

def pm(m): return MODEL_PRETTY.get(m, m)
def pd_ds(d): return DATASET_PRETTY.get(d, d)

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'))
    print(f'  ✅ {name}.pdf / .png')


# =====================================================================
# Fig 1: KNN Best F1-Micro — Horizontal Bar (2×3)
# =====================================================================
def fig1_knn(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=['Dataset', 'Model', 'F1_Micro'])
    df = df[df['Dataset'].isin(DATASET_ORDER) & df['Model'].isin(MODEL_ORDER)]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.0))
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASET_ORDER):
        sub = df[df['Dataset'] == ds].copy()
        sub['Model'] = pd.Categorical(sub['Model'], categories=MODEL_ORDER[::-1], ordered=True)
        sub = sub.sort_values('Model')

        scores = sub['F1_Micro'].values * 100
        models = sub['Model'].values
        colors = [PAL.get(m, '#555') for m in models]
        y_pos = np.arange(len(models))

        bars = ax.barh(y_pos, scores, color=colors, edgecolor='white', linewidth=0.3, height=0.72)

        best_idx = np.argmax(scores)
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(1.0)

        for i, (bar, val) in enumerate(zip(bars, scores)):
            fw = 'bold' if i == best_idx else 'normal'
            ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', ha='left', fontsize=6, fontweight=fw)

        lo = max(scores.min() - 4, 0)
        ax.set_xlim(lo, 103)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([pm(m) for m in models], fontsize=7)
        ax.set_title(pd_ds(ds), fontsize=9, fontweight='bold')
        ax.tick_params(axis='y', length=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Only bottom row gets x-label
        if ax in axes[1]:
            ax.set_xlabel('F1-Micro (%)', fontsize=8)

    fig.suptitle('(a) KNN Evaluation', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, 'fig1_knn_best')
    plt.close(fig)


# =====================================================================
# Fig 2 / Fig 3: Single-method vertical bar chart (2×3, reusable)
# =====================================================================
def _plot_single_method(csv_path, title_prefix, fig_label, out_name, score_col='Score'):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Dataset', 'Model', score_col])
    df = df[df['Dataset'].isin(DATASET_ORDER) & df['Model'].isin(MODEL_ORDER)]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.5))
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASET_ORDER):
        sub = df[df['Dataset'] == ds].copy()
        models = [m for m in MODEL_ORDER if m in sub['Model'].values]
        scores = [sub[sub['Model'] == m][score_col].values[0] for m in models]
        colors = [PAL.get(m, '#555') for m in models]
        x = np.arange(len(models))

        bars = ax.bar(x, scores, color=colors, edgecolor='white', linewidth=0.3, width=0.7)

        best_idx = int(np.argmax(scores))
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(1.0)

        ax.text(best_idx, scores[best_idx] + 0.3, f'{scores[best_idx]:.1f}',
                ha='center', va='bottom', fontsize=6.5, fontweight='bold')

        vmin = min(scores)
        ax.set_ylim(max(vmin - 6, 0), 102)
        ax.set_xticks(x)
        ax.set_xticklabels([pm(m) for m in models], rotation=55, ha='right', fontsize=7)
        ax.set_title(pd_ds(ds), fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # y-label only on left column
        if ax in axes[:, 0]:
            ax.set_ylabel('Accuracy (%)', fontsize=8)

    fig.suptitle(f'({fig_label}) {title_prefix}', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout(h_pad=1.2, w_pad=0.8)
    save(fig, out_name)
    plt.close(fig)


def fig2_linear(csv_path):
    _plot_single_method(csv_path, 'Linear Probe', 'b', 'fig2_linear_probe')


def fig3_full(csv_path):
    _plot_single_method(csv_path, 'Full Fine-tuning', 'c', 'fig3_full_finetune')


# =====================================================================
# Fig 4: Data Efficiency Curves (2×3)
# =====================================================================
def fig4_efficiency(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=['Dataset', 'Model', 'Datafraction', 'Score'])
    df = df.groupby(['Dataset', 'Model', 'Datafraction'])['Score'].max().reset_index()
    df = df[df['Dataset'].isin(DATASET_ORDER) & df['Model'].isin(MODEL_ORDER)]

    frac_order = ['0.01', '0.05', '0.20', '0.50', 'full']
    frac_labels = ['1%', '5%', '20%', '50%', '100%']
    df['Datafraction'] = df['Datafraction'].astype(str)
    df = df[df['Datafraction'].isin(frac_order)]

    show = ['dinov3_sat', 'dinov3_lvd', 'dinov3_sat_1cha', 'dinov3_lvd_1cha',
            'dinov2_l14', 'dinov1_b8', 'mars_mae', 'openclip', 'siglip2']

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.5))
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASET_ORDER):
        sub = df[df['Dataset'] == ds]
        for model in show:
            msub = sub[sub['Model'] == model].copy()
            if msub.empty:
                continue
            msub['fi'] = msub['Datafraction'].map({f: i for i, f in enumerate(frac_order)})
            msub = msub.dropna(subset=['fi']).sort_values('fi')
            if len(msub) < 2:
                continue
            ax.plot(msub['fi'], msub['Score'],
                    marker=MARKERS.get(model, 'o'), markersize=4.5, linewidth=1.3,
                    linestyle=LSTYLES.get(model, '-'),
                    color=PAL.get(model, '#555'),
                    label=pm(model), markeredgewidth=0.3, markeredgecolor='white')

        ax.set_xticks(range(len(frac_labels)))
        ax.set_xticklabels(frac_labels, fontsize=7)
        ax.set_title(pd_ds(ds), fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linewidth=0.3, alpha=0.5)
        # Labels
        if ax in axes[1]:
            ax.set_xlabel('Data Fraction', fontsize=8)
        if ax in axes[:, 0]:
            ax.set_ylabel('Accuracy (%)', fontsize=8)

    # Shared legend below figure
    all_h, all_l, seen = [], [], set()
    for a in axes_flat:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in seen:
                all_h.append(h); all_l.append(l); seen.add(l)
    fig.legend(all_h, all_l, loc='lower center', ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.06), frameon=True, edgecolor='0.8')

    fig.suptitle('(d) Data Efficiency — Linear Probe', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout(h_pad=1.0, w_pad=0.6)
    save(fig, 'fig4_data_efficiency')
    plt.close(fig)


# =====================================================================
# Fig 5: Heatmap Summary — 3 methods side by side
# =====================================================================
def fig5_heatmap(knn_csv, linear_csv, full_csv):
    dk = pd.read_csv(knn_csv).dropna(subset=['Dataset', 'Model', 'F1_Micro'])
    dk = dk[dk['Dataset'].isin(DATASET_ORDER) & dk['Model'].isin(MODEL_ORDER)]
    dk['Score'] = dk['F1_Micro'] * 100

    def load_train(path):
        d = pd.read_csv(path, skipinitialspace=True)
        d.columns = d.columns.str.strip()
        return d.dropna(subset=['Dataset', 'Model', 'Score'])

    dl = load_train(linear_csv)
    dl = dl[dl['Dataset'].isin(DATASET_ORDER) & dl['Model'].isin(MODEL_ORDER)]
    df_ = load_train(full_csv)
    df_ = df_[df_['Dataset'].isin(DATASET_ORDER) & df_['Model'].isin(MODEL_ORDER)]

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[0, 3])

    titles = ['KNN', 'Linear Probe', 'Full Fine-tune']
    for idx, (ax, title, src) in enumerate(zip(axes, titles, [dk, dl, df_])):
        piv = src.pivot_table(index='Model', columns='Dataset', values='Score', aggfunc='max')
        models_p = [m for m in MODEL_ORDER if m in piv.index]
        ds_p = [d for d in DATASET_ORDER if d in piv.columns]
        piv = piv.loc[models_p, ds_p]

        annot_plain = piv.map(lambda x: f'{x:.1f}' if pd.notna(x) else '')

        piv.index = [pm(m) for m in piv.index]
        piv.columns = [pd_ds(d) for d in piv.columns]
        annot_plain.index = piv.index
        annot_plain.columns = piv.columns

        vmin_val = piv.min().min()

        is_last = (idx == 2)
        sns.heatmap(piv, annot=annot_plain, fmt='', cmap='YlOrRd',
                    linewidths=0.4, linecolor='white',
                    cbar=is_last, cbar_ax=cbar_ax if is_last else None,
                    cbar_kws={'label': 'Score (%)'} if is_last else {},
                    ax=ax, vmin=max(vmin_val - 3, 0), vmax=100,
                    annot_kws={'fontsize': 6})

        # Bold best per column
        for j, col in enumerate(piv.columns):
            best_row = piv[col].idxmax()
            i = list(piv.index).index(best_row)
            ax.texts[i * len(piv.columns) + j].set_fontweight('bold')

        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', rotation=35)
        ax.tick_params(axis='y', rotation=0)
        if idx != 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('')

    fig.suptitle('(e) Performance Summary Across Evaluation Protocols',
                 fontsize=11, fontweight='bold', y=1.02)
    save(fig, 'fig5_heatmap_summary')
    plt.close(fig)


# =====================================================================
# Fig 6: DoMars16k KNN Comparison Table (publication-style)
# =====================================================================
def fig6_knn_table(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=['Dataset', 'Model', 'F1_Micro'])
    df = df[df['Dataset'] == 'mb-domars16k']
    df = df[df['Model'].isin(MODEL_ORDER)]

    # Sort by F1_Micro descending
    df = df.sort_values('F1_Micro', ascending=False).reset_index(drop=True)

    # Build table data
    rows = []
    for _, r in df.iterrows():
        rows.append([
            pm(r['Model']),
            r['Pooling'].upper() if r['Pooling'] != 'gem' else 'GeM',
            int(r['K']),
            f"{r['F1_Micro']*100:.2f}",
            f"{r['F1_Macro']*100:.2f}",
        ])

    col_labels = ['Model', 'Pooling', 'K', 'Micro F1 (%)', 'Macro F1 (%)']
    n_rows = len(rows)
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(5.0, 0.38 * n_rows + 0.8))
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=8.5)
        cell.set_edgecolor('white')
        cell.set_height(0.06)

    # Style data rows
    for i in range(1, n_rows + 1):
        bg = '#f9f9f9' if i % 2 == 0 else '#ffffff'
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor('#dddddd')
            cell.set_height(0.055)
            # Bold the best row (first row = rank 1)
            if i == 1:
                cell.set_text_props(fontweight='bold')

    # Column widths
    col_widths = [0.30, 0.14, 0.10, 0.22, 0.22]
    for j, w in enumerate(col_widths):
        for i in range(n_rows + 1):
            table[i, j].set_width(w)

    table.auto_set_column_width(list(range(n_cols)))

    ax.set_title('KNN Evaluation on DoMars16k', fontsize=10, fontweight='bold',
                 pad=12, loc='center')

    plt.tight_layout()
    save(fig, 'fig6_knn_table_domars16k')
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))

    knn_best    = os.path.join(base, 'knn_checkpoints', 'best_summary_knn.csv')
    linear_best = os.path.join(base, 'train_all_logs', 'best_results_linear.csv')
    full_best   = os.path.join(base, 'train_all_logs', 'best_results_full.csv')
    gen_all     = os.path.join(base, 'logs_generalization', 'generalization_results_all.csv')

    print('=' * 60)
    print('  Mars Benchmark — Publication Figures')
    print('=' * 60)

    fig1_knn(knn_best)
    fig2_linear(linear_best)
    fig3_full(full_best)
    fig4_efficiency(gen_all)
    fig5_heatmap(knn_best, linear_best, full_best)
    fig6_knn_table(knn_best)

    print(f'\n🎉 All figures → {OUT_DIR}/')
