"""
DrawMe - Training Metrics & Model Analysis Visualization
Generates meaningful charts from training history, model weights, and metadata.
"""

import os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# ─── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")

# ─── Styling ───────────────────────────────────────────────────────────────────

C = {
    'bg': '#0f0f1a', 'card': '#1a1a2e', 'card2': '#16213e',
    'blue': '#4fc3f7', 'purple': '#ab47bc', 'green': '#66bb6a',
    'orange': '#ffa726', 'red': '#ef5350', 'cyan': '#26c6da',
    'pink': '#ec407a', 'yellow': '#ffee58', 'teal': '#26a69a',
    'txt': '#e0e0e0', 'txt2': '#9e9e9e', 'grid': '#2a2a4a',
}
CAT_COLORS = [
    '#4fc3f7', '#ab47bc', '#66bb6a', '#ffa726', '#ef5350',
    '#26c6da', '#ec407a', '#ffee58', '#7e57c2', '#ff7043',
    '#42a5f5', '#9ccc65', '#5c6bc0', '#29b6f6', '#8d6e63'
]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial'],
    'font.size': 11, 'axes.facecolor': C['card'], 'axes.edgecolor': C['grid'],
    'axes.labelcolor': C['txt'], 'text.color': C['txt'],
    'xtick.color': C['txt2'], 'ytick.color': C['txt2'],
    'figure.facecolor': C['bg'], 'grid.color': C['grid'], 'grid.alpha': 0.3,
    'lines.linewidth': 2.5, 'lines.antialiased': True,
})


def load_json(filename):
    path = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_model_safe():
    """Load Keras model with monkey-patch for BatchNormalization compat."""
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        from tensorflow import keras
        orig = keras.layers.BatchNormalization.from_config
        @classmethod
        def patched(cls, config):
            for k in ['renorm', 'renorm_clipping', 'renorm_momentum', 'fused']:
                config.pop(k, None)
            return orig(config)
        keras.layers.BatchNormalization.from_config = patched
        model_path = os.path.join(SAVE_DIR, "drawme_best.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(SAVE_DIR, "drawme_model.keras")
        if os.path.exists(model_path):
            return keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"  ⚠ Could not load model: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: Training Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

def chart_accuracy(history):
    """01: Training & Validation Accuracy."""
    fig, ax = plt.subplots(figsize=(12, 7))
    epochs = range(1, len(history['accuracy']) + 1)

    ax.fill_between(epochs, history['accuracy'], alpha=0.15, color=C['blue'])
    ax.fill_between(epochs, history['val_accuracy'], alpha=0.15, color=C['purple'])
    ax.plot(epochs, history['accuracy'], color=C['blue'], label='Training',
            linewidth=2.5, marker='o', markersize=6, zorder=5)
    ax.plot(epochs, history['val_accuracy'], color=C['purple'], label='Validation',
            linewidth=2.5, marker='s', markersize=6, zorder=5)

    best_i = int(np.argmax(history['val_accuracy']))
    best_v = history['val_accuracy'][best_i]
    ax.annotate(f'Best: {best_v:.4f}', xy=(best_i+1, best_v),
                xytext=(max(1, best_i-1), best_v - 0.015),
                fontsize=11, fontweight='bold', color=C['green'],
                arrowprops=dict(arrowstyle='->', color=C['green'], lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc=C['card2'], ec=C['green'], alpha=0.9))

    final_t, final_v = history['accuracy'][-1], history['val_accuracy'][-1]
    stats = f'Train: {final_t*100:.2f}%\nVal:   {final_v*100:.2f}%\nBest:  {best_v*100:.2f}%'
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10, va='top',
            fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.5', fc=C['card2'], ec=C['cyan'], alpha=0.9))

    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('📈 Model Accuracy Over Training', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9, facecolor=C['card2'], edgecolor=C['grid'])
    ax.set_ylim(min(history['accuracy'] + history['val_accuracy']) - 0.02, 1.0)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '01_accuracy.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 01_accuracy.png")


def chart_loss(history):
    """02: Training & Validation Loss."""
    fig, ax = plt.subplots(figsize=(12, 7))
    epochs = range(1, len(history['loss']) + 1)

    ax.fill_between(epochs, history['loss'], alpha=0.15, color=C['blue'])
    ax.fill_between(epochs, history['val_loss'], alpha=0.15, color=C['purple'])
    ax.plot(epochs, history['loss'], color=C['blue'], label='Training',
            linewidth=2.5, marker='o', markersize=6, zorder=5)
    ax.plot(epochs, history['val_loss'], color=C['purple'], label='Validation',
            linewidth=2.5, marker='s', markersize=6, zorder=5)

    best_i = int(np.argmin(history['val_loss']))
    best_v = history['val_loss'][best_i]
    ax.annotate(f'Best: {best_v:.4f}', xy=(best_i+1, best_v),
                xytext=(max(1, best_i-1), best_v + 0.02),
                fontsize=11, fontweight='bold', color=C['green'],
                arrowprops=dict(arrowstyle='->', color=C['green'], lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc=C['card2'], ec=C['green'], alpha=0.9))

    gap = history['val_loss'][-1] - history['loss'][-1]
    stats = f'Train: {history["loss"][-1]:.4f}\nVal:   {history["val_loss"][-1]:.4f}\nGap:   {gap:.4f}'
    ax.text(0.98, 0.98, stats, transform=ax.transAxes, fontsize=10, va='top', ha='right',
            fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.5', fc=C['card2'], ec=C['orange'], alpha=0.9))

    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss (Cross-Entropy)', fontsize=13, fontweight='bold')
    ax.set_title('📉 Model Loss Over Training', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, facecolor=C['card2'], edgecolor=C['grid'])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '02_loss.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 02_loss.png")


def chart_lr(history):
    """03: Learning Rate Schedule."""
    if 'learning_rate' not in history:
        print("  ⊘ 03_learning_rate.png (no LR data)")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    epochs = range(1, len(history['learning_rate']) + 1)
    lr = history['learning_rate']

    ax.step(list(epochs), lr, color=C['cyan'], linewidth=2.5, where='mid', zorder=5)
    ax.fill_between(epochs, lr, alpha=0.2, color=C['cyan'], step='mid')

    for i in range(1, len(lr)):
        if lr[i] < lr[i-1]:
            ax.axvline(x=i+1, color=C['red'], linestyle='--', alpha=0.5)
            ax.annotate(f'→ {lr[i]:.1e}', xy=(i+1, lr[i]),
                        xytext=(i+1.3, lr[i]*2), fontsize=9, color=C['red'],
                        arrowprops=dict(arrowstyle='->', color=C['red'], lw=1),
                        bbox=dict(boxstyle='round,pad=0.2', fc=C['card2'], ec=C['red'], alpha=0.8))

    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_title('🔧 Learning Rate Schedule', fontsize=16, fontweight='bold', pad=15)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '03_learning_rate.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 03_learning_rate.png")


def chart_overfitting(history):
    """04: Overfitting Gap Analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    epochs = range(1, len(history['accuracy']) + 1)

    acc_gap = np.array(history['accuracy']) - np.array(history['val_accuracy'])
    colors_a = [C['green'] if g < 0.01 else C['orange'] if g < 0.03 else C['red'] for g in acc_gap]
    ax1.bar(epochs, acc_gap, color=colors_a, alpha=0.85, edgecolor='none')
    ax1.axhline(y=0, color=C['txt2'], linestyle='-', alpha=0.5)
    ax1.axhline(y=0.01, color=C['orange'], linestyle='--', alpha=0.4, label='Warning (1%)')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy Gap (Train − Val)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Overfitting Gap', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, facecolor=C['card2'], edgecolor=C['grid'])
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))

    loss_gap = np.array(history['val_loss']) - np.array(history['loss'])
    colors_l = [C['green'] if g < 0.02 else C['orange'] if g < 0.05 else C['red'] for g in loss_gap]
    ax2.bar(epochs, loss_gap, color=colors_l, alpha=0.85, edgecolor='none')
    ax2.axhline(y=0, color=C['txt2'], linestyle='-', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss Gap (Val − Train)', fontsize=12, fontweight='bold')
    ax2.set_title('Loss Overfitting Gap', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.2)

    fig.suptitle('🔍 Overfitting Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '04_overfitting.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 04_overfitting.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: Model Introspection
# ═══════════════════════════════════════════════════════════════════════════════

def chart_layer_params(model):
    """05: Parameter count per layer."""
    fig, ax = plt.subplots(figsize=(12, 8))
    names, counts, colors = [], [], []
    color_map = {'conv': C['blue'], 'bn': C['teal'], 'dense': C['orange'],
                 'output': C['green'], 'dropout': C['txt2'], 'pool': C['purple']}

    for layer in model.layers:
        pc = int(layer.count_params())
        if pc == 0:
            continue
        n = layer.name
        names.append(n)
        counts.append(pc)
        col = C['txt2']
        for key, c in color_map.items():
            if key in n.lower():
                col = c
                break
        colors.append(col)

    y = np.arange(len(names))
    bars = ax.barh(y, counts, color=colors, alpha=0.85, height=0.7, edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, fontfamily='monospace')
    ax.set_xlabel('Parameters', fontsize=13, fontweight='bold')
    ax.set_title('🧮 Layer Parameter Distribution', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, axis='x')
    ax.invert_yaxis()

    for bar, cnt in zip(bars, counts):
        label = f'{cnt:,}' if cnt < 100000 else f'{cnt/1000:.0f}K'
        ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9, color=C['txt2'])

    total = sum(counts)
    ax.text(0.98, 0.02, f'Total: {total:,} params', transform=ax.transAxes, fontsize=12,
            ha='right', va='bottom', fontweight='bold', color=C['cyan'],
            bbox=dict(boxstyle='round,pad=0.4', fc=C['card2'], ec=C['cyan'], alpha=0.9))

    # Legend
    handles = [mpatches.Patch(color=c, label=k.capitalize()) for k, c in color_map.items() if k != 'dropout']
    ax.legend(handles=handles, fontsize=9, loc='lower right', facecolor=C['card2'], edgecolor=C['grid'])

    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '05_layer_params.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 05_layer_params.png")


def chart_architecture(model):
    """06: Architecture flowchart from actual model layers."""
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7)
    ax.axis('off')

    blocks = []
    color_map = {'input': C['cyan'], 'conv': C['blue'], 'batch': C['teal'],
                 'activation': '#455a64', 'max_pool': C['purple'],
                 'global': C['pink'], 'dense': C['orange'], 'dropout': '#455a64',
                 'output': C['green']}

    for layer in model.layers:
        n = layer.name
        out_shape = layer.output_shape if hasattr(layer, 'output_shape') else None
        params = int(layer.count_params())

        # Determine color
        col = C['txt2']
        for key, c_val in color_map.items():
            if key in n.lower():
                col = c_val
                break
        if n == 'output' or (hasattr(layer, 'activation') and 'softmax' in str(getattr(layer, 'activation', ''))):
            col = C['green']

        # Build label
        cfg = layer.get_config()
        if 'conv' in n.lower():
            f = cfg.get('filters', '?')
            k = cfg.get('kernel_size', '?')
            label = f'{n}\n{f} × {k}'
        elif 'dense' in n.lower() or n == 'output':
            u = cfg.get('units', '?')
            label = f'{n}\n{u} units'
        elif 'dropout' in n.lower():
            r = cfg.get('rate', '?')
            label = f'{n}\n{r}'
        elif 'pool' in n.lower():
            label = n
        elif 'bn' in n.lower() or 'batch' in n.lower():
            label = n
        else:
            label = n

        blocks.append({'name': n, 'label': label, 'color': col, 'params': params})

    # Filter to important layers only (skip augmentation sub-layers)
    important = [b for b in blocks if b['params'] > 0 or
                 any(k in b['name'].lower() for k in ['input', 'pool', 'gap', 'global', 'relu', 'activation'])]
    if len(important) > 14:
        # Merge BN+Activation into conv/dense blocks
        merged = []
        for b in important:
            if 'bn' in b['name'].lower() or 'relu' in b['name'].lower() or 'activation' in b['name'].lower():
                continue
            merged.append(b)
        important = merged

    n_blocks = len(important)
    if n_blocks == 0:
        important = blocks[:12]
        n_blocks = len(important)

    spacing = min(1.4, 16.0 / max(n_blocks, 1))
    bw = spacing * 0.75

    for i, b in enumerate(important):
        x = 0.5 + i * spacing
        rect = mpatches.FancyBboxPatch((x, 2.0), bw, 3, boxstyle="round,pad=0.1",
                                        facecolor=b['color'], edgecolor='white', alpha=0.85, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + bw/2, 3.5, b['label'], ha='center', va='center',
                fontsize=max(7, 10 - n_blocks//5), fontweight='bold', color='white', linespacing=1.3)
        if i < n_blocks - 1:
            ax.annotate('', xy=(0.5 + (i+1)*spacing - 0.05, 3.5),
                        xytext=(x + bw + 0.05, 3.5),
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.2))

    total = sum(int(l.count_params()) for l in model.layers)
    trainable = sum(int(w.numpy().size) for w in model.trainable_weights)
    info = f'Total: {total:,}  |  Trainable: {trainable:,}  |  Non-trainable: {total-trainable:,}'
    ax.text(9.0, 0.6, info, ha='center', va='center', fontsize=11, color=C['txt2'],
            bbox=dict(boxstyle='round,pad=0.5', fc=C['card2'], ec=C['grid'], alpha=0.9))
    ax.set_title('🧠 DrawMe CNN — Architecture', fontsize=18, fontweight='bold', pad=20, color=C['txt'])

    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '06_architecture.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 06_architecture.png")


def chart_weight_distributions(model):
    """07: Kernel weight distributions per conv/dense layer."""
    weight_layers = [(l.name, l.get_weights()[0]) for l in model.layers
                     if l.get_weights() and ('conv' in l.name.lower() or 'dense' in l.name.lower())]
    if not weight_layers:
        print("  ⊘ 07_weight_distributions.png (no weight layers)")
        return

    n = len(weight_layers)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, w) in enumerate(weight_layers):
        ax = axes[i]
        flat = w.flatten()
        col = C['blue'] if 'conv' in name else C['orange']
        ax.hist(flat, bins=60, color=col, alpha=0.8, edgecolor='none', density=True)
        ax.axvline(x=0, color=C['red'], linestyle='--', alpha=0.5, linewidth=1)

        mu, sigma = float(np.mean(flat)), float(np.std(flat))
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.text(0.97, 0.95, f'μ={mu:.4f}\nσ={sigma:.4f}\nn={flat.size:,}',
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                fontfamily='monospace', color=C['txt2'],
                bbox=dict(boxstyle='round,pad=0.2', fc=C['card2'], ec=C['grid'], alpha=0.8))
        ax.grid(True, alpha=0.2)
        ax.set_ylabel('Density', fontsize=9)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('⚖️ Weight Distributions per Layer', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '07_weight_distributions.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 07_weight_distributions.png")


def chart_feature_shapes(model):
    """08: Feature map spatial dimensions through the network."""
    fig, ax = plt.subplots(figsize=(14, 6))

    shapes = []
    for layer in model.layers:
        out = layer.output_shape if hasattr(layer, 'output_shape') else None
        if out and isinstance(out, tuple) and len(out) == 4:
            shapes.append((layer.name, out[1], out[2], out[3]))
        elif out and isinstance(out, tuple) and len(out) == 2:
            shapes.append((layer.name, 1, 1, out[1]))

    if not shapes:
        print("  ⊘ 08_feature_shapes.png (no shape data)")
        return

    # Deduplicate consecutive same-spatial-size layers, keep important ones
    filtered = [shapes[0]]
    for s in shapes[1:]:
        if s[1] != filtered[-1][1] or s[2] != filtered[-1][2] or s[3] != filtered[-1][3]:
            filtered.append(s)
    shapes = filtered

    x_pos = np.arange(len(shapes))
    spatial = [s[1] * s[2] for s in shapes]
    channels = [s[3] for s in shapes]
    names = [s[0] for s in shapes]

    ax2 = ax.twinx()

    bars = ax.bar(x_pos - 0.2, spatial, width=0.4, color=C['blue'], alpha=0.8, label='Spatial (H×W)')
    ax2.bar(x_pos + 0.2, channels, width=0.4, color=C['orange'], alpha=0.8, label='Channels/Units')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Spatial Size (H×W)', fontsize=12, fontweight='bold', color=C['blue'])
    ax2.set_ylabel('Channels / Units', fontsize=12, fontweight='bold', color=C['orange'])
    ax.set_title('🔬 Feature Map Dimensions Through Network', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper right', fontsize=10, facecolor=C['card2'], edgecolor=C['grid'])

    # Annotate shapes
    for i, s in enumerate(shapes):
        label = f'{s[1]}×{s[2]}×{s[3]}' if s[1] > 1 else f'{s[3]}'
        ax.text(i, max(spatial)*1.05, label, ha='center', fontsize=7, color=C['txt2'], rotation=0)

    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '08_feature_shapes.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 08_feature_shapes.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: Dataset & Runtime Context
# ═══════════════════════════════════════════════════════════════════════════════

def chart_dataset(metadata):
    """09: Dataset overview."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    ds = metadata.get('dataset', {})
    tr = metadata.get('training', {})
    cats_path = os.path.join(SAVE_DIR, "categories.json")
    categories = load_json("categories.json") or []

    spc = int(ds.get('samples_per_class', 20000))
    n_cats = int(ds.get('categories', len(categories) or 15))
    test_split = float(tr.get('test_split', 0.2))
    total = spc * n_cats
    train_n = int(total * (1 - test_split))
    test_n = total - train_n

    # Donut
    wedges, texts, autotexts = ax1.pie(
        [train_n, test_n], labels=[f'Train\n{train_n:,}', f'Test\n{test_n:,}'],
        explode=(0.03, 0.03), colors=[C['blue'], C['purple']],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 12, 'color': C['txt']}, pctdistance=0.75)
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(13)
    ax1.add_artist(plt.Circle((0, 0), 0.55, fc=C['card']))
    ax1.text(0, 0, f'{total:,}\nTotal', ha='center', va='center',
             fontsize=14, fontweight='bold', color=C['txt'])
    ax1.set_title('Train / Test Split', fontsize=14, fontweight='bold', pad=10)

    # Bars
    if categories:
        y_pos = np.arange(len(categories))
        bars = ax2.barh(y_pos, [spc]*len(categories), color=CAT_COLORS[:len(categories)],
                         alpha=0.85, height=0.7, edgecolor='none')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([c.capitalize() for c in categories], fontsize=11)
        ax2.invert_yaxis()
        for bar in bars:
            ax2.text(bar.get_width() + spc*0.01, bar.get_y() + bar.get_height()/2,
                     f'{int(bar.get_width()):,}', ha='left', va='center', fontsize=9, color=C['txt2'])
    ax2.set_xlabel('Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Samples per Category', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.2, axis='x')

    fig.suptitle('📦 Dataset Overview', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '09_dataset.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 09_dataset.png")


def chart_infrastructure(metadata):
    """10: Training infrastructure info card."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    rt = metadata.get('runtime', {})
    md = metadata.get('model', {})
    tr = metadata.get('training', {})
    ev = metadata.get('evaluation', {})
    tm = metadata.get('timing', {})
    ds = metadata.get('dataset', {})

    sections = [
        ('🖥️  Runtime', [
            ('GPU', f"{'✓ Enabled' if rt.get('gpu_enabled') else '✗ Disabled'} ({rt.get('gpu_count', 0)} device{'s' if rt.get('gpu_count', 0) != 1 else ''})"),
            ('Mixed Precision', '✓ FP16' if rt.get('mixed_precision') else '✗ FP32'),
            ('Seed', str(rt.get('seed', '?'))),
        ]),
        ('🧠  Model', [
            ('Name', md.get('name', '?')),
            ('Conv Filters', ' → '.join(str(f) for f in md.get('conv_filters', []))),
            ('Dense Units', ' → '.join(str(u) for u in md.get('dense_units', []))),
            ('Dropout', str(md.get('dropout', '?'))),
            ('Learning Rate', str(md.get('learning_rate', '?'))),
        ]),
        ('📋  Training', [
            ('Epochs', f"{tr.get('epochs', '?')} (ran {tm.get('epochs_ran', '?')})"),
            ('Batch Size', str(tr.get('batch_size', '?'))),
            ('LR Schedule', f"ReduceLROnPlateau (factor={tr.get('lr_factor')}, patience={tr.get('lr_patience')})"),
            ('Early Stopping', f"patience={tr.get('early_stopping_patience', '?')}"),
        ]),
        ('📊  Results', [
            ('Test Accuracy', f"{ev.get('test_accuracy', 0)*100:.2f}%"),
            ('Test Loss', f"{ev.get('test_loss', 0):.4f}"),
            ('Total Time', f"{tm.get('total_training_sec', 0)/60:.1f} min"),
            ('Avg Epoch', f"{tm.get('avg_epoch_sec', 0):.1f} sec"),
        ]),
    ]

    y = 0.95
    for section_title, items in sections:
        ax.text(0.05, y, section_title, transform=ax.transAxes, fontsize=14,
                fontweight='bold', color=C['cyan'])
        y -= 0.04
        for label, value in items:
            ax.text(0.08, y, f'{label}:', transform=ax.transAxes, fontsize=11,
                    color=C['txt2'], fontfamily='monospace')
            ax.text(0.45, y, value, transform=ax.transAxes, fontsize=11,
                    color=C['txt'], fontweight='bold', fontfamily='monospace')
            y -= 0.035
        y -= 0.02

    rect = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                                    fc=C['card2'], ec=C['grid'], alpha=0.6, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.set_title('⚙️ Training Infrastructure & Configuration', fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '10_infrastructure.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 10_infrastructure.png")


def chart_epoch_timing(metadata):
    """11: Per-epoch wall-clock time and throughput."""
    tm = metadata.get('timing', {})
    durations = tm.get('epoch_durations_sec', [])
    if not durations:
        print("  ⊘ 11_epoch_timing.png (no timing data)")
        return

    ds = metadata.get('dataset', {})
    tr = metadata.get('training', {})
    spc = int(ds.get('samples_per_class', 20000))
    n_cats = int(ds.get('categories', 15))
    test_split = float(tr.get('test_split', 0.2))
    val_split = float(tr.get('validation_split', 0.1))
    train_samples = int(spc * n_cats * (1 - test_split) * (1 - val_split))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    epochs = range(1, len(durations) + 1)

    # Duration bars
    bars = ax1.bar(epochs, durations, color=C['blue'], alpha=0.85, edgecolor='none')
    avg = np.mean(durations)
    ax1.axhline(y=avg, color=C['orange'], linestyle='--', alpha=0.7, label=f'Avg: {avg:.1f}s')
    for bar, d in zip(bars, durations):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(durations)*0.02,
                 f'{d:.0f}s', ha='center', fontsize=10, color=C['txt2'], fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Duration (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Wall-Clock Time per Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, facecolor=C['card2'], edgecolor=C['grid'])
    ax1.grid(True, alpha=0.2)

    # Throughput
    throughput = [train_samples / d for d in durations]
    bars2 = ax2.bar(epochs, throughput, color=C['green'], alpha=0.85, edgecolor='none')
    for bar, t in zip(bars2, throughput):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughput)*0.02,
                 f'{t:,.0f}', ha='center', fontsize=10, color=C['txt2'], fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Samples / Second', fontsize=12, fontweight='bold')
    ax2.set_title('Training Throughput', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.2)

    total = sum(durations)
    fig.suptitle(f'⏱️ Epoch Timing  (Total: {total/60:.1f} min)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '11_epoch_timing.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 11_epoch_timing.png")


def chart_dashboard(history, metadata, model=None):
    """12: Combined summary dashboard."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    epochs = range(1, len(history['accuracy']) + 1)

    # Panel 1: Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['accuracy'], color=C['blue'], label='Train', lw=2, marker='o', ms=5)
    ax1.plot(epochs, history['val_accuracy'], color=C['purple'], label='Val', lw=2, marker='s', ms=5)
    ax1.fill_between(epochs, history['accuracy'], alpha=0.1, color=C['blue'])
    ax1.fill_between(epochs, history['val_accuracy'], alpha=0.1, color=C['purple'])
    ax1.set_title('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=9, facecolor=C['card2'], edgecolor=C['grid'])
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # Panel 2: Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['loss'], color=C['blue'], label='Train', lw=2, marker='o', ms=5)
    ax2.plot(epochs, history['val_loss'], color=C['purple'], label='Val', lw=2, marker='s', ms=5)
    ax2.fill_between(epochs, history['loss'], alpha=0.1, color=C['blue'])
    ax2.fill_between(epochs, history['val_loss'], alpha=0.1, color=C['purple'])
    ax2.set_title('Loss', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(fontsize=9, facecolor=C['card2'], edgecolor=C['grid'])
    ax2.grid(True, alpha=0.2)

    # Panel 3: Parameter breakdown (pie)
    ax3 = fig.add_subplot(gs[0, 2])
    if model:
        layer_types = {}
        for l in model.layers:
            pc = int(l.count_params())
            if pc == 0:
                continue
            t = 'Conv' if 'conv' in l.name else 'BatchNorm' if 'bn' in l.name or 'batch' in l.name else \
                'Dense' if 'dense' in l.name else 'Output' if l.name == 'output' else 'Other'
            layer_types[t] = layer_types.get(t, 0) + pc
        if layer_types:
            type_colors = {'Conv': C['blue'], 'BatchNorm': C['teal'], 'Dense': C['orange'],
                          'Output': C['green'], 'Other': C['txt2']}
            labels = list(layer_types.keys())
            sizes = list(layer_types.values())
            colors = [type_colors.get(l, C['txt2']) for l in labels]
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                                startangle=90, textprops={'fontsize': 10, 'color': C['txt']})
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            total = sum(sizes)
            ax3.set_title(f'Parameters ({total:,})', fontsize=13, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No params', transform=ax3.transAxes, ha='center')
    else:
        ax3.text(0.5, 0.5, 'Model not loaded', transform=ax3.transAxes, ha='center', fontsize=12, color=C['txt2'])
        ax3.set_title('Parameters', fontsize=13, fontweight='bold')
    ax3.axis('equal')

    # Panel 4: Key Metrics
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    ev = metadata.get('evaluation', {})
    metrics = [
        ('Train Accuracy', f'{history["accuracy"][-1]*100:.2f}%', C['blue']),
        ('Val Accuracy', f'{history["val_accuracy"][-1]*100:.2f}%', C['purple']),
        ('Test Accuracy', f'{ev.get("test_accuracy", 0)*100:.2f}%', C['green']),
        ('Train Loss', f'{history["loss"][-1]:.4f}', C['blue']),
        ('Val Loss', f'{history["val_loss"][-1]:.4f}', C['purple']),
        ('Test Loss', f'{ev.get("test_loss", 0):.4f}', C['green']),
        ('Total Epochs', f'{len(history["accuracy"])}', C['cyan']),
        ('Categories', f'{metadata.get("dataset", {}).get("categories", "?")}', C['orange']),
    ]
    for i, (label, value, color) in enumerate(metrics):
        y = 0.92 - i * 0.11
        ax4.text(0.05, y, label, transform=ax4.transAxes, fontsize=11, color=C['txt2'], fontfamily='monospace')
        ax4.text(0.95, y, value, transform=ax4.transAxes, fontsize=12, color=color,
                 fontweight='bold', fontfamily='monospace', ha='right')
    rect = mpatches.FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.02",
                                    fc=C['card2'], ec=C['grid'], alpha=0.6, transform=ax4.transAxes)
    ax4.add_patch(rect)
    ax4.set_title('Key Metrics', fontsize=13, fontweight='bold')

    # Panel 5: Epoch timing
    ax5 = fig.add_subplot(gs[1, 1])
    tm = metadata.get('timing', {})
    durations = tm.get('epoch_durations_sec', [])
    if durations:
        ep = range(1, len(durations)+1)
        ax5.bar(ep, [d/60 for d in durations], color=C['cyan'], alpha=0.85, edgecolor='none')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Minutes')
        ax5.set_title('Epoch Duration', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.2)
    else:
        ax5.text(0.5, 0.5, 'No timing data', transform=ax5.transAxes, ha='center', color=C['txt2'])
        ax5.set_title('Epoch Duration', fontsize=13, fontweight='bold')

    # Panel 6: Accuracy per epoch change
    ax6 = fig.add_subplot(gs[1, 2])
    val_acc = history['val_accuracy']
    improvements = [0] + [val_acc[i] - val_acc[i-1] for i in range(1, len(val_acc))]
    colors_imp = [C['green'] if imp > 0 else C['red'] for imp in improvements]
    ax6.bar(epochs, improvements, color=colors_imp, alpha=0.85, edgecolor='none')
    ax6.axhline(y=0, color=C['txt2'], linestyle='-', alpha=0.5)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Δ Val Accuracy')
    ax6.set_title('Per-Epoch Change', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.2, axis='y')
    ax6.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))

    fig.suptitle('🎨 DrawMe — Training Summary Dashboard', fontsize=20, fontweight='bold', y=1.01, color=C['txt'])
    fig.savefig(os.path.join(CHARTS_DIR, '12_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 12_dashboard.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  🎨  DrawMe — Chart Generation (v2)")
    print("=" * 60)

    os.makedirs(CHARTS_DIR, exist_ok=True)
    print(f"\n📂 Output: {CHARTS_DIR}\n")

    # Load data sources
    history = load_json("training_history.json")
    if not history:
        print("✗ No training_history.json found. Train the model first.")
        sys.exit(1)
    print(f"📊 Training history: {len(history.get('accuracy', []))} epochs")

    metadata = load_json("training_metadata.json") or {}
    print(f"📋 Metadata: {'loaded' if metadata else 'not found'}")

    print("🧠 Loading model for introspection...")
    model = load_model_safe()
    print(f"   Model: {'loaded ✓' if model else 'skipped (model charts will be omitted)'}\n")

    print("🖼️  Generating charts...\n")

    # Tier 1: Training Dynamics
    chart_accuracy(history)
    chart_loss(history)
    chart_lr(history)
    chart_overfitting(history)

    # Tier 2: Model Introspection
    if model:
        chart_layer_params(model)
        chart_architecture(model)
        chart_weight_distributions(model)
        chart_feature_shapes(model)
    else:
        print("  ⊘ Skipping model introspection charts (05-08)")

    # Tier 3: Dataset & Runtime
    chart_dataset(metadata)
    chart_infrastructure(metadata)
    chart_epoch_timing(metadata)

    # Dashboard
    chart_dashboard(history, metadata, model)

    print(f"\n✅ All charts saved to: {CHARTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
