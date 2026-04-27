"""
DrawMe - Training Metrics Visualization Script
Generates comprehensive charts and graphs showing model performance,
training dynamics, and architecture details.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# ─── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")

CATEGORIES = [
    "cloud", "sun", "tree", "car", "fish",
    "cat", "dog", "house", "star", "flower",
    "bird", "bicycle", "guitar", "moon", "hat"
]

# ─── Styling ───────────────────────────────────────────────────────────────────

# Premium color palette
COLORS = {
    'bg_dark': '#0f0f1a',
    'bg_card': '#1a1a2e',
    'bg_card_alt': '#16213e',
    'accent_blue': '#4fc3f7',
    'accent_purple': '#ab47bc',
    'accent_green': '#66bb6a',
    'accent_orange': '#ffa726',
    'accent_red': '#ef5350',
    'accent_cyan': '#26c6da',
    'accent_pink': '#ec407a',
    'accent_yellow': '#ffee58',
    'text_primary': '#e0e0e0',
    'text_secondary': '#9e9e9e',
    'grid': '#2a2a4a',
    'train_line': '#4fc3f7',
    'val_line': '#ab47bc',
}

CATEGORY_COLORS = [
    '#4fc3f7', '#ab47bc', '#66bb6a', '#ffa726', '#ef5350',
    '#26c6da', '#ec407a', '#ffee58', '#7e57c2', '#ff7043',
    '#42a5f5', '#9ccc65', '#5c6bc0', '#29b6f6', '#8d6e63'
]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'],
    'font.size': 11,
    'axes.facecolor': COLORS['bg_card'],
    'axes.edgecolor': COLORS['grid'],
    'axes.labelcolor': COLORS['text_primary'],
    'text.color': COLORS['text_primary'],
    'xtick.color': COLORS['text_secondary'],
    'ytick.color': COLORS['text_secondary'],
    'figure.facecolor': COLORS['bg_dark'],
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'lines.antialiased': True,
})


def load_history():
    """Load training history from JSON."""
    history_path = os.path.join(SAVE_DIR, "training_history.json")
    with open(history_path, "r") as f:
        return json.load(f)


def create_accuracy_chart(history):
    """Chart 1: Training & Validation Accuracy over Epochs."""
    fig, ax = plt.subplots(figsize=(12, 7))
    epochs = range(1, len(history['accuracy']) + 1)

    # Fill between
    ax.fill_between(epochs, history['accuracy'], alpha=0.15, color=COLORS['train_line'])
    ax.fill_between(epochs, history['val_accuracy'], alpha=0.15, color=COLORS['val_line'])

    # Lines
    ax.plot(epochs, history['accuracy'], color=COLORS['train_line'], 
            label='Training Accuracy', linewidth=2.5, marker='o', markersize=4, zorder=5)
    ax.plot(epochs, history['val_accuracy'], color=COLORS['val_line'], 
            label='Validation Accuracy', linewidth=2.5, marker='s', markersize=4, zorder=5)

    # Best epoch marker
    best_val_idx = np.argmax(history['val_accuracy'])
    best_val = history['val_accuracy'][best_val_idx]
    ax.annotate(f'Best: {best_val:.4f}', 
                xy=(best_val_idx + 1, best_val),
                xytext=(best_val_idx - 4, best_val - 0.04),
                fontsize=11, fontweight='bold', color=COLORS['accent_green'],
                arrowprops=dict(arrowstyle='->', color=COLORS['accent_green'], lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg_card_alt'], 
                         edgecolor=COLORS['accent_green'], alpha=0.9))

    # Final accuracy annotation
    final_train = history['accuracy'][-1]
    final_val = history['val_accuracy'][-1]
    ax.axhline(y=final_val, color=COLORS['val_line'], linestyle='--', alpha=0.3)

    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('📈 Model Accuracy Over Training', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9, 
              facecolor=COLORS['bg_card_alt'], edgecolor=COLORS['grid'])
    ax.set_ylim(0.5, 0.96)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # Stats box
    stats_text = f'Final Train: {final_train*100:.2f}%\nFinal Val: {final_val*100:.2f}%\nBest Val: {best_val*100:.2f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg_card_alt'], 
                     edgecolor=COLORS['accent_cyan'], alpha=0.9))

    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '01_accuracy.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 01_accuracy.png")


def create_loss_chart(history):
    """Chart 2: Training & Validation Loss over Epochs."""
    fig, ax = plt.subplots(figsize=(12, 7))
    epochs = range(1, len(history['loss']) + 1)

    ax.fill_between(epochs, history['loss'], alpha=0.15, color=COLORS['train_line'])
    ax.fill_between(epochs, history['val_loss'], alpha=0.15, color=COLORS['val_line'])

    ax.plot(epochs, history['loss'], color=COLORS['train_line'], 
            label='Training Loss', linewidth=2.5, marker='o', markersize=4, zorder=5)
    ax.plot(epochs, history['val_loss'], color=COLORS['val_line'], 
            label='Validation Loss', linewidth=2.5, marker='s', markersize=4, zorder=5)

    # Best validation loss marker
    best_val_loss_idx = np.argmin(history['val_loss'])
    best_val_loss = history['val_loss'][best_val_loss_idx]
    ax.annotate(f'Best: {best_val_loss:.4f}', 
                xy=(best_val_loss_idx + 1, best_val_loss),
                xytext=(best_val_loss_idx - 5, best_val_loss + 0.15),
                fontsize=11, fontweight='bold', color=COLORS['accent_green'],
                arrowprops=dict(arrowstyle='->', color=COLORS['accent_green'], lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg_card_alt'], 
                         edgecolor=COLORS['accent_green'], alpha=0.9))

    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss (Cross-Entropy)', fontsize=13, fontweight='bold')
    ax.set_title('📉 Model Loss Over Training', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9,
              facecolor=COLORS['bg_card_alt'], edgecolor=COLORS['grid'])
    ax.grid(True, alpha=0.2)

    # Overfitting gap annotation
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    gap = final_val_loss - final_train_loss
    stats_text = f'Final Train: {final_train_loss:.4f}\nFinal Val: {final_val_loss:.4f}\nGap: {gap:.4f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg_card_alt'], 
                     edgecolor=COLORS['accent_orange'], alpha=0.9))

    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '02_loss.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 02_loss.png")


def create_learning_rate_chart(history):
    """Chart 3: Learning Rate Schedule."""
    fig, ax = plt.subplots(figsize=(12, 5))
    epochs = range(1, len(history['learning_rate']) + 1)

    ax.step(list(epochs), history['learning_rate'], color=COLORS['accent_cyan'], 
            linewidth=2.5, where='mid', zorder=5)
    ax.fill_between(epochs, history['learning_rate'], alpha=0.2, 
                    color=COLORS['accent_cyan'], step='mid')

    # Annotate LR drops
    lr_vals = history['learning_rate']
    for i in range(1, len(lr_vals)):
        if lr_vals[i] < lr_vals[i-1]:
            ax.axvline(x=i+1, color=COLORS['accent_red'], linestyle='--', alpha=0.5)
            ax.annotate(f'LR halved\n→ {lr_vals[i]:.1e}',
                        xy=(i+1, lr_vals[i]), xytext=(i+1.5, lr_vals[i]*3),
                        fontsize=9, color=COLORS['accent_red'],
                        arrowprops=dict(arrowstyle='->', color=COLORS['accent_red'], lw=1),
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['bg_card_alt'], 
                                 edgecolor=COLORS['accent_red'], alpha=0.8))

    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_title('🔧 Learning Rate Schedule (ReduceLROnPlateau)', fontsize=16, fontweight='bold', pad=15)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '03_learning_rate.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 03_learning_rate.png")


def create_overfitting_gap_chart(history):
    """Chart 4: Overfitting Analysis — Train vs Val Gap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    epochs = range(1, len(history['accuracy']) + 1)

    # Accuracy gap
    acc_gap = np.array(history['accuracy']) - np.array(history['val_accuracy'])
    ax1.bar(epochs, acc_gap, color=[COLORS['accent_green'] if g < 0.01 else 
            COLORS['accent_orange'] if g < 0.02 else COLORS['accent_red'] for g in acc_gap],
            alpha=0.8, edgecolor='none')
    ax1.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', alpha=0.5)
    ax1.axhline(y=0.01, color=COLORS['accent_orange'], linestyle='--', alpha=0.4, label='Warning threshold')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy Gap (Train - Val)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Overfitting Gap', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, facecolor=COLORS['bg_card_alt'], edgecolor=COLORS['grid'])
    ax1.grid(True, alpha=0.2)

    # Loss gap
    loss_gap = np.array(history['val_loss']) - np.array(history['loss'])
    ax2.bar(epochs, loss_gap, color=[COLORS['accent_green'] if g < 0.02 else 
            COLORS['accent_orange'] if g < 0.05 else COLORS['accent_red'] for g in loss_gap],
            alpha=0.8, edgecolor='none')
    ax2.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss Gap (Val - Train)', fontsize=12, fontweight='bold')
    ax2.set_title('Loss Overfitting Gap', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.2)

    fig.suptitle('🔍 Overfitting Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '04_overfitting_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 04_overfitting_analysis.png")


def create_epoch_improvement_chart(history):
    """Chart 5: Per-epoch improvement in validation accuracy."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    val_acc = history['val_accuracy']
    improvements = [0] + [val_acc[i] - val_acc[i-1] for i in range(1, len(val_acc))]
    epochs = range(1, len(val_acc) + 1)
    
    colors = [COLORS['accent_green'] if imp > 0 else COLORS['accent_red'] for imp in improvements]
    bars = ax.bar(epochs, improvements, color=colors, alpha=0.8, edgecolor='none', width=0.7)
    
    ax.axhline(y=0, color=COLORS['text_secondary'], linestyle='-', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Δ Validation Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('📊 Per-Epoch Validation Accuracy Change', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, axis='y')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
    
    # Legend
    pos_patch = mpatches.Patch(color=COLORS['accent_green'], alpha=0.8, label='Improvement')
    neg_patch = mpatches.Patch(color=COLORS['accent_red'], alpha=0.8, label='Regression')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=10, 
              facecolor=COLORS['bg_card_alt'], edgecolor=COLORS['grid'])
    
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '05_epoch_improvement.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 05_epoch_improvement.png")


def create_dataset_overview(history):
    """Chart 6: Dataset composition and sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    samples_per_class = 20000
    total_samples = samples_per_class * len(CATEGORIES)
    train_samples = int(total_samples * 0.8)
    test_samples = total_samples - train_samples
    
    # Donut chart for train/test split
    sizes = [train_samples, test_samples]
    labels = [f'Train\n{train_samples:,}', f'Test\n{test_samples:,}']
    explode = (0.03, 0.03)
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, explode=explode,
                                        colors=[COLORS['accent_blue'], COLORS['accent_purple']],
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 12, 'color': COLORS['text_primary']},
                                        pctdistance=0.75)
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
    
    # Inner circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.55, fc=COLORS['bg_card'])
    ax1.add_artist(centre_circle)
    ax1.text(0, 0, f'{total_samples:,}\nTotal', ha='center', va='center',
             fontsize=14, fontweight='bold', color=COLORS['text_primary'])
    ax1.set_title('Train / Test Split', fontsize=14, fontweight='bold', pad=10)
    
    # Bar chart for samples per category
    y_pos = np.arange(len(CATEGORIES))
    bars = ax2.barh(y_pos, [samples_per_class] * len(CATEGORIES), 
                     color=CATEGORY_COLORS, alpha=0.85, edgecolor='none', height=0.7)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([c.capitalize() for c in CATEGORIES], fontsize=11)
    ax2.set_xlabel('Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Samples per Category', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.2, axis='x')
    ax2.invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 200, bar.get_y() + bar.get_height()/2, f'{int(width):,}',
                ha='left', va='center', fontsize=9, color=COLORS['text_secondary'])
    
    fig.suptitle('📦 Dataset Overview', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '06_dataset_overview.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 06_dataset_overview.png")


def create_model_architecture_chart():
    """Chart 7: Visual representation of the CNN architecture."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Architecture layers
    layers_info = [
        {'name': 'Input\n28×28×1', 'color': COLORS['accent_cyan'], 'x': 0.5, 'w': 1.2},
        {'name': 'Data\nAugment', 'color': '#455a64', 'x': 2.0, 'w': 1.2},
        {'name': 'Conv2D\n32 filters\n3×3', 'color': COLORS['accent_blue'], 'x': 3.5, 'w': 1.3},
        {'name': 'BN+ReLU\n+MaxPool', 'color': '#37474f', 'x': 5.0, 'w': 1.3},
        {'name': 'Conv2D\n64 filters\n3×3', 'color': COLORS['accent_purple'], 'x': 6.5, 'w': 1.3},
        {'name': 'BN+ReLU\n+MaxPool', 'color': '#37474f', 'x': 8.0, 'w': 1.3},
        {'name': 'Conv2D\n128 filters\n3×3', 'color': COLORS['accent_pink'], 'x': 9.5, 'w': 1.3},
        {'name': 'BN+ReLU\n+GAP', 'color': '#37474f', 'x': 11.0, 'w': 1.3},
        {'name': 'Dense\n256\n+Dropout', 'color': COLORS['accent_orange'], 'x': 12.5, 'w': 1.3},
        {'name': 'Output\n15 classes\nSoftmax', 'color': COLORS['accent_green'], 'x': 14.0, 'w': 1.3},
    ]
    
    for i, layer in enumerate(layers_info):
        rect = mpatches.FancyBboxPatch(
            (layer['x'], 2.5), layer['w'], 3,
            boxstyle="round,pad=0.1",
            facecolor=layer['color'], edgecolor='white',
            alpha=0.85, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(layer['x'] + layer['w']/2, 4.0, layer['name'],
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', linespacing=1.4)
        
        # Arrow to next layer
        if i < len(layers_info) - 1:
            next_layer = layers_info[i + 1]
            ax.annotate('', xy=(next_layer['x'] - 0.05, 4.0),
                        xytext=(layer['x'] + layer['w'] + 0.05, 4.0),
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    
    ax.set_title('🧠 DrawMe CNN v2 — Model Architecture', fontsize=18, fontweight='bold', 
                 pad=20, color=COLORS['text_primary'])
    
    # Parameters info
    param_text = ('Total Parameters: ~160K\n'
                  'Optimizer: Adam (lr=0.001)\n'
                  'Loss: Sparse Categorical Cross-Entropy\n'
                  'Regularization: Dropout (0.4) + BatchNorm + Data Augmentation')
    ax.text(8.0, 0.8, param_text, ha='center', va='center', fontsize=10,
            color=COLORS['text_secondary'], linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg_card_alt'], 
                     edgecolor=COLORS['grid'], alpha=0.9))
    
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '07_architecture.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 07_architecture.png")


def create_training_summary_dashboard(history):
    """Chart 8: Combined dashboard with all key metrics."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # ── Panel 1: Accuracy ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['accuracy'], color=COLORS['train_line'], label='Train', linewidth=2)
    ax1.plot(epochs, history['val_accuracy'], color=COLORS['val_line'], label='Val', linewidth=2)
    ax1.fill_between(epochs, history['accuracy'], alpha=0.1, color=COLORS['train_line'])
    ax1.fill_between(epochs, history['val_accuracy'], alpha=0.1, color=COLORS['val_line'])
    ax1.set_title('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=9, facecolor=COLORS['bg_card_alt'], edgecolor=COLORS['grid'])
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    
    # ── Panel 2: Loss ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['loss'], color=COLORS['train_line'], label='Train', linewidth=2)
    ax2.plot(epochs, history['val_loss'], color=COLORS['val_line'], label='Val', linewidth=2)
    ax2.fill_between(epochs, history['loss'], alpha=0.1, color=COLORS['train_line'])
    ax2.fill_between(epochs, history['val_loss'], alpha=0.1, color=COLORS['val_line'])
    ax2.set_title('Loss', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=9, facecolor=COLORS['bg_card_alt'], edgecolor=COLORS['grid'])
    ax2.grid(True, alpha=0.2)
    
    # ── Panel 3: Learning Rate ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.step(list(epochs), history['learning_rate'], color=COLORS['accent_cyan'], linewidth=2, where='mid')
    ax3.fill_between(epochs, history['learning_rate'], alpha=0.15, color=COLORS['accent_cyan'], step='mid')
    ax3.set_title('Learning Rate', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('LR')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.2)
    
    # ── Panel 4: Accuracy Convergence (last 10 epochs) ──
    ax4 = fig.add_subplot(gs[1, 0])
    last_n = 10
    last_epochs = list(epochs)[-last_n:]
    last_train = history['accuracy'][-last_n:]
    last_val = history['val_accuracy'][-last_n:]
    ax4.plot(last_epochs, last_train, color=COLORS['train_line'], label='Train', linewidth=2.5, marker='o', markersize=5)
    ax4.plot(last_epochs, last_val, color=COLORS['val_line'], label='Val', linewidth=2.5, marker='s', markersize=5)
    ax4.set_title(f'Convergence (Last {last_n} Epochs)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend(fontsize=9, facecolor=COLORS['bg_card_alt'], edgecolor=COLORS['grid'])
    ax4.grid(True, alpha=0.2)
    ax4.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    
    # ── Panel 5: Key Metrics Summary ──
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    metrics = [
        ('Final Train Accuracy', f'{history["accuracy"][-1]*100:.2f}%', COLORS['accent_blue']),
        ('Final Val Accuracy', f'{history["val_accuracy"][-1]*100:.2f}%', COLORS['accent_purple']),
        ('Best Val Accuracy', f'{max(history["val_accuracy"])*100:.2f}%', COLORS['accent_green']),
        ('Final Train Loss', f'{history["loss"][-1]:.4f}', COLORS['accent_blue']),
        ('Final Val Loss', f'{history["val_loss"][-1]:.4f}', COLORS['accent_purple']),
        ('Best Val Loss', f'{min(history["val_loss"]):.4f}', COLORS['accent_green']),
        ('Total Epochs', f'{len(history["accuracy"])}', COLORS['accent_cyan']),
        ('Categories', f'{len(CATEGORIES)}', COLORS['accent_orange']),
        ('Samples/Class', '20,000', COLORS['accent_yellow']),
    ]
    
    y_start = 0.92
    for i, (label, value, color) in enumerate(metrics):
        y = y_start - i * 0.1
        ax5.text(0.05, y, label, transform=ax5.transAxes, fontsize=11,
                color=COLORS['text_secondary'], fontfamily='monospace')
        ax5.text(0.95, y, value, transform=ax5.transAxes, fontsize=12,
                color=color, fontweight='bold', fontfamily='monospace', ha='right')
    
    ax5.set_title('Key Metrics', fontsize=13, fontweight='bold')
    rect = mpatches.FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
                                     boxstyle="round,pad=0.02",
                                     facecolor=COLORS['bg_card_alt'],
                                     edgecolor=COLORS['grid'],
                                     alpha=0.6, transform=ax5.transAxes)
    ax5.add_patch(rect)
    
    # ── Panel 6: Convergence Speed ──
    ax6 = fig.add_subplot(gs[1, 2])
    val_acc = history['val_accuracy']
    thresholds = [0.80, 0.85, 0.90, 0.92, 0.925]
    reached_epochs = []
    threshold_labels = []
    for t in thresholds:
        epoch_reached = None
        for i, acc in enumerate(val_acc):
            if acc >= t:
                epoch_reached = i + 1
                break
        if epoch_reached is not None:
            reached_epochs.append(epoch_reached)
            threshold_labels.append(f'{t*100:.0f}%')
    
    bars = ax6.barh(range(len(reached_epochs)), reached_epochs, 
                    color=[CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i in range(len(reached_epochs))],
                    alpha=0.85, height=0.6, edgecolor='none')
    ax6.set_yticks(range(len(reached_epochs)))
    ax6.set_yticklabels(threshold_labels, fontsize=11)
    ax6.set_xlabel('Epoch Reached', fontsize=12, fontweight='bold')
    ax6.set_title('Convergence Milestones', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.2, axis='x')
    ax6.invert_yaxis()
    
    for bar, ep in zip(bars, reached_epochs):
        ax6.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'Epoch {ep}', va='center', fontsize=10, color=COLORS['text_secondary'])
    
    fig.suptitle('🎨 DrawMe — Training Summary Dashboard', 
                 fontsize=20, fontweight='bold', y=1.01, color=COLORS['text_primary'])
    
    fig.savefig(os.path.join(CHARTS_DIR, '08_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 08_dashboard.png")


def create_training_phases_chart(history):
    """Chart 9: Training phases based on learning rate changes."""
    fig, ax = plt.subplots(figsize=(14, 7))
    epochs = range(1, len(history['accuracy']) + 1)
    
    lr_vals = history['learning_rate']
    
    # Identify phases
    phases = []
    phase_start = 0
    current_lr = lr_vals[0]
    for i in range(len(lr_vals)):
        if lr_vals[i] != current_lr or i == len(lr_vals) - 1:
            end = i if lr_vals[i] != current_lr else i + 1
            phases.append((phase_start, end, current_lr))
            phase_start = i
            current_lr = lr_vals[i]
    if phase_start < len(lr_vals) - 1:
        phases.append((phase_start, len(lr_vals), current_lr))
    
    phase_colors = [COLORS['accent_blue'], COLORS['accent_purple'], 
                    COLORS['accent_green'], COLORS['accent_orange'], COLORS['accent_cyan']]
    
    for idx, (start, end, lr) in enumerate(phases):
        color = phase_colors[idx % len(phase_colors)]
        ax.axvspan(start + 1, end, alpha=0.15, color=color)
        mid = (start + end) / 2 + 0.5
        ax.text(mid, 0.97, f'LR={lr:.1e}', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=9, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['bg_card_alt'], 
                         edgecolor=color, alpha=0.8))
    
    ax.plot(epochs, history['val_accuracy'], color=COLORS['val_line'], 
            linewidth=2.5, marker='s', markersize=4, label='Val Accuracy', zorder=5)
    ax.plot(epochs, history['accuracy'], color=COLORS['train_line'], 
            linewidth=2.5, marker='o', markersize=4, label='Train Accuracy', zorder=5)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('🎯 Training Phases & Accuracy Progression', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11, facecolor=COLORS['bg_card_alt'], edgecolor=COLORS['grid'])
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '09_training_phases.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 09_training_phases.png")


def create_convergence_radar_chart(history):
    """Chart 10: Radar chart summarizing model quality metrics."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Metrics (normalized to 0-1 scale)
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    best_val_acc = max(history['val_accuracy'])
    
    # Generalization = 1 - (train_acc - val_acc) / train_acc
    generalization = 1 - abs(final_train_acc - final_val_acc) / final_train_acc
    
    # Convergence stability = 1 - std(last 5 val_acc) * 100
    stability = max(0, 1 - np.std(history['val_accuracy'][-5:]) * 100)
    
    # Training efficiency = best_val_acc / epochs_to_reach_90%
    epochs_to_90 = next((i+1 for i, acc in enumerate(history['val_accuracy']) if acc >= 0.9), len(history['accuracy']))
    efficiency = min(1.0, 10.0 / epochs_to_90)  # faster convergence = higher score
    
    categories_r = ['Train Acc', 'Val Acc', 'Best Val Acc', 
                     'Generalization', 'Stability', 'Efficiency']
    values = [final_train_acc, final_val_acc, best_val_acc, 
              generalization, stability, efficiency]
    
    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, len(categories_r), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2.5, color=COLORS['accent_cyan'], markersize=8)
    ax.fill(angles, values, alpha=0.2, color=COLORS['accent_cyan'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_r, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color=COLORS['text_secondary'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    
    ax.set_title('⭐ Model Quality Radar', fontsize=16, fontweight='bold', pad=25)
    
    # Add value annotations
    for angle, value, label in zip(angles[:-1], values[:-1], categories_r):
        ax.annotate(f'{value:.3f}', xy=(angle, value), xytext=(angle, value + 0.08),
                    ha='center', fontsize=9, color=COLORS['accent_yellow'], fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '10_quality_radar.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ 10_quality_radar.png")


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🎨  DrawMe — Chart Generation")
    print("=" * 60)
    
    os.makedirs(CHARTS_DIR, exist_ok=True)
    print(f"\n📂 Output directory: {CHARTS_DIR}\n")
    
    print("📊 Loading training history...")
    history = load_history()
    print(f"   Found {len(history['accuracy'])} epochs of data\n")
    
    print("🖼️  Generating charts...\n")
    
    create_accuracy_chart(history)
    create_loss_chart(history)
    create_learning_rate_chart(history)
    create_overfitting_gap_chart(history)
    create_epoch_improvement_chart(history)
    create_dataset_overview(history)
    create_model_architecture_chart()
    create_training_summary_dashboard(history)
    create_training_phases_chart(history)
    create_convergence_radar_chart(history)
    
    print(f"\n✅ All charts saved to: {CHARTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
