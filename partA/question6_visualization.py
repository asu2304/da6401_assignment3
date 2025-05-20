import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 1) Example data
data = {
    'Input': [
        'vidhansabhaen',
        'bhraantiyan',
        'mass',
        'beast',
        'sikud'
    ],
    'Reference': [
        'विधानसभाएं',
        'भ्रांतियां',
        'मास',
        'बीस्ट',
        'सिकुड़'
    ],
    'Vanilla Prediction': [
        'विधांमाान',
        'भ्रततियां',
        'मैस',
        'बेस्ट',
        'सिकुद'
    ],
    'Attention Prediction': [
        'विधानसभाएं',
        'भ्रांतियां',
        'मास',
        'बीस्ट',
        'सिकुड़'
    ]
}
df = pd.DataFrame(data)

# 2) Compute per-cell colors
cell_colors = []
for _, row in df.iterrows():
    vanilla_ok = row['Vanilla Prediction'] == row['Reference']
    attn_ok    = row['Attention Prediction'] == row['Reference']
    # white for Input & Reference
    cols = [
        'white', 'white',
        '#FFCCCC' if not vanilla_ok else '#CCFFCC',
        '#CCFFCC' if attn_ok    else '#FFCCCC'
    ]
    cell_colors.append(cols)

# 3) Load a Devanagari font for correct rendering
dev_font = None
for path in (
    '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
    'C:/Windows/Fonts/Mangal.ttf'
):
    if os.path.exists(path):
        dev_font = fm.FontProperties(fname=path)
        break

# 4) Plot the table
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')

tbl = ax.table(
    cellText    = df.values,
    colLabels   = df.columns,
    cellColours = cell_colors,
    cellLoc     = 'center',
    loc         = 'center'
)

# 5) Style header row and cells
for (r, c), cell in tbl.get_celld().items():
    # header
    if r == 0:
        cell.set_facecolor('#333333')
        cell.get_text().set_color('white')
        cell.get_text().set_fontweight('bold')
    # Devanagari columns (Reference & preds)
    if c in (1, 2, 3) and dev_font:
        cell.get_text().set_fontproperties(dev_font)

# 6) Final adjustments
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 2)

plt.title('Input / Reference / Vanilla vs Attention Predictions',
          pad=12, fontsize=14)
plt.tight_layout()

# 7) Save figure
out = 'comparison_grid.png'
plt.savefig(out, dpi=300)
print(f"Saved {out}")
