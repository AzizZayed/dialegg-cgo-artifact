import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

image_conversion = pd.read_csv("bench/image_conversion/image_conversion.csv")
_2mm = pd.read_csv("bench/2mm/2mm.csv")
_3mm = pd.read_csv("bench/3mm/3mm.csv")
vector_norm = pd.read_csv("bench/vector_norm/vector_norm.csv")
polynomial = pd.read_csv("bench/polynomial/polynomial.csv")

df = pd.concat([image_conversion, _2mm, _3mm, vector_norm, polynomial], ignore_index=True)

# Renanem names to nice names
df['name'] = df['name'].replace({
    "image_conversion": "Image\nConversion",
    "2mm": "2MM",
    "3mm": "3MM",
    "vector_norm": "Vector\nNorm",
    "polynomial": "Polynomial"
})

df['opt'] = df['opt'].replace({
    "noopt": "No Optimization",
    "canon": "Canonicalization",
    "eqsat": "DialEgg",
    "eqsat+canon": "DialEgg + Canonicalization",
    "canon+eqsat": "Canonicalization + DialEgg",
    "cpp": "MLIR Hand-Writen Pass"
})

# Filter the dataframe for the best optimization level
df_opt = df[['name', 'opt', "-O3"]]

# Pivot the dataframe to have optimizations as columns
df_pivot = df_opt.pivot(index='name', columns='opt', values="-O3")
df_pivot = df_pivot.drop(columns=['DialEgg + Canonicalization'], errors='ignore')

# Calculate speedup relative to 'noopt'
df_speedup = (1.0 / df_pivot).mul(df_pivot['No Optimization'], axis=0) 
df_speedup = df_speedup.drop(columns=['No Optimization'], errors='ignore') 

# order is Image Conversion, vector norm, polynomial, 2MM, 3MM
df_speedup = df_speedup.reindex(['Image\nConversion', 'Vector\nNorm', 'Polynomial', '2MM', '3MM'])
df_speedup

for i in range(5):
    # Create the speedup plot
    fig, ax = plt.subplots(figsize=(10, 7))

    bar_width = 0.20
    space = 0.050

    index = np.array([0, 1, 2, 3, 4.25])

    for i, col in enumerate(df_speedup.columns):
        bars = ax.bar(index + i * (bar_width + space), df_speedup[col] - 1, bar_width, 
                    label=col, bottom=1, edgecolor='black', linewidth=0.5)  # Bars start from 1

        # Add text labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):  # Skip NaN values
                ax.text(bar.get_x() + bar.get_width() / 2, max(0, height) + 1, f'{height + 1:.2f}', ha='center', va='bottom', fontsize=12)

    ax.grid(True, which='both', axis='y', zorder=0, linestyle=':', linewidth=0.5, alpha=0.7)

    plt.rc('text', usetex=True) # tex support
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 20})

    # Customize the plot
    plt.xlabel(r'\textbf{Benchmark}', labelpad=10, fontdict={'family': 'Libertine'})
    plt.ylabel(r'\textbf{Speedup w.r.t. no optimization (log x)}', fontdict={'family': 'Libertine', 'weight': 'bold'})
    ax.axhline(y=1, color='black', linestyle='-', label='Baseline (No Optimization)')

    # Rotate x-axis labels for better readabilit
    ax.set_xticks(index + (bar_width / 2 + space + bar_width / 2) * np.array([1, 1, 1, 1.5, 1.5]))
    ax.set_xticklabels(df_speedup.index, rotation=0, ha='center')

    plt.yscale('log')
    plt.ylim(0.5, 15)
    plt.yticks([0.5, 1, 5, 10])

    ax.legend(title='Optimization Passes', loc='upper left', fontsize=20)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # save the plot to a file
    plt.savefig("bench/speedup.pdf")

    # plt.show()