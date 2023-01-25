"""Combine and plot RF

Combine all RF into a single file and plot the output

Usage:
    enrich_odml_epochs.py --csv-list=LIST \
                          --mapping=FILE \
                          --plt=INT \
                          --monkey=STR \
                          --out=FILE

Options:
  -h --help        Show this screen and terminate script.
  --csv-list=LIST  Trial information .csv file.
  --mapping=FILE   Path to .csv array mapping.
  --plt=INT        Output plot
  --monkey=STR     String identifier of monkey
  --out=FILE       Output file path.

"""
from docopt import docopt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

xing_L_colors = ['#bf0001ff', '#7d3e22ff',
                 '#9a5e45ff', '#dcdb04ff',
                 '#81d404ff', '#166e06ff',
                 '#fe7300ff', '#8b4412ff',
                 '#07abc0ff', '#25b8fdff',
                 '#0f71f0ff', '#046db9ff',
                 '#8988e5ff', '#e200feff',
                 '#ea0586ff', '#fe3ccaff']

xing_A_colors = ['#bf0001ff', '#7d3e22ff',
                 '#dcdb04ff', '#81d404ff',
                 '#9a5e45ff', '#166e06ff',
                 '#fe7300ff', '#8b4412ff',
                 '#07abc0ff', '#25b8fdff',
                 '#046db9ff', '#8988e5ff',
                 '#e200feff', '#ea0586ff',
                 '#fe3ccaff', '#000104ff']


def _sort_by_elec(df):
    sorted_df = df.sort_values(by=['Electrode_ID'])
    sorted_df['index'] = range(1024)
    sorted_df.set_index('index', inplace=True)
    return sorted_df


if __name__ == "__main__":

    # Get arguments
    vargs = docopt(__doc__)
    csv_paths = [path for path in vargs['--csv-list'].split(';')]
    map_path = vargs['--mapping']
    plt_path = vargs['--plt']
    monkey = vargs['--monkey']
    out_path = vargs['--out']

    # Load each RF session and the array mapping file
    A = pd.read_csv(csv_paths[0])
    B = pd.read_csv(csv_paths[1])
    mapping = pd.read_csv(map_path)

    # Get them sorted (to avoid inconsistencies when merging)
    A = _sort_by_elec(A)
    B = _sort_by_elec(B)
    mapping = _sort_by_elec(mapping)

    # Get sum of SNR values for each array in all directions
    directions = ['rightward', 'leftward', 'downward', 'upward']
    A['SNR'] = np.sum(np.array([A[f'SNR_fromRF_{d}'] for d in directions]),
                      axis=0)
    B['SNR'] = np.sum(np.array([B[f'SNR_fromRF_{d}'] for d in directions]),
                      axis=0)
    combined = []
    # Select best session for each array (small/large bars)
    # Criteria include most non-excluded electrodes or highest sum of SNR
    for array in range(1, 17):
        a = A[A['Array_ID'] == array]
        b = B[B['Array_ID'] == array]
        if monkey == 'L' and array in [2, 3]:
            if a['date'].iloc[0] == '2017-06-26':
                combined.append(a)
            else:
                combined.append(b)
        elif monkey == 'A' and array in [2, 5]:
            if a['date'].iloc[0] == '2018-08-28':
                combined.append(a)
            else:
                combined.append(b)
        elif len(a.dropna()) > len(b.dropna()):
            combined.append(a)
        elif len(a.dropna()) < len(b.dropna()):
            combined.append(b)
        else:
            if np.sum(a['SNR']) > np.sum(b['SNR']):
                combined.append(a)
            elif np.sum(a['SNR']) < np.sum(b['SNR']):
                combined.append(b)

    # Cast into a single table
    df = pd.concat(combined)
    df = _sort_by_elec(df)

    # Include ID mapping
    df = df.merge(mapping)
    df['Date and Area'] = df['date'] + ', V' + df['Area'].astype(str)
    df.to_csv(out_path, index=False)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(4, 6))
    if monkey == 'L':
        pal = sns.color_palette(xing_L_colors)
        pal_V1 = [p for i, p in enumerate(pal) if i not in [1, 2]]
        pal_V4 = [p for i, p in enumerate(pal) if i in [1, 2]]
    elif monkey == 'A':
        pal = sns.color_palette(xing_A_colors)
        pal_V1 = [p for i, p in enumerate(pal) if i not in [1, 4]]
        pal_V4 = [p for i, p in enumerate(pal) if i in [1, 4]]

    V1 = df[df['Area'] == 1]
    V4 = df[df['Area'] == 4]
    sns.scatterplot(data=V1, x='RF center X (degrees)',
                    y='RF center Y (degrees)', hue='Array_ID',
                    style='Date and Area', markers=['s', 'o'], palette=pal_V1,
                    ax=axs[0], s=10, legend=False)
    sns.scatterplot(data=V4, x='RF center X (degrees)',
                    y='RF center Y (degrees)', hue='Array_ID',
                    style='Date and Area', palette=pal_V4, ax=axs[1], s=10,
                    legend=False)

    # Create legend manually
    patches = [Line2D([0], [0], marker='o', lw=0, color=p, label='Scatter',
                      markerfacecolor=p, markersize=5)
               for i, p in enumerate(pal)]
    extra = [Line2D([0], [0], marker='s', lw=0, color='white', label='Scatter',
                    markerfacecolor='white', markersize=5),
             Line2D([0], [0], marker='s', lw=0, color='gray', label='Scatter',
                    markerfacecolor='gray', markersize=5),
             Line2D([0], [0], marker='o', lw=0, color='gray', label='Scatter',
                    markerfacecolor='gray', markersize=5)]
    labels = ['Array ' + str(i) for i in range(1, 17)]
    extra_lbl = ['', 'Small bars', 'Large bars']
    axs[0].legend(patches+extra, labels+extra_lbl, loc='center left',
                  bbox_to_anchor=(1.03, 0.5), fontsize=7)

    for ax in axs:
        # Draw background grid
        ax.axhline(0, color='k', ls='--', lw=0.5, zorder=-101)
        ax.axvline(0, color='k', ls='--', lw=0.5, zorder=-101)
        for l in np.arange(2, 24, step=2):
            x = np.linspace(-l, l, num=1000)
            ax.plot(x, -np.sqrt(l**2 - x**2), color='k', ls='--', lw=0.2,
                    zorder=-101, scaley=False, scalex=False)
            ax.plot(x, np.sqrt(l**2 - x**2), color='k', ls='--', lw=0.2,
                    zorder=-101, scaley=False, scalex=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    if monkey == 'L':
        axs[0].set_xlim(-0.9, 7.8)
        axs[0].set_ylim(-7.8, 0.9)
        axs[1].set_xlim(-0.9, 7.8)
        axs[1].set_ylim(-7.8, 0.9)
    if monkey == 'A':
        axs[0].set_xlim(-0.6, 5.8)
        axs[0].set_ylim(-5.8, 0.6)
        axs[1].set_xlim(-0.6, 11)
        axs[1].set_ylim(-11, 0.6)
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    axs[0].set_title(f'Monkey {monkey}')

    plt.tight_layout()
    plt.savefig(plt_path)
