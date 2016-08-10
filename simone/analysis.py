# coding=utf-8
import os.path as op
from itertools import product

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from simone import MY_DIR, EXPS_NON_STAIRCASE_COLUMNS, read_experiments_df, read_colors_df


def labs2rgbs():
    # N.B. we ignore the luminant, as we just need a bare idea of how the color looks like...
    from skimage.color import lab2rgb
    colors_df = read_colors_df()
    # lab2rgb expects an image of floats...
    labs = colors_df[['photoshop_l', 'photoshop_a', 'photoshop_b']].values[None, :].astype(float)
    # we will just save tuples
    colors_df['rgb'] = map(tuple, lab2rgb(labs)[0])
    return colors_df


def many_distro_plots():
    exps_df = read_experiments_df()

    # Select only some experiments
    exps_df = exps_df.query('not omit and '
                            'reversal_method == "simple-reversals" and '
                            'staircase_repetition in [0]')
    label_names = ['age', 'gender', 'education', 'vision', 'chromatic', 'chromatic_first']
    staircases_summaries = [col for col in exps_df.columns
                            if col not in EXPS_NON_STAIRCASE_COLUMNS + ['series']]

    for label_name, summarizer in product(label_names, staircases_summaries):
        fig, (top, bottom) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False,
                                          figsize=(18, 12))
        for label, label_df in exps_df.groupby(label_name):
            label = '%s (n=%d)' % (label, len(label_df))
            sns.distplot(label_df[summarizer], label=label, hist=False, rug=True, ax=top)
            sns.kdeplot(label_df[summarizer], label=label, shade=True, cumulative=True, ax=bottom)
        plt.suptitle(label_name, fontsize=20)
        plt.savefig(op.join(MY_DIR, 'figures', '%s-%s-distro.png' % (label_name, summarizer)))
        plt.close()
    plt.show()


def dist_factor_plots():
    # Read the colors table and add RGBs for matplotlib to digest
    colors_df = labs2rgbs()

    # Read in the experiments table
    exps_df = read_experiments_df()

    # Select only non imitted experiments with simple-reversal policy
    exps_df = exps_df.query('not omit and '
                            'reversal_method == "simple-reversals" and '
                            'staircase_repetition in [0]').copy()

    # Make nice groupers / labels
    exps_df['chromatic_first'] = exps_df['chromatic_first'].apply(
        lambda x: 'first chromatic' if x else 'first achromatic')
    exps_df['age_groups'] = pd.cut(exps_df['age'],
                                   bins=[0, 30, 35, 45, 90])
    # N.B. the ages span is small in your dataset

    # These are the values whose distro we will look at, with nice human names
    luminances = {
        'measured_l_achromatic__mean__last2':
            u'achromatic luminance (cd/m², mean last 2 reversals)',
        'measured_wall_l_achromatic__mean__last2':
            u'achromatic wall luminance (cd/m², mean last 2 reversals)',
        'measured_scene_l_achromatic__mean__last2':
            u'achromatic scene luminance (cd/m², mean last 2 reversals)',
        'wall_minus_scene_l_achromatic__mean__last2':
            u'achromatic (wall - scene) luminance (cd/m², mean last 2 reversals)',
    }

    # We want the chromatic colors (C1, C2, C3) per row
    row_label = 'chromatic'

    # In each column of our plot we will group by a different factor
    column_labels = ['gender',
                     'education',
                     'vision',
                     'age_groups',
                     'chromatic_first']

    # We won't consider groups with less than these observations
    min_n = 4

    # We can plot either empirical density or CDF functions
    # and plain achromatic luminance or diff with chromatic luminance?

    for luminance, cumulative, use_diff in product(luminances,
                                                   [True, False],
                                                   [True, False]):

        num_rows = exps_df[row_label].nunique()
        num_cols = len(column_labels)

        fig, subplots = plt.subplots(nrows=num_rows,
                                     ncols=num_cols,
                                     sharex=True, sharey=True,
                                     figsize=(24, 16))
        # It is easier to do this by hand than to use, for example,  seaborn FacetGrid

        for row, (chromatic, cdf) in enumerate(exps_df.groupby(row_label)):
            rgb_chromatic = colors_df.loc[chromatic].rgb
            for col, column_label in enumerate(column_labels):
                for labelnum, (label, ldf) in enumerate(cdf.groupby(column_label)):
                    if len(ldf) < min_n:
                        continue
                    ax = subplots[row][col]
                    label = '%s (n=%d)' % (label, len(ldf))
                    # Haaaack!
                    if not use_diff:
                        l_luminances = ldf[luminance]
                    else:
                        chromatic_l_name = (luminance.
                                            partition('__')[0].
                                            replace('achromatic', 'chromatic'))
                        l_luminances = ldf[luminance] - ldf[chromatic_l_name]
                    # End of haaaack!

                    if cumulative:
                        sns.kdeplot(l_luminances,
                                    label=label,
                                    cumulative=True,
                                    ax=ax)
                    else:
                        sns.distplot(l_luminances,
                                     label=label,
                                     hist=False,
                                     rug=True,
                                     ax=ax)
                    if col == 0:
                        ax.set_ylabel(chromatic, fontsize=40, rotation=0, labelpad=30)
                        ax.yaxis.label.set_color(rgb_chromatic)
                    if row == 0:
                        ax.set_title(column_label, fontsize=20)
                    if row < (num_rows - 1):
                        ax.set_xlabel('')
                    else:
                        ax.set_xlabel(u'luminance (cd/m²)')

        fig.suptitle('%s of %s%s' % (
            'Density' if not cumulative else 'CDF',
            luminances[luminance],
            ' difference' if use_diff else ''),
                     fontsize=30)

        plt.savefig(op.join(MY_DIR, 'figures', 'distro-factor', '%s__diff=%r__cumulative=%r.png' %
                            (luminance, use_diff, cumulative)))

        plt.close()


if __name__ == '__main__':
    many_distro_plots()
    dist_factor_plots()
