# coding=utf-8
import os.path as op
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns

from simone import MY_DIR, staircases, EXPS_NON_STAIRCASE_COLUMNS

if __name__ == '__main__':

    colors_df, participants_df, trials_df, exps_df = staircases()

    # Usually you only need to work witht exps_df.
    # exps_df contains the experiment data arranged one "experiment" per row
    # A experiment is a full staircase
    # The staircase + responses are in their own dataframe in the column "series"
    # There are many columns that correspond to staircase statistics

    # Select only some experiments
    exps_df = exps_df.query('not omit and '
                            'reversal_method == "simple-reversals" and '
                            'staircase_repetition in [0, 1]')
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
