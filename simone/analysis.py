# coding=utf-8
import os.path as op
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from itertools import product
from simone import staircases, DATA_DIR

if __name__ == '__main__':

    users_df, colors_df, trials_df, exps_df = staircases()

    # Usually you only need to work witht exps_df.
    # exps_df contains the experiment data arranged one "experiment" per row
    # A experiment is a full staircase
    # The staircase + responses are in their own dataframe in the column "series"

    # Let's summarise the staircases by luminance difference of last 3 reversals
    def last_mean(df, last=3, do_raise=False):
        if df.reversal.sum() < last:
            if do_raise:
                raise Exception('There are not enough reversals in the experiment (%d < %d)' %
                                (df.reversal.sum(), last))
            return np.nan
        achromatic_l = df[df.reversal].measured_l_achromatic.values[-last:].mean()
        return achromatic_l

    def diff(df, last=3):
        # Find the chromatic luminance; one of many possible ways to query for this...
        assert df.measured_l_chromatic.nunique() == 1
        chromatic_l = df.measured_l_chromatic.unique()[0]
        # Find the last 3 reversals mean
        achromatic_l = last_mean(df, last=last)
        return achromatic_l - chromatic_l

    staircases_summaries = []
    for last in [1, 2, 3, 4, 5]:
        exps_df['diff%d' % last] = exps_df.series.apply(partial(diff, last=last))
        exps_df['last%d_mean' % last] = exps_df.series.apply(partial(last_mean, last=last))
        staircases_summaries.extend(['diff%d' % last, 'last%d_mean' % last])

    # Select only some experiments
    exps_df = exps_df.query('not omit and '
                            'reversal_method == "simple-reversals" and '
                            'staircase_repetition in [0, 1]')
    label_names = ['age', 'gender', 'education', 'vision', 'chromatic', 'chromatic_first']

    for label_name, summarizer in product(label_names, staircases_summaries):
        fig, (top, bottom) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False,
                                          figsize=(18, 12))
        for label, label_df in exps_df.groupby(label_name):
            label = '%s (n=%d)' % (label, len(label_df))
            sns.distplot(label_df[summarizer], label=label, hist=False, rug=True, ax=top)
            sns.kdeplot(label_df[summarizer], label=label, shade=True, cumulative=True, ax=bottom)
        plt.suptitle(label_name, fontsize=20)
        plt.savefig(op.join(DATA_DIR, '%s-%s-distro.png' % (label_name, summarizer)))
        plt.close()
    plt.show()
