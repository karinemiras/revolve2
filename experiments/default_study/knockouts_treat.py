import pandas as pd
import argparse
import warnings
import numpy as np

import plotly.graph_objs as go
import plotly.offline as offline

#import plotly.graph_objects as go
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x) if abs(x) > 0.001 else '{:.0f}'.format(x * 1000))
pd.set_option('display.float_format', lambda x: '%.10f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)

parser = argparse.ArgumentParser()
parser.add_argument("study")
parser.add_argument("experiments")
parser.add_argument("tfs")
parser.add_argument("watchruns")
parser.add_argument("generations")
parser.add_argument("mainpath")
args = parser.parse_args()

study = args.study
experiments_name = args.experiments.split(',')
tfs = list(args.tfs.split(','))
runs = args.watchruns.split(',')
generations = [0, 25, 100]
mainpath = args.mainpath
clrs = ['#009900',
        '#EE8610',
        '#434699',
        '#95fc7a',
        '#221210',
        '#87ac65']

path = f'{mainpath}/{study}/analysis/knockouts'
traits = ['disp_y', 'extremities_prop', 'distance']


def calculate():
    origin_file = f'{path}/knockouts_measures.csv'
    df_ori = pd.read_csv(origin_file)
    original = 'o'  # without knockout

    keys = ['experiment_name', 'run', 'gen', 'ranking', 'individual_id', 'geno_size']
    traits = ['disp_y', 'extremities_prop', 'distance', 'symmetry', 'proportion', 'coverage', 'extensiveness_prop',
              'hinge_prop', 'modules_count', 'head_balance']
    others = ['knockout']
    df = df_ori.filter(items=keys + others + traits)

    # df = df[
    #     ( ( df['experiment_name'] == 'reg2m2') & (df['run'] == 1) &  (df['gen'] == 0 )  ) ]

    df = df[((df['gen'] == 0) | (df['gen'] == 100))]

    for trait in traits:
        # sends trait values of each knockout to columns
        pivot_df = df.pivot_table(index=keys,
                                  columns='knockout', values=trait,
                                  # for distance variable, which is not a trait,
                                  # the calculation is idle (compared to 0)
                                  aggfunc='first')

        all_columns = pivot_df.columns
        knock_columns = [col for col in all_columns if col not in keys and col != original]
        # Subtract each knock_columns by the original
        # (positive values mean the mutant had an increase in the trait or growth)
        df_delta = pivot_df.drop(columns=original).sub(pivot_df[original], axis=0)

        double_knocks = [col for col in knock_columns if '.' in col]
        for double_knock in double_knocks:
            genes = double_knock.split('.')

            additive = df_delta[genes[0]] + df_delta[genes[1]]
            df_delta[f'{genes[0]}add{genes[1]}'] = additive
            df_delta[f'{genes[0]}int{genes[1]}'] = df_delta[double_knock] - additive

        int_columns = [col for col in df_delta.columns if 'int' in col]
        positive = df_delta[int_columns] > 0
        neutral = df_delta[int_columns] == 0
        negative = df_delta[int_columns] < 0

        is_finite = np.isfinite(df_delta[int_columns])

        positive = positive & is_finite
        neutral = neutral & is_finite
        negative = negative & is_finite

        count_positive = positive.sum(axis=1)
        count_neutral = neutral.sum(axis=1)
        count_negative = negative.sum(axis=1)

        df_delta['positive'] = count_positive
        df_delta['neutral'] = count_neutral
        df_delta['negative'] = count_negative
        df_delta['total'] = count_positive + count_neutral + count_negative

        df_delta['positive'] = df_delta['positive'] / df_delta['total']
        df_delta['neutral'] = df_delta['neutral'] / df_delta['total']
        df_delta['negative'] = df_delta['negative'] / df_delta['total']

        positive_values = df_delta[int_columns].where(positive)
        negative_values = df_delta[int_columns].where(negative)

        positive_avg = positive_values.mean(axis=1, skipna=True)
        negative_avg = negative_values.mean(axis=1, skipna=True)
        df_delta['avg_positive'] = positive_avg
        df_delta['avg_negative'] = negative_avg

        df_exp = df_delta.reset_index()[keys + ['positive', 'neutral', 'negative', 'avg_positive', 'avg_negative']]
        df_exp.to_csv(f'{path}/effects_{trait}.csv')

        print(trait)
#
# def plot():
#     clrs = ['#009900',
#             '#EE8610',
#             '#434699',
#             '#95fc7a',
#             '#221210',
#             '#87ac65']
#
#     metrics = ['positive', 'avg_positive', 'negative', 'avg_negative']
#
#     keys = ['experiment_name', 'gen', 'ranking']
#
#     dfs_trait = {}
#     for trait in traits:
#         print('>>>>>', trait)
#         dfs_trait[trait] = pd.read_csv(f'effects_{trait}.csv')
#
#
#     # boxes
#
#     import seaborn as sb
#     from statannot import add_stat_annotation
#     from scipy.stats import mannwhitneyu
#     from scipy.stats import wilcoxon, ttest_ind
#     import warnings
#
#     warnings.filterwarnings('ignore')
#
#     # Set IPython to display all outputs
#     InteractiveShell.ast_node_interactivity = "all"
#
#     for trait in traits:
#         print('>>>>>>', trait)
#
#         df_trait = dfs_trait[trait]
#         f_trait = df_trait[(df_trait['ranking'] == 'best')]
#
#         group_columns = ['experiment_name', 'ranking']
#         unique_groups = df_trait[group_columns].drop_duplicates()
#         filtered_dfs = {}
#
#         for _, group in unique_groups.iterrows():
#             print(group)
#             # Filter rows based on the group
#             filter_condition = (df_trait[group_columns] == group).all(axis=1)
#             filtered_df = df_trait[filter_condition]
#             group_key = tuple(group)
#             filtered_dfs[group_key] = filtered_df
#
#             fig, axes = plt.subplots(1, 4, figsize=(15, 5))
#             sb.set(rc={"axes.titlesize": 23, "axes.labelsize": 23, 'ytick.labelsize': 21, 'xtick.labelsize': 21})
#             sb.set_style("whitegrid")
#
#             for idx, metric in enumerate(metrics):
#                 ax = axes[idx]
#
#                 filtered_clean = filtered_df
#                 if metric in ['avg_positive', 'avg_negative']:
#                     filtered_clean = filtered_df[pd.notna(filtered_df[metric])]
#
#                 group1_data = filtered_clean[filtered_clean['gen'] == 0][f'{metric}']
#                 group2_data = filtered_clean[filtered_clean['gen'] == 100][f'{metric}']
#
#                 # _, p_value = wilcoxon(group1_data, group2_data)
#                 _, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
#                 print(f"\nMetric: {metric}, p-value: {round(p_value, 4)}")
#
#                 if p_value < 0.01:
#                     # display(filtered_clean[[metric, 'experiment_name',  'run',  'gen' ]])
#                     #  ax.text(0.5, max(filtered_clean[f'{metric}'].max()) + 0.1, f'p={p_value:.2f}', ha='center', va='bottom', fontsize=12)
#
#                     sb.boxplot(x='gen', y=f'{metric}', data=filtered_clean,
#                                palette=clrs, width=0.4, showmeans=True, linewidth=2, fliersize=6,
#                                meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"}, ax=ax)
#
#                     ax.tick_params(axis='x', labelrotation=10)
#
#                 ax.set_xlabel('')
#                 ax.set_ylabel(f'{metric}')
#
#             plt.tight_layout()
#             plt.show()

calculate()
#plot()

