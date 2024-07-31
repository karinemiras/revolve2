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
    df = pd.read_csv(origin_file)

    keys = ['experiment_name', 'run', 'gen', 'ranking', 'individual_id', 'geno_size']
    others = ['knockout']
    original = 'o'
    df = df.filter(items=keys+traits+others)

    for trait in traits:
        pivot_df = df.pivot_table(index=keys,
                                  columns='knockout', values=trait,
                                  # for distance variable, which is not a trait, the calculation is idle (comapared to 0)
                                  aggfunc='first')  # (original value=o) is the first

        all_columns = pivot_df.columns
        knock_columns = [col for col in all_columns if col not in keys and col != original]
        # Subtract each knock_columns by the original (positive values mean the mutant had an increse in the trait or growth)
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
        nagative = df_delta[int_columns] < 0

        count_positive = positive.sum(axis=1)
        count_neutral = neutral.sum(axis=1)
        count_nagative = nagative.sum(axis=1)

        df_delta['positive'] = count_positive
        df_delta['neutral'] = count_neutral
        df_delta['nagative'] = count_nagative

        positive_values = df_delta[int_columns].where(positive)
        negative_values = df_delta[int_columns].where(nagative)

        positive_values = positive_values.fillna(0)
        negative_values = negative_values.fillna(0)

        positive_avg = positive_values.mean(axis=1)
        negative_avg = negative_values.mean(axis=1)

        df_delta['avg_positive'] = positive_avg
        df_delta['avg_negative'] = negative_avg

        df_delta = df_delta.reset_index()[keys + ['positive', 'neutral', 'nagative', 'avg_positive', 'avg_negative']]
        df_delta.to_csv(f'{path}/effects_{trait}.csv')
        print(trait)

def plot():
    import matplotlib.pyplot as plt
    clrs = ['#009900',
            '#EE8610',
            '#434699',
            '#95fc7a',
            '#221210',
            '#87ac65']

    metrics = ['positive', 'nagative', 'avg_positive', 'avg_negative']

    keys = ['experiment_name', 'gen', 'ranking']

    for trait in traits:

        df = pd.read_csv(f'effects_{trait}.csv')

        grouped = df.groupby(['experiment_name', 'gen', 'ranking'])
        stats = []

        # Group by the keys
        grouped = df.groupby(keys)

        # Initialize a list to store the results
        results = []

        # Iterate over the variables
        for var in metrics:
            # Calculate median, Q1, and Q3
            median = grouped[var].median().rename('median')
            q1 = grouped[var].quantile(0.25).rename('Q1')
            q3 = grouped[var].quantile(0.75).rename('Q3')

            #         mean = grouped[var].mean().rename('mean')
            #         std = grouped[var].std().rename('std')

            # Combine results into a single DataFrame for each variable
            stats = pd.concat([median, q1, q3], axis=1).reset_index()
            # stats = pd.concat([mean, std], axis=1).reset_index()
            stats['variable'] = var

            # Append results to the list
            results.append(stats)

        # Concatenate all results into a single DataFrame
        final_stats_df = pd.concat(results, ignore_index=True)

        # Print the final DataFrame
        # print(final_stats_df)

        # Define the columns to use for grouping
        group_columns = ['experiment_name', 'ranking']

        # Find unique combinations of the specified columns
        unique_groups = final_stats_df[group_columns].drop_duplicates()

        filtered_dfs = {}

        # Iterate over each unique group
        for _, group in unique_groups.iterrows():
            # Filter rows based on the group
            filter_condition = (final_stats_df[group_columns] == group).all(axis=1)
            filtered_df = final_stats_df[filter_condition]

            # Use a tuple of group values as the key in the dictionary
            group_key = tuple(group)
            filtered_dfs[group_key] = filtered_df

            font = {'font.size': 20}
            plt.rcParams.update(font)
            fig, ax = plt.subplots()

            plt.xlabel('')
            plt.ylabel(f' ')

            for idx, metric in enumerate(metrics):
                curve = filtered_df[filtered_df['variable'] == metric]
                print(curve)

                # ax.plot(curve['gen'], curve[f'mean'],  label=f'{metric}',  c=clrs[idx])
                ax.plot(curve['gen'], curve[f'median'], label=f'{metric}', c=clrs[idx])
                # ax.fill_between(curve['gen'],  curve[f'mean']-curve[f'std'],   curve[f'mean']+curve[f'std'],    alpha=0.3, facecolor=clrs[idx])
                ax.fill_between(curve['gen'], curve[f'Q1'], curve[f'Q3'], alpha=0.3, facecolor=clrs[idx])
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5,
                          fontsize=10)

            #  plt.show()

            plt.savefig(f'{group_key}_{trait}.png', bbox_inches='tight')
            plt.clf()
            plt.close(fig)


#calculate()
plot()

