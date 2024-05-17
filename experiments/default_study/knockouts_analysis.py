import pandas as pd
import argparse
import warnings
import numpy as np

import plotly.graph_objs as go
import plotly.offline as offline

import plotly.graph_objects as go
import pandas as pd
from scipy import stats

offline.init_notebook_mode(connected=True)


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
generations = [0, 50]
mainpath = args.mainpath


def load():
    origin_file = f'{mainpath}/{study}/analysis/knockouts/knockouts_measures.csv'

    df = pd.read_csv(origin_file)
    print(df)


    warnings.filterwarnings("ignore")
    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x) if abs(x) > 0.001 else '{:.0f}'.format(x * 1000))
    pd.set_option('display.float_format', lambda x: '%.10f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)

    original = 'o'
    df = pd.read_csv(origin_file)

    filtered_df = df
    # filtered_df = df[
    #     ( df['experiment_name'] == 'reg2m2')# &

    #                  & (df['run'] == 7)
    #                  & (df['gen'] == 0)
    #                  & (df['ranking'] == 'worst')
    #                # & (df['individual_id'] == 8252)
    #                ]
    fixed_columns = ['experiment_name', 'run', 'gen', 'ranking', 'individual_id']
    pivot_df = filtered_df.pivot_table(index=fixed_columns,
                                       columns='knockout', values='distance',# values='speed_y',
                                       aggfunc='first')  # (original value) is the first

    all_columns = pivot_df.columns
    knock_columns = [col for col in all_columns if col not in fixed_columns and col != original]
    first_day_value = pivot_df[original]
    pivot_df.replace(-np.inf, np.nan, inplace=True)

    # def replace_value(x):
    #     if x < 0.001:
    #         return 0.001
    #     else:
    #         return x
    # pivot_df[knock_columns] = pivot_df[knock_columns].applymap(replace_value)
    # diff_df = pivot_df.divide(first_day_value, axis=0)  #TODO: absiolut diff instead of divid? or use mimumum speed 0.001?
    # do not subtract in case of distance
    diff_df = pivot_df.subtract(first_day_value, axis=0)

    diff_df = diff_df.drop(columns=original)

    # number1.nubmer2 means knockout of promotors 1 and 2 simultaneously
    double_knocks = [col for col in knock_columns if '.' in col]
    single_knocks = []
    for knock in double_knocks:
        promotors = knock.split('.')
        diff_df[f'{promotors[0]}_{promotors[1]}'] = diff_df[promotors[0]] + diff_df[promotors[1]]
        single_knocks.append(f'{promotors[0]}_{promotors[1]}')

    average_single_values = diff_df[single_knocks].mean(axis=1, skipna=True)
    #diff_df['single_avg'] = average_values

    average_double_values = diff_df[double_knocks].mean(axis=1, skipna=True)
    diff_df['diff'] = average_double_values - average_single_values

    intra_mean = diff_df.groupby(['experiment_name', 'run', 'gen', 'ranking']).agg(
        {'diff': 'mean'})
    inter_mean = intra_mean.groupby(['experiment_name', 'gen', 'ranking']).agg(
        {'diff': 'mean'})

    intra_mean = intra_mean.reset_index()

    print(intra_mean)






    # Get unique values of experiment_name and ranking
    experiment_names = intra_mean['experiment_name'].unique()
    rankings = intra_mean['ranking'].unique()

    # Loop through each combination of experiment_name and ranking
    for experiment_name in experiment_names:
        for ranking in rankings:
            # Filter data for the current combination of experiment_name and ranking
            filtered_data = intra_mean[
                (intra_mean['experiment_name'] == experiment_name) & (intra_mean['ranking'] == ranking)]

            # Create box traces for single_avg and double_avg
            trace0 = go.Box(
                y=filtered_data['diff'],
                x=filtered_data['gen'],
                name='single knockout _avg',
                marker=dict(color='#3D9970')
            )
            # trace1 = go.Box(
            #     y=filtered_data['double_avg'],
            #     x=filtered_data['gen'],
            #     name='double knockout_avg',
            #     marker=dict(color='#FF4136')
            # )

            # TODO scale plots among tfs
            # TODO separate by run
            
            # Define data and layout
            data = [trace0]
            layout = go.Layout(
                yaxis=go.layout.YAxis(
                    title='body edit tree distance',
                    zeroline=False
                ),
                yaxis2=go.layout.YAxis(
                    side='right',
                    overlaying='y',
                    zeroline=False
                ),
                boxmode='group'
            )

            # Create figure
            fig = go.Figure(data=data, layout=layout)

            # Statistical testing (e.g., t-test) for the final pair of boxes
          #  single_avg_data = filtered_data['single_avg']
           # double_avg_data = filtered_data['double_avg']
           # statistic, p_value = stats.wilcoxon(single_avg_data, double_avg_data)

            # Annotation text for the test result
            # annotation_text = f"statistic: {statistic:.2f}, P-value: {p_value:.4f}"
            #
            # # Add annotation to the figure
            # fig.add_annotation(
            #     xref='paper', yref='paper',
            #     x=0.5, y=0.9,
            #     text=annotation_text,
            #     showarrow=False,
            #     font=dict(size=12),
            #     bgcolor="white",
            #     bordercolor="black",
            #     borderwidth=1
            # )

            # Write image to file
            fig.write_image(f"{mainpath}/{study}/analysis/knockouts/{experiment_name}_{ranking}_plot.png")

    # from itertools import combinations
    # # Get unique values of experiment_name and ranking
    # experiment_names = intra_mean['experiment_name'].unique()
    # rankings = intra_mean['ranking'].unique()
    #
    # # Loop through each combination of experiment_name and ranking
    # for experiment_name in experiment_names:
    #     for ranking in rankings:
    #         # Filter data for the current combination of experiment_name and ranking
    #         filtered_data = intra_mean[
    #             (intra_mean['experiment_name'] == experiment_name) & (intra_mean['ranking'] == ranking)]
    #
    #         # Perform statistical tests for all pairs of single_avg and double_avg
    #         pairs = list(combinations(filtered_data.index, 2))  # Generate all pairs of indices
    #         for pair in pairs:
    #             index1, index2 = pair
    #             single_avg1 = filtered_data.loc[index1, 'single_avg']
    #             single_avg2 = filtered_data.loc[index2, 'single_avg']
    #             double_avg1 = filtered_data.loc[index1, 'double_avg']
    #             double_avg2 = filtered_data.loc[index2, 'double_avg']
    #
    #             # Perform paired t-test
    #             t_statistic_single, p_value_single = stats.ttest_rel([single_avg1, single_avg2], [double_avg1, double_avg2])
    #
    #             # Print results
    #             print(f"Experiment: {experiment_name}, Ranking: {ranking}, Pair: ({index1}, {index2})")
    #             print(
    #                 f"Paired t-test (single_avg vs. double_avg): T-statistic: {t_statistic_single}, p-value: {p_value_single}")
    #
    #             # Create box traces for single_avg and double_avg
    #             trace0 = go.Box(
    #                 y=[single_avg1, single_avg2],
    #                 x=[filtered_data.loc[index1, 'gen'], filtered_data.loc[index2, 'gen']],
    #                 name='single_avg',
    #                 marker=dict(color='#3D9970')
    #             )
    #             trace1 = go.Box(
    #                 y=[double_avg1, double_avg2],
    #                 x=[filtered_data.loc[index1, 'gen'], filtered_data.loc[index2, 'gen']],
    #                 name='double_avg',
    #                 marker=dict(color='#FF4136')
    #             )
    #
    #             # Define data and layout
    #             data = [trace0, trace1]
    #             layout = go.Layout(
    #                 yaxis=go.layout.YAxis(
    #                     title='Normalized Moisture',
    #                     zeroline=False
    #                 ),
    #                 yaxis2=go.layout.YAxis(
    #                     side='right',
    #                     overlaying='y',
    #                     zeroline=False
    #                 ),
    #                 boxmode='group'
    #             )
    #
    #             # Create figure
    #             fig = go.Figure(data=data, layout=layout)
    #
    #             # Perform statistical test
    #             t_statistic, p_value = stats.ttest_rel([single_avg1, single_avg2], [double_avg1, double_avg2])
    #
    #             # Define annotation text
    #             annotation_text = f"T-statistic: {t_statistic:.2f}, p-value: {p_value:.4f}"
    #
    #             # Add annotation to the figure
    #             fig.add_annotation(
    #                 x=1,  # x-coordinate of annotation (top right of the plot)
    #                 y=max(single_avg1, single_avg2),  # y-coordinate of annotation (top of the box plot)
    #                 text=annotation_text,
    #                 showarrow=False,
    #                 font=dict(color="black", size=12),
    #                 xref="paper",
    #                 yref="y",
    #                 xshift=10
    #             )
    #
    #             # Write image to file
    #             fig.write_image(
    #                 f"{mainpath}/{study}/analysis/knockouts/{experiment_name}_{ranking}_{index1}_{index2}_plot.png")


load()