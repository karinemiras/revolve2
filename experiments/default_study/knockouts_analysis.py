import pandas as pd
import argparse
import warnings
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline
import plotly.graph_objects as go
import pandas as pd
from scipy import stats

import sys

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x) if abs(x) > 0.001 else '{:.0f}'.format(x * 1000))
pd.set_option('display.float_format', lambda x: '%.10f' % x)
pd.set_option('display.max_columns', None)


parser = argparse.ArgumentParser()
parser.add_argument("study")
parser.add_argument("experiments")
parser.add_argument("tfs")
parser.add_argument("watchruns")
parser.add_argument("generations")
parser.add_argument("mainpath")
args = parser.parse_args()

study = 'dispbodygenov1' #args.study
experiments_name = args.experiments.split(',')
tfs = list(args.tfs.split(','))
runs = args.watchruns.split(',')
generations = [0, 25, 100]
mainpath = args.mainpath


def load():

    fixed_columns = ['experiment_name', 'run', 'gen', 'ranking', 'individual_id']
    traits_columns = ['speed_y','extremities_prop'] #TODO fix disp_y
    other_columns = ['geno_size', 'n_promotors', 'knockout', 'distance'] #TODO: change n_promotors name

    origin_file = f'{mainpath}/{study}/analysis/knockouts/knockouts_measures.csv'

    df = pd.read_csv(origin_file)


    original = 'o'
    df = pd.read_csv(origin_file)

    # df = df[
    #     ( df['experiment_name'] == 'reg2m2')# &
    #                  & (df['run'] == 10)
    #                  &  (df['gen'] == 100)
    #                 & (df['ranking'] == 'best')
    #             #    & (df['individual_id'] == 28 )
    #     ]

    df = df.filter(items=fixed_columns+traits_columns+other_columns)
    #print(df)

    for trait in traits_columns:
        pivot_df = df.pivot_table(index=fixed_columns,
                                           columns='knockout', values='distance', # replace by trait
                                           aggfunc='first')  # (original value) is the first

        all_columns = pivot_df.columns
        knock_columns = [col for col in all_columns if col not in fixed_columns and col != original]
        original_value = pivot_df[original]
        pivot_df.replace(-np.inf, np.nan, inplace=True)
        diff_df = pivot_df.subtract(original_value, axis=0)

        diff_df = diff_df.drop(columns=original)

        # TODO categorize by antons paper and count how many pairs fall into each categ
        #TODO: plot trends for

        # gene1.number2 means knockout of genes 1 and 2 simultaneously (double_knocks)
        double_knocks = [col for col in knock_columns if '.' in col]
        single_knocks = []
        diffs = []
        for knock in double_knocks:
            genes = knock.split('.')

            diff_df[f'{genes[0]}plus{genes[1]}'] = diff_df[genes[0]] + diff_df[genes[1]]
            single_knocks.append(f'{genes[0]}_{genes[1]}')

            diff_df[f'{genes[0]}diff{genes[1]}'] = diff_df[knock] - diff_df[f'{genes[0]}plus{genes[1]}']
            diffs.append(f'{genes[0]}diff{genes[1]}')

        diff_df['average_diffs'] = diff_df[diffs].mean(axis=1, skipna=True)
        print(diff_df[diffs])

       # print(diff_df)
        intra_mean = diff_df.groupby(['experiment_name', 'run', 'gen', 'ranking']).agg(
            {'average_diffs': 'mean'})
        inter_mean = intra_mean.groupby(['experiment_name', 'gen', 'ranking']).agg(
            {'average_diffs': 'mean'})

        intra_mean = intra_mean.reset_index()
        pd.set_option('display.max_rows', 10000)
     #   print(intra_mean)
       # print(inter_mean)



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
                y=filtered_data['average_diffs'],
                x=filtered_data['gen'],
                name='single knockout _avg',
                marker=dict(color='#3D9970')
            )

            # TODO scale plots among tfs
            # TODO separate by run
            
            # Define data and layout
            data = [trace0]
            layout = go.Layout(
                yaxis=go.layout.YAxis(
                    title='metric',
                    zeroline=False,
                    range=[-4, 4]
                ),
                yaxis2=go.layout.YAxis(
                    side='right',
                    overlaying='y',
                    zeroline=False,
                    range=[-4, 4]
                ),
                boxmode='group'
            )

            # Create figure
            fig = go.Figure(data=data, layout=layout)



            # Write image to file
            fig.write_image(f"{mainpath}/{study}/analysis/knockouts/{experiment_name}_{ranking}_plot.png")


load()

# for index, row in df.iterrows():
#     print(row)
#     filtered_values = df[
#                         (df['run'] >= row['run'])
#                         & (df['gen'] == row['gen'])
#                         & (df['ranking'] == row['ranking'])
#                         & (df['individual_id'] == row['individual_id'])
#                         & (df['knockout'] == 'o')]
#
#     print('lalal', filtered_values)
# print(row['c1'], row['c2'])