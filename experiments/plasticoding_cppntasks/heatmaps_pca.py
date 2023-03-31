import argparse

import pandas as pd
from sqlalchemy.ext.asyncio.session import AsyncSession
import pandas
import matplotlib.pyplot as plt
import seaborn as sb
from statannot import add_stat_annotation
import pprint
import sys
import os
import asyncio
import math
import numpy as np
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEnvconditions
from ast import literal_eval
from matplotlib.cm import ScalarMappable
import itertools
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument("study")
parser.add_argument("experiments")
parser.add_argument("runs")
parser.add_argument("generations")
parser.add_argument("comparison")
parser.add_argument("mainpath")
parser.add_argument("analysis")
args = parser.parse_args()

study = args.study
experiments_name = args.experiments.split(',')
runs = list(range(1, int(args.runs) + 1))
generations = list(map(int, args.generations.split(',')))
comparison = args.comparison
mainpath = args.mainpath

experiments = experiments_name
inner_metrics = ['median', 'max']
include_max = True
merge_lines = True
gens_boxes = generations
path = f'/storage/{mainpath}/{study}'

clrs = ['#0066CC',
        '#663300',
        '#7855fb'
        ]

measures = {
 #   'modules_count': ['Modules count', 0, 1],
 #   'hinge_count': ['Hinge count', 0, 1],
 #   'brick_count': ['Brick count', 0, 1],
    'hinge_prop': ['Hinge prop', 0, 1],
    'hinge_ratio': ['Hinge ratio', 0, 1],
    'brick_prop': ['Brick prop', 0, 1],
  #  'branching_count': ['Branching count', 0, 1],
    'branching_prop': ['Branching prop', 0, 1],
 #   'extremities': ['Extremities', 0, 1],
 #   'extensiveness': ['Extensiveness', 0, 1],
    'extremities_prop': ['Extremities prop', 0, 1],
    'extensiveness_prop': ['Extensiveness prop', 0.4, 0.8],
   # 'width': ['Width', 0, 1],
  #  'height': ['Height', 0, 1],
    'coverage': ['Coverage', 0, 1],
    'proportion': ['Proportion', 0, 1],
    'symmetry': ['Symmetry', 0, 1],
}

env_conditions = {}


async def main() -> None:
    if not os.path.exists(f'{path}/analysisnovel/{comparison}'):
        os.makedirs(f'{path}/analysisnovel/{comparison}')

    if not os.path.exists(f'{path}/analysisnovel/{comparison}/heatmaps'):
        os.makedirs(f'{path}/analysisnovel/{comparison}/heatmaps')

    db = open_async_database_sqlite(f'/storage/{mainpath}/{study}/{experiments[0]}/run_{runs[0]}')
    async with AsyncSession(db) as session:
        rows = ((await session.execute(select(DbEnvconditions).order_by(DbEnvconditions.id))).all())
        for c_row in rows:
            env_conditions[c_row[0].id] = "_".join(literal_eval(c_row[0].conditions))

    plots()


def plots():

    df = pandas.read_csv(f'{path}/analysisnovel/all_df.csv')

    plot_avg(df)
   # plot(df)

    # plot_runs(df)
    # plot_avg_runs(df)

def plot(df):

    print('plotting ...')

    n_components = 2

    for env in env_conditions:

        if comparison == 'forthright':
            if env == 1:
                intensity = 'speed_y'
            else:
                intensity = 'speed_x'
        else:
            intensity = 'speed_y'

        df_env = df[(df['env_conditions_id'] == env) &
                    ((df['experiment'] == experiments[0]) |
                    (df['experiment'] == experiments[1]) |
                    (df['experiment'] == experiments[2]) )
                    ]

        df_env = df_env[list(measures.keys()) + ['experiment', 'speed_y', 'speed_x']]
        df_env.reset_index(drop=True, inplace=True)

        if comparison == 'backforth' and env == 2:
            df_env['speed_y'] = df_env['speed_y']*-1

        df_env_filt = df_env[list(measures.keys())]

        pca = PCA(n_components=n_components)
        res_pca = pca.fit_transform(df_env_filt)
        res_pca = pd.DataFrame(res_pca)

        df_pca = df_env.join(res_pca)

        for pair in list(itertools.combinations(range(0, n_components), 2)):
            measure1 = pair[0]
            measure2 = pair[1]

            font = {'font.size': 10}
            plt.rcParams.update(font)
            fig, ax = plt.subplots()

            g = sb.FacetGrid(df_pca, col='experiment')

            g.map(plt.tricontourf, measure1, measure2, intensity, cmap=plt.cm.plasma)#, vmin = overall_min, vmax = overall_max)
            plt.colorbar(ScalarMappable(cmap=plt.cm.plasma))

            plt.savefig(f'{path}/analysisnovel/{comparison}/heatmaps/c{measure1}_c{measure2}_{env}_{intensity}.png', bbox_inches='tight')
            plt.clf()
            plt.close(fig)

    print(f'plotted!')


def plot_avg(df):

    print('plotting ...')
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    n_components = 2

    dff = df[   (df['experiment'] == experiments[0]) |
                (df['experiment'] == experiments[1]) |
                (df['experiment'] == experiments[2])
                ]

    df_env1 = dff[(dff['env_conditions_id'] == 1)]
    df_env1 = df_env1.rename(columns={'speed_x': 'speed_x_1',
                                      'speed_y': 'speed_y_1',
                                      'hinge_prop': 'hinge_prop_1',
                                      'hinge_ratio': 'hinge_ratio_1',
                                      'brick_prop': 'brick_prop_1',
                                       'branching_prop':  'branching_prop_1',
                                      'extremities_prop': 'extremities_prop_1',
                                      'extensiveness_prop':  'extensiveness_prop_1',
                                      'coverage': 'coverage_1',
                                      'proportion': 'proportion_1',
                                      'symmetry':  'symmetry_1'
                                      })

    df_env2 = df[(df['env_conditions_id'] == 2)]
    df_env2 = df_env2.rename(columns={'speed_x': 'speed_x_2',
                                      'speed_y': 'speed_y_2',
                                      'hinge_prop': 'hinge_prop_2',
                                      'hinge_ratio': 'hinge_ratio_2',
                                      'brick_prop': 'brick_prop_2',
                                      'branching_prop': 'branching_prop_2',
                                      'extremities_prop': 'extremities_prop_2',
                                      'extensiveness_prop': 'extensiveness_prop_2',
                                      'coverage': 'coverage_2',
                                      'proportion': 'proportion_2'
                                      })

    df_env = pd.merge(df_env1, df_env2, how="inner", on=['experiment', 'run', 'generation_index', 'individual_id'])
    df_env["speed_y_2"] = df_env["speed_y_2"]*-1

    if comparison == 'forthright':
        df_env['avgspeed'] = (df_env['speed_y_1'] + df_env['speed_x_2']) / 2
    else:
        df_env['avgspeed'] = (df_env['speed_y_1'] + df_env['speed_y_2']) / 2

    df_env['hinge_prop'] = (df_env['hinge_prop_1'] + df_env['hinge_prop_2']) / 2
    df_env['hinge_ratio'] = (df_env['hinge_ratio_1'] + df_env['hinge_ratio_2']) / 2
    df_env['brick_prop'] = (df_env['brick_prop_1'] + df_env['brick_prop_2']) / 2
    df_env['branching_prop'] = (df_env['branching_prop_1'] + df_env['branching_prop_2']) / 2
    df_env['extremities_prop'] = (df_env['extremities_prop_1'] + df_env['extremities_prop_2']) / 2
    df_env['extensiveness_prop'] = (df_env['extensiveness_prop_1'] + df_env['extensiveness_prop_2']) / 2
    df_env['coverage'] = (df_env['coverage_1'] + df_env['coverage_2']) / 2
    df_env['proportion'] = (df_env['proportion_1'] + df_env['proportion_2']) / 2

    df_env['experiment'] = np.select(
        [df_env['experiment'] == 'novfullplasticforthright',
         df_env['experiment'] == 'novnonplasticforthright',
         df_env['experiment'] == 'novplasticforthright',
         ],
        [
         '3',
         '1',
         '2',
         ], None)
    df_env = df_env.sort_values(by=['experiment'])
    df_env = df_env[list(measures.keys()) + ['experiment', 'avgspeed']]
    df_env.reset_index(drop=True, inplace=True)

    pprint.pprint(df_env)

    df_env_filt = df_env[list(measures.keys())]

    pca = PCA(n_components=n_components)
    res_pca = pca.fit_transform(df_env_filt)
    res_pca = pd.DataFrame(res_pca)

    df_pca = df_env.join(res_pca)
    print('---')
    print(df_pca)

    for pair in list(itertools.combinations(range(0, n_components), 2)):
        measure1 = pair[0]
        measure2 = pair[1]

        font = {'font.size': 10}
        plt.rcParams.update(font)
        fig, ax = plt.subplots()

        g = sb.FacetGrid(df_pca, col='experiment', legend_out=True)

        # PS: tricontourf is buggy, when most values are the same, paints with max color even if values are all zero
        g.map(plt.tricontourf, measure1, measure2, 'avgspeed', cmap=plt.cm.plasma)#, vmin = overall_min, vmax = overall_max)
       # plt.colorbar(ScalarMappable(cmap=plt.cm.plasma),)

        plt.savefig(f'{path}/analysisnovel/{comparison}/heatmaps/c{measure1}_c{measure2}_avgspeed_3.png', bbox_inches='tight')

        plt.clf()
        plt.close(fig)

    print(f'plotted!')


def plot_runs(df):

    print('plotting ...')

    n_components = 2

    for run in runs:

        for env in env_conditions:

            if env == 1:
                intensity = 'speed_y'
            else:
                intensity = 'speed_x'

            df_env = df[(df['env_conditions_id'] == env) & (df['run'] == run)]
            df_env = df_env[list(measures.keys()) + ['experiment', 'speed_y', 'speed_x']]
            df_env.reset_index(drop=True, inplace=True)

            df_env_filt = df_env[list(measures.keys())]

            pca = PCA(n_components=n_components)
            res_pca = pca.fit_transform(df_env_filt)
            res_pca = pd.DataFrame(res_pca)

            df_pca = df_env.join(res_pca)

            for pair in list(itertools.combinations(range(0, n_components), 2)):
                measure1 = pair[0]
                measure2 = pair[1]

                font = {'font.size': 10}
                plt.rcParams.update(font)
                fig, ax = plt.subplots()

                g = sb.FacetGrid(df_pca, col='experiment')
                g.map(plt.tricontourf, measure1, measure2, intensity, cmap=plt.cm.plasma)#, vmin = overall_min, vmax = overall_max)
                plt.colorbar(ScalarMappable(cmap=plt.cm.plasma))

                plt.savefig(f'{path}/analysisnovel/{comparison}/heatmaps/{measure1}_{measure2}_{run}_{intensity}.png', bbox_inches='tight')
                plt.clf()
                plt.close(fig)

def plot_avg_runs(df):

    print('plotting ...')

    for run in runs:
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.max_columns', 100)
        n_components = 2

        df_env1 = df[(df['env_conditions_id'] == 1) & (df['run'] == run)]

        df_env1 = df_env1.drop(columns=['speed_x'])
        df_env2 = df[(df['env_conditions_id'] == 2)]
        df_env2 = df_env2[['experiment', 'run', 'individual_id', 'generation_index', 'env_conditions_id', 'speed_x']]

        df_env = pd.merge(df_env1, df_env2, how="inner", on=['experiment', 'run', 'generation_index', 'individual_id'])
        df_env['avgspeed'] = (df_env['speed_y'] + df_env['speed_x']) / 2

        df_env = df_env[list(measures.keys()) + ['experiment', 'avgspeed']]
        df_env.reset_index(drop=True, inplace=True)

        df_env_filt = df_env[list(measures.keys())]

        pca = PCA(n_components=n_components)
        res_pca = pca.fit_transform(df_env_filt)
        res_pca = pd.DataFrame(res_pca)

        df_pca = df_env.join(res_pca)

        print(df_pca)

        for pair in list(itertools.combinations(range(0, n_components), 2)):
            measure1 = pair[0]
            measure2 = pair[1]

            font = {'font.size': 10}
            plt.rcParams.update(font)
            fig, ax = plt.subplots()

            g = sb.FacetGrid(df_pca, col='experiment')
            g.map(plt.tricontourf, measure1, measure2, 'avgspeed',
                  cmap=plt.cm.plasma)  # , vmin = overall_min, vmax = overall_max)
            plt.colorbar(ScalarMappable(cmap=plt.cm.plasma))

            plt.savefig(f'{path}/analysisnovel/{comparison}/heatmaps/{measure1}_{measure2}_{run}_avgspeed.png',
                        bbox_inches='tight')
            plt.clf()
            plt.close(fig)

    print(f'plotted!')

if __name__ == "__main__":
    asyncio.run(main())



