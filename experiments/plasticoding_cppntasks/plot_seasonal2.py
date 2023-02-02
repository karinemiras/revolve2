import argparse
from sqlalchemy.ext.asyncio.session import AsyncSession
import pandas
import matplotlib.pyplot as plt
import seaborn as sb
from statannot import add_stat_annotation
import pprint
import sys
import os
import numpy as np
import asyncio
import math
import inspect
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEnvconditions
from ast import literal_eval

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
analysis = args.analysis
experiments = experiments_name
inner_metrics = ['median', 'max']
include_max = False
merge_lines = True
gens_boxes = generations
path = f'/storage/{mainpath}/{study}'


measures = {
     'pop_diversity': ['Diversity', False, 0, 1],
     'backforth_dominated': ['BF Dominated individuals', False, 0, 1],
     'forthright_dominated': ['FR Dominated individuals',  False,0, 1],
     'speed_y': ['Speed (cm/s)',  False,-3.5, 3.5],
     'speed_x': ['Speed (cm/s)',  False,-3.5, 3.5],
     'head_balance': ['Balance', False, 0.7, 1],
     'displacement': ['Displacement',  False,-3.5, 3.5],
     'modules_count': ['Modules count',  False,5, 40],
     'body_changes': ['Body Changes', False, 0, 1],

    'hinge_prop': ['Hinge prop',  True,0.3, 0.7],
    'hinge_ratio': ['Hinge ratio', False, 0, 1],
    'brick_prop': ['Brick prop', False, 0, 1],
    'branching_prop': ['Branching prop', False, 0, 1],
    'extremities_prop': ['Extremities prop',  False,0, 1],
    'extensiveness_prop': ['Extensiveness prop', False, 0, 1],
    'coverage': ['Coverage', False, 0, 1],
    'proportion': ['Proportion',  False,0, 1],
    'symmetry': ['Symmetry', False, 0, 1]
}

env_conditions = {}


async def main() -> None:
    if not os.path.exists(f'{path}/{analysis}/{comparison}'):
        os.makedirs(f'{path}/{analysis}/{comparison}')

    db = open_async_database_sqlite(f'/storage/{mainpath}/{study}/{experiments[-1]}/run_{runs[0]}')
    async with AsyncSession(db) as session:
        rows = ((await session.execute(select(DbEnvconditions).order_by(DbEnvconditions.id))).all())
        for c_row in rows:
            env_conditions[c_row[0].id] = "_".join(literal_eval(c_row[0].conditions))

    plots()


def plots():

    df_inner = pandas.read_csv(f'{path}/{analysis}/df_inner.csv')
    plot_boxes(df_inner)
    if comparison == 'forthright':
        plot_boxes2()


def plot_boxes(df_inner):
    print('plotting boxes...')

    clrs = ['#009900',
            '#009900',
            '#EE8610',
            '#EE8610',
            '#7550ff',
            '#7550ff',
            '#808080']

    for gen_boxes in gens_boxes:

        df_concat = df_inner[(df_inner['generation_index'] == gen_boxes)
                             & ((df_inner['experiment'] == experiments[0]) |
                                (df_inner['experiment'] == experiments[1]) |
                                (df_inner['experiment'] == experiments[2]) |
                                (df_inner['experiment'] == experiments[3])
                                )
                             & (df_inner['run'] <= max(runs))
                             ]

        pandas.set_option('display.max_rows', 1000)

        df_concat["env_conditions_id"] = df_concat["env_conditions_id"].apply(lambda x: str(x))
        df_concat['experiment_env'] = df_concat['experiment'] +'_'+ df_concat['env_conditions_id']

        df_concat['normal_speed'] = np.select(
            [df_concat['experiment_env'].str.contains('onlyforth') & df_concat['experiment_env'].str.contains('1'),
             df_concat['experiment_env'].str.contains('backforth') & df_concat['experiment_env'].str.contains('1'),
             df_concat['experiment_env'].str.contains('backforth') & df_concat['experiment_env'].str.contains('2'),
             df_concat['experiment_env'].str.contains('forthright') & df_concat['experiment_env'].str.contains('1'),
             df_concat['experiment_env'].str.contains('forthright') & df_concat['experiment_env'].str.contains('2'),
             ],
            [df_concat['speed_y_median'],
             df_concat['speed_y_median'],
             df_concat['speed_y_median']*-1,
             df_concat['speed_y_median'],
             df_concat['speed_x_median']], None)

        df_concat['experiment_env'] = np.select(
            [df_concat['experiment_env'] == 'onlyforth_1'],
            ['z_onlyforth'], df_concat['experiment_env'])

        df_concat = df_concat.sort_values(by=['experiment_env'])

        print(df_concat.filter(['experiment', 'env_conditions_id', 'experiment_env','speed_x_median', 'speed_y_median', 'normal_speed']))

        plt.clf()

        if comparison == 'forthright':
            tests_combinations = [('fullplasticforthright_1', 'fullplasticforthright_2'),
                                  ('nonplasticforthright_1', 'nonplasticforthright_2'),
                                  ('plasticforthright_1', 'plasticforthright_2')

                                #   ,('fullplasticforthright_1', 'z_onlyforth')
                                #   , ('fullplasticforthright_2', 'z_onlyforth')
                                #   , ('nonplasticforthright_1', 'z_onlyforth')
                                # , ('nonplasticforthright_2', 'z_onlyforth')
                                # , ('plasticforthright_1', 'z_onlyforth')
                                # , ('plasticforthright_2', 'z_onlyforth')
                                  ]
        else:
            tests_combinations = [('fullplasticbackforth_1', 'fullplasticbackforth_2'),
                                  ('nonplasticbackforth_1', 'nonplasticbackforth_2'),
                                  ('plasticbackforth_1', 'plasticbackforth_2')

                                # , ('fullplasticbackforth_1', 'z_onlyforth')
                                # , ('fullplasticbackforth_2', 'z_onlyforth')
                                # , ('nonplasticbackforth_1', 'z_onlyforth')
                                # , ('nonplasticbackforth_2', 'z_onlyforth')
                                # , ('plasticbackforth_1', 'z_onlyforth')
                                # , ('plasticbackforth_2', 'z_onlyforth')
                                  ]

        sb.set(rc={"axes.titlesize": 23, "axes.labelsize": 23, 'ytick.labelsize': 21, 'xtick.labelsize': 21})
        sb.set_style("whitegrid")

        plot = sb.boxplot(x='experiment_env', y=f'normal_speed', data=df_concat,
                          palette=clrs, width=0.4, showmeans=True, linewidth=2, fliersize=6,
                          meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"})
        plot.tick_params(axis='x', labelrotation=90)

        try:
            if len(tests_combinations) > 0:
                add_stat_annotation(plot, data=df_concat, x='experiment_env', y='normal_speed',
                                    box_pairs=tests_combinations,
                                    comparisons_correction=None,
                                    test='Wilcoxon', text_format='star', fontsize='xx-large', loc='inside',
                                    verbose=1)
        except Exception as error:
            print(error)

        # if measures[measure][1]:
        #     if measures[measure][2] != -math.inf and measures[measure][3] != -math.inf:
        #         plot.set_ylim(measures[measure][2], measures[measure][3])

        plt.xlabel('')
        plt.ylabel('Speed (cm/s)')
        plot.get_figure().savefig(f'{path}/{analysis}/{comparison}/box_normal_speed_{gen_boxes}.png', bbox_inches='tight')
        plt.clf()
        plt.close()

        print(f'plotted boxes!')


def plot_boxes2():
    print('plotting boxes2...')
    df_all = pandas.read_csv(f'{path}/{analysis}/all_df.csv')#,  nrows=200000)
    pandas.set_option('display.max_rows', 100)
    clrs = ['#009900',
            '#EE8610',
            '#7550ff']

    for gen_boxes in gens_boxes:

        df_sub = df_all[(df_all['generation_index'] == gen_boxes)
                             & ((df_all['experiment'] == experiments[0]) |
                                (df_all['experiment'] == experiments[1]) |
                                (df_all['experiment'] == experiments[2])
                                )
                             & (df_all['run'] <= max(runs))]

        df_sub['speed'] = np.select(
            [df_sub['env_conditions_id'] == 1,
             df_sub['env_conditions_id'] == 2],
            [df_sub['speed_y'],
             df_sub['speed_x']], None)

        print(df_sub.filter(['experiment', 'run', 'generation_index', 'genotype_id', 'env_conditions_id', 'speed_y', 'speed_x', 'speed']))

        df_inner = df_sub.groupby([df_sub.experiment, df_sub.run, df_sub.generation_index, df_sub.genotype_id])[['speed']].mean().reset_index()

        print(df_inner)

        df_outer = df_inner.groupby([df_inner.experiment, df_inner.run, df_inner.generation_index])[['speed']].mean().reset_index()

        print(df_outer)

        plt.clf()

        tests_combinations = [(experiments[i], experiments[j]) \
                              for i in range(len(experiments)) for j in range(i + 1, len(experiments))]

        sb.set(rc={"axes.titlesize": 23, "axes.labelsize": 23, 'ytick.labelsize': 21, 'xtick.labelsize': 21})
        sb.set_style("whitegrid")

        plot = sb.boxplot(x='experiment', y=f'speed', data=df_outer,
                          palette=clrs, width=0.4, showmeans=True, linewidth=2, fliersize=6,
                          meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"})
        plot.tick_params(axis='x', labelrotation=20)

        try:
            if len(tests_combinations) > 0:
                add_stat_annotation(plot, data=df_outer, x='experiment', y='speed',
                                    box_pairs=tests_combinations,
                                    comparisons_correction=None,
                                    test='t-test_ind', text_format='star', fontsize='xx-large', loc='inside',
                                    verbose=1)
        except Exception as error:
            print(error)


        plt.xlabel('')
        plt.ylabel('Overall speed (cm/s)')
        plot.get_figure().savefig(f'{path}/{analysis}/{comparison}/box_overall_speed_{gen_boxes}.png', bbox_inches='tight')
        plt.clf()
        plt.close()


if __name__ == "__main__":
    asyncio.run(main())



