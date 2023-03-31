import argparse
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
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEnvconditions
from ast import literal_eval
import numpy as np

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


# measures = {
## this one ahs to be generated alone, since it is intersting only for one of the exps
#  'body_changes': ['Body Changes', False, 0, 1],
##
# }

if len(experiments) == 3:
    measures = {
         'backforth_dominated': ['BF Dominated individuals', False, 0, 1],
         'forthright_dominated': ['FR Dominated individuals',  False,0, 1],
         'speed_y': ['Speed (cm/s)',  True,-0.3, 3.5],
         'speed_x': ['Speed (cm/s)',  True,-0.3, 3.5],
         'head_balance': ['Balance', False, 0.7, 1],
         'displacement': ['Displacement',  False,-3.5, 3.5],
         'modules_count': ['Modules count',  False,5, 40],
    }

    clrs = ['#009900',
            '#EE8610',
            '#7550ff']

if len(experiments) == 4:

    measures = {
        'pop_diversity': ['Diversity', False, 0, 1],
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


clrs = [
        '#009900',
        '#EE8610',
        '#7550ff',
        '#808080']

if analysis == 'analysisnovel':
    measures['seasonal_novelty'] = ['Seasonal Novelty',  False,0, 1]

env_conditions = {}


async def main() -> None:
    if not os.path.exists(f'{path}/{analysis}/{comparison}'):
        os.makedirs(f'{path}/{analysis}/{comparison}')

    db = open_async_database_sqlite(f'/storage/{mainpath}/{study}/{experiments[0]}/run_{runs[0]}')
    async with AsyncSession(db) as session:
        rows = ((await session.execute(select(DbEnvconditions).order_by(DbEnvconditions.id))).all())
        for c_row in rows:
            env_conditions[c_row[0].id] = "_".join(literal_eval(c_row[0].conditions))

    plots()


def plots():

    df_inner = pandas.read_csv(f'{path}/{analysis}/df_inner.csv')
    df_outer = pandas.read_csv(f'{path}/{analysis}/df_outer.csv')

    plot_lines(df_outer)
    plot_boxes(df_inner)


def plot_lines(df_outer):

    print('plotting lines...')

    df_outer['experiment'] = np.select(
        [df_outer['experiment'] == 'onlyforth'],
        ['z_onlyforth'], df_outer['experiment'])

    if comparison == 'backforth':
        df_outer['speed_y_median_median'] = np.select([df_outer['env_conditions_id'] == 2], [df_outer['speed_y_median_median']*-1], df_outer['speed_y_median_median'])
        df_outer['speed_y_median_q25'] = np.select([df_outer['env_conditions_id'] == 2], [df_outer['speed_y_median_q25'] * -1], df_outer['speed_y_median_q25'])
        df_outer['speed_y_median_q75'] = np.select([df_outer['env_conditions_id'] == 2], [df_outer['speed_y_median_q75'] * -1], df_outer['speed_y_median_q75'])

    for env in env_conditions:
        for measure in measures.keys():

            if len(env_conditions) > 1:
                file_env = '_'+str(env)+'_' # '_'+env_conditions[env]+'_'
            else:
                file_env = '_'

            font = {'font.size': 20}
            plt.rcParams.update(font)
            fig, ax = plt.subplots()

            plt.xlabel('')
            plt.ylabel(f'{measures[measure][0]}')
            for idx_experiment, experiment in enumerate(experiments):
                if experiment == 'z_onlyforth':
                    data = df_outer[(df_outer['experiment'] == 'z_onlyforth')]
                else:
                    data = df_outer[(df_outer['experiment'] == experiment) & (df_outer['env_conditions_id'] == env)]

                ax.plot(data['generation_index'], data[f'{measure}_{inner_metrics[0]}_median'],
                        label=f'{experiment}_{inner_metrics[0]}', c=clrs[idx_experiment])
                ax.fill_between(data['generation_index'],
                                data[f'{measure}_{inner_metrics[0]}_q25'],
                                data[f'{measure}_{inner_metrics[0]}_q75'],
                                alpha=0.3, facecolor=clrs[idx_experiment])

                if include_max:
                    ax.plot(data['generation_index'], data[f'{measure}_{inner_metrics[1]}_median'],
                            'b--', label=f'{experiment}_{inner_metrics[1]}', c=clrs[idx_experiment])
                    ax.fill_between(data['generation_index'],
                                    data[f'{measure}_{inner_metrics[1]}_q25'],
                                    data[f'{measure}_{inner_metrics[1]}_q75'],
                                    alpha=0.3, facecolor=clrs[idx_experiment])

                if measures[measure][1]:
                    if measures[measure][2] != -math.inf and measures[measure][3] != -math.inf:
                        ax.set_ylim(measures[measure][2], measures[measure][3])

                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),  fancybox=True, shadow=True, ncol=5, fontsize=10)
                if not merge_lines:
                    plt.savefig(f'{path}/{analysis}/{comparison}/line_{experiment}{file_env}{measure}.png', bbox_inches='tight')
                    plt.clf()
                    plt.close(fig)
                    plt.rcParams.update(font)
                    fig, ax = plt.subplots()

            if merge_lines:
                plt.savefig(f'{path}/{analysis}/{comparison}/line{file_env}{measure}.png', bbox_inches='tight')
                plt.clf()
                plt.close(fig)

        print(f'plotted lines for {env}!')


def plot_boxes(df_inner):
    print('plotting boxes...')

    df_inner['experiment'] = np.select(
        [df_inner['experiment'] == 'onlyforth'],
        ['z_onlyforth'], df_inner['experiment'])

    if comparison == 'backforth':
        df_inner['speed_y_median'] = np.select([df_inner['env_conditions_id'] == 2], [df_inner['speed_y_median']*-1], df_inner['speed_y_median'])

    for env in env_conditions:

        if len(env_conditions) > 1:
            file_env = '_'+str(env)+'_' #'_'+env_conditions[env]+'_'
        else:
            file_env = '_'

        for gen_boxes in gens_boxes:

            if len(experiments) == 4:
                df_inner2 = df_inner[(df_inner['generation_index'] == gen_boxes)
                                     & (df_inner['run'] <= max(runs))
                                     & ( (( ((df_inner['experiment'] == experiments[0]) ) |
                                          ((df_inner['experiment'] == experiments[1]) ) |
                                          ((df_inner['experiment'] == experiments[2]) )  )
                                         & (df_inner['env_conditions_id'] == env))
                                        |  (df_inner['experiment'] == 'z_onlyforth')
                                        )
                                     ]
            else:
                df_inner2 = df_inner[(df_inner['generation_index'] == gen_boxes)
                                     & ( (df_inner['experiment'] == experiments[0]) |
                                         (df_inner['experiment'] == experiments[1]) |
                                         (df_inner['experiment'] == experiments[2])
                                         )
                                     & (df_inner['run'] <= max(runs))
                                     & (df_inner['env_conditions_id'] == env)]
            df_inner2 = df_inner2.sort_values(by=['experiment'])

            plt.clf()

            tests_combinations = [(experiments[i], experiments[j]) \
                                  for i in range(len(experiments)) for j in range(i+1, len(experiments))]

            for idx_measure, measure in enumerate(measures.keys()):
                sb.set(rc={"axes.titlesize": 23, "axes.labelsize": 23, 'ytick.labelsize': 21, 'xtick.labelsize': 21})
                sb.set_style("whitegrid")

                plot = sb.boxplot(x='experiment', y=f'{measure}_{inner_metrics[0]}', data=df_inner2,
                                  palette=clrs, width=0.4, showmeans=True, linewidth=2, fliersize=6,
                                  meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"})
                plot.tick_params(axis='x', labelrotation=10)

                try:
                    if len(tests_combinations) > 0:
                        add_stat_annotation(plot, data=df_inner2, x='experiment', y=f'{measure}_{inner_metrics[0]}',
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
                plt.ylabel(f'{measures[measure][0]}')
                plot.get_figure().savefig(f'{path}/{analysis}/{comparison}/box{file_env}{measure}_{gen_boxes}.png', bbox_inches='tight')
                plt.clf()
                plt.close()

        print(f'plotted boxes for {env}!')

    # def min_max_outer( df):
    #     if not include_max:
    #         inner_metrics = [inner_metrics[0]]
    #     else:
    #         inner_metrics = inner_metrics
    #     outer_metrics = ['median', 'q25', 'q75']
    #
    #     for measure in measures:
    #         min = 10000000
    #         max = 0
    #         for inner_metric in inner_metrics:
    #             for outer_metric in outer_metrics:
    #                 value = df[f'{measure}_{inner_metric}_{outer_metric}'].max()
    #                 if value > max:
    #                     max = value
    #                 value = df[f'{measure}_{inner_metric}_{outer_metric}'].min()
    #                 if value < min:
    #                     min = value
    #         measures[measure][1] = min
    #         measures[measure][2] = max
    #
    # def min_max_inner( df):
    #     for measure in measures:
    #         min = 10000000
    #         max = 0
    #         value = df[f'{measure}_mean'].max()
    #         if value > max:
    #             max = value
    #         value = df[f'{measure}_mean'].min()
    #         if value < min:
    #             min = value
    #         measures[measure][1] = min*1.05
    #         measures[measure][2] = max*1.05


if __name__ == "__main__":
    asyncio.run(main())



