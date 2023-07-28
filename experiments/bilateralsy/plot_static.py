import argparse
import pandas
import matplotlib.pyplot as plt
import seaborn as sb
from statannot import add_stat_annotation
import pprint
import os
import sys
from scipy import stats
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("study")
parser.add_argument("experiments")
parser.add_argument("runs")
parser.add_argument("generations")
parser.add_argument("comparison")
parser.add_argument("mainpath")
args = parser.parse_args()

study = args.study
experiments_name = args.experiments.split(',')
#runs = list(map(int, args.runs.split(',')))
generations = list(map(int, args.generations.split(',')))
comparison = args.comparison
mainpath = args.mainpath

experiments = experiments_name
inner_metrics = ['mean', 'max']
include_max = False
merge_lines = True
by_run = True
gens_boxes = generations
clrs = ['#009900',
        '#EE8610']
path = f'{mainpath}/{study}'

measures = {
    'pop_diversity': ['Diversity', 0, 1],
    'dominated_quality_youth': ['Dominated individuals', 0, 1],
    'fullydominated_quality_youth': ['Fully dominated individuals', 0, 1],
    'seasonal_dominated': ['Seasonal Dominated', 0, 1],
    'age': ['Age', 0, 1],
    'speed_y': ['Speed (cm/s)', 0, 1],
    'relative_speed_y': ['Relative speed (cm/s)', 0, 1],
    'displacement': ['Total displacement (m)', 0, 1],
    'average_z': ['Z', 0, 1],
    'head_balance': ['Balance', 0, 1],
    'modules_count': ['Modules count', 0, 1],
    'hinge_count': ['Hinge count', 0, 1],
    'brick_count': ['Brick count', 0, 1],
    'hinge_prop': ['Hinge prop', 0, 1],
    'hinge_ratio': ['Hinge ratio', 0, 1],
    'brick_prop': ['Brick prop', 0, 1],
    'branching_count': ['Branching count', 0, 1],
    'branching_prop': ['Branching prop', 0, 1],
    'extremities': ['Extremities', 0, 1],
    'extensiveness': ['Extensiveness', 0, 1],
    'extremities_prop': ['Extremities prop', 0, 1],
    'extensiveness_prop': ['Extensiveness prop', 0, 1],
    'width': ['Width', 0, 1],
    'height': ['Height', 0, 1],
    'coverage': ['Coverage', 0, 1],
    'proportion': ['Proportion', 0, 1],
    'symmetry': ['Symmetry', 0, 1]}


def plots():
    if not os.path.exists(f'{path}/analysis/{comparison}'):
        os.makedirs(f'{path}/analysis/{comparison}')

    df_inner = pandas.read_csv(f'{path}/analysis/{comparison}/df_inner.csv')
    df_outer = pandas.read_csv(f'{path}/analysis/{comparison}/df_outer.csv')
    #all_df = pandas.read_csv(f'{path}/analysis/{comparison}/all_df.csv')

 #   plot_lines(df_outer)
    plot_boxes(df_inner)
   # plot_corr(all_df)
   

def plot_lines(df_outer):

    print('plotting lines...')

    for measure in measures.keys():

        font = {'font.size': 20}
        plt.rcParams.update(font)
        fig, ax = plt.subplots()

        plt.xlabel('')
        plt.ylabel(f'{measures[measure][0]}')
        for idx_experiment, experiment in enumerate(experiments):
            data = df_outer[(df_outer['experiment'] == experiment)]

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

            # if measures[measure][1] != -math.inf and measures[measure][2] != -math.inf:
            #     ax.set_ylim(measures[measure][1], measures[measure][2])

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),  fancybox=True, shadow=True, ncol=5, fontsize=10)
            if not merge_lines:
                plt.savefig(f'{path}/analysis/{comparison}/line_{experiment}_{measure}.png', bbox_inches='tight')
                plt.clf()
                plt.close(fig)
                plt.rcParams.update(font)
                fig, ax = plt.subplots()

        if merge_lines:
            plt.savefig(f'{path}/analysis/{comparison}/line_{measure}.png', bbox_inches='tight')
            plt.clf()
            plt.close(fig)

    print('plotted lines!')


def plot_boxes(df_inner):
    print('plotting boxes...')

    for gen_boxes in gens_boxes:

        df_inner2 = df_inner[(df_inner['generation_index'] == gen_boxes)]
        plt.clf()
        tests_combinations = [('bilateralbetter', 'bilateralworse')]
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
                                        test='Mann-Whitney',
                                        text_format = 'star', fontsize = 'xx-large', loc = 'inside',
                                        verbose=1)
            except Exception as error:
                print(error)

            # if measures[measure][1] != -math.inf and measures[measure][2] != -math.inf:
            #     plot.set_ylim(measures[measure][1], measures[measure][2])
            plt.xlabel('')
            plt.ylabel(f'{measures[measure][0]}')
            plot.get_figure().savefig(f'{path}/analysis/{comparison}/box_{measure}_{gen_boxes}.png', bbox_inches='tight')
            plt.clf()
            plt.close()

    print('plotted boxes!')


def plot_corr(all_df):

    def corrfunc(x, y, **kws):

        if kws['label'] == 'bilateral':
            xpos = 0.1
            ypos = 0.1
        else:
            xpos = 0.1
            ypos = .3

        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),
                    xy=(xpos, ypos), xycoords=ax.transAxes, fontsize=10)

    font = {'font.size': 20}
    plt.rcParams.update(font)

    measures_names = [
        'speed_y', 'modules_count', 'hinge_prop', 'brick_prop',
        'branching_prop', 'extremities_prop',
        'extensiveness_prop', 'coverage', 'proportion', 'symmetry']

    #sample = all_df[(all_df['generation_index'] >= 140)]
    sample = all_df[(all_df['generation_index'] >= 50)]
    prop = int(len(sample)*0.1)
    sample = sample.sample(n=prop)

    suball = sample.filter(items=measures_names+['experiment'])

    g = sb.PairGrid(suball, hue='experiment')
    g.map_upper(plt.scatter, s=10)
    g.map_diag(sb.distplot, kde=False)
    g.map_lower(sb.kdeplot, cmap="Blues_d")
    g.map_lower(corrfunc)

    plt.savefig(f'{path}/analysis/{comparison}/corr_last100_0.1.png', bbox_inches='tight')
    plt.clf()
    plt.close()


plots()




