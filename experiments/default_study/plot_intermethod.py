import argparse

import matplotlib.pyplot as plt
import pandas
from sqlalchemy.future import select
import os
import inspect
import matplotlib.pyplot as plt
import seaborn as sb
from statannot import add_stat_annotation
from itertools import combinations
import pprint
import sys
import math

from revolve2.core.database import open_database_sqlite
from revolve2.core.database.serializers import DbFloat
from revolve2.core.optimization.ea.generic_ea._database import (
    DbEAOptimizerGeneration,
    DbEAOptimizerIndividual
)
from revolve2.core.config import Config


class Analysis:

    def __init__(self, args, study, experiments, runs):
        self.args = args
        self.study = study
        self.experiments = experiments
        self.inner_metrics = ['median', 'max']
        self.runs = runs
        self.include_max = False
        self.merge_lines = True
        self.gens_boxes = [200]
        self.clrs = ['#009900',
                     '#EE8610',
                     '#7550ff',
                     '#876044']
        self.path = f'/storage/karine/{study}'

        self.measures = {
            'pop_diversity': ['Diversity', 0, 1],
            'pool_diversity': ['Pool Diversity', 0, 1],
            'dominated_quality_youth': ['Dominated individuals', 0, 1],
            'fullydominated_quality_youth': ['Fully dominated individuals', 0, 1],
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

    def plots(self):

        df_inner = pandas.read_csv(f'{self.path}/analysis/df_inner.csv')
        df_outer = pandas.read_csv(f'{self.path}/analysis/df_outer.csv')

      #  self.plot_lines(df_outer)
        self.plot_boxes(df_inner)

    def plot_lines(self, df_outer):

        print('plotting lines...')

        #self.min_max_outer(df_outer)
        for measure in self.measures.keys():

            font = {'font.size': 20}
            plt.rcParams.update(font)
            fig, ax = plt.subplots()

            plt.xlabel('')
            plt.ylabel(f'{self.measures[measure][0]}')
            for idx_experiment, experiment in enumerate(self.experiments):
                data = df_outer[df_outer['experiment'] == experiment]
                ax.plot(data['generation_index'], data[f'{measure}_{self.inner_metrics[0]}_median'],
                        label=f'{experiment}_{self.inner_metrics[0]}', c=self.clrs[idx_experiment])
                ax.fill_between(data['generation_index'],
                                data[f'{measure}_{self.inner_metrics[0]}_q25'],
                                data[f'{measure}_{self.inner_metrics[0]}_q75'],
                                alpha=0.3, facecolor=self.clrs[idx_experiment])

                if self.include_max:
                    ax.plot(data['generation_index'], data[f'{measure}_{self.inner_metrics[1]}_median'],
                            'b--', label=f'{experiment}_{self.inner_metrics[1]}', c=self.clrs[idx_experiment])
                    ax.fill_between(data['generation_index'],
                                    data[f'{measure}_{self.inner_metrics[1]}_q25'],
                                    data[f'{measure}_{self.inner_metrics[1]}_q75'],
                                    alpha=0.3, facecolor=self.clrs[idx_experiment])

                # if self.measures[measure][1] != -math.inf and self.measures[measure][2] != -math.inf:
                #     ax.set_ylim(self.measures[measure][1], self.measures[measure][2])

                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),  fancybox=True, shadow=True, ncol=5, fontsize=10)
                if not self.merge_lines:
                    plt.savefig(f'{self.path}/analysis/basic_plots/line_{experiment}_{measure}.png', bbox_inches='tight')
                    plt.clf()
                    plt.close(fig)
                    plt.rcParams.update(font)
                    fig, ax = plt.subplots()

            if self.merge_lines:
                plt.savefig(f'{self.path}/analysis/basic_plots/line_{measure}.png', bbox_inches='tight')
                plt.clf()
                plt.close(fig)

        print('plotted lines!')

    def plot_boxes(self, df_inner):
        print('plotting boxes...')
        for gen_boxes in self.gens_boxes:
            df_inner2 = df_inner[(df_inner['generation_index'] == gen_boxes) & (df_inner['run'] <= max(self.runs))]
            #self.min_max_inner(df_inner)
            plt.clf()

            tests_combinations = [(self.experiments[i], self.experiments[j]) \
                                  for i in range(len(self.experiments)) for j in range(i+1, len(self.experiments))]
            for idx_measure, measure in enumerate(self.measures.keys()):
                sb.set(rc={"axes.titlesize": 23, "axes.labelsize": 23, 'ytick.labelsize': 21, 'xtick.labelsize': 21})
                sb.set_style("whitegrid")

                plot = sb.boxplot(x='experiment', y=f'{measure}_{self.inner_metrics[0]}', data=df_inner2,
                                  palette=self.clrs, width=0.4, showmeans=True, linewidth=2, fliersize=6,
                                  meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"})

                try:
                    if len(tests_combinations) > 0:
                        add_stat_annotation(plot, data=df_inner2, x='experiment', y=f'{measure}_{self.inner_metrics[0]}',
                                            box_pairs=tests_combinations,
                                            comparisons_correction=None,
                                            test='Wilcoxon', text_format='star', fontsize='xx-large', loc='inside',
                                            verbose=1)
                except Exception as error:
                    print(error)

                # if self.measures[measure][1] != -math.inf and self.measures[measure][2] != -math.inf:
                #     plot.set_ylim(self.measures[measure][1], self.measures[measure][2])
                plt.xlabel('')
                plt.ylabel(f'{self.measures[measure][0]}')
                plot.get_figure().savefig(f'{self.path}/analysis/basic_plots/box_{measure}_{gen_boxes}.png', bbox_inches='tight')
                plt.clf()
                plt.close()

        print('plotted boxes!')

    # def min_max_outer(self, df):
    #     if not self.include_max:
    #         inner_metrics = [self.inner_metrics[0]]
    #     else:
    #         inner_metrics = self.inner_metrics
    #     outer_metrics = ['median', 'q25', 'q75']
    #
    #     for measure in self.measures:
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
    #         self.measures[measure][1] = min
    #         self.measures[measure][2] = max
    #
    # def min_max_inner(self, df):
    #     for measure in self.measures:
    #         min = 10000000
    #         max = 0
    #         value = df[f'{measure}_mean'].max()
    #         if value > max:
    #             max = value
    #         value = df[f'{measure}_mean'].min()
    #         if value < min:
    #             min = value
    #         self.measures[measure][1] = min*1.05
    #         self.measures[measure][2] = max*1.05


parser = argparse.ArgumentParser()
parser.add_argument("study")
parser.add_argument("experiments")
parser.add_argument("runs")
args = parser.parse_args()

study = args.study
experiments = [args.experiments]
runs = list(range(1, int(args.runs)+1))
# TODO: break by environment
analysis = Analysis(args, study, experiments, runs)
analysis.plots()


