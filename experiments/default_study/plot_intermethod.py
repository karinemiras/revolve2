import argparse

import matplotlib.pyplot as plt
import pandas
from sqlalchemy.future import select
import os
import inspect

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
        self.runs = runs
        self.include_max = True

    def consolidate(self):

        path = f'data/{study}/analysis/basic_plots'
        if not os.path.exists(path):
            os.makedirs(path)

        all_df = None
        for experiment in self.experiments:
            for run in self.runs:

                db = open_database_sqlite(f'./data/{study}/{experiment}/run_{run}')

                # read the optimizer data into a pandas dataframe
                df = pandas.read_sql(
                    select(
                        DbEAOptimizerIndividual,
                        DbEAOptimizerGeneration,
                        DbFloat
                    ).filter(
                        (DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                        & (DbFloat.id == DbEAOptimizerIndividual.float_id)

                    ),
                    db,
                )
                df["experiment"] = experiment
                df["run"] = run

                if all_df is None:
                    all_df = df
                else:
                    all_df = pandas.concat([all_df, df], axis=0)

        measures = ['pop_diversity', 'pool_dominated_individuals',
                       'pool_fulldominated_individuals', 'age',
                       'displacement_xy', 'displacement_y', 'relative_displacement_y',
                       'average_z', 'head_balance', 'modules_count', 'hinge_count',
                       'brick_count', 'hinge_prop', 'brick_prop', 'branching_count',
                       'branching_prop', 'extremities', 'extensiveness', 'extremities_prop',
                       'extensiveness_prop', 'width', 'height', 'coverage', 'proportion',
                       'symmetry']
        keys = ['experiment', 'run', 'generation_index']
        inner_metrics = ['mean', 'max']

        def renamer(col):
            if col not in keys:
                if inspect.ismethod(metric):
                    sulfix = metric.__name__
                else:
                    sulfix = metric
                return col + '_' + sulfix
            else:
                return col

        def groupby(data, measures, metric, keys):
            expr = {x: metric for x in measures}
            df_inner_group = data.groupby(keys).agg(expr).reset_index()
            df_inner_group = df_inner_group.rename(mapper=renamer, axis='columns')
            return df_inner_group

        # inner measurements (within runs)

        df_inner = {}
        for metric in inner_metrics:
            df_inner[metric] = groupby(all_df, measures, metric, keys)

        df_inner = pandas.merge(df_inner['mean'], df_inner['max'], on=keys)

        #self.plot_boxes()

        # outer measurements (among runs)

        measures_inner = []
        for measure in measures:
            for metric in inner_metrics:
                measures_inner.append(f'{measure}_{metric}')

        keys = ['experiment', 'generation_index']
        metric = 'median'
        df_outer_median = groupby(df_inner, measures_inner, metric, keys)

        metric = self.q25
        df_outer_q25 = groupby(df_inner, measures_inner, metric, keys)

        metric = self.q75
        df_outer_q75 = groupby(df_inner, measures_inner, metric, keys)

        df_outer = pandas.merge(df_outer_median, df_outer_q25, on=keys)
        df_outer = pandas.merge(df_outer, df_outer_q75, on=keys)

        #self.plot_lines()

    def q25(self, x):
        return x.quantile(0.25)

    def q75(self, x):
        return x.quantile(0.75)


args = Config()._get_params()
study = 'default_study'
experiments = ['diversity']
runs = [1, 18]
# TODO: break by environment
analysis = Analysis(args, study, experiments, runs)
analysis.consolidate()



