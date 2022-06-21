import argparse

import matplotlib.pyplot as plt
import pandas
from sqlalchemy.future import select

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

    def plot_lines(self):
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

                # TODO: make these groupings dynamically

                # df_inner_group = df.groupby(['experiment', 'run', 'generation_index']).agg(
                #      coverage_avg=('coverage', 'mean'),
                #      pop_diversity_avg=('pop_diversity', 'mean'),
                #      pool_dominated_individuals_avg=('pool_dominated_individuals', 'mean'),
                #      pool_fulldominated_individuals_avg=('pool_fulldominated_individuals', 'mean'),
                #      age_avg=('age', 'mean'),
                #      displacement_xy_avg=('displacement_xy', 'mean'),
                #      displacement_y_avg=('displacement_y', 'mean'),
                #      relative_displacement_y_avg=('relative_displacement_y', 'mean'),
                #      average_z_avg=('average_z', 'mean'),
                #      head_balance_avg=('head_balance', 'mean'),
                #      modules_count_avg=('modules_count', 'mean'),
                #      hinge_count_avg=('hinge_count', 'mean'),
                #      brick_count_avg=('brick_count', 'mean'),
                #      hinge_prop_avg=('hinge_prop', 'mean'),
                #      brick_prop_avg=('brick_prop', 'mean'),
                #      branching_count_avg=('branching_count', 'mean'),
                #      branching_prop_avg=('branching_prop', 'mean'),
                #      extremities_avg=('extremities', 'mean'),
                #      extensiveness_avg=('extensiveness', 'mean'),
                #      extremities_prop_avg=('extremities_prop', 'mean'),
                #      extensiveness_prop_avg=('extensiveness_prop', 'mean'),
                #      width_avg=('width', 'mean'),
                #      height_avg=('height', 'mean'),
                #      proportion_avg=('proportion', 'mean'),
                #      symmetry_avg=('symmetry', 'mean'),
                #
                #      pool_dominated_individuals_max=('pool_dominated_individuals', 'max'),
                #      pool_fulldominated_individuals_max=('pool_fulldominated_individuals', 'max'),
                #      displacement_y_max=('displacement_y', 'max'),
                #      relative_displacement_y_max=('relative_displacement_y', 'max'),
                #      average_z_max=('average_z', 'max'),
                #      head_balance_max=('head_balance', 'max'),
                #      modules_count_max=('modules_count', 'max'),
                #      hinge_count_max=('hinge_count', 'max'),
                #      brick_count_max=('brick_count', 'max'),
                #      hinge_prop_max=('hinge_prop', 'max'),
                #      brick_prop_max=('brick_prop', 'max'),
                #      branching_count_max=('branching_count', 'max'),
                #      branching_prop_max=('branching_prop', 'max'),
                #      extremities_max=('extremities', 'max'),
                #      extensiveness_max=('extensiveness', 'max'),
                #      extremities_prop_max=('extremities_prop', 'max'),
                #      extensiveness_prop_max=('extensiveness_prop', 'max'),
                #      width_max=('width', 'max'),
                #      height_max=('height', 'max'),
                #      proportion_max=('proportion', 'max'),
                #      symmetry_max=('symmetry', 'max')
                # ).reset_index()
                #
                # print(df_inner_group)

                measures = ['pop_diversity', 'pool_dominated_individuals',
                               'pool_fulldominated_individuals', 'age',
                               'displacement_xy', 'displacement_y', 'relative_displacement_y',
                               'average_z', 'head_balance', 'modules_count', 'hinge_count',
                               'brick_count', 'hinge_prop', 'brick_prop', 'branching_count',
                               'branching_prop', 'extremities', 'extensiveness', 'extremities_prop',
                               'extensiveness_prop', 'width', 'height', 'coverage', 'proportion',
                               'symmetry']
                metric = 'mean'
                expr = {x: metric for x in measures}
                df_inner_group=df_inner_group.groupby(['generation_index']).agg(expr).reset_index()

                def renamer(col):
                    if col in measures:
                        return col+'_'+sulfix
                    else:
                        return col

                df_inner_group = df_inner_group.rename(mapper=renamer, axis='columns')

                print(df_inner_group)

args = Config()._get_params()
study = 'default_study'
experiments = ['diversity']
runs = [1]
analysis = Analysis(args, study, experiments, runs)
analysis.plot_lines()



