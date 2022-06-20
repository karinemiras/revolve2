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
                #
                # df_inner_group = df.groupby(['experiment', 'run', 'generation']).agg(
                #     steps=('steps', max),
                #     total_success=('total_success', max),
                #     rewards=('rewards', sum)
                # ).reset_index()


                print(df_inner_group)



args = Config()._get_params()
study = 'default_study'
experiments = ['diversity']
runs = [1]
analysis = Analysis(args, study, experiments, runs)
analysis.plot_lines()



