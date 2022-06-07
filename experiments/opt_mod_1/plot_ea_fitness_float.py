"""
Plot average, min, and max fitness over generations, using the results of the evolutionary optimizer.
Assumes fitnesses is a floats.
Installed as ``revolve2_plot_ea_fitness_float``.
See ``revolve2_plot_ea_fitness_float --help`` for usage.
"""

import argparse

import matplotlib.pyplot as plt
import pandas
from sqlalchemy.future import select

from revolve2.core.database import open_database_sqlite
from revolve2.core.database.serializers import DbFloat
from revolve2.core.optimization.ea.generic_ea import (
    DbEAOptimizer,
    DbEAOptimizerGeneration,
    DbEAOptimizerIndividual,
    DbIndividualFloat
)


def plot(experiment_name: str, optimizer_id: int) -> None:
    # open the database
 
    db = open_database_sqlite(f'data/{experiment_name}')

    # read the optimizer data into a pandas dataframe
    df = pandas.read_sql(
        select(
            DbEAOptimizer,
            DbEAOptimizerGeneration,
            DbEAOptimizerIndividual,
            DbFloat,
        ).filter(
            (DbEAOptimizer.process_id == optimizer_id)
            & (DbEAOptimizerGeneration.ea_optimizer_id == DbEAOptimizer.id)
            & (DbEAOptimizerIndividual.ea_optimizer_id == DbEAOptimizer.id)
            & (DbEAOptimizerIndividual.ea_optimizer_id == DbIndividualFloat.ea_optimizer_id)
            & (DbEAOptimizerIndividual.individual_id == DbIndividualFloat.individual_id)
            & (DbIndividualFloat.float_id == DbFloat.id)
            & (DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
        ),
        db,
    )

    print(df)

    # calculate max min avg
    describe = (
        df[["generation_index", "displacement_xy"]]
        .groupby(by="generation_index")
        .describe()["displacement_xy"]
    )
    mean = describe[["mean"]].values.squeeze()
    std = describe[["std"]].values.squeeze()

    # plot max min mean, std
    describe[["max", "mean", "min"]].plot()
    plt.fill_between(range(len(mean)), mean - std, mean + std)
    plt.show()

    plt.savefig(f'data/{experiment_name}/figure_1.pdf', dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        required=False,
        default="default_experiment",
        type=str,
        help="The database to plot.",
    )
    parser.add_argument(
        "--optimizer_id",
        required=False,
        default=0,
        type=int,
        help="The id of the ea optimizer to plot."
    )
    args = parser.parse_args()

    plot(args.experiment_name, args.optimizer_id)


if __name__ == "__main__":
    main()
