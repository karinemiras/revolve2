from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizer, DbEnvconditions
from revolve2.core.modular_robot.render.render import Render
from genotype import GenotypeSerializer, develop
from revolve2.core.database.serializers import DbFloat

import os
import sys
import argparse
from ast import literal_eval
import pprint
import numpy as np
import math
import pandas as pd
import inspect
import matplotlib.pyplot as plt


async def main(parser) -> None:
    await collect_data(parser)
    await plot(parser)


async def plot(parser) -> None:
    args = parser.parse_args()

    study = args.study
    mainpath = args.mainpath
    comparison = args.comparison

    data = pd.read_csv(f'/storage/{mainpath}/{study}/analysisspeed/{comparison}/controllers_diff.txt', sep=";")

    keys = ['experiment_name', 'run', 'gen']
    metric = 'median'

    data = data[data['diff'].notnull()]

    def renamer(col):
        if col not in keys:
            if inspect.isfunction(metric):
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

    def q25(x):
        return x.quantile(0.25)

    def q75(x):
        return x.quantile(0.75)

    pprint.pprint(data)

    data_inner = groupby(data, ['diff'], metric, keys)

    keys = ['experiment_name', 'gen']
    metric = 'median'
    measures_inner = ['diff_median']
    df_outer_median = groupby(data_inner, measures_inner, metric, keys)

    metric = q25
    df_outer_q25 = groupby(data_inner, measures_inner, metric, keys)

    metric = q75
    df_outer_q75 = groupby(data_inner, measures_inner, metric, keys)

    df_outer = pd.merge(df_outer_median, df_outer_q25, on=keys)
    df_outer = pd.merge(df_outer, df_outer_q75, on=keys)

    font = {'font.size': 20}
    plt.rcParams.update(font)
    fig, ax = plt.subplots()

    plt.xlabel('Generation')
    plt.ylabel('Brain Changes')

    ax.plot(df_outer['gen'], df_outer['diff_median_median'], c='#0066CC')
    ax.fill_between(df_outer['gen'],
                    df_outer['diff_median_q25'],
                    df_outer['diff_median_q75'],
                    alpha=0.3, facecolor='#0066CC')

    pprint.pprint(df_outer)

    ax.set_ylim(-0.05, 1.5)

    plt.savefig(f'/storage/{mainpath}/{study}/analysisspeed/{comparison}/controllers_diff.png', bbox_inches='tight')
    plt.clf()
    plt.close(fig)


async def collect_data(parser) -> None:

    args = parser.parse_args()

    study = args.study
    experiments_name = args.experiments.split(',')
    runs = list(range(1, int(args.runs)+1))
    generations = list(map(int, args.generations.split(',')))
    mainpath = args.mainpath
    comparison = args.comparison

    with open(f'/storage/{mainpath}/{study}/analysisspeed/{comparison}/controllers_diff.txt', 'w') as f:
        f.write(f'experiment_name;run;gen;diff\n')

    for experiment_name in experiments_name:
        print(experiment_name)
        for run in runs:
            print(' run: ', run)

            db = open_async_database_sqlite(f'/storage/{mainpath}/{study}/{experiment_name}/run_{run}')

            for gen in range(1, generations[0]+1):
                print('  gen: ', gen)

                async with AsyncSession(db) as session:
                    rows = ((await session.execute(select(DbEnvconditions).order_by(DbEnvconditions.id))).all())
                    env_conditions = {}
                    for c_row in rows:
                        env_conditions[c_row[0].id] = literal_eval(c_row[0].conditions)

                    rows = (
                        (await session.execute(select(DbEAOptimizer))).all()
                    )
                    max_modules = rows[0].DbEAOptimizer.max_modules
                    substrate_radius = rows[0].DbEAOptimizer.substrate_radius
                    plastic_body = rows[0].DbEAOptimizer.plastic_body
                    plastic_brain = rows[0].DbEAOptimizer.plastic_brain

                    query = select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat)\
                        .filter(DbEAOptimizerGeneration.generation_index.in_([gen])) \
                                               .filter((DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                                       & (DbEAOptimizerGeneration.env_conditions_id == DbEAOptimizerIndividual.env_conditions_id)
                                                       & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                                                       )

                    query = query.order_by(
                                           DbEAOptimizerGeneration.individual_id.asc(),
                                           DbEAOptimizerGeneration.env_conditions_id.asc())

                    rows = ((await session.execute(query)).all())

                    controllers = []
                    for idx, r in enumerate(rows):
                        genotype = (
                            await GenotypeSerializer.from_database(
                                session, [r.DbEAOptimizerIndividual.genotype_id]
                            )
                        )[0]

                        phenotype, queried_substrate = develop(genotype, genotype.mapping_seed, max_modules, substrate_radius,
                                            env_conditions[r.DbEAOptimizerGeneration.env_conditions_id],
                                                               len(env_conditions), plastic_body, plastic_brain)
                        internal_weights, external_weights = phenotype.make_controller_return()
                        controller = internal_weights + external_weights
                       # pprint.pprint(internal_weights + external_weights)

                        if (idx % 2) == 0:
                            controllers.append([])
                            controllers[-1].append(controller)
                        else:
                            controllers[-1].append(controller)

                    for c in controllers:

                        c[0] = np.array(c[0])
                        c[1] = np.array(c[1])
                        cont_diff = c[0] - c[1]
                        abs_matrix = np.abs(cont_diff)
                        avg = np.nanmean(abs_matrix)

                        # print('c1')
                        # pprint.pprint(c[0])
                        # print('c2')
                        # pprint.pprint(c[1])
                        # print('diff')
                        # pprint.pprint(cont_diff)
                        # print('abs')
                        # pprint.pprint(abs_matrix)
                        # print('avg')
                        # print(avg)

                        with open(f'/storage/{mainpath}/{study}/analysisspeed/{comparison}/controllers_diff.txt', 'a') as f:
                            f.write(f'{experiment_name};{run};{gen};{avg}\n')


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("study")
    parser.add_argument("experiments")
    parser.add_argument("runs")
    parser.add_argument("generations")
    parser.add_argument("mainpath")
    parser.add_argument("comparison")
    asyncio.run(main(parser))

# can be run from root