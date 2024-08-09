"""
Visualize and run a modular robot using Mujoco.
"""

from pyrr import Quaternion, Vector3
import argparse
from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor

from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizer, DbEnvconditions
from genotype import GenotypeSerializer, develop_knockout
from optimizer import DbOptimizerState
import sys
from revolve2.core.modular_robot.render.render import Render
from revolve2.core.modular_robot import Measure
from revolve2.core.database.serializers import DbFloat
import pprint
import numpy as np
import os
from ast import literal_eval
import math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
from scipy import stats
import inspect
import statsmodels.api as sm
from scipy.stats import f_oneway
from scipy.stats import mannwhitneyu


class Simulator:
    _controller: ActorController

    async def simulate(self) -> None:

        parser = argparse.ArgumentParser()
        parser.add_argument("study")
        parser.add_argument("experiments")
        parser.add_argument("tfs")
        parser.add_argument("watchruns")
        parser.add_argument("generations")
        parser.add_argument("mainpath")

        args = parser.parse_args()

        self.study = args.study
        self.experiments_name = args.experiments.split(',')
        self.runs = args.watchruns.split(',')
        self.mainpath = args.mainpath

        path = f'{self.mainpath}/{self.study}/analysis/geno/'
        if not os.path.exists(path):
            os.makedirs(path)

        self.pfile = f'{self.mainpath}/{self.study}/analysis/geno/genosize.csv'
        header = ['experiment_name', 'run', 'gen', 'individual_id', 'geno_size', 'disp_y' ]
        with open(self.pfile, 'w') as file:
            file.write(','.join(map(str, header)))
            file.write('\n')

        for ids, experiment_name in enumerate(self.experiments_name):
            print('\n', experiment_name)
            for run in self.runs:
                print('run: ', run)

                path = f'{self.mainpath}/{self.study}'

                fpath = f'{path}/{experiment_name}/run_{run}'
                db = open_async_database_sqlite(fpath)

                await self.recover(db, experiment_name, run)

    async def recover(self, db, experiment_name, run):
        async with AsyncSession(db) as session:

            rows = ((await session.execute(select(DbEnvconditions))).all())
            env_conditions = {}
            for c_row in rows:
                env_conditions[c_row[0].id] = literal_eval(c_row[0].conditions)

            query = select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat) \
                .filter((DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                        & (DbEAOptimizerGeneration.env_conditions_id == DbEAOptimizerIndividual.env_conditions_id)
                        & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                        )

            rows = ((await session.execute(query)).all())

            for idx, r in enumerate(rows):

                disp_y = r.DbFloat.disp_y

                try:
                    genotype = (
                        await GenotypeSerializer.from_database(
                            session, [r.DbEAOptimizerIndividual.genotype_id]
                        )
                    )[0]
                    geno_size = len(genotype.body.genotype)
                except (SyntaxError, TypeError) as e:
                    geno_size = -1
                    print(f"Error evaluating serialized genome:{r.DbEAOptimizerIndividual.genotype_id}, {e}")

                data_part1 = [experiment_name, run, r.DbEAOptimizerGeneration.generation_index,
                              r.DbEAOptimizerGeneration.individual_id , geno_size, disp_y]

                with open(self.pfile, 'a') as file:
                    file.write(','.join(map(str, data_part1)))
                    file.write('\n')


async def main() -> None:
    sim = Simulator()
    await sim.simulate()


def analyze():
    parser = argparse.ArgumentParser()
    parser.add_argument("study")
    parser.add_argument("experiments")
    parser.add_argument("tfs")
    parser.add_argument("watchruns")
    parser.add_argument("generations")
    parser.add_argument("mainpath")

    args = parser.parse_args()

    study = args.study
    experiments_name = args.experiments.split(',')
    runs = args.watchruns.split(',')
    mainpath = args.mainpath

    clrs = ['#009900',
            '#EE8610',
            '#434699',
            '#95fc7a',
            '#221210',
            '#87ac65']

    def groupby(data, measures, metric, keys):
        expr = {x: metric for x in measures}
        df_inner_group = data.groupby(keys).agg(expr).reset_index()
        df_inner_group = df_inner_group.rename(mapper=renamer, axis='columns')
        return df_inner_group

    def renamer(col):
        if col not in keys:
            if inspect.ismethod(metric):
                sulfix = metric.__name__
            else:
                sulfix = metric
            return col + '_' + sulfix
        else:
            return col

    def q25(x):
        return x.quantile(0.25)

    def q75(x):
        return x.quantile(0.75)

    origin_file = f'{mainpath}/{study}/analysis/geno/genosize.csv'
    df = pd.read_csv(origin_file)

    metric = 'mean'
    keys = ['experiment_name', 'run', 'gen']
    df_inner = groupby(df, ['geno_size'], metric, keys)

    keys = ['experiment_name', 'gen']
    metric = 'median'
    aggregation_dict = {
        'geno_size_mean': ['median', q25, q75],
    }

    df_aggregated = df_inner.groupby(keys).agg(aggregation_dict)
    df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]
    df_aggregated = df_aggregated.reset_index()

    font = {'font.size': 20}
    plt.rcParams.update(font)
    fig, ax = plt.subplots()
    plt.xlabel('')

    for idx_experiment, experiment in enumerate(experiments_name):

        data = df_aggregated[(df_aggregated['experiment_name'] == experiment)]
        ax.plot(data['gen'], data[f'geno_size_mean_median'], label=f'{experiment}_mean', c=clrs[idx_experiment])
        ax.fill_between(data['gen'],
                        data[f'geno_size_mean_q25'],
                        data[f'geno_size_mean_q75'], alpha=0.3, facecolor=clrs[idx_experiment])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5, fontsize=10)
    plt.savefig(f'{mainpath}/{study}/analysis/geno/geno_size_{experiment}.png', bbox_inches='tight')
    plt.clf()
    plt.close(fig)
    plt.rcParams.update(font)

    with open(f'{mainpath}/{study}/analysis/geno/geno_size_statistics.txt', 'w') as f:
        f.write("")

    for idx_experiment, experiment in enumerate(experiments_name):
        correlations = []
        fitness = []
        genotypes = []
        for run in runs:

            df_filt = df[(df['experiment_name'] == experiment) & (df['run'] == int(run)) & (df['geno_size'] > 0)]

            df_filt = df_filt.dropna(subset=['disp_y'])
            df_filt = df_filt[np.isfinite(df_filt['disp_y'])]

            correlation, p_value = pearsonr(df_filt['geno_size'], df_filt['disp_y'])

            with open(f'{mainpath}/{study}/analysis/geno/geno_size_statistics.txt', 'a') as f:
                f.write(f"\n run {run} pearson: {correlation}  "
                        f"P-value: {p_value:.4f} "
                        f"avggeno {np.mean(np.array(df_filt['geno_size']))} "
                        f"avgfit {np.mean(np.array(df_filt['disp_y']))}")

            if p_value < 0.004:
                correlations.append(correlation)
                fitness.append(np.mean(np.array(df_filt['disp_y'])))
                genotypes.append(np.mean(np.array(df_filt['geno_size'])))

            # Y = df_filt['disp_y']
            # X = df_filt[['geno_size']]
            # X = sm.add_constant(X)
            # model = sm.OLS(Y, X).fit()
            # print(model.summary())

        with open(f'{mainpath}/{study}/analysis/geno/geno_size_statistics.txt', 'a') as f:
            correlations = np.array(correlations)
            fitness = np.array(fitness)
            genotypes = np.array(genotypes)

            min_value = np.min(correlations)
            max_value = np.max(correlations)
            median_value = np.median(correlations)
            p25_value = np.percentile(correlations, 25)
            p75_value = np.percentile(correlations, 75)

            f.write(f"\n\n {experiment}")
            f.write(f"\n total_corr   min {min_value}  p25 {p25_value} median {median_value}  p75 {p75_value} max {max_value} \n")

            #sorted_indices = np.argsort(correlations)
            sorted_indices = np.argsort(genotypes)
            sorted_fitness = fitness[sorted_indices]

            num_parts = 4
            part_size = len(sorted_fitness) // num_parts
            remainder = len(sorted_fitness) % num_parts  # Calculate remainder

            fitness_parts = []

            start_idx = 0
            for i in range(num_parts):

                if i < remainder:
                    part = sorted_fitness[start_idx:start_idx + part_size + 1]

                    f.write(f" \nmean genosize qt{i+1} {np.median(genotypes[sorted_indices][start_idx:start_idx + part_size + 1])}")

                    if i == 0:
                        f.write(f"\n runs of qt1 {sorted_indices[start_idx:start_idx + part_size + 1]}")

                    start_idx += part_size + 1

                else:
                    part = sorted_fitness[start_idx:start_idx + part_size]
                    f.write(f" \nmean genosize qt{i} {np.median(genotypes[sorted_indices][start_idx:start_idx + part_size])}")
                    start_idx += part_size

                fitness_parts.append(part)

            f_statistic, p_value = f_oneway(*fitness_parts)

            # Create boxplot
            plt.figure(figsize=(10, 6))
            plt.boxplot(fitness_parts, patch_artist=True, showmeans=True)
            plt.title(' ')
            plt.xlabel('Parts')
            plt.ylabel('Fitness')
            plt.xticks([1, 2, 3, 4], ['Part 1', 'Part 2', 'Part 3', 'Part 4'])
            plt.grid(True)

            plt.text(0.5, 1.1, f'ANOVA Results:\nF-statistic = {f_statistic:.2f}\nP-value = {p_value:.4f}',
                     horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='lightgray', alpha=0.5))

            significance_level = 0.05
            for i in range(num_parts - 1):
                u_statistic, pvalue = mannwhitneyu(fitness_parts[i], fitness_parts[i + 1], alternative='two-sided')
                if pvalue < significance_level:
                    plt.text(i + 1.5, np.max(np.concatenate((fitness_parts[i], fitness_parts[i + 1]))) + 0.2, pvalue,
                             ha='center', fontsize=12)

            plt.tight_layout()
            plt.savefig(f'{mainpath}/{study}/analysis/geno/geno_fit_{experiment}.png', bbox_inches='tight')

            f.write("\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

    analyze()




