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
import numpy as np
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEnvconditions
from ast import literal_eval
from matplotlib.cm import ScalarMappable
import itertools

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
runs = list(range(1, int(args.runs) + 1))
generations = list(map(int, args.generations.split(',')))
comparison = args.comparison
mainpath = args.mainpath

experiments = experiments_name
inner_metrics = ['median', 'max']
include_max = True
merge_lines = True
gens_boxes = generations
path = f'/storage/{mainpath}/{study}'


clrs = ['#0066CC',
        '#663300',
        '#7855fb'
        ]

measures = {
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
    'extensiveness_prop': ['Extensiveness prop', 0.4, 0.8],
    'width': ['Width', 0, 1],
    'height': ['Height', 0, 1],
    'coverage': ['Coverage', 0, 1],
    'proportion': ['Proportion', 0, 1],
    'symmetry': ['Symmetry', 0, 1],
}

env_conditions = {}


async def main() -> None:
    if not os.path.exists(f'{path}/analysisnovel/{comparison}'):
        os.makedirs(f'{path}/analysisnovel/{comparison}')

    if not os.path.exists(f'{path}/analysisnovel/{comparison}/heatmaps'):
        os.makedirs(f'{path}/analysisnovel/{comparison}/heatmaps')

    db = open_async_database_sqlite(f'/storage/{mainpath}/{study}/{experiments[0]}/run_{runs[0]}')
    async with AsyncSession(db) as session:
        rows = ((await session.execute(select(DbEnvconditions).order_by(DbEnvconditions.id))).all())
        for c_row in rows:
            env_conditions[c_row[0].id] = "_".join(literal_eval(c_row[0].conditions))

    plots()


def plots():

    df = pandas.read_csv(f'{path}/analysisnovel/all_df.csv')
    plot(df)


def plot(df):

    print('plotting ...')

    for pair in list(itertools.combinations(measures.keys(), 2)):
        measure1 = pair[0]
        measure2 = pair[1]

        for env in env_conditions:

            if env == 1:
                intensity = 'speed_y'
            else:
                intensity = 'speed_x'

            df_env = df[(df['env_conditions_id'] == env)]

            font = {'font.size': 10}
            plt.rcParams.update(font)
            fig, ax = plt.subplots()

            g = sb.FacetGrid(df_env, col='experiment')
            g.map(plt.tricontourf, measure1, measure2, intensity, cmap=plt.cm.plasma)#, vmin = overall_min, vmax = overall_max)
            plt.colorbar(ScalarMappable(cmap=plt.cm.plasma))

            plt.savefig(f'{path}/analysisnovel/{comparison}/heatmaps/{measure1}_{measure2}_{intensity}.png', bbox_inches='tight')
            plt.clf()
            plt.close(fig)

        print(f'plotted!')


if __name__ == "__main__":
    asyncio.run(main())



