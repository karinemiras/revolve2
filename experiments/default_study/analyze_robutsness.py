"""
Visualize and run a modular robot using Mujoco.
"""

from pyrr import Quaternion, Vector3
import argparse
from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, Core, Module
from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from  sqlalchemy.sql.expression import func
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizer, DbEnvconditions
from genotype import GenotypeSerializer, develop
import inspect
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy import stats
import seaborn as sb
from statannot import add_stat_annotation
from optimizer import DbOptimizerState
import sys, os, copy
from revolve2.core.modular_robot.render.render import Render
from revolve2.core.modular_robot import Measure
from revolve2.core.database.serializers import DbFloat
import pprint
from random import Random, random
import pandas as pd
import numpy as np
from ast import literal_eval
from revolve2.core.physics.environment_actor_controller import (
    EnvironmentActorController,
)

from revolve2.runners.mujoco import LocalRunner as LocalRunnerM
from revolve2.runners.isaacgym import LocalRunner as LocalRunnerI

from extractstates import *
from body_spider import *

from revolve2.standard_resources import terrains

import statsmodels.stats.api as sms
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
import scipy.stats as st
import math


class Simulator:
    _controller: ActorController

    async def init(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("study")
        parser.add_argument("experiments")
        parser.add_argument("runs")
        parser.add_argument("generations")
        parser.add_argument("mainpath")
        parser.add_argument("simulator")
        parser.add_argument("loop")
        parser.add_argument("body_phenotype")
        parser.add_argument("bisymmetry")
        parser.add_argument("comparison")

        args = parser.parse_args()

        self.study = args.study
        self.experiments_name = args.experiments.split(',')
        self.runs = list(map(int, args.runs.split(',')))
        mainpath = args.mainpath
        self.simulator = args.simulator
        self.loop = args.loop
        self.body_phenotype = args.body_phenotype
        self.bisymmetry = list(map(int, args.bisymmetry.split(',')))
        self.generations = [150]
        self.type_damage = ['body', 'brain']
        self.damages = {'body': [0, 0.2, 0.4], 'brain': [0.2, 0.4]}
        self.sample_size = 10
        self.colors = ['#009900', '#EE8610']
        self.path = f'{mainpath}/{self.study}'
        self.anal_path = f'{self.path}/analysis/damage/all'
        self.comparison = args.comparison
        self.measures_names = [
            'speed_y',  'modules_count',  'hinge_prop',  'brick_prop', 'branching_prop', 'extremities_prop',
            'extensiveness_prop', 'coverage', 'proportion', 'symmetry']

    async def collect_data(self) -> None:

        if not os.path.exists(self.anal_path):
            os.makedirs(self.anal_path)
        if not os.path.exists(self.anal_path+'/images'):
            os.makedirs(self.anal_path+'/images')

        if self.simulator == 'mujoco':
            self._TERRAIN = terrains.flat()

        meas = ';'.join(str(val) for val in self.measures_names)
        with open(f'{self.anal_path}/damage_perf.txt', 'a') as f:
            f.write(f'experiment_name;run;gen;individual_id;type_damage;damage;{meas}\n')

        for ids, experiment_name in enumerate(self.experiments_name):
            print('\n', experiment_name)
            for run in self.runs:
                print('\n run: ', run)
                db = open_async_database_sqlite(f'{self.path}/{experiment_name}/run_{run}')

                for gen in self.generations:
                    print('  in gen: ', gen)
                    await self.recover(db, experiment_name, run, gen, ids)

    async def recover(self, db, experiment_name, run, gen, ids):
        async with AsyncSession(db) as session:

            rows = (
                (await session.execute(select(DbEAOptimizer))).all()
            )
            max_modules = rows[0].DbEAOptimizer.max_modules
            substrate_radius = rows[0].DbEAOptimizer.substrate_radius
            plastic_body = rows[0].DbEAOptimizer.plastic_body
            plastic_brain = rows[0].DbEAOptimizer.plastic_brain

            rows = (
                (await session.execute(select(DbOptimizerState))).all()
            )
            sampling_frequency = rows[0].DbOptimizerState.sampling_frequency
            control_frequency = rows[0].DbOptimizerState.control_frequency
            simulation_time = rows[0].DbOptimizerState.simulation_time

            rows = ((await session.execute(select(DbEnvconditions))).all())
            env_conditions = {}
            for c_row in rows:
                env_conditions[c_row[0].id] = literal_eval(c_row[0].conditions)

            query = select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat) \
                .filter((DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                        & (DbEAOptimizerGeneration.env_conditions_id == DbEAOptimizerIndividual.env_conditions_id)
                        & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                        & DbEAOptimizerGeneration.generation_index.in_([gen])
                        )
            # query = query.order_by(func.random())
            query = query.order_by(DbFloat.speed_y.desc())

            rows = ((await session.execute(query)).all())
            for idx, r in enumerate(rows[0:self.sample_size]):
                env_conditions_id = r.DbEAOptimizerGeneration.env_conditions_id
                print(f'  id:{r.DbEAOptimizerIndividual.individual_id} ' \
                          f' birth:{r.DbFloat.birth} ' \
                          f' cond:{env_conditions_id} ' \
                          f' dom:{r.DbEAOptimizerGeneration.seasonal_dominated} ' \
                          f' speed_y:{r.DbFloat.speed_y}' \
                      )

                genotype = (
                    await GenotypeSerializer.from_database(
                        session, [r.DbEAOptimizerIndividual.genotype_id]
                    )
                )[0]

                for type_damage in self.type_damage:
                    for damage in self.damages[type_damage]:

                        phenotype, queried_substrate = develop(genotype, genotype.mapping_seed, max_modules,
                                                               substrate_radius, env_conditions[env_conditions_id],
                                                               len(env_conditions), plastic_body, plastic_brain,
                                                               self.loop, self.body_phenotype, self.bisymmetry[ids])

                        joints_off = []
                        if damage > 0:
                            if type_damage == "body":
                                await self.damage_body(queried_substrate, damage, genotype.mapping_seed)
                            else:
                                joints_off = await self.damage_brain(queried_substrate, damage, genotype.mapping_seed)

                        if type_damage == "body" and idx == 0:
                            render = Render()
                            img_path = f'{self.anal_path}/images/' \
                                       f'{experiment_name}_{run}_{gen}_{r.DbEAOptimizerIndividual.individual_id}_{damage}.png'
                            render.render_robot(phenotype.body.core, img_path)

                        actor, controller = phenotype.make_actor_and_controller()
                        bounding_box = actor.calc_aabb()
                        env = Environment(EnvironmentActorController(controller))

                        if self.simulator == 'mujoco':
                            env.static_geometries.extend(self._TERRAIN.static_geometry)

                        x_rotation_degrees = float(env_conditions[env_conditions_id][2])
                        robot_rotation = x_rotation_degrees * np.pi / 180

                        env.actors.append(
                            PosedActor(
                                actor,
                                Vector3(
                                    [
                                        0.0,
                                        0.0,
                                        (bounding_box.size.z / 2.0 - bounding_box.offset.z),
                                    ]
                                ),
                                Quaternion.from_eulers([robot_rotation, 0, 0]),
                                [0.0 for _ in controller.get_dof_targets()],
                            )
                        )

                        batch = Batch(
                             simulation_time=simulation_time,
                             sampling_frequency=sampling_frequency,
                             control_frequency=control_frequency,
                         )
                        batch.environments.append(env)

                        if self.simulator == 'isaac':
                            runner = LocalRunnerI(
                                headless=True,
                                env_conditions=env_conditions[env_conditions_id],
                                real_time=False,
                                loop=self.loop,
                                joints_off=joints_off)

                        elif self.simulator == 'mujoco':
                            runner = LocalRunnerM(headless=False, loop=self.loop)

                        states = await runner.run_batch(batch)
                        if self.simulator == 'isaac':
                            states = extracts_states(states)

                        measures = Measure(states=states, genotype_idx=0, phenotype=phenotype,
                                     generation=0, simulation_time=simulation_time)
                        measures = measures.measure_all_non_relative()
                        meas = ';'.join(str(val) for key, val in measures.items() if key in self.measures_names)

                        with open(f'{self.anal_path}/damage_perf.txt', 'a') as f:
                            f.write(f'{experiment_name};{run};{gen};{r.DbEAOptimizerIndividual.individual_id};{type_damage};{damage};{meas}\n')

    async def damage_body(self, queried_substrate, damage, mapping_seed):
        # removes random extremities

        modules_available = len(queried_substrate)-1
        to_remove = int(modules_available * damage)
        removed = 0
        l = list(queried_substrate.items())

        rng = Random()
        rng.seed(mapping_seed)
        rng.shuffle(l)
        queried_substrate = dict(l)

        while removed < to_remove:
            to_pop = []
            for key in queried_substrate:
                remove = False
                if type(queried_substrate[key]) != Core \
                    and ((type(queried_substrate[key]) == ActiveHinge  and queried_substrate[key].children[ActiveHinge.ATTACHMENT] == None) \
                            or
                        ( type(queried_substrate[key]) != ActiveHinge \
                        and queried_substrate[key].children[Core.FRONT] == None \
                        and queried_substrate[key].children[Core.RIGHT] == None \
                        and queried_substrate[key].children[Core.LEFT] == None)
                        ):
                   remove = True

                if remove:
                    if removed < to_remove:
                        queried_substrate[key]._parent.children[queried_substrate[key].direction_from_parent] = None
                        to_pop.append(key)
                        removed += 1
                    else:
                        break
            for key in to_pop:
                queried_substrate.pop(key)

    async def damage_brain(self, queried_substrate, damage, mapping_seed):
        modules_available = 0
        for key in queried_substrate:
            if type(queried_substrate[key]) == ActiveHinge:
                modules_available += 1
        number_to_damage = int(modules_available * damage)
        rng = Random()
        rng.seed(mapping_seed)
        to_damage = rng.sample(range(0, modules_available), number_to_damage)

        return to_damage

    async def treat(self) -> None:

        keys = ['experiment_name', 'run', 'gen', 'individual_id']
        data = pd.read_csv(f'{self.anal_path}/damage_perf.txt', sep=";")
        pprint.pprint(data)
        data = data.astype({'run': int})
        print(self.runs)

        data = data[data['run'].isin(self.runs)]
        non_damaged = data[(data['damage'] == 0)].filter(items=keys+['speed_y'])
        non_damaged = non_damaged.rename(columns={'speed_y': 'speed_y_original'})
        data = pd.merge(data, non_damaged, on=keys)

        data['speed_prop'] = data['speed_y'] / data['speed_y_original']

        keys = ['experiment_name', 'run', 'gen', 'type_damage', 'damage']
        metric = 'median'

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

        data_inner = groupby(data, ['speed_prop'], metric, keys)

        data_inner.to_csv(f'{self.anal_path}/damage_inner.txt', index=False)

        keys = ['experiment_name', 'gen', 'type_damage', 'damage']
        metric = 'median'
        measures_inner = ['speed_prop_median']
        df_outer_median = groupby(data_inner, measures_inner, metric, keys)

        metric = q25
        df_outer_q25 = groupby(data_inner, measures_inner, metric, keys)

        metric = q75
        df_outer_q75 = groupby(data_inner, measures_inner, metric, keys)

        df_outer = pd.merge(df_outer_median, df_outer_q25, on=keys)
        df_outer = pd.merge(df_outer, df_outer_q75, on=keys)

        df_outer.to_csv(f'{self.anal_path}/damage_outer.txt', index=False)

    async def plot(self) -> None:

        outer = pd.read_csv(f'{self.anal_path}/damage_outer.txt')
        inner = pd.read_csv(f'{self.anal_path}/damage_inner.txt')

        for type_damage in self.type_damage:
            print(type_damage)

            if type_damage == 'body':
                damages = self.damages[type_damage][1:]
            else:
                damages = self.damages[type_damage]

            # for idxd, damage in enumerate(damages):
            #     font = {'font.size': 20}
            #     plt.rcParams.update(font)
            #     fig, ax = plt.subplots()
            #     plt.xlabel('Generation')
            #     plt.ylabel('Speed proportion')
            #
            #     for idex, experiment in enumerate(self.experiments_name):
            #         subouter = outer[(outer['experiment_name'] == experiment) & (outer['type_damage'] == type_damage) & (outer['damage'] == damage)]
            #
            #         pprint.pprint(subouter)
            #         ax.plot(subouter['gen'], subouter['speed_prop_median_median'], c=self.colors[idex])
            #         ax.fill_between(subouter['gen'],
            #                         subouter['speed_prop_median_q25'],
            #                         subouter['speed_prop_median_q75'],
            #                         alpha=0.3, facecolor=self.colors[idex])
            #
            #     plt.savefig(f'{self.anal_path}/{type_damage}_{damage}.png', bbox_inches='tight')
            #     plt.clf()
            #     plt.close(fig)

            ####

            comparison = ["all", "better", "worse"]
            runs_pairs = ["1,2,3,4,5,7,8,12,13,14,18,20,24,25,26,28,31,34,35,36,38,39|1,2,3,6,8,9,10,14,16,18,19,20,21,25,26,27,28,29,30,31,32,33,35,38,40",
                         "6,11,15,17,22,27,30,37|1,2,3,6,8,9,10,14,16,18,19,20,21,25,26,27,28,29,30,31,32,33,35,38,40"]

            for idc, comp in enumerate(comparison):
                print(comp)
                if idc > 0:
                    runs_pair = runs_pairs[idc-1]
                    runs_pair = runs_pair.split('|')
                    runs = [list(map(int, runs_pair[0].split(','))),
                            list(map(int, runs_pair[1].split(',')))]
                    print(runs)
                    subinner = inner[( (inner['experiment_name'] == 'bilateral') & (inner['run'].isin(runs[0])) ) | (inner['experiment_name'] == 'notbilateral') & (inner['run'].isin(runs[1])) ]
                    pprint.pprint(subinner)
                else:
                    subinner = inner

                for idxd, damage in enumerate(damages):
                    print(damage)
                    subinner2 = subinner[(subinner['type_damage'] == type_damage) & (subinner['damage'] == damage)]

                    sb.set(rc={"axes.titlesize": 23, "axes.labelsize": 23, 'ytick.labelsize': 21, 'xtick.labelsize': 21})
                    sb.set_style("whitegrid")

                    plot = sb.boxplot(x='experiment_name', y='speed_prop_median', data=subinner2,
                                      palette=self.colors, width=0.4, showmeans=True, linewidth=2, fliersize=6,
                                      meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"})
                    plot.tick_params(axis='x', labelrotation=10)

                    tests_combinations = [('bilateral', 'notbilateral')]

                    pd.set_option('display.max_rows', 500)
                    pprint.pprint(subinner2)
                    try:
                        if len(tests_combinations) > 0:
                            add_stat_annotation(plot, data=subinner2, x='experiment_name', y='speed_prop_median',
                                                box_pairs=tests_combinations,
                                                comparisons_correction=None,
                                                test='Mann-Whitney', text_format='star', fontsize='xx-large', loc='inside',
                                                verbose=1)
                    except Exception as error:
                        print('test:', error)

                    plt.xlabel('')
                    plt.ylabel('')
                    plot.get_figure().savefig(f'{self.anal_path}/box_{comp}_{type_damage}_{damage}.png', bbox_inches='tight')
                    plt.clf()
                    plt.close()

            font = {'font.size': 20}
            plt.rcParams.update(font)

    async def compare(self) -> None:

        pd.set_option('display.max_rows', 500)

        df_inner = pd.read_csv(f'{self.path}/analysis/basic_plots/df_inner.csv')
        df_inner2 = df_inner[(df_inner['generation_index'] == self.generations[0])]

        ref = df_inner2[(df_inner2['experiment'] == 'notbilateral')]
        ref_mean = np.mean(np.array(ref[["speed_y_mean"]]))
        ref_dev = np.std(np.array(ref[["speed_y_mean"]]))
        print('ref mean std', ref_mean, ref_dev, '\n')
        print('ref', shapiro(ref[["speed_y_mean"]]))
        font = {'font.size': 20}
        plt.rcParams.update(font)
        g = sb.distplot(ref["speed_y_mean"])
        plt.xlabel('Speed (cm/s)')

        #compared = df_inner2[(df_inner2['experiment'] == 'bilateral') & (df_inner2['speed_y_mean'] <= 4.44)]
        compared = df_inner2[(df_inner2['experiment'] == 'bilateral')]
        print("\n")
        pprint.pprint(compared[["experiment", "run", "speed_y_mean"]])
        print('compared', shapiro(compared[["speed_y_mean"]]))

        g = sb.distplot(compared["speed_y_mean"])
        plt.xlabel('Speed (cm/s)')
        g.set_ylim(0, 0.6)
        g.set_xlim(0, 8)
        plt.savefig(f'{self.anal_path}/hist.png', bbox_inches='tight')
        plt.clf()
        plt.close()

        wilc = wilcoxon(ref[["speed_y_mean"]], compared[["speed_y_mean"]])
        #wilc = mannwhitneyu(ref[["speed_y_mean"]], compared[["speed_y_mean"]])

        print('wilcoxon', wilc)

        # print(st.t.interval(alpha=0.95, df=len(ref[["speed_y_mean"]])-1,
        #       loc=np.mean(ref[["speed_y_mean"]]),
        #       scale=st.sem(ref[["speed_y_mean"]])))
        # min = sms.DescrStatsW(ref["speed_y_mean"]).tconfint_mean(alpha=0.05)[0]
        # max = sms.DescrStatsW(ref["speed_y_mean"]).tconfint_mean(alpha=0.05)[1]

        min = ref_mean - 1.96 * (ref_dev/math.sqrt(len(ref)))
        max = ref_mean + 1.96 * (ref_dev/math.sqrt(len(ref)))
        print('\n interval', min, max)

        sample = ref[(ref['speed_y_mean'] <= max)] #&  (ref['speed_y_mean'] >= min)
        print("\n max interval runs of ref")
        pprint.pprint(sample[["experiment", "run", "speed_y_mean"]])

        sample = compared[(compared['speed_y_mean'] > max)]
        print("\n better runs of compared")
        pprint.pprint(sample[["experiment", "run", "speed_y_mean"]])

        sample = compared[(compared['speed_y_mean'] < min)]
        print("\n worse runs of compared")
        pprint.pprint(sample[["experiment", "run", "speed_y_mean"]])


async def main() -> None:

    sim = Simulator()
    await sim.init()
  #  await sim.collect_data()
  #  await sim.compare()
  #  await sim.treat()
    await sim.plot()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())



