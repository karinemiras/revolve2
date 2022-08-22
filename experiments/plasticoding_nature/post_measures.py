"""
Visualize and run a modular robot using Mujoco.
"""

from pyrr import Quaternion, Vector3

from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from sqlalchemy import text
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizer, DbEnvconditions
from genotype import GenotypeSerializer, develop
from optimizer import DbOptimizerState
import sys
from revolve2.core.database.serializers import DbFloat
import pprint
import numpy as np
from ast import literal_eval


class Post:

    async def recover(self):

        self.study = 'plasticoding_nature'
        self.experiments_name = ['plastic.200.300.150']
        self.runs = [1] #list(range(1, 10+1))
        mainpath = "karine"

        for experiment_name in self.experiments_name:
            print('\n', experiment_name)
            for run in self.runs:
                print('\n run: ', run)

                path = f'/storage/{mainpath}/{self.study}'
                db = open_async_database_sqlite(f'{path}/{experiment_name}/run_{run}')

                async with AsyncSession(db) as session:

                    rows = (
                        (await session.execute(select(DbEAOptimizer))).all()
                    )

                    max_modules = rows[0].DbEAOptimizer.max_modules
                    substrate_radius = rows[0].DbEAOptimizer.substrate_radius
                    plastic_body = rows[0].DbEAOptimizer.plastic_body
                    plastic_brain = rows[0].DbEAOptimizer.plastic_brain

                    rows = ((await session.execute(select(DbEnvconditions))).all())
                    env_conditions = {}
                    for c_row in rows:
                        env_conditions[c_row[0].id] = literal_eval(c_row[0].conditions)

                    sql_text = text("SELECT distinct individual_id, genotype_id FROM ea_optimizer_individual order by individual_id")
                    rows = await session.execute(sql_text)

                    for r in rows:
                        queried_substrates = []
                        for e in env_conditions:
                            genotype_id = r[1]

                            genotype = (
                                await GenotypeSerializer.from_database(
                                    session, [genotype_id]
                                )
                            )[0]

                            phenotype, queried_substrate = develop(genotype, genotype.mapping_seed, max_modules, substrate_radius,
                                                                   env_conditions[e], plastic_body, plastic_brain)
                            queried_substrates.append(queried_substrate)

                       # print('-------',genotype_id)

                        keys_a = set(queried_substrates[0].keys())
                        keys_b = set(queried_substrates[1].keys())
                        intersection = keys_a & keys_b
                        disjunct_a = [a for a in keys_a if a not in intersection]
                        disjunct_b = [b for b in keys_b if b not in intersection]
                        # print('a\n',keys_a)
                        # print('b\n',keys_b)
                        # print('inter\n',intersection)
                        # print('disjunct_a\n', disjunct_a,len(disjunct_a))
                        # print('disjunct_b\n', disjunct_b,len(disjunct_b))
                        # body_changes = len(disjunct_a) + len(disjunct_b)
                        for i in intersection:
                            print(type(queried_substrates[0][i]), type(queried_substrates[1][i]))
                            if type(queried_substrates[0][i]) != type(queried_substrates[1][i]):
                                print('diff')
                                body_changes += 1
                        print(body_changes)

                       # sys.exit()

                        # for idx_p, p in enumerate(queried_substrates):
                        #     print(idx_p)
                        #
                        #     print(p)

                            # actor, controller = p.make_actor_and_controller()
                            # print(controller._weight_matrix)
                            # pprint.pprint(controller._state)

                        # print('plasticity_brain')
                        #
                        # print('plasticity_body')


async def main() -> None:

    p = Post()
    await p.recover()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())



