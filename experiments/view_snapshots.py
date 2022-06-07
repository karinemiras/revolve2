from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual
from revolve2.core.modular_robot.render.render import Render
#TODO: make import based on param and move file to anal_resources
from opt_mod_1.genotype import GenotypeSerializer, develop


async def main() -> None:

    experiment_name = 'default_experiment'
    generation = 0

    db = open_async_database_sqlite(f'./data/{experiment_name}')
    async with AsyncSession(db) as session:
        rows = (
            (await session.execute(select(DbEAOptimizerGeneration, DbEAOptimizerIndividual)
                                   .filter(DbEAOptimizerGeneration.generation_index.in_([generation]))
                                   .filter(DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)


            )).all()
        )

        for r in rows:
            genotype = (
                await GenotypeSerializer.from_database(
                    session, [r.DbEAOptimizerIndividual.genotype_id]
                )
            )[0]

            phenotype = develop(genotype)
            render = Render()
            # TODO: fix experiment name
            img_path = f'data/{r.DbEAOptimizerIndividual.genotype_id}.png'
            # TODO: create folders structure auto
            render.render_robot(phenotype.body.core, img_path)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
