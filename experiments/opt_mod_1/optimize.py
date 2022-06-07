import argparse
import logging
from random import Random, random
import sys
import multineat

from genotype import random as random_genotype
from optimizer import Optimizer

from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import ProcessIdGen


async def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        required=False,
        default="default_experiment",
        type=str,
        help="Name of the experiment.",
    )

    args = parser.parse_args()

    # number of initial mutations for body and brain CPPNWIN networks
    NUM_INITIAL_MUTATIONS = 10

    SIMULATION_TIME = 1#30
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 5

    POPULATION_SIZE = 10
    OFFSPRING_SIZE = 10#100
    # actually means number of offspring generations
    NUM_GENERATIONS = 1#00

    FITNESS_MEASURE = 'displacement_xy'

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    logging.info(f"Starting optimization")

    # random number generator
    rng = Random()
    rng.seed(random())

    # database

    database = open_async_database_sqlite(f'./data/{args.experiment_name}')


    # process id generator
    process_id_gen = ProcessIdGen()

    # multineat innovation databases
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    process_id = process_id_gen.gen()
    maybe_optimizer = await Optimizer.from_database(
        database=database,
        process_id=process_id,
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
        process_id_gen=process_id_gen,
        fitness_measure=FITNESS_MEASURE,
    )
    if maybe_optimizer is not None:
        optimizer = maybe_optimizer
    else:

        initial_population = [
            random_genotype(innov_db_body, innov_db_brain, rng, NUM_INITIAL_MUTATIONS)
            for _ in range(POPULATION_SIZE)
        ]

        optimizer = await Optimizer.new(
            database=database,
            process_id=process_id,
            initial_population=initial_population,
            rng=rng,
            process_id_gen=process_id_gen,
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_generations=NUM_GENERATIONS,
            offspring_size=OFFSPRING_SIZE,
            fitness_measure=FITNESS_MEASURE,
        )

    logging.info("Starting optimization process..")

    await optimizer.run()

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
