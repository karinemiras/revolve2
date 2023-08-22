import multineat

from ._genotype import Genotype


def mutate_brain(
    genotype: Genotype,
    multineat_params: multineat.Parameters,
    innov_db: multineat.InnovationDatabase,
    rng: multineat.RNG,
) -> Genotype:
    new_genotype = genotype.genotype.MutateWithConstraints(
        False,
        multineat.SearchMode.BLENDED,
        innov_db,
        multineat_params,
        rng,
    )
    return Genotype(new_genotype)

def mutate_body(
        genotype: Genotype,
        rng: multineat.RNG,
) -> Genotype:
    pertub_qt = 1
    positions = rng.sample(range(0, len(genotype.genotype)), pertub_qt)
    for p in positions:
        genotype.genotype[p] = round(rng.uniform(0, 1), 2)

    # mutation_size = 0.05
    # positions = rng.sample(range(0, len(genotype.genotype)), int(len(genotype.genotype) * mutation_size))
    # for p in positions:
    #     genotype.genotype[p] = round(rng.uniform(0, 1), 2)

    return genotype
