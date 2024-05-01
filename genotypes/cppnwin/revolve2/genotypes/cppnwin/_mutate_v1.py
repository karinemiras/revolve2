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


# mutation for unequal crossover
def mutate_body(
        genotype,
        rng,
) -> Genotype:

    position = rng.sample(range(0, len(genotype.genotype)), 1)[0]
    type = rng.sample(['perturbation', 'deletion', 'addition', 'swap'], 1)[0]

    if type == 'perturbation':
        newv = round(genotype.genotype[position]+rng.normalvariate(0, 0.1), 2)
        if newv > 1:
            genotype.genotype[position] = 1
        elif newv < 0:
            genotype.genotype[position] = 0
        else:
            genotype.genotype[position] = newv

    if type == 'deletion':
        genotype.genotype.pop(position)

    if type == 'addition':
        genotype.genotype.insert(position, round(rng.uniform(0, 1), 2))

    if type == 'swap':
        position2 = rng.sample(range(0, len(genotype.genotype)), 1)[0]
        while position == position2:
            position2 = rng.sample(range(0, len(genotype.genotype)), 1)[0]

        position_v = genotype.genotype[position]
        position2_v = genotype.genotype[position2]
        genotype.genotype[position] = position2_v
        genotype.genotype[position2] = position_v

    return genotype




# mutation for point crossover
# def mutate_body(
#         genotype,
#         rng,
# ) -> Genotype:
#
#     mutation_size = 0.1
#     num_mutations = int(len(genotype.genotype) * mutation_size)
#
#     for i in range(0, num_mutations):
#         position = rng.sample(range(0, len(genotype.genotype)), 1)[0]
#         type = rng.sample(['perturbation', 'swap'], 1)[0]
#
#         if type == 'perturbation':
#             newv = round(genotype.genotype[position]+rng.normalvariate(0, 0.1), 2)
#             if newv > 1:
#                 genotype.genotype[position] = 1
#             elif newv < 0:
#                 genotype.genotype[position] = 0
#             else:
#                 genotype.genotype[position] = newv
#
#         if type == 'swap':
#             position2 = rng.sample(range(0, len(genotype.genotype)), 1)[0]
#             while position == position2:
#                 position2 = rng.sample(range(0, len(genotype.genotype)), 1)[0]
#
#             position_v = genotype.genotype[position]
#             position2_v = genotype.genotype[position2]
#             genotype.genotype[position] = position2_v
#             genotype.genotype[position2] = position_v
#
#     return genotype
