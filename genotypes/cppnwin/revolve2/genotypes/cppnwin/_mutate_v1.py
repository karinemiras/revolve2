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
        genotype,
        rng,
) -> Genotype:
    pertub_qt = 1
    #print('----------naaaa')
    positions = rng.sample(range(0, len(genotype.genotype)), int(len(genotype.genotype)*0.03))
    for p in positions:
        #print('coco',p)
        #genotype.genotype[p] = round(rng.uniform(0, 1), 2)
        newv = round(genotype.genotype[p]+rng.normalvariate(0, 0.1), 2)
       # print(genotype.genotype[p],newv)
        if newv > 1:
            genotype.genotype[p] = 1
        elif newv < 0:
            genotype.genotype[p] = 0
        else:
            genotype.genotype[p] = newv
       # print(genotype.genotype[p] )
            # mutation_size = 0.05

    # positions = rng.sample(range(0, len(genotype.genotype)), int(len(genotype.genotype) * mutation_size))
    # for p in positions:
    #     genotype.genotype[p] = round(rng.uniform(0, 1), 2)

    return genotype
