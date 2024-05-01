import sys


from ._genotype import Genotype


# unequal crossover
def crossover_v1(
    parent1,
    parent2,
    rng,
) -> Genotype:

    # TODO: this threshold and types  should match with the develop method automatically
    promoter_threshold = 0.8
    types_nucleotides = 6

    # the first nucleotide is the concentration
    new_genotype = [(parent1.genotype[0]+parent2.genotype[0])/2]
    p1 = parent1.genotype[1:]
    p2 = parent2.genotype[1:]

    for parent in [p1, p2]:
        nucleotide_idx = 0
        promotor_sites = []
        while nucleotide_idx < len(parent):
            if parent[nucleotide_idx] < promoter_threshold:
                # if there are nucleotides enough to compose a gene
                if (len(parent)-1-nucleotide_idx) >= types_nucleotides:
                    promotor_sites.append(nucleotide_idx)
                    nucleotide_idx += types_nucleotides
            nucleotide_idx += 1
        # TODO: allow uniform random choice of keeping material after cut point instead of up to it
        cutpoint = rng.sample(promotor_sites, 1)[0]
        subset = parent[0:cutpoint+types_nucleotides+1]
        new_genotype += subset

    max_geno_size = 1000
    if len(new_genotype) > max_geno_size:
        new_genotype = new_genotype[0:max_geno_size]

    return Genotype(new_genotype)


# point crossover
# def crossover_v1(
#     parent1,
#     parent2,
#     rng,
# ) -> Genotype:
#
#     p1 = parent1.genotype
#     p2 = parent2.genotype
#
#     crossover_points = rng.sample(range(len(p1) + 1), 2)
#     crossover_points.sort()
#
#     new_genotype = p1[:crossover_points[0]] + p2[crossover_points[0]:crossover_points[1]] + p1[crossover_points[1]:]
#
#     return Genotype(new_genotype)