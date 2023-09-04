import sys


from ._genotype import Genotype


def crossover_v1(
    parent1,
    parent2,
    rng,
) -> Genotype:

    # TODO: this threshold and types  should match with the develop method automatically
    promoter_threshold = 0.8
    types_nucleotypes = 6

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
                if (len(parent)-1-nucleotide_idx) >= types_nucleotypes:
                    promotor_sites.append(nucleotide_idx)
                    nucleotide_idx += types_nucleotypes
            nucleotide_idx += 1

        cutpoint = rng.sample(promotor_sites, 1)[0]
        subset = parent[0:cutpoint+types_nucleotypes+1]
        new_genotype += subset

    return Genotype(new_genotype)
