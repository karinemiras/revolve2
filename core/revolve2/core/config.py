import argparse


class Config():

    def _get_params(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--study_name",
            required=False,
            default="default_study",
            type=str,
            help="",
        )

        parser.add_argument(
            "--experiment_name",
            required=False,
            default="default_experiment",
            type=str,
            help="Name of the experiment.",
        )

        parser.add_argument(
            "--run",
            required=False,
            default=1,
            type=int,
            help="",
        )

        parser.add_argument(
            "--headless",
            required=False,
            default=True,
            type=bool,
            help="",
        )

        parser.add_argument(
            "--max_modules",
            required=False,
            default=15,
            type=int,
            help="",
        )

        parser.add_argument(
            "--body_substrate_dimensions",
            required=False,
            default='2d',
            type=str,
            help="2d or 3d",
        )

        # number of initial mutations for body and brain CPPNWIN networks
        parser.add_argument(
            "--num_initial_mutations",
            required=False,
            default=10,
            type=int,
        )

        parser.add_argument(
            "--simulation_time",
            required=False,
            default=30,
            type=int,
        )

        parser.add_argument(
            "--sampling_frequency",
            required=False,
            default=5,
            type=int,
        )

        parser.add_argument(
            "--run_simulation",
            required=False,
            default=1,
            type=int,
            help="If 0, runs optimizer without simulating robots, so behavioral measures are none."
        )

        parser.add_argument(
            "--control_frequency",
            required=False,
            default=5,
            type=int,
        )

        parser.add_argument(
            "--population_size",
            required=False,
            default=100,
            type=int,
        )

        parser.add_argument(
            "--offspring_size",
            required=False,
            default=100,
            type=int,
        )

        parser.add_argument(
            "--num_generations",
            required=False,
            default=100,
            type=int,
        )

        parser.add_argument(
            "--fitness_measure",
            required=False,
            default="pool_dominated_individuals",
            type=str,
        )

        args = parser.parse_args()

        return args

