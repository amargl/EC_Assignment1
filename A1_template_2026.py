"""EC A1 template code - evolving robot morphologies with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo* file for starting you out with assignment 1.
It samples one body at random, decodes it, scores it
against a set of target bodies, and shows you the result.

*Your Job* section at the bottom of this file summarises the programming task. Full assignment description can be found in the pdf file on Canvas.


THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve a robot BODY that is as structurally close as possible to a whole set
of given target bodies at once.

    fitness = mean tree edit distance to every body in TARGET_DIR,
              plus one standard deviation across those per-target distances
"""


# Standard library
import random
import copy
from pathlib import Path
from typing import Literal
import os
import names
import matplotlib as mpl

# Third-party libraries
import mujoco as mj
import networkx as nx
import numpy as np
import torch
from jedi.api import file_name
from mujoco import viewer

from ariel.ec import Individual
from ariel.ec.genotypes.tree import operators
from ariel.ec.genotypes.tree.io import genome_to_networkx_dict
# Local scripts
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.ec.genotypes.tree.operators import random_tree,subtree_swap, crossover_subtree
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder

"""
We import some extra modules to run multipoint crossover
"""
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    ModuleType,
    IDX_OF_CORE,
)

# Type aliases
type GenotypeTypes = Literal["nde", "tree"]
type ViewerTypes = Literal["launcher", "video", "frame", "none"]

from ariel.ec import (
    EA,
    EAOperation,
    EASettings,
    Individual,
    Population,
)

"""
We want to visualize the graphs and therefore import the
Following package
"""
import matplotlib.pyplot as plt

#to make font computer modern like in latex
fpath = Path(mpl.get_data_path(), "fonts/ttf/cmr10.ttf")
prop = mpl.font_manager.FontProperties(fname=fpath)
mpl.font_manager.fontManager.addfont(str(fpath))
mpl.rcParams["font.family"] = prop.get_name()

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
# NOTE: the tree operators use the `random` module, the NDE uses numpy for its
# own genotype vectors AND is a torch.nn.Module for its internal network - that
# network's weight initialisation uses torch's own RNG, entirely separate from
# numpy/random. If you're using "nde", seed all THREE or your runs will not be
# reproducible across separate script runs, even with the same seed value.
SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
TARGET_DIR: Path = HERE / "target_bodies"  # the bodies you must approach
NUM_OF_MODULES: int = 20  # module budget per evolved body
GENOTYPE: GenotypeTypes = "tree"  # "nde" | "tree"
MODE: ViewerTypes = "launcher"  # "launcher" | "frame" | "video"
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]


""""""""""""""""""""""""""
"!!!!!!!TEST PARAMETERS!!!!!--GR16--"
""""""""""""""""""""""""""
P_CROSSOVER = 0.5
P_MUTATION = 0.6
MUTATION_FUNC = "replace_node"
POINTS = 3
AMOUNT_PARENTS = 3
SAMPLE_SIZE = 15
POPULATION_SIZE = 100
""""""""""
:):):):)
"""""""""""""""

# ============================================================================ #
#  1. THE TARGET BODIES
# ============================================================================ #

def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory.

    Returns
    -------
    list of nx.DiGraph
        One graph per JSON file, sorted by filename.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]



# NDE settings
GENOTYPE_SIZE: int = 64  # length of each of the three NDE gene vectors


# Constructed ONCE, at import time, and reused for every decode call below and
# in your own EA. See the "IMPORTANT" note on "nde" in THE GENOTYPE CONTRACT
# above: rebuilding this per individual silently breaks the genotype -> body
# mapping, because its internal network randomises on construction.
_NDE = NeuralDevelopmentalEncoding(
    number_of_modules=NUM_OF_MODULES,
    genotype_size=GENOTYPE_SIZE,
)



# ============================================================================ #
#  3. FITNESS
# ============================================================================ #

def fitness_function(
    body: nx.DiGraph,
    targets: list[nx.DiGraph],
) -> float:
    """Score one body against the whole target set. LOWER IS BETTER.

    Some things worth thinking about:
      * The std term charges for unevenness - body that is mediocre against every target
        and one that is excellent on most but bad on one can still land close
        in fitness, but the latter is penalized a bit more.
      * Nothing here rewards small bodies. Does your EA bloat? Should a size
        penalty be part of fitness, or is that the encoding's job?
    """
    return mean_plus_std_tree_edit_distance(body, targets)


# ============================================================================ #
#  4. LOOKING AT A BODY
# ============================================================================ #


def show_body(
    body: nx.DiGraph,
    mode: ViewerTypes = MODE,
    file_name: str = "body",
) -> None:
    """Build a body graph in MuJoCo and look at it.

    There is no controller and no physics worth speaking of - this exists so
    you can SEE what your fitness function is actually rewarding. Do this
    early and often. A number going down is not evidence that the bodies look
    anything like the targets.
    """
    if mode == "none":
        return

    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(body)
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    match mode:
        case "launcher":
            # Interactive window. Drag the modules around; nothing drives them.
            viewer.launch(model=model, data=data)
        case "frame":
            # A still image - the cheapest way to eyeball a body.
            save_path = str(DATA / f"{file_name}.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
            console.log(f"saved {save_path}")
        case "video":
            # Mostly useful for showing a body slumping under gravity.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(model, data, duration=5.0, video_recorder=recorder)

"""""""""
5. Helper function to show graph --GR16--
"""""""""

def show_body_tree(body: nx.DiGraph, file_name: str = "body"):
    # To better understand the inner workings of ARIEL and how the tree
    # forms a robot, we added the following helper function which lets us plot
    # the tree which represents the robot
    pos = nx.nx_agraph.graphviz_layout(body, prog="dot")

    plt.figure(figsize=(10, 7))

    # For this we first need to get all the information of the different nodes
    # and edges
    node_labels = {
        nid: f"{data.get('type', '')}\n{data.get('rotation', '').split("_")[1]}°"
        for nid, data in body.nodes(data=True)
    }
    edge_labels = nx.get_edge_attributes(body, "face")

    # We draw this info into the graph
    nx.draw(
        body,
        pos,
        labels=node_labels,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        font_size=8,
        font_weight="bold",
        arrows=True,
        arrowsize=15,
        edge_color="gray",
    )

    nx.draw_networkx_edge_labels(
        body,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    )

    # We save a png of the image
    plt.title(file_name)
    plt.axis("off")
    plt.savefig(str(DATA / f"{file_name}_tree.png"), dpi=200, bbox_inches="tight")
    plt.close()

""""""""""""""""""""""""""""
6. Initialization--GR16--
"""""""""""""""""""""""""""

def initialize_population(targets:list, n:int=50) -> Population:
    # We randomly intitialize n bodies
    population = []
    for _ in range(n):
        # We create a random genome
        # Not from the random body thingy.
        # Very enoying :(((
        genome = random_tree(NUM_OF_MODULES)
        # For this genome, we create a individual
        ind = Individual()
        ind.genotype = genome
        fitness = fitness_function(genome.to_networkx(), targets)
        ind.fitness = fitness
        # We add the individual to the population
        population.append(ind)
    # We then create a population object and return it
    return Population(population)

""""""""""""""""""""""""""""
6. Evolution--GR16--
"""""""""""""""""""""""""""

def reproduction(parents: Population,p_mutation: float, mutation_func:str, points:int= 1, baseline:bool = False) -> list[Individual]:
    """Perform standard GP subtree crossover on tree genomes.

    Two offspring are produced by exchanging randomly chosen subtrees from each parent.
    Unlike one-point crossover on linear genomes, this selects any node (including leaves)
    and swaps the entire subtree rooted at that node. Returns a pair ``(child1, child2)``.
    Parents are left unmodified.
    """
    # This method is highly inspired by the crossover_subtree method
    # given with ariel. The method has added the possibility for multiple
    # parents and for multiple cross points
    # parents and for multiple cross points
    # Take the genomes of all the parents as dictionaries
    genomes = [copy.deepcopy(par.genotype) for par in parents]
    child_genomes = [copy.deepcopy(gen) for gen in genomes]

    # This is directly copied from the crossover_subtree method
    # from .ec.tree.operators
    def pick_noncore(gen: TreeGenome) -> int | None:
        candidates = [nid for nid in gen.nodes if nid != IDX_OF_CORE]
        return random.choice(candidates) if candidates else None

    def mutate(genomes: list[TreeGenome], p_mutation: float, mutation_func:str):
        for gen in genomes:
            # We only perform mutation onder certain probability
            if p_mutation < random.random():
                continue
            # We perform one of the predefined mutation
            # functions
            if mutation_func == "replace_node":
                operators.mutate_replace_node(gen)
            elif mutation_func == "hoist":
                operators.mutate_hoist(gen)
            elif mutation_func == "shrink":
                operators.mutate_shrink(gen)
            elif mutation_func == "replace":
                operators.mutate_subtree_replacement(gen)
            else:
                raise ValueError(f"Unknown mutation function: {mutation_func}")

    def to_individuals(genomes: list[TreeGenome]) -> list[Individual]:
        # We create a small helper function that can transform
        # a genome into a individual with a genotype
        result = []
        for g in genomes:
            ind = Individual()
            ind.genotype = g
            fitness = fitness_function(g.to_networkx(), targets)
            ind.fitness = fitness
            result.append(ind)
        return result

    amount_of_parents = len(genomes)
    # We apply n-point crossover
    # This simply means that we doe a crossover between genomes
    # points times
    for _ in range(points):
        # We do not perform crossover for the baseline cases
        if baseline:
            break
        # We then perform Multi-parent recombination
        # It does not matter in our case how many parents the
        for i in range(amount_of_parents):
            # We perform a crossover under certain prbability
            # and when it is not the baseline case
            if random.random() < P_CROSSOVER:
                # We find the nodes where we want to perform the cut:
                cut_1 = pick_noncore(genomes[i])
                cut_2 = pick_noncore(child_genomes[(i + 1)%amount_of_parents])
                subtree_swap(genomes[i], child_genomes[(i + 1)%amount_of_parents], cut_1, cut_2)

    try:
        for g in child_genomes:
            validate_genome_dict(g.to_dict())
    except ValueError as e:
        # If validation fails, return copies of the original
        # Parents. We do apply mutation
        genomes = [copy.deepcopy(par.genotype) for par in parents]
        mutate(genomes, p_mutation, mutation_func)
        return to_individuals(genomes)
    # Last but not least, we apply the mutation
    mutate(child_genomes, p_mutation, mutation_func)
    return to_individuals(child_genomes)

def evolution_step(pop: Population,
                   amount_parents:int = AMOUNT_PARENTS,
                   mutation_func:str = MUTATION_FUNC,
                   p_mutation:float = P_MUTATION,
                   points:int = POINTS,
                   base_line:bool = False) -> Individual:
    # We take a subsection of the population
    sub_pop = pop.sample(SAMPLE_SIZE)
    # from this section, we take n individuals
    # with the lowest fitness
    parents = sub_pop.best(sort="min", n=amount_parents)
    # We reproduce these individuals to
    # get the offspring
    children = reproduction(parents, mutation_func=mutation_func, p_mutation=p_mutation, points=points, baseline=base_line)
    # We add the offspring to the subsection
    # of the population and to the general population
    pop.extend(children)
    sub_pop.extend(children)
    # We then find the worst performing individuals
    # of the subsection
    worst = sub_pop.best(sort="max", n=amount_parents)
    # We kill all the worst individuals in the
    # subset
    for w in worst:
        w.alive = False
        # We then send the dead robots to heaven
        fallen_one = names.get_full_name()
        w.genotype.save_json(str(DATA / "heaven" / fallen_one))
    # We then return all the alive individuals in
    # the population
    return pop.alive

def experiment(experiment:str,targets,
               timesteps:int=50,
               amount_parents:int = AMOUNT_PARENTS,
               points:int = POINTS,
               baseline:bool = False):
    # We initialize a population
    pop = initialize_population(targets, POPULATION_SIZE)
    show_body_tree(pop[0].genotype.to_networkx(),file_name="start")
    os.makedirs(DATA / experiment / "best_genotype", exist_ok=True)
    os.makedirs(DATA / experiment / "statistics", exist_ok=True)



    if baseline:
        stat_type = "baseline"
        # We do not update the target files when we
        # look for the baseline
        target_files = None
    else:
        stat_type = "general"
        # We define the files where we want to place our target data
        target_files = [
            open(DATA / experiment / "statistics" / f"target_{t}_stats", "w")
            for t in range(len(targets))
        ]
    with open(DATA / experiment / "statistics"/f"{stat_type}_stats", "w") as f:
        for i in range(timesteps):
            # We perform an evolution step
            pop = evolution_step(pop,
                                 amount_parents=amount_parents,
                                 points=points,
                                 base_line=baseline)
            # We determine the best and worst of the populatio
            best = pop.best(sort="min", n=1)[0]
            worst = pop.best(sort="max", n=1)[0]
            # We save the genotype of the fittest individual as a
            # jSon file
            best.genotype.save_json(DATA / experiment / "best_genotype"/ f"time_{i}")
            # We keep track of parameters about general
            # statistics of the performance
            total_fitness =  np.array([pop.fitness for pop in pop])
            # We write down the statistics
            f.write(f"{best.fitness:.2f}\t{float(np.mean(total_fitness)):.2f}\t{float(np.std(total_fitness)):.2f}\t{float(worst.fitness):.2f}\n")
            if baseline:
                # we ignore the targets for the baseline cases
                continue
            # We also determine the distance from eah individual to
            # each target
            for t, tar in enumerate(targets):
                distance_to_target = np.array(
                    [tree_edit_distance(ind.genotype.to_networkx(), tar) for ind in pop]
                )
                target_files[t].write(
                    f"{float(np.min(distance_to_target)):.2f}\t"
                    f"{float(np.mean(distance_to_target)):.2f}\t"
                    f"{float(np.std(distance_to_target)):.2f}\t"
                    f"{float(np.max(distance_to_target)):.2f}\n"
                )




"""""
7. Plots and stats --GR16--
"""""

def plot_target_stats(experiment:str):
    # We want to be able to plot multiple plots next to
    # each other (for the target cases)
    amount_of_plots = 5
    files = [f"target_{i}" for i in range(amount_of_plots)]

    fig, axes = plt.subplots(1, amount_of_plots, figsize=(6 * amount_of_plots, 5), sharey=True)
    if amount_of_plots == 1:
        axes = [axes]

    # We define the statistics that are saved
    for ax, file in zip(axes, files):
        best_fitness = []
        mean_std_fitness = []
        std_fitness = []
        worst_fitness = []
        timesteps = 0

        with open(DATA / experiment / "statistics"/f"{file}_stats", "r") as f:
            for l in f.readlines():
                info = l.split("\t")
                best_fitness.append(float(info[0]))
                mean_std_fitness.append(float(info[1]))
                std_fitness.append(float(info[2]))
                worst_fitness.append(float(info[3]))
                timesteps +=1

        # We set the upper and lower bound of the std
        upper = [mean_std_fitness[i] + std_fitness[i] for i in range(timesteps)]
        lower = [mean_std_fitness[i] - std_fitness[i] for i in range(timesteps)]
        # We also plot the mean, best and worst fitnesses
        ax.plot(range(timesteps), mean_std_fitness, label="Mean Fitness", color="gray")
        ax.fill_between(range(timesteps), upper, lower, color="gray", alpha=0.2)
        ax.plot(range(timesteps), best_fitness, label="Best Fitness", color="green")
        ax.plot(range(timesteps), worst_fitness, label="Worst Fitness", color="red")
        """"
        We need to change the titles and to represent the targets when doing target plot
        """
        name = "target " + file.split("_")[1]
        ax.set_title(name)
        ax.legend()

    fig.tight_layout()
    """"
    We want the file type to be svg in the final,
    (these are vectorized drawings so you can zoom in indefenately)
    for testing png works fine :)
    """
    os.makedirs(DATA / experiment / "plots", exist_ok=True)
    fig.savefig(str(DATA / experiment / "plots"/f"target_plot.svg"), dpi=200, bbox_inches="tight")
    fig.savefig(str(DATA / experiment / "plots" / f"target_plot.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_general_stats(experiment:str):
    # We want to be able to plot multiple plots next to
    # each other (for the target cases)
    amount_of_plots = 1
    files = ["general", "baseline"]


    # We define the statistics that are saved
    for file in files:
        best_fitness = []
        mean_std_fitness = []
        std_fitness = []
        worst_fitness = []
        timesteps = 0

        with open(DATA / experiment / "statistics"/f"{file}_stats", "r") as f:
            for l in f.readlines():
                info = l.split("\t")
                best_fitness.append(float(info[0]))
                mean_std_fitness.append(float(info[1]))
                std_fitness.append(float(info[2]))
                worst_fitness.append(float(info[3]))
                timesteps +=1

        if file == "baseline":
            plt.plot(range(timesteps), best_fitness,linestyle = "--", label="Baseline Fitness", color="green")
            break
        # We set the upper and lower bound of the std
        upper = [mean_std_fitness[i] + std_fitness[i] for i in range(timesteps)]
        lower = [mean_std_fitness[i] - std_fitness[i] for i in range(timesteps)]
        # We also plot the mean, best and worst fitnesses
        plt.plot(range(timesteps), mean_std_fitness, label="Mean Fitness", color="gray")
        plt.fill_between(range(timesteps), upper, lower, color="gray", alpha=0.2)
        plt.plot(range(timesteps), best_fitness, label="Best Fitness", color="green")
        plt.plot(range(timesteps), worst_fitness, label="Worst Fitness", color="red")
    """"
    We need to change the titles and to represent the targets when doing target plot
    """
    split = experiment.split("_")
    front = split[0].split("-")
    name = f"Amount of crossovers: {front[1]}"
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    """"
    We want the file type to be svg in the final,
    (these are vectorized drawings so you can zoom in indefenately)
    for testing png works fine :)
    """
    os.makedirs(DATA / experiment / "plots", exist_ok=True)
    plt.savefig(str(DATA / experiment / "plots"/f"general_plot.svg"), dpi=200, bbox_inches="tight")
    plt.savefig(str(DATA / experiment / "plots" / f"general_plot.png"), dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    """""""""
    "Before running a experiment, make sure to change the experiment run:"
    """""""""
    TIMESTEPS = 50
    EXPERIMENTAL_RUNS = 5
    # We create a heaven for all the dead soldiers
    os.makedirs(DATA / "heaven", exist_ok=True)
    # We load the targets
    targets = load_targets()
    # These are the values that we want to test
    point_values = [1]
    amount_of_parents = [2]
    # These are the amout of runs that our experiment will take
    amount_of_runs = len(point_values) * len(amount_of_parents)
    run_so_far= 1
    print("Running...")
    for point in point_values:
        for amount in amount_of_parents:
            print("="*40)
            print(f"Experiment {run_so_far}/{amount_of_runs}: point {point}, parent {amount}")
            # We run an experiment for each combination of parents and
            # points for the amount of experimental runs
            SEED = 42
            for i in range(EXPERIMENTAL_RUNS):
                print(".", end="")
                experiment(f"point-{point}_parent-{amount}/exp{i}",
                           targets,
                           timesteps=TIMESTEPS,
                           amount_parents=amount,
                           points=point)
                # We perform the same experiment, without performing crossover (AKA our baseline)
                # For this we have to set our seed back to the original value
                RNG = np.random.default_rng(SEED)
                random.seed(SEED)
                torch.manual_seed(SEED)
                plot_target_stats(f"point-{point}_parent-{amount}/exp{i}")
                experiment(f"point-{point}_parent-{amount}/exp{i}",
                           targets,
                           timesteps=TIMESTEPS,
                           amount_parents=amount,
                           points=point,
                           baseline=True)
                plot_general_stats(f"point-{point}_parent-{amount}/exp{i}")
                # We change the seed of our random valuebles
                # every time we perform an experiment
                SEED +=1
                RNG = np.random.default_rng(SEED)
                random.seed(SEED)
                torch.manual_seed(SEED)
            print("\nFinished")
            run_so_far += 1


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above samples ONE body at random and scores it. Your task is to
# replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individ
#     https://login.uva.nl/adfs/ls/?SAhttps://login.uva.nl/adfs/ls/?SAMLRequest=lZJBb8IwDIX%2FSpV7SYpKy6JSqYPDkNiooNthtxAMREoTFqeM%2FftlZdPYBWlXx9%2Bz33MKFK0%2B8qrzB7OCtw7QR%2BdWG%2BT9w4R0znArUCE3ogXkXvJ19bjgwwHjR2e9lVaTK%2BQ2IRDBeWUNieazCZkuV81ynI%2FSkUxzwbKNTJJ8I4YpZHnOsixjmRgmG5HcjdOURC%2FgMLATEqSCAGIHc4NeGB9KbJjF7C5ORk3CeJLylL2SaBb8KCN8Tx28PyKnVNu9MoPuJAZGU7HdIdVISVT97Da1BrsW3BrcSUl4Xi1%2BWTCBhQF2bietgbPvNUJ6YLyS%2FSCKRyovEvGV3%2Fo7rHtltsrsb%2Be0uTQhf2iaOq6X64aUxVe8vHftyn%2Fu04IXW%2BFFQa9Fisvxn8L4%2Bay2WsmPqNLavk8dCA8T4l0HhJYX6u8vKT8BMLRequest=lZJBb8IwDIX%2FSpV7SYpKy6JSqYPDkNiooNthtxAMREoTFqeM%2FftlZdPYBWlXx9%2Bz33MKFK0%2B8qrzB7OCtw7QR%2BdWG%2BT9w4R0znArUCE3ogXkXvJ19bjgwwHjR2e9lVaTK%2BQ2IRDBeWUNieazCZkuV81ynI%2FSkUxzwbKNTJJ8I4YpZHnOsixjmRgmG5HcjdOURC%2FgMLATEqSCAGIHc4NeGB9KbJjF7C5ORk3CeJLylL2SaBb8KCN8Tx28PyKnVNu9MoPuJAZGU7HdIdVISVT97Da1BrsW3BrcSUl4Xi1%2BWTCBhQF2bietgbPvNUJ6YLyS%2FSCKRyovEvGV3%2Fo7rHtltsrsb%2Be0uTQhf2iaOq6X64aUxVe8vHftyn%2Fu04IXW%2BFFQa9Fisvxn8L4%2Bay2WsmPqNLavk8dCA8T4l0HhJYX6u8vKT8Bual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# For morphology-specific evolution with the tree encoding, read:
#
#     examples/c_genotypes/1_body_evolution_tree.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- GENOTYPE - DEPENDENT "GOTCHA"S -------------------------
#
#   TREE: VARIABLE LENGTH - Tree genotypes grow; without pressure against it they
#     will grow forever, and every extra module costs an edit.
#   NDE: REPRODUCIBILITY   If you're using "nde": construct
#     `NeuralDevelopmentalEncoding` ONCE for your whole run, never per
#     individual or per generation, AND call `torch.manual_seed(...)` in
#     addition to the numpy/random seeds. See the two "IMPORTANT" notes under
#     "nde" in THE GENOTYPE CONTRACT above - getting either wrong means your
#     your runs won't reproduce cleanly.
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing - repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline, a good standard is at least a random search.
#   Keep the encoding, module budget and target set identical across
#     everything you compare, change one thing at a time.
#
# ============================================================================ #
