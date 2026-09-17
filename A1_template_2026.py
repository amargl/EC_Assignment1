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
P_MUTATION = 0.2
MUTATION_FUNC = "replace_node"
POINTS = 3
AMOUNT_PARENTS = 3
SAMPLE_SIZE = 15
POPULATION_SIZE = 50
""""""""""
:):):):)
"""""""""""""""

# ============================================================================ #
#  1. THE TARGET BODIES
# ============================================================================ #
#
# The targets are plain nx.DiGraph JSON files.
# They vary in size on purpose. A body that just matches the average module
# count will not score well against all of them.
#
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


# ============================================================================ #
#  2. THE GENOTYPE CONTRACT
# ============================================================================ #
#
# You may use EITHER of ARIEL's two body encodings below. You may NOT invent
# your own, and CPPN is not offered for this assignment.
# Whichever you pick, the contract is the same and it is very short:
#
#       your genotype  --(its decoder)-->  nx.DiGraph  -->  fitness
#
# That DiGraph is the phenotype, and it is all the fitness function ever sees:
#
#       nodes carry   type      : "CORE" | "BRICK" | "HINGE"
#                     rotation  : "DEG_0" | "DEG_45" | "DEG_90"
#       edges carry   face      : "FRONT" | "BACK" | "RIGHT" | "LEFT"
#                                 | "TOP" | "BOTTOM"
#
# THE TWO ENCODINGS
#
#   "nde"   NeuralDevelopmentalEncoding + HighProbabilityDecoder
#           Genotype: three fixed-length float vectors (type / connection /
#           rotation genes). An INDIRECT encoding - a small vector is expanded
#           by a fixed neural network into probability matrices, which are
#           then decoded greedily into a body.
#           -> Fixed-length real vector. Standard real-valued operators work
#              out of the box. But the genotype-phenotype map is wildly
#              non-linear: a small mutation can rebuild the robot entirely.
#           -> IMPORTANT: `NeuralDevelopmentalEncoding`'s internal network is
#              randomly (re-)initialised every time you construct it, and NOT
#              derived from the genotype you pass in. If your EA's decode step
#              builds a fresh `NeuralDevelopmentalEncoding(...)` per individual
#              (the natural way to write it - see `random_nde_body` below),
#              the SAME genotype decodes to a DIFFERENT random body every call,
#              and fitness stops reflecting the genotype at all. Construct it
#              ONCE for your whole run and reuse that one instance's
#              `.forward()` for every genotype you decode.
#           -> ALSO IMPORTANT: `NeuralDevelopmentalEncoding` is a
#              `torch.nn.Module`. Its weight initialisation uses torch's own
#              RNG, entirely separate from numpy/random. `np.random.seed(...)`
#              and `random.seed(...)` do NOT control it - you also need
#              `torch.manual_seed(...)`, or your results will not reproduce
#              across separate runs even with "the same" seed.
#
#   "tree"  TreeGenome + its operators
#           Genotype: the tree itself, nodes and edges.
#           A DIRECT encoding - genotype and phenotype are the same shape.
#           -> ariel.ec.genotypes.tree.operators already gives you
#              random_tree, add_node, remove_subtree, subtree_swap,
#              crossover_subtree, mutate_hoist, mutate_shrink,
#              mutate_replace_node, mutate_subtree_replacement.
#              Variable-length genotype, so watch for bloat.
#
#
# Below, each encoding gets ONE random genotype, decoded to a graph. That is
# your starting point, not your solution: your EA has to search this space,
# not sample it once.
#
# ============================================================================ #

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


def random_nde_body(num_modules: int = NUM_OF_MODULES) -> nx.DiGraph:
    """Sample a random NDE genotype and decode it into a body graph.

    THIS IS THE FUNCTION YOUR EA REPLACES. The three vectors below are the
    genotype: that is what you mutate, recombine and select on. Note this
    function does NOT construct its own `NeuralDevelopmentalEncoding` - it
    reuses the module-level `_NDE` instance. Do the same in your EA.

    `num_modules` must match the value `_NDE` was built with (NUM_OF_MODULES).
    """
    genotype = [
        RNG.uniform(-1.0, 1.0, GENOTYPE_SIZE).astype(np.float32)  # module types
        for _ in range(3)  # types, connections, rotations
    ]

    type_p, conn_p, rot_p = _NDE.forward(genotype)

    decoder = HighProbabilityDecoder(num_modules)
    return decoder.probability_matrices_to_graph(type_p, conn_p, rot_p)


def random_tree_body(num_modules: int = NUM_OF_MODULES) -> nx.DiGraph:
    """Sample a random tree genotype and convert it into a body graph.

    THIS IS THE FUNCTION YOUR EA REPLACES. Here the genotype IS the tree, so
    `TreeGenome` is what your population holds - call `.to_networkx()` only
    when it is time to compute fitness.
    """
    genome = random_tree(max_modules=num_modules)
    return genome.to_networkx()


def random_body(
    genotype: GenotypeTypes = GENOTYPE,
    num_modules: int = NUM_OF_MODULES,
) -> nx.DiGraph:
    """Sample one random body using the chosen encoding."""
    match genotype:
        case "nde":
            return random_nde_body(num_modules)
        case "tree":
            return random_tree_body(num_modules)


# ============================================================================ #
#  3. FITNESS
# ============================================================================ #
#
# Fitness is the MEAN tree edit distance to every target body, PLUS one
# standard deviation across those per-target distances. LOWER IS BETTER, and
# 0.0 would mean your body is identical to all of them at once - which, since
# the targets differ from each other, is impossible. There is a floor above
# zero here and you will not reach it. Work out roughly where it is: a body
# cannot be closer to a set than the set is to itself.
#
# The distance itself lives in tree_edit_distance.py.
# Read that file - you cannot reason about your EA's behaviour without knowing what it is climbing.
#
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
Helper function to show graph --GR16--
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

# # ============================================================================ #
# #  5. ENTRY POINT
# # ============================================================================ #
#
#
# def main() -> None:
#     """Score one randomly-sampled body against the target set."""
#     targets = load_targets()
#
#     console.log(f"encoding      : {GENOTYPE}")
#     console.log(f"module budget : {NUM_OF_MODULES}")
#     console.log(f"targets       : {len(targets)} bodies from {TARGET_DIR.name}")
#     console.log(
#         "target sizes  : "
#         + ", ".join(str(t.number_of_nodes()) for t in targets),
#     )
#
#     # How far apart are the targets from each other? Your fitness cannot go
#     # below the best possible compromise, and this is the clue to where that is.
#     spread = [
#         tree_edit_distance(a, b)
#         for i, a in enumerate(targets)
#         for b in targets[i + 1 :]
#     ]
#     console.log(f"target spread : mean pairwise distance {np.mean(spread):.2f}")
#
#     # --- One random body --------------------------------------------------- #
#     body = random_body(GENOTYPE, NUM_OF_MODULES)
#     fitness = fitness_function(body, targets)
#
#     console.log("")
#     console.log(f"random body   : {body.number_of_nodes()} modules")
#     console.log(
#         "per-target    : "
#         + ", ".join(f"{d:.1f}" for d in distances_to_targets(body, targets)),
#     )
#     console.log(f"fitness       : {fitness:.4f}   (lower is better)")
#
#     show_body(body, MODE, file_name=f"random_{GENOTYPE}")

""""""""""""""""""""""""""""
6. Evolution--GR16--
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

def reproduction(parents: Population,p_mutation: float, mutation_func:str, points:int= 1) -> list[Individual]:
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

    genome_length = len(genomes)
    # We apply n-point crossover
    # This simply means that we doe a crossover between genomes
    # points times
    for _ in range(points):
        # We then perform Multi-parent recombination
        # It does not matter in our case how many parents the
        for i in range(genome_length):
            # We find the nodes where we want to perform the cut:
            cut_1 = pick_noncore(genomes[i])
            cut_2 = pick_noncore(child_genomes[(i + 1)%genome_length])
            subtree_swap(genomes[i], child_genomes[(i + 1)%genome_length], cut_1, cut_2)

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
                   p_mutation = P_MUTATION,
                   points = POINTS):
    # We take a subsection of the population
    sub_pop = pop.sample(SAMPLE_SIZE)
    # from this section, we take n individuals
    # with the lowest fitness
    parents = sub_pop.best(sort="min", n=amount_parents)
    # We reproduce these individuals to
    # get the offspring
    children = reproduction(parents, mutation_func= mutation_func, p_mutation=p_mutation, points=points)
    # We add the offspring to the subsection
    # of the population and to the general population
    pop.extend(children)
    sub_pop.extend(children)
    # We then find the worst performing individuals
    # of the subsection
    worst = sub_pop.best(sort="max", n=AMOUNT_PARENTS)
    # We kill all the worst individuals in the
    # subset
    for w in worst:
        w.alive = False
    # We then return all the alive individuals in
    # the population
    return pop.alive

def experiment(experiment:str,targets,
               timesteps:int=50,
               amount_parents:int = AMOUNT_PARENTS,
               mutation_func:str = MUTATION_FUNC,
               p_mutation = P_MUTATION,
               points = POINTS):
    # We initialize a population
    pop = initialize_population(targets, POPULATION_SIZE)
    show_body_tree(pop[0].genotype.to_networkx(),file_name="start")
    os.makedirs(DATA / experiment / "best_genotype", exist_ok=True)
    os.makedirs(DATA / experiment / "statistics", exist_ok=True)

    with open(DATA / experiment / "statistics"/"general_stats", "w") as f:
        for i in range(timesteps):
            # We perform an evolution step
            pop = evolution_step(pop)
            # We determine the best of the populatio
            best = pop.best(sort="min", n=1)[0]
            # We save the genotype of the fittest individual as a
            # jSon file
            best.genotype.save_json(DATA / experiment / "best_genotype"/ f"time_{i}")
            # We keep track of parameters about general
            # statistics of the performance
            total_fitness =  np.array([pop.fitness for pop in pop])
            # We write down the statistics
            f.write(f"{best.fitness:.2f}\t{float(np.mean(total_fitness)):.2f}\n")


def plot_stats(experiment:str):
    best_fitness = []
    mean_std_fitness = []
    timesteps = 0
    with open(DATA / experiment / "statistics"/"general_stats", "r") as f:
        for l in f.readlines():
            info = l.split("\t")
            best_fitness.append(float(info[0]))
            mean_std_fitness.append(float(info[1]))
            timesteps +=1

    # We make a simple plot for all the data
    plt.plot(range(timesteps), mean_std_fitness, label="Mean Fitness")
    plt.plot(range(timesteps), best_fitness, linestyle="--", label="Best Fitness")
    plt.legend()
    plt.savefig(str(DATA / experiment / "statistics"/"general_stats_plot.svg"), dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    """""""""
    "Before running a experiment, make sure to change the experiment run:"
    """""""""
    TIMESTEPS = 50
    EXPERIMENTAL_RUNS = 5
    # We load the targets
    targets = load_targets()
    # These are the values that we want to test
    point_values = [1,2]
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
            for i in range(EXPERIMENTAL_RUNS):
                print(".", end="")
                experiment(f"point-{point}_parent-{amount}/exp{i}",
                           targets,
                           timesteps=TIMESTEPS,
                           amount_parents=amount,
                           points=point)
                plot_stats(f"point-{point}_parent-{amount}/exp{i}")
                # We change the seed of our random valuebles
                # every time we perform an experiment
                SEED +=1
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
