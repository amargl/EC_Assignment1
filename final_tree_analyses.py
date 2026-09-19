#import packages and functions from the A1 file
# Standard library
import random
import copy
from pathlib import Path
from typing import Literal
import os
import names
import matplotlib as mpl
#new?
import json
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

# Third-party libraries
import mujoco as mj
import networkx as nx
import numpy as np
import torch
from jedi.api import file_name
from mujoco import viewer

# Local scripts
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)

from A1_template_2026 import load_targets

#set path
base_path = Path("C:/Users/anton/Documents/EvolutionaryComputing2026/assignments/assignment_1/__data__/__data__/A1_template_2026")
DATA = Path("C:/Users/anton/Documents/EvolutionaryComputing2026/assignments/assignment_1/__data__/__data__/A1_template_2026")

#define function to make files correct format?


def load_time99_graph(filepath):
    """Load a time_99 JSON file into an nx.DiGraph."""

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = nx.DiGraph()

    # Add nodes
    for node_id, node_data in data["nodes"].items():
        graph.add_node(
            int(node_id),
            type=node_data["type"],
            rotation=node_data["rotation"]
        )

    # Add edges
    for edge in data["edges"]:
        graph.add_edge(
            edge["parent"],
            edge["child"],
            face=edge["face"]
        )

    return graph

#prettier version
def show_body_tree(body: nx.DiGraph, file_name: str = "body"):
    # To better understand the inner workings of ARIEL and how the tree
    # forms a robot, we added the following helper function which lets us plot
    # the tree which represents the robot
    pos = nx.nx_agraph.graphviz_layout(body, prog="dot")

    plt.figure(figsize=(10, 7))

    # For this we first need to get all the information of the different nodes
    # and edges
    node_labels = {
        nid: f"{data.get('rotation', '').split("_")[1]}°" #add {data.get('type', '')}\n
        for nid, data in body.nodes(data=True)
    }
    #edge_labels = nx.get_edge_attributes(body, "face")
    edge_labels = {
        edge: label[0]
        for edge, label in nx.get_edge_attributes(body, "face").items()
    }

    # Separate nodes by type
    core_nodes = [
        nid for nid, data in body.nodes(data=True)
        if data.get("type") == "CORE"
    ]

    brick_nodes = [
        nid for nid, data in body.nodes(data=True)
        if data.get("type") == "BRICK"
    ]

    joint_nodes = [
        nid for nid, data in body.nodes(data=True)
        if data.get("type") == "HINGE"
    ]

   # Draw the edges first
    nx.draw_networkx_edges(
        body,
        pos,
        arrows=True,
        arrowsize=15,
        edge_color="gray",
    )

    #draw core nodes as circles
    nx.draw_networkx_nodes(
        body,
        pos,
        nodelist=core_nodes,
        node_shape="o",
        node_size=1500,
        node_color="yellow",
    )

    # Draw brick nodes as squares
    nx.draw_networkx_nodes(
        body,
        pos,
        nodelist=brick_nodes,
        node_shape="s",
        node_size=1000,
        node_color="lightblue",
    )

    # Draw joint nodes as diamonds
    nx.draw_networkx_nodes(
        body,
        pos,
        nodelist=joint_nodes,
        node_shape="D",
        node_size=1000,
        node_color="pink",
    )

    # Draw labels on top of the nodes
    nx.draw_networkx_labels(
        body,
        pos,
        labels=node_labels,
        font_size=8,
        font_weight="bold",
    )

    # Draw edge labels
    nx.draw_networkx_edge_labels(
        body,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",
            ec="none",
            alpha=0.8,
        ),
    )


    # We save a png of the image
    plt.title(file_name)
    plt.axis("off")
    plt.savefig(str(DATA / f"{file_name}_tree.png"), dpi=200, bbox_inches="tight")
    plt.close()


###################################################################################################
#load the final trees for each run from the data folder
genotypes = []




for point_folder in base_path.glob("point-*_parent-3"):

    point = int(
        point_folder.name.split("-")[1].split("_")[0]
    )

    for exp_folder in point_folder.glob("exp*"):

        experiment = int(
            exp_folder.name.replace("exp", "")
        )

        genotype_file = (
            exp_folder
            / "best_genotype"
            / "time_99"
        )

        if genotype_file.exists():

            genotype = load_time99_graph(genotype_file)

            genotypes.append({
                "point": point,
                "experiment": experiment,
                "genotype": genotype
            })

genotypes_df = pd.DataFrame(genotypes)
print(genotypes_df)

#load targets
targets = load_targets()


#calculate the final tree distances to each target
distance_results = []

for _, row in genotypes_df.iterrows():

    genotype = row["genotype"]

    for target_id, target_graph in enumerate(targets):

        distance = tree_edit_distance(
            genotype,
            target_graph
        )

        distance_results.append({
            "point": row["point"],
            "experiment": row["experiment"],
            "target": target_id,
            "distance": distance
        })

distances_df = pd.DataFrame(distance_results)
#print(distances_df)
#save file
distances_df.to_csv("tree_distances_to_targets.csv", index=False)

#tree distances between final trees in the same point condition

distance_results = []

# Group trees by point condition
for point, group in genotypes_df.groupby("point"):

    # Get all trees within this point condition
    trees = list(group[["experiment", "genotype"]].itertuples(index=False))

    # Compare every unique pair of trees
    for (exp1, tree1), (exp2, tree2) in combinations(trees, 2):

        distance = tree_edit_distance(tree1, tree2)

        distance_results.append({
            "point": point,
            "experiment_1": exp1,
            "experiment_2": exp2,
            "distance": distance
        })

# Convert to DataFrame
distances_df = pd.DataFrame(distance_results)

print(distances_df)

# Save file
distances_df.to_csv("tree_distances.csv", index=False)



print("done running distance calculations, making plots now")
#doing the plots once is enough so---
quit()


#make plot showing what they look like
for _, row in genotypes_df.iterrows():

    show_body_tree(
        row["genotype"],
        file_name=f"point_{row['point']}_exp_{row['experiment']}"
    )

for i, target in enumerate(targets):

    show_body_tree(
        target,
        file_name=f"target_{i}"
    )


#
print("done running")
