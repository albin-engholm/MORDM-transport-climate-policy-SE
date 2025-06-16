# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:43:27 2023
A script for computing and visualizing MOEA convergence metrics
@author: aengholm
"""

import seaborn as sns  # Import seaborn again if only doing the visualization
from ema_workbench.em_framework.optimization import (ArchiveLogger)
from ema_workbench import (
    HypervolumeMetric,
    GenerationalDistanceMetric,
    EpsilonIndicatorMetric,
    InvertedGenerationalDistanceMetric,
    SpacingMetric
)
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %% Load and prepare data

policy_types=["All levers", "No transport efficiency"]
policy_types = ["No transport efficiency"]  # ,
policy_types = ["All levers"]
date = '2025-03-07'  # Specify date the MORDM MOEA was started
date_archive = date
nfe_MOEA = 1000000  # Specify the number of nfes used for the MORDM MOEA

all_archives = []

for policy_type in policy_types:
    archives = ArchiveLogger.load_archives(
        f"./archives/{str(nfe_MOEA)}_{policy_type}_{date_archive}.tar.gz")  # load archive
    model_filename = './output_data/moea_results/'+policy_type + \
        str(nfe_MOEA)+"_nfe_"+"directed_search_MORDM_"+date+"model_.p"
    model = pickle.load(open(model_filename, "rb"))  # Load model
    results_filename = './output_data/moea_results/'+policy_type + \
        str(nfe_MOEA)+"_nfe_"+"directed_search_MORDM_"+date+".p"
    results_list, convergence_list, scenarios, epsilons = pickle.load(open(results_filename, "rb"))
    results_final = results_list[0]

    # Check and drop the 'unnamed' column in each dataframe if it exists
    updated_archives = {}
    for key, dataframe in archives.items():
        if "Unnamed: 0" in dataframe.columns:
            # inplace=True: modify the DataFrame in place (do not create a new object)
            dataframe = dataframe.drop(columns=['Unnamed: 0'])

        # Add the modified dataframe to the new dictionary
        updated_archives[key] = dataframe
    all_archives.append(updated_archives)
    # %% Prepare for convergence calculations

    from ema_workbench.em_framework.optimization import to_problem
    # Create a new list of outcomes excluding INFO (0)  kind
    new_outcomes = [o for o in model.outcomes if o.kind != 0]
    problem_model = model
    problem_model.outcomes = new_outcomes
    problem = to_problem(problem_model, searchover="levers")

    results_epsilon = [results_final]
    reference_set = results_epsilon[0]
    hv = HypervolumeMetric(reference_set, problem)
    gd = GenerationalDistanceMetric(reference_set, problem, d=1)
    ei = EpsilonIndicatorMetric(reference_set, problem)
    ig = InvertedGenerationalDistanceMetric(reference_set, problem, d=1)
    sm = SpacingMetric(problem)
    # %%Compute metrics for each archive
    metrics = []
    for archives in all_archives:
        counter = 0
        for nfe, archive in archives.items():
            # Add levers and set values to 0 if not in archive (to handle transport efficent society)
            for lever in model.levers:
                if lever.name not in archive.columns:
                    archive[lever.name] = 0

            # Remove archive columns with INFO outcomes
            for outcome in model.outcomes:
                if (outcome.kind == 0) & (outcome in archive.columns):  # 0 means INFO
                    archive = archive.drop(columns=[outcome.name])

            print("Calculating metrics for: Generation: " + str(counter) +
                  ", nfe: "+str(nfe)+", # solutions: "+str(len(archive)))
            convergence_list_nfe = convergence_list[0][convergence_list[0]["nfe"] == nfe]
            scores = {
                "epsilon_progress": convergence_list_nfe["epsilon_progress"],
                "generational_distance": gd.calculate(archive),
                "hypervolume": hv.calculate(archive),
                "epsilon_indicator": ei.calculate(archive),
                "inverted_gd": ig.calculate(archive),
                "spacing": sm.calculate(archive),
                "nfe": int(nfe),

            }
            metrics.append(scores)
            counter = counter+1
            #  Use break below if testing on a sub-set of generations
            # if counter > 10:
            #     break
        metrics = pd.DataFrame.from_dict(metrics)

        # sort metrics by number of function evaluations
        metrics.sort_values(by="nfe", inplace=True)
    #  Save convergence metrics dataframe to file
    if len(metrics) > 0:
        pickle.dump(metrics,
                    open(f"./output_data/moea_results/{str(nfe_MOEA)}_{policy_type}_{date_archive}_convergence_metrics.p", "wb"))
# %% visualize metrics
for policy_type in policy_types:
    #  Load convergence metrics dataframe
    import pickle
    import seaborn as sns
    import matplotlib.pyplot as plt

    metrics = pickle.load(
        open(f"./output_data/moea_results/{str(nfe_MOEA)}_{policy_type}_{date_archive}_convergence_metrics.p", "rb"))
    if isinstance(metrics, list) and len(metrics) > 0:
        metrics = metrics[0]
    # Exclude the first and last rows
    metrics = metrics.iloc[1:-1]

    sns.set_style("white")
    fig, axes = plt.subplots(nrows=6, figsize=(8, 12), sharex=True)

    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    ax1.plot(metrics.nfe, metrics.hypervolume)
    ax1.set_ylabel("hypervolume")

    ax2.plot(metrics.nfe, metrics.epsilon_progress)
    ax2.set_ylabel("$\epsilon$ progress")

    ax3.plot(metrics.nfe, metrics.generational_distance)
    ax3.set_ylabel("generational distance")

    ax4.plot(metrics.nfe, metrics.epsilon_indicator)
    ax4.set_ylabel("epsilon indicator")

    ax5.plot(metrics.nfe, metrics.inverted_gd)
    ax5.set_ylabel("inverted generational\ndistance")

    ax6.plot(metrics.nfe, metrics.spacing)
    ax6.set_ylabel("spacing")

    ax6.set_xlabel("nfe")

    sns.despine(fig)
    plt.suptitle(f"Convergence metrics for {policy_type}", y=0.9)
    plt.show()
    fig.savefig(f"./figs/paper/convergence_metrics_{str(nfe_MOEA)}_{policy_type}_{date_archive}.png",
                          dpi=300, format="png", bbox_inches="tight", transparent=True)
