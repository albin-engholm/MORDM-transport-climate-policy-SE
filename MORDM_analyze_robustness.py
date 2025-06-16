# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:18:42 2023
Script for performing uncertainty and robustness analysis.
@author: aengholm
"""
# Imports
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import math
import pickle
import statistics
import pandas as pd
import numpy as np
import re
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
import matplotlib.patches as patches
from ema_workbench.analysis import feature_scoring

from ema_workbench.analysis import prim
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from ema_workbench.analysis import parcoords
import matplotlib.pyplot as plt
import seaborn as sns
from sta_policies import df_sta 
# Set plotting style and scale up font size
sns.set_context("paper", font_scale=1.5)

# %% Load data and store in dataframes for subsequent analysis
policy_types = ["All levers", "No transport efficiency"]
# policy_types=["All levers"]
# "No transport efficiency"]

load_results = 1
if load_results == 1:
    from ema_workbench import load_results
    date = "2025-03-10"
    n_scenarios = 10000
    # for policy_type in policy_types:
    t1 = f"./output_data/robustness_analysis_results/X_XP{n_scenarios}_scenarios_MORDM_OE_{date}.p"

    data = pickle.load(open(t1, "rb"))
    if len(data) > 1:
        data_x = data[0][0]  # Load output data for run for parameter set X
        data_xp = data[0][1]  # Load output data for run for parameter set XP

        experiments_x = data_x[1]
        outcomes_x = data_x[0]
        experiments_xp = data_xp[1]
        outcomes_xp = data_xp[0]
        df_outcomes = pd.concat([pd.DataFrame(outcomes_x), pd.DataFrame(outcomes_xp)],
                                axis=0, join="inner").reset_index(drop=True)
        experiments = pd.concat([experiments_x, experiments_xp], axis=0, join="outer").reset_index(drop=True)
    else:
        data_xp = data[0][0]  # Load output data for run for parameter set XP

        experiments_xp = data_xp[1]
        outcomes_xp = data_xp[0]
        df_outcomes = pd.DataFrame(outcomes_xp).reset_index(drop=True)
        experiments = experiments_xp
    # experiments, outcomes = pickle.load(open(t1, "rb"))
    t2 = f"./output_data/robustness_analysis_results/{n_scenarios}_scenarios_MORDM_OE_{date}model_.p"
    model = pickle.load(open(t2, "rb"))

    # DF with both experiments and outcomes
    df_full = pd.concat([experiments, df_outcomes], axis=1, join="inner")
    
    
    df_outcomes['policy'] = experiments["policy"]
    
    #Rename STA policies here
    matching_cols = [
    'L1_bio_share_diesel', 'L2_bio_share_gasoline',
    'L3_fuel_tax_increase_gasoline', 'L4_fuel_tax_increase_diesel',
    'L5_km_tax_cars', 'L6_km_tax_trucks',
    'L7_additional_car_energy_efficiency',
    'L8_additional_truck_energy_efficiency',
    'L9_transport_efficient_planning_cars',
    'L10_transport_efficient_planning_trucks'
    ]
    
    df_full['policy'] = df_full['policy'].astype(str)
    
    # then run your existing code:
    sta_mask = df_full['Policy type'] == 'STA'
    for idx, row in df_full.loc[sta_mask].iterrows():
        match_mask = (df_sta[matching_cols] == row[matching_cols]).all(axis=1)
        if match_mask.any():
            matched_index = df_sta.index[match_mask][0]
            df_full.at[idx, 'policy'] = matched_index

    # Define the replacements mapping
    replacements = {
        'C1': 'BT1',
        'C2': 'BT2',
        'C3': 'BT3',
        'C4': 'BT4',
        'D1': 'BTE1',
        'D2': 'BTE2',
        'D3': 'BTE3'
    }
    
    # Perform the replacement
    df_full['policy'] = df_full['policy'].replace(replacements)
    
    for index, row in df_full.iterrows():
        if math.isnan(row["R1_fuel_price_to_car_electrification"]):
            df_full.loc[index, "Uncertainty set"] = "X"
        else:
            df_full.loc[index, "Uncertainty set"] = "XP"

    df_full["CO2 target not met"] = df_full['M1_CO2_TTW_total'] > 1.89
    df_full["Bio share of fuel energy"] = df_full["M4_energy_use_bio"] / \
        (df_full["Energy total"]-df_full["M5_energy_use_electricity"])
    # Create helper lists of outcomes, levers and uncertainties

    # Dictionary mapping keys to labels with units
    key_outcomes_labels = {
        'M1_CO2_TTW_total': 'M1_CO2_TTW_total [ton]',  # Replace 'metric unit' with the actual unit for M1
        'M2_driving_cost_car': 'M2_driving_cost_car [SEK]',
        'M3_driving_cost_truck': 'M3_driving_cost_truck [SEK]',
        'M4_energy_use_bio': 'M4_energy_use_bio [TWh]',
        'M5_energy_use_electricity': 'M5_energy_use_electricity [TWh]'
    }

    # Dictionary for objective outcomes labels with units
    objective_outcomes_labels = {
        'M2_driving_cost_car': 'M2_driving_cost_car [SEK]',
        'M3_driving_cost_truck': 'M3_driving_cost_truck [SEK]',
        'M4_energy_use_bio': 'M4_energy_use_bio [TWh]',
        'M5_energy_use_electricity': 'M5_energy_use_electricity [TWh]'
    }

    # Different outcome sets
    all_outcomes = model.outcomes.keys()
    key_outcomes = [
        'M1_CO2_TTW_total',
        'M2_driving_cost_car',
        'M3_driving_cost_truck',
        'M4_energy_use_bio',
        'M5_energy_use_electricity']
    objective_outcomes = [
        'M2_driving_cost_car',
        'M3_driving_cost_truck',
        'M4_energy_use_bio',
        'M5_energy_use_electricity']

    # Levers
    levers = model.levers.keys()

    # Uncertainties
    uncertainties = model.uncertainties.keys()

    # Define color coding mapping
    color_coding = {
        "All levers": '#0005CC',
        "No transport efficiency": '#05CC00',
        "STA": '#CC0005'}

    light_color_coding = {
        "All levers": '#6064CC',
        "No transport efficiency": '#7ECC60',
        "STA": '#C77F5D'
    }

    ultra_light_color_coding = {
        "All levers": '#EDEDFF',
        "No transport efficiency": '#EDFFED',
        "STA": '#FFEDED'
    }

    STA_colors = {
        'B': '#ff0000',  # Red for policy B
        'BT1': '#007bff',  # A shade of blue for C1
        'BT2': '#0056b3',  # A darker shade of blue for C2
        'BT3': '#003d80',  # Even darker shade of blue for C3
        'BT4': '#002366',  # Darkest blue for C4
        'BTE1': '#28a745',  # Green for D1
        'BTE2': '#1e7e34',  # A darker shade of green for D2
        'BTE3': '#155724'  # Darkest green for D3
    }
    # Speciy latex math strings for each robustness metric
    rm_dict = {
        "90_percentile_deviation": r"$\mathrm{RM_{90\%\,dev}}$",
        "Mean_stdev": r"$\mathrm{RM_{MeanStdev}}$",
        "Satisficing metric M1_CO2_TTW_total": "RMco$_{2}$sat",
    }

    color_sta_policies = ["green", "lightcoral", "red", "darkred", "coral", "cornflowerblue", "blue", "darkblue"]
    df_full_all = df_full.copy()
    df_full = df_full[df_full["policy"] != "Reference policy"]
    df_full_sta = df_full[df_full["Policy type"] == "STA"]
    df_sta_levers = df_full_sta[list(levers)+["policy"]].drop_duplicates()
    df_reference_subset = df_full[(df_full["scenario"] == "Reference") & (df_full["Uncertainty set"] == "XP")]

#%% Paraxes of policies on policy levers
    # from ema_workbench.analysis import parcoords
    # import matplotlib.pyplot as plt

    # # Levers to be plotted
    # levers = model.levers.keys()
    # # Construct lever-based limits
    # limits_levers = pd.DataFrame(index=['min', 'max'])
    # for item in levers:
    #     limits_levers[item] = [df_full[item].min(), df_full[item].max()]

    # # Initialize parallel coordinates plot
    # paraxes = parcoords.ParallelAxes(limits_levers)
    # paraxes.plot(df_full,
    #              color='lightgray')  # Non-selected policies in gray

    # # Color map for unique policy types
    # colors = plt.cm.tab10(range(len(df_full['Policy type'].unique())))

    # # Plot selected policies with unique colors
    # for idx, policy_type in enumerate(df_full['Policy type'].unique()):
    #     selected_data = df_full[df_full['Policy type'] == policy_type]
    #     paraxes.plot(selected_data, label=f'Policy type: {policy_type}', color=colors[idx])

    # # Add legend, title, and display plot
    # paraxes.legend()
    # plt.title("Parallel Coordinates Plot of Policies")
    # plt.show()

# %% More plots on reference scenario performance
# Pairplot outcomes on outcomes
df_reference_subset = df_reference_subset.copy()
df_reference_subset['Policy type'] = df_reference_subset['Policy type'].astype('category')
df_reference_subset['Policy type'] = df_reference_subset['Policy type'].cat.reorder_categories(
    ['STA', 'All levers', 'No transport efficiency'], ordered=True
)
# Now create the pairplot.
g = sns.pairplot(
    data=df_reference_subset,
    x_vars=objective_outcomes,
    y_vars=objective_outcomes,
    hue="Policy type",
    palette=color_coding,
    diag_kind="hist",
    diag_kws={"common_norm": False}
)

# Move the legend to the right of the figure.
# You can adjust the values inside bbox_to_anchor to best fit your figure's size.
g._legend.remove()
g.add_legend(title='', bbox_to_anchor=(1, .5), loc='center right', fontsize=16, ncols=1)

for ax in g.axes.flatten():
    # Set new font sizes here
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)  # Adjust fontsize as needed for x labels
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    ax.tick_params(axis='x', labelsize=12)  # Adjust labelsize as needed for x ticks
    ax.tick_params(axis='y', labelsize=12)  # Adjust labelsize as needed for y ticks


# Adjust the figure to make space for the legend
# plt.subplots_adjust(right=0.9)  # You might need to tweak this value.

# %% Pairplots
# Levers on levers
plt.figure(figsize=(16, 16))
sns.pairplot(data=df_reference_subset, x_vars=levers, y_vars=levers,
             hue="Policy type", palette=color_coding, diag_kws={"common_norm": False})
# Levers on outcomes
plt.figure(figsize=(16, 8))
sns.pairplot(data=df_reference_subset, x_vars=levers, y_vars=key_outcomes,
             hue="Policy type", palette=color_coding, diag_kws={"common_norm": False})
# %% Outcomes on outcomes
plt.figure(figsize=(8, 6))

# Assuming 'df_reference_subset' is pre-filtered for a specific uncertainty set,
# if not, you would need to filter it as in the previous loop
all_policy_types = df_reference_subset["Policy type"].unique()

# Create the pairplot
plot = sns.pairplot(data=df_reference_subset, x_vars=objective_outcomes, y_vars=objective_outcomes,
                    hue="Policy type", palette=color_coding, diag_kws={"common_norm": False})

# Annotate Pearson correlation coefficient for each policy type
for i, policy_type in enumerate(all_policy_types):
    policy_type_df = df_reference_subset[df_reference_subset['Policy type'] == policy_type]
    spearman_correlation = policy_type_df[objective_outcomes].corr(method='spearman')

    # Offset for the vertical position based on the policy type index
    vertical_offset = 0.95 - i * 0.12  # Adjust offset as needed

    # Loop through axes to annotate. Exclude diagonal.
    for j in range(len(objective_outcomes)):
        for k in range(len(objective_outcomes)):
            if j != k:  # Exclude diagonal
                ax = plot.axes[j, k]
                rho = spearman_correlation.iloc[j, k]
                n = policy_type_df.shape[0]
                # Calculate test statistic
                if n > 2:
                    t_stat = rho * ((n - 2) ** 0.5) / ((1 - rho**2) ** 0.5)
                    # Calculate one-sided p-value for negative alternative hypothesis
                    p_value = stats.t.cdf(t_stat, df=n-2)  # Use CDF to get the probability of t <= t_stat
                    signif = '*' if p_value < 0.01 else ''
                else:
                    signif = ''

                # Annotate in the upper right corner with offset for each policy type
                text = f'ρ = {rho:.2f}{signif}'
                ax.text(0.95, vertical_offset, text,
                        horizontalalignment='right', verticalalignment='center',
                        transform=ax.transAxes, color=color_coding[policy_type], fontsize=12)

# Adjust the title, labels, legend, and layout as needed

# Get handles and labels for the legend
handles, labels = plot.axes[0][0].get_legend_handles_labels()

# Create new legend on the right side
plt.legend(handles=handles, labels=labels, bbox_to_anchor=(2, 0.5), loc='center left', borderaxespad=0.)
plt.tight_layout()


# %%Relationship between biofuels and electrification rate in reference scenario Figure s.3
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_full[df_full["scenario"] == "Reference"], x='Electrified VKT share light vehicles',
                y='M4_energy_use_bio', hue="Policy type", palette=color_coding, legend=False, s=25)
# Annotating STA policies
bbox_props = dict(boxstyle="round,pad=0.1", fc="white", ec="none", lw=0, alpha=0.5)

for sta_policy in df_full[df_full["Policy type"] == "STA"]["policy"].unique():
    df_sta_policy = df_full[df_full['policy'] == sta_policy]

    plt.annotate(
        sta_policy,
        (df_sta_policy["Electrified VKT share light vehicles"].values[0],
         df_sta_policy["M4_energy_use_bio"].values[0]),
        textcoords="offset points",
        xytext=(0, -15),
        ha='center', color=color_coding["STA"], fontsize=12, bbox=bbox_props
    )
plt.axvline(x=0.68, linestyle="--", color="black", linewidth=1, zorder=10)
plt.text(s="X6_car_electrification_rate: 0.68", x=0.68+.001, y=4, color="black")
plt.xlim([0.66, .85])
sns.despine()


# %% Distribution of electrification outcomes
vars_pairplot = ["Electrified VKT share light vehicles", "Bio share of fuel energy"]
g = sns.pairplot(data=df_full, x_vars=vars_pairplot,
                 y_vars=vars_pairplot, hue="CO2 target not met",
                 palette=["green", "red"], plot_kws=dict(s=.01, alpha=1))
# %% Distribution of electrification and bio outcomes
g = sns.pairplot(data=df_full, x_vars=vars_pairplot, kind="kde",
                 y_vars=vars_pairplot, hue="CO2 target not met",
                 palette=["green", "red"], plot_kws=dict(fill=True, alpha=0.5, levels=10, gridsize=100))

# %% Calculate robustness metrics
# Define robustness metrics
all_robustness_metrics = ["90th percentile", "Reference", "90_percentile_deviation",
                          "Max", "Mean", "Standard deviation", "Mean_stdev", "Max_regret"]
robustness_metrics = ["90_percentile_deviation", "Mean_stdev"]

# Initial setup - Calculating Zero regret values

results = []
grouped = df_full.groupby(['Uncertainty set', 'scenario'])
for (uncertainty_set, scenario), group in grouped:
    for outcome in key_outcomes:
        zero_regret_value = group[outcome].min()
        results.append({
            "Outcome": outcome,
            "Uncertainty set": uncertainty_set,
            "scenario": scenario,
            "Zero regret": zero_regret_value
        })

zero_regret_df = pd.DataFrame(results)


# Calculate other metrics for each policy
all_policies = df_full["policy"].unique()
all_uncertainty_sets = df_full["Uncertainty set"].unique()

metrics_data = []
long_metrics_data = []  # For creating the long format DataFrame

for policy in all_policies:
    if policy != "Reference policy":
        for uncertainty_set in all_uncertainty_sets:
            df_temp = df_full[(df_full["policy"] == policy) & (df_full["Uncertainty set"] == uncertainty_set)]
            policy_data = {
                "policy": policy,
                "Policy type": df_temp["Policy type"].iloc[0],
                "Uncertainty set": uncertainty_set
            }

            co2_outcome_data = df_temp["M1_CO2_TTW_total"]
            satisficing_metric = (co2_outcome_data < 1.89).sum() / len(co2_outcome_data)

            # Add satisficing metric to policy_data
            policy_data["Satisficing metric M1_CO2_TTW_total"] = satisficing_metric

            for outcome in key_outcomes:
                outcome_data = df_temp[outcome]
                ref_outcome_data = df_temp[df_temp["scenario"] == "Reference"][outcome]

                metrics = {
                    f"{metric} {outcome}": (
                        outcome_data.quantile(0.9),
                        ref_outcome_data.values[0],
                        (outcome_data.quantile(0.9) - ref_outcome_data.values[0]) / abs(ref_outcome_data.values[0]),
                        outcome_data.max(),
                        outcome_data.mean(),
                        outcome_data.std(),
                        (outcome_data.mean()) * (outcome_data.std()),
                        None  # Placeholder for Max_regret which will be calculated later
                    )[all_robustness_metrics.index(metric)] for metric in all_robustness_metrics
                }

                policy_data.update(metrics)

                regrets = abs(outcome_data - zero_regret_df.loc[
                    (zero_regret_df["Uncertainty set"] == uncertainty_set) &
                    (zero_regret_df["Outcome"] == outcome) &
                    zero_regret_df["scenario"].isin(df_temp["scenario"].unique()),
                    "Zero regret"
                ].values)

                policy_data[f"Max_regret {outcome}"] = regrets.max()
                metrics[f"Max_regret {outcome}"] = regrets.max()  # Update the metrics dictionary

                # Append data to long_metrics_data
                for metric in all_robustness_metrics:
                    long_metrics_data.append({
                        "policy": policy,
                        "Policy type": df_temp["Policy type"].iloc[0],
                        "Uncertainty set": uncertainty_set,
                        "Outcome": outcome,
                        "Robustness metric": metric,
                        "Value": metrics[f"{metric} {outcome}"]
                    })

            metrics_data.append(policy_data)

policy_metrics_df = pd.DataFrame(metrics_data)
policy_metrics_df_long = pd.DataFrame(long_metrics_data)
policy_metrics_sta = policy_metrics_df[policy_metrics_df["Policy type"] == "STA"]
policy_metrics_sta_long = policy_metrics_df_long[policy_metrics_df_long["Policy type"] == "STA"]
# %% FIGURE18 19 visualize relationship between robustness metrics
# Use Whitegrid
sns.set_style("whitegrid")

all_uncertainty_sets = policy_metrics_df["Uncertainty set"].unique()
all_policy_types = policy_metrics_df["Policy type"].unique()

# Loop over each outcome and uncertainty set combination
rm_pairplot = ["90_percentile_deviation", "Mean_stdev"]
for rm in rm_pairplot:
    pairplot_metrics = []
    for outcome in key_outcomes:
        pairplot_metrics.append(f"{rm} {outcome}")

    for uncertainty_set in all_uncertainty_sets:
        # Filter the data
        subset_df = policy_metrics_df[policy_metrics_df["Uncertainty set"] == uncertainty_set]

        # Create the pairplot
        plot = sns.pairplot(data=subset_df, vars=pairplot_metrics, diag_kind="kde", hue="Policy type",
                            height=2.5, palette=color_coding, diag_kws={"common_norm": False})
        plot.fig.suptitle(f"Robustness metric: {rm}", y=1, fontsize=20)

        # Annotate Spearman correlation coefficient and its significance for each policy type
        for i, policy_type in enumerate(all_policy_types):
            # Offset the vertical position of each annotation by the index of the policy type
            vertical_offset = 0.95 - i * 0.12  # Adjust offset as needed

            policy_type_df = subset_df[subset_df['Policy type'] == policy_type]
            spearman_correlation = policy_type_df[pairplot_metrics].corr(method='spearman')

            # Loop through axes to annotate. Exclude diagonal.
            for j in range(len(pairplot_metrics)):
                for k in range(len(pairplot_metrics)):
                    if j != k:  # Exclude diagonal
                        ax = plot.axes[j, k]
                        rho = spearman_correlation.iloc[j, k]
                        n = policy_type_df.shape[0]
                        # Calculate test statistic
                        if n > 2:
                            t_stat = rho * ((n - 2) ** 0.5) / ((1 - rho**2) ** 0.5)
                            # Calculate one-sided p-value for negative alternative hypothesis
                            p_value = stats.t.cdf(t_stat, df=n-2)  # Use CDF to get the probability of t <= t_stat
                            signif = '*' if p_value < 0.01 else ''
                        else:
                            signif = ''
                        text = f'ρ = {rho:.2f}{signif}'
                        ax.text(0.95, vertical_offset, text,
                                horizontalalignment='right', verticalalignment='center',
                                transform=ax.transAxes, color=color_coding[policy_type], fontsize=12)
        # Adjust labels for better readability
        for ax in plot.axes.flat:
            ax.set_xlabel(ax.get_xlabel().replace(" ", "\n"), rotation=0)
            ax.set_ylabel(ax.get_ylabel().replace(" ", "\n"), rotation=90)

        plot._legend.set_bbox_to_anchor((1.15, 0.5))
        plt.tight_layout()

# %% Visualize satisficing CO2
g = sns.displot(policy_metrics_df, x="Satisficing metric M1_CO2_TTW_total",
                hue="Policy type", kind="kde", common_norm=False, bw_adjust=0.5, cut=0,
                palette=color_coding, fill=False)
g._legend.set_bbox_to_anchor((0.7, 0.9, 0, 0))
# Remove the legend title
g._legend.set_title('')


g = sns.FacetGrid(policy_metrics_df, col="Policy type", col_wrap=3, height=4,
                  sharex=True, sharey=False, hue="Policy type", palette=color_coding)

# Plot histograms
g = g.map(sns.histplot, "Satisficing metric M1_CO2_TTW_total",
          stat="count", element="step", bins=10).set(xlim=(None, 1))
for ax in g.axes.flatten():
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# Adjust x lables
g.set_axis_labels(x_var="RMco$_{2}$sat", y_var="").set_titles("Policy type: {col_name}")
g.fig.tight_layout()

plt.show()
# %% Visualize satisficing CO2 as function of car electrification rate
g = sns.displot(df_full,  x="X6_car_electrification_rate",
                hue="CO2 target not met", kind="kde", common_norm=False, bw_adjust=0.5, cut=0,
                fill=False)
g._legend.set_bbox_to_anchor((0.7, 0.9, 0, 0))
# Remove the legend title
g._legend.set_title('')

# %%


df_long_el = df_full.melt(id_vars=["Policy type", "CO2 target not met"],
                          value_vars=["X6_car_electrification_rate", "Electrified VKT share light vehicles"],
                          var_name="Variable", value_name="value")


# Calculate the x_value for vertical lines
x_value = df_reference_subset["X6_car_electrification_rate"].unique()[0]

# Use displot with faceting
custom_palette = ['green', 'red']

g = sns.displot(
    data=df_long_el,
    x="value",
    row="Variable",
    col="Policy type",
    hue="CO2 target not met",
    kind="kde",
    height=3,
    aspect=1.5,
    common_norm=True,
    palette=custom_palette,
    cut=0,
    fill=True,
    facet_kws={'sharex': False, 'sharey': False}
)

g.set_titles("{col_name}")

g.map_dataframe(lambda data, color: plt.axvline(x=x_value, color='black', linestyle='--'))

# Iterate over each row and set the x-axis labels and limits
xlims = [0, 1]  # Get the x-limits from the first subplot
for i, variable in enumerate(df_long_el["Variable"].unique()):
    for j, ax in enumerate(g.axes[i, :]):  # Iterate over each column in the row
        ax.set_xlabel(variable)
        ax.set_xlim(xlims)  # Set the same x-limits for each subplot to align them as if sharex=True
plt.subplots_adjust(hspace=0.5)
# Show plot with adjusted x-axis labels
plt.show()
# %%
for pt in df_full["Policy type"].unique():
    plt.figure()
    num_bins_x = 5  # Number of bins for 'X6_car_electrification_rate'
    num_bins_y = 5  # Number of bins for 'L2_bio_share_gasoline'

    # Create binned categories for the two columns
    df_full['X6_binned'] = pd.cut(df_full['X6_car_electrification_rate'], bins=num_bins_x)
    df_full['L2_binned'] = pd.cut(df_full['L2_bio_share_gasoline'], bins=num_bins_y)

    # Use pivot_table to aggregate duplicate entries
    pivoted_data = df_full[df_full["Policy type"] == pt].pivot_table(
        index='X6_binned', columns='L2_binned', values='CO2 target not met', aggfunc='mean')

    # Create the heatmap
    plt.figure(figsize=(10, 8))  # You can adjust the figure size as needed
    sns.heatmap(pivoted_data, annot=True, fmt=".2f", cmap="viridis")
    plt.tight_layout()

    #sns.scatterplot(df_full[df_full["Policy type"]==pt], x="X6_car_electrification_rate",y="L2_bio_share_gasoline", hue="CO2 target not met",s=2)
# %% FIGURE 5  deprecated Visualize distributions of different policy metrics

# Create the FacetGrid with specified aspect ratio
g = sns.FacetGrid(policy_metrics_df_long[policy_metrics_df_long["Robustness metric"].isin(robustness_metrics)],
                  row='Robustness metric', col='Outcome', sharey=False, aspect=0.5)

# Map the boxplot to the grid
g.map_dataframe(sns.boxplot, x="Policy type", y="Value", hue="Policy type",
                palette=color_coding, width=0.8, dodge=False)

# Set the y-axis label for each row
# Adjust with actual row titles from 'rm_dict'
row_titles = [rm_dict["90_percentile_deviation"], rm_dict["Mean_stdev"]]
for ax, row_title in zip(g.axes[:, 0], row_titles):
    ax.set_ylabel(row_title)

# Set the x-axis labels for each column
# Ensure that 'objective_outcomes' contains the column titles you want to display
for ax, col_title in zip(g.axes[0, :], objective_outcomes):
    ax.set_title(col_title)

# Remove the default Seaborn FacetGrid titles and any redundant x-ticks
for ax in g.axes.flat:
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_title('')

# Set the FacetGrid titles again, without the row titles
g.set_titles(col_template="{col_name}", row_template="")
for ax in g.axes.flat:
    title = ax.get_title()
    if "|" in title:
        new_title = title.replace(" | ", " ")  # Replace the separator with a space
        ax.set_title(new_title)


# Adjust the legend
g.add_legend(title=None, bbox_to_anchor=(0.5, 1.05), loc='center', borderaxespad=0., ncol=3, fontsize=12)

# Improve the layout
plt.tight_layout()

# Despine to clean up the look of the plot
g.despine()
#%% Figure 5 swarmplot Visualize distributions of different policy metrics
sns.set_style("whitegrid")
# Create the FacetGrid with specified aspect ratio
g = sns.FacetGrid(policy_metrics_df_long[policy_metrics_df_long["Robustness metric"].isin(robustness_metrics)],
                  row='Robustness metric', col='Outcome', sharey=False, aspect=1)

# Map the boxplot to the grid
g.map_dataframe(sns.swarmplot, x="Policy type", y="Value", hue="Policy type",
                palette=color_coding, dodge=False, s=5)

# Set the y-axis label for each row
# Adjust with actual row titles from 'rm_dict'
row_titles = [rm_dict["90_percentile_deviation"], rm_dict["Mean_stdev"]]
for ax, row_title in zip(g.axes[:, 0], row_titles):
    ax.set_ylabel(row_title)

# Set the x-axis labels for each column
# Ensure that 'objective_outcomes' contains the column titles you want to display
for ax, col_title in zip(g.axes[0, :], objective_outcomes):
    ax.set_title(col_title)

# Remove the default Seaborn FacetGrid titles and any redundant x-ticks
for ax in g.axes.flat:
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_title('')

# Set the FacetGrid titles again, without the row titles
g.set_titles(col_template="{col_name}", row_template="")
for ax in g.axes.flat:
    title = ax.get_title()
    if "|" in title:
        new_title = title.replace(" | ", " ")  # Replace the separator with a space
        ax.set_title(new_title)


# Adjust the legend
g.add_legend(title=None, bbox_to_anchor=(0.5, 1.05), loc='center', borderaxespad=0., ncol=3, fontsize=16)

# Improve the layout
plt.tight_layout()

# Despine to clean up the look of the plot
g.despine()
# %% FIGURE 6
# Create the boxplot
plt.figure(figsize=(6, 6))  # You can adjust the figure size as needed
g = sns.boxplot(data=policy_metrics_df, x="Policy type", y="Satisficing metric M1_CO2_TTW_total", palette=color_coding)

# Remove the x-axis label and tick labels
g.set_xlabel('')
g.set_xticklabels([])

# Add a custom y-axis label
g.set_ylabel(rm_dict["Satisficing metric M1_CO2_TTW_total"], fontsize=20)

# Get the unique policy types and create a patch handle for each
policy_types = policy_metrics_df['Policy type'].unique()
legend_handles = [mpatches.Patch(color=color_coding[pt], label=pt) for pt in policy_types]

# Create the legend with rectangular handles
plt.legend(handles=legend_handles, title=None, bbox_to_anchor=(0.5, 1.05),
           loc='center', borderaxespad=0., ncol=3, fontsize=16, frameon=False)
plt.ylim([0, 1])

# Despine to clean up the look of the plot
sns.despine()

# Use tight_layout to adjust subplot params for the figure area
plt.tight_layout()
#%%
plt.figure(figsize=(6, 6))  # You can adjust the figure size as needed
g = sns.swarmplot(data=policy_metrics_df, x="Policy type",
                  y="Satisficing metric M1_CO2_TTW_total", 
                  palette=color_coding, s=8)

# Remove the x-axis label and tick labels
g.set_xlabel('')
g.set_xticklabels([])

# Add a custom y-axis label
g.set_ylabel(rm_dict["Satisficing metric M1_CO2_TTW_total"], fontsize=20)

# Get the unique policy types and create a patch handle for each
policy_types = policy_metrics_df['Policy type'].unique()
legend_handles = [mpatches.Patch(color=color_coding[pt], label=pt) for pt in policy_types]

# Create the legend with rectangular handles
plt.legend(handles=legend_handles, title=None, bbox_to_anchor=(0.5, 1.05),
           loc='center', borderaxespad=0., ncol=3, fontsize=16, frameon=False)
plt.ylim([0.5, 1])

# Despine to clean up the look of the plot
sns.despine()

# Use tight_layout to adjust subplot params for the figure area
plt.tight_layout()

# %% Pairplot of robustness
# Create a dictionary for renaming columns

for robustness_metric in robustness_metrics:
    robustness_metric_outcomes = []
    for outcome in key_outcomes:
        robustness_metric_outcomes.append(f"{robustness_metric} {outcome}")
    rename_dict = {long: short for long, short in zip(robustness_metric_outcomes, key_outcomes)}
    sns.pairplot(policy_metrics_df, x_vars=robustness_metric_outcomes,
                 y_vars=robustness_metric_outcomes, hue="Policy type", palette=color_coding)
# %% Perform statistical test of robustness metrics differences ALl levers No transport efficiency
for robustness_metric in robustness_metrics:
    for outcome in key_outcomes:

        # Filter data for the two specific policy types
        filtered_data = policy_metrics_df_long[policy_metrics_df_long['Policy type'].isin(
            ['All levers', 'No transport efficiency'])]

        # Statistical Test - Mann-Whitney U Test
        group1 = filtered_data[(filtered_data['Policy type'] == 'All levers') & (
            filtered_data['Outcome'] == outcome) & (filtered_data["Robustness metric"] == f"{robustness_metric}")]
        group2 = filtered_data[(filtered_data['Policy type'] == 'No transport efficiency') & (
            filtered_data['Outcome'] == outcome) & (filtered_data["Robustness metric"] == f"{robustness_metric}")]

        # Calculate Descriptive Statistics
        group1_stats = group1.describe()
        group2_stats = group2.describe()

        print(
            f"{robustness_metric} {outcome} - All levers: Median = {group1_stats.loc['50%'][0]}, Mean = {group1_stats.loc['mean'][0]}")
        print(
            f"{robustness_metric} {outcome} - No transport efficiency: Median = {group2_stats.loc['50%'][0]}, Mean = {group2_stats.loc['mean'][0]}")

        # Perform the test (use alternative='two-sided' for a two-tailed test)
        u_statistic, p_value = stats.mannwhitneyu(
            group1['Value'].dropna(), group2['Value'].dropna(), alternative='two-sided')
        print(f'Mann-Whitney U test for {robustness_metric} {outcome}: p-value = {p_value}')

# %% Figure 7 Parcoords of all policies and robustness metrics with empahis on STA or a subset of policies, two different graphs
# Save the original default size
original_figsize = plt.rcParams["figure.figsize"]

# Set the figure size
plt.rcParams["figure.figsize"] = [7, 6]
# Max/min/median All Levers and no transport efficiency policies highlighted
# Identify what policies to highlight and store the policy id in a list
highlighted_policy_info = {}
metric_to_use = '90_percentile_deviation M1_CO2_TTW_total'
for policy_type in ["All levers", "No transport efficiency"]:
    df_pt = policy_metrics_df[policy_metrics_df["Policy type"] == policy_type]
    # Max
    # Find indices for max, min, and median
    max_policy_idx = df_pt[metric_to_use].idxmax()
    min_policy_idx = df_pt[metric_to_use].idxmin()
    median_value = df_pt[metric_to_use].median()
    median_policy_idx = (df_pt[metric_to_use] - median_value).abs().idxmin()

    # Store the information in the dictionary
    highlighted_policy_info[policy_type] = [
        (int(df_pt.loc[max_policy_idx, 'policy']), "solid"),
        (int(df_pt.loc[min_policy_idx, 'policy']), "dotted"),
        (int(df_pt.loc[median_policy_idx, 'policy']), "dashdot")
    ]
linestyles = ["solid", "dotted", "dashdot", "solid", "dotted", "dashdot"]

for robustness_metric in robustness_metrics:
    dynamic_outcomes = [f"{robustness_metric} {outcome}" for outcome in key_outcomes]  # Dynamic outcomes

    # for uncertainty_set in policy_metrics_df["Uncertainty set"].unique():
    for uncertainty_set in ["XP"]:
        policy_metrics_subset = policy_metrics_df[policy_metrics_df["Uncertainty set"] == uncertainty_set]

        limits_outcomes = pd.DataFrame()  # Create a dataframe for limits
        for i, item in enumerate(dynamic_outcomes):
            limits_outcomes.loc[0, item] = min(policy_metrics_subset[item])  # Get lower bound
            limits_outcomes.loc[1, item] = max(policy_metrics_subset[item])  # Get upper bound
        # Create a dictionary for renaming columns
        rename_dict = {long: short for long, short in zip(dynamic_outcomes, key_outcomes)}

        # Rename columns of limits_outcomes for display purposes
        limits_outcomes.rename(columns=rename_dict, inplace=True)
        limits = limits_outcomes

        # Create the parallel coordinates
        paraxes = parcoords.ParallelAxes(limits)
        paraxes.ticklabels = key_outcomes
        # Define your color coding for the legend
        labels = list(color_coding.keys())
        color_legend_elements = [Patch(facecolor=color_coding[label], label=label) for label in labels]

        # Create linestyle legend elements. Adjust the labels as per your description.
        linestyle_labels = ["Max", "Min", "Median"]
        linestyle_legend_elements = [Line2D([0], [0], color='black', linewidth=2, linestyle=ls, label=label)
                                     for ls, label in zip(linestyles[:3], linestyle_labels)]

        # Combine color and linestyle legend elements
        legend_elements = color_legend_elements + linestyle_legend_elements
        sta_policies = df_full[(df_full["Policy type"] == "STA")]["policy"].unique().tolist()

        # Loop over rows in dataframe. Plot all policies
        for i, row in policy_metrics_subset.iterrows():
            row_renamed = row.rename(rename_dict)
            policy_type = row["Policy type"]
            data = row_renamed[key_outcomes]

            if policy_type != "STA":
                color = light_color_coding[policy_type]
                paraxes.plot(data, color=color, linewidth=0.5)
            elif policy_type == "STA":
                color = color_coding[policy_type]
                paraxes.plot(data, color=color, linewidth=1)
        for policy_type, policies in highlighted_policy_info.items():
            for policy_id, linestyle in policies:
                row = policy_metrics_subset[policy_metrics_subset['policy'] == str(policy_id)]

                if not row.empty:
                    row_renamed = row.iloc[0].rename(rename_dict)
                    data = row_renamed[key_outcomes]
                    color = color_coding[policy_type]
                    paraxes.plot(data, color=color, linestyle=linestyle, linewidth=3)
        # Add and adjustlegend manually
        plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1),
                   loc="lower right", fontsize=12, frameon=False, ncol=2)

        # for i, ax in enumerate(paraxes.axes):

        #     ax.set_xticklabels([key_outcomes[i]])  # This removes the x-axis tick labels
        #     ax.set_yticklabels([])  #

        plt.suptitle(f"{rm_dict[robustness_metric]}", fontsize=20, x=0, y=1.05, horizontalalignment="left")
        plt.show()

# %% FIGURE 3 Parcoords plot on levers
limits_levers = pd.DataFrame()  # Create a dataframe for lever-based limits
for item in levers:
    limits_levers.loc[0, item] = min(df_reference_subset[item])  # Get lower bound
    limits_levers.loc[1, item] = max(df_reference_subset[item])  # Get upper bound
paraxes = parcoords.ParallelAxes(limits_levers, formatter={"maxima": ".1f", "minima": ".1f"}, fontsize=20, rot=90)
# Plot selected policies with the manually specified colors
for idx, policy_type in enumerate(df_reference_subset['Policy type'].unique()):
    if policy_type != "STA":
        selected_data = df_reference_subset[df_reference_subset['Policy type'] == policy_type]
        paraxes.plot(selected_data, label=f'{policy_type}', color=light_color_coding[policy_type], linewidth=1)
    elif policy_type == "STA":
        selected_data = df_reference_subset[df_reference_subset['Policy type'] == policy_type]
        paraxes.plot(selected_data, label=f'{policy_type}', color=color_coding[policy_type], linewidth=2)
        # Loop over rows in dataframe. Plot all policies

for policy_type, policies in highlighted_policy_info.items():
    for policy_id, linestyle in policies:
        row = policy_metrics_subset[policy_metrics_subset['policy'] == str(policy_id)]

        if not row.empty:
            row_renamed = row.iloc[0].rename(rename_dict)
            data = df_reference_subset[df_reference_subset["policy"] == str(policy_id)]
            color = color_coding[policy_type]
            paraxes.plot(data, color=color, linestyle=linestyle, linewidth=4)
# Get the figure that parcoords is using
parcoords_fig = plt.gcf()  # 'gcf' stands for 'Get Current Figure'
# for ax in paraxes.axes:
#     ax.set_xticklabels([])  # This removes the x-axis tick labels
#     ax.set_yticklabels([])  #

# Set figure size and facecolor
parcoords_fig.set_size_inches(15, 60)
# parcoords_fig.patch.set_facecolor((1, 1, 1, 0))  # Set transparency

# Define your color coding for the legend
labels = list(color_coding.keys())
color_legend_elements = [Patch(facecolor=color_coding[label], label=label) for label in labels]

# Create linestyle legend elements. Adjust the labels as per your description.
linestyle_labels = ["Max", "Min", "Median"]
linestyle_legend_elements = [Line2D([0], [0], color='black', linewidth=2, linestyle=ls, label=label)
                             for ls, label in zip(linestyles[:3], linestyle_labels)]

# Combine color and linestyle legend elements
legend_elements = color_legend_elements + linestyle_legend_elements
plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1),
           loc="lower right", fontsize=18, frameon=False, ncol=2)


# Instead of saving 'fig', now we save 'parcoords_fig' which is the actual figure containing the plot.
parcoords_fig.savefig("./figs/paper/Figure3_parcoords_candidate_policies_reference_levers.png",
                      dpi=300, format="png", bbox_inches="tight", transparent=True)


#%%
# %% FIGURE 3 Parcoords plot on levers

limits_levers = pd.DataFrame()
for item in levers:
    limits_levers.loc[0, item] = min(df_reference_subset[item])
    limits_levers.loc[1, item] = max(df_reference_subset[item])

paraxes = parcoords.ParallelAxes(
    limits_levers,
    formatter={"maxima": ".1f", "minima": ".1f"},
    fontsize=20,
    rot=90
)

# 1) Plot all non‐STA policies in “light” colors
for idx, policy_type in enumerate(df_reference_subset['Policy type'].unique()):
    if policy_type != "STA":
        selected_data = df_reference_subset[df_reference_subset['Policy type'] == policy_type]
        paraxes.plot(
            selected_data,
            label=f'{policy_type}',
            color=light_color_coding[policy_type],
            linewidth=1
        )

# 2) Plot STA policies with a thicker red line
#    (we do this first so that we can annotate them afterward)
selected_sta = df_reference_subset[df_reference_subset['Policy type'] == "STA"]
paraxes.plot(
    selected_sta,
    label='STA',
    color=color_coding["STA"],
    linewidth=2
)

# 3) If you have any “highlighted” policies (e.g. Max/Min/Median), re‐plot them on top:
for policy_type, policies in highlighted_policy_info.items():
    for policy_id, linestyle in policies:
        row = policy_metrics_subset[policy_metrics_subset['policy'] == str(policy_id)]
        if not row.empty:
            data = df_reference_subset[df_reference_subset["policy"] == str(policy_id)]
            paraxes.plot(
                data,
                color=color_coding[policy_type],
                linestyle=linestyle,
                linewidth=4
            )

# ─────────────────────────────────────────────────────────────────────────────
# 4) NOW: Annotate each STA policy just to the LEFT of the first axis.
#    We assume the “first” lever/column is "L1_bio_share_diesel".
first_axis = paraxes.axes[0]
first_lever_name = "L1_bio_share_diesel"

# Get the vertical (y) limits for that first axis:
y_min, y_max = paraxes.limits[first_lever_name]

# For each STA policy row, compute its y‐position (normalized) and put a text label
for idx, row in selected_sta.iterrows():
    #  a) Extract the raw value of the first lever for this policy
    y_val = row[first_lever_name]
    #  b) Convert to a “relative” 0–1 coordinate along that axis
    y_rel = (y_val - y_min) / (y_max - y_min)
    #  c) Choose a slight negative x‐offset so the text sits just to the left.
    #     In data coordinates for the first axis, the x‐value is effectively “0” (all lines enter at x=0).
    #     We pick e.g. –0.02 so the text is a hair to the left.
    x_rel = +.95

    #  d) Place the text. We use transform=first_axis.transData so that x_rel/y_rel
    #     are interpreted in the same “data” coordinates that the axis uses
    paraxes.fig.text(
        x_rel,
        y_rel,
        str(row['policy']),                    # or whichever column has your STA policy ID/name
        transform=first_axis.transData,
        fontsize=10,
        color=color_coding["STA"],
        ha='right',                            # align right so the text “hangs” to the left
        va='center',
        bbox=dict(facecolor='white',           # optional: make text background transparent
                  alpha=0,
                  edgecolor='none',
                  boxstyle='round,pad=0.2')
    )
# ─────────────────────────────────────────────────────────────────────────────

# 5) Build the legend (colors + linestyles)
labels = list(color_coding.keys())
color_legend_elements = [
    Patch(facecolor=color_coding[label], label=label) for label in labels
]

linestyle_labels = ["Max", "Min", "Median"]
linestyle_legend_elements = [
    Line2D([0], [0],
           color='black',
           linewidth=2,
           linestyle=ls,
           label=label)
    for ls, label in zip(linestyles[:3], linestyle_labels)
]

legend_elements = color_legend_elements + linestyle_legend_elements
plt.legend(
    handles=legend_elements,
    bbox_to_anchor=(1, 1),
    loc="lower right",
    fontsize=18,
    frameon=False,
    ncol=2
)

# 6) Final figure adjustments & save
parcoords_fig = plt.gcf()
parcoords_fig.set_size_inches(15, 60)
parcoords_fig.savefig(
    "./figs/paper/Figure3v2_parcoords_candidate_policies_reference_levers.png",
    dpi=300,
    format="png",
    bbox_inches="tight",
    transparent=True
)

# %% FIGURE 4 Visualization of reference scenario performance

# Setup DataFrame limits for plotting
limits_outcomes = pd.DataFrame()  # Create a dataframe for outcome-based limits
for item in objective_outcomes:
    limits_outcomes.loc[0, item] = min(df_reference_subset[item])  # Get lower bound
    limits_outcomes.loc[1, item] = max(df_reference_subset[item])  # Get upper bound

# Initialize the parallel coordinates plot
paraxes = parcoords.ParallelAxes(limits_outcomes, formatter={"maxima": ".1f", "minima": ".1f"}, fontsize=20, rot=90)

# Plot policies with different styles for each policy type
for policy_type in df_reference_subset['Policy type'].unique():
    selected_data = df_reference_subset[df_reference_subset['Policy type'] == policy_type]
    if policy_type == "STA":
        # Highlight "STA" type policies with a thicker line
        lines = paraxes.plot(selected_data, label=f'{policy_type}', color=color_coding[policy_type], linewidth=2)
        for i, (line, policy_name) in enumerate(zip(selected_data[objective_outcomes].values, selected_data['policy'])):
            # Normalize the y-position within the last axis limits
            last_axis_limits = paraxes.limits[objective_outcomes[-1]]
            y_value = line[-1]  # This is the data value at the last axis
            y_relative = (y_value - last_axis_limits[0]) / (last_axis_limits[1] - last_axis_limits[0])
            x_relative = len(objective_outcomes) + 0.02  # Position beyond the last axis
            # Annotate with the policy name
            paraxes.fig.text(x_relative, y_relative, str(policy_name), transform=paraxes.axes[-1].transData,
                             fontsize=18, color=color_coding["STA"], ha='left', va='center',
                             bbox=dict(facecolor='white', alpha=0, edgecolor='none', boxstyle='round,pad=0.2'))
    else:
        paraxes.plot(selected_data, label=f'{policy_type}', color=light_color_coding[policy_type], linewidth=1)


# Highlight specific policies as done in Figure 3
for policy_type, policies in highlighted_policy_info.items():
    for policy_id, linestyle in policies:
        data = df_reference_subset[df_reference_subset["policy"] == str(policy_id)]
        if not data.empty:
            paraxes.plot(data, color=color_coding[policy_type], linestyle=linestyle, linewidth=4)

# Configure figure properties
parcoords_fig = plt.gcf()  # Get the current figure
parcoords_fig.set_size_inches(16, 20)  # Set figure size


# Create legends for color and linestyle
labels = list(color_coding.keys())
color_legend_elements = [Patch(facecolor=color_coding[label], label=label) for label in labels]
linestyle_labels = ["Max", "Min", "Median"]
linestyle_legend_elements = [Line2D([0], [0], color='black', linewidth=2, linestyle=ls, label=label)
                             for ls, label in zip(["-", "--", ":"], linestyle_labels)]
legend_elements = color_legend_elements + linestyle_legend_elements
plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc="lower right", fontsize=18, frameon=False, ncol=2)

# Save the figure
parcoords_fig.savefig("./figs/paper/Figure4_parcoords_candidate_policies_reference_outcomes.png",
                      dpi=300, format="png", bbox_inches="tight", transparent=True)
# %% STA policies highlighted
# Save the original default size
original_figsize = plt.rcParams["figure.figsize"]

# Set the figure size
plt.rcParams["figure.figsize"] = [7, 6]
for robustness_metric in robustness_metrics:
    dynamic_outcomes = [f"{robustness_metric} {outcome}" for outcome in key_outcomes]  # Dynamic outcomes

    # for uncertainty_set in policy_metrics_df["Uncertainty set"].unique():
    for uncertainty_set in ["XP"]:
        policy_metrics_subset = policy_metrics_df[policy_metrics_df["Uncertainty set"] == uncertainty_set]

        limits_outcomes = pd.DataFrame()  # Create a dataframe for limits
        for i, item in enumerate(dynamic_outcomes):
            limits_outcomes.loc[0, item] = min(policy_metrics_subset[item])  # Get lower bound
            limits_outcomes.loc[1, item] = max(policy_metrics_subset[item])  # Get upper bound
        # Create a dictionary for renaming columns
        rename_dict = {long: short for long, short in zip(dynamic_outcomes, key_outcomes)}

        # Rename columns of limits_outcomes for display purposes
        limits_outcomes.rename(columns=rename_dict, inplace=True)
        limits = limits_outcomes

        # Create the parallel coordinates
        paraxes = parcoords.ParallelAxes(limits)
        paraxes.ticklabels = key_outcomes
        # Define your color coding for the legend
        labels = list(color_coding.keys())
        color_legend_elements = [Patch(facecolor=color_coding[label], label=label) for label in labels]

        # Create linestyle legend elements. Adjust the labels as per your description.
        linestyle_labels = ["Max", "Min", "Median"]
        linestyle_legend_elements = [Line2D([0], [0], color='black', linewidth=2, linestyle=ls, label=label)
                                     for ls, label in zip(linestyles[:3], linestyle_labels)]

        # Combine color and linestyle legend elements
        legend_elements = color_legend_elements + linestyle_legend_elements
        sta_policies = df_full[(df_full["Policy type"] == "STA")]["policy"].unique().tolist()

        # Loop over rows in dataframe. Plot all policies
        for i, row in policy_metrics_subset.iterrows():
            row_renamed = row.rename(rename_dict)
            policy_type = row["Policy type"]
            data = row_renamed[key_outcomes]
            color = ultra_light_color_coding[policy_type]
            paraxes.plot(data, color=color)
        # Initialize a variable to control the alternation of label placement
        alternate = False

        # Then plot the STA policies. This is to ensure STA policies are plotted on top
        for i, row in policy_metrics_subset.iterrows():
            policy_type = row["Policy type"]
            if policy_type == 'STA':
                row_renamed = row.rename(rename_dict)
                data = row_renamed[key_outcomes]
                color = color_coding[policy_type]
                paraxes.plot(data, color=color)

                # Get the axis limits for the last outcome
                last_axis_limits = paraxes.limits.iloc[:, -1]

                # Normalize the y-position within the last axis limits
                y_value = data[rename_dict.get(dynamic_outcomes[-1])]  # This is the data value at the last axis
                y_relative = (y_value - last_axis_limits[0]) / (last_axis_limits[1] - last_axis_limits[0])

                # Alternate the x-position to avoid clashing
                x_offset = 0.05 if alternate else 0.  # Alternate the offset value
                x_relative = len(dynamic_outcomes) + x_offset

                # Now we use the axes' transform to place the text correctly
                last_axis_transform = paraxes.axes[-1].transData

                text = policy_metrics_subset.loc[i, "policy"]  # Use the policy name from the current row
                fontsize = 16
                paraxes.fig.text(x_relative, y_relative, text, transform=last_axis_transform,
                                 fontsize=fontsize, color=color, ha='left' if alternate else 'right', va='center',
                                 bbox=dict(facecolor='white', alpha=0, edgecolor='none', boxstyle='round,pad=0.2'))

                # Flip the alternation flag
                alternate = not alternate
        # Add and adjustlegend manually
        legend_elements = color_legend_elements
        plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1),
                   loc="lower right", fontsize=12, frameon=False, ncol=1)

        # for i, ax in enumerate(paraxes.axes):

        #     ax.set_xticklabels([key_outcomes[i]])  # This removes the x-axis tick labels
        #     ax.set_yticklabels([])  #

        plt.suptitle(f"{rm_dict[robustness_metric]}", fontsize=20, x=0, y=1.05, horizontalalignment="left")
        plt.show()


plt.rcParams["figure.figsize"] = original_figsize

# %% Parcoords of all metrics with brushing

# Save the original default size
original_figsize = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = [10, 10]

outcomes = [
    'M1_CO2_TTW_total',
    'M2_driving_cost_car',
    'M3_driving_cost_truck',
    'M4_energy_use_bio',
    'M5_energy_use_electricity'
]
metrics = ["90_percentile_deviation", "Max_regret", "Mean_stdev"]

# 1. Normalize the metrics for each outcome
for metric in metrics:
    policy_metrics_df[f"sum_normalized_{metric}"] = 0  # initialize at 0
    for outcome in outcomes:
        col_name = f"{metric} {outcome}"
        policy_metrics_df[f"normalized_{col_name}"] = (policy_metrics_df[col_name] - policy_metrics_df[col_name].min()) / \
            (policy_metrics_df[col_name].max() - policy_metrics_df[col_name].min())
        policy_metrics_df[f"sum_normalized_{metric}"] = policy_metrics_df[f"sum_normalized_{metric}"] + \
            policy_metrics_df[f"normalized_{metric} {outcome}"]

# Main loop to plot each metric-uncertainty set combination
for metric in metrics:
    dynamic_outcomes = [f"{metric} {outcome}" for outcome in outcomes]

    # for uncertainty_set in policy_metrics_df["Uncertainty set"].unique():
    for uncertainty_set in ["XP"]:
        subset_uncertainty = policy_metrics_df[policy_metrics_df["Uncertainty set"] == uncertainty_set].copy()

        # Normalize data for the robustness metrics
        normalized_data = subset_uncertainty[dynamic_outcomes].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        subset_uncertainty['sum_metric'] = normalized_data.sum(axis=1)

        # Determine top 5 policies for this metric and uncertainty set, for each policy type, excluding "STA"
        top_policies = []
        for ptype in color_coding.keys():
            if ptype != "STA":
                ptype_policies = subset_uncertainty[subset_uncertainty["Policy type"]
                                                    == ptype].nsmallest(5, 'sum_metric').index.tolist()
                top_policies.extend(ptype_policies)
            else:
                ptype_policies = subset_uncertainty[subset_uncertainty["Policy type"] == ptype].index.tolist()
                top_policies.extend(ptype_policies)

        # Plotting
        color_list = [color_coding[ptype] if i in top_policies else light_color_coding[ptype]
                      for i, ptype in subset_uncertainty["Policy type"].items()]
        alpha_list = [0.9 if i in top_policies else 0.1 for i, ptype in subset_uncertainty["Policy type"].items()]
        linewidth_list = [2 if i in top_policies else 1 for i in subset_uncertainty.index]
        zorder_list = [10 if i in top_policies else 1 for i in subset_uncertainty.index]

        limits_outcomes = pd.DataFrame(
            {col: [subset_uncertainty[col].min(), subset_uncertainty[col].max()] for col in dynamic_outcomes})
        paraxes = parcoords.ParallelAxes(limits_outcomes)
        labels = list(color_coding.keys())
        legend_elements = [Patch(facecolor=color_coding[label], label=label) for label in labels]

        # Plotting logic remains unchanged
        for index, (i, row) in enumerate(subset_uncertainty.iterrows()):
            data = pd.DataFrame([row[dynamic_outcomes]])
            color = color_list[index]
            alpha = alpha_list[index]
            linewidth = linewidth_list[index]
            zorder = zorder_list[index]
            paraxes.plot(data.iloc[0], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
            # Annotate only the lines with 'STA' policy type

        plt.legend(handles=legend_elements, loc='upper right')
        #plt.title(f"Metric: {metric}, Uncertainty set: {uncertainty_set}")
        plt.title(f"Metric: {metric}")
        plt.show()

plt.rcParams["figure.figsize"] = original_figsize


# %% Analyze whether we can find a policy that is more robust in all outcomes for all STA policies

# Assuming policy_metrics_df is a pre-defined DataFrame and metrics is a list of metrics
policy_metrics_df_sta = policy_metrics_df[policy_metrics_df["Policy type"] == "STA"]

# Initialize columns for each metric to indicate if a policy is better than all STA policies
for metric in metrics:
    policy_metrics_df[f"{metric}_Is_Better"] = False

for policy in policy_metrics_df["policy"].unique():
    policy_data = policy_metrics_df[policy_metrics_df["policy"] == policy]

    for metric in metrics:
        is_metric_better_for_all_sta = True

        dynamic_outcomes = [f"{metric} {outcome}" for outcome in key_outcomes]

        for sta_policy in policy_metrics_df_sta["policy"].unique():
            sta_policy_data = policy_metrics_df_sta[policy_metrics_df_sta["policy"] == sta_policy]

            is_better_for_this_sta_policy = True
            for outcome in dynamic_outcomes:
                policy_outcome_value = policy_data[outcome].values[0] if not policy_data[outcome].empty else None
                sta_outcome_value = sta_policy_data[outcome].values[0] if not sta_policy_data[outcome].empty else None

                if policy_outcome_value is None or sta_outcome_value is None or not (policy_outcome_value < sta_outcome_value):
                    is_better_for_this_sta_policy = False
                    break

            if not is_better_for_this_sta_policy:
                is_metric_better_for_all_sta = False
                break

        # Update the DataFrame
        policy_metrics_df.loc[policy_metrics_df["policy"] == policy,
                              f"{metric}_Is_Better"] = is_metric_better_for_all_sta

policy_metrics_df
# %% Prepare data for in-dpeth analysis of STA policies
columns_90th_percentile = [
    col for col in policy_metrics_sta.columns if '90_percentile_deviation' in col and 'normalized' not in col]

columns_90th_percentile.append("policy")
policy_metrics_sta_90 = policy_metrics_sta[columns_90th_percentile]
# Set "policy" column as the index for policy_metrics_sta_90
policy_metrics_sta_90.set_index("policy", inplace=True)
# Extract the "Mean/stdev M...." columns
columns_mean_stdev = [col for col in policy_metrics_sta.columns if (
    ('Mean_stdev' in col) & ("normalized" not in col) & ("sum" not in col))]+["policy"]

policy_metrics_mean_stdev = policy_metrics_sta[columns_mean_stdev]
policy_metrics_mean_stdev.set_index("policy", inplace=True)

df_full_sta = df_full[df_full["scenario"] == "Reference"]
df_full_sta = df_full_sta[df_full_sta["Policy type"] == "STA"][key_outcomes+["policy"]]
# Set "policy" column as the index for df_full_sta
df_full_sta.set_index("policy", inplace=True)


df_sta = pd.concat([df_full_sta, policy_metrics_sta_90], axis=1)
df_sta = pd.concat([df_sta, policy_metrics_mean_stdev], axis=1)
df_sta["policy"] = df_sta.index

x_vars = key_outcomes
y_vars = ['90_percentile_deviation M1_CO2_TTW_total',
          '90_percentile_deviation M2_driving_cost_car',
          '90_percentile_deviation M3_driving_cost_truck',
          '90_percentile_deviation M4_energy_use_bio',
          '90_percentile_deviation M5_energy_use_electricity']

y_vars += columns_mean_stdev
# Create a FacetGrid with scatterplots for each metric

# Your existing code for creating df_long
df_long = pd.melt(df_sta, id_vars=["policy"],
                  value_vars=['M1_CO2_TTW_total', 'M2_driving_cost_car', 'M3_driving_cost_truck',
                              'M4_energy_use_bio', 'M5_energy_use_electricity'],
                  var_name='Metric', value_name='Outcome in reference scenario')

# Add the 90th percentile deviation values
# Adjust this based on how your data is structured
df_long['90th Percentile Deviation'] = df_long.apply(
    lambda row: df_sta.at[row['policy'], '90_percentile_deviation ' + row['Metric']], axis=1
)

df_long['Mean_stdev'] = df_long.apply(
    lambda row: df_sta.at[row['policy'], 'Mean_stdev ' + row['Metric']], axis=1
)

# %% FIGURE9 Ranking of reference scenario and 90th percentile deviation

sns.set_style("white")


def rank_values(series):
    return series.rank(method='dense')


# Apply the ranking within each group for each column'
df_long['Ranking outcome in reference scenario'] = df_long.groupby(
    'Metric')['Outcome in reference scenario'].transform(rank_values)
df_long['Ranking 90th percentile deviation'] = df_long.groupby(
    'Metric')['90th Percentile Deviation'].transform(rank_values)

df_long['Ranking mean_stdev'] = df_long.groupby(
    'Metric')['Mean_stdev'].transform(rank_values)

# Sample over different values of w_ref_rank
weights = [1, 0.5, 0]
df_samples = pd.DataFrame()

for w_ref in weights:
    w_rob = 1 - w_ref
    temp_df = df_long.copy()
    temp_df["Weighted ranking"] = w_ref * temp_df['Ranking outcome in reference scenario'] + \
        w_rob * temp_df['Ranking 90th percentile deviation']
    temp_df['w ref'] = w_ref

    df_samples = pd.concat([df_samples, temp_df])

df_samples["Ranking weighted"] = df_samples.groupby([
    'Metric', "w ref"])["Weighted ranking"].transform(rank_values)

# Plot using FacetGrid
g = sns.FacetGrid(df_samples, col='w ref', height=4, aspect=1, sharex=True, sharey=False, col_order=weights)


def plot_lines(data, **kwargs):
    palette = kwargs.pop("palette")
    ax = plt.gca()  # Get the current axis from the context
    for name, group in data.groupby("policy"):
        z = 3 if name == "B" else 1
        sns.lineplot(x='Metric', y='Ranking weighted', data=group, marker='o',
                     ax=ax, color=palette[name], zorder=z, label=name)


# Map the custom function to the FacetGrid
g.map_dataframe(plot_lines, palette=STA_colors)

# Invert y-axis
for ax in g.axes.flat:
    ax.invert_yaxis()
    ax.set_yticks(range(1, 9))

g.fig.subplots_adjust(hspace=0.9, wspace=0.3)
# Enhancing the plot
g.set_xticklabels(rotation=90)
plt.subplots_adjust(top=0.85)
#g.fig.suptitle('Weighted Ranking of Policies for Different w_ref Values 90th percentile dev', size=16)
g.add_legend(title="STA policy", bbox_to_anchor=(0.5, 1.02), loc='center right', borderaxespad=0.1, ncol=8)
g.set_titles(col_template=r"$\mathrm{{W_{{ref}}}}$: {col_name}", row_template="{row_name}")

g.set_axis_labels("", "Ranking")
# Save the figure
g.savefig("./figs/paper/Figure9_policy_rank_weights.png",
          dpi=300, format="png", bbox_inches="tight", transparent=True)

# %% Visualize robustness and reference scenario performance 90th percentile
metric_name_dict = {'90th Percentile Deviation': r'$\mathrm{RM_{90\%\,dev}}$',
                    'Mean_stdev': r'$\mathrm{RM_{MeanStdev}}$'}


def reference_robustness_scatter_plot(df, y_variable, plot_title):
    # Create the FacetGrid
    g = sns.FacetGrid(df, col='Metric', col_wrap=4, sharex=False, sharey=False)
    g = g.map(plt.scatter, 'Outcome in reference scenario', y_variable, color='red')
    g.fig.subplots_adjust(hspace=0.5, wspace=0.6)

    # Annotate each point with its policy
    for ax, (metric, subset) in zip(g.axes.flatten(), df.groupby('Metric')):
        subset.apply(lambda row: ax.text(row['Outcome in reference scenario'],
                                         row[y_variable],
                                         str(row['policy']),
                                         fontsize=12,
                                         ha='left', va="top"), axis=1)

    # Set the subplot titles and x-axis labels for all subplots
    for ax, title in zip(g.axes.flatten(), x_vars):
        ax.set_title(title)
        ax.set_xlabel('Outcome ref. scenario')
        ax.set_ylabel(metric_name_dict[y_variable])
    # Set the subplot titles
    g.set_titles("{col_name}")
    #g.fig.suptitle(plot_title, fontsize=14, y=1.05)


for robustness_metric, plot_title in [
    ('90th Percentile Deviation', r'STA policies: $\mathrm{RM_{90\%\,dev}}$ vs reference scenario performance'),
    ('Mean_stdev', r'STA policies: $\mathrm{RM_{MeanStdev}}$ vs reference scenario performance')
]:
    reference_robustness_scatter_plot(df_long[df_long["Metric"].isin(
        objective_outcomes)], robustness_metric, plot_title)
# %% Visualie relationship between mean and stdev
for metric in key_outcomes:
    plt.figure()
    sns.scatterplot(policy_metrics_sta, x=f"Mean {metric}",
                    y=f"Standard deviation {metric}", hue="policy", palette=STA_colors,
                    size=f"Mean_stdev {metric}", sizes=(100, 500))
    for index, row in policy_metrics_sta.iterrows():
        policy = row["policy"]
        plt.text(row[f"Mean {metric}"], row[f"Standard deviation {metric}"],
                 f"{policy}: {round(row[f'Mean_stdev {metric}'],1)}", color='black', ha='right', va='bottom', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# %% Visualize 90th percentile deviation metric
df_temp = df_full[df_full["Policy type"] == "STA"]

df_temp = df_full[df_full["Policy type"] == "STA"]
df_temp["policy"] = df_temp["policy"].astype(str)

for outcome in key_outcomes:  # Ensure key_outcomes is defined
    plt.figure(figsize=(10, 6))

    plot = sns.displot(df_temp, x=outcome, hue="policy", kind="kde", height=6, aspect=1.5, palette=STA_colors)
    plt.title(f'Visualization of 90th percentile deviation {outcome} for STA policies')
    max_y_value = plot.ax.get_ylim()[1]
    offset_step = max_y_value * 0.05  # Adjust the multiplier as necessary for visibility

    for index, policy in enumerate(df_temp["policy"].unique()):
        policy_data = df_temp[df_temp['policy'] == policy]
        policy_data_ref = policy_data[policy_data["scenario"] == "Reference"]

        # Retrieve the 90th percentile deviation value
        deviation_col_name = f'90_percentile_deviation {outcome}'
        percentlile_90_col_name = f'90th percentile {outcome}'
        percentile_deviation_value = policy_metrics_sta.loc[policy_metrics_sta['policy']
                                                            == policy, deviation_col_name].values[0]
        percentile_90_value = policy_metrics_sta.loc[policy_metrics_sta['policy']
                                                     == policy, percentlile_90_col_name].values[0]
        line_x = percentile_90_value
        plt.axvline(x=line_x, color=STA_colors[policy], linestyle=':', label=f'90th % deviation of {policy}')

        vertical_position = max_y_value - (index + 1) * offset_step
        plt.text(line_x, vertical_position, f'{percentile_deviation_value:.1%}',
                 horizontalalignment='center', color=STA_colors[policy], fontsize=12, backgroundcolor='white')

        ref_value = policy_data_ref[outcome].values[0]
        plt.axvline(x=ref_value, color=STA_colors[policy], linestyle='-',
                    label=f'Reference scenario value of {policy}')

        arrow_properties = {
            # Smaller head width for a less blocky appearance
            'head_width':  abs(plot.ax.get_ylim()[1]-plot.ax.get_ylim()[0])*0.03,
            'head_length': abs(line_x - ref_value) * 0.02,  # Smaller head length proportional to arrow length
            'width': abs(plot.ax.get_ylim()[1]-plot.ax.get_ylim()[0])*0.0025,  # Width of the arrow line
            'length_includes_head': True,  # The head will be part of the specified length
            'shape': 'full',  # Full shape of the arrow
            'overhang': 0,  # How much the arrow is past the end point
            'zorder': 5,  # Ensure the arrow is drawn on top of other elements
        }

        # Draw an arrow from the 90th percentile line to the reference scenario line
        plt.arrow(ref_value, vertical_position*0.98, line_x - ref_value, 0,
                  fc=STA_colors[policy], ec=STA_colors[policy], **arrow_properties)

    plt.legend()

    plt.title(f'{outcome} by STA policy')

    # %% Visualzie mean stdev
df_temp = df_full[df_full["Policy type"] == "STA"]

df_temp["policy"] = df_temp["policy"].astype(str)

for outcome in key_outcomes:  # Ensure key_outcomes is defined
    g = sns.displot(df_temp, x=outcome, hue="policy", kind="kde", height=6, aspect=1.5, palette=STA_colors)
    plt.title(f'Distribution and Mean Values of {outcome} by Policy')
    #plt.legend(title='', fontsize=8)

    # Calculate and plot the mean and 90th percentile deviation values for each policy with corresponding colors
    for index, policy in enumerate(sorted(df_temp["policy"].unique())):
        policy_data = df_temp[df_temp['policy'] == policy]
        policy_data_ref = policy_data[policy_data["scenario"] == "Reference"]
        mean_stdev_value = policy_metrics_sta.loc[policy_metrics_sta['policy']
                                                  == policy, f"Mean_stdev {outcome}"].values[0]

        deviation_col_name = f'90_percentile_deviation {outcome}'
        percentile_deviation_value = policy_metrics_sta.loc[policy_metrics_sta['policy']
                                                            == policy, deviation_col_name].values[0]

        # Adjust vertical position based on the index
        vertical_position = 0.90 - index * 0.05  # Adjust spacing here as needed
        if index == 0:
            plt.text(0.75, 0.95, rm_dict["Mean_stdev"], transform=g.ax.transAxes,
                     horizontalalignment='left', verticalalignment='top', color="black",
                     fontsize=12, backgroundcolor='white')
            plt.text(0.88, 0.95, rm_dict["90_percentile_deviation"], transform=g.ax.transAxes,
                     horizontalalignment='left', verticalalignment='top', color="black",
                     fontsize=12, backgroundcolor='white')
        plt.text(0.70, vertical_position, f'{policy}', transform=g.ax.transAxes,
                 horizontalalignment='left', verticalalignment='top', color=STA_colors[policy],
                 fontsize=12, backgroundcolor='white')
        plt.text(0.75, vertical_position, f'{mean_stdev_value:.2f}', transform=g.ax.transAxes,
                 horizontalalignment='left', verticalalignment='top', color=STA_colors[policy],
                 fontsize=12, backgroundcolor='white')

        plt.text(0.88, vertical_position, f'{percentile_deviation_value:.2f}', transform=g.ax.transAxes,
                 horizontalalignment='left', verticalalignment='top', color=STA_colors[policy],
                 fontsize=12, backgroundcolor='white')

        ref_value = policy_data_ref[outcome].values[0]
        plt.axvline(x=ref_value, color=STA_colors[policy], linestyle='--')

        # plt.text(ref_value, 0.025, f'{policy}',
        #          horizontalalignment='right', verticalalignment='top', color=STA_colors[policy],
        #          fontsize=12, backgroundcolor='white')
    plt.title(f'Distribution of {outcome} for STA policies over scenarios ')
# %% Vulnerability analysis
# Policies selected for analysis
# vulnerability_policies = [291, 3051, "B"]

# # Parcoords of vulnerability policies
# paraxes = parcoords.ParallelAxes(limits_levers, formatter={"maxima": ".1f", "minima": ".1f"}, fontsize=20, rot=90)
# # Plot selected policies with the manually specified colors
# for idx, policy in enumerate(vulnerability_policies):
#     selected_data = df_full[df_full['policy'] == str(policy)].iloc[0]
#     policy_type = selected_data["Policy type"]
#     paraxes.plot(selected_data, label=f'Policy type: {policy_type}', color=color_coding[policy_type])

# # Get the figure that parcoords is using
# parcoords_fig = plt.gcf()  # 'gcf' stands for 'Get Current Figure'
# # for ax in paraxes.axes:
# #     ax.set_xticklabels([])  # This removes the x-axis tick labels
# #     ax.set_yticklabels([])  #
# # Set figure size and facecolor
# parcoords_fig.set_size_inches(10, 10)
# # parcoords_fig.patch.set_facecolor((1, 1, 1, 0))  # Set transparency

# # Optionally, you can add a legend if you need it
# paraxes.legend()

# %% Find most common vulnerabilities

df_policy_vulnerabilities = pd.DataFrame(index=df_full["policy"].unique())
n_vulnerabilities = 2
vulnerabilities_list = []
for policy in df_full["policy"].unique():
    df_policy_vulnerabilities.loc[policy, "policy"] = policy
    df_policy_vulnerabilities.loc[policy, "Policy type"] = df_full[df_full["policy"] == policy]["Policy type"].unique()[
        0]
    df_policy_vulnerabilities.loc[policy, "CO2 target not met count"] = sum(
        df_full[df_full["policy"] == policy]["CO2 target not met"])
    df_v = df_full[df_full["policy"] == policy]
    df_v = df_v[df_v["scenario"] != "Reference"]
    df_v = df_v.reset_index(drop=True)

    x = df_v[model.uncertainties.keys()]
    y = df_v["CO2 target not met"]
    prim_alg = prim.Prim(x, y, threshold=0.5)
    box1 = prim_alg.find_box()

    # Track the found vulnerabilities to avoid duplicates
    found_vulnerabilities = []

    # Determine the maximum number of dimensions (vulnerabilities) found
    max_res_dim = max(box1.peeling_trajectory["res_dim"], default=0)

    for i in range(1, min(n_vulnerabilities, max_res_dim) + 1):
        box_i = box1.peeling_trajectory[box1.peeling_trajectory["res_dim"]
                                        == i].sort_values(by="density", ascending=False).head(1)
        if not box_i.empty:
            # Choose point for inspection
            i1 = int(box_i["id"].iloc[0])
            box_data = box1.inspect(i1, style="data")[0][1]
            box_data.columns = box_data.columns.droplevel(0)

            # Get the vulnerability that hasn't been found yet
            for v in box_data.index:
                if v not in found_vulnerabilities:
                    found_vulnerabilities.append(v)
                    df_policy_vulnerabilities.loc[policy, f"Vulnerability {i}"] = v
                    break
            if len(found_vulnerabilities) < i:
                # If we didn't find a new vulnerability, we break out of the loop
                break
        else:
            # If there are no more vulnerabilities to find, we break out of the loop
            break
df_policy_vulnerabilities["CO2 target not met share"] = df_policy_vulnerabilities["CO2 target not met count"] / \
    len(df_full["scenario"].unique())
# %% countplot of vulnerabilities

sns.countplot(df_policy_vulnerabilities, x="Vulnerability 1", hue="Policy type")
plt.figure()
sns.countplot(df_policy_vulnerabilities, x="Vulnerability 2", hue="Policy type")

# Melt the dataframe
df_policy_vulnerabilities_long = pd.melt(df_policy_vulnerabilities, id_vars=['Policy type'],
                                         value_vars=['Vulnerability 1', 'Vulnerability 2'],
                                         var_name='Vulnerability Type', value_name='Vulnerability')

sns.catplot(

    data=df_policy_vulnerabilities_long, x="Vulnerability", row="Vulnerability Type",
    kind="count", hue="Policy type", height=4, aspect=1.5, sharex=True, sharey=True
)
plt.xticks(rotation=90)  # Rotate labels for better readability

# Across all policy types
# Create a combination count table
combination_counts = df_policy_vulnerabilities.groupby(
    ['Vulnerability 1', 'Vulnerability 2']).size().reset_index(name='counts')

# Pivot the table to create a matrix form suitable for a heatmap
matrix_counts = combination_counts.pivot_table(
    index='Vulnerability 1', columns='Vulnerability 2', values='counts', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(12, 10))
# You can change cmap to another colormap like 'viridis' if you prefer
sns.heatmap(matrix_counts, annot=True, cmap='viridis', fmt='d')
plt.title('Heatmap of Policy Vulnerability Combinations')
plt.xlabel('Vulnerability 2')
plt.ylabel('Vulnerability 1')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Per policy type
# Iterate through each Policy type and create a heatmap

# Step 1: Identify the unique vulnerabilities for 'Vulnerability 1' and 'Vulnerability 2', excluding 'nan'
unique_vuln1 = df_policy_vulnerabilities['Vulnerability 1'].dropna().unique()
unique_vuln2 = df_policy_vulnerabilities['Vulnerability 2'].dropna().unique()

# Step 2: Create standardized pivot tables and plot heatmaps for each Policy type
for policy_type in df_policy_vulnerabilities['Policy type'].unique():
    # Filter the DataFrame for the current Policy type
    df_filtered = df_policy_vulnerabilities[df_policy_vulnerabilities['Policy type'] == policy_type]

    # Create a combination count table for this Policy type, using the unique vulnerabilities
    combination_counts = df_filtered.groupby(['Vulnerability 1', 'Vulnerability 2']).size().reset_index(name='counts')

    # Calculate the total number of occurrences for the current Policy type
    total_counts = combination_counts['counts'].sum()

    # Calculate the share of each combination
    combination_counts['share'] = combination_counts['counts'] / total_counts

    # Create a full combination matrix with all unique vulnerabilities in 'Vulnerability 1' and 'Vulnerability 2'
    combination_matrix = pd.DataFrame(index=unique_vuln1, columns=unique_vuln2).fillna(0)

    # Populate the matrix with the shares
    for _, row in combination_counts.iterrows():
        vuln1 = row['Vulnerability 1']
        vuln2 = row['Vulnerability 2']
        combination_matrix.at[vuln1, vuln2] = row['share']

    # Step 3: Plot the heatmap for this Policy type
    plt.figure(figsize=(5, 4))
    sns.heatmap(combination_matrix, annot=True, cmap='viridis', fmt='.0%', vmax=.45)  # Use percentage format
    plt.title(f'{policy_type}')
    plt.xlabel('Vulnerability 2')
    plt.ylabel('Vulnerability 1')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()
# The number of top combinations you want to extract
n = 5

# Assuming df_policy_vulnerabilities is your DataFrame that contains the raw data
# Calculate the share of each vulnerability combination within each policy type
policy_vuln_combinations = (
    df_policy_vulnerabilities
    .groupby(['Policy type', 'Vulnerability 1', 'Vulnerability 2'])
    .size()
    .reset_index(name='counts')
)

# Calculate the total counts for each policy type to normalize and get the share
policy_totals = policy_vuln_combinations.groupby('Policy type')['counts'].transform('sum')
policy_vuln_combinations['share'] = policy_vuln_combinations['counts'] / policy_totals

# Now create a DataFrame to hold the top n combinations per policy type
top_combinations = pd.DataFrame()

# For each policy type, sort by share and take the top n
for policy in policy_vuln_combinations['Policy type'].unique():
    # Get the sorted combinations for the current policy
    sorted_combinations = (
        policy_vuln_combinations[policy_vuln_combinations['Policy type'] == policy]
        .sort_values(by='share', ascending=False)
    )

    # Get the top n combinations for the current policy
    top_for_policy = sorted_combinations.head(n)

    # Calculate cumulative count and share
    top_for_policy['cumulative_count'] = top_for_policy['counts'].cumsum()
    top_for_policy['cumulative_share'] = top_for_policy['share'].cumsum()

    # Append to the top_combinations DataFrame
    top_combinations = pd.concat([top_combinations, top_for_policy], ignore_index=True)

# Optionally, sort the final DataFrame for better readability
top_combinations = top_combinations.sort_values(by=['Policy type', 'share'], ascending=[True, False])

# Reset the index for the final DataFrame
top_combinations.reset_index(drop=True, inplace=True)

# Show the DataFrame
print(top_combinations)
# %% Vulnerability analysis and comparison comapred to policy B
vulnerability_analysis_policy = "B"  # 99 All levers, #183 no transport efficiency "B" STA

df_v = df_full[df_full["policy"].astype(str) == str(vulnerability_analysis_policy)]
df_v = df_v[df_v["scenario"] != "Reference"]
# Perform PRIM analysis
# Set up PRIM
# Reset the index to ensure alignment
df_v = df_v.reset_index(drop=True)

# Set up PRIM with aligned indices
x = df_v[model.uncertainties.keys()]
y = df_v["CO2 target not met"]
x_rotated, rotation_matrix = prim.pca_preprocess(x, y)
prim_alg = prim.Prim(x, y, threshold=0.9)

# Find 1st box
box1 = prim_alg.find_box()
#
# Visualizations of Box1
box1.show_tradeoff()

# for i in range(0, len(box1.peeling_trajectory.T.columns)):
#     s = int(box1.peeling_trajectory.T[i].id)
#     if (i % 2) == 0:
#         plt.text(box1.peeling_trajectory.T[i].coverage, box1.peeling_trajectory.T[i].density + .0, s, fontsize=8)
#     else:
#         plt.text(box1.peeling_trajectory.T[i].coverage, box1.peeling_trajectory.T[i].density + .0, s, fontsize=8)
plt.show()
# Inspect box with restrictions on 2 dimensions and max density (i.e. "last" box with res_dim=2)
box_i = box1.peeling_trajectory[box1.peeling_trajectory["res_dim"]
                                == 2].sort_values(by="density", ascending=False).head(1)

# Choose point for inspection
i1 = int(box_i["id"].iloc[0])
# or choose box manually
#i1 = 15
box1.inspect(i1)
box_data = box1.inspect(i1, style="data")[0][1]
# box data is now a df with each row data for each of the restricted dimensions
box_data.columns = box_data.columns.droplevel(0)

box1.inspect(i1, style='graph')
plt.show()
box1.show_ppt()
# Save the original palette
original_palette = sns.color_palette()
custom_palette = ['green', 'red']
sns.set_palette(sns.color_palette(custom_palette))
#%%
ax = box1.show_pairs_scatter(i1,lower="scatter")
ax.legend.set_title("")
# for row in ax.axes:
#     for subplot in row:
#         subplot.set_facecolor('none')  # For the subplot background
#         subplot.grid(False)  # To remove the grid lines
#         for spine in subplot.spines.values():
#             spine.set_edgecolor('none')  # For the edges of the subplot
#plt.savefig('test_prim_box.png', transparent=True)

# Calculate statistics for boxes
# Initialize the filtered DataFrame to all True
mask = pd.Series(True, index=df_full.index)

# Apply the filtering criteria for each row in box_data
for index, row in box_data.iterrows():
    min_val = row['min']
    max_val = row['max']
    # Update the mask to check if each entry is within the 'min' and 'max' range
    mask = mask & (df_full[index] >= min_val) & (df_full[index] <= max_val)

# Apply the mask to df_full to filter the rows
df_v_box = df_full[mask]
# %% Visualize loadings in case PCA PRIM was used
# Get loadings
loadings = {
    "Variable": rotation_matrix.index,
    "r_19": rotation_matrix["r_19"],
    "r_20": rotation_matrix["r_20"]
}
df_loadings = pd.DataFrame(loadings)

# Plot for r_19
plt.figure(figsize=(10, 8))
plt.barh(df_loadings["Variable"], df_loadings["r_19"], color='skyblue')
plt.xlabel('Loadings')
plt.ylabel('Variables')
plt.title('PCA Loadings for Component r_19')
plt.tight_layout()
plt.show()

# Plot for r_20
plt.figure(figsize=(10, 8))
plt.barh(df_loadings["Variable"], df_loadings["r_20"], color='lightgreen')
plt.xlabel('Loadings')
plt.ylabel('Variables')
plt.title('PCA Loadings for Component r_20')
plt.tight_layout()
plt.show()

# Preparing the data
variables = rotation_matrix.index  # The original variables
r_19_loadings = rotation_matrix['r_19']
r_20_loadings = rotation_matrix['r_20']

# Setting up the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8), sharey=True)
fig.suptitle('Comparison of PCA Loadings for Components r_19 and r_20')

# Plot for r_19
axes[0].barh(variables, r_19_loadings, color='skyblue')
axes[0].set_title('Component r_19')
axes[0].set_xlabel('Loadings')
axes[0].set_ylabel('Variables')

# Plot for r_20
axes[1].barh(variables, r_20_loadings, color='lightgreen')
axes[1].set_title('Component r_20')
axes[1].set_xlabel('Loadings')
# No need to set y-label for the second plot as it shares the axis with the first

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the global title
plt.show()
# %% Calculate  coverage and density for all policies for the scenario box

for index, row in df_policy_vulnerabilities.iterrows():

    subset_box = df_v_box[df_v_box["policy"] == index]
    subset_full = df_full[df_full["policy"] == index]
    df_policy_vulnerabilities.loc[index, "Coverage"] = sum(
        subset_box["CO2 target not met"])/sum(subset_full["CO2 target not met"])
    df_policy_vulnerabilities.loc[index, "Density"] = sum(
        subset_box["CO2 target not met"])/len(subset_box["CO2 target not met"])


# %% Scatterplot of coverage and density
# Plotting the scatterplot with the specified color coding
plt.figure(figsize=(8, 6))
sns.set_style("white")
df_policy_vulnerabilities["RMco$_{2}$sat"] = 1 - \
    df_policy_vulnerabilities["CO2 target not met share"]
sns.scatterplot(
    data=df_policy_vulnerabilities,
    x="Coverage",
    y="Density",
    hue="Policy type",
    palette=color_coding,
    size="RMco$_{2}$sat",
)
sns.despine()
# Annotating and emphasizing the policy named "B"
for sta_policy in df_full[df_full["Policy type"] == "STA"]["policy"].unique():
    df_sta_policy = df_policy_vulnerabilities[df_policy_vulnerabilities['policy'] == sta_policy]
    # if sta_policy == "B":
    #     plt.scatter(
    #         df_sta_policy["Coverage"],
    #         df_sta_policy["Density"],
    #         color='red',  # Or any color that makes it stand out
    #         s=100,          # Size of the marker
    #         #label='Policy B',
    #         edgecolor='black',
    #         linewidth=2,
    #         zorder=5  # Make sure the point is on top
    #     )

    plt.annotate(
        sta_policy,
        (df_sta_policy["Coverage"].values[0], df_sta_policy["Density"].values[0]),
        textcoords="offset points",
        xytext=(0, -13),
        ha='center', color=color_coding["STA"]
    )

# Show the legend
#plt.title('Density vs Coverage by Policy Type')
plt.legend(frameon=False, loc="upper left", title="")
plt.show()
# %% Feature scorings


def alphanumeric_sorter(key):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(k): return [convert(c) for c in re.split('([0-9]+)', k)]
    return alphanum_key(key)


# Prepare your data
x_raw = experiments.drop(columns=["Policy type", "policy", "model"])
sorted_columns = sorted(x_raw.columns, key=alphanumeric_sorter)
x_sorted = x_raw[sorted_columns]

# Define outcomes and calculate feature scores
keys_to_include = key_outcomes
y = {key: value for key, value in outcomes_xp.items() if key in keys_to_include}
fs = feature_scoring.get_feature_scores_all(x_sorted, y)

# Sort both the index and columns of the fs DataFrame using the alphanumeric sorter
# fs_sorted = fs.sort_index(key=alphanumeric_sorter)
# sorted_fs_columns = sorted(fs_sorted.columns, key=alphanumeric_sorter)
# fs_sorted = fs_sorted[sorted_fs_columns]
# print(fs_sorted.index)  # Should show the sorted feature names
# print(fs_sorted.columns)  # Should show the sorted outcome names
# Now create the heatmap
sns.heatmap(fs, cmap="viridis", annot=True, fmt=".0%")


# all X and L on all outcomes
x = experiments.drop(columns=["Policy type", "policy", "model"])
keys_to_remove = ["Delta CS light vehicles", "Delta CS trucks", "Delta CS total", "Delta tax increase total"]
y = {key: value for key, value in outcomes_xp.items() if key not in keys_to_remove}

fs = feature_scoring.get_feature_scores_all(x, y)
sns.heatmap(fs, cmap="viridis", annot=True)
plt.show()

# all X and L on all outcomes
x = experiments.drop(columns=["Policy type", "policy", "model"])
x = df_full_all[uncertainties]
y = {'CO2 target not met': np.array(df_full_all["CO2 target not met"].astype(int))}


fs = feature_scoring.get_feature_scores_all(x, y)
sns.heatmap(fs, cmap="viridis", annot=True)
plt.show()


# %% plot for poster

# Define your color coding for the legend
color_coding = {
    "All levers": '#610100',
    "No transport efficiency": '#E9C2C0',

}

# Create a dictionary for the light colors
light_color_coding = {
    "All levers": 'lightgrey',
    "No transport efficiency": 'lightgrey',
}

# Save the original default size
original_figsize = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = [16, 12]

outcomes = [
    'M1_CO2_TTW_total',
    'M2_driving_cost_car',
    'M3_driving_cost_truck',
    'M4_energy_use_bio',
    'M5_energy_use_electricity'
]
metrics = ["Mean/stdev"]
policy_metrics_df2 = policy_metrics_df[(policy_metrics_df["Policy type"] == "All levers") | (
    policy_metrics_df["Policy type"] == "No transport efficiency")]
# 1. Normalize the metrics for each outcome
for metric in metrics:
    policy_metrics_df[f"sum_normalized_{metric}"] = 0  # initialize at 0
    for outcome in outcomes:
        col_name = f"{metric} {outcome}"
        policy_metrics_df2[f"normalized_{col_name}"] = (policy_metrics_df2[col_name] - policy_metrics_df2[col_name].min()) / \
            (policy_metrics_df2[col_name].max() - policy_metrics_df2[col_name].min())
        policy_metrics_df2[f"sum_normalized_{metric}"] = policy_metrics_df2[f"sum_normalized_{metric}"] + \
            policy_metrics_df2[f"normalized_{metric} {outcome}"]

# Main loop to plot each metric-uncertainty set combination
for metric in metrics:
    dynamic_outcomes = [f"{metric} {outcome}" for outcome in outcomes]

    for uncertainty_set in policy_metrics_df2["Uncertainty set"].unique():
        subset_uncertainty = policy_metrics_df2[policy_metrics_df2["Uncertainty set"] == uncertainty_set].copy()

        # Normalize data for the robustness metrics
        normalized_data = subset_uncertainty[dynamic_outcomes].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        subset_uncertainty['sum_metric'] = normalized_data.sum(axis=1)

        # Determine top 5 policies for this metric and uncertainty set, for each policy type including "STA"
        top_policies = []
        for ptype in color_coding.keys():
            ptype_policies = subset_uncertainty[subset_uncertainty["Policy type"]
                                                == ptype].nsmallest(10, 'sum_metric').index.tolist()
            top_policies.extend(ptype_policies)

        # Plotting
        color_list = [color_coding[ptype] if i in top_policies else light_color_coding[ptype]
                      for i, ptype in subset_uncertainty["Policy type"].items()]
        alpha_list = [0.8 if i in top_policies else 0.2 for i, ptype in subset_uncertainty["Policy type"].items()]
        linewidth_list = [5 if i in top_policies else 2 for i in subset_uncertainty.index]

        limits_outcomes = pd.DataFrame(
            {col: [subset_uncertainty[col].min(), subset_uncertainty[col].max()] for col in dynamic_outcomes})
        paraxes = parcoords.ParallelAxes(limits_outcomes, formatter={
                                         "maxima": ".1f", "minima": ".1f"}, fontsize=20, rot=45)
        labels = list(color_coding.keys())
        # legend_elements = [Patch(facecolor=color_coding[label], label=label) for label in labels]

        # Plotting logic remains unchanged
        for index, (i, row) in enumerate(subset_uncertainty.iterrows()):
            data = pd.DataFrame([row[dynamic_outcomes]])
            color = color_list[index]
            alpha = alpha_list[index]
            linewidth = linewidth_list[index]
            paraxes.plot(data.iloc[0], color=color, alpha=alpha, linewidth=linewidth)

        # plt.legend(handles=legend_elements, loc='upper right')
        # plt.title(f"Metric: {metric}, Uncertainty set: {uncertainty_set}")

    # Get the figure that parcoords is using
    parcoords_fig = plt.gcf()  # 'gcf' stands for 'Get Current Figure'
    # for ax in paraxes.axes:
    #     ax.set_xticklabels([])  # This removes the x-axis tick labels
    #     ax.set_yticklabels([])  #
    # Set figure size and facecolor
    parcoords_fig.set_size_inches(15, 10)
    parcoords_fig.patch.set_facecolor((1, 1, 1, 0))  # Set transparency

    # Optionally, you can add a legend if you need it
    # paraxes.legend()

    # Instead of saving 'fig', now we save 'parcoords_fig' which is the actual figure containing the plot.
    parcoords_fig.savefig("./figs/robustness metrics.png", dpi=500,
                          format="png", bbox_inches="tight", transparent=True)
    plt.show()
plt.rcParams["figure.figsize"] = original_figsize
# %%
df_temp = df_full[df_full["policy"].isin(["B", "C1", "C2", "C3", "C4", "D1", "D2", "D3"])]
df_temp["policy"] = df_temp["policy"].astype(str)
for outcome in objective_outcomes:
    sns.displot(df_temp, x=outcome, hue="policy", kind="kde")
