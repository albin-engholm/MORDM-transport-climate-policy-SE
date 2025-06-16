# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:05:08 2022

@author: aengholm
This is a script for setting up and running robust search over policy levers for a static 
excel model with parametric uncertainties. It is designed for the TRV scenario model.
Results are saved and can be loaded for analysis in separate script - 
e.g. scenario_exploration_excel_static.py'
"""

from ema_workbench import (RealParameter, 
                           ScalarOutcome, ema_logging,
                           perform_experiments, Constant, Constraint,
                           Scenario,MultiprocessingEvaluator)
from ema_workbench.em_framework import samplers, sample_uncertainties
import pickle
from datetime import date
        
from ema_workbench.em_framework.optimization import (EpsilonProgress, ArchiveLogger)
#from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
from datetime import date
import time
from platypus import SBX, PM, GAOperator
from ema_workbench.connectors.excel import ExcelModel
import os
import pandas as pd
import numpy as np
import shutil

#%% define functions for robustness calcls

def mean_stdev(x):
    if x is None or np.any(pd.isnull(x)):
        return np.inf  # or a large penalty
    return np.mean(x) * np.std(x)

def satisficing_metric(x):
    if x is None or np.any(pd.isnull(x)):
        return 0
    return np.mean(np.array(x) < 1.89)
#%%Specify model 
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)
    # Set model
    model = ExcelModel("scenarioModel", wd="../models",
                       model_file='Master - Cleanup.xlsx')
    model.default_sheet = "EMA"

#%% specify parameters
    n_processes = -4  # set # of parallel threads
    nfe = 1000
    date = date.today()  # Date to use for storing files
    # What set of policies should the MOEA be run for?
    #policy_types = ["All levers", "No transport efficiency"]
    #policy_types = ["No transport efficiency"]
    policy_types = ["All levers"]

    # Optimization parameters
    # Set epsilons
    # Epsilons for M2-M5, 5% of range in pilot run M2 range ~14, M3 range 82, m4 range 11, m5 range 3
    #Refernce scenario outcome values for M2-M4: 20, 170, 11.1, 7.1
    reference_outcomes = [20,170,11.1,7.1]
    epsilons = [.75, 4, 0.55, 0.15]
    epsilons = [x*3 for x in epsilons]
    epsilons = [x*.1 for x in reference_outcomes]
    # Create instances of the crossover and mutation operators
    crossover = SBX(probability=1, distribution_index=20)
    mutation = PM(probability=1, distribution_index=20)
    
     #%% Specify inputs
    #Set parametric uncetainties
    model.uncertainties = [
        # External factors
        RealParameter("X1_car_demand", -0.3, 0.2, variable_name="C3"),
        RealParameter("X2_truck_demand", -0.3, 0.2, variable_name="C4"),
        RealParameter("X3_fossil_fuel_price", 0.4, 1.3, variable_name="C7"),
        RealParameter("X4_bio_fuel_price", 0.8, 2.2, variable_name="C8"),
        RealParameter("X5_electricity_price", 0.75, 1.25, variable_name="C9"),
        RealParameter("X6_car_electrification_rate", 0.35, 0.9, variable_name="C5"),
        RealParameter("X7_truck_electrification_rate", 0.1, 0.60, variable_name="C6"),
        RealParameter("X8_SAV_market_share", 0, 0.40, variable_name="C15"),
        RealParameter("X9_SAV_driving_cost", -.1, 1, variable_name="C18"),
        RealParameter("X10_SAV_energy_efficiency", 0, 0.25, variable_name="C17"),
        RealParameter("X11_VKT_per_SAV", .5, 2, variable_name="C16"),
        RealParameter("X12_driverless_truck_market_share", 0, 0.60, variable_name="C11"),
        RealParameter("X13_driverless_truck_driving_costs", -0.50, -0.20, variable_name="C14"),
        RealParameter("X14_driverless_truck_energy_efficiency", -0.2, -0.1, variable_name="C12"),
        RealParameter("X15_VKT_per_driverless_truck", 0.2, 0.5, variable_name="C13"),
        
        # Model relationships
        RealParameter("R1_fuel_price_to_car_electrification", 0.1, 0.4, variable_name="C22"),
        RealParameter("R2_fuel_price_to_truck_electrification", 0, 0.5, variable_name="C26"),
        RealParameter("R3_fuel_price_to_car_fuel_consumption", -0.15, 0, variable_name="C27"),
        RealParameter("R4_car_driving_cost_to_car_ownership", -0.2, -0, variable_name="C23"),
        RealParameter("R5_car_driving_cost_to_car_VKT", -0.7, -0.1, variable_name="C24"),
        RealParameter("R6_truck_driving_cost_to_truck_VKT", -1.2, -0.2, variable_name="C25")
    ]
    # Make sure external electrification parameter in model is "Yes" and bus energy use to Level 1
    default_constants = [Constant("C10", "yes"),  Constant("C62", "Level 1")]

    # Specify outcomes to use in optimization. Info is required when outcome is used as constraint
    model.outcomes = [
        ScalarOutcome("M1_CO2_TTW_total", kind=ScalarOutcome.INFO, variable_name="M1_CO2_TTW_total"),
        ScalarOutcome("M2_driving_cost_car", kind=ScalarOutcome.MINIMIZE, variable_name="M2_driving_cost_car"),
        ScalarOutcome("M3_driving_cost_truck", kind=ScalarOutcome.MINIMIZE, variable_name="M3_driving_cost_truck"),
        ScalarOutcome("M4_energy_use_bio", kind=ScalarOutcome.MINIMIZE, variable_name="M4_energy_use_bio"),
        ScalarOutcome("M5_energy_use_electricity", kind=ScalarOutcome.MINIMIZE, variable_name="M5_energy_use_electricity"),
    ]

    # Specify base levers = same between policy types
    base_levers = [
        RealParameter("L1_bio_share_diesel",
                      0, 1,
                      variable_name="C75"),
        RealParameter("L2_bio_share_gasoline",
                      0, 1,
                      variable_name="C76"),
        RealParameter("L3_fuel_tax_increase_gasoline",
                      0, .12, variable_name="C69"),
        RealParameter("L4_fuel_tax_increase_diesel",
                      0, .12, variable_name="C70"),
        RealParameter("L5_km_tax_cars",
                      0, 1, variable_name="C67"),
        RealParameter("L6_km_tax_trucks",
                      0, 2, variable_name="C68"),

    ]
    
    

    #%% Create robustness metrics

    
    robustness_functions = [
        ScalarOutcome('Mean_stdev_CO2', kind=ScalarOutcome.MINIMIZE,
                      variable_name='M1_CO2_TTW_total', function=mean_stdev),
        ScalarOutcome('Mean_stdev_DrivingCostCar', kind=ScalarOutcome.MINIMIZE,
                      variable_name='M2_driving_cost_car', function=mean_stdev),
        ScalarOutcome('Mean_stdev_DrivingCostTrucks', kind=ScalarOutcome.MINIMIZE,
                      variable_name='M3_driving_cost_truck', function=mean_stdev),
        ScalarOutcome('Mean_stdev_EnergyBio', kind=ScalarOutcome.MINIMIZE,
                      variable_name='M4_energy_use_bio', function=mean_stdev),
        ScalarOutcome('Mean_stdev_EnergyElectricity', kind=ScalarOutcome.MINIMIZE,
                      variable_name='M5_energy_use_electricity', function=mean_stdev),
    
        ScalarOutcome('Satisficing_CO2', kind=ScalarOutcome.MAXIMIZE,
                      variable_name='M1_CO2_TTW_total', function=satisficing_metric),
    ]
    # %% Specify constraints
    # Define CO2 target to use for CO2 constraint
    CO2_target = 0.1*18.9  # Set CO2 target [m ton CO2 eq / year], 2040 target is 10% of 2010 emission levels

    # Define constraint for how much the diesel and gasoline lever might differ
    def bio_levers_constraint(L1_bio_share_diesel, L2_bio_share_gasoline):
        # Specify the threshold for the diff in gasoline and diesel biofuel admixture in percentage points
        bio_lever_diff = 0.05
        # Extract the first element if Series are provided, ensuring scalar values
        if isinstance(L1_bio_share_diesel, pd.Series):
            L1_bio_share_diesel = L1_bio_share_diesel.iloc[0]
        if isinstance(L2_bio_share_gasoline, pd.Series):
            L2_bio_share_gasoline = L2_bio_share_gasoline.iloc[0]

        # Calculate the absolute difference between the shares
        difference = abs(L1_bio_share_diesel - L2_bio_share_gasoline)

        # Calculate the distance from the constraint threshold
        distance = max(0, difference - bio_lever_diff)

        return distance

    def fuel_tax_levers_constraint(L3_fuel_tax_increase_gasoline, L4_fuel_tax_increase_diesel):
        # Specify the threshold for the diff in gasoline and diesel biofuel admixture in percentage points
        fuel_tax_lever_diff = 0.02
        # Extract the first element if Series are provided, ensuring scalar values
        if isinstance(L3_fuel_tax_increase_gasoline, pd.Series):
            L3_fuel_tax_increase_gasoline = L3_fuel_tax_increase_gasoline.iloc[0]
        if isinstance(L4_fuel_tax_increase_diesel, pd.Series):
            L4_fuel_tax_increase_diesel = L4_fuel_tax_increase_diesel.iloc[0]

        # Calculate the absolute difference between the shares
        difference = abs(L3_fuel_tax_increase_gasoline - L4_fuel_tax_increase_diesel)

        # Calculate the distance from the constraint threshold
        distance = max(0, difference - fuel_tax_lever_diff)

        return distance

    # Specify the set of constraints

    constraints = [#Constraint("max CO2", outcome_names="M1_CO2_TTW_total",
                   #           function=lambda x: max(0, x-CO2_target)),
                   Constraint("bio levers",
                              parameter_names=["L1_bio_share_diesel", "L2_bio_share_gasoline"],
                              function=bio_levers_constraint),
                   Constraint("fuel tax levers",
                              parameter_names=["L3_fuel_tax_increase_gasoline", "L4_fuel_tax_increase_diesel"],
                              function=fuel_tax_levers_constraint),

                   ]
    #%% Create scenarios to evaluate search over
    scenario_list = [] # placeholder list to store scenarios
    # Specify reference scenario
    reference_scenario = {
        # External factors
        "X1_car_demand": 0,
        "X2_truck_demand": 0,
        "X3_fossil_fuel_price": 1,
        "X4_bio_fuel_price": 1,
        "X5_electricity_price": 1,
        "X6_car_electrification_rate": .68,
        "X7_truck_electrification_rate": .3,
        "X8_SAV_market_share": 0,
        "X9_SAV_driving_cost": 0,
        "X10_SAV_energy_efficiency": 0,
        "X11_VKT_per_SAV": 0,
        "X12_driverless_truck_market_share": 0,
        "X13_driverless_truck_driving_costs": 0,
        "X14_driverless_truck_energy_efficiency": 0,
        "X15_VKT_per_driverless_truck": 0,
        # Model relationships
        "R1_fuel_price_to_car_electrification": 0.19,
        "R2_fuel_price_to_truck_electrification": 0,
        "R3_fuel_price_to_car_fuel_consumption": -0.05,
        "R4_car_driving_cost_to_car_ownership": -0.1,
        "R5_car_driving_cost_to_car_VKT": -0.2,
        "R6_truck_driving_cost_to_truck_VKT": -1.14
    }
    # Add reference scenario to scenario list
    scenario_list.append(Scenario("reference", **reference_scenario))
    #select number of scenarios (per policy)
    nr_scenarios = 10
    scenarios = samplers.sample_uncertainties(model, nr_scenarios, sampler=samplers.LHSSampler())
    # Create scenario objects of the sampled scenarios and add to scenarioset
    
    for i in range(nr_scenarios):
        s_dict = {k: v for k, v in zip(scenarios.params, scenarios.designs[i])}
        scenario_list.append(Scenario(str(i), **s_dict))
  
    #%% run robust optimizaiton
    run_robust_opt=True
    if run_robust_opt:

        # Settings
        os.makedirs("../archives_robust", exist_ok=True)
        os.makedirs("../output_data/robust_opt", exist_ok=True)
        
        temp_dir_root = os.getcwd()
        for d in os.listdir(temp_dir_root):
            if d.startswith("tmp") and os.path.isdir(os.path.join(temp_dir_root, d)):
                try:
                    shutil.rmtree(os.path.join(temp_dir_root, d))
                except Exception as e:
                    print(f"Could not delete {d}: {e}")
        save_files = True
        date_tag = str(date.today())
        
        # Assumes you've already defined: model, base_levers, default_constants, robustness_functions, scenario_list
        
        tic = time.perf_counter()
        
        for policy_type in policy_types:
            print(f"Running ROBUST optimization for policy type: {policy_type}")
        
            # Apply policy-specific model constants and levers
            if policy_type == "All levers":
                model.levers = base_levers
                model.constants = default_constants + [
                    Constant("C71", .05),
                    Constant("C72", .05),
                    Constant("C73", .26),
                    Constant("C74", .17),
                ]
            elif policy_type == "No transport efficiency":
                model.levers = base_levers
                model.constants = default_constants + [
                    Constant("C71", 0),
                    Constant("C72", 0),
                    Constant("C73", 0),
                    Constant("C74", 0),
                ]
        
            # Assign robust outcomes
            model.outcomes.clear()
            model.outcomes = robustness_functions
            

            tmp_folder = "../archives_robust/tmp"
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
                        
            # Setup convergence metrics
            convergence_metrics = [
                ArchiveLogger(
                    "../archives_robust",
                    [lever.name for lever in model.levers],
                    [outcome.name for outcome in model.outcomes],
                    base_filename=f"robust_{nfe}_{policy_type}_{date_tag}.tar.gz",
                ),
                EpsilonProgress(),
            ]
        
            # Run the robust optimization
            with MultiprocessingEvaluator(model, n_processes=n_processes) as evaluator:
                results, convergence = evaluator.robust_optimize(
                    robustness_functions=robustness_functions,
                    scenarios=scenario_list,
                    nfe=nfe,
                    epsilons=[0.025] * len(robustness_functions),
                    convergence=convergence_metrics,
                    searchover="levers",
                    population_size=100,
                )
        
            # Save results
            if save_files:
                filename = f"{policy_type}_{nfe}_robustopt_{date_tag}"
                pickle.dump([results, convergence, scenario_list, model.outcomes],
                            open(f"../output_data/robust_opt/{filename}.p", "wb"))
                pickle.dump(model, open(f"../output_data/robust_opt/{filename}_model.p", "wb"))
        
            print(f"✅ Finished {policy_type} | {len(results)} results")
            print(f"⏱️ Runtime: {(time.perf_counter() - tic)/60:.2f} minutes")
