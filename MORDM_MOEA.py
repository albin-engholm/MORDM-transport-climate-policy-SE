# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:11:08 2022

@author: aengholm
This is a script for setting up and performing a MOEA for the TRV Scenario tool. This is the first step of MORDM.
Results are saved and can be used for further analysis in separate scripts. 
"""
# %% Imports
from ema_workbench import (RealParameter, CategoricalParameter,
                           ScalarOutcome, ema_logging,
                           perform_experiments, Constant,
                           Scenario, Constraint,
                           ema_logging, save_results,
                           MultiprocessingEvaluator)
from ema_workbench.em_framework import samplers
from ema_workbench.em_framework.optimization import (
    EpsilonProgress, ArchiveLogger)
from ema_workbench.connectors.excel import ExcelModel
from platypus import SBX, PM, GAOperator
import matplotlib.pyplot as plt
from datetime import date
import time
import pandas as pd
import pickle
# %% Specify model and optimization parameters
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)
    # Set model
    model = ExcelModel("scenarioModel", wd="./models",
                       model_file='Master - Cleanup.xlsx')
    model.default_sheet = "EMA"
    sample_scenarios = False  # Should reference scenario(s) be sampled?
    manual_scenarios = True  # Should reference scenario(s) be specified manually?
    if sample_scenarios:
        n_scenarios = 8  # Number of scenarios to sample if sampling is used
        sampler = samplers.LHSSampler()
    load_diverse_scenarios = False  # Should a pre-generated set of diverse scenarios be loaded and used as reference?

    n_p = -4  # set # of parallel threads
    nfe = 1000000  # Set number of nfes  in optimization
    #nfe = 10003
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
    # %% #Specify inputs
    model.uncertainties = [
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
        RealParameter("X15_VKT_per_driverless_truck", 0.2, 0.5, variable_name="C13")
    ]
    # Make sure external electrification parameter in model is "Yes" and bus energy use to Level 1
    default_constants = [Constant("C10", "yes"),  Constant("C62", "Level 1")]

    # Specify outcomes to use in optimization. Info is required when outcome is used as constraint
    model.outcomes = [
        ScalarOutcome("M1_CO2_TTW_total", ScalarOutcome.INFO,
                      variable_name="C57"),  # Min / info

        ScalarOutcome("M2_driving_cost_car", ScalarOutcome.MINIMIZE,
                      variable_name="C53"),

        ScalarOutcome("M3_driving_cost_truck", ScalarOutcome.MINIMIZE,
                      variable_name="C54"),

        ScalarOutcome("M4_energy_use_bio", ScalarOutcome.MINIMIZE,
                      variable_name="C41"),

        ScalarOutcome("M5_energy_use_electricity", ScalarOutcome.MINIMIZE,
                      variable_name="C43"),  # min,
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
    # %% Create reference scenario(s)
    scenario_list = []  # list of scenario dicts to store scenarios

    # If scenarios should be sampled, sample them
    if sample_scenarios is True:
        def Extract(lst, i):
            return [item[i] for item in lst]

        scenarios = samplers.sample_uncertainties(
            model, n_scenarios, sampler=sampler)
        scenarios_dict = dict.fromkeys(scenarios.params)
        # create a dict with all scenario parameters based on scenarios
        count = 0
        for i in scenarios_dict.keys():
            scenarios_dict[str(i)] = Extract(scenarios.designs, count)
            count = count+1

        # create a scenario-dict for a single scenario
        scenario_dict = dict.fromkeys(scenarios.params)
        for j in range(len(scenarios.designs)):
            scenario_dict = dict.fromkeys(scenarios.params)
            for key in scenario_dict.keys():
                scenario_dict[str(key)] = scenarios_dict[str(key)][j]
            scenario_list.append(scenario_dict)
        df_scenarios = pd.DataFrame.from_dict(scenarios_dict)

    # If pre-generated diverse scenarios should be used, load them
    elif load_diverse_scenarios is True:
        import pickle
        df_scenarios = pickle.load(
            open("./output_data/diverse_scenarios_3.p", "rb"))
        scenario_list = []
        for i, row in df_scenarios.iterrows():
            scenario_list.append(row.to_dict())

    # If manual specification of scenario(s), specify it / them
    elif manual_scenarios is True:
        reference_scenario = {  # This is the reference scenario
            "X1_car_demand": 0,
            "X2_truck_demand": 0,
            "X3_fossil_fuel_price": 1,
            "X4_bio_fuel_price": 1,
            "X5_electricity_price": 1,
            "X6_car_electrification_rate": .68,
            "X7_truck_electrification_rate": .30,
            "X8_SAV_market_share": 0,
            "X9_SAV_driving_cost": 0,
            "X10_SAV_energy_efficiency": 0,
            "X11_VKT_per_SAV": 0,
            "X12_driverless_truck_market_share": 0,
            "X13_driverless_truck_driving_costs": 0,
            "X14_driverless_truck_energy_efficiency": 0,
            "X15_VKT_per_driverless_truck": 0
        }
        scenario_list.append(reference_scenario)
    # Store a dataframe of scenarios and number of total scenarios
    df_scenarios = pd.DataFrame(scenario_list)
    n_scenarios = len(scenario_list)
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

    constraints = [Constraint("max CO2", outcome_names="M1_CO2_TTW_total",
                              function=lambda x: max(0, x-CO2_target)),
                   Constraint("bio levers",
                              parameter_names=["L1_bio_share_diesel", "L2_bio_share_gasoline"],
                              function=bio_levers_constraint),
                   Constraint("fuel tax levers",
                              parameter_names=["L3_fuel_tax_increase_gasoline", "L4_fuel_tax_increase_diesel"],
                              function=fuel_tax_levers_constraint),

                   ]

# %% Run MOEA for each policy type and scenario
    tic = time.perf_counter()
    # TODO add support for reading model object depending on what policy types should be used
    for policy_type in policy_types:
        results_list = []  # List to store different sets of results
        convergence_list = []  # List to store different sets of convergence metrics
        print("Estimated total model evaluations: ", len(
            policy_types*nfe*len(df_scenarios)))
        scenario_count = 0
        print("Running optimization for policy type ", policy_type)
        if policy_type == "All levers":
            # Specification of levers depending on policy type
            model.levers.clear()
            model.levers = base_levers
            model.constants.clear()
            model.constants = default_constants+[
                Constant("C71", .05),  # "L7_additional_car_energy_efficiency"
                Constant("C72", .05),  # "L8_additional_truck_energy_efficiency"

                Constant("C73", .26),  # "L9_transport_efficient_planning_cars"
                Constant("C74", .17)  # L10_transport_efficient_planning_trucks"

            ]

        if policy_type == "No transport efficiency":

            model.levers.clear()
            model.levers = base_levers

            model.constants.clear()
            model.constants = default_constants+[

                Constant("C71", 0),  # "L7_additional_car_energy_efficiency"
                Constant("C72", 0),  # "L8_additional_truck_energy_efficiency"
                Constant("C73", 0),  # "L9_transport_efficient_planning_cars"
                Constant("C74", 0)  # L10_transport_efficient_planning_trucks"
            ]

        for scenario in scenario_list:
            print("Scenario: ", scenario_count)
            reference = Scenario()
            reference.data = scenario

            convergence_metrics = [
                ArchiveLogger(
                    "./archives",
                    [lever.name for lever in model.levers],
                    [outcome.name for outcome in model.outcomes if outcome.kind != 0],
                    base_filename=f"{str(nfe)}_{policy_type}_{str(date.today())}.tar.gz",
                ),
                EpsilonProgress(),
            ]

            # Create an instance of GAOperator with the operators
            variator_instance = GAOperator(crossover, mutation)

            with MultiprocessingEvaluator(msis=model, n_processes=n_p) as evaluator:
                results, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                                          epsilons=epsilons,
                                                          convergence=convergence_metrics,
                                                          constraints=constraints,
                                                          reference=reference,
                                                          population_size=100,
                                                          variator=variator_instance
                                                          )

            scenario_count = scenario_count+1
            # Reset model constants
            model.constants.clear()
            model.constants = default_constants

            # Add L7-L10 as model levers
            model.levers = [
                RealParameter("L7_additional_car_energy_efficiency",
                              0.0, .05, variable_name="C71"),
                RealParameter("L8_additional_truck_energy_efficiency",
                              0.0, .05, variable_name="C72"),
                RealParameter("L9_transport_efficient_planning_cars",
                              0, .26, variable_name="C73"),
                RealParameter("L10_transport_efficient_planning_trucks",
                              0, .17, variable_name="C74")
            ]
            if policy_type == "All levers":
                results["L7_additional_car_energy_efficiency"] = 0.05
                results["L8_additional_truck_energy_efficiency"] = 0.05
                results["L9_transport_efficient_planning_cars"] = 0.26
                results["L10_transport_efficient_planning_trucks"] = 0.17
            elif policy_type == "No transport efficiency":
                results["L7_additional_car_energy_efficiency"] = 0
                results["L8_additional_truck_energy_efficiency"] = 0
                results["L9_transport_efficient_planning_cars"] = 0
                results["L10_transport_efficient_planning_trucks"] = 0

            # plot epsilon progress
            fig, (ax1) = plt.subplots(ncols=1, sharex=True, figsize=(8, 4))
            ax1.plot(convergence.nfe, convergence.epsilon_progress)
            ax1.set_ylabel('$\epsilon$-progress')
            ax1.set_xlabel('number of function evaluations')
        results_list.append(results)
        convergence_list.append(convergence)

        toc = time.perf_counter()
        print("Runtime [s] = " + str(toc-tic))
        print("Runtime [h] = " + str(round((toc-tic)/3600, 1)))

        # Save results to file?
        save_files = True
        if save_files:
            filename = f"{policy_type}{nfe}_nfe_directed_search_MORDM_{date}"
            filename1 = filename+'.p'
            pickle.dump([results_list, convergence_list, df_scenarios, epsilons], open(
                "./output_data/moea_results/"+filename1, "wb"))
            filename2 = filename+'model_'+".p"
            pickle.dump(model, open("./output_data/moea_results/"+filename2, "wb"))
