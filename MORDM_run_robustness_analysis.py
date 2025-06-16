# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:18:42 2023

@author: aengholm
"""
from ema_workbench import (RealParameter, CategoricalParameter,
                           ScalarOutcome, ema_logging,
                           perform_experiments, Constant, Policy, MultiprocessingEvaluator)

from ema_workbench.connectors.excel import ExcelModel
if __name__ == "__main__":
    ema_logging.log_to_stderr(level=ema_logging.INFO)
    # %% Load candidate policies and model from previous optimization
    import pandas as pd
    df_full = pd.DataFrame()
    policy_types = ["All levers", "No transport efficiency"]
    # policy_types = ["All levers"]  # ,"No transport efficiency"]
    load_results = 1
    if load_results == 1:

        count = 0
        for policy_type in policy_types:
            if count == 0:
                date = "2025-03-07"
                nfe = 1000000
                t1 = f"./output_data/moea_results/{policy_type}{nfe}_nfe_directed_search_MORDM_{date}.p"
                import pickle
                results_list, convergence, scenarios, epsilons = pickle.load(
                    open(t1, "rb"))
                t2 = f"./output_data/moea_results/{policy_type}{nfe}_nfe_directed_search_MORDM_{date}model_.p"
                model = pickle.load(open(t2, "rb"))
                scenario_count = 0
                for results in results_list:
                    results["Policy type"] = policy_type
                    df_full = pd.concat([df_full, results], ignore_index=True)
                    scenario_count = scenario_count+1

            if count == 1:
                date = "2025-03-05"
                nfe = 1000000
                t1 = f"./output_data/moea_results/{policy_type}{nfe}_nfe_directed_search_MORDM_{date}.p"
                import pickle
                results_list, convergence, scenarios, epsilons = pickle.load(
                    open(t1, "rb"))
                t2 = f"./output_data/moea_results/{policy_type}{nfe}_nfe_directed_search_MORDM_{date}model_.p"
                model = pickle.load(open(t2, "rb"))
                for results in results_list:
                    results["Policy type"] = policy_type
                    df_full = pd.concat([df_full, results], ignore_index=True)
            count = count+1

        # Load candidate policy dataframe
        date = "2025-03-05"
        nfe = 1000000
        filename = f"./output_data/candidate_policies/{date}_{nfe}_candidate_policies.p"
        candidate_policy_data = pickle.load(open(filename, "rb"))
        model.working_directory = "./models"
    # The model object already contains all information about levers and uncertainties

    # %% Create a list of all policies
    candidate_policies = []
    for i, policy in candidate_policy_data.iterrows():
        candidate_policies.append(Policy(str(i), **policy.to_dict()))

        # %% Add other interesting outcomes to calculate

    model.outcomes = [
        ScalarOutcome("CO2 TTW change light vehicles", ScalarOutcome.INFO,
                      variable_name="C35"),

        ScalarOutcome("CO2 TTW change trucks", ScalarOutcome.INFO,
                      variable_name="C36"),

        ScalarOutcome("VKT light vehicles", ScalarOutcome.INFO,
                      variable_name="C38"),

        ScalarOutcome("VKT trucks", ScalarOutcome.INFO,
                      variable_name="C39"),

        ScalarOutcome("VKT total", ScalarOutcome.INFO,
                      variable_name="C40"),

        ScalarOutcome("Energy fossile total", ScalarOutcome.INFO,
                      variable_name="C42"),

        ScalarOutcome("Energy total", ScalarOutcome.INFO,
                      variable_name="C44"),

        ScalarOutcome("Electrified VKT share light vehicles", ScalarOutcome.INFO,
                      variable_name="C45"),
        ScalarOutcome("Electrified VKT share trucks", ScalarOutcome.INFO,
                      variable_name="C46"),
        ScalarOutcome("Electric share of total energy", ScalarOutcome.INFO,
                      variable_name="C47"),


        ScalarOutcome("Fossile fuel price relative reference trucks", ScalarOutcome.INFO,
                      variable_name="C51"),
        ScalarOutcome("Fossile fuel price relative reference light vehicles", ScalarOutcome.INFO,
                      variable_name="C52"),

        ScalarOutcome("Delta tax income", ScalarOutcome.INFO,
                      variable_name="C50")
    ]

    # scenarios=[scenarios,reference]
    # %% Set up two runs for various uncertainty parameters #Run model - for open exploration
    n_p = -4
    # Run
    import time
    tic = time.perf_counter()
    results_list_OE = []
    uncertainty_sets = ["X", "XP"]
    uncertainty_sets = ["XP"]

    model.uncertainties.clear()
    model.uncertainties = [
        RealParameter("X1_car_demand", -0.3, 0.2, variable_name="C3"),
        RealParameter("X2_truck_demand", -0.3, 0.2, variable_name="C4"),
        RealParameter("X3_fossil_fuel_price", 0.4, 1.3, variable_name="C7"),
        RealParameter("X4_bio_fuel_price", 0.8, 2.2, variable_name="C8"),
        RealParameter("X5_electricity_price", 0.75, 1.25, variable_name="C9"),
        RealParameter("X6_car_electrification_rate", 0.35, 0.9, variable_name="C5"),
        RealParameter("X7_truck_electrification_rate", 0.1, 0.60, variable_name="C6"),
        RealParameter("X8_SAV_market_share", 0, 0.40, variable_name="C15"),
        RealParameter("X9_SAV_driving_cost", -0.1, 1, variable_name="C18"),
        RealParameter("X10_SAV_energy_efficiency", 0, 0.25, variable_name="C17"),
        RealParameter("X11_VKT_per_SAV", 0.5, 2, variable_name="C16"),
        RealParameter("X12_driverless_truck_market_share", 0, 0.60, variable_name="C11"),
        RealParameter("X13_driverless_truck_driving_costs", -0.50, -0.20, variable_name="C14"),
        RealParameter("X14_driverless_truck_energy_efficiency", -0.1, 0, variable_name="C12"),
        RealParameter("X15_VKT_per_driverless_truck", 0.2, 0.5, variable_name="C13")
    ]
    reference_scenario = {
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
        "X15_VKT_per_driverless_truck": 0
    }
    for uncertainty_set in uncertainty_sets:
        if uncertainty_set == "XP":

            # Add R parameters
            model.uncertainties = [
                RealParameter("R1_fuel_price_to_car_electrification", 0.1, 0.4, variable_name="C22"),
                RealParameter("R2_fuel_price_to_truck_electrification", 0, 0.5, variable_name="C26"),
                RealParameter("R3_fuel_price_to_car_fuel_consumption", -0.15, 0, variable_name="C27"),
                RealParameter("R4_car_driving_cost_to_car_ownership", -0.2, -0, variable_name="C23"),
                RealParameter("R5_car_driving_cost_to_car_VKT", -0.7, -0.1, variable_name="C24"),
                RealParameter("R6_truck_driving_cost_to_truck_VKT", -1.2, -0.2, variable_name="C25")]
            reference_scenario.update({
                "R1_fuel_price_to_car_electrification": 0.19,
                "R2_fuel_price_to_truck_electrification": 0,
                "R3_fuel_price_to_car_fuel_consumption": -0.05,
                "R4_car_driving_cost_to_car_ownership": -0.1,
                "R5_car_driving_cost_to_car_VKT": -0.2,
                "R6_truck_driving_cost_to_truck_VKT": -1.14
            })

        #  Generate scenarios to  explore
        from ema_workbench.em_framework import samplers
        from ema_workbench import Scenario

        # Create the reference scenario and add to scenario list

        scenario_list = []

        reference = Scenario("Reference", **reference_scenario)
        scenario_list.append(reference)
        # Sample additional scenarios
        nr_scenarios_per_uncertainty = 100
        nr_scenarios = int(nr_scenarios_per_uncertainty *
                           len(model.uncertainties.keys()))  # select number of scenarios (per policy)
        nr_scenarios = 10000
        scenarios = samplers.sample_uncertainties(model, nr_scenarios, sampler=samplers.LHSSampler())
        # Create scenario objects of the sampled scenarios and add to scenarioset
        for i in range(nr_scenarios):
            s_dict = {k: v for k, v in zip(scenarios.params, scenarios.designs[i])}
            scenario_list.append(Scenario(str(i), **s_dict))

        with MultiprocessingEvaluator(msis=model, n_processes=n_p) as evaluator:
            experiments, outcomes = perform_experiments(model,
                                                        scenarios=scenario_list,
                                                        evaluator=evaluator,
                                                        policies=candidate_policies)

         #  Add policy type to experiments
        experiments_backup = experiments.copy()
        experiments = experiments_backup
        count = 0
        policy_type_list = []
        policy_names_sta = list(candidate_policy_data[candidate_policy_data["Policy type"] == "STA"].index)
        for policy in experiments["policy"]:
            # if policy not in policy_names_sta
            if policy not in policy_names_sta:
                policy_type_list.append(candidate_policy_data.at[int(policy), "Policy type"])
            else:
                policy_type_list.append(candidate_policy_data.at[policy, "Policy type"])
            count = count+1
        experiments["Policy type"] = policy_type_list

        results = [outcomes, experiments]
        results_list_OE.append(results)
        # %% Save results
    save_results = 1
    if save_results == 1:
        from datetime import date
        from ema_workbench import save_results
        filename_output = f"{nr_scenarios}_scenarios_MORDM_OE_{date.today()}"
        filename = str(nr_scenarios)+'_scenarios_MORDM_OE_'+str(date.today())
        filename1 = filename+'.p'
        pickle.dump([experiments, outcomes], open("./output_data/robustness_analysis_results/"+filename1, "wb"))
        import pickle
        filename2 = filename+'model_'+".p"
        pickle.dump(model, open("./output_data/robustness_analysis_results/"+filename2, "wb"))
        pickle.dump([results_list_OE], open("./output_data/robustness_analysis_results/"+"X_XP"+filename1, "wb"))
