#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{BufWriter, Read, Write};
use std::iter;
use std::path::Path;
use std::process::exit;
use std::rc::Rc;
use std::time::{Duration, Instant};
use cgp_master::components::cgp_components::chromosome::Chromosome;
use cgp_master::components::evo_operators_for_population::adaptation_operators::adaptation_trait::GeneralAdaptation;
use cgp_master::components::evo_operators_for_population::adaptation_operators::adaptation_types::AdaptationTypes;
use cgp_master::components::evo_operators_for_population::adaptation_operators::mutation_rate_adaptation::BaeckCoupled;
use cgp_master::components::evo_operators_for_population::mutation_operators::mutation_types::MutationTypes;
use cgp_master::components::evo_operators_for_population::mutation_operators::mutate_population_general;
use cgp_master::components::evo_operators_for_population::restart_operators::phenotype_restart::PhenotypicDiversityRestart;
use cgp_master::components::evo_operators_for_population::restart_operators::restart_trait::GeneralRestart;
use cgp_master::components::evo_operators_for_population::selection_operators::elitist_selection_saga4_random::PopulationSelectionSAGA4Random;
use cgp_master::components::evo_operators_for_population::selection_operators::selection_types::SelectionTypes;
use cgp_master::datasets::real_world_uci::crossvalidation::CrossValidation;
use cgp_master::function_set::regression_function_set;
use cgp_master::components::cgp_components::cgp_node_mutation_operators::*;
use cgp_master::global_params::CgpParameters;
use cgp_master::components::cgp_components::cgp_types::CGPType;
use cgp_master::components::cgp_components::chromosome_evaluator_operators::*;
use cgp_master::components::cgp_components::chromosome_find_active_node_operators::*;
use cgp_master::components::cgp_components::chromosome_mutation_operators::*;
use cgp_master::components::cgp_components::chromosome_reorder_operators::*;
use cgp_master::components::evo_operators_for_population::crossover_operators::crossover_trait::PopulationGeneralCrossover;
use cgp_master::components::evo_operators_for_population::crossover_operators::crossover_types::CrossoverType;
use cgp_master::components::evo_operators_for_population::evaluation_operators::eval_population_oneplusfour::PopulationForwardPassOnePlusFour;
use cgp_master::components::evo_operators_for_population::evaluation_operators::eval_population_trait::PopulationGeneralForwardPass;
use cgp_master::components::evo_operators_for_population::general_operators::clone_parent_to_child::{self, CloneParentToChild, CloneParentToChildFull, ClonePopulation};
use cgp_master::components::evo_operators_for_population::general_operators::reorder_population::GeneralReorderPopulationTrait;
use cgp_master::components::evo_operators_for_population::mutation_operators::mutate_population_general::PopulationMutationGeneral;
use cgp_master::components::evo_operators_for_population::selection_operators::elitist_selection_oneplusfour::PopulationElitistSelectionOnePlusFour;
use cgp_master::components::evo_operators_for_population::mutation_operators::mutation_trait::PopulationGeneralMutation;
use cgp_master::components::evo_operators_for_population::selection_operators::selection_trait::PopulationGeneralSelection;

use cgp_master::utils::runner::{ProgramState, get_best_parent_chromosome};
use cgp_master::utils::{configurator::*, elitist_archive};
use cgp_master::datasets::boolean_datasets;
use cgp_master::datasets::regression_benchmarks;
use cgp_master::datasets::real_world_uci;
use cgp_master::utils::cli_functions::{AdaptationArgs, Cli, CrossoverArgs, DatasetArgs, MutationArgs, RestartArgs, SelectionArgs, get_arguments};
use cgp_master::utils::logger_functions::{LoggerActiveNodes, LoggerArchive, LoggerFitness, LoggerRestart};
use cgp_master::utils::checkpoint::*;
use cgp_master::utils::utility_funcs::transpose;
use std::collections::HashMap;
use nohash_hasher::BuildNoHashHasher;
use cgp_master::components::cgp_components::cgp_node_types::NodeType;
use cgp_master::components::cgp_components::cgp_node::CGPNode;
use cgp_master::function_set::function_trait::Function;
use cgp_master::components::cgp_components::chromosome_evaluator_operators::ChromosomeEvaluatorGeneral;
use cgp_master::components::cgp_components::chromosome_find_active_node_operators::ChromosomeActiveNode;

pub fn run(mut args: Cli) {
    let (data, labels) = match cgp_master::datasets::load_dataset(&args.dataset_args) {
        Ok(res) => res,
        Err(err) => {
            println!("Invalid dataset: {}", err);
            return;
        }
    };

    let save_dir = Path::new("")
        .join(format!("Experiments_Output_{}", args.dataset_args.dataset_type))
        .join(format!("cgp_extension_type_{}", args.cgp_extension_type))
        .join(format!("dataset_{}", args.dataset_args.dataset))
        .join(format!("mutation_type_{}", args.mutation_args.mutation_type))
        .join(format!("selection_type_{}", args.selection_args.selection_type))
        .join(format!("crossover_type_{}", args.crossover_args.crossover_type))
        .join(format!("adaptation_type_{}", args.adaptation_args.adaptation_type))
        .join(format!("restart_type_{}", args.restart_args.restart_type));

    create_dir_all(&save_dir).expect("Cannot create experiment output directory");
    let summary_path = save_dir.join(args.dataset_args.dataset.clone() + ".csv");
    let summary_es_path = save_dir.join(format!("{}_early_stopping.csv", args.dataset_args.dataset));

    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(summary_path)
        .expect("Cannot open summary file");

    let mut file_es = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(summary_es_path)
        .expect("Cannot open early stopping summary file");

    if file.metadata().unwrap().len() == 0 {
        if args.dataset_args.problem_type == "classification" {
            writeln!(file, "SelectionType,Fold,Seed,Converged,Generations,Evaluations,GenerationBestIndividualFound,AgeBestIndividual,NmbrActiveNodesBestIndividual,Time,BestTrainMCC,BestTrainMAE,TestMCC,TestMAE,TestMSE")
                .expect("Cannot write header to summary file");
        } else {
            writeln!(file, "SelectionType,Fold,Seed,Converged,Generations,Evaluations,GenerationBestIndividualFound,AgeBestIndividual,NmbrActiveNodesBestIndividual,Time,BestTrainFitness,TestFitness,TestMSE")
                .expect("Cannot write header to summary file");
        }
    }

    if file_es.metadata().unwrap().len() == 0 {
        if args.dataset_args.problem_type == "classification" {
            writeln!(file_es, "SelectionType,Fold,Seed,Converged,Generations,Evaluations,GenerationBestIndividualFound,AgeBestIndividual,NmbrActiveNodesBestIndividual,Time,BestTrainMCC,BestTrainMAE,TestMCC,TestMAE,TestMSE")
                .expect("Cannot write header to early stopping summary file");
        } else {
            writeln!(file_es, "SelectionType,Fold,Seed,Converged,Generations,Evaluations,GenerationBestIndividualFound,AgeBestIndividual,NmbrActiveNodesBestIndividual,Time,BestTrainFitness,TestFitness,TestMSE")
                .expect("Cannot write header to early stopping summary file");
        }
    }

    let function_set = regression_function_set::get_regression_function_set();

    let mut cv = CrossValidation::new(data.len(), args.folds, 42);

    for fold in 0.. args.folds {
        
        let (train_data, train_label_raw, test_data, test_label_raw) = 
            cv.split(data.clone(), labels.clone());

        // Adapt labels to the format ProgramState is expecting
        let train_label_rows: Vec<Vec<f32>> = train_label_raw.into_iter().map(|l| vec![l]).collect();
        let train_label = transpose(train_label_rows);

        let test_label_rows: Vec<Vec<f32>> = test_label_raw.into_iter().map(|l| vec![l]).collect();
        let test_label = Some(transpose(test_label_rows));

        let params = make_cgp_params(&args, train_data[0].len(), train_label.len(), function_set.len());
        
        let node_mutation_op = Rc::new(get_node_mutation_operator(&params));
        let chromosome_active_op = Rc::new(get_active_node_finder_operator(&params));
        let chromosome_mutation_op: Rc<Box<dyn ChromosomeMutation>> = Rc::new(get_chromosome_mutation_operator(&params));
        let chromosome_eval_op: Rc<Box<dyn ChromosomeEvaluation<f32>>> = Rc::from(get_chromosome_evaluator_operator(&args));
        let mut mutation_operator = PopulationMutationGeneral::new();
        let eval_operator = get_population_evaluator_operator(&params);
        let selection_operator = get_population_selection_operator(&params);
        let clone_parent_to_child: Box<dyn ClonePopulation<f32>> = CloneParentToChild::new();
        let mut mutation_rate_adaptation: Box<dyn GeneralAdaptation<f32>> = get_adaptation_operator(&params);
        let restart_op: Box<dyn GeneralRestart<f32>> = get_restart_operator(&params);

        for rnd_seed in 0..args.seeds {
            
            let mut runner = ProgramState::new(
                params.clone(), 
                train_data.clone(), 
                train_label.clone(), 
                Some(test_data.clone()), 
                test_label.clone(), 
                Rc::clone(&function_set), 
                Rc::clone(&chromosome_eval_op), 
                Rc::clone(&chromosome_active_op)
            );

            let mut logger_fitness = LoggerFitness::new(&args, &params);
            let mut logger_archive = LoggerArchive::new(&args); 
            let mut logger_restart = LoggerRestart::new(&args);
            
            let start_time = Instant::now();
            let mut iteration_number = 1;
            let mut converged = false;
            let mut best_fitness: f32 = runner.get_best_fitness(); // Best fitness of the *current* restart cycle. Can be worse than the lifetime best
            let mut generation_best_individual_found: usize = 1; // Iteration in which the lifetime best individual was found
            let mut evaluations_best_chromosome: usize = runner.total_evaluations; //Evaluations at which the best chromosome was found
            let mut evaluation_budget: usize = args.eval_budget; // Number of evaluations the run is allowed to continue; Actual number may be slightly higher since iterations are allowed to finish as long as they started below the treshold 
            let mut unproductive_restarts: usize = 0;
            let mut total_restarts_count: usize = 0;
            let mut lifetime_best_fitness = runner.get_best_fitness();
            let mut lifetime_best_chromosome: Chromosome = get_best_parent_chromosome(&runner);
            let mut pre_restart_fitness = lifetime_best_fitness;
            let mut current_cycle_improved = false;
            let mut early_stop_logged = false;

            PhenotypicDiversityRestart::compute_diversity(&mut runner); // Calculates diversity of initial population before a restart

            while runner.total_evaluations < evaluation_budget {
                let current_best_fitness = runner.get_best_fitness();

                if current_best_fitness < lifetime_best_fitness {
                    lifetime_best_fitness = current_best_fitness;
                    lifetime_best_chromosome = get_best_parent_chromosome(&runner);
                    generation_best_individual_found = iteration_number;
                    unproductive_restarts = 0;
                    current_cycle_improved = true;
                    logger_archive.write_best_individual(iteration_number, lifetime_best_fitness, &lifetime_best_chromosome);
                }

                if current_best_fitness < best_fitness {
                    best_fitness = current_best_fitness;
                    evaluations_best_chromosome = runner.total_evaluations;
                }

                if current_best_fitness <=  params.fitness_threshold{
                    converged = true;
                    println!("Run SUCCESSFUL: Converged to target fitness {} at eval {}", current_best_fitness, runner.total_evaluations);
                    break;
                }

                let evaluations_stagnant = runner.total_evaluations - evaluations_best_chromosome;
                let train_mae = if args.dataset_args.problem_type == "classification" {
                    Some(get_best_parent_chromosome(&runner).mae)
                } else {
                    None
                };
                logger_fitness.write_fitness(iteration_number, &runner, train_mae, evaluations_stagnant, false);

                iteration_number += 1;

                clone_parent_to_child.execute(&mut runner); 
                mutation_rate_adaptation.execute(&mut runner); // Adapt mutation rate before mutation happens so it can be evaluated properly
                mutation_operator.execute(&mut runner, Rc::clone(&node_mutation_op), Rc::clone(&chromosome_mutation_op));
                eval_operator.execute(&mut runner, Rc::clone(&chromosome_eval_op), Rc::clone(&chromosome_active_op), Rc::clone(&function_set));
                selection_operator.execute(&mut runner);
                runner.update_elitist_archive();

                let max_stagnant_evals = ((evaluation_budget as f32) * (params.max_stagnant_evals_pct / 100.0)) as usize;

                if evaluations_stagnant >= max_stagnant_evals {
                    let cycle_best_before = best_fitness;
                    let fitness_at_restart = runner.get_best_fitness();
                    let lifetime_best_before = lifetime_best_fitness;
                    let evals_before = runner.total_evaluations;
                    let pre_restart_active_rate = get_best_parent_chromosome(&runner).active_mutation_rate;

                    restart_op.execute(&mut runner, Rc::clone(&function_set), Rc::clone(&chromosome_eval_op), Rc::clone(&chromosome_active_op));

                    // Reinitialize variables if restart occured
                    if runner.total_evaluations > evals_before + 1 {
                        total_restarts_count += 1;
                        let post_restart_initial_fitness = runner.get_best_fitness();
                        let post_restart_active_rate = get_best_parent_chromosome(&runner).active_mutation_rate;

                        logger_restart.write_restart_event(
                            total_restarts_count, 
                            runner.total_evaluations, 
                            cycle_best_before,
                            fitness_at_restart, 
                            lifetime_best_before, 
                            post_restart_initial_fitness, 
                            pre_restart_active_rate,
                            post_restart_active_rate,
                            current_cycle_improved, 
                            runner.diversity
                        );
                        
                        let restart_train_mae = if args.dataset_args.problem_type == "classification" {
                            Some(get_best_parent_chromosome(&runner).mae)
                        } else {
                            None
                        };
                        logger_fitness.write_fitness(iteration_number, &runner, restart_train_mae, evaluations_stagnant, true);
                        
                        if !current_cycle_improved {
                            unproductive_restarts += 1;
                            println!("Restart cycle #{} failed to improve lifetime best (Successive unproductive restarts: {})", total_restarts_count, unproductive_restarts);

                            if unproductive_restarts == 3 && !early_stop_logged {
                                early_stop_logged = true;
                                let mut es_chromo = lifetime_best_chromosome.clone();
                                let mut es_test_fit = 0.0;
                                let mut es_test_mse = 0.0;
                                if let (Some(inputs), Some(labels)) = (&runner.eval_data, &runner.eval_label) {
                                    es_test_fit = chromosome_eval_op.evaluate(&mut es_chromo, Rc::clone(&chromosome_active_op), inputs, labels, Rc::clone(&function_set));
                                    es_test_mse = chromosome_eval_op.evaluate_mse(&mut es_chromo, Rc::clone(&chromosome_active_op), inputs, labels, Rc::clone(&function_set));
                                }
                                let es_test_mae = if runner.eval_data.is_some() { es_chromo.mae } else { 0.0 };

                                if args.dataset_args.problem_type == "classification" {
                                    writeln!(file_es, "{},{},{},false,{},{},{},{},{},{:.2?},{},{},{},{},{}",
                                        args.selection_args.selection_type, fold + 1, rnd_seed + 1,
                                        iteration_number, runner.total_evaluations, generation_best_individual_found,
                                        iteration_number - generation_best_individual_found, es_chromo.active_nodes.len(),
                                        start_time.elapsed(), 1.0 - lifetime_best_fitness, lifetime_best_chromosome.mae,
                                        1.0 - es_test_fit, es_test_mae, es_test_mse
                                    ).expect("Failed to write to early stopping summary file");
                                } else {
                                    writeln!(file_es, "{},{},{},false,{},{},{},{},{},{:.2?},{},{},{}",
                                        args.selection_args.selection_type, fold + 1, rnd_seed + 1,
                                        iteration_number, runner.total_evaluations, generation_best_individual_found,
                                        iteration_number - generation_best_individual_found, es_chromo.active_nodes.len(),
                                        start_time.elapsed(), lifetime_best_fitness, es_test_fit, es_test_mse
                                    ).expect("Failed to write to early stopping summary file");
                                }
                            }
                        }

                        // Reset cycle state for the new restart phase
                        current_cycle_improved = false;
                        evaluations_best_chromosome = runner.total_evaluations;
                        best_fitness = post_restart_initial_fitness;

                        // Re-compute diversity for reinitialized population
                        PhenotypicDiversityRestart::compute_diversity(&mut runner);
                    }
                }
            }

            // Check if the final generation evaluated produced a new lifetime best
            let final_best_fitness = runner.get_best_fitness();
            if final_best_fitness < lifetime_best_fitness {
                lifetime_best_fitness = final_best_fitness;
                lifetime_best_chromosome = get_best_parent_chromosome(&runner);
                generation_best_individual_found = iteration_number;
                logger_archive.write_best_individual(iteration_number, lifetime_best_fitness, &lifetime_best_chromosome);
            }

            let elapsed = start_time.elapsed();
            
            let mut best_test_fitness = 0.0;
            let mut best_test_mse = 0.0;

            let best_train_mae = lifetime_best_chromosome.mae;

            if let (Some(inputs), Some(labels)) = (&runner.eval_data, &runner.eval_label) {
                best_test_fitness = chromosome_eval_op.evaluate(
                    &mut lifetime_best_chromosome,
                    Rc::clone(&chromosome_active_op),
                    inputs,
                    labels,
                    Rc::clone(&function_set)
                );
                best_test_mse = chromosome_eval_op.evaluate_mse(
                    &mut lifetime_best_chromosome,
                    Rc::clone(&chromosome_active_op),
                    inputs,
                    labels,
                    Rc::clone(&function_set)
                );
            };

            let best_test_mae = if runner.eval_data.is_some() {
                lifetime_best_chromosome.mae
            } else {
                0.0
            };

            if args.dataset_args.problem_type == "classification" {

                // Reports MCC, not 1 - MCC! This differs from the fitness reported in fitness_history.csv
                println!(
                    "Fold {}/{}, Seed {}/{}: Converged={}, Evals={}, Time={:.2?}, TrainMCC={:.4}, TrainMAE={:.4}, TestMCC={:.4}, TestMAE={:.4}, TestMSE={:.4}", 
                    fold + 1, args.folds, rnd_seed + 1, args.seeds, converged, runner.total_evaluations, elapsed, 1.0 - lifetime_best_fitness, best_train_mae, 1.0 - best_test_fitness, best_test_mae, best_test_mse
                );

                writeln!(
                    file, 
                    "{},{},{},{},{},{},{},{},{},{:.2?},{},{},{},{},{}", 
                    args.selection_args.selection_type,
                    fold + 1, 
                    rnd_seed + 1, 
                    converged, 
                    iteration_number, 
                    runner.total_evaluations,
                    generation_best_individual_found,
                    (iteration_number - generation_best_individual_found), 
                    lifetime_best_chromosome.active_nodes.len(),
                    elapsed,
                    1.0 - lifetime_best_fitness,
                    best_train_mae,
                    1.0 - best_test_fitness,
                    best_test_mae,
                    best_test_mse
                ).expect("Failed to write to summary file");
            } else {
                println!(
                    "Fold {}/{}, Seed {}/{}: Converged={}, Evals={}, Time={:.2?}, TrainFit={:.4}, TestFit={:.4}, TestMSE={:.4}", 
                    fold + 1, args.folds, rnd_seed + 1, args.seeds, converged, runner.total_evaluations, elapsed, lifetime_best_fitness, best_test_fitness, best_test_mse
                );

                writeln!(
                    file, 
                    "{},{},{},{},{},{},{},{},{},{:.2?},{},{},{}", 
                    args.selection_args.selection_type,
                    fold + 1, 
                    rnd_seed + 1, 
                    converged, 
                    iteration_number, 
                    runner.total_evaluations,
                    generation_best_individual_found,
                    (iteration_number - generation_best_individual_found), 
                    lifetime_best_chromosome.active_nodes.len(),
                    elapsed,
                    lifetime_best_fitness, // Write true lifetime best train fitness
                    best_test_fitness,
                    best_test_mse
                ).expect("Failed to write to summary file");
            }

            logger_fitness.write_finished_fitness(iteration_number, Some(lifetime_best_fitness), Some(best_fitness), Some(best_test_fitness));
            args.run_id += 1; 
        }
    }
}

fn main() {
    let args = get_arguments();
    run(args);
}
