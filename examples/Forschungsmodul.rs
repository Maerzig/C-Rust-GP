#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::process::exit;
use std::rc::Rc;
use std::time::{Duration, Instant};
use cgp_master::components::cgp_components::chromosome::Chromosome;
use cgp_master::components::evo_operators_for_population::mutation_operators::mutation_types::MutationTypes;
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
use cgp_master::utils::configurator::*;
use cgp_master::datasets::boolean_datasets;
use cgp_master::datasets::regression_benchmarks;
use cgp_master::datasets::real_world_uci;
use cgp_master::utils::cli_functions::{Cli, CrossoverArgs, DatasetArgs, MutationArgs, SelectionArgs, get_arguments};
use cgp_master::utils::logger_functions::{LoggerActiveNodes, LoggerArchive, LoggerFitness};
use cgp_master::utils::checkpoint::*;
use cgp_master::utils::utility_funcs::transpose;
use std::collections::HashMap;
use nohash_hasher::BuildNoHashHasher;
use cgp_master::components::cgp_components::cgp_node_types::NodeType;
use cgp_master::components::cgp_components::cgp_node::CGPNode;
use cgp_master::function_set::function_trait::Function;
use cgp_master::components::cgp_components::chromosome_evaluator_operators::ChromosomeEvaluatorGeneral;
use cgp_master::components::cgp_components::chromosome_find_active_node_operators::ChromosomeActiveNode;

fn run(dataset: String, selection_type: String, copy_full: bool, nbr_elitists: usize, pop_size: usize, tournament_size: usize, nbr_parent_elitists: usize) {
    
    let summary_path = dataset.clone() + ".txt";
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(summary_path)
        .expect("Cannot open summary file");

    if file.metadata().unwrap().len() == 0 {
        writeln!(file, "SelectionType,Fold,Seed,Converged,Generations,Evaluations,GenerationBestIndividualFound,AgeBestIndividual,NmbrActiveNodesBestIndividual,Time,BestTrainFitness,TestFitness,TestMSE")
            .expect("Cannot write header to summary file");
    }

    let data: Vec<Vec<f32>>;
    let labels: Vec<f32>;

    if dataset == "apnea" {
        (data, labels, _, _) = real_world_uci::apnea2::get_dataset(
        "src/datasets/real_world_uci/data/556_analcatdata_apnea2.tsv".to_string(), 
        false 
    );
    } else if dataset == "forestfire" {
        (data, labels, _, _) = real_world_uci::forestfires::get_dataset(
        "src/datasets/real_world_uci/data/forestfires.csv".to_string(), 
        false 
    );
    } else {
        println!("Invalid dataset");
        return;
    }
    
    let function_set = regression_function_set::get_regression_function_set();

    let mut dataset_args = DatasetArgs {
        dataset_type: "f32".to_string(),
        dataset: dataset.clone(),
    };
    let mut mutation_args = MutationArgs {
        mutation_type: "Single".to_string(),
        bioma_mutation_multi_n: 0,
        bioma_mutation_prob_active: 0.0,
        bioma_mutation_prob_inactive: 0.0,
        bioma_mutation_rate: 0.05,
    };
    let mut crossover_args = CrossoverArgs {
        crossover_type: "NoCrossover".to_string(),
        crossover_rate: 0.0,
        multi_point_n: 0,
    };
    let mut selection_args = SelectionArgs {
        selection_type: selection_type.clone(),
        tournament_size: tournament_size,
        elitism_number: nbr_elitists,
        population_size: pop_size,
        parent_elitists: nbr_parent_elitists,
    };
    let mut args = Cli { 
        cgp_extension_type: "Standard".to_string(), 
        nbr_nodes: 50, 
        run_id: 1,
        dataset_args: dataset_args,
        mutation_args: mutation_args,
        crossover_args: crossover_args,
        selection_args: selection_args, 
    };

    let n_folds = 5;
    let n_seeds = 5; 
    let mut cv = CrossValidation::new(data.len(), n_folds, 42);

    for fold in 0..n_folds {
        
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
        let chromosome_eval_op = Rc::new(ChromosomeEvaluatorGeneral::new());
        let mut mutation_operator = PopulationMutationGeneral::new();
        let eval_operator = get_population_evaluator_operator(&params);
        let selection_operator = get_population_selection_operator(&params);
        let clone_parent2child: Box<dyn ClonePopulation<f32>> = match copy_full {
            true => CloneParentToChildFull::new(),
            false => CloneParentToChild::new(),
        };

        for rnd_seed in 0..n_seeds {
            
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
            
            let start_time = Instant::now();
            let mut iteration_number = 1;
            let mut converged = false;
            let mut best_fitness: f32 = runner.get_best_fitness();
            let mut best_chromosome: Chromosome = get_best_parent_chromosome(&runner);
            let mut age_best_chromosome: usize = 0; // Iteration in which the currently best individual was found
            let mut evaluations_best_chromosome: usize = runner.total_evaluations; //Iterations at which the best chromosome was found
            let mut evaluation_budget: usize = 8_000_000; // Number of evaluations the run is allowed to continue; Actual number may be slightly higher since iterations are allowed to finish as long as they started below the treshold 

            while runner.total_evaluations < evaluation_budget {
                let current_best_fitness = runner.get_best_fitness();
                if current_best_fitness < best_fitness {
                    best_fitness = current_best_fitness;
                    best_chromosome = get_best_parent_chromosome(&runner);
                    age_best_chromosome = iteration_number;
                    evaluations_best_chromosome = runner.total_evaluations;
                    logger_archive.write_best_individual(iteration_number, best_fitness, &best_chromosome);
                }
                logger_fitness.write_fitness(iteration_number, current_best_fitness);
                
                if iteration_number % 500 == 0 {
                    println!("i: {}, fitness: {}, popsize: {}", iteration_number, current_best_fitness, runner.population.len());
                }

                iteration_number += 1;

                clone_parent2child.execute(&mut runner); 
                mutation_operator.execute(&mut runner, Rc::clone(&node_mutation_op), Rc::clone(&chromosome_mutation_op));
                eval_operator.execute(&mut runner, Rc::clone(&chromosome_eval_op), Rc::clone(&chromosome_active_op), Rc::clone(&function_set));
                selection_operator.execute(&mut runner);

                // If more than 10% of the evaluation budget has passed since the last time a better individual was found the run is treated as converged
                if ((runner.total_evaluations - evaluations_best_chromosome) * 10 >= evaluation_budget) && (runner.get_best_fitness() >= best_fitness) 
                {
                    converged = true;
                    break;
                }
            }

            let elapsed = start_time.elapsed();
            
            let mut best_test_fitness = 0.0;
            let mut best_test_mse = 0.0;

            if let (Some(inputs), Some(labels)) = (&runner.eval_data, &runner.eval_label) {
                best_test_fitness = chromosome_eval_op.evaluate(
                    &mut best_chromosome,
                    Rc::clone(&chromosome_active_op),
                    inputs,
                    labels,
                    Rc::clone(&function_set)
                );
                best_test_mse = chromosome_eval_op.evaluate_mse(
                    &mut best_chromosome,
                    Rc::clone(&chromosome_active_op),
                    inputs,
                    labels,
                    Rc::clone(&function_set)
                );
            };

            println!(
                "Fold {}/{}, Seed {}/{}: Converged={}, Evals={}, Time={:.2?}, TrainFit={:.4}, TestFit={:.4}, TestMSE={:.4}", 
                fold + 1, n_folds, rnd_seed + 1, n_seeds, converged, runner.total_evaluations, elapsed, best_fitness, best_test_fitness, best_test_mse
            );

            writeln!(
                file, 
                "{},{},{},{},{},{},{},{},{},{:.2?},{},{},{}", 
                selection_type,
                fold + 1, 
                rnd_seed + 1, 
                converged, 
                iteration_number, 
                runner.total_evaluations,
                age_best_chromosome,
                (iteration_number - age_best_chromosome), 
                best_chromosome.active_nodes.len(),
                elapsed,
                best_fitness, // This is train fitness
                best_test_fitness,
                best_test_mse
            ).expect("Failed to write to summary file");

            logger_fitness.write_finished_fitness(iteration_number, Some(best_fitness), Some(best_test_fitness));
            args.run_id += 1; 
        }
    }
}
fn main() {
    run("apnea".to_string(), "OnePlusFour".to_string(), false, 1, 4, 0, 0);
    run("forestfire".to_string(), "OnePlusFour".to_string(), false, 1, 4, 0, 0);
    run("apnea".to_string(), "MuCommaLambda".to_string(), false, 3, 16, 0, 1);
    run("forestfire".to_string(), "MuCommaLambda".to_string(), false, 3, 16, 0, 1);
    run("apnea".to_string(), "SAGA4Random".to_string(), true, 3, 16, 0, 0);
    run("forestfire".to_string(), "SAGA4Random".to_string(), true, 3, 16, 0, 0);
    run("apnea".to_string(), "SAGA4ParentTournament".to_string(), true, 3, 16, 4, 0);
    run("forestfire".to_string(), "SAGA4ParentTournament".to_string(), true, 3, 16, 4, 0);
    run("apnea".to_string(), "SAGA4SurvivorTournament".to_string(), true, 3, 16, 4, 0);
    run("forestfire".to_string(), "SAGA4SurvivorTournament".to_string(), true, 3, 16, 4, 0);
}
