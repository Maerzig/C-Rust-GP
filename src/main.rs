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
use cgp_master::components::evo_operators_for_population::mutation_operators::mutation_types::MutationTypes;
use cgp_master::components::evo_operators_for_population::selection_operators::elitist_selection_saga4_random::PopulationSelectionSAGA4Random;
use cgp_master::components::evo_operators_for_population::selection_operators::selection_types::SelectionTypes;
use cgp_master::datasets::real_world_uci::crossvalidation::CrossValidation;
use cgp_master::function_set::boolean_function_set;
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
use cgp_master::utils::logger_functions::{LoggerActiveNodes, LoggerFitness};
use cgp_master::utils::checkpoint::*;
use cgp_master::utils::utility_funcs::transpose;

fn breast_cancer(selection_type: String, copy_full: bool, nbr_elitists: usize, pop_size: usize, tournament_size: usize, nbr_parent_elitists: usize) {
    
    let summary_path = "Summary.txt";
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(summary_path)
        .expect("Cannot open summary file");

    if file.metadata().unwrap().len() == 0 {
        writeln!(file, "SelectionType,Fold,Seed,Converged,Generations,Evaluations,Time,BestTrainFitness,TestFitness")
            .expect("Cannot write header to summary file");
    }

    let (data, labels, _, _) = real_world_uci::breast_cancer::get_dataset(
        "src/datasets/real_world_uci/data/breast+cancer+wisconsin+diagnostic.data".to_string(), 
        false 
    );

    let function_set = regression_function_set::get_regression_function_set();

    let mut dataset_args = DatasetArgs {
        dataset_type: "f32".to_string(),
        dataset: "breast_cancer".to_string(),
    };
    let mut mutation_args = MutationArgs {
        mutation_type: "Single".to_string(),
        bioma_mutation_multi_n: 0,
        bioma_mutation_prob_active: 0.0,
        bioma_mutation_prob_inactive: 0.0,
        bioma_mutation_rate: 0.0,
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
    let mut cv = CrossValidation::new(data.len(), n_folds);

    for fold in 0..n_folds {
        
        let (train_data, train_label_raw, test_data, test_label_raw) = 
            cv.split(data.clone(), labels.clone());

        // Adapt labels to the format ProgramState is expecting
        let train_label_rows: Vec<Vec<f32>> = train_label_raw.into_iter().map(|l| vec![l as f32]).collect();
        let train_label = transpose(train_label_rows);

        let test_label_rows: Vec<Vec<f32>> = test_label_raw.into_iter().map(|l| vec![l as f32]).collect();
        let test_label = Some(transpose(test_label_rows));

        let params = make_cgp_params(&args, train_data[0].len(), train_label[0].len(), function_set.len());
        
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
            
            let start_time = Instant::now();
            let mut iteration_number = 0;
            let mut converged = false;
            let mut total_evals = runner.population.len();

            for i in 0..500_000 {
                logger_fitness.write_fitness(iteration_number, runner.get_best_fitness());
                if i % 500 == 0 {
                    println!("i: {}, fitness: {}", i, runner.get_best_fitness());
                }

                iteration_number += 1;

                clone_parent2child.execute(&mut runner); 
                mutation_operator.execute(&mut runner, Rc::clone(&node_mutation_op), Rc::clone(&chromosome_mutation_op));
                eval_operator.execute(&mut runner, Rc::clone(&chromosome_eval_op), Rc::clone(&chromosome_active_op), Rc::clone(&function_set));
                total_evals += runner.child_ids.len();
                selection_operator.execute(&mut runner);

                if runner.get_best_fitness() < params.fitness_threshold {
                    converged = true;
                    break;
                }
            }

            let elapsed = start_time.elapsed();
            
            let best_train_fitness = runner.get_best_fitness();
            let mut best_test_fitness = 0.0;

            if let (Some(inputs), Some(labels)) = (&runner.eval_data, &runner.eval_label) {
                let mut best_chromosome = get_best_parent_chromosome(&runner);
                best_test_fitness = chromosome_eval_op.evaluate(
                    &mut best_chromosome,
                    Rc::clone(&chromosome_active_op),
                    inputs,
                    labels,
                    Rc::clone(&function_set)
                );
            };

            println!(
                "Fold {}/{}, Seed {}/{}: Converged={}, Evals={}, Time={:.2?}, TrainFit={:.4}, TestFit={:.4}", 
                fold + 1, n_folds, rnd_seed + 1, n_seeds, converged, total_evals, elapsed, best_train_fitness, best_test_fitness
            );

            writeln!(
                file, 
                "{},{},{},{},{},{},{:.2?},{},{}", 
                selection_type,
                fold + 1, 
                rnd_seed + 1, 
                converged, 
                iteration_number, 
                total_evals, 
                elapsed,
                best_train_fitness,
                best_test_fitness
            ).expect("Failed to write to summary file");

            logger_fitness.write_finished_fitness(iteration_number, Some(best_train_fitness), Some(best_test_fitness));
            args.run_id += 1; 
        }
    }
}
fn main() {
    breast_cancer( "OnePlusFour".to_string(), false, 1, 4, 0, 0);
}
