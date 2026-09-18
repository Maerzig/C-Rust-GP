use std::fs;
use std::fs::File;
use std::path::Path;
use std::rc::Rc;
use crate::components::cgp_components::chromosome::Chromosome;
use crate::components::cgp_components::chromosome_find_active_node_operators::ChromosomeActiveNode;
use crate::function_set::function_trait::Function;
use crate::utils::runner::{get_best_parent_chromosome, ProgramState};
use std::io::{BufWriter, Write};
use crate::global_params::CgpParameters;
use crate::utils::cli_functions::Cli;
use serde_json::json;

pub struct LoggerActiveNodes {
    file: BufWriter<File>,
}
pub struct LoggerFitness {
    file: BufWriter<File>,
}

pub struct LoggerArchive {
    file: BufWriter<File>,
}

pub struct LoggerRestart {
    file: BufWriter<File>,
}

impl LoggerActiveNodes {
    pub fn new(args: &Cli, params: &CgpParameters) -> Self {
        let save_path = Path::new("")
            .join(format!("Experiments_Output_{}", args.dataset_args.dataset_type))
            .join(format!("cgp_extension_type_{}", args.cgp_extension_type))
            .join(format!("dataset_{}", args.dataset_args.dataset))
            .join(format!("mutation_type_{}", args.mutation_args.mutation_type))
            .join(format!("selection_type_{}", args.selection_args.selection_type))
            .join(format!("crossover_type_{}", args.crossover_args.crossover_type))
            .join(format!("run_id_{}", args.run_id));

        fs::create_dir_all(save_path.clone()).unwrap();
        let save_file_iteration = "active_nodes.txt".to_string();
        let file = BufWriter::new(File::create(save_path.join(save_file_iteration))
            .expect("cannot create file"));

        // if no parameter file exisists, create one
        let parameter_file_path = "parameters.txt".to_string();
        let parameter_file_path = save_path.join(parameter_file_path);
        if !parameter_file_path.exists() {
            let mut params_file = File::create(parameter_file_path).expect("cannot create file");
            writeln!(params_file, "{}", params).expect("cannot write to file");
        }

        Self {
            file,
        }
    }

    pub fn write_active_nodes<T>(&mut self,
                                 runner: &mut ProgramState<T>,
                                 active_node_func: Rc<Box<dyn ChromosomeActiveNode<T>>>,
                                 function_set: Rc<Vec<Box<dyn Function<T>>>>)
    {
        let mut parent = get_best_parent_chromosome(runner);

        active_node_func.execute(&mut parent, Rc::clone(&function_set));

        write!(self.file, "{:?}", parent.active_nodes).expect("cannot write");
    }
}

impl LoggerFitness {
    pub fn new(args: &Cli, params: &CgpParameters) -> Self {
        let save_path = Path::new("")
            .join(format!("Experiments_Output_{}", args.dataset_args.dataset_type))
            .join(format!("cgp_extension_type_{}", args.cgp_extension_type))
            .join(format!("dataset_{}", args.dataset_args.dataset))
            .join(format!("mutation_type_{}", args.mutation_args.mutation_type))
            .join(format!("selection_type_{}", args.selection_args.selection_type))
            .join(format!("crossover_type_{}", args.crossover_args.crossover_type))
            .join(format!("adaptation_type_{}", args.adaptation_args.adaptation_type))
            .join(format!("restart_type_{}", args.restart_args.restart_type))
            .join(format!("run_id_{}", args.run_id));

        fs::create_dir_all(save_path.clone()).unwrap();

        let save_file_iteration = "fitness_history.csv".to_string();
        let mut file = BufWriter::new(File::create(save_path.join(save_file_iteration))
            .expect("cannot create file"));

        if args.dataset_args.problem_type == "classification" {
            writeln!(file, "Iteration,Evaluations,BestFitnessMCC,BestFitnessMAE,Diversity,StagnantEvals,ActiveMutationRate,InactiveMutationRate,MeanActiveRate,MeanInactiveRate,IsRestart").expect("cannot write header");
        } else {
            writeln!(file, "Iteration,Evaluations,BestFitness,Diversity,StagnantEvals,ActiveMutationRate,InactiveMutationRate,MeanActiveRate,MeanInactiveRate,IsRestart").expect("cannot write header");
        }

        // if no parameter file exisists, create one
        let parameter_file_path = "parameters.txt".to_string();
        let parameter_file_path = save_path.join(parameter_file_path);
        if !parameter_file_path.exists() {
            let mut params_file = File::create(parameter_file_path).expect("cannot create file");
            writeln!(params_file, "{}", params).expect("cannot write to file");
        }

        Self {
            file,
        }
    }

    pub fn write_fitness<T: Clone>(&mut self, iteration_number: usize, runner: &ProgramState<T>, best_mae: Option<f32>, stagnant_evals: usize, is_restart: bool) {
        let parent = get_best_parent_chromosome(&runner);
        let active_rate = parent.active_mutation_rate; // Log the active mutation rate of the currently best individual
        let inactive_rate = parent.inactive_mutation_rate;

        let mean_active_rate: f32 = runner.population.iter()
            .map(|c| c.active_mutation_rate)
            .sum::<f32>() / (runner.population.len() as f32);

        let mean_inactive_rate: f32 = runner.population.iter()
            .map(|c| c.inactive_mutation_rate)
            .sum::<f32>() / (runner.population.len() as f32);

        if let Some(mae) = best_mae {
            writeln!(self.file, "{},{},{},{},{},{},{},{},{},{},{}", 
                iteration_number, runner.total_evaluations, runner.get_best_fitness(), mae, runner.diversity, stagnant_evals, active_rate, inactive_rate, mean_active_rate, mean_inactive_rate, if is_restart {1} else {0}).expect("cannot write");
        } else {
            writeln!(self.file, "{},{},{},{},{},{},{},{},{},{}", 
                iteration_number, runner.total_evaluations, runner.get_best_fitness(), runner.diversity, stagnant_evals, active_rate, inactive_rate, mean_active_rate, mean_inactive_rate, if is_restart {1} else {0}).expect("cannot write");
        }
    }

    pub fn write_finished_fitness(&mut self, iteration_number: usize, fitness_train_lifetime: Option<f32>, fitness_train_current: Option<f32>, fitness_test: Option<f32>) {
        writeln!(self.file, "End at iteration: {}", iteration_number).expect("cannot write");

        if let Some(f_life) = fitness_train_lifetime {
            writeln!(self.file, "Lifetime Best Fitness Train: {}", f_life).expect("cannot write");
        }
        if let Some(f_curr) = fitness_train_current {
            writeln!(self.file, "Final Cycle Best Fitness Train: {}", f_curr).expect("cannot write");
        }
        if let Some(f_test) = fitness_test {
            writeln!(self.file, "Fitness Eval: {}", f_test).expect("cannot write");
        }
    }
}

impl LoggerArchive {
    pub fn new(args: &Cli) -> Self {
        let save_path = Path::new("")
            .join(format!("Experiments_Output_{}", args.dataset_args.dataset_type))
            .join(format!("cgp_extension_type_{}", args.cgp_extension_type))
            .join(format!("dataset_{}", args.dataset_args.dataset))
            .join(format!("mutation_type_{}", args.mutation_args.mutation_type))
            .join(format!("selection_type_{}", args.selection_args.selection_type))
            .join(format!("crossover_type_{}", args.crossover_args.crossover_type))
            .join(format!("adaptation_type_{}", args.adaptation_args.adaptation_type))
            .join(format!("restart_type_{}", args.restart_args.restart_type))
            .join(format!("run_id_{}", args.run_id));

        fs::create_dir_all(save_path.clone()).unwrap();
        
        let save_file_iteration = "archive.jsonl".to_string();
        let file = BufWriter::new(File::create(save_path.join(save_file_iteration))
            .expect("cannot create file"));

        Self { file }
    }

pub fn write_best_individual(&mut self, iteration_number: usize, fitness: f32, chromosome: &Chromosome) {
    let record = json!({
        "iteration": iteration_number,
        "fitness": fitness,
        "chromosome": chromosome
    });

    writeln!(self.file, "{}", record).expect("cannot write to archive");
}
}


impl LoggerRestart {
    pub fn new(args: &Cli) -> Self {
        
        let save_path = Path::new("")
            .join(format!("Experiments_Output_{}", args.dataset_args.dataset_type))
            .join(format!("cgp_extension_type_{}", args.cgp_extension_type))
            .join(format!("dataset_{}", args.dataset_args.dataset))
            .join(format!("mutation_type_{}", args.mutation_args.mutation_type))
            .join(format!("selection_type_{}", args.selection_args.selection_type))
            .join(format!("crossover_type_{}", args.crossover_args.crossover_type))
            .join(format!("adaptation_type_{}", args.adaptation_args.adaptation_type))
            .join(format!("restart_type_{}", args.restart_args.restart_type))
            .join(format!("run_id_{}", args.run_id));

        fs::create_dir_all(save_path.clone()).unwrap();
        let file_path = save_path.join("restarts_summary.csv");
        let is_new = !file_path.exists();
        let mut file = BufWriter::new(File::create(file_path).expect("cannot create file"));

        if is_new {
            writeln!(file, "RestartNum,Evaluations,CycleBestBefore,FitnessAtRestart,LifetimeBestBefore,PostRestartInitialFitness,PreRestartActiveRate,PostRestartActiveRate,IsProductive,DiversityAtRestart").expect("cannot write header");
        }

        Self { file }
    }

    pub fn write_restart_event(&mut self, restart_num: usize, evaluations: usize, cycle_best_before: f32, fitness_at_restart: f32, lifetime_best_before: f32, post_restart_initial_fitness: f32, pre_restart_active_rate: f32, post_restart_active_rate: f32, is_productive: bool, diversity: f32)
    {
        writeln!(self.file, "{},{},{},{},{},{},{},{},{},{}", restart_num, evaluations, cycle_best_before, fitness_at_restart, lifetime_best_before, post_restart_initial_fitness, pre_restart_active_rate, post_restart_active_rate, if is_productive {1} else {0}, diversity).expect("cannot write restart event");
    }
}