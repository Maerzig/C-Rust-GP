use std::{collections::HashMap, rc::Rc};

use crate::{components::{cgp_components::{chromosome_evaluator_operators::ChromosomeEvaluation, chromosome_find_active_node_operators::ChromosomeActiveNode}, evo_operators_for_population::restart_operators::restart_trait::GeneralRestart}, function_set::function_trait::Function, utils::runner::ProgramState};


// Baseline: No restart operator (never reinitializes population)
pub struct NoRestart;

impl<T: Clone> GeneralRestart<T> for NoRestart {
    fn new() -> Box<dyn GeneralRestart<T>> where Self: Sized {
        Box::new(Self)
    }

    fn execute(&self, 
        _runner: &mut ProgramState<T>,
        _function_set: Rc<Vec<Box<dyn Function<T>>>>, 
        _evaluator: Rc<Box<dyn ChromosomeEvaluation<T>>>,
        _active_node_func: Rc<Box<dyn ChromosomeActiveNode<T>>>) {
            // No-op: population is never restarted
    }
}

// Based on the phenotype diversity introduced in 10.1109/SSCI.2015.201
// Instead of the textual representation Kalkreuth et al. use, this uses a fingerprint of the output nodes to focus on behavioural diversity
// For real-valued data phenotypes get binned to avoid floating point imprecision by masking the last 4 bits of the mantissa 
pub struct PhenotypicDiversityRestart;

impl<T: Clone> GeneralRestart<T> for PhenotypicDiversityRestart {
    fn new() -> Box<dyn GeneralRestart<T>> where Self: Sized {
        Box::new(Self)
    }
    // Check these formulas again to make sure they're implemented correctly
    fn execute(&self, 
        runner: &mut ProgramState<T>,
        function_set: Rc<Vec<Box<dyn Function<T>>>>, 
        evaluator: Rc<Box<dyn ChromosomeEvaluation<T>>>,
        active_node_func: Rc<Box<dyn ChromosomeActiveNode<T>>>) {

            let diversity = Self::compute_diversity(runner);
            if diversity < runner.params.diversity_threshold {
                runner.reinitialize_population(function_set, evaluator, active_node_func);
            }
    }

}

impl PhenotypicDiversityRestart {
    pub fn compute_diversity<T: Clone>(runner: &mut ProgramState<T>) -> f32 {
        let mut counts = HashMap::new();
        for chromosome in &runner.population {
            *counts.entry(chromosome.phenotype_hash).or_insert(0usize) += 1;
        }

        let unique_phenotypes = counts.len() as f32;
        let best_fitness = runner.get_best_fitness();
        let worst_fitness = runner.get_worst_fitness();

        let mut fitness_equal = false;
        if (best_fitness - worst_fitness).abs() < f32::EPSILON {
            fitness_equal = true;
        }

        let mut temp_sum = 0.0;

        for id in 0..runner.population.len() {
            let count = counts.get(&runner.population[id].phenotype_hash).copied().unwrap_or(0) as f32;
            let frequency = (runner.params.phenotype_diversity_amplifier * (count / unique_phenotypes)).min(1.0);
            let fitness_rate: f32;
            if fitness_equal {
                fitness_rate = 1.0;
            } else {
                fitness_rate = (runner.fitness_vals[id] - worst_fitness) / (best_fitness - worst_fitness);
            }

            temp_sum += (1.0 - frequency) * fitness_rate;
        }

        let healthy_phenotype_diversity = (1.0 / runner.population.len() as f32) * temp_sum;
        let standard_phenotype_diversity = unique_phenotypes / runner.population.len() as f32;

        let diversity = (standard_phenotype_diversity + healthy_phenotype_diversity) / 2.0;
        runner.diversity = diversity;
        return diversity;
    }

}