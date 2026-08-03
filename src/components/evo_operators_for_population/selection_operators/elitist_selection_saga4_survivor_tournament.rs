use core::f32;
use std::cmp::min;
use rand::seq::IteratorRandom;
use crate::components::evo_operators_for_population::selection_operators::selection_trait::PopulationGeneralSelection;
use crate::utils::runner::ProgramState;
use crate::utils::utility_funcs::get_median_from_sorted;

pub struct PopulationSelectionSAGA4SurvivorTournament;
/// Based on approach taken by Smętek et al. in 10.1007/978-3-642-21219-2_16
/// Population is adapted by assigning an age attribute to each chromosome with the following rules:
/// * All chromosomes start with an age of 3
/// * The age of a chromosome gets reduced by one with each generation
/// * A chromosome whose age drops to 0 is eliminated
/// * All individuals above median fitness receive an extra generation to live
/// * Should the population become too large (10 times the original size) all individuals below a dynamic fitness threshold lose two generations
/// Only a fixed amount of individuals of each generation survive; they are decided by survivor tournament selection + elitism
impl<T: Clone> PopulationGeneralSelection<T> for PopulationSelectionSAGA4SurvivorTournament {
    fn new() -> Box<dyn PopulationGeneralSelection<T>> where Self: Sized {
        Box::new(Self)
    }

    fn execute(&self, runner: &mut ProgramState<T>) {
        let median_fitness = get_median_from_sorted(&runner.fitness_vals_sorted).unwrap();
        let mut current_pop_size= runner.population.len();
        let pop_size_max_exceeded = current_pop_size > runner.params.population_size * 10;
        let pop_control_fitness_threshold: f32;

        if pop_size_max_exceeded {
            pop_control_fitness_threshold = runner.fitness_vals_sorted[runner.params.population_size * 10];
        } else {
            pop_control_fitness_threshold = *runner.fitness_vals_sorted.last().unwrap();
        }

        //Run through the population backwards to ensure removing individuals doesn't throw off the ID of individuals that haven't been processed yet
        for id in (0..current_pop_size).rev() {
            let current_fitness_val = runner.fitness_vals[id];
                if current_fitness_val < median_fitness {
                runner.population[id].age += 1;
            }

            if pop_size_max_exceeded && current_fitness_val > pop_control_fitness_threshold {
                runner.population[id].age -= 2;
            }

            runner.population[id].age -= 1;

            if runner.population[id].age <= 0 {
                runner.population.swap_remove(id);
                runner.fitness_vals.swap_remove(id);
            }
        }

        current_pop_size = runner.population.len();
        let mut rng = rand::thread_rng();
        let parent_count = min(runner.params.population_size + runner.params.elitists, current_pop_size); // Runner.params.population_size = initial pop size
        let mut parent_ids: Vec<usize> = Vec::with_capacity(parent_count);

        // Choose elitists
        // Make sure the best individuals are carried over as elitists
        let mut indexed_fitness: Vec<(usize, f32)> = runner.fitness_vals.iter()
            .enumerate()
            .map(|(i, &f)| (i,f))
            .collect();
        indexed_fitness.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        let nmbr_elitists = min(runner.params.elitists, current_pop_size);

        for i in 0..nmbr_elitists {
            let (elitist_id, _) = indexed_fitness[i];
            parent_ids.push(elitist_id);
        }

        // Survivor Tournament selection
        let mut candidates: Vec<usize> = (0..current_pop_size)
            .filter(|&i| !parent_ids.contains(&i))
            .collect();

        // Tournament selection of parents
        for _ in 0..parent_count - nmbr_elitists {
            if candidates.is_empty() {
                break;
            }

            let t_size = min(runner.params.tournament_size, candidates.len());

            let winner_idx = (0..candidates.len())
                .choose_multiple(&mut rng, t_size)
                .into_iter()
                .min_by(|&a, &b| {
                    runner.fitness_vals[candidates[a]]
                        .partial_cmp(&runner.fitness_vals[candidates[b]])
                        .unwrap()
                })
                .unwrap();

            let winner_id = candidates.swap_remove(winner_idx);
            parent_ids.push(winner_id);
        }

        runner.elitist_ids = parent_ids;
        let child_ids: Vec<usize> = (current_pop_size..current_pop_size + runner.elitist_ids.len()).collect();
        runner.child_ids = child_ids;
    }
}