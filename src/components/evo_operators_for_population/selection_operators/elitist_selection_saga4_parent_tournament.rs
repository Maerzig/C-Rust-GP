use std::cmp::min;

use rand::seq::IteratorRandom;

use crate::components::evo_operators_for_population::selection_operators::selection_trait::PopulationGeneralSelection;
use crate::utils::runner::ProgramState;
use crate::utils::utility_funcs::get_median_from_sorted;

pub struct PopulationSelectionSAGA4ParentTournament;

/// Based on approach taken by Heider et al. in https://doi.org/10.1109/CEC60901.2024.10612101
/// Population is adapted by assigning an age attribute to each chromosome with the following rules:
/// * All chromosomes start with an age of 3
/// * The age of a chromosome gets reduced by one with each generation
/// * A chromosome whose age drops to 0 is eliminated
/// * All individuals above median fitness receive an extra generation to live
/// * Should the population become too large (10 times the original size) all individuals below a dynamic fitness threshold lose two generations
/// Only a fixed amount of individuals of each generation are made parents; they are decided by parent tournament selection + elitism
/// NEEDS the full clone option from clone_parent_to_child.rs
impl<T: Clone> PopulationGeneralSelection<T> for PopulationSelectionSAGA4ParentTournament {
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
        let mut selection = vec![];
        let mut keep_mask: Vec<bool> = vec![false; current_pop_size];

        // Make sure the best parent_elitists individuals are carried over as elitists
        let mut indexed_fitness: Vec<(usize, f32)> = runner.fitness_vals.iter()
            .enumerate()
            .map(|(i, &f)| (i,f))
            .collect();
        indexed_fitness.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        let nmbr_elitists = min(runner.params.parent_elitists, current_pop_size);

        for i in 0..nmbr_elitists {
            let (elitist_id, _) = indexed_fitness[i];
            selection.push(elitist_id);
        }

        // Parent Tournament selection
        // Then make a mask and eliminate all individuals that aren't parents by swapping the parents to the front and then truncating
        // I believe this is a relatively efficient way to go about this since it doesn't involve a lot of reallocating memory or copying individuals
        let tournament_candidate_indices: Vec<usize> = (0..current_pop_size)
            .filter(|&i| !selection.contains(&i))
            .collect();

        if !tournament_candidate_indices.is_empty() {
            for _ in 0..current_pop_size - nmbr_elitists {
                let winner_id = tournament_candidate_indices
                    .iter()
                    .choose_multiple(&mut rng, runner.params.tournament_size)
                    .into_iter()
                    .min_by(|&&a, &&b| {
                        runner.fitness_vals[a].partial_cmp(&runner.fitness_vals[b]).unwrap()
                    })
                    .map(|&id| id)
                    .unwrap();

                selection.push(winner_id);
            }
        }
        
        for i in selection {
            keep_mask[i] = true;
        }

        let mut write_index = 0;

        for read_index in 0..current_pop_size {
            if keep_mask[read_index] {
                if read_index != write_index {
                    runner.population.swap(read_index, write_index);
                    runner.fitness_vals.swap(read_index, write_index);
                }
                write_index+=1;
            }
        }

        runner.population.truncate(write_index);
        runner.fitness_vals.truncate(write_index);

        current_pop_size = runner.population.len();
        let child_ids: Vec<usize> = (current_pop_size..current_pop_size * 2).collect();
        runner.child_ids = child_ids;
    }
}