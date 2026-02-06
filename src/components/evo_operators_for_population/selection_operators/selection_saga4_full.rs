use crate::components::evo_operators_for_population::selection_operators::selection_trait::PopulationGeneralSelection;
use crate::utils::runner::ProgramState;
use crate::utils::utility_funcs::get_median_from_sorted;

pub struct PopulationSelectionSAGA4Full;

/// Based on approach taken by Heider et al. in https://doi.org/10.1109/CEC60901.2024.10612101
/// Population is adapted by assigning an age attribute to each chromosome with the following rules:
/// * All chromosomes start with an age of 3
/// * The age of a chromosome gets reduced by one with each generation
/// * A chromosome whose age drops to 0 is eliminated
/// * All individuals above median fitness receive an extra generation to live
/// * Should the population become too large (10 times the original size) all individuals below a dynamic fitness threshold lose two generations
/// All surviving individuals are made parents
impl<T: Clone> PopulationGeneralSelection<T> for PopulationSelectionSAGA4Full {
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
        let child_ids: Vec<usize> = (current_pop_size..current_pop_size * 2).collect();
        runner.child_ids = child_ids;
    }
}