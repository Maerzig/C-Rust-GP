use std::cmp::Ordering;

use crate::components::evo_operators_for_population::selection_operators::selection_trait::PopulationGeneralSelection;
use crate::utils::runner::ProgramState;
use crate::utils::utility_funcs::vect_difference;

pub struct PopulationSelectionMuCommaLambda;

/// Standard μ,λ selection (comma selection): parents of the next generation are chosen strictly
/// from the children (offspring) of the current generation.
impl<T: Clone> PopulationGeneralSelection<T> for PopulationSelectionMuCommaLambda {
    fn new() -> Box<dyn PopulationGeneralSelection<T>> where Self: Sized {
        Box::new(Self)
    }

    fn execute(&self, runner: &mut ProgramState<T>) {
        // Collect child candidates and their fitness
        let mut child_candidates: Vec<(usize, f32)> = runner.child_ids.iter()
            .map(|&id| (id, runner.fitness_vals[id]))
            .collect();

        // Sort by fitness ascending (lowest fitness value is best)
        child_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Select the mu (elitists) best children as the new parents
        let mut new_parent_ids: Vec<usize> = Vec::with_capacity(runner.params.elitists);
        for (child_id, _) in child_candidates.iter().take(runner.params.elitists) {
            new_parent_ids.push(*child_id);
        }

        assert_eq!(runner.params.elitists, new_parent_ids.len());
        runner.elitist_ids = new_parent_ids;

        let child_ids: Vec<usize> = (0..runner.params.elitists + runner.params.population_size).collect();
        let child_ids = vect_difference(&child_ids, &runner.elitist_ids);
        runner.child_ids = child_ids;
    }
}


