use std::cmp::Ordering;

use crate::components::evo_operators_for_population::selection_operators::selection_trait::PopulationGeneralSelection;
use crate::utils::runner::ProgramState;
use crate::utils::utility_funcs::vect_difference;

pub struct PopulationElitistSelectionMuCommaLambda;

/// μ,λ selection with elitists chosen from the parents of the old generation
impl<T: Clone> PopulationGeneralSelection<T> for PopulationElitistSelectionMuCommaLambda {
    fn new() -> Box<dyn PopulationGeneralSelection<T>> where Self: Sized {
        Box::new(Self)
    }

    fn execute(&self, runner: &mut ProgramState<T>) {
        let mut sorted_fitness_vals: Vec<f32> = runner.fitness_vals_sorted.clone();
        sorted_fitness_vals.dedup();
        
        // Find the ids of the elitists from within the parents of the old generation
        let mut old_parents_sorted = runner.elitist_ids.clone();
        old_parents_sorted.sort_by(|&id1, &id2| {
            let fitness1 = runner.fitness_vals[id1];
            let fitness2 = runner.fitness_vals[id2];
            fitness1.partial_cmp(&fitness2).unwrap_or(Ordering::Equal)
        });
        

        let mut new_parent_ids: Vec<usize> = Vec::with_capacity(runner.params.elitists);
        new_parent_ids.extend(old_parents_sorted.iter().take(runner.params.parent_elitists));

        // Fill the rest with ids of the best performing _children_ of the last generation
        let mut child_candidates: Vec<(usize, f32)> = runner.child_ids.iter()
            .map(|&id| (id, runner.fitness_vals[id]))
            .collect();

        child_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let needed = runner.params.elitists - new_parent_ids.len();
        for (child_id, _) in child_candidates.iter().take(needed) {
            new_parent_ids.push(*child_id);
        }
    
        assert_eq!(runner.elitist_ids.len(), new_parent_ids.len());
        runner.elitist_ids = new_parent_ids;

        let child_ids: Vec<usize> = (0..runner.params.elitists + runner.params.population_size).collect();
        let child_ids = vect_difference(&child_ids, &runner.elitist_ids);
        runner.child_ids = child_ids;
    }
}