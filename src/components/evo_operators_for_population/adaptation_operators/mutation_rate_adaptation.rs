use rand_distr::{Normal, Distribution};
use crate::{components::evo_operators_for_population::adaptation_operators::adaptation_trait::{self, GeneralAdaptation}, utils::runner::ProgramState};

// Based on formula 3 in DOI 10.1007/3-540-61286-6_141
pub struct ActiveRateAdaptationBaeck;

impl<T: Clone> GeneralAdaptation<T> for ActiveRateAdaptationBaeck {
    fn new() -> Box<dyn GeneralAdaptation<T>> where Self: Sized {
        Box::new(Self)
    }
    fn execute(&mut self, runner: &mut ProgramState<T>) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for &id in &runner.child_ids {
            let chromosome = &mut runner.population[id];
            let p = chromosome. active_mutation_rate;
            let lr = chromosome.params.mutation_learning_rate;

            chromosome.active_mutation_rate = (1.0 + ((1.0 - p) / p) * (-lr * normal.sample(&mut rng)).exp()).powi(-1).clamp(0.005, 0.05)
        }
    }
}