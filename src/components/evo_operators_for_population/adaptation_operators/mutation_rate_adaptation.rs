use rand_distr::{Normal, Distribution};
use crate::{components::evo_operators_for_population::adaptation_operators::adaptation_trait::GeneralAdaptation, utils::runner::ProgramState};

// Based on formula 3 in DOI 10.1007/3-540-61286-6_141

// Baseline: No adaptation (leaves mutation rates unchanged)
pub struct NoAdaptation;

impl<T: Clone> GeneralAdaptation<T> for NoAdaptation {
    fn new() -> Box<dyn GeneralAdaptation<T>> where Self: Sized {
        Box::new(Self)
    }
    fn execute(&mut self, _runner: &mut ProgramState<T>) {
        // No-op: mutation rates remain static
    }
}

// Coupled adaptation (active rate adapts, inactive rate is coupled via active_inactive_ratio)
pub struct BaeckCoupled;

impl<T: Clone> GeneralAdaptation<T> for BaeckCoupled {
    fn new() -> Box<dyn GeneralAdaptation<T>> where Self: Sized {
        Box::new(Self)
    }
    fn execute(&mut self, runner: &mut ProgramState<T>) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for &id in &runner.child_ids {
            let chromosome = &mut runner.population[id];
            let p = chromosome.active_mutation_rate;
            let lr = chromosome.params.learning_rate;

            let new_active = (1.0 + ((1.0 - p) / p) * (-lr * normal.sample(&mut rng)).exp()).powi(-1).clamp(0.005, 0.05);
            chromosome.active_mutation_rate = new_active;
            chromosome.inactive_mutation_rate = (new_active * chromosome.params.active_inactive_ratio).clamp(0.0, 1.0);
        }
    }
}

// Active rate adapts, inactive mutation rate remains static
pub struct BaeckStaticInactive;

impl<T: Clone> GeneralAdaptation<T> for BaeckStaticInactive {
    fn new() -> Box<dyn GeneralAdaptation<T>> where Self: Sized {
        Box::new(Self)
    }
    fn execute(&mut self, runner: &mut ProgramState<T>) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for &id in &runner.child_ids {
            let chromosome = &mut runner.population[id];
            let p = chromosome.active_mutation_rate;
            let lr = chromosome.params.learning_rate;

            chromosome.active_mutation_rate = (1.0 + ((1.0 - p) / p) * (-lr * normal.sample(&mut rng)).exp()).powi(-1).clamp(0.005, 0.05);
            // inactive_mutation_rate is kept static
        }
    }
}

// Both active and inactive mutation rates adapt independently using Baeck formula
pub struct BaeckBothAdaptive;

impl<T: Clone> GeneralAdaptation<T> for BaeckBothAdaptive {
    fn new() -> Box<dyn GeneralAdaptation<T>> where Self: Sized {
        Box::new(Self)
    }
    fn execute(&mut self, runner: &mut ProgramState<T>) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for &id in &runner.child_ids {
            let chromosome = &mut runner.population[id];
            let lr = chromosome.params.learning_rate;

            let p_active = chromosome.active_mutation_rate;
            chromosome.active_mutation_rate = (1.0 + ((1.0 - p_active) / p_active) * (-lr * normal.sample(&mut rng)).exp()).powi(-1).clamp(0.005, 0.05);

            let p_inactive = chromosome.inactive_mutation_rate;
            chromosome.inactive_mutation_rate = (1.0 + ((1.0 - p_inactive) / p_inactive) * (-lr * normal.sample(&mut rng)).exp()).powi(-1).clamp(0.005, 0.5);
        }
    }
}