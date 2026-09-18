use std::rc::Rc;

use crate::{components::cgp_components::{chromosome_evaluator_operators::ChromosomeEvaluation, chromosome_find_active_node_operators::ChromosomeActiveNode}, function_set::function_trait::Function, utils::runner::ProgramState};

pub trait GeneralRestart<T>
{
    fn new() -> Box<dyn GeneralRestart<T>> where Self: Sized;

    fn execute(&self, 
        runner: &mut ProgramState<T>,
        function_set: Rc<Vec<Box<dyn Function<T>>>>, 
        evaluator: Rc<Box<dyn ChromosomeEvaluation<T>>>,
        active_node_func: Rc<Box<dyn ChromosomeActiveNode<T>>>);
}