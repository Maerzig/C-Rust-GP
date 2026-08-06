use crate::utils::runner::ProgramState;

pub trait GeneralAdaptation<T>
{
    fn new() -> Box<dyn GeneralAdaptation<T>> where Self: Sized;

    fn execute(&mut self, runner: &mut ProgramState<T>);
}