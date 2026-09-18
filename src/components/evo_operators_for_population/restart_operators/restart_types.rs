use std::fmt::{Display, Formatter};
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub enum RestartTypes {
        None,
        PhenotypicDiversityRestart,
}

impl Display for RestartTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RestartTypes::None => write!(f, "No restart (baseline)"),
            RestartTypes::PhenotypicDiversityRestart => write!(f, "Phenotypic diversity restart"),
        }
    }
}