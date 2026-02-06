use std::fmt::{Display, Formatter};
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub enum SelectionTypes {
    OnePlusFour,
    MuPlusLambda,
    MuCommaLambda,
    Tournament,
    SAGA4Random,
    SAGA4ParentTournament,
    SAGA4SurvivorTournament,
    SAGA4Full,
}

impl Display for SelectionTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionTypes::OnePlusFour => write!(f, "OnePlusFour"),
            SelectionTypes::MuPlusLambda => write!(f, "MuPlusLambda"),
            SelectionTypes::MuCommaLambda => write!(f, "MuCommaLambda"),
            SelectionTypes::Tournament => write!(f, "Tournament"),
            SelectionTypes::SAGA4Random => write!(f, "SAGA4Random"),
            SelectionTypes::SAGA4ParentTournament => write!(f, "Saga4ParentTournament"),
            SelectionTypes::SAGA4SurvivorTournament => write!(f, "Saga4SurvivorTournament"),
            SelectionTypes::SAGA4Full => write!(f, "SAGA4Full"),
        }
    }
}