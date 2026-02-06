use std::fmt::{Display, Formatter};
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub enum SelectionTypes {
    OnePlusFour,
    MuPlusLambda,
    MuCommaLambda,
    Tournament,
}

impl Display for SelectionTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionTypes::OnePlusFour => write!(f, "OnePlusFour"),
            SelectionTypes::MuPlusLambda => write!(f, "MuPlusLambda"),
            SelectionTypes::MuCommaLambda => write!(f, "MuCommaLambda"),
            SelectionTypes::Tournament => write!(f, "Tournament"),
        }
    }
}