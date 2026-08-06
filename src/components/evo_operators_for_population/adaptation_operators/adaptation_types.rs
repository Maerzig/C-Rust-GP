use std::fmt::{Display, Formatter};
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub enum AdaptationTypes {
        ActiveRateAdaptationBaeck,
}

impl Display for AdaptationTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AdaptationTypes::ActiveRateAdaptationBaeck => write!(f, "Baeck active mutation rate adaptation"),
        }
    }
}