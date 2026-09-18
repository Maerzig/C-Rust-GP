use std::fmt::{Display, Formatter};
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub enum AdaptationTypes {
    None,                // Baseline: No adaptation (static rates)
    BaeckCoupled,        // Method 1: Coupled (inactive rate = active rate * active_inactive_ratio)
    BaeckStaticInactive, // Method 2: Active rate adaptive, inactive rate static
    BaeckBothAdaptive,   // Method 3: Both active and inactive rates adaptive
}

impl Display for AdaptationTypes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AdaptationTypes::None => write!(f, "No adaptation"),
            AdaptationTypes::BaeckCoupled => write!(f, "Baeck coupled mutation rate adaptation"),
            AdaptationTypes::BaeckStaticInactive => write!(f, "Baeck active / static inactive mutation rate adaptation"),
            AdaptationTypes::BaeckBothAdaptive => write!(f, "Baeck both active and inactive mutation rate adaptation"),
        }
    }
}