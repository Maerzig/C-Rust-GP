use clap::{Args, Parser};

#[derive(Parser)]
#[clap(author, version, about, name = "User Args")]
pub struct Cli {
    #[arg(long,
        default_value = "Standard",
        help = "Default options include:
        - Standard,
        - OriginalReorder,
        - EReorder,
        - LSDReorder,
        - NegBiasReorder,
        - UniformReorder,
        - DAG
        ")]
    pub cgp_extension_type: String,

    #[arg(long, default_value_t = 50)]
    pub nbr_nodes: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Helper value to differentiate multiple runs of the same configuration."
    )]
    pub run_id: usize,

    // #[arg(
    //     long,
    //     default_value = "",
    //     help = "Optional value. If the configuration is derived from a json file, include the path here."
    // )]
    // pub json_file_path: String,

    #[command(flatten)]
    pub dataset_args: DatasetArgs,

    #[command(flatten)]
    pub mutation_args: MutationArgs,

    #[command(flatten)]
    pub crossover_args: CrossoverArgs,

    #[command(flatten)]
    pub selection_args: SelectionArgs,

    #[command(flatten)]
    pub adaptation_args: AdaptationArgs,

    #[command(flatten)]
    pub restart_args: RestartArgs,
}

#[derive(Args)]
#[group(required = false)]
pub struct DatasetArgs {
    #[arg(long,
        default_value = "f32",
        help = "Default options: f32 or bool")]
    pub dataset_type: String,

    #[arg(long,
        default_value = "regression",
        help = "Default options: regression or classification")]
    pub problem_type: String,

    #[arg(long,
        default_value = "abalone",
        help = "Default options include:
        - Parity,
        - Encode,
        - Decode,
        - Multiply,
        - Keijzer6,
        - Koza3,
        - Nguyen7,
        - Pagie1
        ")]
    pub dataset: String,
}

#[derive(Args)]
#[group(required = false)]
pub struct MutationArgs {
    #[arg(long,
        default_value = "Single",
        help = "Default options include:
        - Point,
        - Single,
        - Split,
        - SplitAdaptive,
        - Multi,
        ")]
    pub mutation_type: String,

    #[arg(long, default_value_t = 0)]
    pub bioma_mutation_multi_n: usize,

    #[arg(long, default_value_t = 0.0)]
    pub bioma_mutation_prob_active: f32,

    #[arg(long, default_value_t = 0.0)]
    pub bioma_mutation_prob_inactive: f32,

    #[arg(long, default_value_t = 0.0)]
    pub bioma_mutation_rate: f32,

    #[arg(long, default_value_t = 0.025)]
    pub starting_mutation_rate: f32,

    #[arg(long, default_value_t = 5.0)]
    pub active_inactive_ratio: f32,
}


#[derive(Args)]
#[group(required = false)]
pub struct CrossoverArgs {
    #[arg(long, default_value = "Uniform",
        help = "Default options include:
        - 1-Point,
        - n-Point,
        - Uniform,
        - NoCrossover,
        In case of n-Point crossover: n must be defined with variable multi-point-n
        ")]
    pub crossover_type: String,

    #[arg(long, default_value_t = 0.9)]
    pub crossover_rate: f32,

    // for n-point crossover
    #[arg(long, default_value_t = 0)]
    pub multi_point_n: usize,
}

#[derive(Args)]
#[group(required = false)]
pub struct SelectionArgs {
    #[arg(long, default_value = "OnePlusFour",
        help = "Default options include:
        - OnePlusFour,
        - MuPlusLambda,
        - MuCommaLambda (standard comma selection),
        - ElitistMuCommaLambda (including parent elitists),
        - Tournament (includes elitists; tournament draws from both elitists and normal population),
        - SAGA4Full (selection based on age alone, all survivors are parents),
        - SAGA4Random (random selection),
        - SAGA4ParentTournament (parent tournament selection),
        - SAGA4SurvivorTournament (survivor tournament selection)
        ")]
    pub selection_type: String,

    #[arg(long, default_value_t = 8, help = "Only relevant if selection_type=Tournament")]
    pub tournament_size: usize,

    #[arg(long, default_value_t = 1, help = "Relevant for MuPlusLambda and Tournament Selection.
    Case MuPlusLambda: Mu==Elitism-Number
    Case Tournament: Number of elitists -> Total Population = Elitism-Number + Population-size
    ")]
    pub elitism_number: usize,

    #[arg(long, default_value_t = 3, help = "Relevant for ElitistMuCommaLambda and the elitist SAGA variants
    Decides how many of the parents of the old generations are carried over as elitists.
    ")]
    pub parent_elitists: usize,

    #[arg(long, default_value_t = 4, help = "Relevant for MuPlusLambda and Tournament Selection.
    Case MuPlusLambda: Lambda==Population-size
    Case Tournament: Population size -> Total Population = Elitism-Number + Population-size
    ")]
    pub population_size: usize,

}
#[derive(Args)]
#[group(required = false)]
pub struct AdaptationArgs {
    #[arg(long, default_value = "BaeckCoupled", help = "Default options include:
        - None,
        - BaeckCoupled,
        - BaeckStaticInactive,
        - BaeckBothAdaptive")]
    pub adaptation_type: String,
    #[arg(long, default_value_t = 0.05, help = "Learning rate for self-adaptation with Baeck formula")]
    pub learning_rate: f32,
}

#[derive(Args)]
#[group(required = false)]
pub struct RestartArgs {
    #[arg(long, default_value = "Phenotype", help = "Default options include:
        - None,
        - Phenotype")]
    pub restart_type: String,
    #[arg(long, default_value_t = 3, help = "How many elitists get saved to the archive after a restart")]
    pub archive_elitists: usize,
    // Decide a good default value here
    #[arg(long, default_value_t = 0.1, help = "Minimum threshold of phenotype diversity (see DOI 10.1109/SSCI.2015.201) before a restart is triggered")]
    pub diversity_threshold: f32,
    #[arg(long, default_value_t = false, help = "Whether to keep best individuals in the population after a reset (true) or just save them to the archive (false)")]
    pub keep_elitists: bool,
    #[arg(long, default_value_t = 1.0, help = "Amplifier for the frequency rate in the calculation of healthy phenotype diversity (see DOI 10.1109/SSCI.2015.201 for more information)")]
    pub phenotype_diversity_amplifier: f32,
    #[arg(long, default_value_t = 2.0, help = "Percentage of evaluation budget without improvement before triggering a restart check")]
    pub max_stagnant_evals_pct: f32,
}

pub fn get_arguments() -> Cli {
    

    Cli::parse()
}
