use crate::datasets::boolean_datasets;
use crate::datasets::real_world_uci;
use crate::utils::cli_functions::DatasetArgs;

/// Loads a dataset using DatasetArgs.
pub fn load_dataset(dataset_args: &DatasetArgs) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
    let dataset_lc = dataset_args.dataset.to_lowercase();
    match dataset_lc.as_str() {
        "apnea" | "apnea2" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::apnea2::get_dataset(
                "src/datasets/ba/regression/data/556_analcatdata_apnea2.tsv".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "forestfire" | "forestfires" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::forestfires::get_dataset(
                "src/datasets/ba/regression/data/forestfires.csv".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "bodyfat" | "bodyfat2" | "560_bodyfat" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::bodyfat::get_dataset(
                "src/datasets/ba/regression/data/560_bodyfat.tsv".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "monk2" | "monk_2" | "560_monk2" => {
            let (data, labels_usize, _, _) = crate::datasets::ba::classification::monk2::get_dataset(
                "src/datasets/ba/classification/data/monk2.tsv".to_string(),
                false,
            );
            let labels: Vec<f32> = labels_usize.into_iter().map(|x| x as f32).collect();
            Ok((data, labels))
        }
        "adult" => {
            let (data, labels, _, _) = crate::datasets::ba::classification::adult::get_dataset(
                "src/datasets/ba/classification/data/adult.data".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "bach" => {
            let (data, labels, _, _) = crate::datasets::ba::classification::bach::get_dataset(
                "src/datasets/ba/classification/data/bach.arff".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "car" => {
            let (data, labels, _, _) = crate::datasets::ba::classification::car::get_dataset(
                "src/datasets/ba/classification/data/car.data".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "chronic_kidney" | "chronic_kidney_disease" | "chronic_kidney_disease_full" => {
            let (data, labels, _, _) = crate::datasets::ba::classification::chronic_kidney_disease::get_dataset(
                "src/datasets/ba/classification/data/chronic_kidney_disease_full.arff".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "forest" => {
            let (data, labels, _, _) = crate::datasets::ba::classification::forest::get_dataset(
                "src/datasets/ba/classification/data/forest.arff".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "human" => {
            let (data, labels, _, _) = crate::datasets::ba::classification::human::get_dataset(
                "src/datasets/ba/classification/data/human.arff".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "iris" => {
            let (data, labels, _, _) = crate::datasets::ba::classification::iris::get_dataset(
                "src/datasets/ba/classification/data/iris.arff".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "wall24" => {
            let (data, labels, _, _) = crate::datasets::ba::classification::wall24::get_dataset(
                "src/datasets/ba/classification/data/wall24.arff".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "wine_quality" => {
            let (data, labels, _, _) = crate::datasets::ba::classification::wine_quality::get_dataset(
                "src/datasets/ba/classification/data/wine_quality.arff".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "abalone" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::abalone::get_dataset(
                "src/datasets/ba/regression/data/abalone.data".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "pollution" | "542_pollution" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::pollution::get_dataset(
                "src/datasets/ba/regression/data/542_pollution.tsv".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "bike_sharing_day" | "bikesharingday" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::bike_sharing_day::get_dataset(
                "src/datasets/ba/regression/data/bike_sharing_day.csv".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "bike_sharing_hour" | "bikesharinghour" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::bike_sharing_hour::get_dataset(
                "src/datasets/ba/regression/data/bike_sharing_hour.csv".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "cal_housing" | "calhousing" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::cal_housing::get_dataset(
                "src/datasets/ba/regression/data/cal_housing.data".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "diabetes" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::diabetes::get_dataset(
                "src/datasets/ba/regression/data/diabetes.tsv".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "energydata" | "energydata_complete" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::energydata::get_dataset(
                "src/datasets/ba/regression/data/energydata_complete.csv".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "winequality_red" | "winequality-red" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::winequality_red::get_dataset(
                "src/datasets/ba/regression/data/winequality-red.csv".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "winequality_white" | "winequality-white" => {
            let (data, labels, _, _) = crate::datasets::ba::regression::winequality_white::get_dataset(
                "src/datasets/ba/regression/data/winequality-white.csv".to_string(),
                false,
            );
            Ok((data, labels))
        }
        "breast_cancer" | "breastcancer" => {
            let (data, labels_usize, _, _) = crate::datasets::ba::classification::breast_cancer::get_dataset(
                "src/datasets/ba/classification/data/breast+cancer+wisconsin+diagnostic.data".to_string(),
                false,
            );
            let labels: Vec<f32> = labels_usize.into_iter().map(|x| x as f32).collect();
            Ok((data, labels))
        }
        "page_blocks" | "pageblocks" => {
            let (data, labels_usize, _, _) = real_world_uci::page_blocks::get_dataset(
                "src/datasets/real_world_uci/data/page-blocks.data".to_string(),
                false,
            );
            let labels = labels_usize.into_iter().map(|x| x as f32).collect();
            Ok((data, labels))
        }
        "pendigits" => {
            let (data, labels_usize, _, _) = real_world_uci::pendigits::get_dataset(
                "src/datasets/real_world_uci/data/dataset_32_pendigits.csv".to_string(),
                false,
            );
            let labels = labels_usize.into_iter().map(|x| x as f32).collect();
            Ok((data, labels))
        }
        "waveform" => {
            let (data, labels_usize, _, _) = real_world_uci::waveform::get_dataset(
                "src/datasets/real_world_uci/data/waveform+database+generator+version+1.data".to_string(),
                false,
            );
            let labels = labels_usize.into_iter().map(|x| x as f32).collect();
            Ok((data, labels))
        }
        "keijzer" | "keijzer6" => {
            let (data, labels_vec) = crate::datasets::ba::regression::keijzer::get_dataset();
            let labels = if !labels_vec.is_empty() { labels_vec[0].clone() } else { vec![] };
            Ok((data, labels))
        }
        "koza3" | "koza_3" => {
            let (data, labels_vec) = crate::datasets::ba::regression::koza_3::get_dataset();
            let labels = if !labels_vec.is_empty() { labels_vec[0].clone() } else { vec![] };
            Ok((data, labels))
        }
        "nguyen7" | "nguyen_7" => {
            let (data, labels_vec) = crate::datasets::ba::regression::nguyen_7::get_dataset();
            let labels = if !labels_vec.is_empty() { labels_vec[0].clone() } else { vec![] };
            Ok((data, labels))
        }
        "pagie1" | "pagie_1" => {
            let (data, labels_vec) = crate::datasets::ba::regression::pagie_1::get_dataset();
            let labels = if !labels_vec.is_empty() { labels_vec[0].clone() } else { vec![] };
            Ok((data, labels))
        }
        "parity" => {
            let (data_b, labels_b) = boolean_datasets::parity::get_dataset();
            let data = data_b.into_iter().map(|row| row.into_iter().map(|b| if b { 1.0 } else { 0.0 }).collect()).collect();
            let labels = labels_b.into_iter().map(|row| if row.first().copied().unwrap_or(false) { 1.0 } else { 0.0 }).collect();
            Ok((data, labels))
        }
        "encode" => {
            let (data_b, labels_b) = boolean_datasets::encode::get_dataset();
            let data = data_b.into_iter().map(|row| row.into_iter().map(|b| if b { 1.0 } else { 0.0 }).collect()).collect();
            let labels = labels_b.into_iter().map(|row| if row.first().copied().unwrap_or(false) { 1.0 } else { 0.0 }).collect();
            Ok((data, labels))
        }
        "decode" => {
            let (data_b, labels_b) = boolean_datasets::decode::get_dataset();
            let data = data_b.into_iter().map(|row| row.into_iter().map(|b| if b { 1.0 } else { 0.0 }).collect()).collect();
            let labels = labels_b.into_iter().map(|row| if row.first().copied().unwrap_or(false) { 1.0 } else { 0.0 }).collect();
            Ok((data, labels))
        }
        "multiply" => {
            let (data_b, labels_b) = boolean_datasets::multiply::get_dataset();
            let data = data_b.into_iter().map(|row| row.into_iter().map(|b| if b { 1.0 } else { 0.0 }).collect()).collect();
            let labels = labels_b.into_iter().map(|row| if row.first().copied().unwrap_or(false) { 1.0 } else { 0.0 }).collect();
            Ok((data, labels))
        }
        _ => {
                Err(format!(
                    "Unknown dataset: '{}'. Supported datasets: apnea, forestfire, abalone, breast_cancer, page_blocks, pendigits, waveform, keijzer6, koza3, nguyen7, pagie1, parity, encode, decode, multiply, or a valid file path (.csv, .tsv, etc.)",
                    dataset_args.dataset)
                )
        }
    }
}

/// Returns the number of target classes for classification datasets.
pub fn get_dataset_num_classes(dataset: &str) -> usize {
    let dataset_lc = dataset.to_lowercase();
    match dataset_lc.as_str() {
        "iris" | "waveform" => 3,
        "breast_cancer" | "breastcancer" | "monk2" | "monk_2" | "560_monk2" | "adult" | "chronic_kidney" | "chronic_kidney_disease" | "chronic_kidney_disease_full" => 2,
        "car" | "wall24" | "forest" => 4,
        "page_blocks" | "pageblocks" => 5,
        "human" => 6,
        "wine_quality" => 7,
        "pendigits" => 10,
        "bach" => 102,
        _ => usize::MIN,
    }
}