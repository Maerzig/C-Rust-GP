use std::fs;
use crate::datasets::real_world_uci::dataset_utils::{shuffle, standardize_dataset};

pub fn get_dataset(
    dataset_path: String,
    shuffle_and_split: bool,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let contents = fs::read_to_string(dataset_path)
        .expect("Should have been able to read the file");

    let mut lines = contents.lines();

    // Skip ARFF header until after @data
    for line in lines.by_ref() {
        if line.trim().to_lowercase().starts_with("@data") {
            break;
        }
    }

    let mut datas: Vec<Vec<f32>> = vec![];
    let mut labels: Vec<f32> = vec![];

    for line in lines {
        let line_str = line.trim();
        if line_str.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line_str.split(',').collect();
        if parts.len() < 563 {
            continue;
        }

        // Target label is the 563rd column (index 562)
        let label_str = parts[562].trim().trim_matches('\'').trim();
        let label: f32 = label_str.parse::<f32>().unwrap() - 1.0;
        labels.push(label);

        // First 562 columns are numeric features
        let converted_data: Vec<f32> = parts[..562]
            .iter()
            .map(|val| val.trim().parse::<f32>().unwrap())
            .collect();
        datas.push(converted_data);
    }

    let datas = standardize_dataset(datas);

    if shuffle_and_split {
        let (datas, labels) = shuffle(datas, labels);
        let total_len = datas.len();
        let split_idx = (total_len as f32 * 0.8) as usize;

        let train_data: Vec<Vec<f32>> = datas[0..split_idx].to_vec();
        let train_label: Vec<f32> = labels[0..split_idx].to_vec();
        let test_data: Vec<Vec<f32>> = datas[split_idx..total_len].to_vec();
        let test_label: Vec<f32> = labels[split_idx..total_len].to_vec();

        return (train_data, train_label, test_data, test_label);
    }

    (datas, labels, vec![], vec![])
}
