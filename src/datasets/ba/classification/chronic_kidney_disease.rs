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
        if parts.len() < 25 {
            continue;
        }

        // Target label is the 25th column (index 24)
        let label_str = parts[24].trim().trim_matches('\'').trim();
        let label: f32 = if label_str == "ckd" { 0.0 } else { 1.0 };
        labels.push(label);

        // First 24 columns are features
        let mut converted_data: Vec<f32> = Vec::with_capacity(24);
        for &val in parts[..24].iter() {
            let val_str = val.trim().trim_matches('\'').trim();
            if val_str == "?" || val_str.is_empty() {
                converted_data.push(-1.0);
            } else if val_str == "normal" || val_str == "notpresent" || val_str == "no" || val_str == "good" {
                converted_data.push(0.0);
            } else if val_str == "abnormal" || val_str == "present" || val_str == "yes" || val_str == "poor" {
                converted_data.push(1.0);
            } else {
                converted_data.push(val_str.parse::<f32>().unwrap_or(-1.0));
            }
        }
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
