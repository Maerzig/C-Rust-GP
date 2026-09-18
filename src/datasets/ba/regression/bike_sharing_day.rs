use std::fs;
use crate::datasets::real_world_uci::dataset_utils::{shuffle, standardize_dataset, standardize_labels};

fn parse_date_to_days(date_str: &str) -> f32 {
    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() == 3 {
        let yr: f32 = parts[0].parse().unwrap_or(2011.0);
        let mnth: f32 = parts[1].parse().unwrap_or(1.0);
        let day: f32 = parts[2].parse().unwrap_or(1.0);
        (yr - 2011.0) * 365.25 + (mnth - 1.0) * 30.4375 + (day - 1.0)
    } else {
        0.0
    }
}

pub fn get_dataset(
    dataset_path: String, 
    shuffle_and_split: bool
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let contents = fs::read_to_string(dataset_path)
        .expect("Should have been able to read the file");
    
    let mut lines = contents.lines();
    
    // Skip the header row
    lines.next();

    let mut datas: Vec<Vec<f32>> = vec![];
    let mut labels: Vec<f32> = vec![];
    
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }

        let mut parts: Vec<&str> = line.split(',').collect();
        
        // Pop the last element as the target regression value
        let label: f32 = parts.pop().unwrap().trim().parse::<f32>().unwrap();
        labels.push(label);

        // Skip index 0 (instant row ID column) and last 2 columns (casual, registered) to prevent target leakage (cnt = casual + registered)
        let parts = &parts[1..parts.len() - 2];

        let converted_data: Vec<f32> = parts.iter().enumerate().map(|(i, val)| {
            let val_str = val.trim();
            if i == 0 {
                // dteday at index 0 after skipping instant
                parse_date_to_days(val_str)
            } else {
                val_str.parse::<f32>().unwrap()
            }
        }).collect();
        datas.push(converted_data);
    }

    // Standardize the features and labels
    let datas = standardize_dataset(datas);
    let labels = standardize_labels(&labels);

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
