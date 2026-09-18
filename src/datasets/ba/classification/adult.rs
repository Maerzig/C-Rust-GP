use std::collections::HashMap;
use std::fs;
use crate::datasets::real_world_uci::dataset_utils::{shuffle, standardize_dataset};

pub fn get_dataset(
    dataset_path: String,
    shuffle_and_split: bool,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    let contents = fs::read_to_string(dataset_path)
        .expect("Should have been able to read the file");

    let mut datas: Vec<Vec<f32>> = vec![];
    let mut labels: Vec<f32> = vec![];
    let mut col_maps: Vec<HashMap<String, f32>> = vec![HashMap::new(); 14];

    for line in contents.lines() {
        let line_str = line.trim();
        if line_str.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line_str.split(',').collect();
        if parts.len() < 15 {
            continue;
        }

        // Target label is the 15th column (index 14)
        let label_str = parts[14].trim().trim_matches('.');
        let label: f32 = if label_str == ">50K" { 1.0 } else { 0.0 };
        labels.push(label);

        // First 14 columns are features
        let mut converted_data: Vec<f32> = Vec::with_capacity(14);
        for (col_idx, &val) in parts[..14].iter().enumerate() {
            let val_str = val.trim();
            if val_str == "?" || val_str.is_empty() {
                converted_data.push(-1.0);
            } else if let Ok(num) = val_str.parse::<f32>() {
                converted_data.push(num);
            } else {
                let map = &mut col_maps[col_idx];
                let next_id = map.len() as f32;
                let cat_id = *map.entry(val_str.to_string()).or_insert(next_id);
                converted_data.push(cat_id);
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
