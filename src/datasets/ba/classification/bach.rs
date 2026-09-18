use std::collections::HashMap;
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
    let mut class_map: HashMap<String, f32> = HashMap::new();
    let mut col_maps: Vec<HashMap<String, f32>> = vec![HashMap::new(); 14];

    for line in lines {
        let line_str = line.trim();
        if line_str.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line_str.split(',').collect();
        if parts.len() < 16 {
            continue;
        }

        // Target label is the 16th column (index 15)
        let label_str = parts[15].trim().trim_matches('\'').trim().to_string();
        let next_class_id = class_map.len() as f32;
        let label = *class_map.entry(label_str).or_insert(next_class_id);
        labels.push(label);

        // Column 0 is eventNumber (ID column) -> skip it!
        // Columns 1..15 are feature columns (14 features total)
        let mut converted_data: Vec<f32> = Vec::with_capacity(14);
        for (feat_idx, &val) in parts[1..15].iter().enumerate() {
            let val_str = val.trim().trim_matches('\'').trim();
            if val_str == "YES" {
                converted_data.push(1.0);
            } else if val_str == "NO" {
                converted_data.push(0.0);
            } else if let Ok(num) = val_str.parse::<f32>() {
                converted_data.push(num);
            } else {
                let map = &mut col_maps[feat_idx];
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
