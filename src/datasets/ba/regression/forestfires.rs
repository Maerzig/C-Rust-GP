use std::fs;
use crate::datasets::real_world_uci::dataset_utils::{shuffle, standardize_dataset, standardize_labels};

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
        
        let label: f32 = parts.pop().unwrap().trim().parse::<f32>().unwrap();
        labels.push(label);

        let mut converted_data: Vec<f32> = Vec::with_capacity(parts.len());
        for (i, val) in parts.iter().enumerate() {
            let val_str = val.trim();
            if i == 2 {
                // Convert month to a float
                let m = match val_str.to_lowercase().as_str() {
                    "jan" => 0.0, "feb" => 1.0, "mar" => 2.0, "apr" => 3.0,
                    "may" => 4.0, "jun" => 5.0, "jul" => 6.0, "aug" => 7.0,
                    "sep" => 8.0, "oct" => 9.0, "nov" => 10.0, "dec" => 11.0,
                    _ => val_str.parse::<f32>().unwrap_or(-1.0),
                };
                converted_data.push(m);
            } else if i == 3 {
                // Convert day to a float
                let d = match val_str.to_lowercase().as_str() {
                    "mon" => 0.0, "tue" => 1.0, "wed" => 2.0, "thu" => 3.0,
                    "fri" => 4.0, "sat" => 5.0, "sun" => 6.0,
                    _ => val_str.parse::<f32>().unwrap_or(-1.0),
                };
                converted_data.push(d);
            } else {
                converted_data.push(val_str.parse::<f32>().unwrap());
            }
        }
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