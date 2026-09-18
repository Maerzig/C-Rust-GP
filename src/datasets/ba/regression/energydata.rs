use std::fs;
use crate::datasets::real_world_uci::dataset_utils::{shuffle, standardize_dataset, standardize_labels};

fn parse_datetime_to_minutes(datetime_str: &str) -> f32 {
    let parts: Vec<&str> = datetime_str.split(' ').collect();
    if parts.len() == 2 {
        let date_parts: Vec<&str> = parts[0].split('-').collect();
        let time_parts: Vec<&str> = parts[1].split(':').collect();
        if date_parts.len() == 3 && time_parts.len() == 3 {
            let yr: f32 = date_parts[0].parse().unwrap_or(2016.0);
            let mnth: f32 = date_parts[1].parse().unwrap_or(1.0);
            let day: f32 = date_parts[2].parse().unwrap_or(1.0);
            let hr: f32 = time_parts[0].parse().unwrap_or(0.0);
            let min: f32 = time_parts[1].parse().unwrap_or(0.0);
            let sec: f32 = time_parts[2].parse().unwrap_or(0.0);
            let days = (yr - 2016.0) * 365.25 + (mnth - 1.0) * 30.4375 + (day - 1.0);
            return days * 1440.0 + hr * 60.0 + min + sec / 60.0;
        }
    }
    0.0
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

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        
        // Target regression value is 'Appliances' at column index 1
        let label_str = parts[1].trim().trim_matches('"').trim();
        let label: f32 = label_str.parse::<f32>().unwrap();
        labels.push(label);

        let mut converted_data: Vec<f32> = Vec::with_capacity(parts.len() - 1);
        for (i, val) in parts.iter().enumerate() {
            if i == 1 {
                continue; // Skip target label column
            }
            let val_str = val.trim().trim_matches('"').trim();
            if i == 0 {
                converted_data.push(parse_datetime_to_minutes(val_str));
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
