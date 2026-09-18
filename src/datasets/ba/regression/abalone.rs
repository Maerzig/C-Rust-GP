use std::fs;
use crate::datasets::real_world_uci::dataset_utils::{shuffle, standardize_dataset, standardize_labels};

pub fn get_dataset(dataset_path: String, shuffle_and_split: bool) -> (Vec<Vec<f32>>,
                         Vec<f32>,
                         Vec<Vec<f32>>,
                         Vec<f32>) {
    let contents = fs::read_to_string(dataset_path)
        .expect("Should have been able to read the file");
    let contents = contents.lines();
    let mut datas: Vec<Vec<f32>> = vec![];
    let mut labels: Vec<f32> = vec![];
    for line in contents {
        if line.trim().is_empty() {
            continue;
        }
        let mut line: Vec<&str> = line.split(',').collect();
        let label: f32 = line.pop().unwrap().trim().parse::<f32>().unwrap();
        labels.push(label);

        let converted_data: Vec<f32> = line.iter().enumerate().map(|(i, val)| {
            let val_str = val.trim();
            if i == 0 {
                match val_str {
                    "M" => 0.0,
                    "F" => 1.0,
                    "I" => 2.0,
                    _ => val_str.parse::<f32>().unwrap_or(-1.0),
                }
            } else {
                val_str.parse::<f32>().unwrap()
            }
        }).collect();
        datas.push(converted_data);
    }

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
