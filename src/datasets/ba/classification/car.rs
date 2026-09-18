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

    for line in contents.lines() {
        let line_str = line.trim();
        if line_str.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line_str.split(',').collect();
        if parts.len() < 7 {
            continue;
        }

        // Target label is the 7th column (index 6)
        let label_str = parts[6].trim();
        let label: f32 = match label_str {
            "unacc" => 0.0,
            "acc" => 1.0,
            "good" => 2.0,
            "vgood" => 3.0,
            _ => panic!("Unknown car class label: {}", label_str),
        };
        labels.push(label);

        // 6 input feature columns
        let mut converted_data: Vec<f32> = Vec::with_capacity(6);
        for (i, &val) in parts[..6].iter().enumerate() {
            let v = val.trim();
            let feat_val = match i {
                0 | 1 => match v {
                    "vhigh" => 3.0,
                    "high" => 2.0,
                    "med" => 1.0,
                    "low" => 0.0,
                    _ => v.parse::<f32>().unwrap_or(-1.0),
                },
                2 => match v {
                    "2" => 2.0,
                    "3" => 3.0,
                    "4" => 4.0,
                    "5more" => 5.0,
                    _ => v.parse::<f32>().unwrap_or(-1.0),
                },
                3 => match v {
                    "2" => 2.0,
                    "4" => 4.0,
                    "more" => 5.0,
                    _ => v.parse::<f32>().unwrap_or(-1.0),
                },
                4 => match v {
                    "small" => 0.0,
                    "med" => 1.0,
                    "big" => 2.0,
                    _ => v.parse::<f32>().unwrap_or(-1.0),
                },
                5 => match v {
                    "low" => 0.0,
                    "med" => 1.0,
                    "high" => 2.0,
                    _ => v.parse::<f32>().unwrap_or(-1.0),
                },
                _ => v.parse::<f32>().unwrap_or(-1.0),
            };
            converted_data.push(feat_val);
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
