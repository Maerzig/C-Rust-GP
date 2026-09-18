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

        // Split by tab for the TSV file
        let mut parts: Vec<&str> = line.split('\t').collect();
        
        // Pop the last element as the target regression value
        let label: f32 = parts.pop().unwrap().trim().parse::<f32>().unwrap();
        labels.push(label);

        // Parse remaining elements as feature data
        let converted_data: Vec<f32> = parts.iter().map(|val| val.trim().parse::<f32>().unwrap()).collect();
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
