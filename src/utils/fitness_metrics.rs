pub fn fitness_regression(prediction: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>) -> f32 {
    assert_eq!(prediction.len(), labels.len());
    assert_eq!(prediction[0].len(), labels[0].len());
    let mut fitness: f32 = 0.;
    prediction.iter().zip(labels.iter()).for_each(|(inner_pred, inner_label)|
            inner_pred.iter().zip(inner_label.iter()).for_each(|(x, y)| fitness += (x - y).abs())
    );

    fitness /= (prediction.len() * prediction[0].len()) as f32;

    if fitness.is_nan() {
        fitness = f32::MAX;
    } else if fitness.is_infinite() {
        fitness = f32::MAX;
    }

    fitness
}

pub fn fitness_regression_mse(prediction: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>) -> f32 {
    assert_eq!(prediction.len(), labels.len());
    assert_eq!(prediction[0].len(), labels[0].len());
    let mut fitness: f32 = 0.;
    prediction.iter().zip(labels.iter()).for_each(|(inner_pred, inner_label)|
            inner_pred.iter().zip(inner_label.iter()).for_each(|(x, y)| fitness += (x - y).powi(2))
    );

    fitness /= (prediction.len() * prediction[0].len()) as f32;

    if fitness.is_nan() {
        fitness = f32::MAX;
    } else if fitness.is_infinite() {
        fitness = f32::MAX;
    }

    fitness
}


pub fn fitness_boolean(prediction: &Vec<Vec<bool>>, labels: &Vec<Vec<bool>>) -> f32 {
    assert_eq!(prediction.len(), labels.len());

    let mut fitness: i32 = 0;
    prediction.iter().zip(labels.iter()).for_each(|(inner_pred, inner_label)|
        inner_pred.iter().zip(inner_label.iter()).for_each(|(x, y)| { if x == y { fitness += 1 } })
    );

    let number_bits = labels[0].len() * labels.len();

    
    1. - (fitness as f32 / number_bits as f32)
}


pub fn fitness_classification_mcc(prediction: &Vec<Vec<f32>>, labels: &Vec<Vec<f32>>) -> (f32, f32) {
    if prediction.is_empty() || prediction[0].is_empty() {
        return (2.0, 1.0);
    }

    for row in prediction {
        for &val in row {
            if val.is_nan() || val.is_infinite() {
                return (2.0, 1.0);
            }
        }
    }

    let is_single_output = prediction.len() == 1;
    let nbr_classes = if is_single_output { 2 } else { prediction.len() };
    let nbr_samples = prediction[0].len();

    let mut correct_count: f32 = 0.0;
    let mut true_class_counts: Vec<f32> = vec![0.0; nbr_classes];
    let mut pred_class_counts: Vec<f32> = vec![0.0; nbr_classes];

    for sample_idx in 0..nbr_samples {
        let pred_class = if is_single_output {
            if prediction[0][sample_idx] > 0.0 { 1 } else { 0 }
        } else {
            let mut best_class = 0;
            let mut max_pred_val = prediction[0][sample_idx];
            for class_idx in 1..nbr_classes {
                if prediction[class_idx][sample_idx] > max_pred_val {
                    max_pred_val = prediction[class_idx][sample_idx];
                    best_class = class_idx;
                }
            }
            best_class
        };

        let target_class = if labels.len() == 1 {
            labels[0][sample_idx].round() as usize
        } else {
            let mut best_label_idx = 0;
            let mut max_label_val = labels[0][sample_idx];
            for class_idx in 1..labels.len() {
                if labels[class_idx][sample_idx] > max_label_val {
                    max_label_val = labels[class_idx][sample_idx];
                    best_label_idx = class_idx;
                }
            }
            best_label_idx
        };

        if pred_class == target_class {
            correct_count += 1.0;
        }
        if target_class < nbr_classes {
            true_class_counts[target_class] += 1.0;
        }
        if pred_class < nbr_classes {
            pred_class_counts[pred_class] += 1.0;
        }
    }

    let n = nbr_samples as f32;
    let mae = 1.0 - (correct_count / n);

    let mut mcc_num = correct_count * n;
    let mut pred_sq_sum: f32 = 0.0;
    let mut true_sq_sum: f32 = 0.0;

    for class_idx in 0..nbr_classes {
        mcc_num -= true_class_counts[class_idx] * pred_class_counts[class_idx];
        pred_sq_sum += pred_class_counts[class_idx].powi(2);
        true_sq_sum += true_class_counts[class_idx].powi(2);
    }

    let term1 = (n.powi(2) - pred_sq_sum).max(0.0).sqrt();
    let term2 = (n.powi(2) - true_sq_sum).max(0.0).sqrt();
    let denom = term1 * term2;

    let mcc = if denom > 0.0 && !denom.is_nan() {
        (mcc_num / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    let mut fitness = 1.0 - mcc;
    if fitness.is_nan() || fitness.is_infinite() {
        fitness = 2.0;
    }

    (fitness, mae)
}
