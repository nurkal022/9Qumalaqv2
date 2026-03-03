/// ONNX Runtime inference wrapper for the dual-head neural network.
///
/// Loads an ONNX model and provides predict(features) -> (policy_logits, value).

use std::sync::Mutex;

use crate::features::FEATURE_SIZE;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

pub const NUM_ACTIONS: usize = 9;

/// Neural network inference via ONNX Runtime.
/// Thread-safe via internal Mutex.
pub struct Network {
    session: Mutex<Session>,
}

impl Network {
    /// Load an ONNX model from file
    pub fn load(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        Ok(Network {
            session: Mutex::new(session),
        })
    }

    /// Run inference on a single position.
    /// Returns (policy_logits[9], value).
    pub fn predict(&self, features: &[f32; FEATURE_SIZE]) -> ([f32; NUM_ACTIONS], f32) {
        let input_data: Vec<f32> = features.to_vec();
        let input_tensor = Tensor::from_array(([1usize, FEATURE_SIZE], input_data)).unwrap();

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs!["features" => input_tensor])
            .unwrap();

        let (_shape, policy_data) = outputs["policy_logits"]
            .try_extract_tensor::<f32>()
            .unwrap();
        let mut policy_logits = [0.0f32; NUM_ACTIONS];
        for i in 0..NUM_ACTIONS {
            policy_logits[i] = policy_data[i];
        }

        let (_shape, value_data) = outputs["value"]
            .try_extract_tensor::<f32>()
            .unwrap();
        let value = value_data[0];

        (policy_logits, value)
    }

    /// Batch inference on multiple positions.
    pub fn predict_batch(
        &self,
        features_batch: &[[f32; FEATURE_SIZE]],
    ) -> Vec<([f32; NUM_ACTIONS], f32)> {
        let batch_size = features_batch.len();
        let flat: Vec<f32> = features_batch.iter().flat_map(|f| f.iter().copied()).collect();
        let input_tensor = Tensor::from_array(([batch_size, FEATURE_SIZE], flat)).unwrap();

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs!["features" => input_tensor])
            .unwrap();

        let (_shape, policy_data) = outputs["policy_logits"]
            .try_extract_tensor::<f32>()
            .unwrap();

        let (_shape, value_data) = outputs["value"]
            .try_extract_tensor::<f32>()
            .unwrap();

        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mut policy_logits = [0.0f32; NUM_ACTIONS];
            for i in 0..NUM_ACTIONS {
                policy_logits[i] = policy_data[b * NUM_ACTIONS + i];
            }
            let value = value_data[b];
            results.push((policy_logits, value));
        }

        results
    }
}

/// A dummy network that returns uniform policy and zero value.
/// Used for testing without an ONNX model.
pub struct RandomNetwork;

impl RandomNetwork {
    pub fn predict(&self, _features: &[f32; FEATURE_SIZE]) -> ([f32; NUM_ACTIONS], f32) {
        ([0.0f32; NUM_ACTIONS], 0.0)
    }
}
