/// Central batched GPU evaluator using ONNX Runtime.
///
/// Workers send leaf positions through a channel. The evaluator thread
/// collects batches and runs inference on GPU, returning (policy, value) pairs.

use crate::encoding::ENCODED_SIZE;
use crossbeam_channel::{Receiver, Sender};
use std::time::{Duration, Instant};

/// Request from a worker to evaluate a position
pub struct EvalRequest {
    pub encoded_state: [f32; ENCODED_SIZE],
    pub response_tx: Sender<EvalResponse>,
}

/// Response from the evaluator
#[derive(Clone)]
pub struct EvalResponse {
    pub policy: [f32; 9],
    pub value: f32,
}

/// Evaluator configuration
pub struct EvaluatorConfig {
    pub model_path: String,
    pub batch_size: usize,
    pub max_wait_us: u64,
    pub use_gpu: bool,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        EvaluatorConfig {
            model_path: "model.onnx".to_string(),
            batch_size: 128,
            max_wait_us: 2000,
            use_gpu: true,
        }
    }
}

/// Run the evaluator loop. Call this in a dedicated thread.
pub fn evaluator_loop(
    rx: Receiver<EvalRequest>,
    config: EvaluatorConfig,
) {
    // Initialize ONNX Runtime session
    let mut builder = ort::session::Session::builder()
        .expect("Failed to create ONNX session builder");

    if config.use_gpu {
        builder = builder
            .with_execution_providers([ort::ep::CUDA::default().build()])
            .expect("Failed to set CUDA execution provider");
    }

    let mut session = builder
        .commit_from_file(&config.model_path)
        .unwrap_or_else(|e| panic!("Failed to load ONNX model {}: {}", config.model_path, e));

    eprintln!("ONNX model loaded: {}", config.model_path);
    eprintln!("Batch size: {}, GPU: {}", config.batch_size, config.use_gpu);

    let max_wait = Duration::from_micros(config.max_wait_us);
    let mut batch_count: u64 = 0;
    let mut total_evals: u64 = 0;

    loop {
        // Wait for first request (blocking)
        let first = match rx.recv() {
            Ok(req) => req,
            Err(_) => break, // All senders dropped, exit
        };

        let mut batch: Vec<EvalRequest> = Vec::with_capacity(config.batch_size);
        batch.push(first);

        // Collect more: try_recv first (zero-wait), then short deadline
        while batch.len() < config.batch_size {
            match rx.try_recv() {
                Ok(req) => batch.push(req),
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    // No more queued. If we have a decent batch, go.
                    if batch.len() >= 16 { break; }
                    // Otherwise wait briefly for more
                    let deadline = Instant::now() + max_wait;
                    match rx.recv_deadline(deadline) {
                        Ok(req) => batch.push(req),
                        Err(_) => break,
                    }
                }
                Err(_) => break,
            }
        }

        let batch_len = batch.len();

        // Build input tensor: [batch_len, 7, 9]
        let mut input_data: Vec<f32> = Vec::with_capacity(batch_len * ENCODED_SIZE);
        for req in &batch {
            input_data.extend_from_slice(&req.encoded_state);
        }

        // Create tensor using (shape, data) tuple
        let input_value = ort::value::Tensor::from_array(
            ([batch_len, 7, 9], input_data)
        )
        .expect("Failed to create input tensor");

        // Run inference
        let outputs = session
            .run(ort::inputs![input_value])
            .expect("ONNX inference failed");

        // Parse outputs
        // Output 0: log_policy [batch, 9] (log_softmax)
        // Output 1: value [batch, 1] (tanh)
        let (_policy_shape, log_policy_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract log_policy");
        let (_value_shape, value_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract value");

        batch_count += 1;
        total_evals += batch_len as u64;

        // Send responses
        for (i, req) in batch.into_iter().enumerate() {
            let mut policy = [0.0f32; 9];

            // Convert log_softmax to softmax
            let row = &log_policy_data[i * 9..(i + 1) * 9];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..9 {
                policy[j] = (row[j] - max_val).exp();
                sum += policy[j];
            }
            if sum > 0.0 {
                for j in 0..9 {
                    policy[j] /= sum;
                }
            }

            let value = value_data[i];

            let _ = req.response_tx.send(EvalResponse { policy, value });
        }
    }

    eprintln!("Evaluator thread exiting (processed {} batches, {} evals)", batch_count, total_evals);
}

/// Dummy evaluator for testing (uniform policy, value=0)
pub fn dummy_evaluator_loop(rx: Receiver<EvalRequest>) {
    loop {
        match rx.recv() {
            Ok(req) => {
                let _ = req.response_tx.send(EvalResponse {
                    policy: [1.0 / 9.0; 9],
                    value: 0.0,
                });
            }
            Err(_) => break,
        }
    }
}
