use burn::{
    backend::{Autodiff, NdArray, Wgpu},
    optim::AdamConfig,
};
use mion::{
    model::ModelConfig,
    training::{TrainingConfig, train},
};

fn main() {
    // type MyBackend = Wgpu<f32, i32>;
    type MyBackend = NdArray<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    // let device = burn::backend::wgpu::WgpuDevice::default();
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let artifact_dir = "/tmp/guide";
    train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
