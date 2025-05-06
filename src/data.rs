use anyhow::Result;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use image::GenericImageView;
use std::path::PathBuf;

pub const IMG_WIDTH: usize = 90;
pub const IMG_HEIGHT: usize = 50;

#[derive(Clone, Default)]
pub struct MnistBatcher {}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> Batcher<B, ImageItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, samples: Vec<ImageItem>, device: &B::Device) -> MnistBatch<B> {
        let targets = samples
            .iter()
            .map(|item| Tensor::<B, 1>::from_data([item.label.elem::<B::FloatElem>()], device))
            .collect();
        let images = samples
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, IMG_WIDTH, IMG_HEIGHT]))
            .collect();
        let targets = Tensor::cat(targets, 0);
        let images = Tensor::cat(images, 0);

        MnistBatch { images, targets }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImageItem {
    pub image: [f32; IMG_WIDTH * IMG_HEIGHT],
    pub label: i32,
}

impl ImageItem {
    pub fn from(image_raw: Vec<u8>, label: i32) -> Self {
        assert_eq!(image_raw.len(), IMG_WIDTH * IMG_HEIGHT);
        let mut image = [0.0; IMG_WIDTH * IMG_HEIGHT];
        for (i, v) in image_raw.iter().enumerate() {
            image[i] = (*v as f32) / (u8::MAX as f32);
        }
        Self { image, label }
    }
}

pub struct ImageData {
    data: Vec<ImageItem>,
}

impl ImageData {
    /// Creates a new train dataset.
    pub fn train() -> Result<Self> {
        let data = Self::read()?;
        Ok(Self { data })
    }

    /// Creates a new test dataset.
    pub fn test() -> Result<Self> {
        let data = Self::read()?;
        Ok(Self { data })
    }

    fn read() -> Result<Vec<ImageItem>> {
        let dir = PathBuf::from("./data");
        let mut data = Vec::new();
        for entry in dir.read_dir()? {
            let path = entry?.path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if !name.contains("webp") {
                continue;
            }
            let image = image::open(path)?;
            let image = image.into_luma8().into_raw();
            let label = name
                .split("_")
                .next()
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap();
            data.push(ImageItem::from(image, label));
        }
        Ok(data)
    }
}

impl Dataset<ImageItem> for ImageData {
    fn get(&self, index: usize) -> Option<ImageItem> {
        self.data.get(index).copied()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}
