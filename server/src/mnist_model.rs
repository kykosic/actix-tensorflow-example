use anyhow::Result;
use image::{self, imageops::FilterType};
use serde::Serialize;
use tensorflow::{
    Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor,
    DEFAULT_SERVING_SIGNATURE_DEF_KEY, PREDICT_INPUTS, PREDICT_OUTPUTS,
};

#[derive(Debug)]
pub struct MnistInput(Vec<f32>);

impl MnistInput {
    pub fn from_image_bytes(bytes: Vec<u8>) -> Result<Self> {
        const NORM_SCALE: f32 = 1. / 255.;
        let im = image::load_from_memory(&bytes)?
            .resize_exact(28, 28, FilterType::Nearest)
            .grayscale()
            .to_bytes()
            .into_iter()
            .map(|x| (x as f32) * NORM_SCALE)
            .collect::<Vec<f32>>();
        Ok(Self(im))
    }
}

#[derive(Debug, Serialize)]
pub struct MnistPrediction {
    pub label: u8,
    pub confidence: f32,
}

pub struct MnistModel {
    bundle: SavedModelBundle,
    input_op: Operation,
    input_index: i32,
    output_op: Operation,
    output_index: i32,
}

impl MnistModel {
    pub fn from_dir(export_dir: &str) -> Result<Self> {
        const MODEL_TAG: &str = "serve";
        let mut graph = Graph::new();
        let bundle =
            SavedModelBundle::load(&SessionOptions::new(), &[MODEL_TAG], &mut graph, export_dir)?;

        let sig = bundle
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
        let input_info = sig.get_input(PREDICT_INPUTS)?;
        let output_info = sig.get_output(PREDICT_OUTPUTS)?;
        let input_op = graph.operation_by_name_required(&input_info.name().name)?;
        let output_op = graph.operation_by_name_required(&output_info.name().name)?;
        let input_index = input_info.name().index;
        let output_index = output_info.name().index;

        Ok(Self {
            bundle,
            input_op,
            input_index,
            output_op,
            output_index,
        })
    }

    pub fn predict(&self, image: MnistInput) -> Result<MnistPrediction> {
        const INPUT_DIMS: &[u64] = &[1, 28, 28, 1];
        let input_tensor = Tensor::<f32>::new(INPUT_DIMS).with_values(&image.0)?;
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&self.input_op, self.input_index, &input_tensor);
        let output_fetch = run_args.request_fetch(&self.output_op, self.output_index);
        self.bundle.session.run(&mut run_args)?;

        let output = run_args.fetch::<f32>(output_fetch)?;
        let mut confidence = 0f32;
        let mut label = 0u8;
        for i in 0..output.dims()[1] {
            let conf = output[i as usize];
            if conf > confidence {
                confidence = conf;
                label = i as u8;
            }
        }

        Ok(MnistPrediction { label, confidence })
    }
}
