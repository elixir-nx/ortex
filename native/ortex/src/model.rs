//! Abstractions for creating an ONNX Runtime Session and Environment which can be safely
//! passed to and from the BEAM.
//!
//! # Examples
//!
//! ```
//! let model = init("./models/resnet50.onnx", vec![])?;
//! let (inputs, outputs) = show(model)?;
//! ```

use crate::tensor::OrtexTensor;
use crate::utils::map_opt_level;

use std::convert::TryFrom;
use std::sync::Arc;

use ort::{
    environment, execution_providers::ExecutionProvider, session::SessionBuilder,
    tensor::InputTensor, LoggingLevel, OrtError,
};
use rustler::resource::ResourceArc;
use rustler::Atom;

/// Holds the model state which include onnxruntime session and environment. All
/// are threadsafe so this can be called concurrently from the beam.
pub struct OrtexModel {
    pub env: Arc<environment::Environment>,
    pub session: ort::session::Session,
}

// Since we're only using the session for inference and
// inference is threadsafe, this Sync is safe. Additionally,
// Environment is global and also threadsafe
// https://github.com/microsoft/onnxruntime/issues/114
unsafe impl Sync for OrtexModel {}

/// Creates a model given the path to the model and vector of execution providers.
/// The execution providers are Atoms from Erlang/Elixir.
pub fn init(
    model_path: String,
    eps: Vec<ExecutionProvider>,
    opt: i32,
) -> Result<OrtexModel, OrtError> {
    // TODO: send tracing logs to erlang/elixir _somehow_
    // tracing_subscriber::fmt::init();
    let environment = environment::Environment::builder()
        .with_name("ortex-model")
        .with_log_level(LoggingLevel::Verbose)
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_execution_providers(eps)?
        .with_optimization_level(map_opt_level(opt))?
        .with_model_from_file(model_path)?;

    let state = OrtexModel {
        env: environment.to_owned(),
        session,
    };
    Ok(state)
}

/// Returns input/output information about a model. The result is a Tuple of
/// `inputs` and `outputs` with elements of `(Name, Type, Dimension)` where
/// `Dimension` elements of -1 are dynamic.
pub fn show(
    model: ResourceArc<OrtexModel>,
) -> (
    Vec<(String, String, Vec<Option<u32>>)>,
    Vec<(String, String, Vec<Option<u32>>)>,
) {
    let inputs: Vec<(String, String, Vec<Option<u32>>)> = model
        .session
        .inputs
        .iter()
        .map(|i| {
            let inp_type = i.input_type;
            let dims = &i.dimensions;
            (
                i.name.to_string(),
                format!("{inp_type:#?}"),
                dims.to_owned(),
            )
        })
        .collect();
    let outputs: Vec<(String, String, Vec<Option<u32>>)> = model
        .session
        .outputs
        .iter()
        .map(|i| {
            let out_type = i.output_type;
            let dims = &i.dimensions;
            (
                i.name.to_string(),
                format!("{out_type:#?}"),
                dims.to_owned(),
            )
        })
        .collect();

    (inputs.into(), outputs.into())
}

/// Runs the model with the given inputs. Returns a vector of tensors. Use `model::show`
/// to see what the model expects for input and output shapes.
pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, OrtError> {
    let final_input: Vec<InputTensor> =
        inputs.into_iter().map(|x| InputTensor::from(&*x)).collect();

    // Grab the session and run a forward pass with it
    let session = &model.session;

    // Construct a Vec of ModelOutput enums based on the DynOrtTensor data type
    let result: Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, OrtError> =
        session
            .run(final_input)?
            .iter()
            .map(
                |e| -> Result<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize), OrtError> {
                    let ortextensor = OrtexTensor::try_from(e)?;
                    let shape = ortextensor.shape();
                    let (dtype, bits) = ortextensor.dtype();
                    Ok((ResourceArc::new(ortextensor), shape, dtype, bits))
                },
            )
            .collect();
    result
}
