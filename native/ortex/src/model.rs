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
use std::convert::{Into, TryFrom};

use ort::{Error, ExecutionProviderDispatch, Session, Value};
use rustler::resource::ResourceArc;
use rustler::Atom;

/// Holds the model state which include onnxruntime session and environment. All
/// are threadsafe so this can be called concurrently from the beam.
pub struct OrtexModel {
    pub session: ort::Session,
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
    eps: Vec<ExecutionProviderDispatch>,
    opt: i32,
) -> Result<OrtexModel, Error> {
    // TODO: send tracing logs to erlang/elixir _somehow_
    // tracing_subscriber::fmt::init();
    ort::init()
        .with_execution_providers(&eps)
        .with_name("ortex-model")
        .commit()?;

    let session = Session::builder()?
        .with_execution_providers(&eps)?
        .with_optimization_level(map_opt_level(opt))?
        .with_model_from_file(model_path)?;

    let state = OrtexModel { session };
    Ok(state)
}

/// Returns input/output information about a model. The result is a Tuple of
/// `inputs` and `outputs` with elements of `(Name, Type, Dimension)` where
/// `Dimension` elements of -1 are dynamic.
pub fn show(
    model: ResourceArc<OrtexModel>,
) -> (
    Vec<(String, String, Option<Vec<i64>>)>,
    Vec<(String, String, Option<Vec<i64>>)>,
) {
    let model: &OrtexModel = &*model;

    let mut inputs = Vec::new();
    for input in model.session.inputs.iter() {
        let name = input.name.to_string();
        let repr = format!("{:#?}", input.input_type);
        let dims = Option::<&Vec<i64>>::cloned(input.input_type.tensor_dimensions());
        inputs.push((name, repr, dims));
    }

    let mut outputs = Vec::new();
    for output in model.session.outputs.iter() {
        let name = output.name.to_string();
        let repr = format!("{:#?}", output.output_type);
        let dims = Option::<&Vec<i64>>::cloned(output.output_type.tensor_dimensions());
        outputs.push((name, repr, dims));
    }

    (inputs, outputs)
}

/// Runs the model with the given inputs. Returns a vector of tensors. Use `model::show`
/// to see what the model expects for input and output shapes.
pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Error> {
    // TODO: can we handle an error more elegantly than just .unwrap()?
    let final_input: Vec<Value> = inputs
        .into_iter()
        .map(|x| Value::try_from(&*x).unwrap())
        .collect();

    // Grab the session and run a forward pass with it
    let session = &model.session;

    // Construct a Vec of ModelOutput enums based on the DynOrtTensor data type
    let outputs = session.run(&final_input[..])?;

    outputs
        .iter()
        .map(|(_name, val)| {
            let val: &Value = val;
            let ortextensor: OrtexTensor = OrtexTensor::try_from(val)?;
            let shape = ortextensor.shape();
            let (dtype, bits) = ortextensor.dtype();
            Ok((ResourceArc::new(ortextensor), shape, dtype, bits))
        })
        .collect()
}
