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
use crate::utils::{is_bool_input, map_opt_level};
use std::convert::TryInto;
use std::iter::zip;
use std::sync::Mutex;

use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::{Session, SessionInputValue};
use ort::Error;
use rustler::{Atom, Resource, ResourceArc};

/// Holds the model state which include onnxruntime session and environment. All
/// are threadsafe so this can be called concurrently from the beam.
pub struct OrtexModel {
    pub session: Mutex<Session>,
}

#[rustler::resource_impl(name = "OrtexModel")]
impl Resource for OrtexModel {}

impl std::panic::RefUnwindSafe for OrtexModel {}
impl std::panic::UnwindSafe for OrtexModel {}

/// Creates a model given the path to the model and vector of execution providers.
/// The execution providers are Atoms from Erlang/Elixir.
pub fn init(
    model_path: String,
    eps: Vec<ExecutionProviderDispatch>,
    opt: i32,
) -> Result<OrtexModel, Error> {
    // TODO: send tracing logs to erlang/elixir _somehow_
    // tracing_subscriber::fmt::init();

    let session = Session::builder()?
        .with_optimization_level(map_opt_level(opt))?
        .with_execution_providers(eps)?
        .commit_from_file(model_path)?;

    let state = OrtexModel {
        session: Mutex::new(session),
    };
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

    let session = model.session.lock().unwrap_or_else(|e| e.into_inner());

    let mut inputs = Vec::new();
    for input in session.inputs() {
        let name = input.name().to_string();
        let repr = format!("{:#?}", input.dtype());
        let dims = match input.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => Some(shape.to_vec()),
            _ => None,
        };
        inputs.push((name, repr, dims));
    }

    let mut outputs = Vec::new();
    for output in session.outputs() {
        let name = output.name().to_string();
        let repr = format!("{:#?}", output.dtype());
        let dims = match output.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => Some(shape.to_vec()),
            _ => None,
        };
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
    // Grab the session and run a forward pass with it
    let mut session = model.session.lock().unwrap_or_else(|e| e.into_inner());
    let mut ortified_inputs: Vec<SessionInputValue> = Vec::new();
    let output_names: Vec<String>;

    {
        let session_inputs = session.inputs();
        let expected_inputs = session_inputs.len();
        if inputs.len() != expected_inputs {
            return Err(Error::new(format!(
                "Expected {} input(s), got {}",
                expected_inputs,
                inputs.len()
            )));
        }

        for (elixir_input, onnx_input) in zip(inputs, session_inputs) {
            let derefed_input: &OrtexTensor = &elixir_input;
            if is_bool_input(onnx_input.dtype()) {
                // this assumes that the boolean input isn't huge -- we're cloning it twice;
                // once below, once in the try_into()
                let boolified_input = derefed_input.clone().to_bool()?;
                let v: SessionInputValue = (&boolified_input).try_into()?;
                ortified_inputs.push(v);
            } else {
                let v: SessionInputValue = derefed_input.try_into()?;
                ortified_inputs.push(v);
            }
        }

        output_names = session
            .outputs()
            .iter()
            .map(|output| output.name().to_string())
            .collect();
    }

    // Construct a Vec of ModelOutput enums based on the DynOrtTensor data type
    let outputs = session.run(&ortified_inputs[..])?;
    let mut collected_outputs = Vec::new();

    for output_name in output_names {
        let val = outputs.get(&output_name).ok_or_else(|| {
            Error::new(format!(
                "Expected {} to be in the outputs, but didn't find it",
                output_name
            ))
        })?;

        // NOTE: try_into impl here will implicitly map bool outputs to u8 outputs
        let ortextensor: OrtexTensor = val.try_into()?;
        let shape = ortextensor.shape();
        let (dtype, bits) = ortextensor.dtype();

        let collected_output = (ResourceArc::new(ortextensor), shape, dtype, bits);
        collected_outputs.push(collected_output)
    }

    Ok(collected_outputs)
}
