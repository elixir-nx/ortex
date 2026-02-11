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

use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;
use ort::Error;
use rustler::Atom;
use rustler::Resource;
use rustler::ResourceArc;
use std::error::Error as StdError;
use std::sync::Mutex;

/// Holds the model state which include onnxruntime session and environment. All
/// are threadsafe so this can be called concurrently from the beam.
pub struct OrtexModel {
    pub session: Mutex<ort::session::Session>,
}
impl Resource for OrtexModel {}

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

    let session = Session::builder()?
        .with_optimization_level(map_opt_level(opt))?
        .with_execution_providers(eps)?
        .commit_from_file(model_path)?;

    let state = OrtexModel {
        session: session.into(),
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
    let session: &mut ort::session::Session = &mut model.session.lock().unwrap();

    let mut inputs = Vec::new();
    for input in session.inputs.iter() {
        let name = input.name.to_string();
        let repr = format!("{:#?}", input.input_type);
        let dims: Option<Vec<i64>> = input.input_type.tensor_shape().map(|s| s.to_vec());
        inputs.push((name, repr, dims));
    }

    let mut outputs = Vec::new();
    for output in session.outputs.iter() {
        let name = output.name.to_string();
        let repr = format!("{:#?}", output.output_type);
        let dims: Option<Vec<i64>> = output.output_type.tensor_shape().map(|s| s.to_vec());
        outputs.push((name, repr, dims));
    }

    (inputs, outputs)
}

/// Runs the model with the given inputs. Returns a vector of tensors. Use `model::show`
/// to see what the model expects for input and output shapes.
pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Box<dyn StdError>> {
    let session: &mut ort::session::Session = &mut model.session.lock().unwrap();

    let mut ortified_inputs: Vec<ort::session::SessionInputValue> = Vec::new();

    for (elixir_input, onnx_input) in zip(inputs, &session.inputs) {
        let derefed_input: &OrtexTensor = &elixir_input;
        if is_bool_input(&onnx_input.input_type) {
            let boolified_input: &OrtexTensor = &derefed_input.clone().to_bool();
            let v: ort::session::SessionInputValue = boolified_input.try_into()?;
            ortified_inputs.push(v);
        } else {
            let v: ort::session::SessionInputValue = derefed_input.try_into()?;
            ortified_inputs.push(v);
        }
    }

    let outputs = session.run(&ortified_inputs[..])?;
    let mut collected_outputs = Vec::new();

    for output_name in outputs.keys() {
        let val = outputs.get(output_name).expect(
            &format!(
                "Expected {} to be in the outputs, but didn't find it",
                output_name
            )[..],
        );

        let ortextensor: OrtexTensor = val.try_into()?;
        let shape = ortextensor.shape();
        let (dtype, bits) = ortextensor.dtype();
        let collected_output = (ResourceArc::new(ortextensor), shape, dtype, bits);
        collected_outputs.push(collected_output);
    }

    Ok(collected_outputs)
}
