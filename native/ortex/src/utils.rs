//! Serialization and deserialization to transfer between Ortex and BinaryBackend
//! [Nx](https://hexdocs.pm/nx) backend.

use crate::constants::*;
use crate::tensor::OrtexTensor;
use ndarray::{ArrayViewMut, Ix, IxDyn};

use ndarray::ShapeError;

use rustler::resource::ResourceArc;
use rustler::types::Binary;
use rustler::{Atom, Env, NifResult};

use ort::{ExecutionProviderDispatch, GraphOptimizationLevel};

/// A faster (unsafe) way of creating an Array from an Erlang binary
fn initialize_from_raw_ptr<T>(ptr: *const T, shape: &[Ix]) -> ArrayViewMut<T, IxDyn> {
    let array = unsafe { ArrayViewMut::from_shape_ptr(shape, ptr as *mut T) };
    array
}

/// Given a Binary term, shape, and dtype from the BEAM, constructs an OrtexTensor and
/// returns the reference to be used as an Nx.Backend representation.
///
/// # Example
///
/// ```elixir
/// bin = <<1, 0, 0, 0, 1, 0, 0, 0>>
/// ```
///
/// Create a shape `[2]` u32 OrtexTensor from a binary of 8 bytes
/// ```elixir
/// {:ok, reference} = from_binary(bin, {2}, {:u, 32})
/// ```
pub fn from_binary(
    bin: Binary,
    shape: Vec<usize>,
    dtype_str: String,
    dtype_bits: usize,
) -> Result<ResourceArc<OrtexTensor>, ShapeError> {
    match (dtype_str.as_ref(), dtype_bits) {
        ("bf", 16) => Ok(ResourceArc::new(OrtexTensor::bf16(
            initialize_from_raw_ptr(bin.as_ptr() as *const half::bf16, &shape).to_owned(),
        ))),
        ("f", 16) => Ok(ResourceArc::new(OrtexTensor::f16(
            initialize_from_raw_ptr(bin.as_ptr() as *const half::f16, &shape).to_owned(),
        ))),
        ("f", 32) => Ok(ResourceArc::new(OrtexTensor::f32(
            initialize_from_raw_ptr(bin.as_ptr() as *const f32, &shape).to_owned(),
        ))),
        ("f", 64) => Ok(ResourceArc::new(OrtexTensor::f64(
            initialize_from_raw_ptr(bin.as_ptr() as *const f64, &shape).to_owned(),
        ))),
        ("s", 8) => Ok(ResourceArc::new(OrtexTensor::s8(
            initialize_from_raw_ptr(bin.as_ptr() as *const i8, &shape).to_owned(),
        ))),
        ("s", 16) => Ok(ResourceArc::new(OrtexTensor::s16(
            initialize_from_raw_ptr(bin.as_ptr() as *const i16, &shape).to_owned(),
        ))),
        ("s", 32) => Ok(ResourceArc::new(OrtexTensor::s32(
            initialize_from_raw_ptr(bin.as_ptr() as *const i32, &shape).to_owned(),
        ))),
        ("s", 64) => Ok(ResourceArc::new(OrtexTensor::s64(
            initialize_from_raw_ptr(bin.as_ptr() as *const i64, &shape).to_owned(),
        ))),
        ("u", 8) => Ok(ResourceArc::new(OrtexTensor::u8(
            initialize_from_raw_ptr(bin.as_ptr() as *const u8, &shape).to_owned(),
        ))),
        ("u", 16) => Ok(ResourceArc::new(OrtexTensor::u16(
            initialize_from_raw_ptr(bin.as_ptr() as *const u16, &shape).to_owned(),
        ))),
        ("u", 32) => Ok(ResourceArc::new(OrtexTensor::u32(
            initialize_from_raw_ptr(bin.as_ptr() as *const u32, &shape).to_owned(),
        ))),
        ("u", 64) => Ok(ResourceArc::new(OrtexTensor::u64(
            initialize_from_raw_ptr(bin.as_ptr() as *const u64, &shape).to_owned(),
        ))),
        (&_, _) => unimplemented!(),
    }
}

/// Given a reference to an OrtexTensor return the binary representation to be used
/// by the BinaryBackend of Nx.
pub fn to_binary<'a>(
    env: Env<'a>,
    reference: ResourceArc<OrtexTensor>,
    _bits: usize,
    _limit: usize,
) -> NifResult<Binary<'a>> {
    Ok(reference.make_binary(env, |x| x.to_bytes()))
}

/// Takes a vec of Atoms and transforms them into a vec of ExecutionProvider Enums
pub fn map_eps(env: rustler::env::Env, eps: Vec<Atom>) -> Vec<ExecutionProviderDispatch> {
    eps.iter()
        .map(|e| match &e.to_term(env).atom_to_string().unwrap()[..] {
            CPU => ort::CPUExecutionProvider::default().build(),
            CUDA => ort::CUDAExecutionProvider::default().build(),
            TENSORRT => ort::TensorRTExecutionProvider::default().build(),
            ACL => ort::ACLExecutionProvider::default().build(),
            ONEDNN => ort::OneDNNExecutionProvider::default().build(),
            COREML => ort::CoreMLExecutionProvider::default().build(),
            DIRECTML => ort::DirectMLExecutionProvider::default().build(),
            ROCM => ort::ROCmExecutionProvider::default().build(),
            _ => ort::CPUExecutionProvider::default().build(),
        })
        .collect()
}

/// Take an optimization level and returns the
pub fn map_opt_level(opt: i32) -> GraphOptimizationLevel {
    match opt {
        1 => GraphOptimizationLevel::Level1,
        2 => GraphOptimizationLevel::Level2,
        3 => GraphOptimizationLevel::Level3,
        _ => GraphOptimizationLevel::Disable,
    }
}
