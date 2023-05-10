//! Serialization and deserialization to transfer between Ortex and BinaryBackend
//! [Nx](https://hexdocs.pm/nx) backend.

use crate::constants::*;
use crate::tensor::OrtexTensor;

use ndarray::prelude::*;
use ndarray::ShapeError;

use rustler::resource::ResourceArc;
use rustler::types::{Binary, OwnedBinary};
use rustler::{Atom, Env, Error, NifResult};

use ort::{ExecutionProvider, GraphOptimizationLevel};

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
    // TODO: make this more DRY, pull out into an impl
    match (dtype_str.as_ref(), dtype_bits) {
        ("bf", 16) => Ok(ResourceArc::new(OrtexTensor::bf16(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_ne_bytes([c[0], c[1]]))
                    .collect(),
            )
            .into_shape(shape)?,
        ))),
        ("f", 16) => Ok(ResourceArc::new(OrtexTensor::f16(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(2)
                    .map(|c| half::f16::from_ne_bytes([c[0], c[1]]))
                    .collect(),
            )
            .into_shape(shape)?,
        ))),
        ("f", 32) => Ok(ResourceArc::new(OrtexTensor::f32(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
            .into_shape(shape)?,
        ))),
        ("f", 64) => Ok(ResourceArc::new(OrtexTensor::f64(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(8)
                    .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
            )
            .into_shape(shape)?,
        ))),
        ("s", 8) => Ok(ResourceArc::new(OrtexTensor::s8(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(1)
                    .map(|c| i8::from_ne_bytes([c[0]]))
                    .collect(),
            )
            .into_shape(shape)?,
        ))),
        ("s", 16) => Ok(ResourceArc::new(OrtexTensor::s16(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(2)
                    .map(|c| i16::from_ne_bytes([c[0], c[1]]))
                    .collect(),
            )
            .into_shape(shape)?,
        ))),
        ("s", 32) => Ok(ResourceArc::new(OrtexTensor::s32(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(4)
                    .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
            .into_shape(shape)?,
        ))),
        ("s", 64) => Ok(ResourceArc::new(OrtexTensor::s64(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(8)
                    .map(|c| i64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
            )
            .into_shape(shape)?,
        ))),
        ("u", 8) => Ok(ResourceArc::new(OrtexTensor::u8(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(1)
                    .map(|c| u8::from_ne_bytes([c[0]]))
                    .collect(),
            )
            .into_shape(shape)?,
        ))),
        ("u", 16) => Ok(ResourceArc::new(OrtexTensor::u16(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(2)
                    .map(|c| u16::from_ne_bytes([c[0], c[1]]))
                    .collect(),
            )
            .into_shape(shape)?,
        ))),
        ("u", 32) => Ok(ResourceArc::new(OrtexTensor::u32(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(4)
                    .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
            .into_shape(shape)?,
        ))),
        ("u", 64) => Ok(ResourceArc::new(OrtexTensor::u64(
            Array::from_vec(
                bin.as_slice()
                    .chunks_exact(8)
                    .map(|c| u64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
            )
            .into_shape(shape)?,
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
    // TODO: implement limit and size so we aren't dumping the entire binary on every
    // IO.inspect call
    let bytes = reference.to_bytes();
    let mut bin = OwnedBinary::new(bytes.len()).ok_or(Error::Term(Box::new("Out of memory")))?;
    bin.as_mut_slice().copy_from_slice(&bytes);
    Ok(Binary::from_owned(bin, env))
}

/// Takes a vec of Atoms and transforms them into a vec of ExecutionProvider Enums
pub fn map_eps(env: rustler::env::Env, eps: Vec<Atom>) -> Vec<ExecutionProvider> {
    eps.iter()
        .map(|e| match &e.to_term(env).atom_to_string().unwrap()[..] {
            CPU => ExecutionProvider::cpu(),
            CUDA => ExecutionProvider::cuda(),
            TENSORRT => ExecutionProvider::tensorrt(),
            ACL => ExecutionProvider::acl(),
            DNNL => ExecutionProvider::dnnl(),
            ONEDNN => ExecutionProvider::onednn(),
            COREML => ExecutionProvider::coreml(),
            DIRECTML => ExecutionProvider::directml(),
            ROCM => ExecutionProvider::rocm(),
            _ => ExecutionProvider::cpu(),
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
