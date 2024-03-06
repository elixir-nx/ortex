//! # Ortex
//! Rust bindings between [ONNX Runtime](https://github.com/microsoft/onnxruntime) and
//! Erlang/Elixir using [Ort](https://docs.rs/ort) and [Rustler](https://docs.rs/rustler).
//! These are only meant to be accessed via the NIF interface provided by Rustler and not
//! directly.

mod constants;
mod execution_providers;
mod model;
mod tensor;
mod utils;

use execution_providers::{
    OrtexCPUExecutionProvider, OrtexCUDAExecutionProvider, OrtexExecutionProvider,
    OrtexTensorRTExecutionProvider,
};
use model::OrtexModel;
use tensor::OrtexTensor;

use rustler::resource::ResourceArc;
use rustler::types::Binary;
use rustler::{Atom, Env, NifResult, Term};

#[rustler::nif(schedule = "DirtyIo")]
fn init(
    model_path: String,
    eps: Vec<ResourceArc<OrtexExecutionProvider>>,
    opt: i32,
) -> NifResult<ResourceArc<model::OrtexModel>> {
    Ok(ResourceArc::new(
        model::init(model_path, eps, opt)
            .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
    ))
}

#[rustler::nif]
fn show_session(
    model: ResourceArc<model::OrtexModel>,
) -> NifResult<(
    Vec<(String, String, Option<Vec<i64>>)>,
    Vec<(String, String, Option<Vec<i64>>)>,
)> {
    Ok(model::show(model))
}

#[rustler::nif(schedule = "DirtyIo")]
fn run(
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> NifResult<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>> {
    model::run(model, inputs).map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn from_binary(bin: Binary, shape: Term, dtype: Term) -> NifResult<ResourceArc<OrtexTensor>> {
    let shape: Vec<usize> = rustler::types::tuple::get_tuple(shape)?
        .iter()
        .map(|x| -> NifResult<usize> { Ok(x.decode::<usize>())? })
        .collect::<NifResult<Vec<usize>>>()?;
    let (dtype_t, dtype_bits): (Term, usize) = dtype.decode()?;
    let dtype_str = dtype_t.atom_to_string()?;

    utils::from_binary(bin, shape, dtype_str, dtype_bits)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn to_binary<'a>(
    env: Env<'a>,
    reference: ResourceArc<OrtexTensor>,
    bits: usize,
    limit: usize,
) -> NifResult<Binary<'a>> {
    utils::to_binary(env, reference, bits, limit)
}

#[rustler::nif]
pub fn slice<'a>(
    tensor: ResourceArc<OrtexTensor>,
    start_indicies: Vec<isize>,
    lengths: Vec<isize>,
    strides: Vec<isize>,
) -> NifResult<ResourceArc<OrtexTensor>> {
    Ok(ResourceArc::new(tensor.slice(
        start_indicies,
        lengths,
        strides,
    )))
}

#[rustler::nif]
pub fn reshape<'a>(
    tensor: ResourceArc<OrtexTensor>,
    shape: Vec<usize>,
) -> NifResult<ResourceArc<OrtexTensor>> {
    Ok(ResourceArc::new(tensor.reshape(shape)?))
}

#[rustler::nif]
pub fn concatenate<'a>(
    tensors: Vec<ResourceArc<OrtexTensor>>,
    dtype: Term,
    axis: i32,
) -> NifResult<ResourceArc<OrtexTensor>> {
    let (dtype_t, dtype_bits): (Term, usize) = dtype.decode()?;
    let dtype_str = dtype_t.atom_to_string()?;
    let concatted = tensor::concatenate(tensors, (&dtype_str, dtype_bits), axis as usize);
    Ok(ResourceArc::new(concatted))
}

rustler::init!(
    "Elixir.Ortex.Native",
    [
        run,
        init,
        from_binary,
        to_binary,
        show_session,
        slice,
        reshape,
        concatenate,
        execution_providers::make_cpu_ep,
        execution_providers::make_cuda_ep,
        execution_providers::make_tensorrt_ep,
    ],
    load = |env: Env, _| {
        rustler::resource!(OrtexModel, env);
        rustler::resource!(OrtexTensor, env);
        rustler::resource!(OrtexExecutionProvider, env);
        rustler::resource!(OrtexCPUExecutionProvider, env);
        rustler::resource!(OrtexCUDAExecutionProvider, env);
        rustler::resource!(OrtexTensorRTExecutionProvider, env);
        true
    }
);
