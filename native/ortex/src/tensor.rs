//! Conversions for packing/unpacking `OrtexTensor`s into different types
use ndarray::prelude::*;
use ndarray::{ArrayBase, ArrayView, Data, IxDyn};
use ort::tensor::{DynOrtTensor, FromArray, InputTensor, TensorElementDataType};
use ort::OrtError;
use rustler::Atom;

use crate::constants::ortex_atoms;

#[derive(Debug)]
#[allow(non_camel_case_types)]
/// Enum for wrapping different types to pass back to the BEAM since rustler can't
/// pass type generics back and forth
pub enum OrtexTensor {
    s8(Array<i8, IxDyn>),
    s16(Array<i16, IxDyn>),
    s32(Array<i32, IxDyn>),
    s64(Array<i64, IxDyn>),
    u8(Array<u8, IxDyn>),
    u16(Array<u16, IxDyn>),
    u32(Array<u32, IxDyn>),
    u64(Array<u64, IxDyn>),
    f16(Array<half::f16, IxDyn>),
    bf16(Array<half::bf16, IxDyn>),
    f32(Array<f32, IxDyn>),
    f64(Array<f64, IxDyn>),
}

impl From<&OrtexTensor> for InputTensor {
    fn from(tensor: &OrtexTensor) -> Self {
        match tensor {
            OrtexTensor::s8(y) => InputTensor::from_array(y.clone().into()),
            OrtexTensor::s16(y) => InputTensor::from_array(y.clone().into()),
            OrtexTensor::s32(y) => InputTensor::from_array(y.clone().into()),
            OrtexTensor::s64(y) => InputTensor::from_array(y.clone().into()),
            OrtexTensor::u8(y) => InputTensor::from_array(y.clone().into()),
            OrtexTensor::u16(y) => InputTensor::from_array(y.clone().into()),
            OrtexTensor::u32(y) => InputTensor::from_array(y.clone().into()),
            OrtexTensor::u64(y) => InputTensor::from_array(y.clone().into()),
            OrtexTensor::f16(y) => InputTensor::from_array(y.clone().into()),
            OrtexTensor::bf16(y) => InputTensor::from_array(y.clone().into()),
            OrtexTensor::f32(y) => InputTensor::from_array(y.clone().into()),
            OrtexTensor::f64(y) => InputTensor::from_array(y.clone().into()),
        }
    }
}

impl OrtexTensor {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            OrtexTensor::s8(y) => y.shape().to_owned(),
            OrtexTensor::s16(y) => y.shape().to_owned(),
            OrtexTensor::s32(y) => y.shape().to_owned(),
            OrtexTensor::s64(y) => y.shape().to_owned(),
            OrtexTensor::u8(y) => y.shape().to_owned(),
            OrtexTensor::u16(y) => y.shape().to_owned(),
            OrtexTensor::u32(y) => y.shape().to_owned(),
            OrtexTensor::u64(y) => y.shape().to_owned(),
            OrtexTensor::f16(y) => y.shape().to_owned(),
            OrtexTensor::bf16(y) => y.shape().to_owned(),
            OrtexTensor::f32(y) => y.shape().to_owned(),
            OrtexTensor::f64(y) => y.shape().to_owned(),
        }
    }

    pub fn dtype(&self) -> (Atom, usize) {
        match self {
            OrtexTensor::s8(_) => (ortex_atoms::s(), 8),
            OrtexTensor::s16(_) => (ortex_atoms::s(), 16),
            OrtexTensor::s32(_) => (ortex_atoms::s(), 32),
            OrtexTensor::s64(_) => (ortex_atoms::s(), 64),
            OrtexTensor::u8(_) => (ortex_atoms::u(), 8),
            OrtexTensor::u16(_) => (ortex_atoms::u(), 16),
            OrtexTensor::u32(_) => (ortex_atoms::u(), 32),
            OrtexTensor::u64(_) => (ortex_atoms::u(), 64),
            OrtexTensor::f16(_) => (ortex_atoms::f(), 16),
            OrtexTensor::bf16(_) => (ortex_atoms::bf(), 16),
            OrtexTensor::f32(_) => (ortex_atoms::f(), 32),
            OrtexTensor::f64(_) => (ortex_atoms::f(), 64),
        }
    }

    pub fn to_bytes<'a>(&'a self) -> &'a [u8] {
        let contents: &'a [u8] = match self {
            OrtexTensor::s8(y) => get_bytes(y),
            OrtexTensor::s16(y) => get_bytes(y),
            OrtexTensor::s32(y) => get_bytes(y),
            OrtexTensor::s64(y) => get_bytes(y),
            OrtexTensor::u8(y) => get_bytes(y),
            OrtexTensor::u16(y) => get_bytes(y),
            OrtexTensor::u32(y) => get_bytes(y),
            OrtexTensor::u64(y) => get_bytes(y),
            OrtexTensor::f16(y) => get_bytes(y),
            OrtexTensor::bf16(y) => get_bytes(y),
            OrtexTensor::f32(y) => get_bytes(y),
            OrtexTensor::f64(y) => get_bytes(y),
        };
        contents
    }

    pub fn slice<'a>(
        &'a self,
        start_indicies: Vec<isize>,
        lengths: Vec<isize>,
        strides: Vec<isize>,
    ) -> Self {
        let mut slice_specs: Vec<(isize, Option<isize>, isize)> = vec![];
        for ((start_index, length), stride) in start_indicies
            .iter()
            .zip(lengths.iter())
            .zip(strides.iter())
        {
            slice_specs.push((*start_index, Some(*length + *start_index), *stride));
        }
        match self {
            OrtexTensor::s8(y) => OrtexTensor::s8(slice_array(y, &slice_specs).to_owned()),
            OrtexTensor::s16(y) => OrtexTensor::s16(slice_array(y, &slice_specs).to_owned()),
            OrtexTensor::s32(y) => OrtexTensor::s32(slice_array(y, &slice_specs).to_owned()),
            OrtexTensor::s64(y) => OrtexTensor::s64(slice_array(y, &slice_specs).to_owned()),
            OrtexTensor::u8(y) => OrtexTensor::u8(slice_array(y, &slice_specs).to_owned()),
            OrtexTensor::u16(y) => OrtexTensor::u16(slice_array(y, &slice_specs).to_owned()),
            OrtexTensor::u32(y) => OrtexTensor::u32(slice_array(y, &slice_specs).to_owned()),
            OrtexTensor::u64(y) => OrtexTensor::u64(slice_array(y, &slice_specs).to_owned()),
            OrtexTensor::f16(y) => OrtexTensor::f16(slice_array(y, &slice_specs).to_owned()),
            OrtexTensor::bf16(y) => OrtexTensor::bf16(slice_array(y, &slice_specs).to_owned()),
            OrtexTensor::f32(y) => OrtexTensor::f32(slice_array(y, &slice_specs).to_owned()),
            OrtexTensor::f64(y) => OrtexTensor::f64(slice_array(y, &slice_specs).to_owned()),
        }
    }
}

fn slice_array<'a, T, D>(
    array: &'a Array<T, D>,
    slice_specs: &'a Vec<(isize, Option<isize>, isize)>,
) -> ArrayView<'a, T, D>
where
    D: Dimension,
{
    array.slice_each_axis(|ax: ndarray::AxisDescription| {
        let (start, end, step) = slice_specs[ax.axis.index()];
        ndarray::Slice { start, end, step }
    })
}

fn get_bytes<'a, T>(array: &'a ArrayBase<T, IxDyn>) -> &'a [u8]
where
    T: Data,
{
    let len = array.len();
    let binding = unsafe { std::mem::zeroed() };
    let f = array.get(0).unwrap_or(&binding);
    let size: usize = std::mem::size_of_val(f);
    unsafe { std::slice::from_raw_parts(array.as_ptr() as *const u8, len * size) }
}

impl std::convert::TryFrom<&DynOrtTensor<'_, IxDyn>> for OrtexTensor {
    type Error = OrtError;
    fn try_from(e: &DynOrtTensor<IxDyn>) -> Result<OrtexTensor, Self::Error> {
        let dtype = e.data_type();
        match dtype {
            // TODO: Pull this out into an impl for each OrtexTensor type or some other
            // function to be more DRY
            TensorElementDataType::Float16 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::f16(tensor))
            }
            TensorElementDataType::Bfloat16 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::bf16(tensor))
            }
            TensorElementDataType::Float32 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::f32(tensor))
            }
            TensorElementDataType::Float64 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::f64(tensor))
            }
            TensorElementDataType::Int8 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::s8(tensor))
            }
            TensorElementDataType::Int16 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::s16(tensor))
            }
            TensorElementDataType::Int32 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::s32(tensor))
            }
            TensorElementDataType::Int64 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::s64(tensor))
            }
            TensorElementDataType::Uint8 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::u8(tensor))
            }
            TensorElementDataType::Uint16 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::u16(tensor))
            }
            TensorElementDataType::Uint32 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::u32(tensor))
            }
            TensorElementDataType::Uint64 => {
                let tensor = e.try_extract()?.view().to_owned();
                Ok(OrtexTensor::u64(tensor))
            }
            TensorElementDataType::String | TensorElementDataType::Bool => todo!(),
        }
    }
}
