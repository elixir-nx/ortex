//! Conversions for packing/unpacking `OrtexTensor`s into different types
use core::convert::TryFrom;
use ndarray::prelude::*;
use ndarray::{ArrayBase, ArrayView, Data, IxDyn, IxDynImpl, ViewRepr};
use ort::session::SessionInputValue;
use ort::tensor::TensorElementType;
use ort::value::{Tensor, Value, ValueType};
use ort::Error;
use rustler::{Atom, Error as RustlerError, Resource, ResourceArc};

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
    // the bool input is for internal use only.
    // Nx-facing code treats bool tensors as u8 outputs.
    bool(Array<bool, IxDyn>),
}

#[rustler::resource_impl(name = "OrtexTensor")]
impl Resource for OrtexTensor {}

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
            OrtexTensor::bool(y) => y.shape().to_owned(),
        }
    }

    pub fn reshape(&self, shape: Vec<usize>) -> rustler::NifResult<Self> {
        match self {
            OrtexTensor::s8(y) => Ok(OrtexTensor::s8(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::s16(y) => Ok(OrtexTensor::s16(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::s32(y) => Ok(OrtexTensor::s32(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::s64(y) => Ok(OrtexTensor::s64(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::u8(y) => Ok(OrtexTensor::u8(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::u16(y) => Ok(OrtexTensor::u16(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::u32(y) => Ok(OrtexTensor::u32(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::u64(y) => Ok(OrtexTensor::u64(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::f16(y) => Ok(OrtexTensor::f16(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::bf16(y) => Ok(OrtexTensor::bf16(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::f32(y) => Ok(OrtexTensor::f32(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::f64(y) => Ok(OrtexTensor::f64(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
            OrtexTensor::bool(y) => Ok(OrtexTensor::bool(
                y.clone()
                    .into_shape_with_order(shape)
                    .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?,
            )),
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
            OrtexTensor::bool(_) => (ortex_atoms::u(), 8),
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
            OrtexTensor::bool(y) => get_bytes(y),
        };
        contents
    }

    pub fn slice<'a>(
        &'a self,
        start_indicies: Vec<isize>,
        lengths: Vec<isize>,
        strides: Vec<isize>,
    ) -> rustler::NifResult<Self> {
        let rank = match self {
            OrtexTensor::s8(y) => y.ndim(),
            OrtexTensor::s16(y) => y.ndim(),
            OrtexTensor::s32(y) => y.ndim(),
            OrtexTensor::s64(y) => y.ndim(),
            OrtexTensor::u8(y) => y.ndim(),
            OrtexTensor::u16(y) => y.ndim(),
            OrtexTensor::u32(y) => y.ndim(),
            OrtexTensor::u64(y) => y.ndim(),
            OrtexTensor::f16(y) => y.ndim(),
            OrtexTensor::bf16(y) => y.ndim(),
            OrtexTensor::f32(y) => y.ndim(),
            OrtexTensor::f64(y) => y.ndim(),
            OrtexTensor::bool(y) => y.ndim(),
        };

        if start_indicies.len() != rank || lengths.len() != rank || strides.len() != rank {
            return Err(RustlerError::Term(Box::new(format!(
                "Slice arguments must match tensor rank of {}",
                rank
            ))));
        }

        if strides.iter().any(|s| *s == 0) {
            return Err(RustlerError::Term(Box::new(
                "Slice stride must be non-zero".to_string(),
            )));
        }

        let mut slice_specs: Vec<(isize, Option<isize>, isize)> = vec![];
        for ((start_index, length), stride) in start_indicies
            .iter()
            .zip(lengths.iter())
            .zip(strides.iter())
        {
            slice_specs.push((*start_index, Some(*length + *start_index), *stride));
        }
        let sliced = match self {
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
            OrtexTensor::bool(y) => OrtexTensor::bool(slice_array(y, &slice_specs).to_owned()),
        };
        Ok(sliced)
    }

    pub fn to_bool(self) -> Result<OrtexTensor, Error> {
        match self {
            OrtexTensor::u8(y) => {
                let values: Result<Vec<bool>, Error> = y
                    .iter()
                    .map(|x| match x {
                        0 => Ok(false),
                        1 => Ok(true),
                        _ => Err(Error::new(
                            "Tried to convert a u8 tensor to bool, but not every element is 0 or 1",
                        )),
                    })
                    .collect();

                let bool_tensor =
                    Array::from_shape_vec(y.raw_dim(), values?).map_err(|e| Error::new(e.to_string()))?;
                Ok(OrtexTensor::bool(bool_tensor))
            }
            t => Err(Error::new(format!("Can't convert this type {:?} to bool", t.dtype()))),
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

impl TryFrom<&Value> for OrtexTensor {
    type Error = Error;
    fn try_from(e: &Value) -> Result<Self, Self::Error> {
        let dtype = e.dtype();
        let ty = match dtype {
            ValueType::Tensor { ty, .. } => ty,
            _ => return Err(Error::new(format!("Expected tensor output, got {:?}", dtype))),
        };

        let tensor = match *ty {
            TensorElementType::Bfloat16 => {
                OrtexTensor::bf16(e.try_extract_array::<half::bf16>()?.to_owned())
            }
            TensorElementType::Float16 => {
                OrtexTensor::f16(e.try_extract_array::<half::f16>()?.to_owned())
            }
            TensorElementType::Float32 => {
                OrtexTensor::f32(e.try_extract_array::<f32>()?.to_owned())
            }
            TensorElementType::Float64 => {
                OrtexTensor::f64(e.try_extract_array::<f64>()?.to_owned())
            }
            TensorElementType::Uint8 => {
                OrtexTensor::u8(e.try_extract_array::<u8>()?.to_owned())
            }
            TensorElementType::Uint16 => {
                OrtexTensor::u16(e.try_extract_array::<u16>()?.to_owned())
            }
            TensorElementType::Uint32 => {
                OrtexTensor::u32(e.try_extract_array::<u32>()?.to_owned())
            }
            TensorElementType::Uint64 => {
                OrtexTensor::u64(e.try_extract_array::<u64>()?.to_owned())
            }
            TensorElementType::Int8 => {
                OrtexTensor::s8(e.try_extract_array::<i8>()?.to_owned())
            }
            TensorElementType::Int16 => {
                OrtexTensor::s16(e.try_extract_array::<i16>()?.to_owned())
            }
            TensorElementType::Int32 => {
                OrtexTensor::s32(e.try_extract_array::<i32>()?.to_owned())
            }
            TensorElementType::Int64 => {
                OrtexTensor::s64(e.try_extract_array::<i64>()?.to_owned())
            }
            TensorElementType::String => {
                return Err(Error::new("String tensors are not supported"))
            }
            // map the output into u8 space
            TensorElementType::Bool => {
                let nd_array = e.try_extract_array::<bool>()?;
                OrtexTensor::u8(nd_array.mapv(|x| x as u8))
            }
            other => {
                return Err(Error::new(format!(
                    "Tensor element type {:?} is not supported",
                    other
                )))
            }
        };

        Ok(tensor)
    }
}

impl TryFrom<&OrtexTensor> for SessionInputValue<'_> {
    type Error = Error;
    fn try_from(ort_tensor: &OrtexTensor) -> Result<Self, Self::Error> {
        let r: SessionInputValue = match ort_tensor {
            OrtexTensor::s8(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::s16(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::s32(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::s64(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::f16(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::f32(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::f64(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::bf16(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::u8(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::u16(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::u32(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::u64(arr) => Tensor::from_array(arr.to_owned())?.into(),
            OrtexTensor::bool(arr) => Tensor::from_array(arr.to_owned())?.into(),
        };
        Ok(r)
    }
}

impl Clone for OrtexTensor {
    fn clone(&self) -> Self {
        match self {
            OrtexTensor::s8(t) => OrtexTensor::s8(t.clone()),
            OrtexTensor::s16(t) => OrtexTensor::s16(t.clone()),
            OrtexTensor::s32(t) => OrtexTensor::s32(t.clone()),
            OrtexTensor::s64(t) => OrtexTensor::s64(t.clone()),
            OrtexTensor::bf16(t) => OrtexTensor::bf16(t.clone()),
            OrtexTensor::f16(t) => OrtexTensor::f16(t.clone()),
            OrtexTensor::f32(t) => OrtexTensor::f32(t.clone()),
            OrtexTensor::f64(t) => OrtexTensor::f64(t.clone()),
            OrtexTensor::u8(t) => OrtexTensor::u8(t.clone()),
            OrtexTensor::u16(t) => OrtexTensor::u16(t.clone()),
            OrtexTensor::u32(t) => OrtexTensor::u32(t.clone()),
            OrtexTensor::u64(t) => OrtexTensor::u64(t.clone()),
            OrtexTensor::bool(t) => OrtexTensor::bool(t.clone()),
        }
    }
}

// Currently only supports concatenating tenors of the same type.
//
// This is a similar structure to the above match clauses, except each function
// in map is more complex and needs to be written out explicitly. To reduce
// repetition, the concatenate! macro expands that code and makes the necessary
// minor tweaks

macro_rules! concatenate {
    // `typ` is the actual datatype, `ort_tensor_kind` is the OrtexTensor variant
    ($tensors:expr, $axis:expr, $typ:ty, $ort_tensor_kind:ident) => {{
        type ArrayType<'a> = ArrayBase<ViewRepr<&'a $typ>, Dim<IxDynImpl>>;
        fn filter<'a>(tensor: &'a OrtexTensor) -> Option<ArrayType<'a>> {
            match tensor {
                OrtexTensor::$ort_tensor_kind(x) => Some(x.view()),
                _ => None,
            }
        }
        // hack way to type coalesce. Filters out any ndarray's that don't
        // have the desired type
        let input_len = $tensors.len();
        let tensors: Vec<ArrayType> = $tensors
            .iter()
            .filter_map(|tensor| filter(tensor))
            .collect();
        if tensors.len() != input_len {
            return Err(Error::new(
                "Concatenate called with mixed tensor types",
            ));
        }

        let tensors =
            ndarray::concatenate(Axis($axis), &tensors).map_err(|e| Error::new(e.to_string()))?;
        // data is not contiguous after the concatenation above. To decode
        // properly, need to create a new contiguous vector
        let tensors =
            Array::from_shape_vec(tensors.raw_dim(), tensors.iter().cloned().collect())
                .map_err(|e| Error::new(e.to_string()))?;
        Ok(OrtexTensor::$ort_tensor_kind(tensors))
    }};
}

pub fn concatenate(
    tensors: Vec<ResourceArc<OrtexTensor>>,
    dtype: (&str, usize),
    axis: usize,
) -> Result<OrtexTensor, Error> {
    if tensors.is_empty() {
        return Err(Error::new("Concatenate requires at least one tensor"));
    }
    match dtype {
        ("s", 8) => concatenate!(tensors, axis, i8, s8),
        ("s", 16) => concatenate!(tensors, axis, i16, s16),
        ("s", 32) => concatenate!(tensors, axis, i32, s32),
        ("s", 64) => concatenate!(tensors, axis, i64, s64),
        ("u", 8) => concatenate!(tensors, axis, u8, u8),
        ("u", 16) => concatenate!(tensors, axis, u16, u16),
        ("u", 32) => concatenate!(tensors, axis, u32, u32),
        ("u", 64) => concatenate!(tensors, axis, u64, u64),
        ("f", 16) => concatenate!(tensors, axis, half::f16, f16),
        ("bf", 16) => concatenate!(tensors, axis, half::bf16, bf16),
        ("f", 32) => concatenate!(tensors, axis, f32, f32),
        ("f", 64) => concatenate!(tensors, axis, f64, f64),
        _ => Err(Error::new(format!(
            "Unsupported dtype {} with {} bits for concatenate",
            dtype.0, dtype.1
        ))),
    }
}
