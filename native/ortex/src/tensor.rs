//! Conversions for packing/unpacking `OrtexTensor`s into different types
use ndarray::prelude::*;
use ndarray::{ArrayBase, ArrayView, Data, IxDyn};
use ort::tensor::{DynOrtTensor, FromArray, InputTensor, TensorElementDataType};
use ort::OrtError;
use rustler::resource::ResourceArc;
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

    pub fn reshape(&self, shape: Vec<usize>) -> rustler::NifResult<Self> {
        match self {
            OrtexTensor::s8(y) => {
                Ok(OrtexTensor::s8(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
            OrtexTensor::s16(y) => {
                Ok(OrtexTensor::s16(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
            OrtexTensor::s32(y) => {
                Ok(OrtexTensor::s32(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
            OrtexTensor::s64(y) => {
                Ok(OrtexTensor::s64(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
            OrtexTensor::u8(y) => {
                Ok(OrtexTensor::u8(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
            OrtexTensor::u16(y) => {
                Ok(OrtexTensor::u16(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
            OrtexTensor::u32(y) => {
                Ok(OrtexTensor::u32(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
            OrtexTensor::u64(y) => {
                Ok(OrtexTensor::u64(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
            OrtexTensor::f16(y) => {
                Ok(OrtexTensor::f16(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
            OrtexTensor::bf16(y) => {
                Ok(OrtexTensor::bf16(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
            OrtexTensor::f32(y) => {
                Ok(OrtexTensor::f32(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
            OrtexTensor::f64(y) => {
                Ok(OrtexTensor::f64(y.clone().into_shape(shape).map_err(
                    |e| rustler::Error::Term(Box::new(e.to_string())),
                )?))
            }
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

// Currently only supports concatenating tenors of the same type.
//
// This is a similar structure to the above match clauses, except each function
// in map is more complex and needs to be written out explicitly, see below.
//
// Each fn concatenate_{type} verifies to the compiler that the vec<OrtexTensor>
// all have the same type, and then we can concat easily from there
//
// TODO: make the fn concatenate_{type} a macro?
pub fn concatenate(
    tensors: Vec<ResourceArc<OrtexTensor>>,
    dtype: (&str, usize),
    axis: usize,
) -> OrtexTensor {
    match dtype {
        ("s", 8) => concatenate_s8(tensors, axis),
        ("s", 16) => concatenate_s16(tensors, axis),
        ("s", 32) => concatenate_s32(tensors, axis),
        ("s", 64) => concatenate_s64(tensors, axis),
        ("u", 8) => concatenate_u8(tensors, axis),
        ("u", 16) => concatenate_u16(tensors, axis),
        ("u", 32) => concatenate_u32(tensors, axis),
        ("u", 64) => concatenate_u64(tensors, axis),
        ("f", 16) => concatenate_f16(tensors, axis),
        ("bf", 16) => concatenate_bf16(tensors, axis),
        ("f", 32) => concatenate_f32(tensors, axis),
        ("f", 64) => concatenate_f64(tensors, axis),
        _ => unimplemented!(),
    }
}

// each of the below concatenate_{x} functions are identical except for the
// underlying data-type / OrtexTensor enum
fn concatenate_s8(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    // very hacky way to type coalesce, filter_map using an option
    fn filter_s8(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&i8>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::s8(x) => Some(x.view()),
            _ => None,
        }
    }

    // now all tensors have the same type after filter_map()-ing
    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&i8>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_s8(val)).collect();

    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();

    // because concatenating creates a non-standard data format, we copy the
    // data into a standard format shape. Otherwise, when converting to a
    // binary, the tensor's data is not ordered properly
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::s8(x)
}

fn concatenate_s16(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    fn filter_s16(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&i16>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::s16(x) => Some(x.view()),
            _ => None,
        }
    }

    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&i16>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_s16(val)).collect();

    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::s16(x)
}

fn concatenate_s32(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    fn filter_s32(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&i32>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::s32(x) => Some(x.view()),
            _ => None,
        }
    }
    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&i32>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_s32(val)).collect();

    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::s32(x)
}

fn concatenate_s64(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    fn filter_s64(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&i64>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::s64(x) => Some(x.view()),
            _ => None,
        }
    }

    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&i64>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_s64(val)).collect();
    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::s64(x)
}

fn concatenate_u8(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    fn filter_u8(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&u8>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::u8(x) => Some(x.view()),
            _ => None,
        }
    }

    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&u8>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_u8(val)).collect();

    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::u8(x)
}

fn concatenate_u16(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    fn filter_u16(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&u16>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::u16(x) => Some(x.view()),
            _ => None,
        }
    }

    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&u16>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_u16(val)).collect();

    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::u16(x)
}

fn concatenate_u32(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    fn filter_u32(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&u32>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::u32(x) => Some(x.view()),
            _ => None,
        }
    }

    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&u32>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_u32(val)).collect();

    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::u32(x)
}

fn concatenate_u64(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    fn filter_u64(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&u64>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::u64(x) => Some(x.view()),
            _ => None,
        }
    }

    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&u64>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_u64(val)).collect();

    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::u64(x)
}

fn concatenate_f16(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    fn filter_f16(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&half::f16>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::f16(x) => Some(x.view()),
            _ => None,
        }
    }

    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&half::f16>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_f16(val)).collect();

    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::f16(x)
}

fn concatenate_bf16(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    fn filter_bf16(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&half::bf16>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::bf16(x) => Some(x.view()),
            _ => None,
        }
    }

    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&half::bf16>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_bf16(val)).collect();

    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::bf16(x)
}

fn concatenate_f32(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    fn filter_f32(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&f32>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::f32(x) => Some(x.view()),
            _ => None,
        }
    }

    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&f32>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_f32(val)).collect();

    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::f32(x)
}

fn concatenate_f64(tensors: Vec<ResourceArc<OrtexTensor>>, axis: usize) -> OrtexTensor {
    fn filter_f64(
        of: &OrtexTensor,
    ) -> Option<ArrayBase<ndarray::ViewRepr<&f64>, Dim<ndarray::IxDynImpl>>> {
        match of {
            OrtexTensor::f64(x) => Some(x.view()),
            _ => None,
        }
    }

    let tensors: Vec<ArrayBase<ndarray::ViewRepr<&f64>, Dim<ndarray::IxDynImpl>>> =
        tensors.iter().filter_map(|val| filter_f64(val)).collect();

    let x = ndarray::concatenate(Axis(axis), &tensors).unwrap();
    let x = Array::from_shape_vec(x.raw_dim(), x.iter().cloned().collect()).unwrap();
    OrtexTensor::f64(x)
}
