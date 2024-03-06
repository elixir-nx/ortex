pub mod ortex_atoms {
    rustler::atoms! {
        // Tensor types available
        s8, s16, s32, s64,
        u8, u16, u32, u64,
        f16, f32, f64,
        bf16,
        c64, c128,
        s, u, f, bf, c,
        // Execution provider atoms
        cpu, cuda, tensorrt, acl, dnnl,
        onednn, coreml, directml, rocm
    }
}
