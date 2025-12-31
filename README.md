# xsimd-asm

Portable vector assembler. Compiles a unified instruction set to x86 AVX2/AVX-512, ARM NEON, or RISC-V RVV.

## Status

| Backend | Status |
|---------|--------|
| x86 AVX2 | Tested, working |
| x86 AVX-512 | Partial, untested |
| ARM NEON | Generates code, not tested on hardware |
| RISC-V RVV | Generates code, not tested on hardware |

## Build

```bash
make
./xsimd-asm input.pva -o output.bin
```

## Test (x86 only)

```bash
./xsimd-asm examples/comprehensive_x86_test.pva -o test.bin
./tests/test_runner test.bin
```

136 instructions tested, all pass on x86 AVX2. Sample output:

```
ADD_F32 : [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0]
MUL_I32 : [50, 200, 450, 800, 1250, 1800, 2450, 3200]
FMA     : [4.5, 11.0, 20.5, 33.0, 48.5, 67.0, 88.5, 113.0]
```

## Instruction Set

**Arithmetic:** vadd/vsub/vmul/vdiv for f32, f64, i32, i16  
**Math:** vsqrt, vrsqrt, vrcp, vabs, vneg, vmin, vmax  
**Comparisons:** vcmplt, vcmple, vcmpgt, vcmpge, vcmpeq, vcmpne  
**Bitwise:** vand, vor, vxor, vnot  
**Shifts:** vshl, vshr, vsar  
**FMA:** vfma rd, rs1, rs2, rs3  
**Memory:** vload, vstore (with optional offset)  
**Conversions:** vcvt.f32.i32, vcvt.i32.f32  
**Other:** vmov, vbroadcast, vzero, vone, vshuffle, vblend

## Registers

| PVA | x86-64 | ARM64 | Purpose |
|-----|--------|-------|---------|
| r0-r9 | ymm0-ymm9 | v0-v9 | Vector data |
| r10 | rdi | x0 | Input ptr |
| r11 | rsi | x1 | Output ptr |
| r12-r15 | rdx,rcx,r8,r9 | x2-x5 | Additional ptrs |

## Example

```asm
vload.f32 r0, [r10]
vload.f32 r1, [r10+#32]
vadd.f32 r2, r0, r1
vsub.f32 r3, r0, r1
vmul.f32 r4, r2, r3
vstore.f32 [r11], r4
ret
```

## Known Issues

- ARM/RISC-V backends generate code but haven't been tested on real hardware
- AVX-512 is only partially implemented
- Horizontal ops only work within 128-bit lanes (AVX2 limitation)
- No gather/scatter or masked operations

## License

MIT
