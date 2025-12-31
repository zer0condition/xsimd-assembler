# xsimd-asm

Portable vector assembler. Write once, emit native SIMD for x86 (AVX2/AVX-512), ARM (NEON), or RISC-V (RVV).

Takes `.pva` files (Portable Vector Assembly) and outputs raw machine code. No runtime, no dependencies, just bytes you can mmap and execute.

## How it works

1. Parser reads PVA syntax into IR
2. Optimizer runs DCE, NOP removal, fusion analysis
3. Backend emits native code (VEX/EVEX for x86, etc.)
4. Output is raw executable bytes

The x86 backend handles all the VEX encoding complexity - 2-byte vs 3-byte prefixes, R/X/B bits, vvvv fields, 66/0F/0F38 maps. You write `vadd.f32 r0, r1, r2` and get the correct `vaddps` with proper encoding.

## Status

| Backend | Status |
|---------|--------|
| x86 AVX2 | Tested, working (136 instructions verified) |
| x86 AVX-512 | Partial, untested |
| ARM NEON | Generates code, not tested on hardware |
| RISC-V RVV | Generates code, not tested on hardware |

## Structure

```
src/
  main.c           - CLI, file I/O
  parser.c         - Lexer + parser, outputs IR
  optimizer.c      - DCE, NOP removal, fusion hints
  detect_arch.c    - CPUID, XCR0 checks for AVX state
  backends/
    x86.c          - SSE/AVX2/AVX-512 codegen
    arm.c          - NEON codegen  
    riscv.c        - RVV codegen
include/
  pva.h            - Opcode enum, IR structs, API
examples/
  comprehensive_x86_test.pva   - Full test (136 ops)
  vector_math.pva              - Basic example
tests/
  test_runner.c    - mmap + execute generated code
```

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

136 instructions, 983 bytes generated. Results:

```
=== F32 Arithmetic ===
ADD  : [1.5000, 3.0000, 4.5000, 6.0000, 7.5000, 9.0000, 10.5000, 12.0000]
SUB  : [0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000]
MUL  : [0.5000, 2.0000, 4.5000, 8.0000, 12.5000, 18.0000, 24.5000, 32.0000]
DIV  : [2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000]

=== F32 Math ===
SQRT : [2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000, 9.0000]
MIN  : [0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000]
MAX  : [4.0000, 9.0000, 16.0000, 25.0000, 36.0000, 49.0000, 64.0000, 81.0000]
ABS  : [1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000]
NEG  : [-1.0000, -2.0000, -3.0000, -4.0000, -5.0000, -6.0000, -7.0000, -8.0000]

=== Integer Ops (I32) ===
ADD  : [15, 30, 45, 60, 75, 90, 105, 120]
SUB  : [5, 10, 15, 20, 25, 30, 35, 40]
MUL  : [50, 200, 450, 800, 1250, 1800, 2450, 3200]

=== Bitwise ===
AND  : [0x00, 0x00, 0x0e, 0x00, 0x10, 0x1c, 0x02, 0x00]
OR   : [0x0f, 0x1e, 0x1f, 0x3c, 0x3b, 0x3e, 0x67, 0x78]
XOR  : [0x0f, 0x1e, 0x11, 0x3c, 0x2b, 0x22, 0x65, 0x78]

=== Shifts ===
SHL  : [20, 40, 60, 80, 100, 120, 140, 160]
SHR  : [5, 10, 15, 20, 25, 30, 35, 40]
SAR  : [5, 10, 15, 20, 25, 30, 35, 40]

=== FMA (a*b+c) ===
FMA  : [4.5000, 11.0000, 20.5000, 33.0000, 48.5000, 67.0000, 88.5000, 113.0000]

=== Conversions ===
F2I  : [1, 2, 3, 4, 5, 6, 7, 8]
I2F  : [1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000]
```

## Instructions

80+ opcodes. Full list in `pva.h`.

```
Arithmetic   vadd, vsub, vmul, vdiv (f32/f64/i32/i16)
Math         vsqrt, vrsqrt, vrcp, vabs, vneg, vmin, vmax
Compare      vcmplt, vcmple, vcmpgt, vcmpge, vcmpeq, vcmpne
Bitwise      vand, vor, vxor, vnot
Shifts       vshl, vshr, vsar (immediate)
FMA          vfma rd, rs1, rs2, rs3  (rd = rs1*rs2 + rs3)
Memory       vload, vstore with [base] or [base+#offset]
Convert      vcvt.f32.i32, vcvt.i32.f32
Misc         vmov, vbroadcast, vzero, vone, vshuffle, vblend
```

## Registers

```
r0-r9    Vector regs (ymm0-9 on x86, v0-9 on ARM)
r10      Input pointer (rdi / x0)
r11      Output pointer (rsi / x1)  
r12-r15  Extra pointers (rdx, rcx, r8, r9 / x2-x5)
```

Generated code follows System V AMD64 calling convention on x86. First arg in rdi, second in rsi, etc.

## Example

```asm
; compute (a+b)*(a-b) for 8 floats
vload.f32 r0, [r10]         ; load a
vload.f32 r1, [r10+#32]     ; load b (32 bytes = 8 floats)
vadd.f32 r2, r0, r1         ; a + b
vsub.f32 r3, r0, r1         ; a - b
vmul.f32 r4, r2, r3         ; (a+b) * (a-b)
vstore.f32 [r11], r4
ret
```

## Limitations

- ARM/RISC-V backends emit code but no hardware testing yet
- AVX-512 partial (basic ops only, no masking)
- Horizontal ops work within 128-bit lanes only (AVX2 arch limitation)
- No gather/scatter
- No predicated/masked operations

## Contributing

Help wanted:

- **ARM testing** - Run the test suite on ARM64 hardware with NEON
- **RISC-V testing** - Run tests on RVV-capable hardware
- **AVX-512 testing** - Verify on Skylake-X or newer
- **New instructions** - Gather/scatter, masked ops, more horizontal reductions
- **Bug fixes** - File issues or send PRs

## License

MIT
