# xsimd-asm

Portable vector assembler. Write once, emit native SIMD for x86 (SSE/AVX2/AVX-512), ARM (NEON), or RISC-V (RVV).

Takes `.pva` files (Portable Vector Assembly) and outputs raw machine code. No runtime, no dependencies, just bytes you can mmap and execute.

## How it works

1. Parser reads PVA syntax into IR
2. Optimizer runs DCE, NOP removal, fusion analysis
3. Backend emits native code (VEX/EVEX for x86, etc.)
4. Output is raw executable bytes

The x86 backend handles all the VEX/EVEX encoding complexity:
- **VEX** (2-byte C5, 3-byte C4) for AVX/AVX2 - R/X/B bits, vvvv fields, 0F/0F38/0F3A maps
- **EVEX** (4-byte 62) for AVX-512 - mask registers k0-k7, zeroing masking, 512-bit vectors

You write `vadd.f32 r0, r1, r2` and get the correct `vaddps` with proper encoding.

## Status

| Backend | Status |
|---------|--------|
| x86 SSE | Working (128-bit, 4 elements) |
| x86 AVX2 | Tested, working (135 instructions verified) |
| x86 AVX-512 | Code generation working, needs hardware testing |
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

### Force specific ISA

```bash
./xsimd-asm input.pva -o out.bin --force-avx512  # Force AVX-512 (512-bit)
./xsimd-asm input.pva -o out.bin --force-avx2    # Force AVX2 (256-bit)
./xsimd-asm input.pva -o out.bin --force-sse     # Force SSE (128-bit)
```

## Test (x86 only)

```bash
./xsimd-asm examples/comprehensive_x86_test.pva -o test.bin
./tests/test_runner test.bin
```

135 instructions, 951 bytes generated (AVX2). Results:

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

## AVX-512 Output (Disassembly)

Generated with `--force-avx512`. Requires Skylake-X or newer to execute.

```asm
; AVX-512 uses EVEX prefix (62h) and ZMM registers (512-bit)
vmovaps zmm0,ZMMWORD PTR [rdi]              ; 62 f1 7c 40 28 07
vmovaps zmm1,ZMMWORD PTR [rdi+0x800]        ; 62 f1 7c 40 28 4f 20
vaddps  zmm2,zmm0,zmm1                      ; 62 f1 7c 48 58 d1
vmovaps ZMMWORD PTR [rsi],zmm2              ; 62 f1 7c 40 29 16
vsubps  zmm3,zmm0,zmm1                      ; 62 f1 7c 48 5c d9
vmulps  zmm4,zmm0,zmm1                      ; 62 f1 7c 48 59 e1
vdivps  zmm5,zmm0,zmm1                      ; 62 f1 7c 48 5e e9
vsqrtps zmm2,zmm0                           ; 62 f1 7c 48 51 d0
vminps  zmm3,zmm0,zmm1                      ; 62 f1 7c 48 5d d9
vmaxps  zmm4,zmm0,zmm1                      ; 62 f1 7c 48 5f e1

; ABS: vpternlogd to create 0x7FFFFFFF mask, then vandps
vpternlogd zmm15,zmm15,zmm15,0xff           ; 62 53 05 48 25 ff ff
vpsrld  zmm15,zmm15,0x1                     ; 62 d1 05 48 72 d7 01
vandps  zmm5,zmm0,zmm15                     ; 62 d1 7c 48 54 ef

; NEG: vpternlogd to create 0x80000000 mask, then vxorps  
vpternlogd zmm15,zmm15,zmm15,0xff           ; 62 53 05 48 25 ff ff
vpslld  zmm15,zmm15,0x1f                    ; 62 d1 05 48 72 f7 1f
vxorps  zmm6,zmm0,zmm15                     ; 62 d1 7c 48 57 f7

; Comparisons output to mask register, then expand to vector
vcmpltps k1,zmm0,zmm1                       ; 62 f1 7c 48 c2 c9 01
vpxord  zmm2,zmm2,zmm2                      ; 62 f1 6d 48 ef d2
vpternlogd zmm2{k1},zmm2,zmm2,0xff          ; 62 f3 6d 59 25 d2 ff

; Integer ops use EVEX with 66 prefix
vpaddd  zmm2,zmm0,zmm1                      ; 62 f1 7d 48 fe d1
vpsubd  zmm3,zmm0,zmm1                      ; 62 f1 7d 48 fa d9
vpmulld zmm4,zmm0,zmm1                      ; 62 f2 7d 48 40 e1

; Bitwise - AVX-512 has vpandd/vpord/vpxord
vpandd  zmm5,zmm0,zmm1                      ; 62 f1 7d 48 db e9
vpord   zmm6,zmm0,zmm1                      ; 62 f1 7d 48 eb f1
vpxord  zmm7,zmm0,zmm1                      ; 62 f1 7d 48 ef f9

; NOT uses vpternlogd with 0xFF XOR
vpternlogd zmm8,zmm0,zmm0,0xff              ; 62 73 7d 48 25 c0 ff

; RSQRT/RCP use 14-bit precision AVX-512 versions
vrsqrt14ps zmm5,zmm0                        ; 62 f2 7d 48 4e e8
vrcp14ps zmm6,zmm0                          ; 62 f2 7d 48 4c f0
```

## AVX2 Output (Disassembly)

For comparison, same code compiled for AVX2 (VEX prefix, YMM registers):

```asm
; AVX2 uses VEX prefix (C4/C5) and YMM registers (256-bit)
vmovaps ymm0,YMMWORD PTR [rdi]              ; c4 e1 7c 28 07
vmovaps ymm1,YMMWORD PTR [rdi+0x20]         ; c4 e1 7c 28 4f 20
vaddps  ymm2,ymm0,ymm1                      ; c5 fc 58 d1
vmovaps YMMWORD PTR [rsi],ymm2              ; c4 e1 7c 29 16
vsubps  ymm3,ymm0,ymm1                      ; c5 fc 5c d9
vmulps  ymm4,ymm0,ymm1                      ; c5 fc 59 e1
vdivps  ymm5,ymm0,ymm1                      ; c5 fc 5e e9
vsqrtps ymm2,ymm0                           ; c5 fc 51 d0
vminps  ymm3,ymm0,ymm1                      ; c5 fc 5d d9
vmaxps  ymm4,ymm0,ymm1                      ; c5 fc 5f e1

; ABS: vpcmpeqd for all-ones, vpsrld to make 0x7FFFFFFF
vpcmpeqd ymm15,ymm15,ymm15                  ; c4 41 05 76 ff
vpsrld  ymm15,ymm15,0x1                     ; c4 c1 05 72 d7 01
vandps  ymm5,ymm0,ymm15                     ; c4 c1 7c 54 ef

; Comparisons write to YMM register directly
vcmpltps ymm2,ymm0,ymm1                     ; c4 e1 7c c2 d1 01
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
- AVX-512 generates valid EVEX code but needs hardware verification (Skylake-X or newer)
- AVX-512 masking is used internally (comparisons) but not exposed in PVA syntax
- Horizontal ops work within 128-bit lanes (AVX2 arch limitation)
- No gather/scatter instructions yet
- No user-exposed predicated/masked operations

## Contributing

Help wanted:

- **AVX-512 testing** - Verify on Skylake-X, Ice Lake, or newer with AVX-512 support
- **ARM testing** - Run the test suite on ARM64 hardware with NEON
- **RISC-V testing** - Run tests on RVV-capable hardware
- **New instructions** - Gather/scatter, exposed mask ops, more horizontal reductions
- **Bug fixes** - File issues or send PRs

## License

MIT
