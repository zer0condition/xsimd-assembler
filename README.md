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

Generated with `--force-avx512`. 135 instructions, 1155 bytes. Requires Skylake-X or newer to execute.

```asm
; === FUNCTION PROLOGUE ===
push   rbp
mov    rbp,rsp
sub    rsp,0x20

; === F32 ARITHMETIC ===
vmovaps zmm0,ZMMWORD PTR [rdi]           ; load input A
vmovaps zmm1,ZMMWORD PTR [rdi+0x800]     ; load input B (offset scaled for 512-bit)
vaddps  zmm2,zmm0,zmm1                   ; ADD
vmovaps ZMMWORD PTR [rsi],zmm2           ; store result
vsubps  zmm3,zmm0,zmm1                   ; SUB
vmovaps ZMMWORD PTR [rsi+0x800],zmm3
vmulps  zmm4,zmm0,zmm1                   ; MUL
vmovaps ZMMWORD PTR [rsi+0x1000],zmm4
vdivps  zmm5,zmm0,zmm1                   ; DIV
vmovaps ZMMWORD PTR [rsi+0x1800],zmm5

; === F32 MATH ===
vmovaps zmm0,ZMMWORD PTR [rdi+0x1000]
vsqrtps zmm2,zmm0                        ; SQRT
vmovaps ZMMWORD PTR [rsi+0x80],zmm2
vmovaps zmm1,ZMMWORD PTR [rdi+0x800]
vminps  zmm3,zmm0,zmm1                   ; MIN
vmovaps ZMMWORD PTR [rsi+0xa0],zmm3
vmaxps  zmm4,zmm0,zmm1                   ; MAX
vmovaps ZMMWORD PTR [rsi+0xc0],zmm4

; === ABS (vpternlogd for all-ones, shift for sign mask) ===
vmovaps zmm0,ZMMWORD PTR [rdi+0x1800]
vpternlogd zmm15,zmm15,zmm15,0xff        ; zmm15 = all 1s
vpsrld  zmm15,zmm15,0x1                  ; zmm15 = 0x7FFFFFFF (clear sign bit)
vandps  zmm5,zmm0,zmm15                  ; ABS = clear sign bits
vmovaps ZMMWORD PTR [rsi+0xe0],zmm5

; === NEG (XOR with sign bit mask) ===
vmovaps zmm0,ZMMWORD PTR [rdi]
vpternlogd zmm15,zmm15,zmm15,0xff        ; zmm15 = all 1s
vpslld  zmm15,zmm15,0x1f                 ; zmm15 = 0x80000000 (sign bit only)
vxorps  zmm6,zmm0,zmm15                  ; NEG = flip sign bits
vmovaps ZMMWORD PTR [rsi+0x100],zmm6

; === COMPARISONS (output to mask k1, expand to vector) ===
vmovaps zmm0,ZMMWORD PTR [rdi]
vmovaps zmm1,ZMMWORD PTR [rdi+0x800]
vcmpltps k1,zmm0,zmm1                    ; compare LT -> k1
vpxord  zmm2,zmm2,zmm2                   ; zero zmm2
vpternlogd zmm2{k1},zmm2,zmm2,0xff       ; set all bits where k1=1
vmovaps ZMMWORD PTR [rsi+0x120],zmm2
vcmpltps k1,zmm1,zmm0                    ; GT (reverse operands)
vpxord  zmm3,zmm3,zmm3
vpternlogd zmm3{k1},zmm3,zmm3,0xff
vmovaps ZMMWORD PTR [rsi+0x140],zmm3
vmovaps zmm4,zmm0
vcmpeqps k1,zmm0,zmm4                    ; EQ
vpxord  zmm5,zmm5,zmm5
vpternlogd zmm5{k1},zmm5,zmm5,0xff
vmovaps ZMMWORD PTR [rsi+0x160],zmm5

; === I32 ARITHMETIC ===
vmovaps zmm0,ZMMWORD PTR [rdi+0x80]
vmovaps zmm1,ZMMWORD PTR [rdi+0xa0]
vpaddd  zmm2,zmm0,zmm1                   ; I32 ADD
vmovaps ZMMWORD PTR [rsi+0x180],zmm2
vpsubd  zmm3,zmm0,zmm1                   ; I32 SUB
vmovaps ZMMWORD PTR [rsi+0x1a0],zmm3
vpmulld zmm4,zmm0,zmm1                   ; I32 MUL
vmovaps ZMMWORD PTR [rsi+0x1c0],zmm4

; === BITWISE (AVX-512 uses vpandd/vpord/vpxord) ===
vpandd  zmm5,zmm0,zmm1                   ; AND
vmovaps ZMMWORD PTR [rsi+0x1e0],zmm5
vpord   zmm6,zmm0,zmm1                   ; OR
vmovaps ZMMWORD PTR [rsi+0x200],zmm6
vpxord  zmm7,zmm0,zmm1                   ; XOR
vmovaps ZMMWORD PTR [rsi+0x220],zmm7
vpternlogd zmm8,zmm0,zmm0,0xff           ; NOT (ternlog trick)
vmovaps ZMMWORD PTR [rsi+0x240],zmm8

; === SHIFTS ===
vmovaps zmm0,ZMMWORD PTR [rdi+0x80]
vmovaps zmm1,ZMMWORD PTR [rdi+0xa0]
vpslld  zmm2,zmm0,0x1                    ; SHL by 1
vmovaps ZMMWORD PTR [rsi+0x260],zmm2
vpsrld  zmm3,zmm0,0x1                    ; SHR by 1
vmovaps ZMMWORD PTR [rsi+0x280],zmm3
vpsrad  zmm4,zmm0,0x1                    ; SAR by 1
vmovaps ZMMWORD PTR [rsi+0x2a0],zmm4

; === I32 MIN/MAX/ABS/NEG ===
vpminsd zmm5,zmm0,zmm1                   ; I32 MIN
vmovaps ZMMWORD PTR [rsi+0x2c0],zmm5
vpmaxsd zmm6,zmm0,zmm1                   ; I32 MAX
vmovaps ZMMWORD PTR [rsi+0x2e0],zmm6
vpabsd  zmm7,zmm0                        ; I32 ABS
vmovaps ZMMWORD PTR [rsi+0x300],zmm7
vpxord  zmm8,zmm8,zmm8                   ; I32 NEG (0 - x)
vpsubd  zmm8,zmm8,zmm0
vmovaps ZMMWORD PTR [rsi+0x320],zmm8

; === I32 COMPARISONS (output to mask registers) ===
vpcmpgtd k2,zmm1,zmm0                    ; I32 GT
vpcmpgtd k3,zmm0,zmm1                    ; I32 LT
vpcmpeqd k4,zmm0,zmm0                    ; I32 EQ

; === FMA ===
vmovaps zmm0,ZMMWORD PTR [rdi]
vmovaps zmm1,ZMMWORD PTR [rdi+0x800]
vmovaps zmm2,ZMMWORD PTR [rdi+0x1000]
vmovaps zmm3,zmm2
vfmadd231ps zmm3,zmm0,zmm1               ; FMA: zmm3 += zmm0 * zmm1
vmovaps ZMMWORD PTR [rsi+0x3a0],zmm3

; === DATA MOVEMENT ===
vxorps  zmm4,zmm4,zmm4                   ; VZERO
vmovaps ZMMWORD PTR [rsi+0x3c0],zmm4
vmovaps zmm5,ZMMWORD PTR [rdi]
vmovaps zmm6,zmm5                        ; VMOV
vmovaps ZMMWORD PTR [rsi+0x3e0],zmm6
vbroadcastss zmm7,xmm5                   ; BROADCAST
vmovaps ZMMWORD PTR [rsi+0x400],zmm7

; === TYPE CONVERSIONS ===
vmovaps zmm0,ZMMWORD PTR [rdi]
vcvtps2dq zmm2,zmm0                      ; F32 -> I32
vmovaps ZMMWORD PTR [rsi+0x420],zmm2
vcvtdq2ps zmm3,zmm2                      ; I32 -> F32
vmovaps ZMMWORD PTR [rsi+0x440],zmm3

; === F64 ARITHMETIC ===
vaddpd  zmm2,zmm0,zmm1                   ; F64 ADD
vmovaps ZMMWORD PTR [rsi+0x460],zmm2
vsubpd  zmm3,zmm0,zmm1                   ; F64 SUB
vmovaps ZMMWORD PTR [rsi+0x480],zmm3
vmulpd  zmm4,zmm0,zmm1                   ; F64 MUL
vmovaps ZMMWORD PTR [rsi+0x4a0],zmm4
vdivpd  zmm5,zmm0,zmm1                   ; F64 DIV
vmovaps ZMMWORD PTR [rsi+0x4c0],zmm5
vsqrtpd zmm6,zmm0                        ; F64 SQRT
vmovaps ZMMWORD PTR [rsi+0x4e0],zmm6

; === I16 ARITHMETIC ===
vmovaps zmm0,ZMMWORD PTR [rdi+0x100]
vmovaps zmm1,ZMMWORD PTR [rdi+0x120]
vpaddw  zmm2,zmm0,zmm1                   ; I16 ADD
vmovaps ZMMWORD PTR [rsi+0x500],zmm2
vpsubw  zmm3,zmm0,zmm1                   ; I16 SUB
vmovaps ZMMWORD PTR [rsi+0x520],zmm3
vpmullw zmm4,zmm0,zmm1                   ; I16 MUL
vmovaps ZMMWORD PTR [rsi+0x540],zmm4

; === HORIZONTAL OPS (shuffle + op) ===
vmovaps zmm0,ZMMWORD PTR [rdi]
vshufps zmm2,zmm0,zmm0,0x4e              ; HADD via shuffle+add
vaddps  zmm2,zmm2,zmm0
vmovaps ZMMWORD PTR [rsi+0x560],zmm2
vshufps zmm3,zmm0,zmm0,0x4e              ; HMIN via shuffle+min
vminps  zmm3,zmm3,zmm0
vmovaps ZMMWORD PTR [rsi+0x580],zmm3
vshufps zmm4,zmm0,zmm0,0x4e              ; HMAX via shuffle+max
vmaxps  zmm4,zmm4,zmm0
vmovaps ZMMWORD PTR [rsi+0x5a0],zmm4

; === RSQRT/RCP (14-bit precision versions) ===
vmovaps zmm0,ZMMWORD PTR [rdi+0x1000]
vrsqrt14ps zmm5,zmm0                     ; reciprocal sqrt (14-bit)
vmovaps ZMMWORD PTR [rsi+0x5c0],zmm5
vrcp14ps zmm6,zmm0                       ; reciprocal (14-bit)
vmovaps ZMMWORD PTR [rsi+0x5e0],zmm6

; === SHUFFLE/BLEND ===
vmovaps zmm0,ZMMWORD PTR [rdi]
vmovaps zmm1,ZMMWORD PTR [rdi+0x800]
vshufps zmm2,zmm0,zmm1,0x1
vmovaps ZMMWORD PTR [rsi+0x600],zmm2
vmovaps zmm3,zmm0                        ; blend (copy)
vmovaps ZMMWORD PTR [rsi+0x620],zmm3

; === BROADCAST ===
vmovaps zmm0,ZMMWORD PTR [rdi]
vbroadcastss zmm4,xmm0
vmovaps ZMMWORD PTR [rsi+0x640],zmm4

; === VONE (all 1s) ===
vpternlogd zmm6,zmm6,zmm6,0xff           ; all bits set
vmovaps ZMMWORD PTR [rsi+0x680],zmm6

; === FUNCTION EPILOGUE ===
mov    rsp,rbp
pop    rbp
ret
```

## AVX2 Output (Disassembly)

For comparison, same code compiled for AVX2. 135 instructions, 951 bytes. VEX prefix (C4/C5), YMM registers (256-bit).

```asm
; === FUNCTION PROLOGUE ===
push   rbp
mov    rbp,rsp
sub    rsp,0x20

; === F32 ARITHMETIC ===
vmovaps ymm0,YMMWORD PTR [rdi]           ; load input A
vmovaps ymm1,YMMWORD PTR [rdi+0x20]      ; load input B (32-byte offset for 256-bit)
vaddps  ymm2,ymm0,ymm1                   ; ADD
vmovaps YMMWORD PTR [rsi],ymm2           ; store result
vsubps  ymm3,ymm0,ymm1                   ; SUB
vmovaps YMMWORD PTR [rsi+0x20],ymm3
vmulps  ymm4,ymm0,ymm1                   ; MUL
vmovaps YMMWORD PTR [rsi+0x40],ymm4
vdivps  ymm5,ymm0,ymm1                   ; DIV
vmovaps YMMWORD PTR [rsi+0x60],ymm5

; === F32 MATH ===
vmovaps ymm0,YMMWORD PTR [rdi+0x40]
vsqrtps ymm2,ymm0                        ; SQRT
vmovaps YMMWORD PTR [rsi+0x80],ymm2
vmovaps ymm1,YMMWORD PTR [rdi+0x20]
vminps  ymm3,ymm0,ymm1                   ; MIN
vmovaps YMMWORD PTR [rsi+0xa0],ymm3
vmaxps  ymm4,ymm0,ymm1                   ; MAX
vmovaps YMMWORD PTR [rsi+0xc0],ymm4

; === ABS (vpcmpeqd for all-ones, shift for sign mask) ===
vmovaps ymm0,YMMWORD PTR [rdi+0x60]
vpcmpeqd ymm15,ymm15,ymm15               ; ymm15 = all 1s
vpsrld  ymm15,ymm15,0x1                  ; ymm15 = 0x7FFFFFFF
vandps  ymm5,ymm0,ymm15                  ; ABS = clear sign bits
vmovaps YMMWORD PTR [rsi+0xe0],ymm5

; === NEG (XOR with sign bit mask) ===
vmovaps ymm0,YMMWORD PTR [rdi]
vpcmpeqd ymm15,ymm15,ymm15               ; ymm15 = all 1s
vpslld  ymm15,ymm15,0x1f                 ; ymm15 = 0x80000000
vxorps  ymm6,ymm0,ymm15                  ; NEG = flip sign bits
vmovaps YMMWORD PTR [rsi+0x100],ymm6

; === COMPARISONS (output directly to YMM) ===
vmovaps ymm0,YMMWORD PTR [rdi]
vmovaps ymm1,YMMWORD PTR [rdi+0x20]
vcmpltps ymm2,ymm0,ymm1                  ; compare LT -> ymm2
vmovaps YMMWORD PTR [rsi+0x120],ymm2
vcmpltps ymm3,ymm1,ymm0                  ; GT (reverse operands)
vmovaps YMMWORD PTR [rsi+0x140],ymm3
vmovaps ymm4,ymm0
vcmpeqps ymm5,ymm0,ymm4                  ; EQ
vmovaps YMMWORD PTR [rsi+0x160],ymm5

; === I32 ARITHMETIC ===
vmovaps ymm0,YMMWORD PTR [rdi+0x80]
vmovaps ymm1,YMMWORD PTR [rdi+0xa0]
vpaddd  ymm2,ymm0,ymm1                   ; I32 ADD
vmovaps YMMWORD PTR [rsi+0x180],ymm2
vpsubd  ymm3,ymm0,ymm1                   ; I32 SUB
vmovaps YMMWORD PTR [rsi+0x1a0],ymm3
vpmulld ymm4,ymm0,ymm1                   ; I32 MUL
vmovaps YMMWORD PTR [rsi+0x1c0],ymm4

; === BITWISE ===
vandps  ymm5,ymm0,ymm1                   ; AND
vmovaps YMMWORD PTR [rsi+0x1e0],ymm5
vorps   ymm6,ymm0,ymm1                   ; OR
vmovaps YMMWORD PTR [rsi+0x200],ymm6
vxorps  ymm7,ymm0,ymm1                   ; XOR
vmovaps YMMWORD PTR [rsi+0x220],ymm7
vpcmpeqd ymm8,ymm8,ymm8                  ; NOT (all 1s XOR x)
vpxor   ymm8,ymm8,ymm0
vmovaps YMMWORD PTR [rsi+0x240],ymm8

; === SHIFTS ===
vmovaps ymm0,YMMWORD PTR [rdi+0x80]
vmovaps ymm1,YMMWORD PTR [rdi+0xa0]
vpslld  ymm2,ymm0,0x1                    ; SHL by 1
vmovaps YMMWORD PTR [rsi+0x260],ymm2
vpsrld  ymm3,ymm0,0x1                    ; SHR by 1
vmovaps YMMWORD PTR [rsi+0x280],ymm3
vpsrad  ymm4,ymm0,0x1                    ; SAR by 1
vmovaps YMMWORD PTR [rsi+0x2a0],ymm4

; === I32 MIN/MAX/ABS/NEG ===
vpminsd ymm5,ymm0,ymm1                   ; I32 MIN
vmovaps YMMWORD PTR [rsi+0x2c0],ymm5
vpmaxsd ymm6,ymm0,ymm1                   ; I32 MAX
vmovaps YMMWORD PTR [rsi+0x2e0],ymm6
vpabsd  ymm7,ymm0                        ; I32 ABS
vmovaps YMMWORD PTR [rsi+0x300],ymm7
vpxor   ymm8,ymm8,ymm8                   ; I32 NEG (0 - x)
vpsubd  ymm8,ymm8,ymm0
vmovaps YMMWORD PTR [rsi+0x320],ymm8

; === I32 COMPARISONS ===
vpcmpgtd ymm2,ymm1,ymm0                  ; I32 GT
vmovaps YMMWORD PTR [rsi+0x340],ymm2
vpcmpgtd ymm3,ymm0,ymm1                  ; I32 LT
vmovaps YMMWORD PTR [rsi+0x360],ymm3
vpcmpeqd ymm4,ymm0,ymm0                  ; I32 EQ
vmovaps YMMWORD PTR [rsi+0x380],ymm4

; === FMA ===
vmovaps ymm0,YMMWORD PTR [rdi]
vmovaps ymm1,YMMWORD PTR [rdi+0x20]
vmovaps ymm2,YMMWORD PTR [rdi+0x40]
vmovaps ymm3,ymm2
vfmadd231ps ymm3,ymm0,ymm1               ; FMA: ymm3 += ymm0 * ymm1
vmovaps YMMWORD PTR [rsi+0x3a0],ymm3

; === DATA MOVEMENT ===
vxorps  ymm4,ymm4,ymm4                   ; VZERO
vmovaps YMMWORD PTR [rsi+0x3c0],ymm4
vmovaps ymm5,YMMWORD PTR [rdi]
vmovaps ymm6,ymm5                        ; VMOV
vmovaps YMMWORD PTR [rsi+0x3e0],ymm6
vbroadcastss ymm7,xmm5                   ; BROADCAST
vmovaps YMMWORD PTR [rsi+0x400],ymm7

; === TYPE CONVERSIONS ===
vmovaps ymm0,YMMWORD PTR [rdi]
vcvtps2dq ymm2,ymm0                      ; F32 -> I32
vmovaps YMMWORD PTR [rsi+0x420],ymm2
vcvtdq2ps ymm3,ymm2                      ; I32 -> F32
vmovaps YMMWORD PTR [rsi+0x440],ymm3

; === F64 ARITHMETIC ===
vaddpd  ymm2,ymm0,ymm1                   ; F64 ADD
vmovaps YMMWORD PTR [rsi+0x460],ymm2
vsubpd  ymm3,ymm0,ymm1                   ; F64 SUB
vmovaps YMMWORD PTR [rsi+0x480],ymm3
vmulpd  ymm4,ymm0,ymm1                   ; F64 MUL
vmovaps YMMWORD PTR [rsi+0x4a0],ymm4
vdivpd  ymm5,ymm0,ymm1                   ; F64 DIV
vmovaps YMMWORD PTR [rsi+0x4c0],ymm5
sqrtpd  xmm6,xmm0                        ; F64 SQRT (SSE fallback)
vmovaps YMMWORD PTR [rsi+0x4e0],ymm6

; === I16 ARITHMETIC ===
vmovaps ymm0,YMMWORD PTR [rdi+0x100]
vmovaps ymm1,YMMWORD PTR [rdi+0x120]
vpaddw  ymm2,ymm0,ymm1                   ; I16 ADD
vmovaps YMMWORD PTR [rsi+0x500],ymm2
vpsubw  ymm3,ymm0,ymm1                   ; I16 SUB
vmovaps YMMWORD PTR [rsi+0x520],ymm3
vpmullw ymm4,ymm0,ymm1                   ; I16 MUL
vmovaps YMMWORD PTR [rsi+0x540],ymm4

; === HORIZONTAL OPS ===
vmovaps ymm0,YMMWORD PTR [rdi]
vhaddps ymm2,ymm0,ymm0                   ; HADD (native instruction)
vhaddps ymm2,ymm0,ymm0
vmovaps YMMWORD PTR [rsi+0x560],ymm2
vshufps ymm3,ymm0,ymm0,0x4e              ; HMIN via shuffle+min
vminps  ymm3,ymm3,ymm0
vmovaps YMMWORD PTR [rsi+0x580],ymm3
vshufps ymm4,ymm0,ymm0,0x4e              ; HMAX via shuffle+max
vmaxps  ymm4,ymm4,ymm0
vmovaps YMMWORD PTR [rsi+0x5a0],ymm4

; === RSQRT/RCP ===
vmovaps ymm0,YMMWORD PTR [rdi+0x40]
vrsqrtps ymm5,ymm0                       ; reciprocal sqrt
vmovaps YMMWORD PTR [rsi+0x5c0],ymm5
vrcpps  ymm6,ymm0                        ; reciprocal
vmovaps YMMWORD PTR [rsi+0x5e0],ymm6

; === SHUFFLE/BLEND ===
vmovaps ymm0,YMMWORD PTR [rdi]
vmovaps ymm1,YMMWORD PTR [rdi+0x20]
vshufps ymm2,ymm0,ymm1,0x1
vmovaps YMMWORD PTR [rsi+0x600],ymm2
vmovaps ymm3,ymm0                        ; blend (copy)
vmovaps YMMWORD PTR [rsi+0x620],ymm3

; === BROADCAST ===
vmovaps ymm0,YMMWORD PTR [rdi]
vbroadcastss ymm4,xmm0
vmovaps YMMWORD PTR [rsi+0x640],ymm4

; === MISC ===
vmovaps ymm0,YMMWORD PTR [rdi+0x80]
vmovaps ymm5,ymm0                        ; copy
vmovaps YMMWORD PTR [rsi+0x660],ymm5
vpcmpeqd ymm6,ymm6,ymm6                  ; VONE (all bits set)
vmovaps YMMWORD PTR [rsi+0x680],ymm6

; === FUNCTION EPILOGUE ===
mov    rsp,rbp
pop    rbp
ret
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
