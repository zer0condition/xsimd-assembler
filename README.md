# xsimd-asm

**Cross-Platform SIMD Assembler** - Write vector code once, run on any SIMD architecture.

xsimd-asm is a portable vector assembly language and code generator that compiles a unified instruction set to native SIMD code for multiple architectures. Write your vectorized algorithms once and deploy them on x86, ARM, or RISC-V without rewriting architecture-specific intrinsics.

## Features

- **Unified instruction set** - Single assembly syntax for all platforms
- **Multi-architecture support**:
  - x86-64: AVX-512, AVX2, SSE4.2
  - ARM64: SVE, NEON  
  - RISC-V: RVV (Vector Extension)
- **Runtime detection** - Automatically selects best available SIMD extension
- **OS support verification** - Validates that OS has enabled AVX/AVX-512 state saving
- **Built-in optimizer** - Dead code elimination, common subexpression elimination, strength reduction
- **60+ vector instructions** - Arithmetic, math, comparisons, bitwise, shuffles, conversions, and more

## Building

```bash
make
```

## Usage

```bash
./xsimd-asm input.pva -o output.bin
```

### Example

```bash
# Assemble the mandelbrot example
./xsimd-asm examples/mandelbrot.pva -o mandelbrot.bin

# Assemble the vector math example
./xsimd-asm examples/vector_math.pva -o vector_math.bin
```

## Instruction Set

### Arithmetic
| Instruction | Description |
|-------------|-------------|
| `vadd.f32 rd, rs1, rs2` | Vector float32 addition |
| `vsub.f32 rd, rs1, rs2` | Vector float32 subtraction |
| `vmul.f32 rd, rs1, rs2` | Vector float32 multiplication |
| `vdiv.f32 rd, rs1, rs2` | Vector float32 division |
| `vadd.f64 rd, rs1, rs2` | Vector float64 addition |
| `vadd.i32 rd, rs1, rs2` | Vector int32 addition |
| `vfma rd, rs1, rs2, rs3` | Fused multiply-add (rd = rs1 * rs2 + rs3) |

### Math Functions
| Instruction | Description |
|-------------|-------------|
| `vsqrt rd, rs` | Square root |
| `vrsqrt rd, rs` | Reciprocal square root (fast) |
| `vrcp rd, rs` | Reciprocal (1/x) |
| `vabs rd, rs` | Absolute value |
| `vneg rd, rs` | Negate |
| `vmin rd, rs1, rs2` | Element-wise minimum |
| `vmax rd, rs1, rs2` | Element-wise maximum |

### Comparisons
| Instruction | Description |
|-------------|-------------|
| `vcmplt rd, rs1, rs2` | Less than (produces mask) |
| `vcmple rd, rs1, rs2` | Less than or equal |
| `vcmpgt rd, rs1, rs2` | Greater than |
| `vcmpge rd, rs1, rs2` | Greater than or equal |
| `vcmpeq rd, rs1, rs2` | Equal |
| `vcmpne rd, rs1, rs2` | Not equal |

### Bitwise & Shifts
| Instruction | Description |
|-------------|-------------|
| `vand rd, rs1, rs2` | Bitwise AND |
| `vor rd, rs1, rs2` | Bitwise OR |
| `vxor rd, rs1, rs2` | Bitwise XOR |
| `vnot rd, rs` | Bitwise NOT |
| `vshl rd, rs, #imm` | Shift left by immediate |
| `vshr rd, rs, #imm` | Logical shift right |
| `vsar rd, rs, #imm` | Arithmetic shift right |

### Memory
| Instruction | Description |
|-------------|-------------|
| `vload.f32 rd, [rs]` | Load vector from memory |
| `vstore.f32 [rd], rs` | Store vector to memory |
| `vload.f32 rd, [rs+#off]` | Load with offset |
| `vgather rd, [rs], ridx` | Gather (indexed load) |
| `vscatter [rd], rs, ridx` | Scatter (indexed store) |

### Data Movement
| Instruction | Description |
|-------------|-------------|
| `vmov rd, rs` | Move register |
| `vbroadcast rd, rs` | Broadcast scalar to all lanes |
| `vshuffle rd, rs1, rs2, #imm` | Shuffle/permute elements |
| `vblend rd, rs1, rs2, mask` | Blend based on mask |

### Initialization
| Instruction | Description |
|-------------|-------------|
| `vzero rd` | Set all elements to zero |
| `vone rd` | Set all bits to 1 |
| `vset1 rd, #imm` | Broadcast immediate to all lanes |

### Type Conversion
| Instruction | Description |
|-------------|-------------|
| `vcvt.f32.i32 rd, rs` | Convert int32 to float32 |
| `vcvt.i32.f32 rd, rs` | Convert float32 to int32 |
| `vcvt.f64.f32 rd, rs` | Convert float32 to float64 |
| `vcvt.f32.f64 rd, rs` | Convert float64 to float32 |

### Control Flow
| Instruction | Description |
|-------------|-------------|
| `label name:` | Define a label |
| `jmp label` | Unconditional jump |
| `jmp_if mask, label` | Conditional jump |
| `call label` | Function call |
| `ret` | Return |

## Example Program

```asm
; vector_math.pva - Vector arithmetic example

    ; load input vectors
    vload.f32 r2, [r10]
    vload.f32 r3, [r11]
    
    ; compute: result = (a + b) * (a - b)
    vadd.f32 r4, r2, r3     ; r4 = a + b
    vsub.f32 r5, r2, r3     ; r5 = a - b
    vmul.f32 r6, r4, r5     ; r6 = (a+b) * (a-b)
    
    ; store result
    vstore.f32 [r12], r6
    
    ret
```

## Architecture Detection

xsimd-asm automatically detects the best available SIMD extension at runtime:

```
CPU architecture:
  target: x86-64 AVX2
  vector width: 32 bytes (256 bits)
  elems: 8 floats per vector
```

For AVX2 and AVX-512, it verifies OS support by checking XCR0 to ensure the OS has enabled YMM/ZMM state saving.

## Project Structure

```
xsimd-asm/
├── include/
│   └── pva.h           # Main header (types, opcodes, API)
├── src/
│   ├── main.c          # CLI entry point
│   ├── parser.c        # Lexer and parser
│   ├── optimizer.c     # IR optimization passes
│   ├── detect_arch.c   # Runtime CPU detection
│   └── backends/
│       ├── x86.c       # x86 SSE/AVX2/AVX-512 codegen
│       ├── arm.c       # ARM NEON/SVE codegen
│       └── riscv.c     # RISC-V RVV codegen
├── examples/
│   ├── mandelbrot.pva  # Mandelbrot set example
│   └── vector_math.pva # Vector math operations
├── Makefile
└── README.md
```

## Use Cases

- **High-performance computing** - Portable vectorized kernels
- **Image/signal processing** - SIMD-accelerated filters
- **Scientific computing** - Cross-platform numerical algorithms
- **Game engines** - Portable math libraries
- **Machine learning** - Custom SIMD operators
- **Embedded systems** - Consistent code across ARM/RISC-V targets

## License

MIT License
