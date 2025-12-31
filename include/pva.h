/*
 * xsimd-asm - Cross-Platform SIMD Assembler
 * 
 * Portable Vector Assembly (PVA) instruction set definitions
 * and compiler API declarations.
 */

#ifndef PVA_H
#define PVA_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    PVA_ARCH_UNKNOWN = 0,
    PVA_ARCH_X86_SSE,
    PVA_ARCH_X86_AVX2,
    PVA_ARCH_X86_AVX512,
    PVA_ARCH_ARM_NEON,
    PVA_ARCH_ARM_SVE,
    PVA_ARCH_RISCV_RVV
} pva_arch_t;

typedef enum {
    // Arithmetic - Float32
    PVA_ADD_F32 = 1, PVA_SUB_F32, PVA_MUL_F32, PVA_DIV_F32,
    // Arithmetic - Float64
    PVA_ADD_F64, PVA_SUB_F64, PVA_MUL_F64, PVA_DIV_F64,
    // Arithmetic - Int32
    PVA_ADD_I32, PVA_SUB_I32, PVA_MUL_I32,
    // Arithmetic - Int16
    PVA_ADD_I16, PVA_SUB_I16, PVA_MUL_I16,

    // Math operations
    PVA_SQRT_F32, PVA_SQRT_F64,         // Square root
    PVA_RSQRT_F32,                       // Reciprocal square root (fast)
    PVA_RCP_F32,                         // Reciprocal (1/x)
    PVA_ABS_F32, PVA_ABS_I32,           // Absolute value
    PVA_NEG_F32, PVA_NEG_I32,           // Negate
    PVA_FMA_F32, PVA_FMA_F64,           // Fused multiply-add
    PVA_MIN_F32, PVA_MIN_I32,           // Element-wise minimum
    PVA_MAX_F32, PVA_MAX_I32,           // Element-wise maximum
    PVA_CLAMP_F32,                       // Clamp between min/max

    // Memory operations
    PVA_LOAD_F32, PVA_STORE_F32,
    PVA_LOAD_F64, PVA_STORE_F64,
    PVA_LOAD_I32, PVA_STORE_I32,
    PVA_GATHER_F32, PVA_SCATTER_F32,    // Indexed load/store

    // Comparisons - Float32
    PVA_CMP_LT_F32, PVA_CMP_LE_F32,     // Less than, less equal
    PVA_CMP_GT_F32, PVA_CMP_GE_F32,     // Greater than, greater equal
    PVA_CMP_EQ_F32, PVA_CMP_NE_F32,     // Equal, not equal
    // Comparisons - Int32
    PVA_CMP_LT_I32, PVA_CMP_GT_I32, PVA_CMP_EQ_I32,

    // Mask/Logic operations
    PVA_AND_MASK, PVA_OR_MASK, PVA_XOR_MASK, PVA_NOT_MASK,
    
    // Bitwise operations
    PVA_AND_I32, PVA_OR_I32, PVA_XOR_I32, PVA_NOT_I32,
    PVA_SHL_I32, PVA_SHR_I32, PVA_SAR_I32,  // Shift left, logical right, arithmetic right

    // Horizontal reductions
    PVA_HADD_F32,                        // Horizontal add (sum all elements)
    PVA_HMIN_F32, PVA_HMAX_F32,         // Horizontal min/max

    // Data movement
    PVA_BROADCAST_F32, PVA_BROADCAST_I32, // Broadcast scalar to all lanes
    PVA_SHUFFLE,                          // Permute elements
    PVA_BLEND,                            // Blend based on mask
    PVA_MOV,                              // Move register to register

    // Type conversion
    PVA_CVT_F32_I32,                     // Int32 to Float32
    PVA_CVT_I32_F32,                     // Float32 to Int32 (truncate)
    PVA_CVT_F64_F32,                     // Float32 to Float64
    PVA_CVT_F32_F64,                     // Float64 to Float32

    // Initialization
    PVA_SETZERO,
    PVA_SETONE,                          // Set all bits to 1
    PVA_SET1_F32, PVA_SET1_I32,         // Set all elements to immediate

    // Control flow
    PVA_LABEL,                           // Define label
    PVA_JMP,                             // Unconditional jump
    PVA_JMP_IF,                          // Conditional jump (if mask true)
    PVA_LOOP_BEGIN, PVA_LOOP_END,       // Loop markers
    PVA_CALL, PVA_RET,                  // Function call/return

    PVA_NOP,
    PVA_UNKNOWN                          // Unknown/invalid opcode
} pva_opcode_t;

// Data type for instruction operands
typedef enum {
    PVA_TYPE_F32 = 0,
    PVA_TYPE_F64,
    PVA_TYPE_I32,
    PVA_TYPE_I16,
    PVA_TYPE_I8,
    PVA_TYPE_MASK
} pva_dtype_t;

typedef struct {
    pva_opcode_t op;
    uint8_t dst, src1, src2, src3;      // Added src3 for FMA
    uint32_t imm;
    int mask_reg;                        // Mask register for predication (-1 = no mask)
    char* label;                         // For jumps and labels
} pva_instr_t;

typedef struct {
    char* name;
    size_t offset;                       // Byte offset in generated code
} pva_label_t;

typedef struct {
    pva_instr_t* code;
    size_t size, capacity;
    pva_arch_t arch;
    int vec_width_bytes;
    char* filename;
    
    // Label table for jumps
    pva_label_t* labels;
    size_t label_count, label_capacity;
    
    // Generated code size (set by backends)
    size_t code_size;
} pva_module_t;

pva_arch_t pva_detect_arch(int* vec_width_bytes);
pva_module_t* pva_parse_file(const char* filename);
void pva_optimize(pva_module_t* mod);
size_t pva_emit_x86(pva_module_t* mod, uint8_t* buffer);
size_t pva_emit_arm(pva_module_t* mod, uint8_t* buffer);
size_t pva_emit_riscv(pva_module_t* mod, uint8_t* buffer);
void pva_free(pva_module_t* mod);

#endif