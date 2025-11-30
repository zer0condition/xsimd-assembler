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
    PVA_ADD_F32 = 1, PVA_SUB_F32, PVA_MUL_F32, PVA_DIV_F32,
    PVA_LOAD_F32, PVA_STORE_F32,
    PVA_CMP_LT_F32, PVA_CMP_EQ_F32,
    PVA_AND_MASK, PVA_OR_MASK,
    PVA_SETZERO, PVA_LOOP_BEGIN, PVA_LOOP_END,
    PVA_NOP
} pva_opcode_t;

typedef struct {
    pva_opcode_t op;
    uint8_t dst, src1, src2;
    uint32_t imm;
    int mask_reg;
} pva_instr_t;

typedef struct {
    pva_instr_t* code;
    size_t size, capacity;
    pva_arch_t arch;
    int vec_width_bytes;
    char* filename;
} pva_module_t;

pva_arch_t pva_detect_arch(int* vec_width_bytes);
pva_module_t* pva_parse_file(const char* filename);
void pva_optimize(pva_module_t* mod);
void pva_emit_x86(pva_module_t* mod, uint8_t* buffer);
void pva_emit_arm(pva_module_t* mod, uint8_t* buffer);
void pva_emit_riscv(pva_module_t* mod, uint8_t* buffer);
void pva_free(pva_module_t* mod);

#endif