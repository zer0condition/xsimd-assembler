#include "pva.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define MAX_REGS_SSE 16
#define MAX_REGS_AVX2 16
#define MAX_REGS_AVX512 32
#define FULL_MASK 0xFFFF

static void write_bytes(uint8_t** buf, const uint8_t* data, size_t len) {
    memcpy(*buf, data, len);
    *buf += len;
}

// emit EVEX prefix for AVX512 instructions with register bits and mask
static void emit_evex_prefix(uint8_t** pbuf, uint8_t p0, uint8_t p1, uint8_t p2,
                             uint8_t r, uint8_t x, uint8_t b, uint8_t r2,
                             uint8_t mask, uint8_t zeroing, uint8_t vector_length) {
    uint8_t evex[4];
    evex[0] = 0x62;
    evex[1] = ((~r & 1) << 7) | ((~x & 1) << 6) | ((~b & 1) << 5) | (p0 & 0x1F);
    evex[2] = ((~r2 & 1) << 7) | (p1 & 0x7F);
    evex[3] = ((mask & 0xF) & 0x0F)         // mask in bits 0-3
              | ((zeroing & 1) << 4)        // zeroing/merging bit
              | ((vector_length & 3) << 5) // vector length bits
              | 0x08;                      // EVEX bit

    write_bytes(pbuf, evex, 4);
}

// emit ModRM byte for register encoding
static void emit_modrm(uint8_t** pbuf, uint8_t reg, uint8_t rm) {
    uint8_t modrm = 0xC0 | ((reg & 0x7) << 3) | (rm & 0x7);
    write_bytes(pbuf, &modrm, 1);
}

// helper to encode prefixes and opcode for AVX2 instructions
static void emit_avx2_instr(uint8_t** pbuf, uint8_t opcode, uint8_t dst, uint8_t src1, uint8_t src2) {
    // VEX prefix: 0xC5 0xF4 fixed for this instruction variant
    uint8_t vex[2] = {0xC5, 0xF4};
    write_bytes(pbuf, vex, 2);

    write_bytes(pbuf, &opcode, 1);

    uint8_t modrm = 0xC0 | ((src2 & 0x7) << 3) | (dst & 0x7);
    write_bytes(pbuf, &modrm, 1);
}

static void emit_sse_addps(uint8_t** pbuf) {
    uint8_t instr[3] = {0x0F, 0x58, 0xC1};
    write_bytes(pbuf, instr, 3);
}

static void emit_sse_subps(uint8_t** pbuf) {
    uint8_t instr[3] = {0x0F, 0x5C, 0xC1};
    write_bytes(pbuf, instr, 3);
}

static void emit_sse_mulps(uint8_t** pbuf) {
    uint8_t instr[3] = {0x0F, 0x59, 0xC1};
    write_bytes(pbuf, instr, 3);
}

static void emit_sse_divps(uint8_t** pbuf) {
    uint8_t instr[3] = {0x0F, 0x5E, 0xC1};
    write_bytes(pbuf, instr, 3);
}

static void emit_sse_setzero(uint8_t** pbuf) {
    uint8_t instr[3] = {0x0F, 0x57, 0xC0};
    write_bytes(pbuf, instr, 3);
}

static void emit_avx512_instr(uint8_t** pbuf, uint8_t opcode,
                              uint8_t dst, uint8_t src1, uint8_t src2,
                              uint8_t mask) {
    uint8_t r = (dst >> 3) & 1;
    uint8_t x = 0;
    uint8_t b = (src2 >> 3) & 1;
    uint8_t r2 = (src1 >> 4) & 1;

    uint8_t vector_length = 2; 
    uint8_t zeroing = 0;      

    emit_evex_prefix(pbuf, 0x7D, 0x48, opcode, r, x, b, r2, mask, zeroing, vector_length);
    write_bytes(pbuf, &opcode, 1);
    emit_modrm(pbuf, dst & 7, src2 & 7);
}

// emit AVX512 load/store using EVEX + opcode + ModRM
static void emit_avx512_load(uint8_t** pbuf, uint8_t dst, uint8_t base_reg, uint8_t mask) {
    uint8_t r = (dst >> 3) & 1;
    uint8_t x = 0;
    uint8_t b = (base_reg >> 3) & 1;
    uint8_t r2 = 0;

    uint8_t vector_length = 2;
    uint8_t zeroing = 0;

    emit_evex_prefix(pbuf, 0x7D, 0x48, 0x10, r, x, b, r2, mask, zeroing, vector_length);

    uint8_t opcode = 0x10;
    write_bytes(pbuf, &opcode, 1);

    uint8_t modrm = (dst & 7) << 3 | (base_reg & 7);
    write_bytes(pbuf, &modrm, 1);
}

static void emit_avx512_store(uint8_t** pbuf, uint8_t src, uint8_t base_reg, uint8_t mask) {
    uint8_t r = (src >> 3) & 1;
    uint8_t x = 0;
    uint8_t b = (base_reg >> 3) & 1;
    uint8_t r2 = 0;

    uint8_t vector_length = 2;
    uint8_t zeroing = 0;

    emit_evex_prefix(pbuf, 0x7D, 0x48, 0x11, r, x, b, r2, mask, zeroing, vector_length);

    uint8_t opcode = 0x11;
    write_bytes(pbuf, &opcode, 1);

    uint8_t modrm = (src & 7) << 3 | (base_reg & 7);
    write_bytes(pbuf, &modrm, 1);
}

// for AVX2 and SSE, we do not handle masked loads/stores - add later 

static void emit_prologue(uint8_t** pbuf) {
    uint8_t prologue[] = {
        0x55,                   // push rbp
        0x48, 0x89, 0xe5,       // mov rbp, rsp
        0x48, 0x83, 0xec, 0x20  // sub rsp, 32
    };
    write_bytes(pbuf, prologue, sizeof(prologue));
}

static void emit_epilogue(uint8_t** pbuf) {
    uint8_t epilogue[] = {
        0x48, 0x89, 0xec,       // mov rsp, rbp
        0x5d,                   // pop rbp
        0xc3                    // ret
    };
    write_bytes(pbuf, epilogue, sizeof(epilogue));
}

void pva_emit_x86(pva_module_t* mod, uint8_t* buffer) {
    if (!mod || !buffer) return;

    uint8_t* ptr = buffer;
    memset(buffer, 0x90, 8192); // fill with NOPs

    printf("[codegen] generating x86 code for %zu instructions\n", mod->size);
    printf("[codegen] target vector width: %d bytes\n", mod->vec_width_bytes);

    emit_prologue(&ptr);

    for (size_t i = 0; i < mod->size; i++) {
        pva_instr_t* instr = &mod->code[i];

        switch (instr->op) {
            case PVA_ADD_F32:
                if (mod->vec_width_bytes == 64) { // AVX512
                    emit_avx512_instr(&ptr, 0x58, instr->dst, instr->src1, instr->src2, FULL_MASK);
                } else if (mod->vec_width_bytes == 32) { // AVX2/AVX
                    emit_avx2_instr(&ptr, 0x58, instr->dst, instr->src1, instr->src2);
                } else { // SSE
                    emit_sse_addps(&ptr);
                }
                break;

            case PVA_SUB_F32:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_instr(&ptr, 0x5C, instr->dst, instr->src1, instr->src2, FULL_MASK);
                } else if (mod->vec_width_bytes == 32) {
                    emit_avx2_instr(&ptr, 0x5C, instr->dst, instr->src1, instr->src2);
                } else {
                    emit_sse_subps(&ptr);
                }
                break;

            case PVA_MUL_F32:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_instr(&ptr, 0x59, instr->dst, instr->src1, instr->src2, FULL_MASK);
                } else if (mod->vec_width_bytes == 32) {
                    emit_avx2_instr(&ptr, 0x59, instr->dst, instr->src1, instr->src2);
                } else {
                    emit_sse_mulps(&ptr);
                }
                break;

            case PVA_DIV_F32:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_instr(&ptr, 0x5E, instr->dst, instr->src1, instr->src2, FULL_MASK);
                } else if (mod->vec_width_bytes == 32) {
                    emit_avx2_instr(&ptr, 0x5E, instr->dst, instr->src1, instr->src2);
                } else {
                    // SSE divps - you can implement similarly
                }
                break;

            case PVA_LOAD_F32:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_load(&ptr, instr->dst, instr->src1, FULL_MASK);
                } else if (mod->vec_width_bytes == 32 || mod->vec_width_bytes == 16) {
                    // simplified: Use movaps xmm registers with fixed offsets.
                    // extend to dynamic addressing lats
                    // ex: opcode movaps xmm0, [rsi]
                    uint8_t opcode[] = {0x0F, 0x28, 0x06}; // SSE movaps xmm0, [rsi]
                    write_bytes(&ptr, opcode, 3);
                }
                break;

            case PVA_STORE_F32:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_store(&ptr, instr->src1, instr->dst, FULL_MASK);
                } else if (mod->vec_width_bytes == 32 || mod->vec_width_bytes == 16) {
                    // simplified
                    uint8_t opcode[] = {0x0F, 0x29, 0x06}; // SSE movaps [rsi], xmm0
                    write_bytes(&ptr, opcode, 3);
                }
                break;

            case PVA_SETZERO:
                if (mod->vec_width_bytes == 64) {
                    uint8_t setzero[] = {0x62, 0xf2, 0x7d, 0x48, 0x57, 0xc0}; // vxorps zmm0,zmm0,zmm0
                    write_bytes(&ptr, setzero, 6);
                } else if (mod->vec_width_bytes == 32) {
                    uint8_t setzero[] = {0xc5, 0xf4, 0x57, 0xc0}; // vxorps ymm0,ymm0,ymm0
                    write_bytes(&ptr, setzero, 4);
                } else {
                    emit_sse_setzero(&ptr);
                }
                break;

            default:
                // ignore unsupported instructions for now
                break;
        }
    }

    emit_epilogue(&ptr);

    printf("[codegen] generated %ld bytes of code\n", ptr - buffer);
}
