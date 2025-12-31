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

// Helper for SSE two-operand instructions
static void emit_sse_unary(uint8_t** pbuf, uint8_t op1, uint8_t op2, uint8_t dst, uint8_t src) {
    uint8_t modrm = 0xC0 | ((dst & 0x7) << 3) | (src & 0x7);
    uint8_t instr[3] = {0x0F, op1, modrm};
    if (op2 != 0) {
        uint8_t instr4[4] = {0x0F, op1, op2, modrm};
        write_bytes(pbuf, instr4, 4);
    } else {
        write_bytes(pbuf, instr, 3);
    }
}

// Helper for SSE binary ops with register encoding
static void emit_sse_binop(uint8_t** pbuf, uint8_t op, uint8_t dst, uint8_t src1, uint8_t src2) {
    // For SSE, we need to move src1 to dst first if they differ, then operate with src2
    if (dst != src1) {
        // movaps dst, src1
        uint8_t mov[3] = {0x0F, 0x28, (uint8_t)(0xC0 | ((dst & 7) << 3) | (src1 & 7))};
        write_bytes(pbuf, mov, 3);
    }
    // op dst, src2
    uint8_t instr[3] = {0x0F, op, (uint8_t)(0xC0 | ((dst & 7) << 3) | (src2 & 7))};
    write_bytes(pbuf, instr, 3);
}

size_t pva_emit_x86(pva_module_t* mod, uint8_t* buffer) {
    if (!mod || !buffer) return 0;

    uint8_t* ptr = buffer;
    uint8_t* start = buffer;

    printf("[codegen] generating x86 code for %zu instructions\n", mod->size);
    printf("[codegen] target vector width: %d bytes\n", mod->vec_width_bytes);

    emit_prologue(&ptr);

    for (size_t i = 0; i < mod->size; i++) {
        pva_instr_t* instr = &mod->code[i];

        switch (instr->op) {
            // ========== ARITHMETIC F32 ==========
            case PVA_ADD_F32:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_instr(&ptr, 0x58, instr->dst, instr->src1, instr->src2, FULL_MASK);
                } else if (mod->vec_width_bytes == 32) {
                    emit_avx2_instr(&ptr, 0x58, instr->dst, instr->src1, instr->src2);
                } else {
                    emit_sse_binop(&ptr, 0x58, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_SUB_F32:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_instr(&ptr, 0x5C, instr->dst, instr->src1, instr->src2, FULL_MASK);
                } else if (mod->vec_width_bytes == 32) {
                    emit_avx2_instr(&ptr, 0x5C, instr->dst, instr->src1, instr->src2);
                } else {
                    emit_sse_binop(&ptr, 0x5C, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_MUL_F32:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_instr(&ptr, 0x59, instr->dst, instr->src1, instr->src2, FULL_MASK);
                } else if (mod->vec_width_bytes == 32) {
                    emit_avx2_instr(&ptr, 0x59, instr->dst, instr->src1, instr->src2);
                } else {
                    emit_sse_binop(&ptr, 0x59, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_DIV_F32:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_instr(&ptr, 0x5E, instr->dst, instr->src1, instr->src2, FULL_MASK);
                } else if (mod->vec_width_bytes == 32) {
                    emit_avx2_instr(&ptr, 0x5E, instr->dst, instr->src1, instr->src2);
                } else {
                    emit_sse_binop(&ptr, 0x5E, instr->dst, instr->src1, instr->src2);
                }
                break;

            // ========== ARITHMETIC F64 ==========
            case PVA_ADD_F64:
                if (mod->vec_width_bytes >= 32) {
                    // vaddpd ymm/zmm
                    emit_avx2_instr(&ptr, 0x58, instr->dst, instr->src1, instr->src2);
                } else {
                    // addpd xmm (0x66 prefix)
                    uint8_t prefix = 0x66;
                    write_bytes(&ptr, &prefix, 1);
                    emit_sse_binop(&ptr, 0x58, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_SUB_F64:
                if (mod->vec_width_bytes >= 32) {
                    emit_avx2_instr(&ptr, 0x5C, instr->dst, instr->src1, instr->src2);
                } else {
                    uint8_t prefix = 0x66;
                    write_bytes(&ptr, &prefix, 1);
                    emit_sse_binop(&ptr, 0x5C, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_MUL_F64:
                if (mod->vec_width_bytes >= 32) {
                    emit_avx2_instr(&ptr, 0x59, instr->dst, instr->src1, instr->src2);
                } else {
                    uint8_t prefix = 0x66;
                    write_bytes(&ptr, &prefix, 1);
                    emit_sse_binop(&ptr, 0x59, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_DIV_F64:
                if (mod->vec_width_bytes >= 32) {
                    emit_avx2_instr(&ptr, 0x5E, instr->dst, instr->src1, instr->src2);
                } else {
                    uint8_t prefix = 0x66;
                    write_bytes(&ptr, &prefix, 1);
                    emit_sse_binop(&ptr, 0x5E, instr->dst, instr->src1, instr->src2);
                }
                break;

            // ========== ARITHMETIC I32 ==========
            case PVA_ADD_I32: {
                // paddd xmm, xmm (0x66 0x0F 0xFE)
                uint8_t instr_bytes[] = {0x66, 0x0F, 0xFE, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            case PVA_SUB_I32: {
                // psubd xmm, xmm (0x66 0x0F 0xFA)
                uint8_t instr_bytes[] = {0x66, 0x0F, 0xFA, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            case PVA_MUL_I32: {
                // pmulld xmm, xmm (0x66 0x0F 0x38 0x40)
                uint8_t instr_bytes[] = {0x66, 0x0F, 0x38, 0x40, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                write_bytes(&ptr, instr_bytes, 5);
                break;
            }

            // ========== MATH OPERATIONS ==========
            case PVA_SQRT_F32: {
                // sqrtps xmm, xmm (0x0F 0x51)
                uint8_t instr_bytes[] = {0x0F, 0x51, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                write_bytes(&ptr, instr_bytes, 3);
                break;
            }

            case PVA_RSQRT_F32: {
                // rsqrtps xmm, xmm (0x0F 0x52)
                uint8_t instr_bytes[] = {0x0F, 0x52, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                write_bytes(&ptr, instr_bytes, 3);
                break;
            }

            case PVA_RCP_F32: {
                // rcpps xmm, xmm (0x0F 0x53)
                uint8_t instr_bytes[] = {0x0F, 0x53, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                write_bytes(&ptr, instr_bytes, 3);
                break;
            }

            case PVA_MIN_F32: {
                // minps xmm, xmm (0x0F 0x5D)
                emit_sse_binop(&ptr, 0x5D, instr->dst, instr->src1, instr->src2);
                break;
            }

            case PVA_MAX_F32: {
                // maxps xmm, xmm (0x0F 0x5F)
                emit_sse_binop(&ptr, 0x5F, instr->dst, instr->src1, instr->src2);
                break;
            }

            case PVA_ABS_F32: {
                // andps with sign mask (clear sign bit)
                // Load 0x7FFFFFFF mask and AND
                // Simplified: andps with constant
                uint8_t instr_bytes[] = {0x0F, 0x54, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                write_bytes(&ptr, instr_bytes, 3);
                break;
            }

            case PVA_NEG_F32: {
                // xorps with sign mask (flip sign bit)
                uint8_t instr_bytes[] = {0x0F, 0x57, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                write_bytes(&ptr, instr_bytes, 3);
                break;
            }

            // ========== COMPARISONS ==========
            case PVA_CMP_LT_F32: {
                // cmpltps xmm, xmm (cmpps with imm8=1)
                uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), 0x01};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            case PVA_CMP_LE_F32: {
                // cmpleps (imm8=2)
                uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), 0x02};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            case PVA_CMP_EQ_F32: {
                // cmpeqps (imm8=0)
                uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), 0x00};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            case PVA_CMP_NE_F32: {
                // cmpneqps (imm8=4)
                uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), 0x04};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            case PVA_CMP_GT_F32: {
                // No direct GT, use LT with swapped operands (cmpltps src2, src1)
                uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7)), 0x01};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            case PVA_CMP_GE_F32: {
                // No direct GE, use LE with swapped operands
                uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7)), 0x02};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            // ========== BITWISE ==========
            case PVA_AND_MASK:
            case PVA_AND_I32: {
                // andps (0x0F 0x54) or pand (0x66 0x0F 0xDB)
                uint8_t instr_bytes[] = {0x0F, 0x54, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                write_bytes(&ptr, instr_bytes, 3);
                break;
            }

            case PVA_OR_MASK:
            case PVA_OR_I32: {
                // orps (0x0F 0x56)
                uint8_t instr_bytes[] = {0x0F, 0x56, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                write_bytes(&ptr, instr_bytes, 3);
                break;
            }

            case PVA_XOR_MASK:
            case PVA_XOR_I32: {
                // xorps (0x0F 0x57)
                uint8_t instr_bytes[] = {0x0F, 0x57, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                write_bytes(&ptr, instr_bytes, 3);
                break;
            }

            case PVA_SHL_I32: {
                // pslld xmm, imm8 (0x66 0x0F 0x72 /6 ib)
                uint8_t instr_bytes[] = {0x66, 0x0F, 0x72, (uint8_t)(0xF0 | (instr->dst & 7)), (uint8_t)instr->imm};
                write_bytes(&ptr, instr_bytes, 5);
                break;
            }

            case PVA_SHR_I32: {
                // psrld xmm, imm8 (0x66 0x0F 0x72 /2 ib)
                uint8_t instr_bytes[] = {0x66, 0x0F, 0x72, (uint8_t)(0xD0 | (instr->dst & 7)), (uint8_t)instr->imm};
                write_bytes(&ptr, instr_bytes, 5);
                break;
            }

            case PVA_SAR_I32: {
                // psrad xmm, imm8 (0x66 0x0F 0x72 /4 ib)
                uint8_t instr_bytes[] = {0x66, 0x0F, 0x72, (uint8_t)(0xE0 | (instr->dst & 7)), (uint8_t)instr->imm};
                write_bytes(&ptr, instr_bytes, 5);
                break;
            }

            // ========== DATA MOVEMENT ==========
            case PVA_MOV: {
                // movaps dst, src
                uint8_t instr_bytes[] = {0x0F, 0x28, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                write_bytes(&ptr, instr_bytes, 3);
                break;
            }

            case PVA_BROADCAST_F32: {
                if (mod->vec_width_bytes >= 32) {
                    // vbroadcastss (AVX)
                    uint8_t instr_bytes[] = {0xC4, 0xE2, 0x79, 0x18, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 5);
                } else {
                    // SSE: shufps to broadcast
                    uint8_t instr_bytes[] = {0x0F, 0xC6, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7)), 0x00};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_SHUFFLE: {
                // shufps xmm, xmm, imm8
                uint8_t instr_bytes[] = {0x0F, 0xC6, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), (uint8_t)instr->imm};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            case PVA_BLEND: {
                // blendps xmm, xmm, imm8 (0x66 0x0F 0x3A 0x0C)
                uint8_t instr_bytes[] = {0x66, 0x0F, 0x3A, 0x0C, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), (uint8_t)instr->src3};
                write_bytes(&ptr, instr_bytes, 6);
                break;
            }

            // ========== TYPE CONVERSION ==========
            case PVA_CVT_F32_I32: {
                // cvtdq2ps xmm, xmm (0x0F 0x5B)
                uint8_t instr_bytes[] = {0x0F, 0x5B, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                write_bytes(&ptr, instr_bytes, 3);
                break;
            }

            case PVA_CVT_I32_F32: {
                // cvtps2dq xmm, xmm (0x66 0x0F 0x5B)
                uint8_t instr_bytes[] = {0x66, 0x0F, 0x5B, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            // ========== MEMORY ==========
            case PVA_LOAD_F32:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_load(&ptr, instr->dst, instr->src1, FULL_MASK);
                } else {
                    // movaps xmm, [reg] - with proper ModRM
                    uint8_t modrm = (instr->dst & 7) << 3;
                    if (instr->imm == 0) {
                        modrm |= (instr->src1 & 7);  // [base]
                    } else {
                        modrm |= 0x40 | (instr->src1 & 7);  // [base+disp8]
                    }
                    uint8_t opcode[] = {0x0F, 0x28, modrm};
                    write_bytes(&ptr, opcode, 3);
                    if (instr->imm != 0 && instr->imm < 128) {
                        uint8_t disp = (uint8_t)instr->imm;
                        write_bytes(&ptr, &disp, 1);
                    }
                }
                break;

            case PVA_STORE_F32:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_store(&ptr, instr->dst, instr->src1, FULL_MASK);
                } else {
                    // movaps [reg], xmm
                    uint8_t modrm = (instr->dst & 7) << 3;
                    if (instr->imm == 0) {
                        modrm |= (instr->src1 & 7);
                    } else {
                        modrm |= 0x40 | (instr->src1 & 7);
                    }
                    uint8_t opcode[] = {0x0F, 0x29, modrm};
                    write_bytes(&ptr, opcode, 3);
                    if (instr->imm != 0 && instr->imm < 128) {
                        uint8_t disp = (uint8_t)instr->imm;
                        write_bytes(&ptr, &disp, 1);
                    }
                }
                break;

            // ========== INITIALIZATION ==========
            case PVA_SETZERO:
                if (mod->vec_width_bytes == 64) {
                    uint8_t setzero[] = {0x62, 0xF1, 0x7C, 0x48, 0x57, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7))};
                    write_bytes(&ptr, setzero, 6);
                } else if (mod->vec_width_bytes == 32) {
                    uint8_t setzero[] = {0xC5, 0xFC, 0x57, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7))};
                    write_bytes(&ptr, setzero, 4);
                } else {
                    uint8_t setzero[] = {0x0F, 0x57, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7))};
                    write_bytes(&ptr, setzero, 3);
                }
                break;

            case PVA_SETONE: {
                // pcmpeqd xmm, xmm (sets all bits to 1)
                uint8_t setone[] = {0x66, 0x0F, 0x76, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7))};
                write_bytes(&ptr, setone, 4);
                break;
            }

            // ========== CONTROL FLOW ==========
            case PVA_RET: {
                emit_epilogue(&ptr);
                break;
            }

            case PVA_NOP:
            default:
                break;
        }
    }

    emit_epilogue(&ptr);

    size_t code_size = ptr - start;
    printf("[codegen] generated %zu bytes of code\n", code_size);
    return code_size;
}
