#include "pva.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define MAX_REGS_SSE 16
#define MAX_REGS_AVX2 16
#define MAX_REGS_AVX512 32
#define FULL_MASK 0xFFFF

/*
 * PVA Calling Convention for x86-64 (System V AMD64 ABI):
 * 
 * Pointer registers (for memory operations):
 *   PVA r10 -> x86 rdi (1st argument)
 *   PVA r11 -> x86 rsi (2nd argument)  
 *   PVA r12 -> x86 rdx (3rd argument)
 *   PVA r13 -> x86 rcx (4th argument)
 *   PVA r14 -> x86 r8  (5th argument)
 *   PVA r15 -> x86 r9  (6th argument)
 *
 * Vector registers:
 *   PVA r0-r7   -> XMM0-XMM7 / YMM0-YMM7 / ZMM0-ZMM7
 *   PVA r8-r9   -> XMM8-XMM9 (but also available as scratch)
 *
 * x86 GPR encoding (for memory addressing):
 *   rax=0, rcx=1, rdx=2, rbx=3, rsp=4, rbp=5, rsi=6, rdi=7
 *   r8=8, r9=9, r10=10, r11=11, r12=12, r13=13, r14=14, r15=15
 */

// Map PVA pointer register to x86 GPR encoding
static uint8_t pva_ptr_to_gpr(uint8_t pva_reg) {
    switch (pva_reg) {
        case 10: return 7;   // r10 -> rdi
        case 11: return 6;   // r11 -> rsi
        case 12: return 2;   // r12 -> rdx
        case 13: return 1;   // r13 -> rcx
        case 14: return 8;   // r14 -> r8
        case 15: return 9;   // r15 -> r9
        default: return 7;   // default to rdi
    }
}

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

// helper to encode VEX prefixes and opcode for AVX2 three-operand instructions
// VEX.256.0F.WIG opcode /r  
// dst = destination ymm register
// src1 = first source (encoded in VEX.vvvv, inverted)
// src2 = second source (encoded in ModRM.rm)
static void emit_avx2_instr(uint8_t** pbuf, uint8_t opcode, uint8_t dst, uint8_t src1, uint8_t src2) {
    // For 3-operand AVX instructions: op dst, src1, src2
    // VEX.vvvv = ~src1 (inverted, 4 bits)
    // ModRM.reg = dst
    // ModRM.rm = src2
    
    // Check if we need 3-byte VEX (for extended registers)
    if (dst >= 8 || src2 >= 8) {
        // 3-byte VEX: C4 RXB.mmmmm W.vvvv.L.pp
        uint8_t vex_r = (dst < 8) ? 1 : 0;
        uint8_t vex_x = 1;  // not using SIB
        uint8_t vex_b = (src2 < 8) ? 1 : 0;
        uint8_t vex_vvvv = (~src1) & 0xF;
        
        uint8_t vex[3] = {
            0xC4,
            (uint8_t)((vex_r << 7) | (vex_x << 6) | (vex_b << 5) | 0x01),  // R.X.B.00001 (0F map)
            (uint8_t)((0 << 7) | (vex_vvvv << 3) | (1 << 2) | 0x00)  // W=0, vvvv, L=1(256-bit), pp=00
        };
        write_bytes(pbuf, vex, 3);
    } else {
        // 2-byte VEX: C5 R.vvvv.L.pp
        uint8_t vex_r = 1;  // dst < 8
        uint8_t vex_vvvv = (~src1) & 0xF;
        
        uint8_t vex[2] = {
            0xC5,
            (uint8_t)((vex_r << 7) | (vex_vvvv << 3) | (1 << 2) | 0x00)  // R=1, vvvv, L=1(256-bit), pp=00
        };
        write_bytes(pbuf, vex, 2);
    }

    write_bytes(pbuf, &opcode, 1);

    // ModRM: mod=11 (register), reg=dst, rm=src2
    uint8_t modrm = 0xC0 | ((dst & 0x7) << 3) | (src2 & 0x7);
    write_bytes(pbuf, &modrm, 1);
}

// Helper for VEX.256.66.0F38 instructions (integer ops like pminsd, pmaxsd, pabsd)
// These are 3-byte VEX with map=0F38, pp=01 (66 prefix)
static void emit_avx2_66_0f38(uint8_t** pbuf, uint8_t opcode, uint8_t dst, uint8_t src1, uint8_t src2) {
    // 3-byte VEX: C4 RXB.mmmmm W.vvvv.L.pp
    // mmmmm = 00010 for 0F38 map
    uint8_t vex_r = (dst < 8) ? 1 : 0;
    uint8_t vex_x = 1;  // not using SIB
    uint8_t vex_b = (src2 < 8) ? 1 : 0;
    uint8_t vex_vvvv = (~src1) & 0xF;
    
    uint8_t vex[3] = {
        0xC4,
        (uint8_t)((vex_r << 7) | (vex_x << 6) | (vex_b << 5) | 0x02),  // R.X.B.00010 (0F38 map)
        (uint8_t)((0 << 7) | (vex_vvvv << 3) | (1 << 2) | 0x01)  // W=0, vvvv, L=1(256-bit), pp=01(66)
    };
    write_bytes(pbuf, vex, 3);
    write_bytes(pbuf, &opcode, 1);

    // ModRM: mod=11 (register), reg=dst, rm=src2
    uint8_t modrm = 0xC0 | ((dst & 0x7) << 3) | (src2 & 0x7);
    write_bytes(pbuf, &modrm, 1);
}

// Helper for VEX.256.66.0F instructions (integer ops like paddd, psubd, pmulld, pcmpeqd, pcmpgtd)
// These are 3-byte VEX with map=0F, pp=01 (66 prefix)
static void emit_avx2_66_0f(uint8_t** pbuf, uint8_t opcode, uint8_t dst, uint8_t src1, uint8_t src2) {
    // 3-byte VEX: C4 RXB.mmmmm W.vvvv.L.pp
    // mmmmm = 00001 for 0F map
    uint8_t vex_r = (dst < 8) ? 1 : 0;
    uint8_t vex_x = 1;  // not using SIB
    uint8_t vex_b = (src2 < 8) ? 1 : 0;
    uint8_t vex_vvvv = (~src1) & 0xF;
    
    uint8_t vex[3] = {
        0xC4,
        (uint8_t)((vex_r << 7) | (vex_x << 6) | (vex_b << 5) | 0x01),  // R.X.B.00001 (0F map)
        (uint8_t)((0 << 7) | (vex_vvvv << 3) | (1 << 2) | 0x01)  // W=0, vvvv, L=1(256-bit), pp=01(66)
    };
    write_bytes(pbuf, vex, 3);
    write_bytes(pbuf, &opcode, 1);

    // ModRM: mod=11 (register), reg=dst, rm=src2
    uint8_t modrm = 0xC0 | ((dst & 0x7) << 3) | (src2 & 0x7);
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
                if (mod->vec_width_bytes >= 32) {
                    // vpaddd ymm, ymm, ymm - VEX.256.66.0F.WIG FE /r
                    // 2-byte VEX: C5 R.vvvv.L.pp  (R=1 for regs<8, L=1 for 256-bit, pp=01 for 66)
                    uint8_t vex_byte2 = (uint8_t)(0x85 | ((~instr->src1 & 0xF) << 3));  // R=1, vvvv, L=1, pp=01
                    uint8_t instr_bytes[] = {0xC5, vex_byte2, 0xFE, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                } else {
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0xFE, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_SUB_I32: {
                if (mod->vec_width_bytes >= 32) {
                    // vpsubd ymm, ymm, ymm - VEX.256.66.0F.WIG FA /r
                    uint8_t vex_byte2 = (uint8_t)(0x85 | ((~instr->src1 & 0xF) << 3));  // R=1, vvvv, L=1, pp=01
                    uint8_t instr_bytes[] = {0xC5, vex_byte2, 0xFA, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                } else {
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0xFA, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_MUL_I32: {
                if (mod->vec_width_bytes >= 32) {
                    // vpmulld ymm, ymm, ymm - VEX.256.66.0F38.WIG 40 /r
                    // Use the helper for 66.0F38 map
                    emit_avx2_66_0f38(&ptr, 0x40, instr->dst, instr->src1, instr->src2);
                } else {
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x38, 0x40, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 5);
                }
                break;
            }

            // ========== MATH OPERATIONS ==========
            case PVA_SQRT_F32: {
                if (mod->vec_width_bytes == 32) {
                    // vsqrtps ymm, ymm - VEX.256.0F.WIG 51
                    emit_avx2_instr(&ptr, 0x51, instr->dst, instr->src1, instr->src1);
                } else {
                    // sqrtps xmm, xmm (0x0F 0x51)
                    uint8_t instr_bytes[] = {0x0F, 0x51, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_RSQRT_F32: {
                if (mod->vec_width_bytes == 32) {
                    // vrsqrtps ymm, ymm - VEX.256.0F.WIG 52
                    emit_avx2_instr(&ptr, 0x52, instr->dst, instr->src1, instr->src1);
                } else {
                    // rsqrtps xmm, xmm (0x0F 0x52)
                    uint8_t instr_bytes[] = {0x0F, 0x52, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_RCP_F32: {
                if (mod->vec_width_bytes == 32) {
                    // vrcpps ymm, ymm - VEX.256.0F.WIG 53
                    emit_avx2_instr(&ptr, 0x53, instr->dst, instr->src1, instr->src1);
                } else {
                    // rcpps xmm, xmm (0x0F 0x53)
                    uint8_t instr_bytes[] = {0x0F, 0x53, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_MIN_F32: {
                if (mod->vec_width_bytes == 32) {
                    // vminps ymm, ymm, ymm
                    emit_avx2_instr(&ptr, 0x5D, instr->dst, instr->src1, instr->src2);
                } else {
                    // minps xmm, xmm (0x0F 0x5D)
                    emit_sse_binop(&ptr, 0x5D, instr->dst, instr->src1, instr->src2);
                }
                break;
            }

            case PVA_MAX_F32: {
                if (mod->vec_width_bytes == 32) {
                    // vmaxps ymm, ymm, ymm
                    emit_avx2_instr(&ptr, 0x5F, instr->dst, instr->src1, instr->src2);
                } else {
                    // maxps xmm, xmm (0x0F 0x5F)
                    emit_sse_binop(&ptr, 0x5F, instr->dst, instr->src1, instr->src2);
                }
                break;
            }

            case PVA_ABS_F32: {
                // ABS: clear sign bit = AND with 0x7FFFFFFF
                // For AVX: create mask in temp register, then vandps
                if (mod->vec_width_bytes == 32) {
                    // Use a scratch register (ymm15) to create mask
                    // vpcmpeqd ymm15, ymm15, ymm15 (all ones)
                    uint8_t pcmpeq[] = {0xC4, 0x41, 0x05, 0x76, 0xFF};  // VEX.256.66.0F.WIG 76 /r
                    write_bytes(&ptr, pcmpeq, 5);
                    // vpsrld ymm15, ymm15, 1 (shift right to get 0x7FFFFFFF)
                    uint8_t psrld[] = {0xC4, 0xC1, 0x05, 0x72, 0xD7, 0x01};  // VEX.256.66.0F.WIG 72 /2 ib
                    write_bytes(&ptr, psrld, 6);
                    // vandps dst, src, ymm15
                    uint8_t vandps[] = {0xC4, 0xC1, (uint8_t)(0x04 | ((~instr->src1 & 0xF) << 3)), 0x54, 
                                        (uint8_t)(0xC7 | ((instr->dst & 7) << 3))};
                    write_bytes(&ptr, vandps, 5);
                } else {
                    // SSE: similar approach with xmm15
                    // pcmpeqd xmm7, xmm7
                    uint8_t pcmpeq[] = {0x66, 0x0F, 0x76, 0xFF};
                    write_bytes(&ptr, pcmpeq, 4);
                    // psrld xmm7, 1
                    uint8_t psrld[] = {0x66, 0x0F, 0x72, 0xD7, 0x01};
                    write_bytes(&ptr, psrld, 5);
                    // andps dst, xmm7
                    uint8_t instr_bytes[] = {0x0F, 0x54, (uint8_t)(0xC7 | ((instr->dst & 7) << 3))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_NEG_F32: {
                // NEG: flip sign bit = XOR with 0x80000000
                if (mod->vec_width_bytes == 32) {
                    // Use scratch register (ymm15) to create mask
                    // vpcmpeqd ymm15, ymm15, ymm15 (all ones)
                    uint8_t pcmpeq[] = {0xC4, 0x41, 0x05, 0x76, 0xFF};
                    write_bytes(&ptr, pcmpeq, 5);
                    // vpslld ymm15, ymm15, 31 (shift left to get 0x80000000)
                    uint8_t pslld[] = {0xC4, 0xC1, 0x05, 0x72, 0xF7, 0x1F};  // VEX.256.66.0F.WIG 72 /6 ib
                    write_bytes(&ptr, pslld, 6);
                    // vxorps dst, src, ymm15
                    uint8_t vxorps[] = {0xC4, 0xC1, (uint8_t)(0x04 | ((~instr->src1 & 0xF) << 3)), 0x57,
                                        (uint8_t)(0xC7 | ((instr->dst & 7) << 3))};
                    write_bytes(&ptr, vxorps, 5);
                } else {
                    // SSE: similar approach
                    uint8_t pcmpeq[] = {0x66, 0x0F, 0x76, 0xFF};
                    write_bytes(&ptr, pcmpeq, 4);
                    uint8_t pslld[] = {0x66, 0x0F, 0x72, 0xF7, 0x1F};
                    write_bytes(&ptr, pslld, 5);
                    uint8_t instr_bytes[] = {0x0F, 0x57, (uint8_t)(0xC7 | ((instr->dst & 7) << 3))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            // ========== COMPARISONS ==========
            case PVA_CMP_LT_F32: {
                if (mod->vec_width_bytes == 32) {
                    // vcmpps ymm, ymm, ymm, imm8 - VEX.256.0F.WIG C2 /r ib
                    // Using vcmpps with imm8=1 (LT)
                    // Byte3: W=0 (bit7), vvvv=~src1 (bits 6-3), L=1 (bit2), pp=00 (bits1-0)
                    uint8_t vex[3] = {
                        0xC4,
                        (uint8_t)(0xE1),  // R=1, X=1, B=1, mmmmm=01 (0F)
                        (uint8_t)(0x04 | ((~instr->src1 & 0xF) << 3))  // W=0, vvvv=~src1, L=1, pp=00
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0xC2;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7));
                    write_bytes(&ptr, &modrm, 1);
                    uint8_t imm = 0x01;  // LT
                    write_bytes(&ptr, &imm, 1);
                } else {
                    // cmpltps xmm, xmm (cmpps with imm8=1)
                    uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), 0x01};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_CMP_LE_F32: {
                if (mod->vec_width_bytes == 32) {
                    uint8_t vex[3] = {
                        0xC4, 0xE1,
                        (uint8_t)(0x04 | ((~instr->src1 & 0xF) << 3))
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0xC2;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7));
                    write_bytes(&ptr, &modrm, 1);
                    uint8_t imm = 0x02;  // LE
                    write_bytes(&ptr, &imm, 1);
                } else {
                    uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), 0x02};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_CMP_EQ_F32: {
                if (mod->vec_width_bytes == 32) {
                    uint8_t vex[3] = {
                        0xC4, 0xE1,
                        (uint8_t)(0x04 | ((~instr->src1 & 0xF) << 3))
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0xC2;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7));
                    write_bytes(&ptr, &modrm, 1);
                    uint8_t imm = 0x00;  // EQ
                    write_bytes(&ptr, &imm, 1);
                } else {
                    uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), 0x00};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_CMP_NE_F32: {
                if (mod->vec_width_bytes == 32) {
                    uint8_t vex[3] = {
                        0xC4, 0xE1,
                        (uint8_t)(0x04 | ((~instr->src1 & 0xF) << 3))
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0xC2;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7));
                    write_bytes(&ptr, &modrm, 1);
                    uint8_t imm = 0x04;  // NEQ
                    write_bytes(&ptr, &imm, 1);
                } else {
                    uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), 0x04};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_CMP_GT_F32: {
                if (mod->vec_width_bytes == 32) {
                    // GT: swap operands and use LT
                    uint8_t vex[3] = {
                        0xC4, 0xE1,
                        (uint8_t)(0x04 | ((~instr->src2 & 0xF) << 3))  // swap: use src2 as vvvv
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0xC2;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7));  // swap: src1 as rm
                    write_bytes(&ptr, &modrm, 1);
                    uint8_t imm = 0x01;  // LT (with swapped operands = GT)
                    write_bytes(&ptr, &imm, 1);
                } else {
                    uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7)), 0x01};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_CMP_GE_F32: {
                if (mod->vec_width_bytes == 32) {
                    // GE: swap operands and use LE
                    uint8_t vex[3] = {
                        0xC4, 0xE1,
                        (uint8_t)(0x04 | ((~instr->src2 & 0xF) << 3))
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0xC2;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7));
                    write_bytes(&ptr, &modrm, 1);
                    uint8_t imm = 0x02;  // LE (with swapped operands = GE)
                    write_bytes(&ptr, &imm, 1);
                } else {
                    uint8_t instr_bytes[] = {0x0F, 0xC2, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7)), 0x02};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            // ========== BITWISE ==========
            case PVA_AND_MASK:
            case PVA_AND_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vandps ymm, ymm, ymm - VEX.256.0F.WIG 54
                    emit_avx2_instr(&ptr, 0x54, instr->dst, instr->src1, instr->src2);
                } else {
                    // andps (0x0F 0x54)
                    uint8_t instr_bytes[] = {0x0F, 0x54, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_OR_MASK:
            case PVA_OR_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vorps ymm, ymm, ymm - VEX.256.0F.WIG 56
                    emit_avx2_instr(&ptr, 0x56, instr->dst, instr->src1, instr->src2);
                } else {
                    // orps (0x0F 0x56)
                    uint8_t instr_bytes[] = {0x0F, 0x56, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_XOR_MASK:
            case PVA_XOR_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vxorps ymm, ymm, ymm - VEX.256.0F.WIG 57
                    emit_avx2_instr(&ptr, 0x57, instr->dst, instr->src1, instr->src2);
                } else {
                    // xorps (0x0F 0x57)
                    uint8_t instr_bytes[] = {0x0F, 0x57, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_SHL_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vpslld ymm_dst, ymm_src, imm8 - VEX.256.66.0F.WIG 72 /6 ib
                    // VEX.vvvv = ~dst, ModRM.rm = src1
                    uint8_t vex[3] = {
                        0xC4, 0xE1,
                        (uint8_t)(0x05 | ((~instr->dst & 0xF) << 3))  // W=0, vvvv=~dst, L=1, pp=01
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0x72;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xF0 | (instr->src1 & 7));  // /6 = 110 in reg field, rm=src1
                    write_bytes(&ptr, &modrm, 1);
                    uint8_t imm = (uint8_t)instr->imm;
                    write_bytes(&ptr, &imm, 1);
                } else {
                    // pslld xmm, imm8 (0x66 0x0F 0x72 /6 ib)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x72, (uint8_t)(0xF0 | (instr->dst & 7)), (uint8_t)instr->imm};
                    write_bytes(&ptr, instr_bytes, 5);
                }
                break;
            }

            case PVA_SHR_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vpsrld ymm_dst, ymm_src, imm8 - VEX.256.66.0F.WIG 72 /2 ib
                    uint8_t vex[3] = {
                        0xC4, 0xE1,
                        (uint8_t)(0x05 | ((~instr->dst & 0xF) << 3))  // W=0, vvvv=~dst, L=1, pp=01
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0x72;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xD0 | (instr->src1 & 7));  // /2 = 010 in reg field, rm=src1
                    write_bytes(&ptr, &modrm, 1);
                    uint8_t imm = (uint8_t)instr->imm;
                    write_bytes(&ptr, &imm, 1);
                } else {
                    // psrld xmm, imm8 (0x66 0x0F 0x72 /2 ib)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x72, (uint8_t)(0xD0 | (instr->dst & 7)), (uint8_t)instr->imm};
                    write_bytes(&ptr, instr_bytes, 5);
                }
                break;
            }

            case PVA_SAR_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vpsrad ymm_dst, ymm_src, imm8 - VEX.256.66.0F.WIG 72 /4 ib
                    uint8_t vex[3] = {
                        0xC4, 0xE1,
                        (uint8_t)(0x05 | ((~instr->dst & 0xF) << 3))  // W=0, vvvv=~dst, L=1, pp=01
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0x72;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xE0 | (instr->src1 & 7));  // /4 = 100 in reg field, rm=src1
                    write_bytes(&ptr, &modrm, 1);
                    uint8_t imm = (uint8_t)instr->imm;
                    write_bytes(&ptr, &imm, 1);
                } else {
                    // psrad xmm, imm8 (0x66 0x0F 0x72 /4 ib)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x72, (uint8_t)(0xE0 | (instr->dst & 7)), (uint8_t)instr->imm};
                    write_bytes(&ptr, instr_bytes, 5);
                }
                break;
            }

            // ========== DATA MOVEMENT ==========
            case PVA_MOV: {
                if (mod->vec_width_bytes == 32) {
                    // vmovaps ymm, ymm - VEX.256.0F.WIG 28
                    uint8_t vex[3] = {
                        0xC4,
                        (uint8_t)(0xE1),  // R=1, X=1, B=1, mmmmm=01
                        (uint8_t)(0x7C)   // W=0, vvvv=1111, L=1, pp=00
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0x28;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7));
                    write_bytes(&ptr, &modrm, 1);
                } else {
                    // movaps dst, src
                    uint8_t instr_bytes[] = {0x0F, 0x28, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_BROADCAST_F32: {
                if (mod->vec_width_bytes >= 32) {
                    // vbroadcastss ymm, xmm - VEX.256.66.0F38.W0 18 /r
                    // C4 E2 7D 18 modrm
                    uint8_t instr_bytes[] = {0xC4, 0xE2, 0x7D, 0x18, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 5);
                } else {
                    // SSE: shufps to broadcast
                    uint8_t instr_bytes[] = {0x0F, 0xC6, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7)), 0x00};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_SHUFFLE: {
                if (mod->vec_width_bytes == 32) {
                    // vshufps ymm, ymm, ymm, imm8 - VEX.256.0F.WIG C6
                    emit_avx2_instr(&ptr, 0xC6, instr->dst, instr->src1, instr->src2);
                    uint8_t imm = (uint8_t)instr->imm;
                    write_bytes(&ptr, &imm, 1);
                } else {
                    // shufps xmm, xmm, imm8
                    uint8_t instr_bytes[] = {0x0F, 0xC6, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), (uint8_t)instr->imm};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_BLEND: {
                if (mod->vec_width_bytes == 32) {
                    // vblendps ymm, ymm, ymm, imm8 - VEX.256.66.0F3A.WIG 0C
                    // 3-byte VEX: C4 E3 xD 0C modrm imm (x = ~vvvv, D = L=1, pp=01)
                    uint8_t vex[3] = {
                        0xC4, 0xE3,
                        (uint8_t)(0x05 | ((~instr->src1 & 0xF) << 3))  // W=0, vvvv, L=1, pp=01
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0x0C;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7));
                    write_bytes(&ptr, &modrm, 1);
                    uint8_t imm = (uint8_t)instr->src3;
                    write_bytes(&ptr, &imm, 1);
                } else {
                    // blendps xmm, xmm, imm8 (0x66 0x0F 0x3A 0x0C)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x3A, 0x0C, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7)), (uint8_t)instr->src3};
                    write_bytes(&ptr, instr_bytes, 6);
                }
                break;
            }

            // ========== TYPE CONVERSION ==========
            case PVA_CVT_F32_I32: {
                // vcvtps2dq - convert packed f32 to i32
                if (mod->vec_width_bytes >= 32) {
                    // VEX.256.66.0F.WIG 5B /r
                    uint8_t instr_bytes[] = {0xC5, 0xFD, 0x5B, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                } else {
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x5B, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_CVT_I32_F32: {
                // vcvtdq2ps - convert packed i32 to f32
                if (mod->vec_width_bytes >= 32) {
                    // VEX.256.0F.WIG 5B /r
                    uint8_t instr_bytes[] = {0xC5, 0xFC, 0x5B, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                } else {
                    uint8_t instr_bytes[] = {0x0F, 0x5B, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_CVT_F64_F32: {
                // cvtps2pd xmm, xmm (0x0F 0x5A) - widen f32 to f64
                uint8_t instr_bytes[] = {0x0F, 0x5A, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                write_bytes(&ptr, instr_bytes, 3);
                break;
            }

            case PVA_CVT_F32_F64: {
                // cvtpd2ps xmm, xmm (0x66 0x0F 0x5A) - narrow f64 to f32
                uint8_t instr_bytes[] = {0x66, 0x0F, 0x5A, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            // ========== I16 ARITHMETIC ==========
            case PVA_ADD_I16: {
                if (mod->vec_width_bytes == 32) {
                    // vpaddw ymm, ymm, ymm - VEX.256.66.0F.WIG FD
                    emit_avx2_66_0f(&ptr, 0xFD, instr->dst, instr->src1, instr->src2);
                } else {
                    // paddw xmm, xmm (0x66 0x0F 0xFD)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0xFD, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_SUB_I16: {
                if (mod->vec_width_bytes == 32) {
                    // vpsubw ymm, ymm, ymm - VEX.256.66.0F.WIG F9
                    emit_avx2_66_0f(&ptr, 0xF9, instr->dst, instr->src1, instr->src2);
                } else {
                    // psubw xmm, xmm (0x66 0x0F 0xF9)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0xF9, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_MUL_I16: {
                if (mod->vec_width_bytes == 32) {
                    // vpmullw ymm, ymm, ymm - VEX.256.66.0F.WIG D5
                    emit_avx2_66_0f(&ptr, 0xD5, instr->dst, instr->src1, instr->src2);
                } else {
                    // pmullw xmm, xmm (0x66 0x0F 0xD5)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0xD5, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            // ========== ADDITIONAL MATH ==========
            case PVA_SQRT_F64: {
                // sqrtpd xmm, xmm (0x66 0x0F 0x51)
                uint8_t instr_bytes[] = {0x66, 0x0F, 0x51, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                write_bytes(&ptr, instr_bytes, 4);
                break;
            }

            case PVA_ABS_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vpabsd ymm, ymm - VEX.256.66.0F38.WIG 1E
                    emit_avx2_66_0f38(&ptr, 0x1E, instr->dst, 0, instr->src1);
                } else {
                    // pabsd xmm, xmm (0x66 0x0F 0x38 0x1E) - SSSE3
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x38, 0x1E, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 5);
                }
                break;
            }

            case PVA_NEG_I32: {
                if (mod->vec_width_bytes == 32) {
                    // AVX2: vpxor to zero dst, then vpsubd dst, dst, src1
                    // vpxor ymm, ymm, ymm (zero dst)
                    emit_avx2_66_0f(&ptr, 0xEF, instr->dst, instr->dst, instr->dst);
                    // vpsubd ymm, ymm, ymm (dst = 0 - src1)
                    emit_avx2_66_0f(&ptr, 0xFA, instr->dst, instr->dst, instr->src1);
                } else {
                    // pxor + psubd
                    uint8_t xor_bytes[] = {0x66, 0x0F, 0xEF, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7))};
                    write_bytes(&ptr, xor_bytes, 4);
                    uint8_t sub_bytes[] = {0x66, 0x0F, 0xFA, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, sub_bytes, 4);
                }
                break;
            }

            case PVA_FMA_F32: {
                // vfma dst, src1, src2, src3 means dst = src1 * src2 + src3
                // We need to: 1) copy src3 to dst, 2) use vfmadd231ps dst, src1, src2
                // vfmadd231ps dst, src1, src2 = src1 * src2 + dst
                if (mod->vec_width_bytes >= 32) {
                    // First: vmovaps dst, src3 (copy addend to destination)
                    uint8_t mov[5] = {
                        0xC4, 0xE1, 0x7C, 0x28,  // VEX.256.0F.WIG 28
                        (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src3 & 7))
                    };
                    write_bytes(&ptr, mov, 5);
                    
                    // Then: vfmadd231ps dst, src1, src2
                    // VEX.256.66.0F38.W0 B8 /r
                    uint8_t vvvv = (~instr->src1) & 0xF;
                    uint8_t instr_bytes[] = {
                        0xC4, 0xE2, 
                        (uint8_t)(0x75 | (vvvv << 3)),  // W=0, vvvv, L=1(256-bit), pp=01(66)
                        0xB8,
                        (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))
                    };
                    write_bytes(&ptr, instr_bytes, 5);
                } else {
                    // SSE fallback: mul then add
                    // First copy src3 to dst
                    uint8_t mov[] = {0x0F, 0x28, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src3 & 7))};
                    write_bytes(&ptr, mov, 3);
                    // Multiply src1 * src2 into a temp (clobber dst since we saved src3)
                    // Actually simpler: dst = src1 * src2, then dst += src3 (if we save src3 first)
                    emit_sse_binop(&ptr, 0x59, instr->dst, instr->src1, instr->src2);  // dst = src1 * src2
                    emit_sse_binop(&ptr, 0x58, instr->dst, instr->dst, instr->src3);    // dst += src3
                }
                break;
            }

            case PVA_FMA_F64: {
                // vfma dst, src1, src2, src3 means dst = src1 * src2 + src3
                if (mod->vec_width_bytes >= 32) {
                    // First: vmovapd dst, src3 (copy addend)
                    uint8_t mov[5] = {
                        0xC4, 0xE1, 0x7D, 0x28,  // VEX.256.66.0F.WIG 28
                        (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src3 & 7))
                    };
                    write_bytes(&ptr, mov, 5);
                    
                    // vfmadd231pd dst, src1, src2
                    // VEX.256.66.0F38.W1 B8 /r
                    uint8_t vvvv = (~instr->src1) & 0xF;
                    uint8_t instr_bytes[] = {
                        0xC4, 0xE2,
                        (uint8_t)(0xF5 | (vvvv << 3)),  // W=1, vvvv, L=1, pp=01
                        0xB8,
                        (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))
                    };
                    write_bytes(&ptr, instr_bytes, 5);
                }
                break;
            }

            case PVA_MIN_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vpminsd ymm, ymm, ymm - VEX.256.66.0F38.WIG 39
                    emit_avx2_66_0f38(&ptr, 0x39, instr->dst, instr->src1, instr->src2);
                } else {
                    // pminsd xmm, xmm (0x66 0x0F 0x38 0x39) - SSE4.1
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x38, 0x39, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 5);
                }
                break;
            }

            case PVA_MAX_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vpmaxsd ymm, ymm, ymm - VEX.256.66.0F38.WIG 3D
                    emit_avx2_66_0f38(&ptr, 0x3D, instr->dst, instr->src1, instr->src2);
                } else {
                    // pmaxsd xmm, xmm (0x66 0x0F 0x38 0x3D) - SSE4.1
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x38, 0x3D, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 5);
                }
                break;
            }

            case PVA_NOT_MASK:
            case PVA_NOT_I32: {
                if (mod->vec_width_bytes == 32) {
                    // Create all 1s in dst, then XOR with src1
                    // vpcmpeqd dst, dst, dst to get all 1s
                    emit_avx2_66_0f(&ptr, 0x76, instr->dst, instr->dst, instr->dst);
                    // vpxor dst, dst, src1 
                    emit_avx2_66_0f(&ptr, 0xEF, instr->dst, instr->dst, instr->src1);
                } else {
                    // pcmpeqd dst, dst (all 1s)
                    uint8_t eq_bytes[] = {0x66, 0x0F, 0x76, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7))};
                    write_bytes(&ptr, eq_bytes, 4);
                    // pxor dst, src1
                    uint8_t xor_bytes[] = {0x66, 0x0F, 0xEF, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, xor_bytes, 4);
                }
                break;
            }

            // ========== HORIZONTAL REDUCTIONS ==========
            case PVA_HADD_F32: {
                if (mod->vec_width_bytes == 32) {
                    // vhaddps ymm, ymm, ymm - VEX.256.F2.0F.WIG 7C
                    // 3-byte VEX: C4 E1 xF 7C (x = ~vvvv, F = L=1, pp=11 for F2)
                    uint8_t vex[3] = {
                        0xC4, 0xE1,
                        (uint8_t)(0x07 | ((~instr->src1 & 0xF) << 3))  // W=0, vvvv, L=1, pp=11(F2)
                    };
                    write_bytes(&ptr, vex, 3);
                    uint8_t opcode = 0x7C;
                    write_bytes(&ptr, &opcode, 1);
                    uint8_t modrm = (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7));
                    write_bytes(&ptr, &modrm, 1);
                    // Need second hadd for full reduction
                    write_bytes(&ptr, vex, 3);
                    write_bytes(&ptr, &opcode, 1);
                    write_bytes(&ptr, &modrm, 1);
                } else {
                    // haddps xmm, xmm (0xF2 0x0F 0x7C) - SSE3
                    uint8_t instr_bytes[] = {0xF2, 0x0F, 0x7C, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                    // Need two haddps for full reduction
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_HMIN_F32: {
                if (mod->vec_width_bytes == 32) {
                    // Use AVX2 vshufps and vminps
                    // vshufps ymm, ymm, ymm, imm8 - VEX.256.0F.WIG C6
                    emit_avx2_instr(&ptr, 0xC6, instr->dst, instr->src1, instr->src1);
                    uint8_t imm = 0x4E;
                    write_bytes(&ptr, &imm, 1);
                    // vminps ymm, ymm, ymm
                    emit_avx2_instr(&ptr, 0x5D, instr->dst, instr->dst, instr->src1);
                } else {
                    // Simplified: shufps + minps
                    uint8_t shuf[] = {0x0F, 0xC6, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7)), 0x4E};
                    write_bytes(&ptr, shuf, 4);
                    uint8_t minp[] = {0x0F, 0x5D, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, minp, 3);
                }
                break;
            }

            case PVA_HMAX_F32: {
                if (mod->vec_width_bytes == 32) {
                    // Use AVX2 vshufps and vmaxps
                    emit_avx2_instr(&ptr, 0xC6, instr->dst, instr->src1, instr->src1);
                    uint8_t imm = 0x4E;
                    write_bytes(&ptr, &imm, 1);
                    // vmaxps ymm, ymm, ymm
                    emit_avx2_instr(&ptr, 0x5F, instr->dst, instr->dst, instr->src1);
                } else {
                    // Similar to HMIN but with maxps
                    uint8_t shuf[] = {0x0F, 0xC6, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7)), 0x4E};
                    write_bytes(&ptr, shuf, 4);
                    uint8_t maxp[] = {0x0F, 0x5F, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, maxp, 3);
                }
                break;
            }

            // ========== ADDITIONAL DATA MOVEMENT ==========
            case PVA_BROADCAST_I32: {
                if (mod->vec_width_bytes >= 32) {
                    // vpbroadcastd ymm, xmm (AVX2) - VEX.256.66.0F38.W0 58 /r
                    // C4 E2 7D 58 modrm (0x7D = W=0, vvvv=1111, L=1, pp=01)
                    uint8_t instr_bytes[] = {0xC4, 0xE2, 0x7D, 0x58, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 5);
                } else {
                    // pshufd with 0x00 immediate
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x70, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7)), 0x00};
                    write_bytes(&ptr, instr_bytes, 5);
                }
                break;
            }

            case PVA_SET1_F32: {
                // Load immediate and broadcast - complex, simplified here
                // movss xmm, mem; shufps xmm, xmm, 0
                // For now, assume immediate is pre-loaded
                uint8_t shuf[] = {0x0F, 0xC6, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7)), 0x00};
                write_bytes(&ptr, shuf, 4);
                break;
            }

            case PVA_SET1_I32: {
                // Similar approach for integers
                uint8_t shuf[] = {0x66, 0x0F, 0x70, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7)), 0x00};
                write_bytes(&ptr, shuf, 5);
                break;
            }

            // ========== I32 COMPARISONS ==========
            case PVA_CMP_LT_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vpcmpgtd with swapped operands (a < b => b > a)
                    // vpcmpgtd ymm, ymm, ymm - VEX.256.66.0F.WIG 66
                    emit_avx2_66_0f(&ptr, 0x66, instr->dst, instr->src2, instr->src1);
                } else {
                    // pcmpgtd with swapped operands (a < b => b > a)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x66, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_CMP_GT_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vpcmpgtd ymm, ymm, ymm - VEX.256.66.0F.WIG 66
                    emit_avx2_66_0f(&ptr, 0x66, instr->dst, instr->src1, instr->src2);
                } else {
                    // pcmpgtd xmm, xmm (0x66 0x0F 0x66)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x66, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_CMP_EQ_I32: {
                if (mod->vec_width_bytes == 32) {
                    // vpcmpeqd ymm, ymm, ymm - VEX.256.66.0F.WIG 76
                    emit_avx2_66_0f(&ptr, 0x76, instr->dst, instr->src1, instr->src2);
                } else {
                    // pcmpeqd xmm, xmm (0x66 0x0F 0x76)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x76, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            // ========== MEMORY ==========
            case PVA_LOAD_F32: {
                // Map PVA pointer register to x86 GPR
                uint8_t base_gpr = pva_ptr_to_gpr(instr->src1);
                uint8_t xmm_dst = instr->dst & 0xF;
                
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_load(&ptr, xmm_dst, base_gpr, FULL_MASK);
                } else if (mod->vec_width_bytes == 32) {
                    // vmovaps ymm, [gpr] - VEX encoded
                    // VEX.256.0F.WIG 28 /r
                    uint8_t rex_needed = (xmm_dst >= 8) || (base_gpr >= 8);
                    uint8_t vex_r = (xmm_dst >= 8) ? 0 : 1;
                    uint8_t vex_b = (base_gpr >= 8) ? 0 : 1;
                    
                    // 3-byte VEX: C4 RXB mmmmm WvvvvLpp
                    uint8_t vex[3] = {
                        0xC4,
                        (uint8_t)(0x01 | (vex_r << 7) | (1 << 6) | (vex_b << 5)),  // RXB + 0F map
                        (uint8_t)(0x04 | (0xF << 3))  // W=0, vvvv=1111(unused), L=1(256-bit), pp=00
                    };
                    write_bytes(&ptr, vex, 3);
                    
                    uint8_t opcode = 0x28;  // vmovaps
                    write_bytes(&ptr, &opcode, 1);
                    
                    // ModRM: mod=00 (indirect), reg=dst, rm=base
                    // Special handling for rsp(4) and rbp(5) which need SIB or disp
                    uint8_t modrm;
                    if (instr->imm == 0 && base_gpr != 5 && base_gpr != 13) {
                        modrm = ((xmm_dst & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        // SIB byte needed for rsp-based addressing
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;  // scale=0, index=rsp(none), base=rsp
                            write_bytes(&ptr, &sib, 1);
                        }
                    } else if (instr->imm >= -128 && instr->imm < 128) {
                        // 8-bit signed displacement
                        modrm = 0x40 | ((xmm_dst & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;
                            write_bytes(&ptr, &sib, 1);
                        }
                        uint8_t disp = (uint8_t)(int8_t)instr->imm;
                        write_bytes(&ptr, &disp, 1);
                    } else {
                        // 32-bit displacement
                        modrm = 0x80 | ((xmm_dst & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;
                            write_bytes(&ptr, &sib, 1);
                        }
                        uint32_t disp = (uint32_t)instr->imm;
                        write_bytes(&ptr, (uint8_t*)&disp, 4);
                    }
                } else {
                    // SSE: movaps xmm, [gpr]
                    // Need REX prefix if using extended registers
                    if (xmm_dst >= 8 || base_gpr >= 8) {
                        uint8_t rex = 0x40 | ((xmm_dst >= 8) ? 0x04 : 0) | ((base_gpr >= 8) ? 0x01 : 0);
                        write_bytes(&ptr, &rex, 1);
                    }
                    
                    uint8_t opcode[2] = {0x0F, 0x28};
                    write_bytes(&ptr, opcode, 2);
                    
                    uint8_t modrm;
                    if (instr->imm == 0 && (base_gpr & 7) != 5) {
                        modrm = ((xmm_dst & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;
                            write_bytes(&ptr, &sib, 1);
                        }
                    } else if (instr->imm >= -128 && instr->imm < 128) {
                        modrm = 0x40 | ((xmm_dst & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;
                            write_bytes(&ptr, &sib, 1);
                        }
                        uint8_t disp = (uint8_t)(int8_t)instr->imm;
                        write_bytes(&ptr, &disp, 1);
                    } else {
                        modrm = 0x80 | ((xmm_dst & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;
                            write_bytes(&ptr, &sib, 1);
                        }
                        uint32_t disp = (uint32_t)instr->imm;
                        write_bytes(&ptr, (uint8_t*)&disp, 4);
                    }
                }
                break;
            }

            case PVA_STORE_F32: {
                // For store: instr->dst is the base pointer register, instr->src1 is the XMM source
                uint8_t base_gpr = pva_ptr_to_gpr(instr->dst);
                uint8_t xmm_src = instr->src1 & 0xF;
                
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_store(&ptr, xmm_src, base_gpr, FULL_MASK);
                } else if (mod->vec_width_bytes == 32) {
                    // vmovaps [gpr], ymm - VEX encoded
                    uint8_t vex_r = (xmm_src >= 8) ? 0 : 1;
                    uint8_t vex_b = (base_gpr >= 8) ? 0 : 1;
                    
                    uint8_t vex[3] = {
                        0xC4,
                        (uint8_t)(0x01 | (vex_r << 7) | (1 << 6) | (vex_b << 5)),
                        (uint8_t)(0x04 | (0xF << 3))  // L=1 for 256-bit
                    };
                    write_bytes(&ptr, vex, 3);
                    
                    uint8_t opcode = 0x29;  // vmovaps store
                    write_bytes(&ptr, &opcode, 1);
                    
                    uint8_t modrm;
                    if (instr->imm == 0 && (base_gpr & 7) != 5) {
                        modrm = ((xmm_src & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;
                            write_bytes(&ptr, &sib, 1);
                        }
                    } else if (instr->imm >= -128 && instr->imm < 128) {
                        // 8-bit displacement (signed)
                        modrm = 0x40 | ((xmm_src & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;
                            write_bytes(&ptr, &sib, 1);
                        }
                        uint8_t disp = (uint8_t)(int8_t)instr->imm;
                        write_bytes(&ptr, &disp, 1);
                    } else {
                        // 32-bit displacement
                        modrm = 0x80 | ((xmm_src & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;
                            write_bytes(&ptr, &sib, 1);
                        }
                        uint32_t disp = (uint32_t)instr->imm;
                        write_bytes(&ptr, (uint8_t*)&disp, 4);
                    }
                } else {
                    // SSE: movaps [gpr], xmm
                    if (xmm_src >= 8 || base_gpr >= 8) {
                        uint8_t rex = 0x40 | ((xmm_src >= 8) ? 0x04 : 0) | ((base_gpr >= 8) ? 0x01 : 0);
                        write_bytes(&ptr, &rex, 1);
                    }
                    
                    uint8_t opcode[2] = {0x0F, 0x29};
                    write_bytes(&ptr, opcode, 2);
                    
                    uint8_t modrm;
                    if (instr->imm == 0 && (base_gpr & 7) != 5) {
                        modrm = ((xmm_src & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;
                            write_bytes(&ptr, &sib, 1);
                        }
                    } else if (instr->imm >= -128 && instr->imm < 128) {
                        modrm = 0x40 | ((xmm_src & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;
                            write_bytes(&ptr, &sib, 1);
                        }
                        uint8_t disp = (uint8_t)(int8_t)instr->imm;
                        write_bytes(&ptr, &disp, 1);
                    } else {
                        modrm = 0x80 | ((xmm_src & 7) << 3) | (base_gpr & 7);
                        write_bytes(&ptr, &modrm, 1);
                        if ((base_gpr & 7) == 4) {
                            uint8_t sib = 0x24;
                            write_bytes(&ptr, &sib, 1);
                        }
                        uint32_t disp = (uint32_t)instr->imm;
                        write_bytes(&ptr, (uint8_t*)&disp, 4);
                    }
                }
                break;
            }

            // ========== INITIALIZATION ==========
            case PVA_SETZERO:
                if (mod->vec_width_bytes == 64) {
                    uint8_t setzero[] = {0x62, 0xF1, 0x7C, 0x48, 0x57, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7))};
                    write_bytes(&ptr, setzero, 6);
                } else if (mod->vec_width_bytes == 32) {
                    // vxorps ymm_dst, ymm_dst, ymm_dst
                    // 2-byte VEX: C5 R.vvvv.L.pp (R=1 for regs<8, L=1 for 256-bit, pp=00)
                    uint8_t vex_byte2 = (uint8_t)(0x84 | ((~instr->dst & 0xF) << 3));  // R=1, vvvv=~dst, L=1, pp=00
                    uint8_t setzero[] = {0xC5, vex_byte2, 0x57, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7))};
                    write_bytes(&ptr, setzero, 4);
                } else {
                    uint8_t setzero[] = {0x0F, 0x57, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7))};
                    write_bytes(&ptr, setzero, 3);
                }
                break;

            case PVA_SETONE: {
                // vpcmpeqd ymm, ymm, ymm - sets all bits to 1
                if (mod->vec_width_bytes >= 32) {
                    // VEX.256.66.0F.WIG 76 /r
                    uint8_t setone[] = {0xC5, (uint8_t)(0xFD ^ ((instr->dst & 7) << 3)), 0x76, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7))};
                    write_bytes(&ptr, setone, 4);
                } else {
                    // pcmpeqd xmm, xmm (sets all bits to 1)
                    uint8_t setone[] = {0x66, 0x0F, 0x76, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7))};
                    write_bytes(&ptr, setone, 4);
                }
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
