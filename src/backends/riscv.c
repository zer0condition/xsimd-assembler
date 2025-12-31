#include "pva.h"
#include <stdio.h>
#include <string.h>

// helper to emit a 32-bit RISC-V instruction in little-endian
static void emit_rv_instr(uint8_t** ptr, uint32_t opcode) {
    *(*ptr)++ = (opcode >> 0) & 0xff;
    *(*ptr)++ = (opcode >> 8) & 0xff;
    *(*ptr)++ = (opcode >> 16) & 0xff;
    *(*ptr)++ = (opcode >> 24) & 0xff;
}

// RVV vector-vector float operation helper
static void emit_rvv_vvf(uint8_t** ptr, uint32_t base, uint8_t vd, uint8_t vs1, uint8_t vs2) {
    uint32_t opcode = base;
    opcode |= ((vd & 0x1f) << 7);
    opcode |= ((vs1 & 0x1f) << 15);
    opcode |= ((vs2 & 0x1f) << 20);
    emit_rv_instr(ptr, opcode);
}

// RVV single-source operation helper
static void emit_rvv_v1(uint8_t** ptr, uint32_t base, uint8_t vd, uint8_t vs) {
    uint32_t opcode = base;
    opcode |= ((vd & 0x1f) << 7);
    opcode |= ((vs & 0x1f) << 15);
    emit_rv_instr(ptr, opcode);
}

size_t pva_emit_riscv(pva_module_t* mod, uint8_t* buffer) {
    if (!mod || !buffer) return 0;

    uint8_t* ptr = buffer;
    uint8_t* start = buffer;

    printf("[codegen] generating RISC-V RVV code for %zu instructions\n", mod->size);
    printf("[codegen] target vector width: %d bytes\n", mod->vec_width_bytes);

    // prologue
    // addi sp, sp, -16
    emit_rv_instr(&ptr, 0xff010113);
    // sd ra, 8(sp)
    emit_rv_instr(&ptr, 0x00113423);

    // init vector length (setvl instruction)
    // vsetvli t0, x0, e32, m1 (set 32-bit elements, LMUL=1)
    emit_rv_instr(&ptr, 0xc0007257);

    for (size_t i = 0; i < mod->size; i++) {
        pva_instr_t* instr = &mod->code[i];

        switch (instr->op) {
            // ========== ARITHMETIC F32 ==========
            case PVA_ADD_F32:
                // vfadd.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x00001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_SUB_F32:
                // vfsub.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x08001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_MUL_F32:
                // vfmul.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x90001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_DIV_F32:
                // vfdiv.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x80001057, instr->dst, instr->src1, instr->src2);
                break;

            // ========== ARITHMETIC F64 ==========
            case PVA_ADD_F64:
                // vfadd.vv (with e64 SEW)
                emit_rvv_vvf(&ptr, 0x00001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_SUB_F64:
                emit_rvv_vvf(&ptr, 0x08001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_MUL_F64:
                emit_rvv_vvf(&ptr, 0x90001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_DIV_F64:
                emit_rvv_vvf(&ptr, 0x80001057, instr->dst, instr->src1, instr->src2);
                break;

            // ========== ARITHMETIC I32 ==========
            case PVA_ADD_I32:
                // vadd.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x00000057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_SUB_I32:
                // vsub.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x08000057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_MUL_I32:
                // vmul.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x94002057, instr->dst, instr->src1, instr->src2);
                break;

            // ========== MATH OPERATIONS ==========
            case PVA_SQRT_F32:
                // vfsqrt.v vd, vs
                emit_rvv_v1(&ptr, 0x4c001057, instr->dst, instr->src1);
                break;

            case PVA_MIN_F32:
                // vfmin.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x10001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_MAX_F32:
                // vfmax.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x18001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_FMA_F32:
                // vfmacc.vv vd, vs1, vs2 (vd = vd + vs1 * vs2)
                emit_rvv_vvf(&ptr, 0xb0001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_ABS_F32:
                // vfsgnjx.vv vd, vs, vs (abs via sign injection)
                emit_rvv_vvf(&ptr, 0x24001057, instr->dst, instr->src1, instr->src1);
                break;

            case PVA_NEG_F32:
                // vfsgnjn.vv vd, vs, vs (neg via negated sign injection)
                emit_rvv_vvf(&ptr, 0x20001057, instr->dst, instr->src1, instr->src1);
                break;

            // ========== COMPARISONS ==========
            case PVA_CMP_LT_F32:
                // vmflt.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x6c001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_CMP_LE_F32:
                // vmfle.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x64001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_CMP_GT_F32:
                // vmflt.vv vd, vs2, vs1 (swap operands)
                emit_rvv_vvf(&ptr, 0x6c001057, instr->dst, instr->src2, instr->src1);
                break;

            case PVA_CMP_GE_F32:
                // vmfle.vv vd, vs2, vs1 (swap operands)
                emit_rvv_vvf(&ptr, 0x64001057, instr->dst, instr->src2, instr->src1);
                break;

            case PVA_CMP_EQ_F32:
                // vmfeq.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x60001057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_CMP_NE_F32:
                // vmfne.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x70001057, instr->dst, instr->src1, instr->src2);
                break;

            // ========== BITWISE ==========
            case PVA_AND_MASK:
            case PVA_AND_I32:
                // vand.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x24000057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_OR_MASK:
            case PVA_OR_I32:
                // vor.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x28000057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_XOR_MASK:
            case PVA_XOR_I32:
                // vxor.vv vd, vs1, vs2
                emit_rvv_vvf(&ptr, 0x2c000057, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_NOT_MASK:
            case PVA_NOT_I32:
                // vnot.v vd, vs (pseudo: vxor.vi vd, vs, -1)
                emit_rvv_v1(&ptr, 0x2c003057 | (0x1f << 20), instr->dst, instr->src1);
                break;

            case PVA_SHL_I32: {
                // vsll.vi vd, vs, imm
                uint32_t opcode = 0x94003057;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                opcode |= ((instr->imm & 0x1f) << 20);
                emit_rv_instr(&ptr, opcode);
                break;
            }

            case PVA_SHR_I32: {
                // vsrl.vi vd, vs, imm
                uint32_t opcode = 0xa0003057;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                opcode |= ((instr->imm & 0x1f) << 20);
                emit_rv_instr(&ptr, opcode);
                break;
            }

            case PVA_SAR_I32: {
                // vsra.vi vd, vs, imm
                uint32_t opcode = 0xa4003057;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                opcode |= ((instr->imm & 0x1f) << 20);
                emit_rv_instr(&ptr, opcode);
                break;
            }

            // ========== DATA MOVEMENT ==========
            case PVA_MOV:
                // vmv.v.v vd, vs
                emit_rvv_v1(&ptr, 0x5e000057, instr->dst, instr->src1);
                break;

            case PVA_BROADCAST_F32: {
                // vfmv.v.f vd, fs (broadcast scalar)
                uint32_t opcode = 0x5e005057;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                emit_rv_instr(&ptr, opcode);
                break;
            }

            // ========== TYPE CONVERSION ==========
            case PVA_CVT_F32_I32:
                // vfcvt.f.x.v vd, vs (int to float)
                emit_rvv_v1(&ptr, 0x48019057, instr->dst, instr->src1);
                break;

            case PVA_CVT_I32_F32:
                // vfcvt.x.f.v vd, vs (float to int, truncate)
                emit_rvv_v1(&ptr, 0x48011057, instr->dst, instr->src1);
                break;

            // ========== MEMORY ==========
            case PVA_LOAD_F32:
            case PVA_LOAD_I32: {
                // vle32.v vd, (rs1)
                uint32_t opcode = 0x02006007;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                emit_rv_instr(&ptr, opcode);
                break;
            }

            case PVA_LOAD_F64: {
                // vle64.v vd, (rs1)
                uint32_t opcode = 0x02007007;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                emit_rv_instr(&ptr, opcode);
                break;
            }

            case PVA_STORE_F32:
            case PVA_STORE_I32: {
                // vse32.v vs, (rs1)
                uint32_t opcode = 0x02006027;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                emit_rv_instr(&ptr, opcode);
                break;
            }

            case PVA_STORE_F64: {
                // vse64.v vs, (rs1)
                uint32_t opcode = 0x02007027;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                emit_rv_instr(&ptr, opcode);
                break;
            }

            // ========== INITIALIZATION ==========
            case PVA_SETZERO: {
                // vmv.v.x vd, x0 (move zero to vector)
                uint32_t opcode = 0x5e004057;
                opcode |= ((instr->dst & 0x1f) << 7);
                emit_rv_instr(&ptr, opcode);
                break;
            }

            case PVA_SETONE: {
                // vmv.v.i vd, -1 (all ones)
                uint32_t opcode = 0x5e003057 | (0x1f << 15);
                opcode |= ((instr->dst & 0x1f) << 7);
                emit_rv_instr(&ptr, opcode);
                break;
            }

            // ========== CONTROL FLOW ==========
            case PVA_RET:
                // ret (jalr x0, ra, 0)
                emit_rv_instr(&ptr, 0x00008067);
                break;

            default:
                break;
        }
    }

    // epilogue
    // ld ra, 8(sp)
    emit_rv_instr(&ptr, 0x00813083);
    // addi sp, sp, 16
    emit_rv_instr(&ptr, 0x01010113);
    // ret
    emit_rv_instr(&ptr, 0x00008067);

    size_t code_size = ptr - start;
    printf("[codegen] generated %zu bytes of RISC-V RVV code\n", code_size);
    return code_size;
}