#include "pva.h"
#include <stdio.h>
#include <string.h>

// helper to emit a 32-bit ARM instruction in little-endian
static void emit_arm_instr(uint8_t** ptr, uint32_t opcode) {
    *(*ptr)++ = (opcode >> 0) & 0xff;
    *(*ptr)++ = (opcode >> 8) & 0xff;
    *(*ptr)++ = (opcode >> 16) & 0xff;
    *(*ptr)++ = (opcode >> 24) & 0xff;
}

// NEON 3-operand float instruction helper
static void emit_neon_f32_3op(uint8_t** ptr, uint32_t base, uint8_t dst, uint8_t src1, uint8_t src2) {
    uint32_t opcode = base;
    opcode |= ((dst & 0x1f) << 0);
    opcode |= ((src1 & 0x1f) << 5);
    opcode |= ((src2 & 0x1f) << 16);
    emit_arm_instr(ptr, opcode);
}

// NEON 2-operand instruction helper
static void emit_neon_f32_2op(uint8_t** ptr, uint32_t base, uint8_t dst, uint8_t src) {
    uint32_t opcode = base;
    opcode |= ((dst & 0x1f) << 0);
    opcode |= ((src & 0x1f) << 5);
    emit_arm_instr(ptr, opcode);
}

size_t pva_emit_arm(pva_module_t* mod, uint8_t* buffer) {
    if (!mod || !buffer) return 0;

    uint8_t* ptr = buffer;
    uint8_t* start = buffer;

    printf("[codegen] generating ARM NEON/SVE code for %zu instructions\n", mod->size);
    printf("[codegen] target vector width: %d bytes\n", mod->vec_width_bytes);

    // prologue: save callee-saved registers
    // stp fp, lr, [sp, #-16]!
    emit_arm_instr(&ptr, 0xa9bf7bfd);
    // mov fp, sp
    emit_arm_instr(&ptr, 0x910003fd);
    // sub sp, sp, #0x100
    emit_arm_instr(&ptr, 0xd10403ff);

    for (size_t i = 0; i < mod->size; i++) {
        pva_instr_t* instr = &mod->code[i];

        switch (instr->op) {
            // ========== ARITHMETIC F32 ==========
            case PVA_ADD_F32:
                // fadd v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x4e20d400, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_SUB_F32:
                // fsub v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x4ea0d400, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_MUL_F32:
                // fmul v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x6e20dc00, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_DIV_F32:
                // fdiv v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x6e20fc00, instr->dst, instr->src1, instr->src2);
                break;

            // ========== ARITHMETIC F64 ==========
            case PVA_ADD_F64:
                // fadd v<dst>.2d, v<src1>.2d, v<src2>.2d
                emit_neon_f32_3op(&ptr, 0x4e60d400, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_SUB_F64:
                emit_neon_f32_3op(&ptr, 0x4ee0d400, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_MUL_F64:
                emit_neon_f32_3op(&ptr, 0x6e60dc00, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_DIV_F64:
                emit_neon_f32_3op(&ptr, 0x6e60fc00, instr->dst, instr->src1, instr->src2);
                break;

            // ========== ARITHMETIC I32 ==========
            case PVA_ADD_I32:
                // add v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x4ea08400, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_SUB_I32:
                // sub v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x6ea08400, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_MUL_I32:
                // mul v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x4ea09c00, instr->dst, instr->src1, instr->src2);
                break;

            // ========== MATH OPERATIONS ==========
            case PVA_SQRT_F32:
                // fsqrt v<dst>.4s, v<src>.4s
                emit_neon_f32_2op(&ptr, 0x6ea1f800, instr->dst, instr->src1);
                break;

            case PVA_ABS_F32:
                // fabs v<dst>.4s, v<src>.4s
                emit_neon_f32_2op(&ptr, 0x4ea0f800, instr->dst, instr->src1);
                break;

            case PVA_NEG_F32:
                // fneg v<dst>.4s, v<src>.4s
                emit_neon_f32_2op(&ptr, 0x6ea0f800, instr->dst, instr->src1);
                break;

            case PVA_MIN_F32:
                // fmin v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x4ea0f400, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_MAX_F32:
                // fmax v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x4e20f400, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_FMA_F32:
                // fmla v<dst>.4s, v<src1>.4s, v<src2>.4s (accumulate into dst)
                emit_neon_f32_3op(&ptr, 0x4e20cc00, instr->dst, instr->src1, instr->src2);
                break;

            // ========== COMPARISONS ==========
            case PVA_CMP_LT_F32:
                // fcmgt v<dst>.4s, v<src2>.4s, v<src1>.4s (swap for LT)
                emit_neon_f32_3op(&ptr, 0x6ea0e400, instr->dst, instr->src2, instr->src1);
                break;

            case PVA_CMP_LE_F32:
                // fcmge v<dst>.4s, v<src2>.4s, v<src1>.4s (swap for LE)
                emit_neon_f32_3op(&ptr, 0x6e20e400, instr->dst, instr->src2, instr->src1);
                break;

            case PVA_CMP_GT_F32:
                // fcmgt v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x6ea0e400, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_CMP_GE_F32:
                // fcmge v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x6e20e400, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_CMP_EQ_F32:
                // fcmeq v<dst>.4s, v<src1>.4s, v<src2>.4s
                emit_neon_f32_3op(&ptr, 0x4e20e400, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_CMP_NE_F32:
                // fcmeq then NOT for not-equal
                emit_neon_f32_3op(&ptr, 0x4e20e400, instr->dst, instr->src1, instr->src2);
                emit_neon_f32_2op(&ptr, 0x6e205800, instr->dst, instr->dst);
                break;

            // ========== BITWISE ==========
            case PVA_AND_MASK:
            case PVA_AND_I32:
                // and v<dst>.16b, v<src1>.16b, v<src2>.16b
                emit_neon_f32_3op(&ptr, 0x4e201c00, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_OR_MASK:
            case PVA_OR_I32:
                // orr v<dst>.16b, v<src1>.16b, v<src2>.16b
                emit_neon_f32_3op(&ptr, 0x4ea01c00, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_XOR_MASK:
            case PVA_XOR_I32:
                // eor v<dst>.16b, v<src1>.16b, v<src2>.16b
                emit_neon_f32_3op(&ptr, 0x6e201c00, instr->dst, instr->src1, instr->src2);
                break;

            case PVA_NOT_MASK:
            case PVA_NOT_I32:
                // not v<dst>.16b, v<src>.16b (mvn)
                emit_neon_f32_2op(&ptr, 0x6e205800, instr->dst, instr->src1);
                break;

            case PVA_SHL_I32: {
                // shl v<dst>.4s, v<src>.4s, #imm
                uint32_t opcode = 0x4f205400 | (instr->dst & 0x1f) | ((instr->src1 & 0x1f) << 5) | ((instr->imm & 0x1f) << 16);
                emit_arm_instr(&ptr, opcode);
                break;
            }

            case PVA_SHR_I32: {
                // ushr v<dst>.4s, v<src>.4s, #imm
                uint32_t opcode = 0x6f200400 | (instr->dst & 0x1f) | ((instr->src1 & 0x1f) << 5) | (((32 - instr->imm) & 0x1f) << 16);
                emit_arm_instr(&ptr, opcode);
                break;
            }

            // ========== DATA MOVEMENT ==========
            case PVA_MOV:
                // mov v<dst>.16b, v<src>.16b (orr with itself)
                emit_neon_f32_3op(&ptr, 0x4ea01c00, instr->dst, instr->src1, instr->src1);
                break;

            case PVA_BROADCAST_F32: {
                // dup v<dst>.4s, v<src>.s[0]
                uint32_t opcode = 0x4e040400 | (instr->dst & 0x1f) | ((instr->src1 & 0x1f) << 5);
                emit_arm_instr(&ptr, opcode);
                break;
            }

            // ========== TYPE CONVERSION ==========
            case PVA_CVT_F32_I32:
                // scvtf v<dst>.4s, v<src>.4s
                emit_neon_f32_2op(&ptr, 0x4e21d800, instr->dst, instr->src1);
                break;

            case PVA_CVT_I32_F32:
                // fcvtzs v<dst>.4s, v<src>.4s
                emit_neon_f32_2op(&ptr, 0x4ea1b800, instr->dst, instr->src1);
                break;

            // ========== MEMORY ==========
            case PVA_LOAD_F32:
            case PVA_LOAD_F64:
            case PVA_LOAD_I32: {
                // ldr q<dst>, [x<base>, #offset]
                uint32_t opcode = 0x3dc00000;
                opcode |= (instr->dst & 0x1f);
                opcode |= ((instr->src1 & 0x1f) << 5);
                opcode |= (((instr->imm >> 4) & 0xfff) << 10);  // scaled offset
                emit_arm_instr(&ptr, opcode);
                break;
            }

            case PVA_STORE_F32:
            case PVA_STORE_F64:
            case PVA_STORE_I32: {
                // str q<src>, [x<base>, #offset]
                uint32_t opcode = 0x3d800000;
                opcode |= (instr->dst & 0x1f);
                opcode |= ((instr->src1 & 0x1f) << 5);
                opcode |= (((instr->imm >> 4) & 0xfff) << 10);
                emit_arm_instr(&ptr, opcode);
                break;
            }

            // ========== INITIALIZATION ==========
            case PVA_SETZERO: {
                // eor v<dst>.16b, v<dst>.16b, v<dst>.16b
                emit_neon_f32_3op(&ptr, 0x6e201c00, instr->dst, instr->dst, instr->dst);
                break;
            }

            case PVA_SETONE: {
                // cmpeq v<dst>.4s, v<dst>.4s, v<dst>.4s (all bits = 1)
                emit_neon_f32_3op(&ptr, 0x6ea08c00, instr->dst, instr->dst, instr->dst);
                break;
            }

            // ========== CONTROL FLOW ==========
            case PVA_RET:
                // ret
                emit_arm_instr(&ptr, 0xd65f03c0);
                break;

            default:
                break;
        }
    }

    // ABI epilogue
    // add sp, sp, #0x100
    emit_arm_instr(&ptr, 0x910403ff);
    // ldp fp, lr, [sp], #16
    emit_arm_instr(&ptr, 0xa8c17bfd);
    // ret
    emit_arm_instr(&ptr, 0xd65f03c0);

    size_t code_size = ptr - start;
    printf("[codegen] generated %zu bytes of ARM code\n", code_size);
    return code_size;
}