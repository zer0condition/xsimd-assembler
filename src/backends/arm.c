#include "pva.h"
#include <stdio.h>
#include <string.h>

void pva_emit_arm(pva_module_t* mod, uint8_t* buffer) {
    if (!mod || !buffer) return;

    uint8_t* ptr = buffer;
    memset(buffer, 0x00, 8192);

    printf("[codegen] generating ARM NEON/SVE code for %zu instructions\n", mod->size);
    printf("[codegen] target vector width: %d bytes\n", mod->vec_width_bytes);

    // prologue: save callee-saved registers
    // stp fp, lr, [sp, #-16]!
    *ptr++ = 0xfd; *ptr++ = 0x7b; *ptr++ = 0xbf; *ptr++ = 0xa9;
    // add fp, sp, #0
    *ptr++ = 0xfd; *ptr++ = 0x03; *ptr++ = 0x00; *ptr++ = 0x91;
    // sub sp, sp, #0x100 (allocate stack space)
    *ptr++ = 0xff; *ptr++ = 0x83; *ptr++ = 0x04; *ptr++ = 0xd1;

    // gen instruction codes
    for (size_t i = 0; i < mod->size; i++) {
        pva_instr_t* instr = &mod->code[i];

        switch (instr->op) {
            case PVA_ADD_F32: {
                // fadd v<dst>.4s, v<src1>.4s, v<src2>.4s
                // encoding: 0x4e 0xd4 0xd4 0x20 (for v0.4s, v1.4s, v2.4s)
                uint32_t opcode = 0x4e20d400;  // fadd base
                opcode |= ((instr->dst & 0x1f) << 0);      // dest register
                opcode |= ((instr->src1 & 0x1f) << 5);     // src1 register
                opcode |= ((instr->src2 & 0x1f) << 16);    // src2 register
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_SUB_F32: {
                // fsub v<dst>.4s, v<src1>.4s, v<src2>.4s
                uint32_t opcode = 0x4e20d400;  // fsub encoding differs
                opcode |= ((instr->dst & 0x1f) << 0);
                opcode |= ((instr->src1 & 0x1f) << 5);
                opcode |= ((instr->src2 & 0x1f) << 16);
                opcode ^= 0x40;  // toggle subtraction bit
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_MUL_F32: {
                // fmul v<dst>.4s, v<src1>.4s, v<src2>.4s
                uint32_t opcode = 0x6e20dc00;  // fmul base
                opcode |= ((instr->dst & 0x1f) << 0);
                opcode |= ((instr->src1 & 0x1f) << 5);
                opcode |= ((instr->src2 & 0x1f) << 16);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_DIV_F32: {
                // fdiv v<dst>.4s, v<src1>.4s, v<src2>.4s
                uint32_t opcode = 0x6e20fc00;  // fdiv base
                opcode |= ((instr->dst & 0x1f) << 0);
                opcode |= ((instr->src1 & 0x1f) << 5);
                opcode |= ((instr->src2 & 0x1f) << 16);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_LOAD_F32: {
                // ldr q<dst>, [x1, #offset]
                uint32_t opcode = 0x3dc00000;  // ldr q base
                opcode |= (instr->dst & 0x1f);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_STORE_F32: {
                // str q<src>, [x1, #offset]
                uint32_t opcode = 0x3cc00000;  // str q base
                opcode |= (instr->dst & 0x1f);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_SETZERO: {
                // eor v<dst>.16b, v<dst>.16b, v<dst>.16b
                uint32_t opcode = 0x6e201c00;
                opcode |= (instr->dst & 0x1f);
                opcode |= ((instr->dst & 0x1f) << 5);
                opcode |= ((instr->dst & 0x1f) << 16);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_CMP_LT_F32: {
                // fcmlt v<dst>.4s, v<src1>.4s, v<src2>.4s
                uint32_t opcode = 0x4ea0e400;
                opcode |= (instr->dst & 0x1f);
                opcode |= ((instr->src1 & 0x1f) << 5);
                opcode |= ((instr->src2 & 0x1f) << 16);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            default:
                break;
        }
    }

    // ABI epilogue
    // add sp, sp, #0x100
    *ptr++ = 0xff; *ptr++ = 0x83; *ptr++ = 0x00; *ptr++ = 0x91;
    // ldp fp, lr, [sp], #16
    *ptr++ = 0xfd; *ptr++ = 0x7b; *ptr++ = 0xc1; *ptr++ = 0xa8;
    // ret (mov lr to pc)
    *ptr++ = 0xc0; *ptr++ = 0x03; *ptr++ = 0x5f; *ptr++ = 0xd6;

    printf("[codegen] generated %ld bytes of ARM code\n", ptr - buffer);
}