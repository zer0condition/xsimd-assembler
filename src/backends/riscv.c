#include "pva.h"
#include <stdio.h>
#include <string.h>

void pva_emit_riscv(pva_module_t* mod, uint8_t* buffer) {
    if (!mod || !buffer) return;

    uint8_t* ptr = buffer;
    memset(buffer, 0x00, 8192);

    printf("[codegen] generating RISC-V RVV code for %zu instructions\n", mod->size);
    printf("[codegen] target vector width: %d bytes\n", mod->vec_width_bytes);

    // prologue
    // addi sp, sp, -16
    *ptr++ = 0x13; *ptr++ = 0x01; *ptr++ = 0x01; *ptr++ = 0xff;
    // sd ra, 8(sp)
    *ptr++ = 0x23; *ptr++ = 0x34; *ptr++ = 0x11; *ptr++ = 0x00;

    // init vector length (setvl instruction)
    // set VL based on vector width
    // vsetvli t0, x0, e32, m1  (set 32-bit elements, LMUL=1)
    *ptr++ = 0x57; *ptr++ = 0x72; *ptr++ = 0x00; *ptr++ = 0xc0;

    // gen instruction codes
    for (size_t i = 0; i < mod->size; i++) {
        pva_instr_t* instr = &mod->code[i];

        switch (instr->op) {
            case PVA_ADD_F32: {
                // vfadd.vv v<dst>, v<src1>, v<src2>
                // encoding: opcode[6:0]=0x57, funct3=0x10, funct6=0x00
                uint32_t opcode = 0x00001057;
                opcode |= ((instr->dst & 0x1f) << 7);       // vd
                opcode |= ((instr->src1 & 0x1f) << 15);     // vs1
                opcode |= ((instr->src2 & 0x1f) << 20);     // vs2
                opcode |= (0 << 25);                         // vm=0 (unmasked)
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_SUB_F32: {
                // vfsub.vv v<dst>, v<src1>, v<src2>
                // encoding: funct6=0x08 for subtract
                uint32_t opcode = 0x08001057;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                opcode |= ((instr->src2 & 0x1f) << 20);
                opcode |= (0 << 25);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_MUL_F32: {
                // vfmul.vv v<dst>, v<src1>, v<src2>
                // encoding: funct6=0x10 for multiply
                uint32_t opcode = 0x10001057;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                opcode |= ((instr->src2 & 0x1f) << 20);
                opcode |= (0 << 25);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_DIV_F32: {
                // vfdiv.vv v<dst>, v<src1>, v<src2>
                // encoding: funct6=0x18 for divide
                uint32_t opcode = 0x18001057;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                opcode |= ((instr->src2 & 0x1f) << 20);
                opcode |= (0 << 25);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_LOAD_F32: {
                // vle32.v v<dst>, (x1)
                // unit-stride load encoding
                uint32_t opcode = 0x06000007;  // vle32.v base
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= (1 << 20);  // x1 register
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_STORE_F32: {
                // vse32.v v<src>, (x1)
                // unit-stride store encoding
                uint32_t opcode = 0x04000027;  // vse32.v base
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= (1 << 20);  // x1 register
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_SETZERO: {
                // vmv.v.x v<dst>, x0 (move immediate/scalar to vector)
                // uses x0 (zero) as source
                uint32_t opcode = 0x40005057;
                opcode |= ((instr->dst & 0x1f) << 7);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_CMP_LT_F32: {
                // vmflt.vv v<dst>, v<src1>, v<src2>
                // floating-point less-than comparison
                uint32_t opcode = 0x6e005057;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                opcode |= ((instr->src2 & 0x1f) << 20);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_AND_MASK: {
                // vand.vv v<dst>, v<src1>, v<src2>
                // logical AND for masks
                uint32_t opcode = 0x24001057;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                opcode |= ((instr->src2 & 0x1f) << 20);
                
                *ptr++ = (opcode >> 0) & 0xff;
                *ptr++ = (opcode >> 8) & 0xff;
                *ptr++ = (opcode >> 16) & 0xff;
                *ptr++ = (opcode >> 24) & 0xff;
                break;
            }

            case PVA_OR_MASK: {
                // vor.vv v<dst>, v<src1>, v<src2>
                // logical OR for masks
                uint32_t opcode = 0x28001057;
                opcode |= ((instr->dst & 0x1f) << 7);
                opcode |= ((instr->src1 & 0x1f) << 15);
                opcode |= ((instr->src2 & 0x1f) << 20);
                
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

    // epilogue
    // ld ra, 8(sp)
    *ptr++ = 0x83; *ptr++ = 0x30; *ptr++ = 0x81; *ptr++ = 0x00;
    // addi sp, sp, 16
    *ptr++ = 0x13; *ptr++ = 0x01; *ptr++ = 0x01; *ptr++ = 0x01;
    // jalr x0, x1, 0 (ret)
    *ptr++ = 0x67; *ptr++ = 0x80; *ptr++ = 0x00; *ptr++ = 0x00;

    printf("[codegen] generated %ld bytes of RISC-V RVV code\n", ptr - buffer);
}