#include "pva.h"
#include <stdio.h>
#include <string.h>

void pva_emit_x86(pva_module_t* mod, uint8_t* buffer) {
    if (!mod || !buffer) return;

    uint8_t* ptr = buffer;
    memset(buffer, 0x90, 8192);  // NOP fill

    printf("[codegen] generating x86 code for %zu instructions\n", mod->size);
    printf("[codegen] target vector width: %d bytes\n", mod->vec_width_bytes);

    // prologue: push rbp; mov rbp, rsp; sub rsp, 0x20
    *ptr++ = 0x55;                          // push rbp
    *ptr++ = 0x48; *ptr++ = 0x89; *ptr++ = 0xe5;  // mov rbp, rsp
    *ptr++ = 0x48; *ptr++ = 0x83; *ptr++ = 0xec; *ptr++ = 0x20;  // sub rsp, 0x20

    // gen instruction codes
    for (size_t i = 0; i < mod->size; i++) {
        pva_instr_t* instr = &mod->code[i];

        switch (instr->op) {
            case PVA_ADD_F32:
                if (mod->vec_width_bytes == 64) {
                    // vaddps zmm0, zmm1, zmm2
                    *ptr++ = 0x62; *ptr++ = 0xf2; *ptr++ = 0x7d; *ptr++ = 0x48;
                    *ptr++ = 0x59; *ptr++ = 0xc2;
                } else if (mod->vec_width_bytes == 32) {
                    // vaddps ymm0, ymm1, ymm2
                    *ptr++ = 0xc5; *ptr++ = 0xf4; *ptr++ = 0x58; *ptr++ = 0xc2;
                } else {
                    // addps xmm0, xmm1
                    *ptr++ = 0x0f; *ptr++ = 0x58; *ptr++ = 0xc1;
                }
                break;

            case PVA_SUB_F32:
                if (mod->vec_width_bytes == 64) {
                    *ptr++ = 0x62; *ptr++ = 0xf2; *ptr++ = 0x7d; *ptr++ = 0x48;
                    *ptr++ = 0x5c; *ptr++ = 0xc2;  // vsubps
                } else if (mod->vec_width_bytes == 32) {
                    *ptr++ = 0xc5; *ptr++ = 0xf4; *ptr++ = 0x5c; *ptr++ = 0xc2;  // vsubps ymm
                } else {
                    *ptr++ = 0x0f; *ptr++ = 0x5c; *ptr++ = 0xc1;  // subps xmm
                }
                break;

            case PVA_MUL_F32:
                if (mod->vec_width_bytes == 64) {
                    *ptr++ = 0x62; *ptr++ = 0xf2; *ptr++ = 0x7d; *ptr++ = 0x48;
                    *ptr++ = 0x59; *ptr++ = 0xc2;  // vmulps
                } else if (mod->vec_width_bytes == 32) {
                    *ptr++ = 0xc5; *ptr++ = 0xf4; *ptr++ = 0x59; *ptr++ = 0xc2;  // vmulps ymm
                } else {
                    *ptr++ = 0x0f; *ptr++ = 0x59; *ptr++ = 0xc1;  // mulps xmm
                }
                break;

            case PVA_LOAD_F32:
                if (mod->vec_width_bytes == 64) {
                    *ptr++ = 0x62; *ptr++ = 0xf2; *ptr++ = 0x7d; *ptr++ = 0x48;
                    *ptr++ = 0x10; *ptr++ = 0x06;  // vmovaps zmm0, [rsi]
                } else if (mod->vec_width_bytes == 32) {
                    *ptr++ = 0xc5; *ptr++ = 0xfc; *ptr++ = 0x28; *ptr++ = 0x06;  // vmovaps ymm0, [rsi]
                } else {
                    *ptr++ = 0x0f; *ptr++ = 0x28; *ptr++ = 0x06;  // movaps xmm0, [rsi]
                }
                break;

            case PVA_STORE_F32:
                if (mod->vec_width_bytes == 64) {
                    *ptr++ = 0x62; *ptr++ = 0xf2; *ptr++ = 0x7d; *ptr++ = 0x48;
                    *ptr++ = 0x11; *ptr++ = 0x06;  // vmovaps [rsi], zmm0
                } else if (mod->vec_width_bytes == 32) {
                    *ptr++ = 0xc5; *ptr++ = 0xfc; *ptr++ = 0x29; *ptr++ = 0x06;  // vmovaps [rsi], ymm0
                } else {
                    *ptr++ = 0x0f; *ptr++ = 0x29; *ptr++ = 0x06;  // movaps [rsi], xmm0
                }
                break;

            case PVA_SETZERO:
                if (mod->vec_width_bytes == 64) {
                    *ptr++ = 0x62; *ptr++ = 0xf2; *ptr++ = 0x7d; *ptr++ = 0x48;
                    *ptr++ = 0x57; *ptr++ = 0xc0;  // vxorps zmm0, zmm0, zmm0
                } else if (mod->vec_width_bytes == 32) {
                    *ptr++ = 0xc5; *ptr++ = 0xf4; *ptr++ = 0x57; *ptr++ = 0xc0;  // vxorps ymm0, ymm0, ymm0
                } else {
                    *ptr++ = 0x0f; *ptr++ = 0x57; *ptr++ = 0xc0;  // xorps xmm0, xmm0
                }
                break;

            default:
                break;
        }
    }

    // epilogue: mov rax, 0; mov rsp, rbp; pop rbp; ret
    *ptr++ = 0x48; *ptr++ = 0xc7; *ptr++ = 0xc0; *ptr++ = 0x00; *ptr++ = 0x00;
    *ptr++ = 0x00; *ptr++ = 0x00;  // mov rax, 0
    *ptr++ = 0x48; *ptr++ = 0x89; *ptr++ = 0xec;  // mov rsp, rbp
    *ptr++ = 0x5d;  // pop rbp
    *ptr++ = 0xc3;  // ret

    printf("[codegen] generated %ld bytes of x86 code\n", ptr - buffer);
}