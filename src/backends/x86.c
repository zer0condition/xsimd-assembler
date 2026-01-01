#include "pva.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define MAX_REGS_SSE 16
#define MAX_REGS_AVX2 16
#define MAX_REGS_AVX512 32
#define FULL_MASK 0x00  // k0 = no masking (all elements active)

/*
 * x86 Instruction Encoding Reference:
 * 
 * VEX Prefix (2-byte): C5 [RvvvvLpp]
 *   R: inverted REX.R (0 = extended reg field)
 *   vvvv: inverted additional source register (1111 = unused)
 *   L: vector length (0 = 128-bit, 1 = 256-bit)
 *   pp: opcode prefix (00=none, 01=66, 10=F3, 11=F2)
 *
 * VEX Prefix (3-byte): C4 [RXBmmmmm] [WvvvvLpp]
 *   R: inverted REX.R (0 = extended reg field)
 *   X: inverted REX.X (0 = extended SIB index)
 *   B: inverted REX.B (0 = extended rm/base)
 *   mmmmm: opcode map (00001=0F, 00010=0F38, 00011=0F3A)
 *   W: REX.W equivalent
 *   vvvv: inverted additional source register
 *   L: vector length
 *   pp: opcode prefix
 *
 * EVEX Prefix (4-byte): 62 [RXBR'0mmm] [Wvvvv1pp] [zL'Lbv'aaa]
 *   R: inverted REX.R (reg[3])
 *   X: inverted REX.X (index[3])
 *   B: inverted REX.B (rm[3]/base[3])
 *   R': inverted reg[4] (high bit for ZMM16-31)
 *   mmm: opcode map (01=0F, 10=0F38, 11=0F3A)
 *   W: operand size (0=32-bit, 1=64-bit for most)
 *   vvvv: inverted src2 register
 *   pp: prefix (00=none, 01=66, 10=F3, 11=F2)
 *   z: zeroing-masking (1=zero, 0=merge)
 *   L'L: vector length (00=128, 01=256, 10=512)
 *   b: broadcast/rounding control
 *   v': inverted vvvv[4] (high bit for ZMM16-31 as src2)
 *   aaa: opmask register k0-k7
 *
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
 *   PVA r0-r31  -> XMM0-XMM31 / YMM0-YMM31 / ZMM0-ZMM31 (AVX-512)
 *   PVA r0-r15  -> XMM0-XMM15 / YMM0-YMM15 (SSE/AVX2)
 *
 * x86 GPR encoding (for memory addressing):
 *   rax=0, rcx=1, rdx=2, rbx=3, rsp=4, rbp=5, rsi=6, rdi=7
 *   r8=8, r9=9, r10=10, r11=11, r12=12, r13=13, r14=14, r15=15
 */

// Opcode map values for VEX/EVEX
#define MAP_0F     0x01
#define MAP_0F38   0x02
#define MAP_0F3A   0x03

// EVEX vector length encoding
#define VL_128     0x00
#define VL_256     0x01
#define VL_512     0x02

// VEX/EVEX prefix types (pp field)
#define PP_NONE    0x00
#define PP_66      0x01
#define PP_F3      0x02
#define PP_F2      0x03

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

static void write_byte(uint8_t** buf, uint8_t byte) {
    **buf = byte;
    (*buf)++;
}

static void write_uint32_le(uint8_t** buf, uint32_t val) {
    write_byte(buf, (uint8_t)(val & 0xFF));
    write_byte(buf, (uint8_t)((val >> 8) & 0xFF));
    write_byte(buf, (uint8_t)((val >> 16) & 0xFF));
    write_byte(buf, (uint8_t)((val >> 24) & 0xFF));
}

/*
 * EVEX Prefix Emitter - Full AVX-512 support
 * 
 * Parameters:
 *   dst: destination register (0-31 for ZMM)
 *   src1: first source register (encoded in vvvv, 0-31)
 *   src2: second source register (in ModRM.rm, 0-31)
 *   map: opcode map (MAP_0F, MAP_0F38, MAP_0F3A)
 *   pp: prefix (PP_NONE, PP_66, PP_F3, PP_F2)
 *   w: W bit (0 or 1, operand size)
 *   vl: vector length (VL_128, VL_256, VL_512)
 *   mask: opmask register k0-k7 (0 = no masking)
 *   zeroing: zeroing-masking (0 = merge, 1 = zero)
 *   broadcast: broadcast bit (0 = no broadcast)
 */
static void emit_evex(uint8_t** pbuf, uint8_t dst, uint8_t src1, uint8_t src2,
                      uint8_t map, uint8_t pp, uint8_t w, uint8_t vl,
                      uint8_t mask, uint8_t zeroing, uint8_t broadcast) {
    // Extract register bits
    uint8_t R  = (~(dst >> 3)) & 1;      // inverted dst[3]
    uint8_t R2 = (~(dst >> 4)) & 1;      // inverted dst[4] (R')
    uint8_t X  = 1;                       // inverted index[3], 1 = not used
    uint8_t B  = (~(src2 >> 3)) & 1;     // inverted src2[3]
    uint8_t V4 = (~(src1 >> 4)) & 1;     // inverted src1[4] (V')
    
    uint8_t vvvv = (~src1) & 0x0F;       // inverted src1[3:0]
    
    uint8_t evex[4];
    evex[0] = 0x62;
    
    // Byte 1: R.X.B.R'.0.0.m.m (mmm = map)
    evex[1] = (R << 7) | (X << 6) | (B << 5) | (R2 << 4) | (map & 0x07);
    
    // Byte 2: W.vvvv.1.pp
    evex[2] = ((w & 1) << 7) | (vvvv << 3) | 0x04 | (pp & 0x03);
    
    // Byte 3: z.L'L.b.V'.aaa
    evex[3] = ((zeroing & 1) << 7) | ((vl & 0x03) << 5) | 
              ((broadcast & 1) << 4) | ((V4 & 1) << 3) | (mask & 0x07);
    
    write_bytes(pbuf, evex, 4);
}

// Emit ModRM byte for register-to-register operations
static void emit_modrm_reg(uint8_t** pbuf, uint8_t reg, uint8_t rm) {
    uint8_t modrm = 0xC0 | ((reg & 0x07) << 3) | (rm & 0x07);
    write_byte(pbuf, modrm);
}

// Emit ModRM + optional SIB + displacement for memory operations
static void emit_modrm_mem(uint8_t** pbuf, uint8_t reg, uint8_t base, int32_t disp) {
    uint8_t base_low = base & 0x07;
    uint8_t reg_low = reg & 0x07;
    
    // Check if we need SIB byte (base = rsp/r12 uses encoding 100)
    int need_sib = (base_low == 4);
    
    if (disp == 0 && base_low != 5 && base_low != 13) {
        // [base] - no displacement (except rbp/r13 which need disp8=0)
        uint8_t modrm = (reg_low << 3) | base_low;
        write_byte(pbuf, modrm);
        if (need_sib) {
            write_byte(pbuf, 0x24);  // SIB: scale=0, index=rsp(none), base=rsp
        }
    } else if (disp >= -128 && disp <= 127) {
        // [base + disp8]
        uint8_t modrm = 0x40 | (reg_low << 3) | base_low;
        write_byte(pbuf, modrm);
        if (need_sib) {
            write_byte(pbuf, 0x24);
        }
        write_byte(pbuf, (uint8_t)(int8_t)disp);
    } else {
        // [base + disp32]
        uint8_t modrm = 0x80 | (reg_low << 3) | base_low;
        write_byte(pbuf, modrm);
        if (need_sib) {
            write_byte(pbuf, 0x24);
        }
        write_uint32_le(pbuf, (uint32_t)disp);
    }
}

/*
 * AVX-512 Instruction Emitter
 * Emits a complete EVEX-encoded instruction with register operands
 */
static void emit_avx512_reg3(uint8_t** pbuf, uint8_t opcode,
                             uint8_t dst, uint8_t src1, uint8_t src2,
                             uint8_t map, uint8_t pp, uint8_t w,
                             uint8_t mask, uint8_t zeroing) {
    emit_evex(pbuf, dst, src1, src2, map, pp, w, VL_512, mask, zeroing, 0);
    write_byte(pbuf, opcode);
    emit_modrm_reg(pbuf, dst, src2);
}

/*
 * AVX-512 2-operand instruction (dst = op(src))
 */
static void emit_avx512_reg2(uint8_t** pbuf, uint8_t opcode,
                             uint8_t dst, uint8_t src,
                             uint8_t map, uint8_t pp, uint8_t w,
                             uint8_t mask, uint8_t zeroing) {
    // For 2-operand, vvvv is unused (1111)
    emit_evex(pbuf, dst, 0, src, map, pp, w, VL_512, mask, zeroing, 0);
    write_byte(pbuf, opcode);
    emit_modrm_reg(pbuf, dst, src);
}

/*
 * AVX-512 Memory Load
 * vmovups/vmovaps zmm, [base + disp]
 */
static void emit_avx512_load_mem(uint8_t** pbuf, uint8_t dst, uint8_t base_gpr,
                                  int32_t disp, uint8_t aligned,
                                  uint8_t mask, uint8_t zeroing) {
    // vmovups: EVEX.512.0F.W0 10 /r (pp=00)
    // vmovaps: EVEX.512.0F.W0 28 /r (pp=00, not 66!)
    // Note: vmovaps is 0F 28 with pp=00, same as vmovups but aligned
    uint8_t opcode = aligned ? 0x28 : 0x10;
    
    // EVEX for memory: B bit uses base_gpr[3], X is 1 (no index)
    uint8_t R  = (~(dst >> 3)) & 1;
    uint8_t R2 = (~(dst >> 4)) & 1;
    uint8_t X  = 1;
    uint8_t B  = (~(base_gpr >> 3)) & 1;
    
    uint8_t evex[4];
    evex[0] = 0x62;
    evex[1] = (R << 7) | (X << 6) | (B << 5) | (R2 << 4) | MAP_0F;
    evex[2] = (0 << 7) | (0xF << 3) | 0x04 | PP_NONE;  // W=0, vvvv=1111, bit2=1, pp=00
    evex[3] = ((zeroing & 1) << 7) | (VL_512 << 5) | (0 << 4) | (mask & 0x07);  // z, L'L=10, b=0, aaa
    
    write_bytes(pbuf, evex, 4);
    write_byte(pbuf, opcode);
    emit_modrm_mem(pbuf, dst, base_gpr, disp);
}

/*
 * AVX-512 Memory Store
 * vmovups/vmovaps [base + disp], zmm
 */
static void emit_avx512_store_mem(uint8_t** pbuf, uint8_t src, uint8_t base_gpr,
                                   int32_t disp, uint8_t aligned, uint8_t mask) {
    // vmovups store: EVEX.512.0F.W0 11 /r
    // vmovaps store: EVEX.512.0F.W0 29 /r
    uint8_t opcode = aligned ? 0x29 : 0x11;
    
    uint8_t R  = (~(src >> 3)) & 1;
    uint8_t R2 = (~(src >> 4)) & 1;
    uint8_t X  = 1;
    uint8_t B  = (~(base_gpr >> 3)) & 1;
    
    uint8_t evex[4];
    evex[0] = 0x62;
    evex[1] = (R << 7) | (X << 6) | (B << 5) | (R2 << 4) | MAP_0F;
    evex[2] = (0 << 7) | (0xF << 3) | 0x04 | PP_NONE;  // W=0, vvvv=1111, bit2=1, pp=00
    evex[3] = (VL_512 << 5) | (0 << 4) | (mask & 0x07);  // z=0, L'L=10, b=0, aaa
    
    write_bytes(pbuf, evex, 4);
    write_byte(pbuf, opcode);
    emit_modrm_mem(pbuf, src, base_gpr, disp);
}

/*
 * Helper for AVX2 2-operand instructions (dst = op src)
 * VEX.256.0F.WIG opcode /r
 * vvvv is unused (set to 1111)
 */
static void emit_avx2_2op(uint8_t** pbuf, uint8_t opcode, uint8_t dst, uint8_t src) {
    // For 2-operand instructions, vvvv must be 1111 (unused)
    // Check if we need 3-byte VEX (for extended registers)
    if (dst >= 8 || src >= 8) {
        // 3-byte VEX: C4 RXB.mmmmm W.vvvv.L.pp
        uint8_t vex_r = (dst < 8) ? 1 : 0;
        uint8_t vex_x = 1;  // not using SIB
        uint8_t vex_b = (src < 8) ? 1 : 0;
        
        uint8_t vex[3] = {
            0xC4,
            (uint8_t)((vex_r << 7) | (vex_x << 6) | (vex_b << 5) | 0x01),  // R.X.B.00001 (0F map)
            (uint8_t)((0 << 7) | (0xF << 3) | (1 << 2) | 0x00)  // W=0, vvvv=1111, L=1(256-bit), pp=00
        };
        write_bytes(pbuf, vex, 3);
    } else {
        // 2-byte VEX: C5 R.vvvv.L.pp
        uint8_t vex[2] = {
            0xC5,
            (uint8_t)((1 << 7) | (0xF << 3) | (1 << 2) | 0x00)  // R=1, vvvv=1111, L=1(256-bit), pp=00
        };
        write_bytes(pbuf, vex, 2);
    }
    
    write_bytes(pbuf, &opcode, 1);
    
    // ModRM: mod=11 (register), reg=dst, rm=src
    uint8_t modrm = 0xC0 | ((dst & 0x7) << 3) | (src & 0x7);
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

/*
 * AVX-512 Float32 packed instruction emitter
 * vaddps, vsubps, vmulps, vdivps, etc.
 * EVEX.512.0F.W0 opcode /r
 */
static void emit_avx512_ps(uint8_t** pbuf, uint8_t opcode,
                           uint8_t dst, uint8_t src1, uint8_t src2,
                           uint8_t mask) {
    emit_avx512_reg3(pbuf, opcode, dst, src1, src2, MAP_0F, PP_NONE, 0, mask, 0);
}

/*
 * AVX-512 Float64 packed instruction emitter  
 * vaddpd, vsubpd, vmulpd, vdivpd, etc.
 * EVEX.512.66.0F.W1 opcode /r
 */
static void emit_avx512_pd(uint8_t** pbuf, uint8_t opcode,
                           uint8_t dst, uint8_t src1, uint8_t src2,
                           uint8_t mask) {
    emit_avx512_reg3(pbuf, opcode, dst, src1, src2, MAP_0F, PP_66, 1, mask, 0);
}

/*
 * AVX-512 Integer32 packed instruction emitter (66.0F map)
 * vpaddd, vpsubd, etc.
 * EVEX.512.66.0F.W0 opcode /r
 */
static void emit_avx512_d_0f(uint8_t** pbuf, uint8_t opcode,
                              uint8_t dst, uint8_t src1, uint8_t src2,
                              uint8_t mask) {
    emit_avx512_reg3(pbuf, opcode, dst, src1, src2, MAP_0F, PP_66, 0, mask, 0);
}

/*
 * AVX-512 Integer32 packed instruction emitter (66.0F38 map)
 * vpmulld, vpminsd, vpmaxsd, etc.
 * EVEX.512.66.0F38.W0 opcode /r
 */
static void emit_avx512_d_0f38(uint8_t** pbuf, uint8_t opcode,
                                uint8_t dst, uint8_t src1, uint8_t src2,
                                uint8_t mask) {
    emit_avx512_reg3(pbuf, opcode, dst, src1, src2, MAP_0F38, PP_66, 0, mask, 0);
}

/*
 * AVX-512 Compare instruction emitter
 * vcmpps with immediate
 * EVEX.512.0F.W0 C2 /r ib
 * Note: This writes to a mask register (k0-k7)
 */
static void emit_avx512_cmpps_to_mask(uint8_t** pbuf, uint8_t k_dst, uint8_t src1, 
                               uint8_t src2, uint8_t imm, uint8_t mask) {
    // For AVX-512, compare writes to a mask register (k0-k7)
    emit_evex(pbuf, k_dst, src1, src2, MAP_0F, PP_NONE, 0, VL_512, mask, 0, 0);
    write_byte(pbuf, 0xC2);
    emit_modrm_reg(pbuf, k_dst, src2);
    write_byte(pbuf, imm);
}

/*
 * AVX-512 Compare to vector result
 * Compares src1 vs src2, then expands the mask to a vector of all-ones/all-zeros
 * Uses: vcmpps k1, src1, src2, imm; vpternlogd dst, dst, dst, 0xFF {k1}{z}
 */
static void emit_avx512_cmpps(uint8_t** pbuf, uint8_t dst, uint8_t src1, 
                               uint8_t src2, uint8_t imm, uint8_t mask) {
    (void)mask;  // We use k1 internally
    
    // Step 1: Compare to mask register k1
    emit_avx512_cmpps_to_mask(pbuf, 1, src1, src2, imm, 0);  // k1 = comparison result
    
    // Step 2: Zero the destination first (vpxord dst, dst, dst)
    emit_evex(pbuf, dst, dst, dst, MAP_0F, PP_66, 0, VL_512, 0, 0, 0);
    write_byte(pbuf, 0xEF);  // vpxord
    emit_modrm_reg(pbuf, dst, dst);
    
    // Step 3: Set all-ones in dst where mask k1 is set, using vpternlogd with zeroing
    // vpternlogd dst, dst, dst, 0xFF {k1} - sets all bits where k1=1
    // EVEX.512.66.0F3A.W0 25 /r ib with mask k1 and zeroing
    emit_evex(pbuf, dst, dst, dst, MAP_0F3A, PP_66, 0, VL_512, 1, 0, 1);  // k1, zeroing
    write_byte(pbuf, 0x25);  // vpternlogd
    emit_modrm_reg(pbuf, dst, dst);
    write_byte(pbuf, 0xFF);  // immediate: result = all ones
}

/*
 * AVX-512 FMA instruction emitter
 * vfmadd231ps: EVEX.512.66.0F38.W0 B8 /r
 * vfmadd231pd: EVEX.512.66.0F38.W1 B8 /r
 */
static void emit_avx512_fma_ps(uint8_t** pbuf, uint8_t dst, uint8_t src1,
                                uint8_t src2, uint8_t mask) {
    // vfmadd231ps dst, src1, src2 = src1 * src2 + dst
    emit_avx512_reg3(pbuf, 0xB8, dst, src1, src2, MAP_0F38, PP_66, 0, mask, 0);
}

static void emit_avx512_fma_pd(uint8_t** pbuf, uint8_t dst, uint8_t src1,
                                uint8_t src2, uint8_t mask) {
    // vfmadd231pd dst, src1, src2 = src1 * src2 + dst
    emit_avx512_reg3(pbuf, 0xB8, dst, src1, src2, MAP_0F38, PP_66, 1, mask, 0);
}

/*
 * AVX-512 broadcast instruction emitter
 * vbroadcastss: EVEX.512.66.0F38.W0 18 /r
 */
static void emit_avx512_broadcast_ss(uint8_t** pbuf, uint8_t dst, uint8_t src, uint8_t mask) {
    emit_avx512_reg2(pbuf, 0x18, dst, src, MAP_0F38, PP_66, 0, mask, 0);
}

/*
 * AVX-512 broadcast i32 instruction emitter
 * vpbroadcastd: EVEX.512.66.0F38.W0 58 /r
 */
static void emit_avx512_broadcast_d(uint8_t** pbuf, uint8_t dst, uint8_t src, uint8_t mask) {
    emit_avx512_reg2(pbuf, 0x58, dst, src, MAP_0F38, PP_66, 0, mask, 0);
}

/*
 * AVX-512 absolute value for i32
 * vpabsd: EVEX.512.66.0F38.W0 1E /r
 */
static void emit_avx512_pabsd(uint8_t** pbuf, uint8_t dst, uint8_t src, uint8_t mask) {
    emit_avx512_reg2(pbuf, 0x1E, dst, src, MAP_0F38, PP_66, 0, mask, 0);
}

/*
 * AVX-512 move register
 * vmovaps zmm, zmm: EVEX.512.0F.W0 28 /r
 */
static void emit_avx512_mov(uint8_t** pbuf, uint8_t dst, uint8_t src, uint8_t mask) {
    emit_avx512_reg2(pbuf, 0x28, dst, src, MAP_0F, PP_NONE, 0, mask, 0);
}

/*
 * AVX-512 XOR for zeroing
 * vxorps zmm, zmm, zmm: EVEX.512.0F.W0 57 /r
 */
static void emit_avx512_xorps(uint8_t** pbuf, uint8_t dst, uint8_t src1, uint8_t src2, uint8_t mask) {
    emit_avx512_reg3(pbuf, 0x57, dst, src1, src2, MAP_0F, PP_NONE, 0, mask, 0);
}

/*
 * AVX-512 shift left for i32
 * vpslld zmm, zmm, imm8: EVEX.512.66.0F.W0 72 /6 ib
 */
static void emit_avx512_pslld_imm(uint8_t** pbuf, uint8_t dst, uint8_t src, uint8_t imm, uint8_t mask) {
    emit_evex(pbuf, 6, dst, src, MAP_0F, PP_66, 0, VL_512, mask, 0, 0);
    write_byte(pbuf, 0x72);
    emit_modrm_reg(pbuf, 6, src);  // /6 = reg field is 6
    write_byte(pbuf, imm);
}

/*
 * AVX-512 shift right logical for i32
 * vpsrld zmm, zmm, imm8: EVEX.512.66.0F.W0 72 /2 ib
 */
static void emit_avx512_psrld_imm(uint8_t** pbuf, uint8_t dst, uint8_t src, uint8_t imm, uint8_t mask) {
    emit_evex(pbuf, 2, dst, src, MAP_0F, PP_66, 0, VL_512, mask, 0, 0);
    write_byte(pbuf, 0x72);
    emit_modrm_reg(pbuf, 2, src);  // /2 = reg field is 2
    write_byte(pbuf, imm);
}

/*
 * AVX-512 shift right arithmetic for i32
 * vpsrad zmm, zmm, imm8: EVEX.512.66.0F.W0 72 /4 ib
 */
static void emit_avx512_psrad_imm(uint8_t** pbuf, uint8_t dst, uint8_t src, uint8_t imm, uint8_t mask) {
    emit_evex(pbuf, 4, dst, src, MAP_0F, PP_66, 0, VL_512, mask, 0, 0);
    write_byte(pbuf, 0x72);
    emit_modrm_reg(pbuf, 4, src);  // /4 = reg field is 4
    write_byte(pbuf, imm);
}

/*
 * AVX-512 square root
 * vsqrtps: EVEX.512.0F.W0 51 /r
 */
static void emit_avx512_sqrtps(uint8_t** pbuf, uint8_t dst, uint8_t src, uint8_t mask) {
    emit_avx512_reg2(pbuf, 0x51, dst, src, MAP_0F, PP_NONE, 0, mask, 0);
}

/*
 * AVX-512 reciprocal (approximate)
 * vrcp14ps: EVEX.512.66.0F38.W0 4C /r
 */
static void emit_avx512_rcpps(uint8_t** pbuf, uint8_t dst, uint8_t src, uint8_t mask) {
    emit_avx512_reg2(pbuf, 0x4C, dst, src, MAP_0F38, PP_66, 0, mask, 0);
}

/*
 * AVX-512 reciprocal square root (approximate)
 * vrsqrt14ps: EVEX.512.66.0F38.W0 4E /r
 */
static void emit_avx512_rsqrtps(uint8_t** pbuf, uint8_t dst, uint8_t src, uint8_t mask) {
    emit_avx512_reg2(pbuf, 0x4E, dst, src, MAP_0F38, PP_66, 0, mask, 0);
}

/*
 * AVX-512 ternary logic (for NOT, AND, OR, XOR combinations)
 * vpternlogd: EVEX.512.66.0F3A.W0 25 /r ib
 */
static void emit_avx512_ternlogd(uint8_t** pbuf, uint8_t dst, uint8_t src1, 
                                  uint8_t src2, uint8_t imm, uint8_t mask) {
    emit_evex(pbuf, dst, src1, src2, MAP_0F3A, PP_66, 0, VL_512, mask, 0, 0);
    write_byte(pbuf, 0x25);
    emit_modrm_reg(pbuf, dst, src2);
    write_byte(pbuf, imm);
}

/*
 * AVX-512 blend using mask
 * vpblendmd: EVEX.512.66.0F38.W0 64 /r
 */
static void emit_avx512_blendmd(uint8_t** pbuf, uint8_t dst, uint8_t src1,
                                 uint8_t src2, uint8_t mask) {
    emit_avx512_reg3(pbuf, 0x64, dst, src1, src2, MAP_0F38, PP_66, 0, mask, 0);
}

/*
 * AVX-512 permute/shuffle
 * vpermd: EVEX.512.66.0F38.W0 36 /r
 */
static void emit_avx512_permd(uint8_t** pbuf, uint8_t dst, uint8_t idx,
                               uint8_t src, uint8_t mask) {
    emit_avx512_reg3(pbuf, 0x36, dst, idx, src, MAP_0F38, PP_66, 0, mask, 0);
}

/*
 * AVX-512 gather float32
 * vgatherdps: EVEX.512.66.0F38.W0 92 /r
 */
static void emit_avx512_gatherdps(uint8_t** pbuf, uint8_t dst, uint8_t base,
                                   uint8_t idx, uint8_t mask) {
    // Complex encoding - simplified for register form
    emit_evex(pbuf, dst, idx, base, MAP_0F38, PP_66, 0, VL_512, mask, 0, 0);
    write_byte(pbuf, 0x92);
    // Need VSIB encoding - simplified here
    emit_modrm_reg(pbuf, dst, base);
}

/*
 * AVX-512 scatter float32
 * vscatterdps: EVEX.512.66.0F38.W0 A2 /r
 */
static void emit_avx512_scatterdps(uint8_t** pbuf, uint8_t src, uint8_t base,
                                    uint8_t idx, uint8_t mask) {
    emit_evex(pbuf, src, idx, base, MAP_0F38, PP_66, 0, VL_512, mask, 0, 0);
    write_byte(pbuf, 0xA2);
    emit_modrm_reg(pbuf, src, base);
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
    
    // Get mask register from instruction if specified, otherwise use k0 (no masking)
    #define GET_MASK(instr) ((instr)->mask_reg >= 0 ? (uint8_t)(instr)->mask_reg : 0)

    printf("[codegen] generating x86 code for %zu instructions\n", mod->size);
    printf("[codegen] target vector width: %d bytes\n", mod->vec_width_bytes);

    emit_prologue(&ptr);

    for (size_t i = 0; i < mod->size; i++) {
        pva_instr_t* instr = &mod->code[i];
        uint8_t mask = GET_MASK(instr);

        switch (instr->op) {
            // ========== ARITHMETIC F32 ==========
            case PVA_ADD_F32:
                if (mod->vec_width_bytes == 64) {
                    // vaddps zmm, zmm, zmm - EVEX.512.0F.W0 58 /r
                    emit_avx512_ps(&ptr, 0x58, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    emit_avx2_instr(&ptr, 0x58, instr->dst, instr->src1, instr->src2);
                } else {
                    emit_sse_binop(&ptr, 0x58, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_SUB_F32:
                if (mod->vec_width_bytes == 64) {
                    // vsubps zmm, zmm, zmm - EVEX.512.0F.W0 5C /r
                    emit_avx512_ps(&ptr, 0x5C, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    emit_avx2_instr(&ptr, 0x5C, instr->dst, instr->src1, instr->src2);
                } else {
                    emit_sse_binop(&ptr, 0x5C, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_MUL_F32:
                if (mod->vec_width_bytes == 64) {
                    // vmulps zmm, zmm, zmm - EVEX.512.0F.W0 59 /r
                    emit_avx512_ps(&ptr, 0x59, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    emit_avx2_instr(&ptr, 0x59, instr->dst, instr->src1, instr->src2);
                } else {
                    emit_sse_binop(&ptr, 0x59, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_DIV_F32:
                if (mod->vec_width_bytes == 64) {
                    // vdivps zmm, zmm, zmm - EVEX.512.0F.W0 5E /r
                    emit_avx512_ps(&ptr, 0x5E, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    emit_avx2_instr(&ptr, 0x5E, instr->dst, instr->src1, instr->src2);
                } else {
                    emit_sse_binop(&ptr, 0x5E, instr->dst, instr->src1, instr->src2);
                }
                break;

            // ========== ARITHMETIC F64 ==========
            case PVA_ADD_F64:
                if (mod->vec_width_bytes == 64) {
                    // vaddpd zmm, zmm, zmm - EVEX.512.66.0F.W1 58 /r
                    emit_avx512_pd(&ptr, 0x58, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vaddpd ymm - need VEX.256.66.0F
                    uint8_t vex_vvvv = (~instr->src1) & 0xF;
                    uint8_t vex[3] = {0xC4, 0xE1, (uint8_t)(0x05 | (vex_vvvv << 3))};  // W=0, L=1, pp=01
                    write_bytes(&ptr, vex, 3);
                    write_byte(&ptr, 0x58);
                    emit_modrm_reg(&ptr, instr->dst, instr->src2);
                } else {
                    // addpd xmm (0x66 prefix)
                    uint8_t prefix = 0x66;
                    write_bytes(&ptr, &prefix, 1);
                    emit_sse_binop(&ptr, 0x58, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_SUB_F64:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_pd(&ptr, 0x5C, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    uint8_t vex_vvvv = (~instr->src1) & 0xF;
                    uint8_t vex[3] = {0xC4, 0xE1, (uint8_t)(0x05 | (vex_vvvv << 3))};
                    write_bytes(&ptr, vex, 3);
                    write_byte(&ptr, 0x5C);
                    emit_modrm_reg(&ptr, instr->dst, instr->src2);
                } else {
                    uint8_t prefix = 0x66;
                    write_bytes(&ptr, &prefix, 1);
                    emit_sse_binop(&ptr, 0x5C, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_MUL_F64:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_pd(&ptr, 0x59, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    uint8_t vex_vvvv = (~instr->src1) & 0xF;
                    uint8_t vex[3] = {0xC4, 0xE1, (uint8_t)(0x05 | (vex_vvvv << 3))};
                    write_bytes(&ptr, vex, 3);
                    write_byte(&ptr, 0x59);
                    emit_modrm_reg(&ptr, instr->dst, instr->src2);
                } else {
                    uint8_t prefix = 0x66;
                    write_bytes(&ptr, &prefix, 1);
                    emit_sse_binop(&ptr, 0x59, instr->dst, instr->src1, instr->src2);
                }
                break;

            case PVA_DIV_F64:
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_pd(&ptr, 0x5E, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    uint8_t vex_vvvv = (~instr->src1) & 0xF;
                    uint8_t vex[3] = {0xC4, 0xE1, (uint8_t)(0x05 | (vex_vvvv << 3))};
                    write_bytes(&ptr, vex, 3);
                    write_byte(&ptr, 0x5E);
                    emit_modrm_reg(&ptr, instr->dst, instr->src2);
                } else {
                    uint8_t prefix = 0x66;
                    write_bytes(&ptr, &prefix, 1);
                    emit_sse_binop(&ptr, 0x5E, instr->dst, instr->src1, instr->src2);
                }
                break;

            // ========== ARITHMETIC I32 ==========
            case PVA_ADD_I32: {
                if (mod->vec_width_bytes == 64) {
                    // vpaddd zmm, zmm, zmm - EVEX.512.66.0F.W0 FE /r
                    emit_avx512_d_0f(&ptr, 0xFE, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vpaddd ymm, ymm, ymm - VEX.256.66.0F.WIG FE /r
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
                if (mod->vec_width_bytes == 64) {
                    // vpsubd zmm, zmm, zmm - EVEX.512.66.0F.W0 FA /r
                    emit_avx512_d_0f(&ptr, 0xFA, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vpmulld zmm, zmm, zmm - EVEX.512.66.0F38.W0 40 /r
                    emit_avx512_d_0f38(&ptr, 0x40, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vpmulld ymm, ymm, ymm - VEX.256.66.0F38.WIG 40 /r
                    emit_avx2_66_0f38(&ptr, 0x40, instr->dst, instr->src1, instr->src2);
                } else {
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x38, 0x40, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))};
                    write_bytes(&ptr, instr_bytes, 5);
                }
                break;
            }

            // ========== MATH OPERATIONS ==========
            case PVA_SQRT_F32: {
                if (mod->vec_width_bytes == 64) {
                    // vsqrtps zmm, zmm - EVEX.512.0F.W0 51 /r
                    emit_avx512_sqrtps(&ptr, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vsqrtps ymm, ymm - VEX.256.0F.WIG 51 (2-operand)
                    emit_avx2_2op(&ptr, 0x51, instr->dst, instr->src1);
                } else {
                    // sqrtps xmm, xmm (0x0F 0x51)
                    uint8_t instr_bytes[] = {0x0F, 0x51, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_RSQRT_F32: {
                if (mod->vec_width_bytes == 64) {
                    // vrsqrt14ps zmm, zmm - EVEX.512.66.0F38.W0 4E /r
                    emit_avx512_rsqrtps(&ptr, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vrsqrtps ymm, ymm - VEX.256.0F.WIG 52 (2-operand)
                    emit_avx2_2op(&ptr, 0x52, instr->dst, instr->src1);
                } else {
                    // rsqrtps xmm, xmm (0x0F 0x52)
                    uint8_t instr_bytes[] = {0x0F, 0x52, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_RCP_F32: {
                if (mod->vec_width_bytes == 64) {
                    // vrcp14ps zmm, zmm - EVEX.512.66.0F38.W0 4C /r
                    emit_avx512_rcpps(&ptr, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vrcpps ymm, ymm - VEX.256.0F.WIG 53 (2-operand)
                    emit_avx2_2op(&ptr, 0x53, instr->dst, instr->src1);
                } else {
                    // rcpps xmm, xmm (0x0F 0x53)
                    uint8_t instr_bytes[] = {0x0F, 0x53, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_MIN_F32: {
                if (mod->vec_width_bytes == 64) {
                    // vminps zmm, zmm, zmm - EVEX.512.0F.W0 5D /r
                    emit_avx512_ps(&ptr, 0x5D, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vminps ymm, ymm, ymm
                    emit_avx2_instr(&ptr, 0x5D, instr->dst, instr->src1, instr->src2);
                } else {
                    // minps xmm, xmm (0x0F 0x5D)
                    emit_sse_binop(&ptr, 0x5D, instr->dst, instr->src1, instr->src2);
                }
                break;
            }

            case PVA_MAX_F32: {
                if (mod->vec_width_bytes == 64) {
                    // vmaxps zmm, zmm, zmm - EVEX.512.0F.W0 5F /r
                    emit_avx512_ps(&ptr, 0x5F, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // AVX-512: Use vpternlogd to set all ones, then shift
                    // vpternlogd zmm15, zmm15, zmm15, 0xFF (all ones)
                    emit_avx512_ternlogd(&ptr, 15, 15, 15, 0xFF, 0);
                    // vpsrld zmm15, zmm15, 1 (shift right to get 0x7FFFFFFF)
                    emit_avx512_psrld_imm(&ptr, 15, 15, 1, 0);
                    // vandps dst, src, zmm15
                    emit_avx512_ps(&ptr, 0x54, instr->dst, instr->src1, 15, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // AVX-512: Use vpternlogd to set all ones, then shift left
                    // vpternlogd zmm15, zmm15, zmm15, 0xFF (all ones)
                    emit_avx512_ternlogd(&ptr, 15, 15, 15, 0xFF, 0);
                    // vpslld zmm15, zmm15, 31 (shift left to get 0x80000000)
                    emit_avx512_pslld_imm(&ptr, 15, 15, 31, 0);
                    // vxorps dst, src, zmm15
                    emit_avx512_ps(&ptr, 0x57, instr->dst, instr->src1, 15, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vcmpps zmm/k, zmm, zmm, imm8 - EVEX.512.0F.W0 C2 /r ib
                    // AVX-512 writes result to mask register, but for compatibility write to vector
                    emit_avx512_cmpps(&ptr, instr->dst, instr->src1, instr->src2, 0x01, mask);  // LT
                } else if (mod->vec_width_bytes == 32) {
                    // vcmpps ymm, ymm, ymm, imm8 - VEX.256.0F.WIG C2 /r ib
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
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_cmpps(&ptr, instr->dst, instr->src1, instr->src2, 0x02, mask);  // LE
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_cmpps(&ptr, instr->dst, instr->src1, instr->src2, 0x00, mask);  // EQ
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    emit_avx512_cmpps(&ptr, instr->dst, instr->src1, instr->src2, 0x04, mask);  // NEQ
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // GT: use NLT (not less than) = imm8 0x0D, or swap operands and use LT
                    emit_avx512_cmpps(&ptr, instr->dst, instr->src2, instr->src1, 0x01, mask);  // swapped LT = GT
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // GE: swap operands and use LE
                    emit_avx512_cmpps(&ptr, instr->dst, instr->src2, instr->src1, 0x02, mask);  // swapped LE = GE
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vpandd zmm, zmm, zmm - EVEX.512.66.0F.W0 DB /r
                    emit_avx512_d_0f(&ptr, 0xDB, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vpord zmm, zmm, zmm - EVEX.512.66.0F.W0 EB /r
                    emit_avx512_d_0f(&ptr, 0xEB, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vpxord zmm, zmm, zmm - EVEX.512.66.0F.W0 EF /r
                    emit_avx512_d_0f(&ptr, 0xEF, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vpslld zmm, zmm, imm8 - EVEX.512.66.0F.W0 72 /6 ib
                    emit_avx512_pslld_imm(&ptr, instr->dst, instr->src1, (uint8_t)instr->imm, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vpslld ymm_dst, ymm_src, imm8 - VEX.256.66.0F.WIG 72 /6 ib
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
                if (mod->vec_width_bytes == 64) {
                    // vpsrld zmm, zmm, imm8 - EVEX.512.66.0F.W0 72 /2 ib
                    emit_avx512_psrld_imm(&ptr, instr->dst, instr->src1, (uint8_t)instr->imm, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vpsrad zmm, zmm, imm8 - EVEX.512.66.0F.W0 72 /4 ib
                    emit_avx512_psrad_imm(&ptr, instr->dst, instr->src1, (uint8_t)instr->imm, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vmovaps zmm, zmm - EVEX.512.0F.W0 28 /r
                    emit_avx512_mov(&ptr, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vbroadcastss zmm, xmm - EVEX.512.66.0F38.W0 18 /r
                    emit_avx512_broadcast_ss(&ptr, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vbroadcastss ymm, xmm - VEX.256.66.0F38.W0 18 /r
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
                if (mod->vec_width_bytes == 64) {
                    // vshufps zmm, zmm, zmm, imm8 - EVEX.512.0F.W0 C6 /r ib
                    emit_avx512_ps(&ptr, 0xC6, instr->dst, instr->src1, instr->src2, mask);
                    write_byte(&ptr, (uint8_t)instr->imm);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // AVX-512: vpblendmd zmm, zmm, zmm {k} - blend using mask register
                    // For immediate blend, use vpternlogd or vblendmps
                    emit_avx512_blendmd(&ptr, instr->dst, instr->src1, instr->src2, (uint8_t)instr->src3);
                } else if (mod->vec_width_bytes == 32) {
                    // vblendps ymm, ymm, ymm, imm8 - VEX.256.66.0F3A.WIG 0C
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
                if (mod->vec_width_bytes == 64) {
                    // EVEX.512.66.0F.W0 5B /r
                    emit_avx512_reg2(&ptr, 0x5B, instr->dst, instr->src1, MAP_0F, PP_66, 0, mask, 0);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // EVEX.512.0F.W0 5B /r
                    emit_avx512_reg2(&ptr, 0x5B, instr->dst, instr->src1, MAP_0F, PP_NONE, 0, mask, 0);
                } else if (mod->vec_width_bytes == 32) {
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
                // vcvtps2pd - widen f32 to f64
                if (mod->vec_width_bytes == 64) {
                    // EVEX.512.0F.W0 5A /r (actually takes xmm/ymm and outputs zmm)
                    emit_avx512_reg2(&ptr, 0x5A, instr->dst, instr->src1, MAP_0F, PP_NONE, 0, mask, 0);
                } else {
                    // cvtps2pd xmm, xmm (0x0F 0x5A)
                    uint8_t instr_bytes[] = {0x0F, 0x5A, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 3);
                }
                break;
            }

            case PVA_CVT_F32_F64: {
                // vcvtpd2ps - narrow f64 to f32
                if (mod->vec_width_bytes == 64) {
                    // EVEX.512.66.0F.W1 5A /r
                    emit_avx512_reg2(&ptr, 0x5A, instr->dst, instr->src1, MAP_0F, PP_66, 1, mask, 0);
                } else {
                    // cvtpd2ps xmm, xmm (0x66 0x0F 0x5A)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x5A, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            // ========== I16 ARITHMETIC ==========
            case PVA_ADD_I16: {
                if (mod->vec_width_bytes == 64) {
                    // vpaddw zmm, zmm, zmm - EVEX.512.66.0F.W0 FD /r
                    emit_avx512_d_0f(&ptr, 0xFD, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vpsubw zmm, zmm, zmm - EVEX.512.66.0F.W0 F9 /r
                    emit_avx512_d_0f(&ptr, 0xF9, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vpmullw zmm, zmm, zmm - EVEX.512.66.0F.W0 D5 /r
                    emit_avx512_d_0f(&ptr, 0xD5, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vsqrtpd zmm, zmm - EVEX.512.66.0F.W1 51 /r
                    emit_avx512_reg2(&ptr, 0x51, instr->dst, instr->src1, MAP_0F, PP_66, 1, mask, 0);
                } else {
                    // sqrtpd xmm, xmm (0x66 0x0F 0x51)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x51, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_ABS_I32: {
                if (mod->vec_width_bytes == 64) {
                    // vpabsd zmm, zmm - EVEX.512.66.0F38.W0 1E /r
                    emit_avx512_pabsd(&ptr, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // AVX-512: vpxord to zero dst, then vpsubd dst, dst, src1
                    emit_avx512_d_0f(&ptr, 0xEF, instr->dst, instr->dst, instr->dst, 0);
                    emit_avx512_d_0f(&ptr, 0xFA, instr->dst, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // AVX2: vpxor to zero dst, then vpsubd dst, dst, src1
                    emit_avx2_66_0f(&ptr, 0xEF, instr->dst, instr->dst, instr->dst);
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
                // vfmadd231ps dst, src1, src2 = src1 * src2 + dst
                if (mod->vec_width_bytes == 64) {
                    // First: vmovaps dst, src3 (copy addend to destination)
                    emit_avx512_mov(&ptr, instr->dst, instr->src3, 0);
                    // Then: vfmadd231ps dst, src1, src2
                    emit_avx512_fma_ps(&ptr, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // First: vmovaps dst, src3 (copy addend to destination)
                    // Use 3-byte VEX for proper high register support
                    uint8_t vex_r = (instr->dst < 8) ? 1 : 0;
                    uint8_t vex_b = (instr->src3 < 8) ? 1 : 0;
                    uint8_t mov[5] = {
                        0xC4, 
                        (uint8_t)((vex_r << 7) | (1 << 6) | (vex_b << 5) | 0x01),  // R.X.B.00001 (0F map)
                        0x7C,  // W=0, vvvv=1111, L=1(256-bit), pp=00
                        0x28,  // vmovaps opcode
                        (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src3 & 7))
                    };
                    write_bytes(&ptr, mov, 5);
                    
                    // Then: vfmadd231ps dst, src1, src2
                    // VEX.256.66.0F38.W0 B8 /r - use helper for proper high reg support
                    emit_avx2_66_0f38(&ptr, 0xB8, instr->dst, instr->src1, instr->src2);
                } else {
                    // SSE fallback: mul then add
                    // First copy src3 to dst
                    uint8_t mov[] = {0x0F, 0x28, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src3 & 7))};
                    write_bytes(&ptr, mov, 3);
                    emit_sse_binop(&ptr, 0x59, instr->dst, instr->src1, instr->src2);  // dst = src1 * src2
                    emit_sse_binop(&ptr, 0x58, instr->dst, instr->dst, instr->src3);    // dst += src3
                }
                break;
            }

            case PVA_FMA_F64: {
                // vfma dst, src1, src2, src3 means dst = src1 * src2 + src3
                if (mod->vec_width_bytes == 64) {
                    // First: vmovapd dst, src3 (copy addend)
                    emit_avx512_mov(&ptr, instr->dst, instr->src3, 0);
                    // vfmadd231pd dst, src1, src2
                    emit_avx512_fma_pd(&ptr, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // First: vmovapd dst, src3 (copy addend)
                    // Use 3-byte VEX for proper high register support
                    uint8_t vex_r = (instr->dst < 8) ? 1 : 0;
                    uint8_t vex_b = (instr->src3 < 8) ? 1 : 0;
                    uint8_t mov[5] = {
                        0xC4,
                        (uint8_t)((vex_r << 7) | (1 << 6) | (vex_b << 5) | 0x01),  // R.X.B.00001 (0F map)
                        0x7D,  // W=0, vvvv=1111, L=1(256-bit), pp=01(66)
                        0x28,  // vmovapd opcode
                        (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src3 & 7))
                    };
                    write_bytes(&ptr, mov, 5);
                    
                    // vfmadd231pd dst, src1, src2
                    // VEX.256.66.0F38.W1 B8 /r
                    uint8_t vex_r2 = (instr->dst < 8) ? 1 : 0;
                    uint8_t vex_b2 = (instr->src2 < 8) ? 1 : 0;
                    uint8_t vvvv = (~instr->src1) & 0xF;
                    uint8_t instr_bytes[] = {
                        0xC4, 
                        (uint8_t)((vex_r2 << 7) | (1 << 6) | (vex_b2 << 5) | 0x02),  // R.X.B.00010 (0F38 map)
                        (uint8_t)(0x85 | (vvvv << 3)),  // W=1, vvvv, L=1, pp=01
                        0xB8,
                        (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src2 & 7))
                    };
                    write_bytes(&ptr, instr_bytes, 5);
                }
                break;
            }

            case PVA_MIN_I32: {
                if (mod->vec_width_bytes == 64) {
                    // vpminsd zmm, zmm, zmm - EVEX.512.66.0F38.W0 39 /r
                    emit_avx512_d_0f38(&ptr, 0x39, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vpmaxsd zmm, zmm, zmm - EVEX.512.66.0F38.W0 3D /r
                    emit_avx512_d_0f38(&ptr, 0x3D, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // AVX-512: Use vpternlogd with 0xFF to invert
                    // vpternlogd dst, src1, src1, 0xFF = NOT(src1)
                    emit_avx512_ternlogd(&ptr, instr->dst, instr->src1, instr->src1, 0xFF, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                // Note: AVX-512 doesn't have a direct haddps equivalent
                // For now, use permute and add approach for 512-bit
                if (mod->vec_width_bytes == 64) {
                    // AVX-512: Use vshufps and vaddps to emulate horizontal add
                    // vshufps dst, src, src, 0x4E (swap adjacent pairs)
                    emit_avx512_ps(&ptr, 0xC6, instr->dst, instr->src1, instr->src1, 0);
                    write_byte(&ptr, 0x4E);  // immediate for shuffle
                    // vaddps dst, dst, src
                    emit_avx512_ps(&ptr, 0x58, instr->dst, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vhaddps ymm, ymm, ymm - VEX.256.F2.0F.WIG 7C
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
                if (mod->vec_width_bytes == 64) {
                    // AVX-512: Use permute and min operations
                    emit_avx512_ps(&ptr, 0xC6, instr->dst, instr->src1, instr->src1, 0);
                    write_byte(&ptr, 0x4E);
                    emit_avx512_ps(&ptr, 0x5D, instr->dst, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // Use AVX2 vshufps and vminps
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
                if (mod->vec_width_bytes == 64) {
                    // AVX-512: Use permute and max operations
                    emit_avx512_ps(&ptr, 0xC6, instr->dst, instr->src1, instr->src1, 0);
                    write_byte(&ptr, 0x4E);
                    emit_avx512_ps(&ptr, 0x5F, instr->dst, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vpbroadcastd zmm, xmm - EVEX.512.66.0F38.W0 58 /r
                    emit_avx512_broadcast_d(&ptr, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vbroadcastss zmm, xmm - EVEX.512.66.0F38.W0 18 /r
                    emit_avx512_broadcast_ss(&ptr, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vbroadcastss ymm, xmm (AVX) - VEX.256.66.0F38.W0 18 /r
                    uint8_t instr_bytes[] = {0xC4, 0xE2, 0x7D, 0x18, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 5);
                } else {
                    // shufps xmm, xmm, 0 - broadcast lowest element
                    uint8_t shuf[] = {0x0F, 0xC6, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7)), 0x00};
                    write_bytes(&ptr, shuf, 4);
                }
                break;
            }

            case PVA_SET1_I32: {
                if (mod->vec_width_bytes == 64) {
                    // vpbroadcastd zmm, xmm - EVEX.512.66.0F38.W0 58 /r
                    emit_avx512_broadcast_d(&ptr, instr->dst, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vpbroadcastd ymm, xmm (AVX2) - VEX.256.66.0F38.W0 58 /r
                    uint8_t instr_bytes[] = {0xC4, 0xE2, 0x7D, 0x58, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 5);
                } else {
                    // pshufd xmm, xmm, 0x00 - broadcast lowest dword
                    uint8_t shuf[] = {0x66, 0x0F, 0x70, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->dst & 7)), 0x00};
                    write_bytes(&ptr, shuf, 5);
                }
                break;
            }

            // ========== I32 COMPARISONS ==========
            case PVA_CMP_LT_I32: {
                if (mod->vec_width_bytes == 64) {
                    // vpcmpgtd zmm, zmm, zmm with swapped operands (a < b => b > a)
                    // EVEX.512.66.0F.W0 66 /r
                    emit_avx512_d_0f(&ptr, 0x66, instr->dst, instr->src2, instr->src1, mask);
                } else if (mod->vec_width_bytes == 32) {
                    // vpcmpgtd with swapped operands (a < b => b > a)
                    emit_avx2_66_0f(&ptr, 0x66, instr->dst, instr->src2, instr->src1);
                } else {
                    // pcmpgtd with swapped operands (a < b => b > a)
                    uint8_t instr_bytes[] = {0x66, 0x0F, 0x66, (uint8_t)(0xC0 | ((instr->dst & 7) << 3) | (instr->src1 & 7))};
                    write_bytes(&ptr, instr_bytes, 4);
                }
                break;
            }

            case PVA_CMP_GT_I32: {
                if (mod->vec_width_bytes == 64) {
                    // vpcmpgtd zmm, zmm, zmm - EVEX.512.66.0F.W0 66 /r
                    emit_avx512_d_0f(&ptr, 0x66, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                if (mod->vec_width_bytes == 64) {
                    // vpcmpeqd zmm, zmm, zmm - EVEX.512.66.0F.W0 76 /r
                    emit_avx512_d_0f(&ptr, 0x76, instr->dst, instr->src1, instr->src2, mask);
                } else if (mod->vec_width_bytes == 32) {
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
                    // vmovaps zmm, [base + disp] - aligned load
                    emit_avx512_load_mem(&ptr, xmm_dst, base_gpr, (int32_t)instr->imm, 1, FULL_MASK, 0);
                } else if (mod->vec_width_bytes == 32) {
                    // vmovaps ymm, [gpr] - VEX encoded
                    // VEX.256.0F.WIG 28 /r
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
                    } else if ((int32_t)instr->imm >= -128 && (int32_t)instr->imm < 128) {
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
                    } else if ((int32_t)instr->imm >= -128 && (int32_t)instr->imm < 128) {
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
                    // vmovaps [base + disp], zmm - aligned store
                    emit_avx512_store_mem(&ptr, xmm_src, base_gpr, (int32_t)instr->imm, 1, FULL_MASK);
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
                    } else if ((int32_t)instr->imm >= -128 && (int32_t)instr->imm < 128) {
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
                    } else if ((int32_t)instr->imm >= -128 && (int32_t)instr->imm < 128) {
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
                    // vpxord zmm, zmm, zmm - EVEX.512.66.0F.W0 EF /r
                    // Self-XOR to zero out the register
                    emit_avx512_xorps(&ptr, instr->dst, instr->dst, instr->dst, 0);
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
                if (mod->vec_width_bytes == 64) {
                    // AVX-512: vpternlogd zmm, zmm, zmm, 0xFF sets all bits to 1
                    emit_avx512_ternlogd(&ptr, instr->dst, instr->dst, instr->dst, 0xFF, 0);
                } else if (mod->vec_width_bytes == 32) {
                    // vpcmpeqd ymm, ymm, ymm - sets all bits to 1
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
