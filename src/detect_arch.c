#include "pva.h"
#include <string.h>
#include <stdio.h>

#ifdef __x86_64__
#include <cpuid.h>

// XCR0 feature bits for OS-enabled AVX state saving
#define XCR0_SSE_BIT    (1 << 1)  // SSE state (XMM registers)
#define XCR0_AVX_BIT    (1 << 2)  // AVX state (YMM registers)
#define XCR0_OPMASK_BIT (1 << 5)  // AVX-512 opmask registers (k0-k7)
#define XCR0_ZMM_HI_BIT (1 << 6)  // AVX-512 upper 256 bits of ZMM0-15
#define XCR0_HI16_ZMM   (1 << 7)  // AVX-512 ZMM16-31

// read extended control register (xgetbv)
static inline unsigned long long xgetbv(unsigned int xcr) {
    unsigned int eax, edx;
    __asm__ __volatile__(
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(xcr)
    );
    return ((unsigned long long)edx << 32) | eax;
}
#endif

pva_arch_t pva_detect_arch(int* vec_width_bytes) {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, NULL) < 1) {
        *vec_width_bytes = 4;
        return PVA_ARCH_UNKNOWN;
    }

    __cpuid(1, eax, ebx, ecx, edx);

    // check for SSE support (minimum requirement)
    if (!(edx & (1 << 25))) {
        *vec_width_bytes = 4;
        return PVA_ARCH_UNKNOWN;
    }

    // check if OS has enabled XSAVE (OSXSAVE bit in ECX from CPUID.1)
    // this is required before we can call xgetbv
    int os_uses_xsave = (ecx & (1 << 27)) != 0;
    
    unsigned long long xcr0 = 0;
    if (os_uses_xsave) {
        xcr0 = xgetbv(0);  // read XCR0
    }

    // check for AVX-512 Foundation
    if (__get_cpuid_max(0, NULL) >= 7) {
        unsigned int eax7, ebx7, ecx7, edx7;
        __cpuid_count(7, 0, eax7, ebx7, ecx7, edx7);
        
        // CPU supports AVX-512F?
        if (ebx7 & (1 << 16)) {
            // verify OS has enabled AVX-512 state saving
            // SSE + AVX + opmask + ZMM_Hi256 + Hi16_ZMM
            unsigned long long avx512_xcr0_bits = XCR0_SSE_BIT | XCR0_AVX_BIT | 
                                                   XCR0_OPMASK_BIT | XCR0_ZMM_HI_BIT | XCR0_HI16_ZMM;
            if (os_uses_xsave && (xcr0 & avx512_xcr0_bits) == avx512_xcr0_bits) {
                *vec_width_bytes = 64;
                printf("[detect_arch] detected AVX-512 support (CPU + OS enabled)\n");
                return PVA_ARCH_X86_AVX512;
            } else {
                printf("[detect_arch] AVX-512 CPU support found but OS has not enabled ZMM state saving\n");
            }
        }
    }

    // check for AVX2 (requires checking CPUID leaf 7)
    if (__get_cpuid_max(0, NULL) >= 7) {
        unsigned int eax7b, ebx7b, ecx7b, edx7b;
        __cpuid_count(7, 0, eax7b, ebx7b, ecx7b, edx7b);
        
        // CPU supports AVX2?
        if (ebx7b & (1 << 5)) {
            // verify OS has enabled AVX state saving (YMM registers)
            unsigned long long avx_xcr0_bits = XCR0_SSE_BIT | XCR0_AVX_BIT;
            if (os_uses_xsave && (xcr0 & avx_xcr0_bits) == avx_xcr0_bits) {
                *vec_width_bytes = 32;
                printf("[detect_arch] detected AVX2 support (CPU + OS enabled)\n");
                return PVA_ARCH_X86_AVX2;
            } else {
                printf("[detect_arch] AVX2 CPU support found but OS has not enabled YMM state saving\n");
            }
        }
    }

    // default to SSE (doesn't require xgetbv check - always supported on x86_64)
    *vec_width_bytes = 16;
    printf("[detect_arch] detected SSE4.2 support\n");
    return PVA_ARCH_X86_SSE;

#elif defined(__aarch64__)
    // ARM64 
    #ifdef __ARM_FEATURE_SVE
        *vec_width_bytes = 16;
        printf("[detect_arch] detected ARM SVE support\n");
        return PVA_ARCH_ARM_SVE;
    #else
        *vec_width_bytes = 16;
        printf("[detect_arch] detected ARM NEON support\n");
        return PVA_ARCH_ARM_NEON;
    #endif

#elif defined(__riscv)
    // RISC-V 
    *vec_width_bytes = 32;
    printf("[detect_arch] detected RISC-V RVV support\n");
    return PVA_ARCH_RISCV_RVV;

#else
    // unknown 
    *vec_width_bytes = 4;
    printf("[detect_arch] unknown architecture, using scalar fallback\n");
    return PVA_ARCH_UNKNOWN;
#endif
}