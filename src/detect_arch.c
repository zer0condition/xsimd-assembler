#include "pva.h"
#include <cpuid.h>
#include <string.h>
#include <stdio.h>

pva_arch_t pva_detect_arch(int* vec_width_bytes) {
    unsigned int eax, ebx, ecx, edx;

#ifdef __x86_64__
    if (__get_cpuid_max(0, NULL) < 1) {
        *vec_width_bytes = 4;
        return PVA_ARCH_UNKNOWN;
    }

    __cpuid(1, eax, ebx, ecx, edx);

    if (!(edx & (1 << 25))) {
        *vec_width_bytes = 4;
        return PVA_ARCH_UNKNOWN;
    }

    // check for AVX-512 Foundation
    if (__get_cpuid_max(0, NULL) >= 7) {
        unsigned int eax7, ebx7, ecx7, edx7;
        __cpuid_count(7, 0, eax7, ebx7, ecx7, edx7);
        
        if (ebx7 & (1 << 16)) {  // AVX-512F
            *vec_width_bytes = 64;
            printf("[detect_arch] detected AVX-512 support\n");
            return PVA_ARCH_X86_AVX512;
        }
    }

    // check for AVX2
    if (ecx & (1 << 5)) {
        *vec_width_bytes = 32;
        printf("[detect_arch] detected AVX2 support\n");
        return PVA_ARCH_X86_AVX2;
    }

    // default to SSE
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