#include "pva.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 4 || strcmp(argv[2], "-o") != 0) {
        fprintf(stderr, "usage: %s input.pva -o output.bin\n", argv[0]);
        fprintf(stderr, "example: %s mandelbrot.pva -o mandelbrot.bin\n", argv[0]);
        return 1;
    }

    printf("[parser] parsing: %s\n", argv[1]);

    // parse source file
    pva_module_t* mod = pva_parse_file(argv[1]);
    if (!mod) {
        fprintf(stderr, "err: failed to parse %s\n", argv[1]);
        return 1;
    }

    printf("parser]     parsed %zu instructions\n\n", mod->size);

    // check support
    int vec_width = 0;
    mod->arch = pva_detect_arch(&vec_width);
    mod->vec_width_bytes = vec_width;

    printf("CPU architecture:\n");
    switch (mod->arch) {
        case PVA_ARCH_X86_AVX512:
            printf("  target: x86-64 AVX-512\n");
            printf("  vector width: %d bytes (512 bits)\n", vec_width);
            printf("  elems: %d floats per vector\n", vec_width / 4);
            break;
        case PVA_ARCH_X86_AVX2:
            printf("  target: x86-64 AVX2\n");
            printf("  vector width: %d bytes (256 bits)\n", vec_width);
            printf("  elems: %d floats per vector\n", vec_width / 4);
            break;
        case PVA_ARCH_X86_SSE:
            printf("  target: x86-64 SSE4.2\n");
            printf("  vector width: %d bytes (128 bits)\n", vec_width);
            printf("  elems: %d floats per vector\n", vec_width / 4);
            break;
        case PVA_ARCH_ARM_SVE:
            printf("  target: ARM SVE\n");
            printf("  vector width: %d bytes (scalable)\n", vec_width);
            break;
        case PVA_ARCH_ARM_NEON:
            printf("  target: ARM NEON\n");
            printf("  vector width: %d bytes (128 bits)\n", vec_width);
            printf("  elems: %d floats per vector\n", vec_width / 4);
            break;
        case PVA_ARCH_RISCV_RVV:
            printf("  target: RISC-V RVV\n");
            printf("  vector width: %d bytes (scalable)\n", vec_width);
            break;
        default:
            printf("  target: unknown/scalar fallback\n");
            printf("  vector width: %d bytes\n", vec_width);
    }

    // apply IR optimizations
    pva_optimize(mod);

    // gen binary output
    printf("\n");
    uint8_t buffer[8192] = {0};

    if (mod->arch == PVA_ARCH_X86_AVX512 || 
        mod->arch == PVA_ARCH_X86_AVX2 || 
        mod->arch == PVA_ARCH_X86_SSE) {
        pva_emit_x86(mod, buffer);
    } else if (mod->arch == PVA_ARCH_ARM_SVE || 
               mod->arch == PVA_ARCH_ARM_NEON) {
        pva_emit_arm(mod, buffer);
    } else if (mod->arch == PVA_ARCH_RISCV_RVV) {
        pva_emit_riscv(mod, buffer);
    } else {
        fprintf(stderr, "err: unsupported or unknown architecture\n");
        pva_free(mod);
        return 1;
    }

    // output
    FILE* outfp = fopen(argv[3], "wb");
    if (!outfp) {
        perror("err: failed to open output file");
        pva_free(mod);
        return 1;
    }

    size_t written = fwrite(buffer, 1, sizeof(buffer), outfp);
    fclose(outfp);

    printf("\ncompiled successfully!\n");
    printf("    output: %s (%zu bytes)\n", argv[3], written);
    printf("    instructions: %zu\n", mod->size);

    pva_free(mod);
    return 0;
}