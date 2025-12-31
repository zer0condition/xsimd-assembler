/*
 * Test runner for xsimd-asm generated code
 * 
 * This loads the generated binary and executes it with test data
 * to verify the code works correctly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <immintrin.h>

// Function signature for generated PVA code
// Calling convention: r10=input in rdi, r11=output in rsi
typedef void (*pva_func_t)(float* input, float* output);

static void* load_executable(const char* filename, size_t* out_size) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open binary");
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // Allocate executable memory
    void* mem = mmap(NULL, size, 
                     PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap failed");
        fclose(fp);
        return NULL;
    }

    fread(mem, 1, size, fp);
    fclose(fp);

    *out_size = size;
    return mem;
}

static void hexdump(const uint8_t* data, size_t len) {
    printf("Code bytes (%zu):\n", len);
    for (size_t i = 0; i < len; i++) {
        printf("%02x ", data[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    if (len % 16 != 0) printf("\n");
}

static void print_vector_f32(const char* name, float* v, int count) {
    printf("%s: [", name);
    for (int i = 0; i < count; i++) {
        printf("%.4f%s", v[i], i < count-1 ? ", " : "");
    }
    printf("]\n");
}

static void print_vector_hex(const char* name, uint32_t* v, int count) {
    printf("%s: [", name);
    for (int i = 0; i < count; i++) {
        printf("0x%08x%s", v[i], i < count-1 ? ", " : "");
    }
    printf("]\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <binary.bin>\n", argv[0]);
        return 1;
    }

    size_t code_size;
    void* code = load_executable(argv[1], &code_size);
    if (!code) {
        return 1;
    }

    printf("=== xsimd-asm Test Runner ===\n");
    printf("Loaded: %s (%zu bytes)\n\n", argv[1], code_size);

    hexdump((uint8_t*)code, code_size);
    printf("\n");

    // Allocate aligned test data (32-byte aligned for AVX2)
    // Input: 512 bytes = 128 floats at various offsets for comprehensive test
    // Output: 2048 bytes = 512 floats for all test results
    float* input __attribute__((aligned(32))) = aligned_alloc(32, 1024);
    float* output __attribute__((aligned(32))) = aligned_alloc(32, 4096);
    
    memset(input, 0, 1024);
    memset(output, 0, 4096);

    // Set up input data at various offsets
    // Offset 0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    for (int i = 0; i < 8; i++) {
        input[i] = (float)(i + 1);
    }
    
    // Offset 32 (8 floats): [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    for (int i = 0; i < 8; i++) {
        input[8 + i] = (float)(i + 1) * 0.5f;
    }
    
    // Offset 64 (16 floats): [4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0] - perfect squares
    for (int i = 0; i < 8; i++) {
        input[16 + i] = (float)((i + 2) * (i + 2));
    }
    
    // Offset 96 (24 floats): [-1.0, -2.0, 3.0, -4.0, -5.0, 6.0, -7.0, 8.0] - mixed signs
    for (int i = 0; i < 8; i++) {
        input[24 + i] = (i % 3 == 2) ? (float)(i + 1) : -(float)(i + 1);
    }
    
    // Offset 128 (32 floats): integers as float bits [10, 20, 30, 40, ...]
    uint32_t* input_i32 = (uint32_t*)input;
    for (int i = 0; i < 8; i++) {
        input_i32[32 + i] = (uint32_t)(10 * (i + 1));
    }
    
    // Offset 160 (40 floats): integers [5, 10, 15, 20, ...]
    for (int i = 0; i < 8; i++) {
        input_i32[40 + i] = (uint32_t)(5 * (i + 1));
    }
    
    // Offset 192 (48 floats): f64 values [1.0, 2.0, 3.0, 4.0] as doubles
    double* input_f64 = (double*)&input[48];
    for (int i = 0; i < 4; i++) {
        input_f64[i] = (double)(i + 1);
    }
    
    // Offset 224 (56 floats): f64 values [0.5, 1.0, 1.5, 2.0]
    double* input_f64_2 = (double*)&input[56];
    for (int i = 0; i < 4; i++) {
        input_f64_2[i] = (double)(i + 1) * 0.5;
    }
    
    // Offset 256, 288: i16 packed data
    int16_t* input_i16 = (int16_t*)&input[64];
    for (int i = 0; i < 16; i++) {
        input_i16[i] = (int16_t)(i + 1);
    }
    int16_t* input_i16_2 = (int16_t*)&input[72];
    for (int i = 0; i < 16; i++) {
        input_i16_2[i] = (int16_t)(i + 10);
    }

    printf("=== Input Data ===\n");
    print_vector_f32("input[0]   (f32 vec1)", input, 8);
    print_vector_f32("input[32]  (f32 vec2)", input + 8, 8);
    print_vector_f32("input[64]  (squares) ", input + 16, 8);
    print_vector_f32("input[96]  (mixed)   ", input + 24, 8);
    print_vector_hex("input[128] (i32 a)   ", input_i32 + 32, 8);
    print_vector_hex("input[160] (i32 b)   ", input_i32 + 40, 8);
    printf("\n");

    printf("Executing generated code...\n");
    
    // Cast to function pointer and call
    pva_func_t func = (pva_func_t)code;
    
    // Call: r10=input (rdi), r11=output (rsi)
    func(input, output);

    printf("Execution completed!\n\n");

    // Print results by category
    printf("=== F32 Arithmetic ===\n");
    print_vector_f32("ADD  [0]  ", output, 8);
    print_vector_f32("SUB  [32] ", output + 8, 8);
    print_vector_f32("MUL  [64] ", output + 16, 8);
    print_vector_f32("DIV  [96] ", output + 24, 8);
    
    printf("\n=== F32 Math ===\n");
    print_vector_f32("SQRT [128]", output + 32, 8);
    print_vector_f32("MIN  [160]", output + 40, 8);
    print_vector_f32("MAX  [192]", output + 48, 8);
    print_vector_f32("ABS  [224]", output + 56, 8);
    print_vector_f32("NEG  [256]", output + 64, 8);
    
    printf("\n=== F32 Comparisons ===\n");
    print_vector_hex("LT   [288]", (uint32_t*)(output + 72), 8);
    print_vector_hex("GT   [320]", (uint32_t*)(output + 80), 8);
    print_vector_hex("EQ   [352]", (uint32_t*)(output + 88), 8);
    
    printf("\n=== Integer Ops ===\n");
    print_vector_hex("ADD  [384]", (uint32_t*)(output + 96), 8);
    print_vector_hex("SUB  [416]", (uint32_t*)(output + 104), 8);
    print_vector_hex("MUL  [448]", (uint32_t*)(output + 112), 8);
    
    printf("\n=== Bitwise ===\n");
    print_vector_hex("AND  [480]", (uint32_t*)(output + 120), 8);
    print_vector_hex("OR   [512]", (uint32_t*)(output + 128), 8);
    print_vector_hex("XOR  [544]", (uint32_t*)(output + 136), 8);
    print_vector_hex("NOT  [576]", (uint32_t*)(output + 144), 8);
    
    printf("\n=== Shifts ===\n");
    print_vector_hex("SHL  [608]", (uint32_t*)(output + 152), 8);
    print_vector_hex("SHR  [640]", (uint32_t*)(output + 160), 8);
    print_vector_hex("SAR  [672]", (uint32_t*)(output + 168), 8);
    
    printf("\n=== Integer Min/Max/Abs/Neg ===\n");
    print_vector_hex("MIN  [704]", (uint32_t*)(output + 176), 8);
    print_vector_hex("MAX  [736]", (uint32_t*)(output + 184), 8);
    print_vector_hex("ABS  [768]", (uint32_t*)(output + 192), 8);
    print_vector_hex("NEG  [800]", (uint32_t*)(output + 200), 8);
    
    printf("\n=== Integer Comparisons ===\n");
    print_vector_hex("LT   [832]", (uint32_t*)(output + 208), 8);
    print_vector_hex("GT   [864]", (uint32_t*)(output + 216), 8);
    print_vector_hex("EQ   [896]", (uint32_t*)(output + 224), 8);
    
    printf("\n=== FMA ===\n");
    print_vector_f32("FMA  [928]", output + 232, 8);
    
    printf("\n=== Special ===\n");
    print_vector_f32("ZERO [960]", output + 240, 8);
    print_vector_f32("MOV  [992]", output + 248, 8);
    print_vector_f32("BCST[1024]", output + 256, 8);
    
    printf("\n=== Conversions ===\n");
    print_vector_hex("F2I [1056]", (uint32_t*)(output + 264), 8);
    print_vector_f32("I2F [1088]", output + 272, 8);
    
    printf("\n=== F64 Ops (as f32) ===\n");
    print_vector_f32("ADD [1120]", output + 280, 8);
    print_vector_f32("SUB [1152]", output + 288, 8);
    print_vector_f32("MUL [1184]", output + 296, 8);
    print_vector_f32("DIV [1216]", output + 304, 8);
    print_vector_f32("SQRT[1248]", output + 312, 8);
    
    printf("\n=== I16 Ops ===\n");
    print_vector_hex("ADD [1280]", (uint32_t*)(output + 320), 8);
    print_vector_hex("SUB [1312]", (uint32_t*)(output + 328), 8);
    print_vector_hex("MUL [1344]", (uint32_t*)(output + 336), 8);
    
    printf("\n=== Horizontal ===\n");
    print_vector_f32("HADD[1376]", output + 344, 8);
    print_vector_f32("HMIN[1408]", output + 352, 8);
    print_vector_f32("HMAX[1440]", output + 360, 8);
    
    printf("\n=== RSQRT/RCP ===\n");
    print_vector_f32("RSQT[1472]", output + 368, 8);
    print_vector_f32("RCP [1504]", output + 376, 8);
    
    printf("\n=== Shuffle/Blend ===\n");
    print_vector_f32("SHUF[1536]", output + 384, 8);
    print_vector_f32("BLND[1568]", output + 392, 8);
    
    printf("\n=== SET1 ===\n");
    print_vector_f32("F32 [1600]", output + 400, 8);
    print_vector_hex("I32 [1632]", (uint32_t*)(output + 408), 8);
    
    printf("\n=== SETONE ===\n");
    print_vector_hex("ONE [1664]", (uint32_t*)(output + 416), 8);

    // Cleanup
    munmap(code, code_size);
    free(input);
    free(output);

    return 0;
}
