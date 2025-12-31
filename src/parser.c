#include "pva.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

typedef struct {
    const char *input;
    int pos;
    int line;
    int col;
} pva_lexer_t;

static void lexer_skip_whitespace(pva_lexer_t *lex) {
    while (lex->input[lex->pos] && isspace(lex->input[lex->pos])) {
        if (lex->input[lex->pos] == '\n') {
            lex->line++;
            lex->col = 0;
        } else {
            lex->col++;
        }
        lex->pos++;
    }
}

static int lexer_peek(pva_lexer_t *lex) {
    lexer_skip_whitespace(lex);
    return lex->input[lex->pos];
}

static char* lexer_read_token(pva_lexer_t *lex, char *buffer, int max_len) {
    lexer_skip_whitespace(lex);
    
    int i = 0;
    while (i < max_len - 1 && lex->input[lex->pos] && 
           !isspace(lex->input[lex->pos]) && 
           lex->input[lex->pos] != ',' &&
           lex->input[lex->pos] != '[' &&
           lex->input[lex->pos] != ']' &&
           lex->input[lex->pos] != '+' &&
           lex->input[lex->pos] != '#' &&
           lex->input[lex->pos] != ';') {
        buffer[i++] = lex->input[lex->pos++];
        lex->col++;
    }
    buffer[i] = 0;
    return buffer;
}

static int lexer_read_register(pva_lexer_t *lex) {
    char token[16];
    lexer_read_token(lex, token, sizeof(token));
    
    if (token[0] != 'r') return -1;
    if (!isdigit(token[1])) return -1;
    
    int reg = atoi(&token[1]);
    if (reg < 0 || reg > 31) return -1;  // extended to 32 registers
    
    return reg;
}

// read an immediate value (e.g., #42 or #3.14)
static int lexer_read_immediate(pva_lexer_t *lex, uint32_t *imm) {
    lexer_skip_whitespace(lex);
    if (lex->input[lex->pos] != '#') return -1;
    lex->pos++;
    
    char buffer[32];
    int i = 0;
    while (i < 31 && lex->input[lex->pos] && 
           (isdigit(lex->input[lex->pos]) || lex->input[lex->pos] == '.' || 
            lex->input[lex->pos] == '-' || lex->input[lex->pos] == 'x')) {
        buffer[i++] = lex->input[lex->pos++];
    }
    buffer[i] = 0;
    
    if (strchr(buffer, '.')) {
        // float immediate - store as bit pattern
        float f = (float)atof(buffer);
        memcpy(imm, &f, sizeof(float));
    } else if (buffer[0] == '0' && buffer[1] == 'x') {
        // hex immediate
        *imm = (uint32_t)strtoul(buffer, NULL, 16);
    } else {
        *imm = (uint32_t)atoi(buffer);
    }
    return 0;
}

// read a label name
static char* lexer_read_label(pva_lexer_t *lex) {
    lexer_skip_whitespace(lex);
    
    char buffer[64];
    int i = 0;
    while (i < 63 && lex->input[lex->pos] && 
           (isalnum(lex->input[lex->pos]) || lex->input[lex->pos] == '_')) {
        buffer[i++] = lex->input[lex->pos++];
    }
    buffer[i] = 0;
    
    if (i == 0) return NULL;
    
    char* label = malloc(i + 1);
    strcpy(label, buffer);
    return label;
}

static pva_opcode_t map_opcode(const char *opname) {
    // arithmetic - float32
    if (strcmp(opname, "vadd") == 0 || strcmp(opname, "vadd.f32") == 0) return PVA_ADD_F32;
    if (strcmp(opname, "vsub") == 0 || strcmp(opname, "vsub.f32") == 0) return PVA_SUB_F32;
    if (strcmp(opname, "vmul") == 0 || strcmp(opname, "vmul.f32") == 0) return PVA_MUL_F32;
    if (strcmp(opname, "vdiv") == 0 || strcmp(opname, "vdiv.f32") == 0) return PVA_DIV_F32;
    
    // arithmetic - float64
    if (strcmp(opname, "vadd.f64") == 0) return PVA_ADD_F64;
    if (strcmp(opname, "vsub.f64") == 0) return PVA_SUB_F64;
    if (strcmp(opname, "vmul.f64") == 0) return PVA_MUL_F64;
    if (strcmp(opname, "vdiv.f64") == 0) return PVA_DIV_F64;
    
    // arithmetic - int32
    if (strcmp(opname, "vadd.i32") == 0) return PVA_ADD_I32;
    if (strcmp(opname, "vsub.i32") == 0) return PVA_SUB_I32;
    if (strcmp(opname, "vmul.i32") == 0) return PVA_MUL_I32;
    
    // arithmetic - int16
    if (strcmp(opname, "vadd.i16") == 0) return PVA_ADD_I16;
    if (strcmp(opname, "vsub.i16") == 0) return PVA_SUB_I16;
    if (strcmp(opname, "vmul.i16") == 0) return PVA_MUL_I16;
    
    // math operations
    if (strcmp(opname, "vsqrt") == 0 || strcmp(opname, "vsqrt.f32") == 0) return PVA_SQRT_F32;
    if (strcmp(opname, "vsqrt.f64") == 0) return PVA_SQRT_F64;
    if (strcmp(opname, "vrsqrt") == 0) return PVA_RSQRT_F32;
    if (strcmp(opname, "vrcp") == 0) return PVA_RCP_F32;
    if (strcmp(opname, "vabs") == 0 || strcmp(opname, "vabs.f32") == 0) return PVA_ABS_F32;
    if (strcmp(opname, "vabs.i32") == 0) return PVA_ABS_I32;
    if (strcmp(opname, "vneg") == 0 || strcmp(opname, "vneg.f32") == 0) return PVA_NEG_F32;
    if (strcmp(opname, "vneg.i32") == 0) return PVA_NEG_I32;
    if (strcmp(opname, "vfma") == 0 || strcmp(opname, "vfma.f32") == 0) return PVA_FMA_F32;
    if (strcmp(opname, "vfma.f64") == 0) return PVA_FMA_F64;
    if (strcmp(opname, "vmin") == 0 || strcmp(opname, "vmin.f32") == 0) return PVA_MIN_F32;
    if (strcmp(opname, "vmin.i32") == 0) return PVA_MIN_I32;
    if (strcmp(opname, "vmax") == 0 || strcmp(opname, "vmax.f32") == 0) return PVA_MAX_F32;
    if (strcmp(opname, "vmax.i32") == 0) return PVA_MAX_I32;
    if (strcmp(opname, "vclamp") == 0) return PVA_CLAMP_F32;
    
    // memory operations
    if (strcmp(opname, "vload") == 0 || strcmp(opname, "vload.f32") == 0) return PVA_LOAD_F32;
    if (strcmp(opname, "vstore") == 0 || strcmp(opname, "vstore.f32") == 0) return PVA_STORE_F32;
    if (strcmp(opname, "vload.f64") == 0) return PVA_LOAD_F64;
    if (strcmp(opname, "vstore.f64") == 0) return PVA_STORE_F64;
    if (strcmp(opname, "vload.i32") == 0) return PVA_LOAD_I32;
    if (strcmp(opname, "vstore.i32") == 0) return PVA_STORE_I32;
    if (strcmp(opname, "vgather") == 0) return PVA_GATHER_F32;
    if (strcmp(opname, "vscatter") == 0) return PVA_SCATTER_F32;
    
    // comparisons - float32
    if (strcmp(opname, "vlt") == 0 || strcmp(opname, "vcmplt") == 0) return PVA_CMP_LT_F32;
    if (strcmp(opname, "vle") == 0 || strcmp(opname, "vcmple") == 0) return PVA_CMP_LE_F32;
    if (strcmp(opname, "vgt") == 0 || strcmp(opname, "vcmpgt") == 0) return PVA_CMP_GT_F32;
    if (strcmp(opname, "vge") == 0 || strcmp(opname, "vcmpge") == 0) return PVA_CMP_GE_F32;
    if (strcmp(opname, "veq") == 0 || strcmp(opname, "vcmpeq") == 0) return PVA_CMP_EQ_F32;
    if (strcmp(opname, "vne") == 0 || strcmp(opname, "vcmpne") == 0) return PVA_CMP_NE_F32;
    
    // comparisons - int32
    if (strcmp(opname, "vcmplt.i32") == 0) return PVA_CMP_LT_I32;
    if (strcmp(opname, "vcmpgt.i32") == 0) return PVA_CMP_GT_I32;
    if (strcmp(opname, "vcmpeq.i32") == 0) return PVA_CMP_EQ_I32;
    
    // mask/logic operations
    if (strcmp(opname, "vand") == 0) return PVA_AND_MASK;
    if (strcmp(opname, "vor") == 0) return PVA_OR_MASK;
    if (strcmp(opname, "vxor") == 0) return PVA_XOR_MASK;
    if (strcmp(opname, "vnot") == 0) return PVA_NOT_MASK;
    
    // bitwise operations
    if (strcmp(opname, "vand.i32") == 0) return PVA_AND_I32;
    if (strcmp(opname, "vor.i32") == 0) return PVA_OR_I32;
    if (strcmp(opname, "vxor.i32") == 0) return PVA_XOR_I32;
    if (strcmp(opname, "vnot.i32") == 0) return PVA_NOT_I32;
    if (strcmp(opname, "vshl") == 0) return PVA_SHL_I32;
    if (strcmp(opname, "vshr") == 0) return PVA_SHR_I32;
    if (strcmp(opname, "vsar") == 0) return PVA_SAR_I32;
    
    // horizontal reductions
    if (strcmp(opname, "vhadd") == 0) return PVA_HADD_F32;
    if (strcmp(opname, "vhmin") == 0) return PVA_HMIN_F32;
    if (strcmp(opname, "vhmax") == 0) return PVA_HMAX_F32;
    
    // data movement
    if (strcmp(opname, "vbroadcast") == 0) return PVA_BROADCAST_F32;
    if (strcmp(opname, "vbroadcast.i32") == 0) return PVA_BROADCAST_I32;
    if (strcmp(opname, "vshuffle") == 0) return PVA_SHUFFLE;
    if (strcmp(opname, "vblend") == 0) return PVA_BLEND;
    if (strcmp(opname, "vmov") == 0) return PVA_MOV;
    
    // type conversion
    if (strcmp(opname, "vcvt.f32.i32") == 0) return PVA_CVT_F32_I32;
    if (strcmp(opname, "vcvt.i32.f32") == 0) return PVA_CVT_I32_F32;
    if (strcmp(opname, "vcvt.f64.f32") == 0) return PVA_CVT_F64_F32;
    if (strcmp(opname, "vcvt.f32.f64") == 0) return PVA_CVT_F32_F64;
    
    // initialization
    if (strcmp(opname, "vzero") == 0) return PVA_SETZERO;
    if (strcmp(opname, "vone") == 0) return PVA_SETONE;
    if (strcmp(opname, "vset1") == 0 || strcmp(opname, "vset1.f32") == 0) return PVA_SET1_F32;
    if (strcmp(opname, "vset1.i32") == 0) return PVA_SET1_I32;
    
    // control flow
    if (strcmp(opname, "label") == 0) return PVA_LABEL;
    if (strcmp(opname, "jmp") == 0) return PVA_JMP;
    if (strcmp(opname, "jmp_if") == 0) return PVA_JMP_IF;
    if (strcmp(opname, "loop_begin") == 0) return PVA_LOOP_BEGIN;
    if (strcmp(opname, "loop_end") == 0) return PVA_LOOP_END;
    if (strcmp(opname, "call") == 0) return PVA_CALL;
    if (strcmp(opname, "ret") == 0 || strcmp(opname, "vret") == 0) return PVA_RET;
    
    // explicit nop
    if (strcmp(opname, "nop") == 0 || strcmp(opname, "vnop") == 0) return PVA_NOP;
    
    return PVA_UNKNOWN;  // truly unknown opcode
}

static pva_instr_t parse_instruction_line(pva_lexer_t *lex, int line_num) {
    pva_instr_t instr = {0};
    instr.op = PVA_UNKNOWN;
    instr.mask_reg = -1;

    char opname[32];
    lexer_read_token(lex, opname, sizeof(opname));
    
    if (strlen(opname) == 0) {
        instr.op = PVA_NOP;  // Empty line is OK
        return instr;
    }
    
    pva_opcode_t op = map_opcode(opname);
    if (op == PVA_UNKNOWN) {
        fprintf(stderr, "[parser] line %d: unknown opcode '%s'\n", line_num, opname);
        return instr;  // instr.op is already PVA_UNKNOWN
    }

    instr.op = op;

    switch (op) {
        // three-operand instructions: dst, src1, src2
        case PVA_ADD_F32: case PVA_SUB_F32: case PVA_MUL_F32: case PVA_DIV_F32:
        case PVA_ADD_F64: case PVA_SUB_F64: case PVA_MUL_F64: case PVA_DIV_F64:
        case PVA_ADD_I32: case PVA_SUB_I32: case PVA_MUL_I32:
        case PVA_ADD_I16: case PVA_SUB_I16: case PVA_MUL_I16:
        case PVA_MIN_F32: case PVA_MAX_F32: case PVA_MIN_I32: case PVA_MAX_I32:
        case PVA_CMP_LT_F32: case PVA_CMP_LE_F32: case PVA_CMP_GT_F32: case PVA_CMP_GE_F32:
        case PVA_CMP_EQ_F32: case PVA_CMP_NE_F32:
        case PVA_CMP_LT_I32: case PVA_CMP_GT_I32: case PVA_CMP_EQ_I32:
        case PVA_AND_MASK: case PVA_OR_MASK: case PVA_XOR_MASK:
        case PVA_AND_I32: case PVA_OR_I32: case PVA_XOR_I32: {
            int dst = lexer_read_register(lex);
            if (dst < 0) {
                fprintf(stderr, "[parser] line %d: expected register for destination\n", line_num);
                return instr;
            }
            instr.dst = dst;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            
            int src1 = lexer_read_register(lex);
            if (src1 < 0) {
                fprintf(stderr, "[parser] line %d: expected register for source1\n", line_num);
                return instr;
            }
            instr.src1 = src1;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            
            int src2 = lexer_read_register(lex);
            if (src2 < 0) {
                fprintf(stderr, "[parser] line %d: expected register for source2\n", line_num);
                return instr;
            }
            instr.src2 = src2;
            break;
        }

        // four-operand instructions: dst, src1, src2, src3 (fma, clamp, blend)
        case PVA_FMA_F32: case PVA_FMA_F64:
        case PVA_CLAMP_F32:
        case PVA_BLEND: {
            int dst = lexer_read_register(lex);
            if (dst < 0) {
                fprintf(stderr, "[parser] line %d: expected register for destination\n", line_num);
                return instr;
            }
            instr.dst = dst;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            int src1 = lexer_read_register(lex);
            if (src1 < 0) { fprintf(stderr, "[parser] line %d: expected register\n", line_num); return instr; }
            instr.src1 = src1;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            int src2 = lexer_read_register(lex);
            if (src2 < 0) { fprintf(stderr, "[parser] line %d: expected register\n", line_num); return instr; }
            instr.src2 = src2;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            int src3 = lexer_read_register(lex);
            if (src3 < 0) { fprintf(stderr, "[parser] line %d: expected register\n", line_num); return instr; }
            instr.src3 = src3;
            break;
        }

        // two-operand instructions: dst, src1
        case PVA_SQRT_F32: case PVA_SQRT_F64:
        case PVA_RSQRT_F32: case PVA_RCP_F32:
        case PVA_ABS_F32: case PVA_ABS_I32:
        case PVA_NEG_F32: case PVA_NEG_I32:
        case PVA_NOT_MASK: case PVA_NOT_I32:
        case PVA_CVT_F32_I32: case PVA_CVT_I32_F32:
        case PVA_CVT_F64_F32: case PVA_CVT_F32_F64:
        case PVA_MOV:
        case PVA_HADD_F32: case PVA_HMIN_F32: case PVA_HMAX_F32:
        case PVA_BROADCAST_F32: case PVA_BROADCAST_I32: {
            int dst = lexer_read_register(lex);
            if (dst < 0) {
                fprintf(stderr, "[parser] line %d: expected register for destination\n", line_num);
                return instr;
            }
            instr.dst = dst;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            
            int src1 = lexer_read_register(lex);
            if (src1 < 0) {
                fprintf(stderr, "[parser] line %d: expected register for source\n", line_num);
                return instr;
            }
            instr.src1 = src1;
            break;
        }

        // shift instructions: dst, src1, #imm
        case PVA_SHL_I32: case PVA_SHR_I32: case PVA_SAR_I32: {
            int dst = lexer_read_register(lex);
            if (dst < 0) {
                fprintf(stderr, "[parser] line %d: expected register for destination\n", line_num);
                return instr;
            }
            instr.dst = dst;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            
            int src1 = lexer_read_register(lex);
            if (src1 < 0) {
                fprintf(stderr, "[parser] line %d: expected register for source\n", line_num);
                return instr;
            }
            instr.src1 = src1;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            
            uint32_t imm;
            if (lexer_read_immediate(lex, &imm) < 0) {
                fprintf(stderr, "[parser] line %d: expected immediate value\n", line_num);
                return instr;
            }
            instr.imm = imm;
            break;
        }

        // set immediate: dst, #imm
        case PVA_SET1_F32: case PVA_SET1_I32: {
            int dst = lexer_read_register(lex);
            if (dst < 0) {
                fprintf(stderr, "[parser] line %d: expected register\n", line_num);
                return instr;
            }
            instr.dst = dst;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            
            uint32_t imm;
            if (lexer_read_immediate(lex, &imm) < 0) {
                fprintf(stderr, "[parser] line %d: expected immediate value\n", line_num);
                return instr;
            }
            instr.imm = imm;
            break;
        }

        // shuffle: dst, src1, src2, #imm
        case PVA_SHUFFLE: {
            int dst = lexer_read_register(lex);
            if (dst < 0) { fprintf(stderr, "[parser] line %d: expected register\n", line_num); return instr; }
            instr.dst = dst;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            int src1 = lexer_read_register(lex);
            if (src1 < 0) { fprintf(stderr, "[parser] line %d: expected register\n", line_num); return instr; }
            instr.src1 = src1;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            int src2 = lexer_read_register(lex);
            if (src2 < 0) { fprintf(stderr, "[parser] line %d: expected register\n", line_num); return instr; }
            instr.src2 = src2;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            uint32_t imm;
            if (lexer_read_immediate(lex, &imm) < 0) {
                fprintf(stderr, "[parser] line %d: expected shuffle immediate\n", line_num);
                return instr;
            }
            instr.imm = imm;
            break;
        }

        // load operations: dst, [base+offset]
        case PVA_LOAD_F32: case PVA_LOAD_F64: case PVA_LOAD_I32: {
            int dst = lexer_read_register(lex);
            if (dst < 0) {
                fprintf(stderr, "[parser] line %d: expected register for destination\n", line_num);
                return instr;
            }
            instr.dst = dst;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            if (lexer_peek(lex) == '[') lex->pos++;
            
            // try to read base register
            int base = lexer_read_register(lex);
            if (base >= 0) {
                instr.src1 = base;
            }
            
            // check for offset: +offset or +#imm
            lexer_skip_whitespace(lex);
            if (lex->input[lex->pos] == '+') {
                lex->pos++;
                uint32_t offset;
                if (lexer_read_immediate(lex, &offset) == 0) {
                    instr.imm = offset;
                }
            }
            
            // skip to ]
            while (lex->input[lex->pos] && lex->input[lex->pos] != ']') {
                lex->pos++;
            }
            if (lex->input[lex->pos] == ']') lex->pos++;
            break;
        }

        // store operations: [base+offset], src
        case PVA_STORE_F32: case PVA_STORE_F64: case PVA_STORE_I32: {
            // parse memory operand first: [base+offset]
            if (lexer_peek(lex) == '[') lex->pos++;
            
            int base = lexer_read_register(lex);
            if (base < 0) {
                fprintf(stderr, "[parser] line %d: expected base register in memory operand\n", line_num);
                return instr;
            }
            instr.dst = base;  // use dst for base address register
            
            // check for offset: +offset or +#imm
            lexer_skip_whitespace(lex);
            if (lex->input[lex->pos] == '+') {
                lex->pos++;
                uint32_t offset;
                if (lexer_read_immediate(lex, &offset) == 0) {
                    instr.imm = offset;
                }
            }
            
            // skip to ]
            while (lex->input[lex->pos] && lex->input[lex->pos] != ']') {
                lex->pos++;
            }
            if (lex->input[lex->pos] == ']') lex->pos++;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            
            // now read the source register
            int src = lexer_read_register(lex);
            if (src < 0) {
                fprintf(stderr, "[parser] line %d: expected source register\n", line_num);
                return instr;
            }
            instr.src1 = src;
            break;
        }

        // gather/scatter: dst, [base], index_reg
        case PVA_GATHER_F32: case PVA_SCATTER_F32: {
            int dst = lexer_read_register(lex);
            if (dst < 0) { fprintf(stderr, "[parser] line %d: expected register\n", line_num); return instr; }
            instr.dst = dst;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            if (lexer_peek(lex) == '[') lex->pos++;
            
            int base = lexer_read_register(lex);
            if (base < 0) { fprintf(stderr, "[parser] line %d: expected base register\n", line_num); return instr; }
            instr.src1 = base;
            
            while (lex->input[lex->pos] && lex->input[lex->pos] != ']') lex->pos++;
            if (lex->input[lex->pos] == ']') lex->pos++;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            
            int idx = lexer_read_register(lex);
            if (idx < 0) { fprintf(stderr, "[parser] line %d: expected index register\n", line_num); return instr; }
            instr.src2 = idx;
            break;
        }

        // single-operand instructions: dst
        case PVA_SETZERO: case PVA_SETONE: {
            int dst = lexer_read_register(lex);
            if (dst < 0) {
                fprintf(stderr, "[parser] line %d: expected register\n", line_num);
                return instr;
            }
            instr.dst = dst;
            break;
        }

        // label definition: label name:
        case PVA_LABEL: {
            char* name = lexer_read_label(lex);
            if (!name) {
                fprintf(stderr, "[parser] line %d: expected label name\n", line_num);
                return instr;
            }
            // skip colon if present
            lexer_skip_whitespace(lex);
            if (lex->input[lex->pos] == ':') lex->pos++;
            instr.label = name;
            break;
        }

        // jump instructions: jmp label or jmp_if mask, label
        case PVA_JMP: {
            char* name = lexer_read_label(lex);
            if (!name) {
                fprintf(stderr, "[parser] line %d: expected label name\n", line_num);
                return instr;
            }
            instr.label = name;
            break;
        }
        
        case PVA_JMP_IF: {
            int mask = lexer_read_register(lex);
            if (mask < 0) {
                fprintf(stderr, "[parser] line %d: expected mask register\n", line_num);
                return instr;
            }
            instr.src1 = mask;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            
            char* name = lexer_read_label(lex);
            if (!name) {
                fprintf(stderr, "[parser] line %d: expected label name\n", line_num);
                return instr;
            }
            instr.label = name;
            break;
        }

        // call: call label
        case PVA_CALL: {
            char* name = lexer_read_label(lex);
            if (!name) {
                fprintf(stderr, "[parser] line %d: expected function name\n", line_num);
                return instr;
            }
            instr.label = name;
            break;
        }

        // no operands
        case PVA_LOOP_BEGIN: case PVA_LOOP_END:
        case PVA_RET:
        case PVA_NOP:
            break;

        default:
            break;
    }

    return instr;
}

pva_module_t* pva_parse_file(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[parser] err: failed to open file '%s'\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *source = malloc(fsize + 1);
    if (!source) {
        fprintf(stderr, "[parser] err: memory alloc failed\n");
        fclose(fp);
        return NULL;
    }

    fread(source, 1, fsize, fp);
    source[fsize] = 0;
    fclose(fp);

    pva_module_t* mod = calloc(1, sizeof(pva_module_t));
    if (!mod) {
        free(source);
        return NULL;
    }

    mod->capacity = 1024;
    mod->code = calloc(mod->capacity, sizeof(pva_instr_t));
    mod->filename = calloc(strlen(filename) + 1, 1);
    strcpy(mod->filename, filename);

    if (!mod->code) {
        free(source);
        free(mod->filename);
        free(mod);
        return NULL;
    }

    pva_lexer_t lex = {source, 0, 1, 0};
    int errors = 0;

    while (lex.input[lex.pos]) {
        lexer_skip_whitespace(&lex);
        
        // skip comments (both # and ; style)
        if (lex.input[lex.pos] == '#' || lex.input[lex.pos] == ';') {
            while (lex.input[lex.pos] && lex.input[lex.pos] != '\n') {
                lex.pos++;
            }
            continue;
        }

        // skip empty lines
        if (lex.input[lex.pos] == '\n') {
            lex.pos++;
            lex.line++;
            lex.col = 0;
            continue;
        }

        if (!lex.input[lex.pos]) break;

        // parse instruction
        pva_instr_t instr = parse_instruction_line(&lex, lex.line);
        
        if (instr.op == PVA_UNKNOWN) {
            errors++;
            // Skip to next line
            while (lex.input[lex.pos] && lex.input[lex.pos] != '\n') {
                lex.pos++;
            }
            continue;
        }

        // add to module
        if (mod->size >= mod->capacity) {
            mod->capacity *= 2;
            pva_instr_t *new_code = realloc(mod->code, mod->capacity * sizeof(pva_instr_t));
            if (!new_code) {
                fprintf(stderr, "[parser] err: memory alloc failed\n");
                pva_free(mod);
                free(source);
                return NULL;
            }
            mod->code = new_code;
        }

        mod->code[mod->size++] = instr;

        // skip to next line
        while (lex.input[lex.pos] && lex.input[lex.pos] != '\n') {
            lex.pos++;
        }
        if (lex.input[lex.pos] == '\n') {
            lex.pos++;
            lex.line++;
            lex.col = 0;
        }
    }

    if (errors > 0) {
        fprintf(stderr, "[parser] warning: %d parse errors encountered\n", errors);
    }

    printf("[parser] successfully parsed %zu instructions from: '%s'\n", mod->size, filename);
    free(source);
    return mod;
}

void pva_free(pva_module_t* mod) {
    if (!mod) return;
    
    // free instruction labels
    if (mod->code) {
        for (size_t i = 0; i < mod->size; i++) {
            if (mod->code[i].label) {
                free(mod->code[i].label);
            }
        }
        free(mod->code);
    }
    
    // free label table
    if (mod->labels) {
        for (size_t i = 0; i < mod->label_count; i++) {
            if (mod->labels[i].name) {
                free(mod->labels[i].name);
            }
        }
        free(mod->labels);
    }
    
    if (mod->filename) free(mod->filename);
    free(mod);
}