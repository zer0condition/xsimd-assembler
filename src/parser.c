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
           lex->input[lex->pos] != '#') {
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
    if (reg < 0 || reg > 15) return -1;
    
    return reg;
}

static pva_opcode_t map_opcode(const char *opname) {
    if (strcmp(opname, "vadd") == 0) return PVA_ADD_F32;
    if (strcmp(opname, "vsub") == 0) return PVA_SUB_F32;
    if (strcmp(opname, "vmul") == 0) return PVA_MUL_F32;
    if (strcmp(opname, "vdiv") == 0) return PVA_DIV_F32;
    if (strcmp(opname, "vload") == 0) return PVA_LOAD_F32;
    if (strcmp(opname, "vstore") == 0) return PVA_STORE_F32;
    if (strcmp(opname, "vlt") == 0) return PVA_CMP_LT_F32;
    if (strcmp(opname, "veq") == 0) return PVA_CMP_EQ_F32;
    if (strcmp(opname, "vand") == 0) return PVA_AND_MASK;
    if (strcmp(opname, "vor") == 0) return PVA_OR_MASK;
    if (strcmp(opname, "vzero") == 0) return PVA_SETZERO;
    if (strcmp(opname, "loop_begin") == 0) return PVA_LOOP_BEGIN;
    if (strcmp(opname, "loop_end") == 0) return PVA_LOOP_END;
    return PVA_NOP;
}

static pva_instr_t parse_instruction_line(pva_lexer_t *lex, int line_num) {
    pva_instr_t instr = {0};
    instr.op = PVA_NOP;
    instr.mask_reg = -1;

    char opname[32];
    lexer_read_token(lex, opname, sizeof(opname));
    
    if (strlen(opname) == 0) return instr;
    
    pva_opcode_t op = map_opcode(opname);
    if (op == PVA_NOP) {
        fprintf(stderr, "[parser] line %d: unknown opcode '%s'\n", line_num, opname);
        return instr;
    }

    instr.op = op;

    switch (op) {
        case PVA_ADD_F32:
        case PVA_SUB_F32:
        case PVA_MUL_F32:
        case PVA_DIV_F32:
        case PVA_CMP_LT_F32:
        case PVA_CMP_EQ_F32:
        case PVA_AND_MASK:
        case PVA_OR_MASK: {
            // format: dst, src1, src2

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

        case PVA_LOAD_F32:
        case PVA_STORE_F32: {
            // format: reg, [address] or [address], reg
            int reg = lexer_read_register(lex);
            if (reg < 0) {
                fprintf(stderr, "[parser] line %d: expected register\n", line_num);
                return instr;
            }
            instr.dst = reg;
            
            if (lexer_peek(lex) == ',') lex->pos++;
            if (lexer_peek(lex) == '[') lex->pos++;  // Skip '['
            
            // read address (simplified - just skip to ]
            while (lexer_peek(lex) != ']' && lex->input[lex->pos]) {
                lex->pos++;
            }
            if (lexer_peek(lex) == ']') lex->pos++;
            break;
        }

        case PVA_SETZERO: {
            // format: dst
            int dst = lexer_read_register(lex);
            if (dst < 0) {
                fprintf(stderr, "[parser] line %d: expected register\n", line_num);
                return instr;
            }
            instr.dst = dst;
            break;
        }

        case PVA_LOOP_BEGIN:
        case PVA_LOOP_END:
            // :skull: enjoy
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
        
        // skip comments
        if (lex.input[lex.pos] == '#') {
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
        
        if (instr.op == PVA_NOP) {
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
    if (mod->code) free(mod->code);
    if (mod->filename) free(mod->filename);
    free(mod);
}