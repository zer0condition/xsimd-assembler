#include "pva.h"
#include <stdio.h>
#include <string.h>

// uhh not too safe

typedef struct {
    pva_opcode_t op;
    uint8_t src1;
    uint8_t src2;
} instr_key_t;

typedef struct {
    int start_idx;
    int length;
    pva_opcode_t pattern[3];  // LOAD COMPUTE STORE
} FusiblePattern;

static int find_fusible_patterns(pva_module_t* mod, FusiblePattern* patterns, int max_patterns) {
    int count = 0;
    
    for (size_t i = 0; i < mod->size - 2 && count < max_patterns; i++) {
        // pattern: LOAD -> COMPUTE -> STORE
        if (mod->code[i].op == PVA_LOAD_F32 &&
            (mod->code[i+1].op >= PVA_ADD_F32 && mod->code[i+1].op <= PVA_DIV_F32) &&
            mod->code[i+2].op == PVA_STORE_F32) {
            
            // check data flow: LOAD dst must match COMPUTE src1
            if (mod->code[i].dst == mod->code[i+1].src1 ||
                mod->code[i].dst == mod->code[i+1].src2) {
                
                patterns[count].start_idx = i;
                patterns[count].length = 3;
                patterns[count].pattern[0] = PVA_LOAD_F32;
                patterns[count].pattern[1] = mod->code[i+1].op;
                patterns[count].pattern[2] = PVA_STORE_F32;
                count++;
            }
        }
    }
    
    return count;
}

static void eliminate_dead_code(pva_module_t* mod) {
    // track which registers are actually used
    int reg_used[16] = {0};
    
    // mark registers used in output operations
    for (size_t i = 0; i < mod->size; i++) {
        if (mod->code[i].op == PVA_STORE_F32) {
            reg_used[mod->code[i].dst] = 1;
        }
    }
    
    // backtrack to find dependencies
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = (int)mod->size - 1; i >= 0; i--) {
            if (reg_used[mod->code[i].dst]) {
                // If destination is used, mark sources as used
                if (mod->code[i].src1 < 16) {
                    if (!reg_used[mod->code[i].src1]) {
                        reg_used[mod->code[i].src1] = 1;
                        changed = 1;
                    }
                }
                if (mod->code[i].src2 < 16) {
                    if (!reg_used[mod->code[i].src2]) {
                        reg_used[mod->code[i].src2] = 1;
                        changed = 1;
                    }
                }
            }
        }
    }
    
    // remopve instructions whose destinations are never used
    size_t write_idx = 0;
    int removed = 0;
    
    for (size_t i = 0; i < mod->size; i++) {
        int is_dead = 0;
        
        // STORE and LOAD always keep (side effects)
        if (mod->code[i].op != PVA_STORE_F32 && mod->code[i].op != PVA_LOAD_F32) {
            if (!reg_used[mod->code[i].dst]) {
                is_dead = 1;
                removed++;
            }
        }
        
        if (!is_dead) {
            mod->code[write_idx++] = mod->code[i];
        }
    }
    
    mod->size = write_idx;
    
    if (removed > 0) {
        printf("[optimizer]     removed %d dead code instructions\n", removed);
    }
}


#define HASH_TABLE_SIZE 1024

static size_t hash_instr_key(const instr_key_t *key) {
    return (key->op * 31 + key->src1 * 17 + key->src2) % HASH_TABLE_SIZE;
}

void combine_commutative_ops(pva_module_t* mod) {
    int combined = 0;
    pva_instr_t *hash_table[HASH_TABLE_SIZE];
    memset(hash_table, 0, sizeof(hash_table));

    for (size_t i = 0; i < mod->size; i++) {
        pva_instr_t *instr = &mod->code[i];

        if (instr->op != PVA_ADD_F32 && instr->op != PVA_MUL_F32)
            continue;

        instr_key_t key1 = {instr->op, instr->src1, instr->src2};
        instr_key_t key2 = {instr->op, instr->src2, instr->src1}; // commutative order

        size_t h1 = hash_instr_key(&key1);
        size_t h2 = hash_instr_key(&key2);

        pva_instr_t *found = NULL;

        if (hash_table[h1] &&
            ((hash_table[h1]->src1 == instr->src1 && hash_table[h1]->src2 == instr->src2) ||
             (hash_table[h1]->src1 == instr->src2 && hash_table[h1]->src2 == instr->src1)) ) {
            found = hash_table[h1];
        } else if (hash_table[h2] &&
            ((hash_table[h2]->src1 == instr->src1 && hash_table[h2]->src2 == instr->src2) ||
             (hash_table[h2]->src1 == instr->src2 && hash_table[h2]->src2 == instr->src1)) ) {
            found = hash_table[h2];
        }

        if (found) {
            // replace result register with existing computed register
            instr->op = PVA_NOP; // mark as removed
            instr->dst = 0;
            instr->src1 = 0;
            instr->src2 = 0;
            combined++;
        } else {
            // insert into hash table
            hash_table[h1] = instr;
        }
    }

    if (combined > 0) {
        printf("[optimizer] found and removed %d common subexpressions\n", combined);
    }
}

int calculate_instruction_level_parallelism(pva_module_t* mod) {
    int max_chain = 0;
    int current_chain = 1;
    int last_dst = -1;

    for (size_t i = 0; i < mod->size; i++) {
        int has_dependency = 0;

        if (last_dst >= 0 &&
            (mod->code[i].src1 == last_dst || mod->code[i].src2 == last_dst)) {
            has_dependency = 1;
        }

        if (has_dependency) {
            current_chain++;
        } else {
            if (current_chain > max_chain) max_chain = current_chain;
            current_chain = 1;
        }

        last_dst = mod->code[i].dst;
    }

    if (current_chain > max_chain) max_chain = current_chain;
    return max_chain;
}

void strength_reduce(pva_module_t* mod) {
    int reductions = 0;

    for (size_t i = 0; i < mod->size; i++) {
        pva_instr_t *instr = &mod->code[i];
        
        // skull emoji
        if (instr->op == PVA_MUL_F32 && instr->imm == 2) {
            // replace vmul dst, src, #2 with vadd dst, src, src
            instr->op = PVA_ADD_F32;
            instr->src2 = instr->src1;
            instr->imm = 0;
            reductions++;
        }
    }

    if (reductions > 0) {
        printf("[optimizer] applied %d strength reductions\n", reductions);
    }
}

void pva_optimize(pva_module_t* mod) {
    if (!mod || mod->size == 0) return;

    printf("\n[optimizer] starting optimization pass...\n");
    printf("[optimizer] input: %zu instructions\n", mod->size);

    // Pass 1: remove NOPs
    printf("[optimizer] pass 1: removing NOPs...\n");
    size_t write_idx = 0;
    int nop_count = 0;
    
    for (size_t read_idx = 0; read_idx < mod->size; read_idx++) {
        if (mod->code[read_idx].op == PVA_NOP) {
            nop_count++;
            continue;
        }
        mod->code[write_idx++] = mod->code[read_idx];
    }
    
    mod->size = write_idx;
    if (nop_count > 0) {
        printf("[optimizer]     removed %d NOPs\n", nop_count);
    }

    // Pass 2: dead code elimination
    printf("[optimizer] pass 2: dead code elimination...\n");
    eliminate_dead_code(mod);

    // Pass 3: detect fusible patterns
    printf("[optimizer] pass 3: instruction fusion analysis...\n");
    FusiblePattern patterns[256];
    int pattern_count = find_fusible_patterns(mod, patterns, 256);
    if (pattern_count > 0) {
        printf("[optimizer]     found %d fusible patterns (LOAD->COMPUTE->STORE)\n", pattern_count);
    }

    // Pass 4: common subexpression elimination 
    printf("[optimizer] pass 4: common subexpression elimination?...\n");
    combine_commutative_ops(mod);

    // Pass 5: instruction level parallelism analysis
    printf("[optimizer] pass 5: parallelism analysis...\n");
    int max_chain = calculate_instruction_level_parallelism(mod);
    printf("[optimizer]   max dependency chain: %d instructions\n", max_chain);

    // Pass 6: strength reduction 
    printf("[optimizer] pass 6: strength reduction...\n");
    strength_reduce(mod);

    printf("[optimizer] optimization complete!\n");
    printf("[optimizer] output: %zu instructions\n", mod->size);
}