CC = gcc
CFLAGS = -O3 -Wall -Wextra -mavx512f -mavx2 -march=native -Iinclude
LDFLAGS = -lm

# Source files
SRCS = src/main.c \
       src/detect_arch.c \
       src/parser.c \
       src/optimizer.c \
       src/backends/x86.c \
       src/backends/arm.c \
       src/backends/riscv.c

OBJS = $(SRCS:.c=.o)
TARGET = xsimd-asm

# Default target
.PHONY: all clean run help

all: $(TARGET)

# Link executable
$(TARGET): $(OBJS)
	@echo "[link] building xsimd-asm..."
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "[success] executable: $(TARGET)"

# Compile source files
%.o: %.c
	@echo "[compile] $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Run example
run: $(TARGET)
	@echo "[run] assembling mandelbrot.pva..."
	./$(TARGET) examples/mandelbrot.pva -o mandelbrot.bin
	@echo "[done] output: mandelbrot.bin"

# Clean build artifacts
clean:
	@echo "[clean] removing objects and executable..."
	rm -f $(OBJS) $(TARGET)
	@echo "[done]"

help:
	@echo "xsimd-asm - Cross-Platform SIMD Assembler"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build the assembler"
	@echo "  make run      - Build and run example"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Assembler usage:"
	@echo "  ./xsimd-asm input.pva -o output.bin"
	@echo ""
	@echo "Supported architectures:"
	@echo "  - x86-64: AVX-512, AVX2, SSE4.2"
	@echo "  - ARM64: SVE, NEON"
	@echo "  - RISC-V: RVV"
