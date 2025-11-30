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
TARGET = pva

# Default target
.PHONY: all clean run help

all: $(TARGET)

# Link executable
$(TARGET): $(OBJS)
	@echo "[Link] Building PVA compiler..."
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "[Success] Executable: $(TARGET)"

# Compile source files
%.o: %.c
	@echo "[Compile] $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Run example
run: $(TARGET)
	@echo "[Run] Compiling mandelbrot.pva..."
	./$(TARGET) examples/mandelbrot.pva -o mandelbrot.bin
	@echo "[Done] Output: mandelbrot.bin"

# Clean build artifacts
clean:
	@echo "[Clean] Removing objects and executable..."
	rm -f $(OBJS) $(TARGET)
	@echo "[Done]"

help:
	@echo "PVA Compiler - Portable Vector Assembly"
	@echo "Usage:"
	@echo "  make          - Build the compiler"
	@echo "  make run      - Build and run example"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Compiler usage:"
	@echo "  ./pva input.pva -o output.bin"
	@echo ""
	@echo "Supported architectures:"
	@echo "  - x86-64: AVX512, AVX2, SSE4.2"
	@echo "  - ARM64: SVE, NEON"
	@echo "  - RISC-V: RVV"
