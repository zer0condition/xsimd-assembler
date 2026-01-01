CC = gcc
CFLAGS = -O3 -Wall -Wextra -Iinclude
LDFLAGS = -lm

# Native x86 build
ifeq ($(TARGET_ARCH),)
  CFLAGS += -mavx512f -mavx2 -march=native
endif

# Cross-compile for ARM64
ifeq ($(TARGET_ARCH),arm64)
  CC = aarch64-linux-gnu-gcc
  CFLAGS += -DTARGET_ARM64 -static
endif

# Cross-compile for RISC-V
ifeq ($(TARGET_ARCH),riscv64)
  CC = riscv64-linux-gnu-gcc
  CFLAGS += -DTARGET_RISCV64 -static -march=rv64gcv
endif

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
.PHONY: all clean run help test-arm test-riscv cross-arm cross-riscv

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
	rm -f $(OBJS) $(TARGET) xsimd-asm-arm64 xsimd-asm-riscv64
	rm -f tests/test_runner_arm64 tests/test_runner_riscv64
	@echo "[done]"

# Cross-compile targets
cross-arm:
	@echo "[cross] building for ARM64..."
	$(MAKE) clean
	$(MAKE) TARGET_ARCH=arm64
	mv xsimd-asm xsimd-asm-arm64
	aarch64-linux-gnu-gcc -O2 -static -o tests/test_runner_arm64 tests/test_runner.c
	@echo "[done] xsimd-asm-arm64, tests/test_runner_arm64"

cross-riscv:
	@echo "[cross] building for RISC-V..."
	$(MAKE) clean
	$(MAKE) TARGET_ARCH=riscv64
	mv xsimd-asm xsimd-asm-riscv64
	riscv64-linux-gnu-gcc -O2 -static -march=rv64gcv -o tests/test_runner_riscv64 tests/test_runner.c
	@echo "[done] xsimd-asm-riscv64, tests/test_runner_riscv64"

# Test with QEMU
test-arm: cross-arm
	@echo "[test] running ARM64 test with QEMU..."
	./xsimd-asm examples/comprehensive_x86_test.pva -o test_arm.bin
	qemu-aarch64-static tests/test_runner_arm64 test_arm.bin

test-riscv: cross-riscv
	@echo "[test] running RISC-V test with QEMU..."
	./xsimd-asm examples/comprehensive_x86_test.pva -o test_riscv.bin
	qemu-riscv64-static tests/test_runner_riscv64 test_riscv.bin

help:
	@echo "xsimd-asm - Cross-Platform SIMD Assembler"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build for native x86"
	@echo "  make cross-arm    - Cross-compile for ARM64"
	@echo "  make cross-riscv  - Cross-compile for RISC-V"
	@echo "  make test-arm     - Test ARM backend with QEMU"
	@echo "  make test-riscv   - Test RISC-V backend with QEMU"
	@echo "  make run          - Build and run example"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Assembler usage:"
	@echo "  ./xsimd-asm input.pva -o output.bin"
	@echo ""
	@echo "Supported architectures:"
	@echo "  - x86-64: AVX-512, AVX2, SSE4.2"
	@echo "  - ARM64: SVE, NEON"
	@echo "  - RISC-V: RVV"
