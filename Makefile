# q38 — Qwen3.8-Flash-Next / DGX Spark prototype (M0).
#
# Single supported target: DGX Spark / GB10 / Linux aarch64 / CUDA.
# No Metal, no ROCm, no CPU backend, no auto-detection. `make spark` builds
# ./q38 plus the M0 test binaries.

CC ?= cc
CFLAGS ?= -O3 -g -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -I.
MODEL_DIR ?= /home/lvx/q38model

# --- CUDA toolchain -----------------------------------------------------
CUDA_HOME ?= $(shell if [ -x /usr/local/cuda/bin/nvcc ]; then \
	printf '%s' /usr/local/cuda; \
	elif command -v nvcc >/dev/null 2>&1; then \
	dirname "$$(dirname "$$(command -v nvcc)")"; \
	else \
	printf '%s' /usr/local/cuda; \
	fi)
NVCC ?= $(CUDA_HOME)/bin/nvcc

# The build prints the chosen arch and refuses to link if it cannot produce
# code for the target. sm_121 is the DGX Spark (GB10) arch; the nvcc syntax
# is fixed to compute_121a/sm_121a (verified against the installed toolkit on
# Spark; see BASELINE.md).
CUDA_ARCH ?= sm_121
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS)

CUDA_LDLIBS ?= -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart

# --- Objects ------------------------------------------------------------
# q38_cuda.o is compiled by nvcc; the rest by cc.
C_OBJS := q38.o q38_gguf.o q38_memory.o q38_platform.o
CUDA_OBJS := q38_cuda.o
Q38_OBJS := $(C_OBJS) $(CUDA_OBJS)

TEST_DIR := tests
TEST_BINS := $(TEST_DIR)/test_platform $(TEST_DIR)/test_gguf \
	$(TEST_DIR)/test_memory $(TEST_DIR)/test_model_config
ARTIFACT_DIR := artifacts/m0
M1_ARTIFACT_DIR := artifacts/m1

.PHONY: all spark test clean m0-acceptance m1-inventory

all: spark

# --- CUDA object ----------------------------------------------------------
q38_cuda.o: q38_cuda.cu q38_cuda.h q38.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_cuda.cu

# --- C objects ------------------------------------------------------------
q38.o: q38.c q38.h q38_gguf.h q38_memory.h q38_platform.h q38_cuda.h
	$(CC) $(CFLAGS) -c -o $@ q38.c

q38_gguf.o: q38_gguf.c q38_gguf.h
	$(CC) $(CFLAGS) -c -o $@ q38_gguf.c

q38_memory.o: q38_memory.c q38_memory.h q38.h
	$(CC) $(CFLAGS) -c -o $@ q38_memory.c

q38_platform.o: q38_platform.c q38_platform.h q38_cuda.h q38.h
	$(CC) $(CFLAGS) -c -o $@ q38_platform.c

q38_model_config.o: q38_model_config.c q38_model_config.h
	$(CC) $(CFLAGS) -c -o $@ q38_model_config.c

# --- Executable -----------------------------------------------------------
q38: $(Q38_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(Q38_OBJS) $(CUDA_LDLIBS)

spark: q38 $(TEST_BINS)
	@echo "q38: built ./q38 and M0 test binaries"

# --- Tests ------------------------------------------------------------------
$(TEST_DIR)/test_platform: $(TEST_DIR)/test_platform.c q38_platform.o q38_cuda.o q38.h q38_platform.h q38_cuda.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_platform.c q38_platform.o q38_cuda.o $(CUDA_LDLIBS)

$(TEST_DIR)/test_gguf: $(TEST_DIR)/test_gguf.c q38_gguf.o q38.h q38_gguf.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_gguf.c q38_gguf.o

$(TEST_DIR)/test_memory: $(TEST_DIR)/test_memory.c q38_memory.o q38.h q38_memory.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_memory.c q38_memory.o

$(TEST_DIR)/test_model_config: $(TEST_DIR)/test_model_config.c q38_model_config.o q38_model_config.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_model_config.c q38_model_config.o

test: $(TEST_BINS)
	./$(TEST_DIR)/test_gguf
	./$(TEST_DIR)/test_memory
	./$(TEST_DIR)/test_model_config

m0-acceptance: spark
	@set -eu; \
	mkdir -p $(ARTIFACT_DIR); \
	echo "q38: running M0 acceptance"; \
	./$(TEST_DIR)/test_platform; \
	./$(TEST_DIR)/test_gguf; \
	./$(TEST_DIR)/test_memory; \
	printf '\107\107\125\106\003\000\000\000\000\000\000\000\000\000\000\000' > $(ARTIFACT_DIR)/test.gguf; \
	truncate -s 1G $(ARTIFACT_DIR)/test.gguf; \
	./q38 --inspect $(ARTIFACT_DIR)/test.gguf --json > $(ARTIFACT_DIR)/inspect.json; \
	./q38 --list-tensors $(ARTIFACT_DIR)/test.gguf --json > $(ARTIFACT_DIR)/tensors.json; \
	./q38 --memory-plan $(ARTIFACT_DIR)/test.gguf --json > $(ARTIFACT_DIR)/memory.json; \
	./q38 --platform-json > $(ARTIFACT_DIR)/platform.json; \
	grep -q '"version":3' $(ARTIFACT_DIR)/inspect.json; \
	grep -q '"tensors":0' $(ARTIFACT_DIR)/inspect.json; \
	grep -q '"tensors":\[\]' $(ARTIFACT_DIR)/tensors.json; \
	for key in phase rss_bytes mem_available_bytes cuda_free_bytes cuda_total_bytes \
		model_file_bytes model_mapped_bytes cuda_allocated_bytes peak_internal_bytes; do \
		grep -q "\"$$key\"" $(ARTIFACT_DIR)/memory.json; \
	done; \
	test "$$(stat -c '%s' $(ARTIFACT_DIR)/test.gguf)" -eq 1073741824; \
	/usr/bin/time -f '%M' -o $(ARTIFACT_DIR)/rss-kb.txt \
		./q38 --inspect $(ARTIFACT_DIR)/test.gguf >/dev/null; \
	rss_kb="$$(cat $(ARTIFACT_DIR)/rss-kb.txt)"; \
	test "$${rss_kb:-0}" -lt 262144; \
	for i in $$(seq 1 100); do ./$(TEST_DIR)/test_gguf >/dev/null 2>&1; done; \
	if grep -RInE 'cudaHostRegister[[:space:]]*\(' --exclude-dir=.git \
		--exclude-dir=to_be_deleted --exclude='*.md' .; then \
		echo "q38: forbidden whole-file host registration found" >&2; exit 1; \
	fi; \
	if nm -u q38 tests/test_platform | grep -E 'ds4_|DeepSeek|GLM|DSpark|MTP|Metal|ROCm'; then \
		echo "q38: out-of-scope symbols linked" >&2; exit 1; \
	fi; \
	echo "q38: M0 acceptance passed"

m1-inventory:
	@mkdir -p $(M1_ARTIFACT_DIR)
	python3 tools/q38_inventory.py --model-dir $(MODEL_DIR) \
		--output $(M1_ARTIFACT_DIR)/source_inventory.json

clean:
	rm -f q38 *.o $(TEST_DIR)/test_platform $(TEST_DIR)/test_gguf \
		$(TEST_DIR)/test_memory $(TEST_DIR)/test_model_config
	rm -rf $(ARTIFACT_DIR)
