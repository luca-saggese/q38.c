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
	$(TEST_DIR)/test_memory $(TEST_DIR)/test_model_config \
	$(TEST_DIR)/test_quant_blocks
ARTIFACT_DIR := artifacts/m0
M1_ARTIFACT_DIR := artifacts/m1
M2_ARTIFACT_DIR := artifacts/m2

.PHONY: all spark test clean m0-acceptance m1-inventory m1-validate m1-subset \
	m1-bind m1-quant-block m1-full m1-memory-matrix m1-acceptance m2-c00 m2-c01

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

q38_golden.o: q38_golden.c q38_golden.h
	$(CC) $(CFLAGS) -c -o $@ q38_golden.c

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

$(TEST_DIR)/test_quant_blocks: $(TEST_DIR)/test_quant_blocks.c \
		to_be_deleted/gguf-tools/quants.c to_be_deleted/gguf-tools/quants.h
	$(CC) $(CFLAGS) -Ito_be_deleted/gguf-tools -o $@ \
		$(TEST_DIR)/test_quant_blocks.c to_be_deleted/gguf-tools/quants.c \
		-lm -lpthread

$(TEST_DIR)/test_m2_golden: $(TEST_DIR)/test_m2_golden.c q38_golden.o q38_golden.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m2_golden.c q38_golden.o

$(TEST_DIR)/test_m2_weights: $(TEST_DIR)/test_m2_weights.c q38_weights.o \
		q38_gguf.o q38_model_config.o q38_weights.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m2_weights.c q38_weights.o \
		q38_gguf.o q38_model_config.o

test: $(TEST_BINS)
	./$(TEST_DIR)/test_gguf
	./$(TEST_DIR)/test_memory
	./$(TEST_DIR)/test_model_config
	./$(TEST_DIR)/test_quant_blocks

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

m1-validate: m1-inventory
	python3 tools/q38_classify.py $(M1_ARTIFACT_DIR)/source_inventory.json \
		--output $(M1_ARTIFACT_DIR)/tensor_classes.json
	python3 tools/q38_validate_manifest.py \
		$(M1_ARTIFACT_DIR)/tensor_classes.json tools/quant_manifest_q2.json

m1-quant-block: m1-validate $(TEST_DIR)/test_quant_blocks
	python3 tools/q38_quant_block_test.py \
		$(M1_ARTIFACT_DIR)/tensor_classes.json tools/quant_manifest_q2.json
	./$(TEST_DIR)/test_quant_blocks

m1-subset: m1-validate tools/q38_quantize
	python3 tools/convert_q38_gguf.py --model-dir $(MODEL_DIR) \
		--inventory $(M1_ARTIFACT_DIR)/source_inventory.json \
		--classes $(M1_ARTIFACT_DIR)/tensor_classes.json \
		--manifest tools/quant_manifest_q2.json \
		--output $(M1_ARTIFACT_DIR)/unused.gguf \
		--max-layer 3 --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
		--quantize --plan-output $(M1_ARTIFACT_DIR)/subset_conversion_plan.json
	@test "$$(python3 -c 'import json; print(json.load(open("$(M1_ARTIFACT_DIR)/subset_conversion_plan.json"))["status"])')" = pass
	python3 tools/convert_q38_gguf.py --model-dir $(MODEL_DIR) \
		--inventory $(M1_ARTIFACT_DIR)/source_inventory.json \
		--classes $(M1_ARTIFACT_DIR)/tensor_classes.json \
		--manifest tools/quant_manifest_q2.json \
		--output $(M1_ARTIFACT_DIR)/qwen38-runtime-only-layers0-3-Q2Experts-BF16Core-BF16PLE.gguf \
		--max-layer 3 \
		--revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
		--quantize --quantizer ./tools/q38_quantize

tools/q38_quantize: tools/q38_quantize.c to_be_deleted/gguf-tools/quants.c \
		to_be_deleted/gguf-tools/quants.h
	$(CC) $(CFLAGS) -Ito_be_deleted/gguf-tools -o $@ \
		tools/q38_quantize.c to_be_deleted/gguf-tools/quants.c -lm -lpthread

m1-full: m1-validate tools/q38_quantize
	python3 tools/convert_q38_gguf.py --model-dir $(MODEL_DIR) \
		--inventory $(M1_ARTIFACT_DIR)/source_inventory.json \
		--classes $(M1_ARTIFACT_DIR)/tensor_classes.json \
		--manifest tools/quant_manifest_q2.json \
		--output $(M1_ARTIFACT_DIR)/unused.gguf \
		--max-layer 47 --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
		--quantize --plan-output $(M1_ARTIFACT_DIR)/full_conversion_plan.json
	@test "$$(python3 -c 'import json; print(json.load(open("$(M1_ARTIFACT_DIR)/full_conversion_plan.json"))["status"])')" = pass
	python3 tools/convert_q38_gguf.py --model-dir $(MODEL_DIR) \
		--inventory $(M1_ARTIFACT_DIR)/source_inventory.json \
		--classes $(M1_ARTIFACT_DIR)/tensor_classes.json \
		--manifest tools/quant_manifest_q2.json \
		--output $(M1_ARTIFACT_DIR)/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		--max-layer 47 --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
		--quantize --quantizer ./tools/q38_quantize

m1-memory-matrix: m1-subset
	sh tools/run_memory_matrix.sh \
		$(M1_ARTIFACT_DIR)/qwen38-runtime-only-layers0-3-Q2Experts-BF16Core-BF16PLE.gguf \
		$(M1_ARTIFACT_DIR)

m1-acceptance: m1-validate m1-quant-block m1-bind
	python3 tools/m1_acceptance.py --artifact-dir $(M1_ARTIFACT_DIR) \
		--inventory $(M1_ARTIFACT_DIR)/source_inventory.json \
		--classes $(M1_ARTIFACT_DIR)/tensor_classes.json \
		--manifest tools/quant_manifest_q2.json

q38_weights.o: q38_weights.c q38_weights.h q38_gguf.h q38_model_config.h
	$(CC) $(CFLAGS) -c -o $@ q38_weights.c

$(TEST_DIR)/test_weights: $(TEST_DIR)/test_weights.c q38_weights.o q38_gguf.o \
	q38_model_config.o q38_weights.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_weights.c q38_weights.o \
		q38_gguf.o q38_model_config.o

m1-bind: m1-subset $(TEST_DIR)/test_weights
	./$(TEST_DIR)/test_weights \
		$(M1_ARTIFACT_DIR)/qwen38-runtime-only-layers0-3-Q2Experts-BF16Core-BF16PLE.gguf

m2-c00: $(TEST_DIR)/test_m2_golden
	@mkdir -p $(M2_ARTIFACT_DIR)
	./$(TEST_DIR)/test_m2_golden
	printf '%s\n' \
		'{"format":"q38-golden-binary","version":1,"endianness":"little","checksum":"FNV-1a-64","metadata":"fixed q38_golden_meta"}' \
		> $(M2_ARTIFACT_DIR)/golden_format_version.json

m2-c01: m2-c00 $(TEST_DIR)/test_weights $(TEST_DIR)/test_m2_weights
	./$(TEST_DIR)/test_m2_weights
	@test -f $(M1_ARTIFACT_DIR)/runtime-layers0-q2-test.gguf || \
		{ echo "M2-C01: required M1 subset artifact is unavailable" >&2; exit 1; }
	./$(TEST_DIR)/test_weights \
		$(M1_ARTIFACT_DIR)/runtime-layers0-q2-test.gguf 0
	@mkdir -p $(M2_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M2-C01","binding":"strict","subset_max_layer":0,"required_layers":1,"tensor_count":29,"payload":"mmap views only","status":"pass"}' \
		> $(M2_ARTIFACT_DIR)/binding_report.json

clean:
	rm -f q38 *.o $(TEST_DIR)/test_platform $(TEST_DIR)/test_gguf \
		$(TEST_DIR)/test_memory $(TEST_DIR)/test_model_config \
		$(TEST_DIR)/test_weights $(TEST_DIR)/test_quant_blocks \
		$(TEST_DIR)/test_m2_golden \
		$(TEST_DIR)/test_m2_weights \
		tools/q38_quantize
	rm -rf $(ARTIFACT_DIR) $(M2_ARTIFACT_DIR)
