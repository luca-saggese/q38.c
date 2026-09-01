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
M3_ARTIFACT_DIR := artifacts/m3
M4_ARTIFACT_DIR := artifacts/m4

.PHONY: all spark test clean m0-acceptance m1-inventory m1-validate m1-subset \
	m1-bind m1-quant-block m1-full m1-memory-matrix m1-acceptance m2-c00 m2-c01 \
	m2-c02 m2-c03 m2-c04 m2-c05 m2-c06 m2-c07 m2-c08 m2-c09 m2-c10 \
	m2-c11 m2-acceptance m3-c00 m3-c01 m3-c02 m3-c03 m3-c04 m3-c05 \
	m3-c06 m3-c07 m3-c08 m3-c09 m3-c10 m3-c11 m3-c12 m3-c13 m3-audit \
	m3-acceptance m4-c00 m4-c01 m4-c02 m4-c03 m4-c04 m4-c05 m4-c06 m4-c07 \
	m4-c08

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

q38_session.o: q38_session.c q38_session.h
	$(CC) $(CFLAGS) -c -o $@ q38_session.c

q38_ple_ref.o: q38_ple_ref.c q38_ple_ref.h q38_session.h q38_quant.h
	$(CC) $(CFLAGS) -c -o $@ q38_ple_ref.c

q38_ple.o: q38_ple.c q38_ple.h q38_gguf.h
	$(CC) $(CFLAGS) -c -o $@ q38_ple.c

q38_ple_cache.o: q38_ple_cache.c q38_ple_cache.h
	$(CC) $(CFLAGS) -c -o $@ q38_ple_cache.c

q38_golden.o: q38_golden.c q38_golden.h
	$(CC) $(CFLAGS) -c -o $@ q38_golden.c

q38_tokenizer.o: q38_tokenizer.c q38_tokenizer.h
	$(CC) $(CFLAGS) -c -o $@ q38_tokenizer.c

q38_quant.o: q38_quant.c q38_quant.h
	$(CC) $(CFLAGS) -c -o $@ q38_quant.c

q38_oracle.o: q38_oracle.c q38_oracle.h
	$(CC) $(CFLAGS) -c -o $@ q38_oracle.c

q38_cuda_primitives.o: q38_cuda_primitives.cu q38_cuda_primitives.h q38_quant.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_cuda_primitives.cu

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

$(TEST_DIR)/test_m2_tokenizer: $(TEST_DIR)/test_m2_tokenizer.c \
		q38_tokenizer.o q38_tokenizer.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m2_tokenizer.c q38_tokenizer.o

$(TEST_DIR)/test_m4_session: $(TEST_DIR)/test_m4_session.c q38_session.o q38_session.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m4_session.c q38_session.o

m4-c00:
	@test -f docs/qwen_ple_semantics.md
	@grep -q "mixed_n" docs/qwen_ple_semantics.md
	@grep -q "build_ple" docs/qwen_ple_semantics.md
	@echo "M4-C00: PLE semantics freeze passed"

m4-c01: m4-c00 $(TEST_DIR)/test_m4_session
	@./$(TEST_DIR)/test_m4_session
	@mkdir -p $(M4_ARTIFACT_DIR)
	printf '%s\n' '{"gate":"M4-C01","history":"two-token session history","eos_reset":true,"chunk_boundary":true,"status":"pass"}' > $(M4_ARTIFACT_DIR)/session_history.json

$(TEST_DIR)/test_m4_ple_ref: $(TEST_DIR)/test_m4_ple_ref.c q38_ple_ref.o \
		q38_session.o q38_quant.o q38_ple_ref.h q38_session.h q38_quant.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m4_ple_ref.c \
		q38_ple_ref.o q38_session.o q38_quant.o -lm

m4-c02: m4-c01 $(TEST_DIR)/test_m4_ple_ref
	@./$(TEST_DIR)/test_m4_ple_ref
	@mkdir -p $(M4_ARTIFACT_DIR)
	printf '%s\n' '{"gate":"M4-C02","oracle":"scalar uint64 XOR/multiply modulo indexing","heads":16,"orders":["bigram","trigram"],"status":"pass"}' > $(M4_ARTIFACT_DIR)/ple_ref.json

$(TEST_DIR)/test_m4_ple_store: $(TEST_DIR)/test_m4_ple_store.c q38_ple.o \
		q38_ple.h q38_gguf.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m4_ple_store.c q38_ple.o

m4-c03: m4-c02 $(TEST_DIR)/test_m4_ple_store
	@./$(TEST_DIR)/test_m4_ple_store
	@mkdir -p $(M4_ARTIFACT_DIR)
	printf '%s\n' '{"gate":"M4-C03","table":"dedicated PLE descriptor","row_width":2560,"strict_geometry":true,"status":"pass"}' > $(M4_ARTIFACT_DIR)/ple_store.json

$(TEST_DIR)/test_m4_ple_row: $(TEST_DIR)/test_m4_ple_row.c \
		q38_ple_ref.o q38_session.o q38_quant.o q38_ple_ref.h \
		q38_session.h q38_quant.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m4_ple_row.c \
		q38_ple_ref.o q38_session.o q38_quant.o -lm

m4-c04: m4-c03 $(TEST_DIR)/test_m4_ple_row
	@./$(TEST_DIR)/test_m4_ple_row
	@mkdir -p $(M4_ARTIFACT_DIR)
	printf '%s\n' '{"gate":"M4-C04","decoder":"scalar PLE row","qtypes":["Q2_K","Q4_K"],"row_width":2560,"status":"pass"}' > $(M4_ARTIFACT_DIR)/ple_row_quant_tests.json

q38_ple_cuda.o: q38_ple_cuda.cu q38_ple_cuda.h q38_quant.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_ple_cuda.cu

$(TEST_DIR)/test_m4_ple_cuda: $(TEST_DIR)/test_m4_ple_cuda.cu \
		q38_ple_cuda.o q38_ple_ref.o q38_session.o q38_quant.o \
		q38_ple_cuda.h q38_ple_ref.h q38_quant.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m4_ple_cuda.cu \
		q38_ple_cuda.o q38_ple_ref.o q38_session.o q38_quant.o \
		$(CUDA_LDLIBS) -lm

m4-c05: m4-c04 $(TEST_DIR)/test_m4_ple_cuda
	@./$(TEST_DIR)/test_m4_ple_cuda
	@mkdir -p $(M4_ARTIFACT_DIR)
	printf '%s\n' '{"gate":"M4-C05","lookup":"naive CUDA row decode","qtypes":["Q2_K","Q4_K"],"cache":"disabled","status":"pass"}' > $(M4_ARTIFACT_DIR)/ple_cuda_lookup.json

M4_C06_GOLDEN := $(M4_ARTIFACT_DIR)/ple_injection_golden.json

$(M4_C06_GOLDEN): tools/generate_m4_c06_goldens.py \
		$(MODEL_DIR)/config.json $(MODEL_DIR)/model.safetensors.index.json
	@mkdir -p $(M4_ARTIFACT_DIR)
	python3 tools/generate_m4_c06_goldens.py --model-dir $(MODEL_DIR) \
		--output $@

$(TEST_DIR)/q38_forward_probe: q38_forward_probe.c q38_ple_ref.o \
		q38_session.o q38_quant.o q38_ple_ref.h q38_session.h q38_quant.h
	$(CC) $(CFLAGS) -o $@ q38_forward_probe.c \
		q38_ple_ref.o q38_session.o q38_quant.o

m4-c06: m4-c05 $(M4_C06_GOLDEN) $(TEST_DIR)/q38_forward_probe
	@./$(TEST_DIR)/q38_forward_probe $(M4_C06_GOLDEN)
	@printf '%s\n' '{"gate":"M4-C06","probe":"q38_forward_probe","golden":"ple_injection_golden.json","coverage":"PLE IDs and exact injection boundary metadata","hidden_vectors":"unavailable; full model forward not implemented","status":"pass"}' > $(M4_ARTIFACT_DIR)/ple_injection_probe.json

$(TEST_DIR)/test_ple_chunking: $(TEST_DIR)/test_ple_chunking.c \
		q38_ple_ref.o q38_session.o q38_quant.o q38_ple_ref.h \
		q38_session.h q38_quant.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_ple_chunking.c \
		q38_ple_ref.o q38_session.o q38_quant.o

m4-c07: m4-c06 $(TEST_DIR)/test_ple_chunking
	@./$(TEST_DIR)/test_ple_chunking $(M4_C06_GOLDEN)
	@mkdir -p $(M4_ARTIFACT_DIR)
	@printf '%s\n' '{"gate":"M4-C07","harness":"test_ple_chunking","cases":"independent M4-C06 golden corpus","partitions":["single","one-token","two-tail","three-five-tail","four-token","random-seed-1","random-seed-2"],"ids":"exact","hidden_logits":"unavailable; full model forward not implemented","status":"pass"}' > $(M4_ARTIFACT_DIR)/ple_chunk_invariance.json

$(TEST_DIR)/test_m4_ple_cache: $(TEST_DIR)/test_m4_ple_cache.c \
		q38_ple_cache.o q38_ple_ref.o q38_session.o q38_quant.o \
		q38_ple_cache.h q38_ple_ref.h q38_quant.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m4_ple_cache.c \
		q38_ple_cache.o q38_ple_ref.o q38_session.o q38_quant.o -lm

m4-c08: m4-c07 $(TEST_DIR)/test_m4_ple_cache
	@mkdir -p $(M4_ARTIFACT_DIR)
	@./$(TEST_DIR)/test_m4_ple_cache \
		$(M4_ARTIFACT_DIR)/ple_cache_stats_cold.json \
		$(M4_ARTIFACT_DIR)/ple_cache_stats_warm.json
	@printf '%s\n' '{"gate":"M4-C08","cache":"bounded deterministic quantized-row cache","value":"quantized row bytes","replacement":"round-robin","cold_warm":"equivalent","full_dequant_mirror":false,"status":"pass"}' > $(M4_ARTIFACT_DIR)/ple_cache_stats.json

.PHONY: tokenizer-runtime-gate
tokenizer-runtime-gate:
	@$(MAKE) --no-print-directory q38_tokenizer.o tests/test_m2_tokenizer
	@! nm -u q38_tokenizer.o 2>/dev/null | grep -E 'fork|exec|python|transformers|tokenizers' >/dev/null
	@! grep -nE 'fork|exec|python|transformers|tokenizers' q38_tokenizer.c q38_tokenizer.h >/dev/null
	@./$(TEST_DIR)/test_m2_tokenizer
	@echo "native tokenizer runtime gate: passed"

$(TEST_DIR)/test_m2_quant: $(TEST_DIR)/test_m2_quant.c q38_quant.o \
		q38_oracle.o q38_quant.h q38_oracle.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m2_quant.c q38_quant.o \
		q38_oracle.o -lm

$(TEST_DIR)/test_m2_cuda: $(TEST_DIR)/test_m2_cuda.cu q38_cuda_primitives.o \
		q38_quant.o q38_oracle.o q38_cuda_primitives.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m2_cuda.cu \
		q38_cuda_primitives.o q38_quant.o q38_oracle.o $(CUDA_LDLIBS) -lm

$(TEST_DIR)/test_m2_norm: $(TEST_DIR)/test_m2_norm.cu q38_cuda_primitives.o \
		q38_oracle.o q38_cuda_primitives.h q38_oracle.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m2_norm.cu \
		q38_cuda_primitives.o q38_oracle.o $(CUDA_LDLIBS) -lm

$(TEST_DIR)/test_m2_matvec: $(TEST_DIR)/test_m2_matvec.cu \
		q38_cuda_primitives.o q38_quant.o q38_oracle.o q38_cuda_primitives.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m2_matvec.cu \
		q38_cuda_primitives.o q38_quant.o q38_oracle.o $(CUDA_LDLIBS) -lm

$(TEST_DIR)/test_m2_embedding: $(TEST_DIR)/test_m2_embedding.c \
		q38_weights.o q38_gguf.o q38_model_config.o q38_weights.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m2_embedding.c q38_weights.o \
		q38_gguf.o q38_model_config.o

$(TEST_DIR)/test_m2_lm_head: $(TEST_DIR)/test_m2_lm_head.cu \
		q38_cuda_primitives.o q38_oracle.o q38_weights.o q38_gguf.o \
		q38_model_config.o q38_cuda_primitives.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m2_lm_head.cu \
		q38_cuda_primitives.o q38_oracle.o q38_weights.o q38_gguf.o \
		q38_model_config.o $(CUDA_LDLIBS) -lm

$(TEST_DIR)/test_m2_memory: $(TEST_DIR)/test_m2_memory.c \
		q38_weights.o q38_gguf.o q38_model_config.o q38_weights.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m2_memory.c q38_weights.o \
		q38_gguf.o q38_model_config.o

$(TEST_DIR)/test_m3_gr_binding: $(TEST_DIR)/test_m3_gr_binding.c \
		q38_weights.o q38_gguf.o q38_model_config.o q38_weights.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m3_gr_binding.c q38_weights.o \
		q38_gguf.o q38_model_config.o

q38_gr_ref.o: q38_gr_ref.c q38_gr_ref.h
	$(CC) $(CFLAGS) -c -o $@ q38_gr_ref.c

$(TEST_DIR)/test_m3_gr_ref: $(TEST_DIR)/test_m3_gr_ref.c q38_gr_ref.o \
		q38_gr_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m3_gr_ref.c q38_gr_ref.o -lm

q38_gr.o: q38_gr.cu q38_gr.h q38_gr_ref.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_gr.cu

$(TEST_DIR)/test_m3_gr_cuda: $(TEST_DIR)/test_m3_gr_cuda.cu q38_gr.o \
		q38_gr_ref.o q38_oracle.o q38_gr.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m3_gr_cuda.cu \
		q38_gr.o q38_gr_ref.o q38_oracle.o $(CUDA_LDLIBS) -lm

$(TEST_DIR)/test_m3_gdn_binding: $(TEST_DIR)/test_m3_gdn_binding.c \
		q38_weights.o q38_gguf.o q38_model_config.o q38_weights.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m3_gdn_binding.c q38_weights.o \
		q38_gguf.o q38_model_config.o

q38_state.o: q38_state.c q38_state.h
	$(CC) $(CFLAGS) -c -o $@ q38_state.c

$(TEST_DIR)/test_m3_state: $(TEST_DIR)/test_m3_state.c q38_state.o q38_state.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m3_state.c q38_state.o

q38_gdn_ref.o: q38_gdn_ref.c q38_gdn_ref.h q38_state.h
	$(CC) $(CFLAGS) -c -o $@ q38_gdn_ref.c

$(TEST_DIR)/test_m3_gdn_ref: $(TEST_DIR)/test_m3_gdn_ref.c q38_gdn_ref.o \
		q38_gdn_ref.h q38_state.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m3_gdn_ref.c q38_gdn_ref.o -lm

q38_gdn.o: q38_gdn.cu q38_gdn.h q38_cuda_primitives.h q38_quant.h q38_state.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_gdn.cu

$(TEST_DIR)/test_m3_gdn_cuda: $(TEST_DIR)/test_m3_gdn_cuda.cu q38_gdn.o \
		q38_cuda_primitives.o q38_oracle.o q38_gdn.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m3_gdn_cuda.cu \
		q38_gdn.o q38_cuda_primitives.o q38_oracle.o $(CUDA_LDLIBS) -lm

m3-c08: m3-c07 $(TEST_DIR)/test_m3_gdn_recurrence_cuda
	./$(TEST_DIR)/test_m3_gdn_recurrence_cuda
	@mkdir -p $(M3_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M3-C08","kernel":"simple CUDA FP32 GDN recurrence baseline","state_logical_shape":["sequence","value_head","row","column"],"state_dtype":"F32","state_layout":"contiguous logical row-major; sequence=1","inputs":{"q":["token",48,128],"k":["token",48,128],"v":["token",48,128],"decay":["token",48],"beta":["token",48]},"equations":["Sbar=decay*S_prev","prediction=Sbar^T*k","delta=(v-prediction)*beta","S_next=Sbar+k*delta^T","y=scale*S_next^T*q"],"token_cases":[1,2,4,5],"checks":["output parity","state parity","zero reset","stream synchronization","CUDA launch/runtime errors"],"status":"pass"}' \
		> $(M3_ARTIFACT_DIR)/gdn_recurrence_cuda.json

$(TEST_DIR)/test_m3_gdn_recurrence_cuda: \
		$(TEST_DIR)/test_m3_gdn_recurrence_cuda.cu q38_gdn.o q38_gdn_ref.o \
		q38_cuda_primitives.o q38_gdn.h q38_gdn_ref.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ \
		$(TEST_DIR)/test_m3_gdn_recurrence_cuda.cu q38_gdn.o q38_gdn_ref.o \
		q38_cuda_primitives.o \
		$(CUDA_LDLIBS) -lm

$(TEST_DIR)/test_m3_forward_probe: $(TEST_DIR)/test_m3_forward_probe.cu \
		q38_gr.o q38_gr_ref.o q38_gdn.o q38_gdn_ref.o \
		q38_cuda_primitives.o q38_oracle.o \
		q38_gr.h q38_gdn.h q38_gdn_ref.h q38_oracle.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m3_forward_probe.cu \
		q38_gr.o q38_gr_ref.o q38_gdn.o q38_gdn_ref.o \
		q38_cuda_primitives.o q38_oracle.o $(CUDA_LDLIBS) -lm

m3-c09: m3-c08 $(TEST_DIR)/test_m3_forward_probe
	@mkdir -p $(M3_ARTIFACT_DIR)
	./$(TEST_DIR)/test_m3_forward_probe \
		$(M3_ARTIFACT_DIR)/layer0_intermediates.json

$(TEST_DIR)/test_m3_multigdn: $(TEST_DIR)/test_m3_multigdn.cu \
		q38_gdn.o q38_gdn_ref.o q38_cuda_primitives.o \
		q38_gdn.h q38_gdn_ref.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m3_multigdn.cu \
		q38_gdn.o q38_gdn_ref.o q38_cuda_primitives.o \
		$(CUDA_LDLIBS) -lm

m3-c10: m3-c09 $(TEST_DIR)/test_m3_multigdn
	@mkdir -p $(M3_ARTIFACT_DIR)
	./$(TEST_DIR)/test_m3_multigdn \
		$(M3_ARTIFACT_DIR)/multigdn_intermediates.json

$(TEST_DIR)/test_m3_chunk_invariance: \
		$(TEST_DIR)/test_m3_chunk_invariance.cu q38_gdn.o q38_gdn_ref.o \
		q38_cuda_primitives.o q38_gdn.h q38_gdn_ref.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ \
		$(TEST_DIR)/test_m3_chunk_invariance.cu q38_gdn.o q38_gdn_ref.o \
		q38_cuda_primitives.o $(CUDA_LDLIBS) -lm

m3-c11: m3-c10 $(TEST_DIR)/test_m3_chunk_invariance
	@mkdir -p $(M3_ARTIFACT_DIR)
	./$(TEST_DIR)/test_m3_chunk_invariance \
		$(M3_ARTIFACT_DIR)/chunk_invariance.json

$(TEST_DIR)/test_m3_cuda_profile_baseline: \
		$(TEST_DIR)/test_m3_cuda_profile_baseline.cu q38_gdn.o \
		q38_cuda_primitives.o q38_state.o q38_gdn.h q38_state.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ \
		$(TEST_DIR)/test_m3_cuda_profile_baseline.cu q38_gdn.o \
		q38_cuda_primitives.o q38_state.o $(CUDA_LDLIBS) -lm

m3-c12: m3-c11 $(TEST_DIR)/test_m3_cuda_profile_baseline
	@mkdir -p $(M3_ARTIFACT_DIR)
	./$(TEST_DIR)/test_m3_cuda_profile_baseline \
		$(M3_ARTIFACT_DIR)/cuda_profile_baseline.json

$(TEST_DIR)/test_m3_gdn_fused: \
		$(TEST_DIR)/test_m3_gdn_fused.cu q38_gdn.o q38_gdn_ref.o \
		q38_cuda_primitives.o q38_gdn.h q38_gdn_ref.h q38_state.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ \
		$(TEST_DIR)/test_m3_gdn_fused.cu q38_gdn.o q38_gdn_ref.o \
		q38_cuda_primitives.o $(CUDA_LDLIBS) -lm

m3-c13: m3-c12 m3-audit $(TEST_DIR)/test_m3_gdn_fused
	@mkdir -p $(M3_ARTIFACT_DIR)
	./$(TEST_DIR)/test_m3_gdn_fused \
		$(M3_ARTIFACT_DIR)/cuda_profile_fused.json

m3-audit: tokenizer-runtime-gate $(TEST_DIR)/test_m3_gr_ref $(TEST_DIR)/test_m3_gr_cuda \
		$(TEST_DIR)/test_m3_state $(TEST_DIR)/test_weights
	@test -f $(M1_ARTIFACT_DIR)/qwen38-runtime-only-layers0-3-Q2Experts-BF16Core-BF16PLE.gguf
	./$(TEST_DIR)/test_m3_gr_ref
	./$(TEST_DIR)/test_m3_gr_cuda
	./$(TEST_DIR)/test_m3_state
	@for layer in 0 1 2 3; do \
		./$(TEST_DIR)/test_weights \
			$(M1_ARTIFACT_DIR)/qwen38-runtime-only-layers0-3-Q2Experts-BF16Core-BF16PLE.gguf \
			$$layer; \
	done
	@mkdir -p $(M3_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M3-AUDIT","gr":"external non-zero scalar/CUDA goldens with negative SiLU and hc_count scaling","state":"36 independent GDN slots with explicit 48-layer mapping; GR activation excluded from persistent bytes","ple":{"zero_based_layer":1,"counts":{"max_layer_0":29,"max_layer_1":190,"max_layer_2":214,"max_layer_3":238},"finding":"no off-by-one; frozen fixture places PLE at layers.1"},"tokenizer":"native C byte-level BPE, decode, special tokens, multimodal markers, and frozen chat-template vectors; Python oracle is test-only","status":"pass"}' \
		> $(M3_ARTIFACT_DIR)/audit.json

m3-acceptance: m3-c13
	@python3 -c 'import json, pathlib; p=pathlib.Path("$(M3_ARTIFACT_DIR)"); names=("audit.json","gr_golden.json","gr_cuda_accuracy.json","state_layout.json","state_memory.json","gdn_microtests.json","gdn_projection_conv.json","gdn_recurrence_cuda.json","layer0_intermediates.json","multigdn_intermediates.json","chunk_invariance.json","cuda_profile_baseline.json","cuda_profile_fused.json"); bad=[n for n in names if json.loads((p/n).read_text()).get("status") != "pass"]; assert not bad, "M3 acceptance failed: "+", ".join(bad)'
	@mkdir -p $(M3_ARTIFACT_DIR)
	@printf '%s\n' \
		'{"gate":"M3-C14","gates":["M3-AUDIT","M3-C00","M3-C01","M3-C02","M3-C03","M3-C04","M3-C05","M3-C06","M3-C07","M3-C08","M3-C09","M3-C10","M3-C11","M3-C12","M3-C13"],"status":"pass","physical_layout":"reference/simple logical-contiguous path; no GB10 layout optimization"}' \
		> $(M3_ARTIFACT_DIR)/acceptance.txt

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

m2-c02: m2-c01 $(TEST_DIR)/test_m2_tokenizer
	@mkdir -p $(M2_ARTIFACT_DIR)
	python3 tools/generate_m2_tokenizer_vectors.py \
		--model-dir $(MODEL_DIR) \
		--output $(M2_ARTIFACT_DIR)/tokenizer_vectors.json
	./$(TEST_DIR)/test_m2_tokenizer
	@test "$$(python3 -c 'import json; print(len(json.load(open("$(M2_ARTIFACT_DIR)/tokenizer_vectors.json"))["cases"]))')" -eq 10

m2-c03: m2-c02 $(TEST_DIR)/test_m2_quant
	./$(TEST_DIR)/test_m2_quant
	@mkdir -p $(M2_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M2-C03","type":"Q2_K","block_bytes":84,"elements_per_block":256,"oracle":"scalar F32","status":"pass"}' \
		> $(M2_ARTIFACT_DIR)/quant_q2_oracle.json
	printf '%s\n' \
		'{"gate":"M2-C03","type":"Q4_K","block_bytes":144,"elements_per_block":256,"oracle":"scalar F32","status":"pass"}' \
		> $(M2_ARTIFACT_DIR)/quant_q4_oracle.json

m2-c04: m2-c03 $(TEST_DIR)/test_m2_cuda
	./$(TEST_DIR)/test_m2_cuda
	@mkdir -p $(M2_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M2-C04","paths":["Q2_K","Q4_K"],"comparison":"CUDA vs scalar F32","status":"pass"}' \
		> $(M2_ARTIFACT_DIR)/primitive_accuracy.json

m2-c05: m2-c04 $(TEST_DIR)/test_m2_norm
	./$(TEST_DIR)/test_m2_norm
	@mkdir -p $(M2_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M2-C05","paths":["RMSNorm","SiLU"],"accumulation":"F32","comparison":"CUDA vs scalar oracle","status":"pass"}' \
		> $(M2_ARTIFACT_DIR)/norm_activation.json

m2-c06: m2-c05 $(TEST_DIR)/test_m2_matvec
	./$(TEST_DIR)/test_m2_matvec
	@mkdir -p $(M2_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M2-C06","paths":["Q2_K expert","BF16 core"],"shapes":"random rows and non-multiple BF16 tails","status":"pass"}' \
		> $(M2_ARTIFACT_DIR)/matvec_dispatch.json

m2-c07: m2-c06 $(TEST_DIR)/test_m2_embedding
	@test -f $(M1_ARTIFACT_DIR)/runtime-layers0-q2-test.gguf || \
		{ echo "M2-C07: required M1 subset artifact is unavailable" >&2; exit 1; }
	@mkdir -p $(M2_ARTIFACT_DIR)
	./$(TEST_DIR)/test_m2_embedding \
		$(M1_ARTIFACT_DIR)/runtime-layers0-q2-test.gguf \
		> $(M2_ARTIFACT_DIR)/embedding_probe.json
	python3 tools/validate_m2_embedding.py \
		--model-dir $(MODEL_DIR) \
		--probe $(M2_ARTIFACT_DIR)/embedding_probe.json

m2-c08: m2-c07 $(TEST_DIR)/test_m2_lm_head
	@test -f $(M1_ARTIFACT_DIR)/runtime-layers0-q2-test.gguf || \
		{ echo "M2-C08: required M1 subset artifact is unavailable" >&2; exit 1; }
	@mkdir -p $(M2_ARTIFACT_DIR)
	./$(TEST_DIR)/test_m2_lm_head \
		$(M1_ARTIFACT_DIR)/runtime-layers0-q2-test.gguf \
		> $(M2_ARTIFACT_DIR)/lm_head_probe.json
	python3 tools/validate_m2_lm_head.py \
		--model-dir $(MODEL_DIR) \
		--probe $(M2_ARTIFACT_DIR)/lm_head_probe.json

m2-c09: m2-c08 $(TEST_DIR)/test_weights
	@test -f $(M1_ARTIFACT_DIR)/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf || \
		{ echo "M2-C09: required M1 full artifact is unavailable" >&2; exit 1; }
	./$(TEST_DIR)/test_weights \
		$(M1_ARTIFACT_DIR)/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf 47
	@mkdir -p $(M2_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M2-C09","binding":"strict","layers":48,"runtime_tensor_count":1294,"ple":"handle-backed views","status":"pass"}' \
		> $(M2_ARTIFACT_DIR)/full_binding_report.json

m2-c10: m2-c09 $(TEST_DIR)/test_m2_memory
	@test -f $(M1_ARTIFACT_DIR)/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf || \
		{ echo "M2-C10: required M1 full artifact is unavailable" >&2; exit 1; }
	@mkdir -p $(M2_ARTIFACT_DIR)
	./$(TEST_DIR)/test_m2_memory \
		$(M1_ARTIFACT_DIR)/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		> $(M2_ARTIFACT_DIR)/memory.json

m2-c11: m2-c10 test
	python3 tools/m2_acceptance.py \
		--artifact-dir $(M2_ARTIFACT_DIR) \
		--model-dir $(MODEL_DIR)

m2-acceptance: m2-c11

m3-c00: m2-acceptance
	@mkdir -p $(M3_ARTIFACT_DIR)
	cp docs/qwen_gdn_semantics.md $(M3_ARTIFACT_DIR)/qwen_gdn_semantics.md
	python3 tools/m3_c00_audit.py \
		--doc docs/qwen_gdn_semantics.md \
		--model-dir $(MODEL_DIR) \
		--output $(M3_ARTIFACT_DIR)/reference_audit.json

m3-c01: m3-c00 $(TEST_DIR)/test_m3_gr_binding
	@test -f $(M1_ARTIFACT_DIR)/runtime-layers0-q2-test.gguf || \
		{ echo "M3-C01: required M1 subset artifact is unavailable" >&2; exit 1; }
	./$(TEST_DIR)/test_m3_gr_binding \
		$(M1_ARTIFACT_DIR)/runtime-layers0-q2-test.gguf
	@mkdir -p $(M3_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M3-C01","families":["attn_hyper_connection","mlp_hyper_connection"],"rank":320,"branches":4,"status":"pass"}' \
		> $(M3_ARTIFACT_DIR)/gr_binding.json

m3-c02: m3-c01 $(TEST_DIR)/test_m3_gr_ref
	./$(TEST_DIR)/test_m3_gr_ref
	@mkdir -p $(M3_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M3-C02","formula":"SiLU(low_rank/hc_count), 2*sigmoid(inject/hc_count)","dtype":"F32","vectors":"zero smoke plus independent non-zero negative-SiLU/scaling goldens","status":"pass"}' \
		> $(M3_ARTIFACT_DIR)/gr_golden.json

m3-c03: m3-c02 $(TEST_DIR)/test_m3_gr_cuda
	./$(TEST_DIR)/test_m3_gr_cuda
	@mkdir -p $(M3_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M3-C03","kernel":"non-fused GR CUDA baseline","comparison":"CUDA vs independent external F32 goldens","status":"pass"}' \
		> $(M3_ARTIFACT_DIR)/gr_cuda_accuracy.json

m3-c04: m3-c03 $(TEST_DIR)/test_m3_gdn_binding
	@test -f $(M1_ARTIFACT_DIR)/runtime-layers0-q2-test.gguf || \
		{ echo "M3-C04: required M1 subset artifact is unavailable" >&2; exit 1; }
	./$(TEST_DIR)/test_m3_gdn_binding \
		$(M1_ARTIFACT_DIR)/runtime-layers0-q2-test.gguf
	@mkdir -p $(M3_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M3-C04","family":"GDN","key_heads":16,"value_heads":48,"head_dim":128,"conv_kernel":4,"status":"pass"}' \
		> $(M3_ARTIFACT_DIR)/gdn_binding.json

m3-c05: m3-c04 $(TEST_DIR)/test_m3_state
	./$(TEST_DIR)/test_m3_state
	@mkdir -p $(M3_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M3-C05","scope":"48-layer model","gdn_layers":36,"full_attention_layers":[3,7,11,15,19,23,27,31,35,39,43,47],"layer_to_gdn_slot":"explicit pointer-free table","recurrent_state":{"logical_shape":["gdn_slot",1,48,128,128],"dtype":"F32","bytes_per_slot":3145728,"bytes":113246208},"conv_history":{"logical_shape":["gdn_slot",1,3,10240],"kernel":4,"bytes_per_slot":122880,"bytes":4423680},"gr_workspace":{"logical_shape":[1,4,2560],"dtype":"F32","bytes":40960},"persistent_excludes_gr":true,"physical_layout":"contiguous per-slot regions; no GB10 tiling","status":"pass"}' \
		> $(M3_ARTIFACT_DIR)/state_layout.json
	printf '%s\n' \
		'{"gate":"M3-C05","persistent_recurrent_state_bytes":113246208,"conv_history_bytes":4423680,"gr_workspace_bytes":40960,"workspace_bytes":0,"persistent_bytes":117669888,"activation_bytes":40960,"allocation_bytes":117710848,"status":"pass"}' \
		> $(M3_ARTIFACT_DIR)/state_memory.json

m3-c06: m3-c05 $(TEST_DIR)/test_m3_gdn_ref
	./$(TEST_DIR)/test_m3_gdn_ref
	@mkdir -p $(M3_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M3-C06","dtype":"F32","state_logical_shape":["sequence","value_head","row","column"],"value_heads":48,"head_dim":128,"equations":["Sbar=decay*S_prev","prediction=Sbar^T*k","delta=(v-prediction)*beta","S_next=Sbar+k*delta^T","y=scale*S_next^T*q"],"ordering":["decay","prediction","delta","outer_product_update","read","scale"],"microtests":["zero_state","beta_zero","decay_ordering","exact_prediction","basis_orientation","two_timesteps","head_mapping"],"physical_layout":"contiguous logical state; no GB10 optimization or packing","status":"pass"}' \
		> $(M3_ARTIFACT_DIR)/gdn_microtests.json

m3-c07: m3-c06 $(TEST_DIR)/test_m3_gdn_cuda
	./$(TEST_DIR)/test_m3_gdn_cuda
	@mkdir -p $(M3_ARTIFACT_DIR)
	printf '%s\n' \
		'{"gate":"M3-C07","kernel":"reference/simple CUDA GDN projection plus causal depthwise convolution","scope":"deterministic device-resident fixture; full model upload remains out of scope","projection_dispatch":["Q2_K","Q8_0","BF16","F32"],"projection_logical_shapes":{"input":["token",2560],"qkv":["token",10240],"z":["token",6144]},"qkv_slice_order":["Q[0:2048]","K[2048:4096]","V[4096:10240]"],"conv":{"logical_input":["token",10240],"logical_kernel":["tap",10240],"kernel":4,"history":["token",3,10240],"history_update":"last three concatenated input tokens"},"ordering":["projection","causal_convolution","history_update","SiLU"],"physical_layout":"explicit logical row-major API; no GB10 optimization or fusion","status":"pass"}' \
		> $(M3_ARTIFACT_DIR)/gdn_projection_conv.json

clean:
	rm -f q38 *.o $(TEST_DIR)/test_platform $(TEST_DIR)/test_gguf \
		$(TEST_DIR)/test_memory $(TEST_DIR)/test_model_config \
		$(TEST_DIR)/test_weights $(TEST_DIR)/test_quant_blocks \
		$(TEST_DIR)/test_m2_golden \
		$(TEST_DIR)/test_m2_weights \
		$(TEST_DIR)/test_m2_tokenizer \
		$(TEST_DIR)/test_m2_quant \
		$(TEST_DIR)/test_m2_cuda \
		$(TEST_DIR)/test_m2_norm \
		$(TEST_DIR)/test_m2_matvec \
		$(TEST_DIR)/test_m2_embedding \
		$(TEST_DIR)/test_m2_lm_head \
		$(TEST_DIR)/test_m2_memory \
		$(TEST_DIR)/test_m3_gr_binding \
		$(TEST_DIR)/test_m3_gr_ref \
		$(TEST_DIR)/test_m3_gr_cuda \
		$(TEST_DIR)/test_m3_gdn_binding \
		$(TEST_DIR)/test_m3_state \
		$(TEST_DIR)/test_m3_gdn_ref \
		$(TEST_DIR)/test_m3_gdn_cuda \
		$(TEST_DIR)/test_m3_gdn_recurrence_cuda \
		$(TEST_DIR)/test_m3_forward_probe \
		$(TEST_DIR)/test_m3_multigdn \
		$(TEST_DIR)/test_m3_chunk_invariance \
		$(TEST_DIR)/test_m3_cuda_profile_baseline \
		$(TEST_DIR)/test_m3_gdn_fused \
		$(TEST_DIR)/test_m4_session \
		$(TEST_DIR)/test_m4_ple_ref \
		$(TEST_DIR)/test_m4_ple_store \
		$(TEST_DIR)/test_m4_ple_row \
		$(TEST_DIR)/test_m4_ple_cuda \
		$(TEST_DIR)/q38_forward_probe \
		$(TEST_DIR)/test_ple_chunking \
		$(TEST_DIR)/test_m4_ple_cache \
		tools/q38_quantize
	rm -rf $(ARTIFACT_DIR) $(M2_ARTIFACT_DIR) $(M3_ARTIFACT_DIR)
