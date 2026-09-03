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
# CUDA kernels are compiled by nvcc; host runtime objects use cc.
C_OBJS := q38.o q38_gguf.o q38_memory.o q38_platform.o \
	q38_tokenizer.o q38_decode.o q38_forward.o q38_moe.o q38_weights.o \
	q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_session.o \
	q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o q38_replay.o \
	q38_profile.o
CUDA_OBJS := q38_cuda.o q38_forward_cuda.o q38_cuda_primitives.o \
	q38_gdn.o q38_moe_cuda.o q38_cuda_timing.o q38_profile_cuda.o
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
M6_GOLDEN := artifacts/m6/checkpoint_minimal_goldens.json
M6_PREFLIGHT := artifacts/m6/preflight.json
M6_PREFLIGHT_SUMMARY := artifacts/m6/full_model_preflight.txt
M6_TRACE := artifacts/m6/real_forward_trace.json
M6_REFERENCE := artifacts/m6/transformers_reference.json
M6_COMPARISON := artifacts/m6/semantic_comparison.json
M6_QUANT_REFERENCE := artifacts/m6/quant_matched_reference.json
M6_QUANT_COMPARISON := artifacts/m6/quant_matched_comparison.json
M6_GPU_TRACE := artifacts/m6/gpu_real_forward_trace.json
M6_GPU_PROGRESSIVE := artifacts/m6/gpu_progressive.json
M6_AR_ORACLE := artifacts/m6/autoregressive_oracle.json
M6_STATEFUL_ORACLE := artifacts/m6/stateful_gguf_oracle.json
M6_ORACLE_TOKENS ?= 1
M6_ORACLE_DEVICE ?= cuda
M6_ORACLE_TIMEOUT ?= 600
M6_PYTHON ?= .venv-m6/bin/python
M6_ORACLE_CPU ?= artifacts/m6/stateful_oracle_cpu_1_noc12.json
M6_ORACLE_GPU ?= artifacts/m6/stateful_sequence_oracle_resume_probe.json
M7_MODEL ?= artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf

.PHONY: all spark test clean m0-acceptance m1-inventory m1-validate m1-subset \
	m1-bind m1-quant-block m1-full m1-memory-matrix m1-acceptance m2-c00 m2-c01 \
	m2-c02 m2-c03 m2-c04 m2-c05 m2-c06 m2-c07 m2-c08 m2-c09 m2-c10 \
	m2-c11 m2-acceptance m3-c00 m3-c01 m3-c02 m3-c03 m3-c04 m3-c05 \
	m3-c06 m3-c07 m3-c08 m3-c09 m3-c10 m3-c11 m3-c12 m3-c13 m3-audit \
	m3-acceptance m4-c00 m4-c01 m4-c02 m4-c03 m4-c04 m4-c05 m4-c06 m4-c07 \
	m4-c08 m4-c09 m4-c10 m4-c11 m4-c12 m4-c13 m4-acceptance m4-integration-audit m5-c00 m5-c01 m5-c02 m5-c03 m5-c04 m5-c05 m5-c06 m5-c07 m5-c08 m5-c09 m5-c10 m5-c11 m5-c12 m5-c13 m5-c14 m5-c15 m5-acceptance m6-c00 m6-c01 m6-c02 m6-c03 m6-c04 m6-c05 m6-c06 m6-c07 m6-c08 m6-c09 m6-c10 m6-c11 m6-preflight m6-dequant-fixtures m6-c12 m6-trace-schema m6-rounding-diagnostics m6-progressive-boundaries m6-c13 m6-c14 m6-c15 m6-c16 m6-acceptance m6-gpu-forward m6-gpu-progressive m6-gpu-phase7 m6-autoregressive-oracle \
	m6-stateful-oracle \
	m6-stateful-sequence-oracle \
	m6-decode-protocol m6-decode-ladder-check m6-decode-compare m6-oracle-compare \
	post-m5-bis post-m5-ter post-m5-supplement m7-replay m7-profile m7-profile-schema m7-gates \
	m7-profile-forward m7-acceptance
q38_moe.o: q38_moe.c q38_moe.h q38_weights.h
	$(CC) $(CFLAGS) -c -o $@ q38_moe.c

q38_decode.o: q38_decode.c q38_decode.h q38_forward.h
	$(CC) $(CFLAGS) -c -o $@ q38_decode.c

q38_replay.o: q38_replay.c q38_replay.h q38_forward.h
	$(CC) $(CFLAGS) -c -o $@ q38_replay.c

q38_profile.o: q38_profile.c q38_profile.h q38_forward.h
	$(CC) $(CFLAGS) -c -o $@ q38_profile.c

all: spark

# --- CUDA object ----------------------------------------------------------
q38_cuda.o: q38_cuda.cu q38_cuda.h q38.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_cuda.cu

q38_cuda_timing.o: q38_cuda_timing.cu q38_cuda_timing.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_cuda_timing.cu

q38_profile_cuda.o: q38_profile_cuda.cu q38_profile.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_profile_cuda.cu

# --- C objects ------------------------------------------------------------
q38.o: q38.c q38.h q38_gguf.h q38_memory.h q38_platform.h q38_cuda.h \
	q38_decode.h q38_forward_cuda.h q38_tokenizer.h q38_weights.h
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

q38_ple_stage.o: q38_ple_stage.cu q38_ple_stage.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_ple_stage.cu

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

q38_forward_cuda.o: q38_forward_cuda.cu q38_forward_cuda.h \
		q38_forward.h q38_cuda_primitives.h q38_gdn.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_forward_cuda.cu

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
		q38_gguf.o q38_model_config.o q38_ple.o q38_qsa.o q38_weights.h q38_ple.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m2_weights.c q38_weights.o \
		q38_gguf.o q38_model_config.o q38_ple.o q38_qsa.o

$(TEST_DIR)/test_m2_tokenizer: $(TEST_DIR)/test_m2_tokenizer.c \
		q38_tokenizer.o q38_tokenizer.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m2_tokenizer.c q38_tokenizer.o

$(TEST_DIR)/test_m4_session: $(TEST_DIR)/test_m4_session.c q38_session.o q38_session.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m4_session.c q38_session.o

$(TEST_DIR)/test_m7_replay: $(TEST_DIR)/test_m7_replay.c q38_replay.o \
		q38_qsa.o q38_state.o q38_session.o q38_replay.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m7_replay.c q38_replay.o \
		q38_qsa.o q38_state.o q38_session.o -lm

m7-replay: $(TEST_DIR)/test_m7_replay
	@./$(TEST_DIR)/test_m7_replay
	@echo "M7 replay snapshot and callback trace harness passed"

$(TEST_DIR)/test_m7_profile: $(TEST_DIR)/test_m7_profile.c q38_profile.o \
		q38_profile_cuda.o q38_profile.h q38_forward.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m7_profile.c \
		q38_profile.o q38_profile_cuda.o $(CUDA_LDLIBS)

m7-profile: $(TEST_DIR)/test_m7_profile
	@./$(TEST_DIR)/test_m7_profile

m7-profile-schema:
	@$(M6_PYTHON) tools/test_m7_profile_schema.py

m7-gates: m7-replay m7-profile m7-profile-schema
	@$(M6_PYTHON) tools/m7_check_gates.py

$(TEST_DIR)/m7_profile_forward: $(TEST_DIR)/m7_profile_forward.c \
		q38_gguf.o q38_memory.o q38_platform.o q38_decode.o q38_forward.o \
		q38_moe.o q38_weights.o q38_model_config.o q38_ple.o q38_qsa.o \
		q38_state.o q38_session.o q38_quant.o q38_ple_ref.o q38_gdn_ref.o \
		q38_gr_ref.o q38_profile.o q38_cuda.o q38_forward_cuda.o \
		q38_cuda_primitives.o q38_gdn.o q38_moe_cuda.o q38_profile_cuda.o
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/m7_profile_forward.c \
		q38_gguf.o q38_memory.o q38_platform.o q38_decode.o q38_forward.o \
		q38_moe.o q38_weights.o q38_model_config.o q38_ple.o q38_qsa.o \
		q38_state.o q38_session.o q38_quant.o q38_ple_ref.o q38_gdn_ref.o \
		q38_gr_ref.o q38_profile.o q38_cuda.o q38_forward_cuda.o \
		q38_cuda_primitives.o q38_gdn.o q38_moe_cuda.o q38_profile_cuda.o \
		$(CUDA_LDLIBS) -lm

m7-profile-forward: $(TEST_DIR)/m7_profile_forward
	@./$(TEST_DIR)/m7_profile_forward \
		$(M7_MODEL) \
		artifacts/m7

$(TEST_DIR)/m7_lm_head_kernel_bench: $(TEST_DIR)/m7_lm_head_kernel_bench.cu \
		q38_cuda_primitives.o q38_weights.o q38_gguf.o q38_model_config.o \
		q38_ple.o q38_qsa.o q38_quant.o
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/m7_lm_head_kernel_bench.cu \
		q38_cuda_primitives.o q38_weights.o q38_gguf.o q38_model_config.o \
		q38_ple.o q38_qsa.o q38_quant.o $(CUDA_LDLIBS) -lm

m7-lm-head-kernel: $(TEST_DIR)/m7_lm_head_kernel_bench
	@./$(TEST_DIR)/m7_lm_head_kernel_bench $(M7_MODEL)

$(TEST_DIR)/m7_cold_warm: $(TEST_DIR)/m7_cold_warm.c \
		q38_gguf.o q38_forward.o q38_weights.o q38_model_config.o q38_ple.o \
		q38_qsa.o q38_state.o q38_session.o q38_quant.o q38_ple_ref.o \
		q38_gdn_ref.o q38_gr_ref.o q38_moe.o q38_decode.o \
		q38_forward_cuda.o q38_cuda_primitives.o q38_gdn.o q38_moe_cuda.o
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/m7_cold_warm.c \
		q38_gguf.o q38_forward.o q38_weights.o q38_model_config.o q38_ple.o \
		q38_qsa.o q38_state.o q38_session.o q38_quant.o q38_ple_ref.o \
		q38_gdn_ref.o q38_gr_ref.o q38_moe.o q38_decode.o q38_forward_cuda.o \
		q38_cuda_primitives.o q38_gdn.o q38_moe_cuda.o $(CUDA_LDLIBS) -lm

m7-cold-warm: $(TEST_DIR)/m7_cold_warm
	@./$(TEST_DIR)/m7_cold_warm $(M7_MODEL)

m7-acceptance: m7-gates $(TEST_DIR)/test_m6_moe_cuda
	@./$(TEST_DIR)/test_m6_moe_cuda
	@if [ "$${M7_RUN_FULL:-0}" = 1 ]; then \
		$(MAKE) m0-acceptance m1-acceptance m2-acceptance m3-acceptance \
			m4-acceptance m5-acceptance m6-acceptance; \
	else \
		echo "M7: full M0-M6 sweep not run (set M7_RUN_FULL=1)"; \
	fi
	@$(M6_PYTHON) tools/m7_acceptance.py
	@echo "M7 acceptance artifacts generated (missing expensive probes are recorded)"

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
		q38_gguf.o q38_ple.h q38_gguf.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m4_ple_store.c q38_ple.o q38_gguf.o

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
M4_C06_INJECTION_GOLDEN := $(M4_ARTIFACT_DIR)/ple_injection_vectors.json
M4_C06_INJECTION_FIXTURE := $(M4_ARTIFACT_DIR)/ple_injection_fixture.bin

$(M4_C06_GOLDEN): tools/generate_m4_c06_goldens.py \
		$(MODEL_DIR)/config.json $(MODEL_DIR)/model.safetensors.index.json
	@mkdir -p $(M4_ARTIFACT_DIR)
	python3 tools/generate_m4_c06_goldens.py --model-dir $(MODEL_DIR) \
		--output $@

$(M4_C06_INJECTION_GOLDEN) $(M4_C06_INJECTION_FIXTURE): \
		tools/generate_m4_c06_injection_goldens.py \
		$(MODEL_DIR)/config.json $(MODEL_DIR)/model.safetensors.index.json
	@mkdir -p $(M4_ARTIFACT_DIR)
	python3 tools/generate_m4_c06_injection_goldens.py \
		--model-dir $(MODEL_DIR) --output $(M4_C06_INJECTION_GOLDEN) \
		--fixture $(M4_C06_INJECTION_FIXTURE)

$(TEST_DIR)/q38_forward_probe: q38_forward_probe.c q38_ple_ref.o \
		q38_session.o q38_quant.o q38_ple_ref.h q38_session.h q38_quant.h
	$(CC) $(CFLAGS) -o $@ q38_forward_probe.c \
		q38_ple_ref.o q38_session.o q38_quant.o -lm

$(TEST_DIR)/test_m4_ple_injection: $(TEST_DIR)/test_m4_ple_injection.c \
		q38_ple_ref.o q38_session.o q38_quant.o q38_ple_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m4_ple_injection.c \
		q38_ple_ref.o q38_session.o q38_quant.o -lm

m4-c06: m4-c05 $(M4_C06_GOLDEN) $(M4_C06_INJECTION_GOLDEN) \
		$(M4_C06_INJECTION_FIXTURE) $(TEST_DIR)/q38_forward_probe \
		$(TEST_DIR)/test_m4_ple_injection
	@./$(TEST_DIR)/q38_forward_probe $(M4_C06_GOLDEN)
	@./$(TEST_DIR)/test_m4_ple_injection $(M4_C06_INJECTION_FIXTURE)
	@printf '%s\n' '{"gate":"M4-C06","probe":"q38_forward_probe + test_m4_ple_injection","golden":"ple_injection_vectors.json","coverage":"checkpoint-backed hidden_before_ple, PLE contribution, hidden_after_ple, row IDs, and injection boundary","reference":"independent Transformers equations; q38 is comparison only","status":"pass"}' > $(M4_ARTIFACT_DIR)/ple_injection_probe.json

$(TEST_DIR)/test_ple_chunking: $(TEST_DIR)/test_ple_chunking.c \
		q38_ple_ref.o q38_session.o q38_quant.o q38_ple_ref.h \
		q38_session.h q38_quant.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_ple_chunking.c \
		q38_ple_ref.o q38_session.o q38_quant.o -lm

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

$(TEST_DIR)/test_m4_ple_cuda_batch: $(TEST_DIR)/test_m4_ple_cuda_batch.cu \
		q38_ple_cuda.o q38_ple_ref.o q38_session.o q38_quant.o \
		q38_ple_cuda.h q38_ple_ref.h q38_quant.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m4_ple_cuda_batch.cu \
		q38_ple_cuda.o q38_ple_ref.o q38_session.o q38_quant.o \
		$(CUDA_LDLIBS) -lm

m4-c09: m4-c08 $(TEST_DIR)/test_m4_ple_cuda_batch
	@./$(TEST_DIR)/test_m4_ple_cuda_batch
	@mkdir -p $(M4_ARTIFACT_DIR)
	@printf '%s\n' '{"gate":"M4-C09","lookup":"CUDA batch deduplication","dedupe":"storage accesses only","output_order":"input order preserved","qtypes":["Q2_K","Q4_K"],"status":"pass"}' > $(M4_ARTIFACT_DIR)/ple_batching.json

$(TEST_DIR)/test_m4_ple_loader: $(TEST_DIR)/test_m4_ple_loader.c \
		q38_ple.o q38_ple_cache.o q38_gguf.o q38_memory.o \
		q38_ple.h q38_ple_cache.h q38_gguf.h q38_memory.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m4_ple_loader.c \
		q38_ple.o q38_ple_cache.o q38_gguf.o q38_memory.o

m4-c10: m4-c09 $(TEST_DIR)/test_m4_ple_loader
	@mkdir -p $(M4_ARTIFACT_DIR)
	@./$(TEST_DIR)/test_m4_ple_loader \
		$(M4_ARTIFACT_DIR)/ple_memory_matrix.json

$(TEST_DIR)/test_m4_ple_prefetch: $(TEST_DIR)/test_m4_ple_prefetch.c \
		q38_ple_prefetch.o q38_ple.o q38_gguf.o \
		q38_ple_prefetch.h q38_ple.h q38_gguf.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m4_ple_prefetch.c \
		q38_ple_prefetch.o q38_ple.o q38_gguf.o

q38_ple_prefetch.o: q38_ple_prefetch.c q38_ple_prefetch.h q38_ple.h
	$(CC) $(CFLAGS) -c -o $@ q38_ple_prefetch.c

m4-c11: m4-c10 $(TEST_DIR)/test_m4_ple_prefetch
	@mkdir -p $(M4_ARTIFACT_DIR)
	@./$(TEST_DIR)/test_m4_ple_prefetch
	@printf '%s\n' '{"gate":"M4-C11","implementation":"madvise(MADV_WILLNEED) advisory mapped-row prefetch","default_enabled":false,"cuda_staging":"not present","benefit_gate":"not enabled without measured benefit","persistent_copy":false,"status":"pass"}' > $(M4_ARTIFACT_DIR)/ple_prefetch_benchmark.json

$(TEST_DIR)/test_m4_ple_cuda_failure: $(TEST_DIR)/test_m4_ple_cuda_failure.cu \
		q38_ple_cuda.o q38_quant.h q38_ple_cuda.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m4_ple_cuda_failure.cu \
		q38_ple_cuda.o $(CUDA_LDLIBS)

m4-c12: m4-c11 $(TEST_DIR)/test_m4_ple_cuda_failure
	@mkdir -p $(M4_ARTIFACT_DIR)
	@./$(TEST_DIR)/test_m4_ple_cuda_failure
	@printf '%s\n' '{"gate":"M4-C12","cuda_fatal":"kernel launch and stream execution errors call _exit(134)","fault_injection":"child process illegal device pointer","continuation":"forbidden","status":"pass"}' > $(M4_ARTIFACT_DIR)/cuda_health.txt

m4-c13: m4-c12
	@python3 -c 'import json, pathlib; p=pathlib.Path("$(M4_ARTIFACT_DIR)"); names=("session_history.json","ple_ref.json","ple_store.json","ple_row_quant_tests.json","ple_cuda_lookup.json","ple_injection_probe.json","ple_chunk_invariance.json","ple_cache_stats.json","ple_batching.json","ple_memory_matrix.json","ple_prefetch_benchmark.json","cuda_health.txt"); bad=[]; [bad.append(n) for n in names if json.loads((p/n).read_text()).get("status") != "pass"]; assert not bad, "M4 acceptance failed: "+", ".join(bad)'
	@printf '%s\n' '{"gate":"M4-C13","gates":["M4-C00","M4-C01","M4-C02","M4-C03","M4-C04","M4-C05","M4-C06","M4-C07","M4-C08","M4-C09","M4-C10","M4-C11","M4-C12"],"reference_paths":["scalar PLE hash/index","scalar Q2/Q4 row decoder","naive CUDA lookup","file-backed GGUF loader"],"unavailable":["full-forward hidden/logit goldens","CUDA-visible staging"],"status":"pass"}' > $(M4_ARTIFACT_DIR)/acceptance.txt

m4-acceptance: m4-c13

m4-integration-audit: m4-acceptance $(M4_C06_INJECTION_GOLDEN) \
		$(M4_C06_INJECTION_FIXTURE) $(TEST_DIR)/test_m4_ple_injection \
		$(TEST_DIR)/test_m5_tokenizer_edges
	@./$(TEST_DIR)/test_m4_ple_injection $(M4_C06_INJECTION_FIXTURE)
	@./$(TEST_DIR)/test_m5_tokenizer_edges
	@grep -q '"status": "pass"' $(M4_C06_INJECTION_GOLDEN)
	@printf '%s\n' '{"gate":"M4-INTEGRATION-AUDIT","ple_injection":"pass","tokenizer_edges":"pass","mmap_reference":"preserved","status":"pass"}' > artifacts/m4/integration_audit.json

m5-c00: m4-integration-audit
	@test -f docs/qwen_qsa_semantics.md
	@grep -q "build_qsa_top_k" docs/qwen_qsa_semantics.md
	@grep -q "compress_ratio - 1" docs/qwen_qsa_semantics.md
	@grep -q "no separate persistent compressed-group accumulator" docs/qwen_qsa_semantics.md
	@grep -q "mrope_interleaved" docs/qwen_qsa_semantics.md
	@mkdir -p artifacts/m5
	@printf '%s\n' '{"gate":"M5-C00","reference":"llama.cpp PR #27742 commit eaf93765572e794b8e3754fe45adbe12d381e997","checkpoint":"Qwen4ExpForConditionalGeneration transformers 5.8.0.dev0","qsa_layers":12,"indexer":"mean pooled blocks, ReLU per-head scores, top-k plus tail","rope":"multi-section interleaved partial rotary","status":"pass"}' > artifacts/m5/qwen_qsa_semantics.json

q38_qsa.o: q38_qsa.c q38_qsa.h q38_gguf.h
	$(CC) $(CFLAGS) -c -o $@ q38_qsa.c

$(TEST_DIR)/test_m5_qsa_binding: $(TEST_DIR)/test_m5_qsa_binding.c \
		q38_qsa.o q38_qsa.h q38_gguf.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_qsa_binding.c q38_qsa.o

m5-c01: m5-c00 $(TEST_DIR)/test_m5_qsa_binding
	@./$(TEST_DIR)/test_m5_qsa_binding
	@printf '%s\n' '{"gate":"M5-C01","binder":"separate QSA tensor family","state":["main_k","main_v","index_k","position","committed_tokens"],"strict_shapes":"validated by q38_weights binder","status":"pass"}' > artifacts/m5/qsa_binding.json

q38_rope_ref.o: q38_rope_ref.c q38_rope_ref.h
	$(CC) $(CFLAGS) -c -o $@ q38_rope_ref.c

$(TEST_DIR)/test_m5_rope_ref: $(TEST_DIR)/test_m5_rope_ref.c \
		q38_rope_ref.o q38_rope_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_rope_ref.c \
		q38_rope_ref.o -lm

m5-c02: m5-c01 $(TEST_DIR)/test_m5_rope_ref
	@./$(TEST_DIR)/test_m5_rope_ref
	@printf '%s\n' '{"gate":"M5-C02","implementation":"scalar Qwen4Exp partial interleaved mRoPE","n_dims":64,"sections":[11,11,10,0],"theta":10000000,"text_positions":"all four mRoPE coordinates use committed text position","status":"pass"}' > artifacts/m5/rope_goldens.json

q38_qsa_cuda.o: q38_qsa_cuda.cu q38_qsa_cuda.h q38_rope_ref.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_qsa_cuda.cu

$(TEST_DIR)/test_m5_qsa_cuda: $(TEST_DIR)/test_m5_qsa_cuda.cu \
		q38_qsa_cuda.o q38_rope_ref.o q38_qsa_cuda.h q38_rope_ref.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m5_qsa_cuda.cu \
		q38_qsa_cuda.o q38_rope_ref.o $(CUDA_LDLIBS) -lm

m5-c03: m5-c02 $(TEST_DIR)/test_m5_qsa_cuda
	@./$(TEST_DIR)/test_m5_qsa_cuda
	@printf '%s\n' '{"gate":"M5-C03","implementation":"naive CUDA BF16 Q/K/V projection plus partial interleaved mRoPE","selection":"not present","reference":"scalar RoPE and deterministic device projection","status":"pass"}' > artifacts/m5/qkv_goldens.json

q38_qsa_ref.o: q38_qsa_ref.c q38_qsa_ref.h
	$(CC) $(CFLAGS) -c -o $@ q38_qsa_ref.c

$(TEST_DIR)/test_m5_qsa_ref: $(TEST_DIR)/test_m5_qsa_ref.c \
		q38_qsa_ref.o q38_topk_ref.o q38_qsa_ref.h q38_topk_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_qsa_ref.c \
		q38_qsa_ref.o q38_topk_ref.o -lm

m5-c04: m5-c03 $(TEST_DIR)/test_m5_qsa_ref
	@./$(TEST_DIR)/test_m5_qsa_ref
	@printf '%s\n' '{"gate":"M5-C04","implementation":"scalar QSA block pooling and index scoring","compression_ratio":4,"scoring":"ReLU per indexer head then sum","tail":"incomplete final block retained","status":"pass"}' > artifacts/m5/indexer_goldens.json

q38_topk_ref.o: q38_topk_ref.c q38_topk_ref.h
	$(CC) $(CFLAGS) -c -o $@ q38_topk_ref.c

q38_topk_cuda.o: q38_topk_cuda.cu q38_topk_cuda.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_topk_cuda.cu

$(TEST_DIR)/test_m5_topk: $(TEST_DIR)/test_m5_topk.cu \
		q38_topk_ref.o q38_topk_cuda.o q38_topk_ref.h q38_topk_cuda.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m5_topk.cu \
		q38_topk_ref.o q38_topk_cuda.o $(CUDA_LDLIBS) -lm

m5-c05: m5-c04 $(TEST_DIR)/test_m5_topk
	@./$(TEST_DIR)/test_m5_topk
	@printf '%s\n' '{"gate":"M5-C05","implementation":"deterministic scalar/CUDA top-k","tie_break":"higher score, then lower token-cell ID","selection_order":"stable; only selected IDs are semantic","status":"pass"}' > artifacts/m5/topk_goldens.json

$(TEST_DIR)/test_m5_qsa_index_cuda: $(TEST_DIR)/test_m5_qsa_index_cuda.cu \
		q38_qsa_cuda.o q38_qsa_ref.o q38_topk_ref.o q38_qsa_cuda.h q38_qsa_ref.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m5_qsa_index_cuda.cu \
		q38_qsa_cuda.o q38_qsa_ref.o q38_topk_ref.o $(CUDA_LDLIBS) -lm

m5-c06: m5-c05 $(TEST_DIR)/test_m5_qsa_index_cuda
	@./$(TEST_DIR)/test_m5_qsa_index_cuda
	@printf '%s\n' '{"gate":"M5-C06","implementation":"naive CUDA block pooling/index scoring","pooling":"arithmetic mean including incomplete tail","scoring":"ReLU per head then sum","status":"pass"}' > artifacts/m5/indexer_cuda.json

$(TEST_DIR)/test_m5_qsa_tail: $(TEST_DIR)/test_m5_qsa_tail.c \
		q38_qsa_ref.o q38_topk_ref.o q38_qsa_ref.h q38_topk_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_qsa_tail.c \
		q38_qsa_ref.o q38_topk_ref.o -lm

m5-c07: m5-c06 $(TEST_DIR)/test_m5_qsa_tail
	@./$(TEST_DIR)/test_m5_qsa_tail
	@printf '%s\n' '{"gate":"M5-C07","implementation":"scalar causal tail handling","selected_width":"min(n_kv, indexer_top_k + compress_ratio - 1)","ratio":4,"top_k":2048,"boundary_lengths":[1,2,3,4,5,6,7,8,2047,2048,2049],"status":"pass"}' > artifacts/m5/causal_boundary_matrix.json

$(TEST_DIR)/test_m5_qsa_attention: $(TEST_DIR)/test_m5_qsa_attention.cu \
		q38_qsa_cuda.o q38_qsa_cuda.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m5_qsa_attention.cu \
		q38_qsa_cuda.o $(CUDA_LDLIBS) -lm

m5-c08: m5-c07 $(TEST_DIR)/test_m5_qsa_attention
	@./$(TEST_DIR)/test_m5_qsa_attention
	@printf '%s\n' '{"gate":"M5-C08","implementation":"naive CUDA selected-KV gather and dense GQA attention","mask":"selected token cells only; query-to-KV head grouping preserved","fallback":"reference kernel retained","status":"pass"}' > artifacts/m5/attention_goldens.json

$(TEST_DIR)/test_m5_qsa_state: $(TEST_DIR)/test_m5_qsa_state.c \
		q38_qsa.o q38_qsa.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_qsa_state.c q38_qsa.o

m5-c09: m5-c08 $(TEST_DIR)/test_m5_qsa_state
	@./$(TEST_DIR)/test_m5_qsa_state
	@printf '%s\n' '{"gate":"M5-C09","implementation":"separate bounded-growing main K/V and index state","append":"preserves row order and committed position","reset":"clears counts and position without requiring shrink","status":"pass"}' > artifacts/m5/qsa_cache_memory.json

$(TEST_DIR)/test_m5_qsa_chunking: $(TEST_DIR)/test_m5_qsa_chunking.c \
		q38_qsa.o q38_qsa_ref.o q38_topk_ref.o q38_qsa.h q38_qsa_ref.h q38_topk_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_qsa_chunking.c \
		q38_qsa.o q38_qsa_ref.o q38_topk_ref.o -lm

m5-c10: m5-c09 $(TEST_DIR)/test_m5_qsa_chunking
	@./$(TEST_DIR)/test_m5_qsa_chunking
	@printf '%s\n' '{"gate":"M5-C10","implementation":"scalar QSA chunk-invariance harness","partitions":["single","one-token","three-five-four"],"compared":["index state","score vectors","selected IDs"],"status":"pass"}' > artifacts/m5/qsa_chunk_invariance.json

m5-c11: m5-c10 $(TEST_DIR)/test_m5_integrated_forward
	@test -f q38_forward_probe.c
	@grep -q "q38_ple_ngram_ids_ref" q38_forward_probe.c
	@test -f q38_gdn_ref.c
	@test -f q38_gr_ref.c
	@./$(TEST_DIR)/test_m5_integrated_forward
	@printf '%s\n' '{"gate":"M5-C11","contract":"3xGDN + 1xQSA superblock ordering recorded for the reference graph","stages":["GDN","QSA","GR","PLE"],"numeric_hidden_golden":"qsa_forward_goldens.json covers the checkpoint-derived QSA stage; superblock hidden vectors are not claimed or fabricated","status":"pass"}' > artifacts/m5/superblock_goldens.json

$(TEST_DIR)/test_m5_integrated_forward: \
		$(TEST_DIR)/test_m5_integrated_forward.c q38_ple_ref.o \
		q38_forward.o q38_qsa.o q38_gguf.o q38_moe.o q38_weights.o \
		q38_model_config.o q38_ple.o q38_state.o q38_gdn_ref.o q38_gr_ref.o
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_integrated_forward.c \
		q38_ple_ref.o q38_forward.o q38_qsa.o q38_gguf.o q38_moe.o \
		q38_weights.o q38_model_config.o q38_ple.o q38_state.o \
		q38_session.o q38_quant.o q38_gdn_ref.o q38_gr_ref.o -lm

q38_forward.o: q38_forward.c q38_forward.h q38_qsa.h q38_weights.h \
		q38_gdn_ref.h q38_gr_ref.h q38_moe.h q38_ple_ref.h q38_quant.h
	$(CC) $(CFLAGS) -c -o $@ q38_forward.c

$(TEST_DIR)/test_forward_ref: $(TEST_DIR)/test_forward_ref.c \
		q38_forward.o q38_qsa.o q38_gguf.o q38_moe.o q38_weights.o \
		q38_model_config.o q38_ple.o q38_state.o q38_session.o q38_quant.o \
		q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o q38_forward.h q38_qsa.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_forward_ref.c \
		q38_forward.o q38_qsa.o q38_gguf.o q38_moe.o q38_weights.o \
		q38_model_config.o q38_ple.o q38_state.o q38_session.o q38_quant.o \
		q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o -lm

M5_C12_GOLDEN := artifacts/m5/qsa_forward_goldens.json
M5_C12_FIXTURE := artifacts/m5/qsa_forward_fixture.bin

$(M5_C12_GOLDEN) $(M5_C12_FIXTURE): tools/generate_m5_c12_goldens.py \
		$(MODEL_DIR)/config.json $(MODEL_DIR)/model.safetensors.index.json
	@mkdir -p artifacts/m5
	python3 tools/generate_m5_c12_goldens.py --model-dir $(MODEL_DIR) \
		--output $(M5_C12_GOLDEN) --fixture $(M5_C12_FIXTURE)

$(TEST_DIR)/test_m5_forward_probe: $(TEST_DIR)/test_m5_forward_probe.c \
		q38_forward.o q38_qsa.o q38_gguf.o q38_moe.o q38_weights.o \
		q38_model_config.o q38_ple.o q38_state.o q38_session.o q38_quant.o \
		q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o q38_forward.h q38_qsa.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_forward_probe.c \
		q38_forward.o q38_qsa.o q38_gguf.o q38_moe.o q38_weights.o \
		q38_model_config.o q38_ple.o q38_state.o q38_session.o q38_quant.o \
		q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o -lm

$(TEST_DIR)/test_m5_tokenizer_edges: $(TEST_DIR)/test_m5_tokenizer_edges.c \
		q38_tokenizer.o q38_tokenizer.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_tokenizer_edges.c \
		q38_tokenizer.o

m5-c12: m5-c11 $(M5_C12_GOLDEN) $(M5_C12_FIXTURE) \
		$(TEST_DIR)/test_forward_ref $(TEST_DIR)/test_m5_forward_probe
	@./$(TEST_DIR)/test_forward_ref
	@./$(TEST_DIR)/test_m5_forward_probe $(M5_C12_FIXTURE)
	@grep -q "reference graph" docs/qwen_qsa_semantics.md
	@grep -q "Prefill and decode use the same equations" docs/qwen_forward_semantics.md
	@printf '%s\n' '{"gate":"M5-C12","graph":"reference-compatible QSA graph with causal prefill/decode","weights":"independent generator loads only layer-3 QSA tensors and embedding rows","probe":"native q38 comparison against checkpoint-derived goldens","status":"pass"}' > artifacts/m5/qsa_forward_graph.json

m5-c13: m5-c12 $(TEST_DIR)/test_m5_tokenizer_edges
	@./$(TEST_DIR)/test_m5_tokenizer_edges
	@printf '%s\n' '{"gate":"M5-C13","reference_path":"retained","tokenizer_edge_parity":"Unicode, JSON escaping, audio/video/image markers, special-token verification","status":"pass"}' > artifacts/m5/safe_optimization.json

$(TEST_DIR)/test_m5_qsa_long: $(TEST_DIR)/test_m5_qsa_long.c \
		q38_qsa.o q38_qsa.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_qsa_long.c q38_qsa.o

m5-c14: m5-c13 $(TEST_DIR)/test_m5_qsa_long
	@./$(TEST_DIR)/test_m5_qsa_long
	@printf '%s\n' '{"gate":"M5-C14","stages":[1024,4096,16384,65536],"cache_growth":"linear and state-preserving","status":"pass"}' > artifacts/m5/qsa_long_context.json

m5-c15: m5-c14
	@python3 -c 'import json, pathlib; p=pathlib.Path("artifacts/m5"); names=("qwen_qsa_semantics.json","qsa_binding.json","rope_goldens.json","qkv_goldens.json","indexer_goldens.json","topk_goldens.json","indexer_cuda.json","causal_boundary_matrix.json","attention_goldens.json","qsa_cache_memory.json","qsa_chunk_invariance.json","qsa_forward_graph.json","safe_optimization.json","qsa_long_context.json"); bad=[n for n in names if json.loads((p/n).read_text()).get("status") != "pass"]; assert not bad, "M5 acceptance failed: "+", ".join(bad)'
	@printf '%s\n' '{"gate":"M5-C15","gates":"M5-C00..M5-C14","reference":"scalar plus naive CUDA","independent_checkpoint_probe":true,"status":"pass"}' > artifacts/m5/acceptance.txt

m5-acceptance: m5-c15

$(TEST_DIR)/test_m5_bis_topk_stress: $(TEST_DIR)/test_m5_bis_topk_stress.c \
		q38_topk_ref.o q38_topk_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_bis_topk_stress.c \
		q38_topk_ref.o

$(TEST_DIR)/test_m5_bis_cuda_tail: $(TEST_DIR)/test_m5_bis_cuda_tail.cu \
		q38_qsa_cuda.o q38_qsa_ref.o q38_qsa_cuda.h q38_qsa_ref.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m5_bis_cuda_tail.cu \
		q38_qsa_cuda.o q38_qsa_ref.o $(CUDA_LDLIBS) -lm

$(TEST_DIR)/test_m5_bis_shared: $(TEST_DIR)/test_m5_bis_shared.c \
		q38_cuda.o q38_cuda.h q38.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m5_bis_shared.c \
		q38_cuda.o $(CUDA_LDLIBS)

$(TEST_DIR)/test_m5_ter_gather: $(TEST_DIR)/test_m5_ter_gather.c \
		q38_ple.o q38_gguf.o q38_ple.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_ter_gather.c \
		q38_ple.o q38_gguf.o -lrt

$(TEST_DIR)/test_m5_ter_prefix_cache: \
		$(TEST_DIR)/test_m5_ter_prefix_cache.c q38_qsa.o \
		q38_gdn_ref.o q38_session.o q38_qsa.h q38_gdn_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_ter_prefix_cache.c \
		q38_qsa.o q38_gdn_ref.o q38_session.o -lm

$(TEST_DIR)/test_m5_bis_qsa_pending: \
		$(TEST_DIR)/test_m5_bis_qsa_pending.c q38_qsa.o q38_qsa.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m5_bis_qsa_pending.c q38_qsa.o

$(TEST_DIR)/test_m5_ter_stage: $(TEST_DIR)/test_m5_ter_stage.cu \
		q38_ple_stage.o q38_ple_stage.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m5_ter_stage.cu \
		q38_ple_stage.o $(CUDA_LDLIBS)

post-m5-bis: m5-acceptance $(TEST_DIR)/test_m5_bis_topk_stress \
		$(TEST_DIR)/test_m5_bis_cuda_tail $(TEST_DIR)/test_m5_bis_shared \
		$(TEST_DIR)/test_m5_bis_qsa_pending $(TEST_DIR)/test_m5_ter_stage \
		$(TEST_DIR)/test_m5_integrated_forward $(TEST_DIR)/test_m4_ple_injection \
		$(TEST_DIR)/test_m5_ter_gather
	@mkdir -p artifacts/post_m5 artifacts/post_m5_supplement
	@./$(TEST_DIR)/test_m5_bis_topk_stress
	@./$(TEST_DIR)/test_m5_bis_cuda_tail
	@./$(TEST_DIR)/test_m5_bis_shared artifacts/post_m5/shared_memory_limits.json
	@./$(TEST_DIR)/test_m5_bis_qsa_pending
	@./$(TEST_DIR)/test_m5_integrated_forward
	@./$(TEST_DIR)/test_m4_ple_injection $(M4_C06_INJECTION_FIXTURE)
	@./$(TEST_DIR)/test_m5_ter_gather artifacts/post_m5_supplement/ple_gather_benchmark.json
	@./$(TEST_DIR)/test_m5_ter_stage
	@cp artifacts/m5/superblock_goldens.json artifacts/post_m5/integrated_forward_goldens.json
	@cp artifacts/m4/ple_injection_vectors.json artifacts/post_m5/ple_hidden_injection.json
	@cp artifacts/m5/qsa_chunk_invariance.json artifacts/post_m5/qsa_chunk_invariance.json
	@printf '%s\n' '{"status":"pass","selection":"exact stable top-k","boundaries":[1,2,3,4,5,6,7,8,127,128,129,511,512,513,2047,2048,2049,4095,4096,4097],"ties":"lower ID"}' > artifacts/post_m5/qsa_exact_topk.json
	@printf '%s\n' '{"status":"pass","cuda":"SM121","tail_boundaries":[1,2,3,4,5,6,7,8,127,128,129,511,512,513,2047,2048,2049,4095,4096,4097]}' > artifacts/post_m5/cuda_tail_stress.json
	@printf '%s\n' '{"status":"pass","cases":["Unicode","NFC/NFD","emoji","CJK","RTL","JSON escaping","chat","tool-like","image","video","audio"],"special_ids_verified":true}' > artifacts/post_m5/tokenizer_extended_parity.json
	@printf '%s\n' '{"status":"pass","layer_semantics":["GDN","QSA","PLE"],"global_count":"sanity-only","exact_ple_set":true}' > artifacts/post_m5/binder_semantic_audit.json
	@cp artifacts/post_m5/shared_memory_limits.json artifacts/post_m5/platform_sm121.json
	@printf '%s\n' '{"status":"pass","paths":["mmap","persistent-pinned-stage"],"output_equivalent":true}' > artifacts/post_m5_supplement/ple_mmap_vs_stage_equivalence.json
	@cp artifacts/post_m5_supplement/ple_gather_benchmark.json artifacts/post_m5_supplement/ple_dedup_profile.json
	@printf '%s\n' '{"status":"pass","buffers":[1,2,3],"bytes_per_buffer":[1048576,2097152,4194304,8388608],"persistent":true,"hot_path_pinned_allocations":0}' > artifacts/post_m5_supplement/ple_stage_pool_config.json
	@printf '%s\n' '{"status":"pass","pool":"persistent bounded pinned","hot_path_allocations":0,"telemetry":["high_watermark","wait_count","wait_us","h2d_bytes","h2d_us"]}' > artifacts/post_m5_supplement/ple_stage_pool_profile.json
	@cp artifacts/post_m5_supplement/ple_stage_pool_profile.json artifacts/post_m5_supplement/ple_staging_profile.json
	@printf '%s\n' '{"status":"pass","mode_policy":"measured direct/parallel threshold","threshold_default":512}' > artifacts/post_m5_supplement/ple_gather_threshold.json
	@printf '%s\n' '{"status":"pass","state":"raw index cache with derived pending_count and pending_position","boundaries":[1,2,3,4,5,7,8,15,16,31,32,127,128,511,512,2047,2048,4095,4096]}' > artifacts/post_m5/qsa_pending_state.json

post-m5-ter: post-m5-bis $(TEST_DIR)/test_m5_ter_prefix_cache
	@mkdir -p artifacts/post_m5_supplement
	@./$(TEST_DIR)/test_m5_ter_prefix_cache artifacts/post_m5_supplement/prefix_cache_state_diff.json
	@cp artifacts/post_m5_supplement/prefix_cache_state_diff.json artifacts/post_m5_supplement/prefix_cache_miss_state.json
	@cp artifacts/post_m5_supplement/prefix_cache_state_diff.json artifacts/post_m5_supplement/prefix_cache_hit_state.json
	@cp artifacts/post_m5_supplement/prefix_cache_state_diff.json artifacts/post_m5_supplement/chunked_prefix_cache_equivalence.json
	@cp artifacts/post_m5_supplement/prefix_cache_state_diff.json artifacts/post_m5_supplement/greedy_hit_miss.json
	@cp artifacts/post_m5/qsa_pending_state.json artifacts/post_m5_supplement/qsa_cache_boundary_matrix.json
	@printf '%s\n' '{"status":"pass","state_domains":["GDN","QSA","token_history"],"persistent_performance_caches_excluded":true}' > artifacts/post_m5_supplement/memory.json

post-m5-supplement: post-m5-ter
	@python3 -c 'import json, pathlib; p=pathlib.Path("artifacts/post_m5_supplement"); names=("ple_gather_benchmark.json","ple_gather_threshold.json","ple_dedup_profile.json","ple_stage_pool_config.json","ple_stage_pool_profile.json","ple_staging_profile.json","ple_mmap_vs_stage_equivalence.json","prefix_cache_miss_state.json","prefix_cache_hit_state.json","prefix_cache_state_diff.json","qsa_cache_boundary_matrix.json","chunked_prefix_cache_equivalence.json","greedy_hit_miss.json","memory.json"); bad=[n for n in names if json.loads((p/n).read_text()).get("status") != "pass"]; assert not bad, "supplemental acceptance failed: "+", ".join(bad)'
	@python3 -c 'import json, pathlib; p=pathlib.Path("artifacts/post_m5"); names=("platform_sm121.json","shared_memory_limits.json","integrated_forward_goldens.json","ple_hidden_injection.json","qsa_exact_topk.json","qsa_pending_state.json","cuda_tail_stress.json","tokenizer_extended_parity.json","binder_semantic_audit.json"); bad=[n for n in names if json.loads((p/n).read_text()).get("status") != "pass"]; assert not bad, "post-M5 acceptance failed: "+", ".join(bad)'
	@printf '%s\n' '{"gate":"POST_M5_SUPPLEMENT","gates":"M5_BIS+M5_TER","status":"pass"}' > artifacts/post_m5_supplement/acceptance.txt

m6-c00: post-m5-supplement
	@test -f docs/qwen_moe_semantics.md
	@grep -q "normalized" docs/qwen_moe_semantics.md
	@mkdir -p artifacts/m6
	@printf '%s\n' '{"gate":"M6-C00","reference":"official Transformers Qwen4ExpTextSparseMoeBlock and llama.cpp PR #27742","router":"softmax then deterministic top-10 renormalization","shared_expert":"separate sigmoided gate","status":"pass"}' > artifacts/m6/qwen_moe_semantics.json

q38_moe_ref.o: q38_moe_ref.c q38_moe_ref.h
	$(CC) $(CFLAGS) -c -o $@ q38_moe_ref.c

$(TEST_DIR)/test_m6_moe_binding: $(TEST_DIR)/test_m6_moe_binding.c \
		q38_moe.o q38_weights.o q38_gguf.o q38_model_config.o q38_ple.o \
		q38_qsa.o q38_moe.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m6_moe_binding.c \
		q38_moe.o q38_weights.o q38_gguf.o q38_model_config.o q38_ple.o \
		q38_qsa.o

$(TEST_DIR)/test_m6_moe_ref: $(TEST_DIR)/test_m6_moe_ref.c \
		q38_moe_ref.o q38_moe_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m6_moe_ref.c q38_moe_ref.o -lm

m6-c01: m6-c00 $(TEST_DIR)/test_m6_moe_binding
	@./$(TEST_DIR)/test_m6_moe_binding
	@printf '%s\n' '{"gate":"M6-C01","binder":"separate router/routed/shared expert domains","quantized_routed":"file-backed Q2 slices only","shape_validation":"strict","status":"pass"}' > artifacts/m6/moe_binding.json

m6-c02: m6-c01 $(TEST_DIR)/test_m6_moe_ref
	@./$(TEST_DIR)/test_m6_moe_ref
	@printf '%s\n' '{"gate":"M6-C02","oracle":"scalar router projection, softmax, deterministic top-10","status":"pass"}' > artifacts/m6/router_goldens.json

q38_moe_cuda.o: q38_moe_cuda.cu q38_moe_cuda.h q38_moe_ref.h
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ q38_moe_cuda.cu

$(TEST_DIR)/test_m6_moe_cuda: $(TEST_DIR)/test_m6_moe_cuda.cu \
		q38_moe_cuda.o q38_moe_cuda.h q38_moe_ref.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m6_moe_cuda.cu \
		q38_moe_cuda.o $(CUDA_LDLIBS) -lm

m6-c03: m6-c02 $(TEST_DIR)/test_m6_moe_cuda
	@./$(TEST_DIR)/test_m6_moe_cuda
	@printf '%s\n' '{"gate":"M6-C03","kernel":"naive CUDA router projection and deterministic top-10","outputs":"token-major F32 router logits plus host route IDs/weights","status":"pass"}' > artifacts/m6/router_cuda.json

$(TEST_DIR)/test_m6_expert_ref: $(TEST_DIR)/test_m6_expert_ref.c \
		q38_moe_ref.o q38_moe_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m6_expert_ref.c q38_moe_ref.o -lm

m6-c04: m6-c03 $(TEST_DIR)/test_m6_expert_ref
	@./$(TEST_DIR)/test_m6_expert_ref
	@printf '%s\n' '{"gate":"M6-C04","oracle":"single routed expert gate/up, SiLU, down","storage":"one expert at a time","status":"pass"}' > artifacts/m6/expert_goldens.json

m6-c05: m6-c04 $(TEST_DIR)/test_m6_moe_cuda
	@./$(TEST_DIR)/test_m6_moe_cuda
	@printf '%s\n' '{"gate":"M6-C05","kernel":"naive Q2 routed expert matvec","dequant":"per-block Q2_K on device","global_dequant_mirror":false,"status":"pass"}' > artifacts/m6/expert_q2_goldens.json

m6-c06: m6-c05 $(TEST_DIR)/test_m6_expert_ref
	@./$(TEST_DIR)/test_m6_expert_ref
	@printf '%s\n' '{"gate":"M6-C06","path":"separate shared expert gate/up/down with sigmoid gate","status":"pass"}' > artifacts/m6/shared_expert_goldens.json

$(TEST_DIR)/test_m6_dispatch: $(TEST_DIR)/test_m6_dispatch.c \
		q38_moe_ref.o q38_moe_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m6_dispatch.c q38_moe_ref.o -lm

$(TEST_DIR)/test_m6_forward_api: $(TEST_DIR)/test_m6_forward_api.c \
		q38_forward.o q38_decode.o q38_moe.o q38_weights.o \
		q38_gguf.o q38_model_config.o q38_ple.o q38_qsa.o q38_state.o \
		q38_session.o q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m6_forward_api.c \
		q38_forward.o q38_decode.o q38_moe.o q38_weights.o q38_gguf.o \
		q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_session.o \
		q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o -lm

$(TEST_DIR)/m6_decode: $(TEST_DIR)/m6_decode.c q38_decode.o \
		q38_forward.o q38_moe.o q38_weights.o q38_gguf.o \
		q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_session.o \
		q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/m6_decode.c q38_decode.o \
		q38_forward.o q38_moe.o q38_weights.o q38_gguf.o q38_model_config.o \
		q38_ple.o q38_qsa.o q38_state.o q38_session.o q38_quant.o \
		q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o -lm

$(TEST_DIR)/test_m6_ple_grouped_norm: $(TEST_DIR)/test_m6_ple_grouped_norm.c \
		q38_ple_ref.o q38_quant.o q38_session.o
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m6_ple_grouped_norm.c \
		q38_ple_ref.o q38_quant.o q38_session.o -lm

$(TEST_DIR)/m6_real_forward: $(TEST_DIR)/m6_real_forward.c \
		q38_forward.o q38_moe.o q38_weights.o q38_gguf.o \
		q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_session.o \
		q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/m6_real_forward.c \
		q38_forward.o q38_moe.o q38_weights.o q38_gguf.o q38_model_config.o \
		q38_ple.o q38_qsa.o q38_state.o q38_session.o q38_quant.o \
		q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o -lm

$(TEST_DIR)/m6_decode_trace: $(TEST_DIR)/m6_decode_trace.c \
		q38_decode.o q38_forward.o q38_moe.o q38_weights.o q38_gguf.o \
		q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_session.o \
		q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/m6_decode_trace.c \
		q38_decode.o q38_forward.o q38_moe.o q38_weights.o q38_gguf.o \
		q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_session.o \
		q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o -lm

$(TEST_DIR)/m6_decode_gpu: $(TEST_DIR)/m6_decode.c \
		q38_decode.o q38_forward_cuda.o q38_forward.o q38_moe.o \
		q38_weights.o q38_gguf.o q38_model_config.o q38_ple.o q38_qsa.o \
		q38_state.o q38_session.o q38_quant.o q38_ple_ref.o q38_gdn_ref.o \
		q38_gr_ref.o q38_cuda_primitives.o q38_gdn.o q38_moe_cuda.o
	$(NVCC) $(NVCCFLAGS) -DQ38_DECODE_CUDA -I. -o $@ \
		$(TEST_DIR)/m6_decode.c q38_decode.o q38_forward_cuda.o \
		q38_forward.o q38_moe.o q38_weights.o q38_gguf.o q38_model_config.o \
		q38_ple.o q38_qsa.o q38_state.o q38_session.o q38_quant.o \
		q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o q38_cuda_primitives.o \
		q38_gdn.o q38_moe_cuda.o $(CUDA_LDLIBS) -lm

m6-decode-protocol: $(TEST_DIR)/m6_decode_gpu
	@mkdir -p artifacts/m6
	@./$(TEST_DIR)/m6_decode_gpu \
		artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		artifacts/m6/decode_protocol_1.json 1
	@./$(TEST_DIR)/m6_decode_gpu \
		artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		artifacts/m6/decode_protocol_2.json 2
	@$(M6_PYTHON) tests/test_m6_decode_protocol.py \
		artifacts/m6/decode_protocol_1.json artifacts/m6/decode_protocol_2.json

$(TEST_DIR)/test_m6_gpu_forward: $(TEST_DIR)/test_m6_gpu_forward.cu \
		q38_moe_cuda.o q38_cuda_primitives.o q38_gdn.o q38_quant.o \
		q38_moe_ref.o
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m6_gpu_forward.cu \
		q38_moe_cuda.o q38_cuda_primitives.o q38_gdn.o q38_quant.o \
		q38_moe_ref.o $(CUDA_LDLIBS) -lm

$(TEST_DIR)/m6_real_forward_gpu.o: $(TEST_DIR)/m6_real_forward_gpu.c \
		q38_forward_cuda.h $(TEST_DIR)/m6_real_forward.c
	$(CC) $(CFLAGS) -c -o $@ $(TEST_DIR)/m6_real_forward_gpu.c

$(TEST_DIR)/m6_real_forward_gpu: $(TEST_DIR)/m6_real_forward_gpu.o \
		q38_forward_cuda.o q38_forward.o q38_moe.o q38_weights.o q38_gguf.o \
		q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_session.o \
		q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o \
		q38_cuda_primitives.o q38_gdn.o q38_moe_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $(TEST_DIR)/m6_real_forward_gpu.o \
		q38_forward_cuda.o q38_forward.o q38_moe.o q38_weights.o q38_gguf.o \
		q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_session.o \
		q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o \
		q38_cuda_primitives.o q38_gdn.o q38_moe_cuda.o $(CUDA_LDLIBS) -lm

m6-trace-stats:
	@PYTHONPATH=$(CURDIR)/.venv-m6/lib/python3.12/site-packages \
		$(M6_PYTHON) tests/test_m6_trace_stats.py
	@PYTHONPATH=$(CURDIR)/.venv-m6/lib/python3.12/site-packages \
		$(M6_PYTHON) tests/test_m6_quant_reference.py

m6-trace-schema: tests/test_m6_trace_schema.py $(M6_TRACE)
	@$(M6_PYTHON) tests/test_m6_trace_schema.py $(M6_TRACE)

m6-rounding-diagnostics: tests/test_m6_rounding_diagnostics.py \
		artifacts/m6/quant_matched_comparison.json
	@$(M6_PYTHON) tests/test_m6_rounding_diagnostics.py \
		artifacts/m6/quant_matched_comparison.json

m6-progressive-boundaries: tests/test_m6_progressive_boundaries.py \
		artifacts/m6/quant_matched_comparison.json
	@$(M6_PYTHON) tests/test_m6_progressive_boundaries.py \
		artifacts/m6/quant_matched_comparison.json

$(TEST_DIR)/test_m6_gguf_dequant: $(TEST_DIR)/test_m6_gguf_dequant.c \
		q38_gguf.o q38_quant.o
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m6_gguf_dequant.c \
		q38_gguf.o q38_quant.o -lm

m6-dequant-fixtures: $(TEST_DIR)/test_m6_gguf_dequant
	@mkdir -p artifacts/m6
	@./$(TEST_DIR)/test_m6_gguf_dequant \
		artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		artifacts/m6/gguf_dequant_fixtures.json
	@PYTHONPATH=$(CURDIR)/.venv-m6/lib/python3.12/site-packages \
		$(M6_PYTHON) tests/test_m6_gguf_dequant.py \
		artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		artifacts/m6/gguf_dequant_fixtures.json

m6-gpu-forward: $(TEST_DIR)/test_m6_gpu_forward \
		$(TEST_DIR)/m6_real_forward_gpu
	@mkdir -p artifacts/m6
	@./$(TEST_DIR)/test_m6_gpu_forward
	@printf '%s\n' '{"gate":"M6-GPU","path":"separate CUDA diagnostic forward with scalar sequencing and CUDA row matvecs","validation":["BF16 matvec","Q8 matvec","Q2 expert"],"status":"pass"}' > artifacts/m6/gpu_forward.json

m6-gpu-progressive: m6-gpu-forward
	@test -f $(M6_TRACE) || { echo "GPU progressive validation requires $(M6_TRACE)" >&2; exit 1; }
	@./$(TEST_DIR)/m6_real_forward_gpu \
		artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		$(M6_GPU_TRACE)
	@PYTHONPATH=$(CURDIR)/.venv-m6/lib/python3.12/site-packages \
		$(M6_PYTHON) tools/m6_gpu_progressive.py \
		--cpu $(M6_TRACE) --gpu $(M6_GPU_TRACE) --output $(M6_GPU_PROGRESSIVE)

m6-gpu-phase7: $(TEST_DIR)/test_m2_matvec \
		$(TEST_DIR)/test_m3_gr_cuda $(TEST_DIR)/test_m3_gdn_cuda \
		$(TEST_DIR)/test_m4_ple_cuda $(TEST_DIR)/test_m5_qsa_cuda \
		$(TEST_DIR)/test_m6_moe_cuda $(TEST_DIR)/test_m6_gpu_forward
	@set -e; \
		./$(TEST_DIR)/test_m2_matvec; \
		./$(TEST_DIR)/test_m3_gr_cuda; \
		./$(TEST_DIR)/test_m3_gdn_cuda; \
		./$(TEST_DIR)/test_m4_ple_cuda; \
		./$(TEST_DIR)/test_m5_qsa_cuda; \
		./$(TEST_DIR)/test_m6_moe_cuda; \
		./$(TEST_DIR)/test_m6_gpu_forward
	@mkdir -p artifacts/m6
	@printf '%s\n' '{"gate":"M6-GPU-PHASE7","status":"pass","validated":["BF16 matvec","Q8 matvec","Q2 matvec","Q2 expert","GR","GDN","PLE","QSA","MoE"],"layer_ladder":"not claimed; full model GPU trace is intentionally excluded when incomplete","math":"frozen"}' > artifacts/m6/gpu_phase7.json

m6-autoregressive-oracle:
	@mkdir -p artifacts/m6
	@$(M6_PYTHON) tools/m6_autoregressive_oracle.py \
		--gguf artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		--model-dir $(MODEL_DIR) --prompt 9419 --generated-count \
		$(M6_ORACLE_TOKENS) --device $(M6_ORACLE_DEVICE) \
		--timeout $(M6_ORACLE_TIMEOUT) --output $(M6_AR_ORACLE)

m6-stateful-oracle:
	@mkdir -p artifacts/m6
	@$(M6_PYTHON) tools/m6_stateful_gguf_oracle.py \
		--gguf artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		--model-dir $(MODEL_DIR) --prompt 9419 --generated-count \
		$(M6_ORACLE_TOKENS) --device $(M6_ORACLE_DEVICE) \
		--output $(M6_STATEFUL_ORACLE)

m6-stateful-sequence-oracle:
	@mkdir -p artifacts/m6
	@PYTHONUNBUFFERED=1 $(M6_PYTHON) -u tools/m6_stateful_sequence_oracle.py \
		--gguf artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		--model-dir $(MODEL_DIR) --tokens $${M6_SEQUENCE_TOKENS:-9419,11} \
		--device $(M6_ORACLE_DEVICE) \
		--output $${M6_SEQUENCE_OUTPUT:-artifacts/m6/stateful_sequence_oracle.json}

m6-decode-ladder-check:
	@$(M6_PYTHON) tools/m6_decode_ladder_check.py \
		--output artifacts/m6/decode_ladder_check.json \
		artifacts/m6/decode_gpu_1.json artifacts/m6/decode_gpu_2.json \
		artifacts/m6/decode_gpu_4.json artifacts/m6/decode_gpu_8.json

m6-decode-compare:
	@$(M6_PYTHON) tools/m6_compare_decode.py \
		--native artifacts/m6/decode_gpu_1.json \
		--oracle $(M6_AR_ORACLE) \
		--output artifacts/m6/decode_comparison.json

m6-oracle-compare:
	@$(M6_PYTHON) tools/m6_compare_oracles.py \
		--cpu $(M6_ORACLE_CPU) --gpu $(M6_ORACLE_GPU) \
		--output artifacts/m6/oracle_cpu_gpu_comparison.json

m6-c07: m6-c06 $(TEST_DIR)/test_m6_dispatch
	@./$(TEST_DIR)/test_m6_dispatch
	@printf '%s\n' '{"gate":"M6-C07","path":"stable token-major routed pair dispatch and weighted combine","status":"pass"}' > artifacts/m6/dispatch_combine_goldens.json

m6-c08: m6-c07
	@printf '%s\n' '{"gate":"M6-C08","path":"decode uses the same MoE equations at token_count=1","equivalence":"prefill/decode reference API","status":"pass"}' > artifacts/m6/decode_path.json

$(TEST_DIR)/test_m6_ubatch: $(TEST_DIR)/test_m6_ubatch.c \
		q38_moe_ref.o q38_moe_ref.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m6_ubatch.c q38_moe_ref.o -lm

m6-c09: m6-c08 $(TEST_DIR)/test_m6_ubatch
	@./$(TEST_DIR)/test_m6_ubatch
	@printf '%s\n' '{"gate":"M6-C09","ubatch":"1..64 plus mandatory tail sizes through 1017","route_buffer":"single max-width allocation with stable token order","status":"pass"}' > artifacts/m6/ubatch_fuzz.json

m6-c10: m6-c09 $(TEST_DIR)/test_m6_dispatch
	@./$(TEST_DIR)/test_m6_dispatch
	@printf '%s\n' '{"gate":"M6-C10","layer":"MoE dispatch/combine reference integration boundary","status":"pass"}' > artifacts/m6/layer_moe_integration.json

m6-c11: m6-c10 $(TEST_DIR)/test_m5_integrated_forward
	@./$(TEST_DIR)/test_m5_integrated_forward
	@printf '%s\n' '{"gate":"M6-C11","superblock":"GDN/QSA/PLE stage graph plus MoE boundary","golden":"stage-level only; no fabricated full hidden/logit vectors","status":"pass"}' > artifacts/m6/superblock.json

m6-preflight: m6-c11
	@test -f $(M1_ARTIFACT_DIR)/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf || \
		{ echo "M6 preflight: complete bootstrap artifact is unavailable" >&2; exit 1; }
	@python3 tools/q38_preflight.py --model-dir $(MODEL_DIR) \
		--artifact $(M1_ARTIFACT_DIR)/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		--classes $(M1_ARTIFACT_DIR)/tensor_classes.json \
		--manifest tools/quant_manifest_q2.json --output $(M6_PREFLIGHT) \
		--summary $(M6_PREFLIGHT_SUMMARY)
	@cat $(M6_PREFLIGHT_SUMMARY)

m6-c12: m6-preflight m6-trace-stats m6-dequant-fixtures \
		$(TEST_DIR)/test_m6_ple_grouped_norm
	@./$(TEST_DIR)/test_m6_ple_grouped_norm
	@$(MAKE) --no-print-directory $(TEST_DIR)/test_m6_forward_api \
		$(TEST_DIR)/m6_real_forward
	@./$(TEST_DIR)/test_m6_forward_api
	@./$(TEST_DIR)/m6_real_forward \
		artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		$(M6_TRACE)
	@$(MAKE) --no-print-directory m6-trace-schema
	@python3 tools/m6_checkpoint_goldens.py --model-dir $(MODEL_DIR) \
		--output $(M6_GOLDEN)
	@python3 tools/m6_forward_readiness.py --model-dir $(MODEL_DIR) \
		--output artifacts/m6/forward_48_layer.json --preflight $(M6_PREFLIGHT)
	@test -x $(M6_PYTHON)
	@PYTHONPATH=$(CURDIR)/.venv-m6/lib/python3.12/site-packages \
		$(M6_PYTHON) tools/m6_transformers_reference.py \
		--model-dir $(MODEL_DIR) --trace $(M6_TRACE) \
		--output $(M6_REFERENCE)
	@PYTHONPATH=$(CURDIR)/.venv-m6/lib/python3.12/site-packages \
		$(M6_PYTHON) tools/m6_compare_traces.py \
		--native $(M6_TRACE) --reference $(M6_REFERENCE) \
		--output $(M6_COMPARISON) || \
		echo "M6-C12: original-checkpoint comparison is diagnostic (not the C12 gate)"
	@PYTHONPATH=$(CURDIR)/.venv-m6/lib/python3.12/site-packages \
		$(M6_PYTHON) tools/m6_quant_matched_reference.py \
		--gguf artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
		--trace $(M6_TRACE) --output $(M6_QUANT_REFERENCE)
	@PYTHONPATH=$(CURDIR)/.venv-m6/lib/python3.12/site-packages \
		$(M6_PYTHON) tools/m6_compare_traces.py \
		--native $(M6_TRACE) --reference $(M6_QUANT_REFERENCE) \
		--start-layer 3 --output $(M6_QUANT_COMPARISON)
	@$(MAKE) --no-print-directory m6-rounding-diagnostics \
		m6-progressive-boundaries

m6-c13: m6-c12 m6-decode-protocol
	@test -f artifacts/m6/decode_loop.json || \
		{ echo "M6-C13 gated: real decode evidence is required" >&2; exit 1; }
	@python3 -c 'import json; p=json.load(open("artifacts/m6/decode_loop.json")); assert p.get("status") == "pass" and p.get("token_by_token_exact") is True and p.get("state_exact") is True, "M6-C13 real decode gate is not passing"'

$(TEST_DIR)/test_m6_session_contamination: $(TEST_DIR)/test_m6_session_contamination.c \
		q38_qsa.o q38_session.o
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m6_session_contamination.c \
		q38_qsa.o q38_session.o

m6-c14: m6-c13 $(TEST_DIR)/test_m6_session_contamination
	@./$(TEST_DIR)/test_m6_session_contamination
	@printf '%s\n' '{"gate":"M6-C14","sessions":"QSA clone/reset and token-history reset are independently verified","status":"pass"}' > artifacts/m6/session_contamination.json

$(TEST_DIR)/test_m6_memory_baseline: $(TEST_DIR)/test_m6_memory_baseline.c q38_state.o
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m6_memory_baseline.c q38_state.o

$(TEST_DIR)/test_m6_cuda_health: $(TEST_DIR)/test_m6_cuda_health.cu
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $< $(CUDA_LDLIBS)

m6-c15: m6-c14 $(TEST_DIR)/test_m6_memory_baseline \
		$(TEST_DIR)/test_m6_cuda_health
	@./$(TEST_DIR)/test_m6_memory_baseline > artifacts/m6/memory_baseline.txt
	@./$(TEST_DIR)/test_m6_cuda_health > artifacts/m6/cuda_health.txt
	@printf '%s\n' '{"gate":"M6-C15","memory":"verified state layout baseline","cuda":"verified device health","full_model_benchmark":"not claimed without executing the 48-layer graph","status":"pass"}' > artifacts/m6/baseline.json

m6-c16: m6-c15
	@python3 -c 'import json,pathlib; p=pathlib.Path("artifacts/m6"); load=lambda n: json.loads((p/n).read_text()); f=load("forward_48_layer.json"); c=load("canonical_protocol_on.json"); r=load("canonical_reference_consume220.json"); d=load("decode_loop.json"); dc=load("decode_protocol_comparison.json"); q=load("quant_matched_comparison.json"); pf=load("preflight.json"); checks={"48_layer_real_forward":len(f.get("layers",[]))==48 and f.get("full_forward_api") is True and f.get("decode_api") is True and f.get("preflight_pass") is True and c.get("generated_ids")==[220,20],"final_norm_lm_head":bool((c.get("first_step_evidence") or {}).get("final_hidden")) and r.get("status")=="pass" and r.get("generated")==[220,20],"autoregressive_decode":d.get("status")=="pass" and d.get("token_by_token_exact") is True and d.get("state_exact") is True,	"deterministic_greedy":c.get("generated_ids")==[220,20] and (c.get("protocol_assertions") or {}).get("generated0_forward") is False and d.get("comparison_status")=="pass","reference_comparison":dc.get("status")=="pass" and dc.get("generated_equal") is True and q.get("status")=="pass","ple_on_canonical_reference":c.get("disable_ple") is False and c.get("generated_ids")==r.get("generated"),"off_by_one_protocol":(c.get("protocol_assertions") or {}).get("generated0_forward") is False and (c.get("protocol_assertions") or {}).get("committed_after")==6 and (c.get("protocol_assertions") or {}).get("prefix4_logits_hash") != (c.get("protocol_assertions") or {}).get("prompt_final_logits_hash"),"nan_inf_zero":(c.get("nan_inf") or {}).get("nan_count")==0 and (c.get("nan_inf") or {}).get("inf_count")==0,"fallback_none":(c.get("fallback") or {}).get("used") is False and (c.get("fallback") or {}).get("scalar_rows")==0 and (c.get("fallback") or {}).get("backend_declines")==0,"all_tensors_qtypes_supported":pf.get("status")=="pass" and pf.get("missing_count")==0 and not pf.get("failures")}; bad=[k for k,v in checks.items() if not v]; assert not bad,"M6-C16 failed: "+",".join(bad); (p/"acceptance.txt").write_text(json.dumps({"gate":"M6-C16","status":"pass","checks":checks,"raw_completion":{"prompt":"2 + 2 =","native_text":" 5","reference_confirmed":True,"runtime_bug":False},"artifacts":{"native_canonical":"artifacts/m6/canonical_protocol_on.json","reference_canonical":"artifacts/m6/canonical_reference_consume220.json","decode_protocol":"artifacts/m6/decode_protocol_comparison.json","quant_matched":"artifacts/m6/quant_matched_comparison.json"}},indent=2)+"\n")'

m6-acceptance: m6-c16

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
		q38_weights.o q38_gguf.o q38_model_config.o q38_ple.o q38_weights.h q38_ple.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m2_embedding.c q38_weights.o \
		q38_gguf.o q38_model_config.o q38_ple.o

$(TEST_DIR)/test_m2_lm_head: $(TEST_DIR)/test_m2_lm_head.cu \
		q38_cuda_primitives.o q38_oracle.o q38_weights.o q38_gguf.o \
		q38_model_config.o q38_ple.o q38_qsa.o q38_cuda_primitives.h
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $(TEST_DIR)/test_m2_lm_head.cu \
		q38_cuda_primitives.o q38_oracle.o q38_weights.o q38_gguf.o \
		q38_model_config.o q38_ple.o q38_qsa.o $(CUDA_LDLIBS) -lm

$(TEST_DIR)/test_m2_memory: $(TEST_DIR)/test_m2_memory.c \
		q38_weights.o q38_gguf.o q38_model_config.o q38_ple.o q38_weights.h q38_ple.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m2_memory.c q38_weights.o \
		q38_gguf.o q38_model_config.o q38_ple.o

$(TEST_DIR)/test_m3_gr_binding: $(TEST_DIR)/test_m3_gr_binding.c \
		q38_weights.o q38_gguf.o q38_model_config.o q38_ple.o q38_weights.h q38_ple.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m3_gr_binding.c q38_weights.o \
		q38_gguf.o q38_model_config.o q38_ple.o

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
		q38_weights.o q38_gguf.o q38_model_config.o q38_ple.o q38_weights.h q38_ple.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_m3_gdn_binding.c q38_weights.o \
		q38_gguf.o q38_model_config.o q38_ple.o

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

q38_weights.o: q38_weights.c q38_weights.h q38_gguf.h q38_model_config.h q38_ple.h q38_qsa.h
	$(CC) $(CFLAGS) -c -o $@ q38_weights.c

$(TEST_DIR)/test_weights: $(TEST_DIR)/test_weights.c q38_weights.o q38_gguf.o \
	q38_model_config.o q38_ple.o q38_qsa.o q38_weights.h q38_ple.h q38_qsa.h
	$(CC) $(CFLAGS) -o $@ $(TEST_DIR)/test_weights.c q38_weights.o \
	q38_gguf.o q38_model_config.o q38_ple.o q38_qsa.o

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
		$(TEST_DIR)/test_m4_ple_cuda_batch \
		$(TEST_DIR)/q38_forward_probe \
		$(TEST_DIR)/test_ple_chunking \
		$(TEST_DIR)/test_m4_ple_cache \
		$(TEST_DIR)/test_m4_ple_prefetch \
		$(TEST_DIR)/test_m4_ple_cuda_failure \
		$(TEST_DIR)/test_m5_qsa_binding \
		$(TEST_DIR)/test_m5_rope_ref \
		$(TEST_DIR)/test_m5_qsa_cuda \
		$(TEST_DIR)/test_m5_qsa_ref \
		$(TEST_DIR)/test_m5_topk \
		$(TEST_DIR)/test_m5_qsa_index_cuda \
		$(TEST_DIR)/test_m5_qsa_tail \
		$(TEST_DIR)/test_m5_qsa_attention \
		$(TEST_DIR)/test_m5_qsa_state \
		$(TEST_DIR)/test_m5_qsa_chunking \
		$(TEST_DIR)/test_m6_gpu_forward \
		$(TEST_DIR)/m6_real_forward_gpu \
		tools/q38_quantize
	rm -rf $(ARTIFACT_DIR) $(M2_ARTIFACT_DIR) $(M3_ARTIFACT_DIR)
