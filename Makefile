# q38 canonical production build and benchmark interface.

CC ?= cc
CFLAGS ?= -O3 -g -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -I. -pthread
MODEL_DIR ?= /home/lvx/q38model

CUDA_HOME ?= $(shell if [ -x /usr/local/cuda/bin/nvcc ]; then \
	printf '%s' /usr/local/cuda; \
	elif command -v nvcc >/dev/null 2>&1; then \
	dirname "$$(dirname "$$(command -v nvcc)")"; \
	else \
	printf '%s' /usr/local/cuda; \
	fi)
NVCC ?= $(CUDA_HOME)/bin/nvcc
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math -I$(CURDIR) $(NVCC_ARCH_FLAGS)
CUDA_LDLIBS ?= -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -Xcompiler -pthread

Q2_CANONICAL_MODEL ?= artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf
Q2_CANONICAL_TOKENIZER ?= $(MODEL_DIR)
Q2_CANONICAL_PROMPT ?= Explain in simple terms why the sky appears blue during the day and red near sunset.
Q2_REFERENCE_OUTPUT_DIR := artifacts/perf/reference_0
Q2_CURRENT_OUTPUT_DIR := artifacts/perf/current

PRODUCTION_C_OBJS := \
	q38.o q38_gguf.o q38_memory.o q38_platform.o q38_tokenizer.o \
	q38_decode.o q38_forward.o q38_ple_prefetch.o q38_moe.o q38_weights.o \
	q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_quant.o \
	q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o q38_replay.o q38_profile.o \
	q38_residency.o q38_session.o
PRODUCTION_CUDA_OBJS := \
	q38_cuda.o q38_forward_cuda.o q38_qsa_cuda.o q38_cuda_primitives.o \
	q38_gdn.o q38_moe_cuda.o q38_cuda_timing.o q38_profile_cuda.o \
	q38_topk_cuda.o
PRODUCTION_OBJS := $(PRODUCTION_C_OBJS) $(PRODUCTION_CUDA_OBJS)

CANONICAL_BENCH_C_OBJS := \
	q38_gguf.o q38_forward.o q38_ple_prefetch.o q38_weights.o \
	q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_session.o \
	q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o q38_moe.o q38_decode.o \
	q38_tokenizer.o
CANONICAL_BENCH_CUDA_OBJS := \
	q38_forward_cuda.o q38_qsa_cuda.o q38_cuda_primitives.o q38_gdn.o \
	q38_moe_cuda.o q38_profile_cuda.o q38_residency.o \
	q38_topk_cuda.o

TEST_BINS := \
	tests/test_platform tests/test_gguf tests/test_memory \
	tests/test_model_config tests/test_quant_blocks tests/test_residency

.PHONY: all q38 q38-server q38-cli spark test test-server clean tools \
	bench-q2-reference-0 bench-q2-decode bench-q2-prefill \
	check-perf-artifacts

all: q38

q38: $(PRODUCTION_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(PRODUCTION_OBJS) $(CUDA_LDLIBS)

SERVER_OBJS := q38_server.o q38_server_protocol.o q38_server_engine_mock.o \
	q38_server_engine.o q38_json.o q38_kvstore.o

q38-server: q38_server_main.o $(SERVER_OBJS)
	$(CC) $(CFLAGS) -o $@ q38_server_main.o $(SERVER_OBJS)

q38-cli: q38_cli.o q38_json.o
	$(CC) $(CFLAGS) -o $@ q38_cli.o q38_json.o

spark: q38 $(TEST_BINS)

test: $(TEST_BINS)
	@set -e; for test in $(TEST_BINS); do ./$$test; done

test-server: tests/test_q38_json tests/test_q38_kvstore tests/test_q38_server_engine \
		tests/test_q38_server_protocol tests/test_q38_server
	@set -e; \
	./tests/test_q38_json; \
	./tests/test_q38_kvstore; \
	./tests/test_q38_server_engine; \
	./tests/test_q38_server_protocol; \
	./tests/test_q38_server

tests/test_q38_json: tests/test_q38_json.c q38_json.o q38_json.h
	$(CC) $(CFLAGS) -o $@ tests/test_q38_json.c q38_json.o

tests/test_q38_kvstore: tests/test_q38_kvstore.c q38_kvstore.o q38_kvstore.h
	$(CC) $(CFLAGS) -o $@ tests/test_q38_kvstore.c q38_kvstore.o

tests/test_q38_server_engine: tests/test_q38_server_engine.c \
		q38_server_engine_mock.o q38_server_engine.o q38_prompt.o \
		q38_json.o q38_server_engine.h q38_prompt.h
	$(CC) $(CFLAGS) -o $@ tests/test_q38_server_engine.c \
		q38_server_engine_mock.o q38_server_engine.o q38_prompt.o q38_json.o

tests/test_q38_server_protocol: tests/test_q38_server_protocol.c \
		q38_server_protocol.o q38_server_engine_mock.o q38_server_engine.o q38_json.o \
		q38_server_protocol.h q38_server_engine.h q38_json.h
	$(CC) $(CFLAGS) -o $@ tests/test_q38_server_protocol.c \
		q38_server_protocol.o q38_server_engine_mock.o q38_server_engine.o q38_json.o

tests/test_q38_server: tests/test_q38_server.c $(SERVER_OBJS) \
		q38_server.h q38_server_protocol.h q38_server_engine.h q38_json.h
	$(CC) $(CFLAGS) -o $@ tests/test_q38_server.c $(SERVER_OBJS)

tools: tools/q38_quantize

tools/q38_quantize: tools/q38_quantize.c third_party/gguf-tools/quants.c \
		third_party/gguf-tools/quants.h
	$(CC) $(CFLAGS) -Ithird_party/gguf-tools -o $@ \
		tools/q38_quantize.c third_party/gguf-tools/quants.c -lm -lpthread

tests/test_platform: tests/test_platform.c q38_platform.o q38_cuda.o \
		q38.h q38_platform.h q38_cuda.h
	$(NVCC) $(NVCCFLAGS) -o $@ tests/test_platform.c \
		q38_platform.o q38_cuda.o $(CUDA_LDLIBS)

tests/test_gguf: tests/test_gguf.c q38_gguf.o q38.h q38_gguf.h
	$(CC) $(CFLAGS) -o $@ tests/test_gguf.c q38_gguf.o

tests/test_memory: tests/test_memory.c q38_memory.o q38.h q38_memory.h
	$(CC) $(CFLAGS) -o $@ tests/test_memory.c q38_memory.o

tests/test_model_config: tests/test_model_config.c q38_model_config.o \
		q38_model_config.h
	$(CC) $(CFLAGS) -o $@ tests/test_model_config.c q38_model_config.o

tests/test_quant_blocks: tests/test_quant_blocks.c \
		third_party/gguf-tools/quants.c third_party/gguf-tools/quants.h
	$(CC) $(CFLAGS) -Ithird_party/gguf-tools -o $@ \
		tests/test_quant_blocks.c third_party/gguf-tools/quants.c -lm -lpthread

tests/test_residency: tests/test_residency.c q38_residency.o \
		q38_residency.h
	$(CC) $(CFLAGS) -o $@ tests/test_residency.c q38_residency.o

tests/q2_canonical_bench: tests/q2_canonical_bench.c \
		$(CANONICAL_BENCH_C_OBJS) $(CANONICAL_BENCH_CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/q2_canonical_bench.c \
		$(CANONICAL_BENCH_C_OBJS) $(CANONICAL_BENCH_CUDA_OBJS) \
		$(CUDA_LDLIBS) -lm

bench-q2-reference-0: tests/q2_canonical_bench
	@test ! -e $(Q2_REFERENCE_OUTPUT_DIR)/q2_decode_reference_0.json
	@test ! -e $(Q2_REFERENCE_OUTPUT_DIR)/q2_prefill_reference_0.json
	@CFLAGS="$(CFLAGS)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" \
		python3 tools/bench_q2_canonical.py \
		--mode reference0 --binary ./tests/q2_canonical_bench \
		--model "$(Q2_CANONICAL_MODEL)" \
		--tokenizer "$(Q2_CANONICAL_TOKENIZER)" \
		--prompt "$(Q2_CANONICAL_PROMPT)" \
		--output-dir "$(Q2_REFERENCE_OUTPUT_DIR)"

bench-q2-decode: tests/q2_canonical_bench
	@CFLAGS="$(CFLAGS)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" \
		python3 tools/bench_q2_canonical.py \
		--mode current-decode --binary ./tests/q2_canonical_bench \
		--model "$(Q2_CANONICAL_MODEL)" \
		--tokenizer "$(Q2_CANONICAL_TOKENIZER)" \
		--prompt "$(Q2_CANONICAL_PROMPT)" \
		--output "$(Q2_CURRENT_OUTPUT_DIR)/q2_decode_canonical.json"

bench-q2-prefill: tests/q2_canonical_bench
	@CFLAGS="$(CFLAGS)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" \
		python3 tools/bench_q2_canonical.py \
		--mode current-prefill --binary ./tests/q2_canonical_bench \
		--model "$(Q2_CANONICAL_MODEL)" \
		--tokenizer "$(Q2_CANONICAL_TOKENIZER)" \
		--prompt "$(Q2_CANONICAL_PROMPT)" \
		--output "$(Q2_CURRENT_OUTPUT_DIR)/q2_prefill_canonical.json"

check-perf-artifacts:
	@python3 tools/check_perf_artifacts.py

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

q38_%.o: cuda/q38_%.cu
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f q38 q38-server q38-cli $(PRODUCTION_OBJS) q38_session.o \
		$(SERVER_OBJS) q38_server_main.o q38_cli.o \
		tests/q2_canonical_bench $(TEST_BINS) \
		tests/test_q38_json tests/test_q38_server_engine \
		tests/test_q38_kvstore \
		tests/test_q38_server_protocol tests/test_q38_server \
		tools/q38_quantize
