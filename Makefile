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
	q38_residency.o q38_directional_steering.o q38_session.o
PRODUCTION_CUDA_OBJS := \
	q38_cuda.o q38_forward_cuda.o q38_qsa_cuda.o q38_cuda_primitives.o \
	q38_gdn.o q38_moe_cuda.o q38_cuda_timing.o q38_profile_cuda.o \
	q38_topk_cuda.o
PRODUCTION_OBJS := $(PRODUCTION_C_OBJS) $(PRODUCTION_CUDA_OBJS)

CANONICAL_BENCH_C_OBJS := \
	q38_gguf.o q38_forward.o q38_ple_prefetch.o q38_weights.o \
	q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_session.o \
	q38_directional_steering.o \
	q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o q38_moe.o q38_decode.o \
	q38_tokenizer.o
CANONICAL_BENCH_CUDA_OBJS := \
	q38_forward_cuda.o q38_qsa_cuda.o q38_cuda_primitives.o q38_gdn.o \
	q38_moe_cuda.o q38_profile_cuda.o q38_residency.o \
	q38_topk_cuda.o

TEST_BINS := \
	tests/test_platform tests/test_gguf tests/test_memory \
	tests/test_model_config tests/test_quant_blocks tests/test_residency

.PHONY: all q38 q38-server q38-server-mock q38-cli spark test test-server clean tools \
	q38-server-real \
	bench-q2-reference-0 bench-q2-decode bench-q2-prefill \
	check-perf-artifacts test-steering test-steering-cuda \
	gr-fixtures gr-bench gr-c1-bench gr-c2-bench gr-c3-bench \
	gr-c3-bundle gr-c4-bench test-gr

all: q38

q38: $(PRODUCTION_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(PRODUCTION_OBJS) $(CUDA_LDLIBS)

SERVER_OBJS := q38_server.o q38_server_protocol.o q38_server_engine_mock.o \
	q38_server_engine.o q38_json.o q38_kvstore.o q38_prompt.o

q38_server.o q38_server_engine.o q38_server_engine_mock.o \
q38_server_protocol.o q38_kvstore.o: q38_server_engine.h
q38_session.o q38_directional_steering.o: q38_directional_steering.h

q38-server-mock: q38_server_main.o $(SERVER_OBJS)
	$(CC) $(CFLAGS) -o $@ q38_server_main.o $(SERVER_OBJS)

SERVER_RUNTIME_C_OBJS := $(filter-out q38.o,$(PRODUCTION_C_OBJS))
SERVER_RUNTIME_OBJS := $(SERVER_RUNTIME_C_OBJS) $(PRODUCTION_CUDA_OBJS)

q38-server-real: q38_server_real_main.o q38_server_engine_q38.o \
		$(SERVER_OBJS) $(SERVER_RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ q38_server_real_main.o \
		q38_server_engine_q38.o $(SERVER_OBJS) $(SERVER_RUNTIME_OBJS) \
		$(CUDA_LDLIBS)

q38-server: q38_server_real_main.o q38_server_engine_q38.o \
		$(SERVER_OBJS) $(SERVER_RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ q38_server_real_main.o \
		q38_server_engine_q38.o $(SERVER_OBJS) $(SERVER_RUNTIME_OBJS) \
		$(CUDA_LDLIBS)

q38-cli: q38_cli.o q38_json.o
	$(CC) $(CFLAGS) -o $@ q38_cli.o q38_json.o

spark: q38 $(TEST_BINS)

test: $(TEST_BINS)
	@set -e; for test in $(TEST_BINS); do ./$$test; done

test-server: tests/test_q38_directional_steering tests/test_q38_json tests/test_q38_kvstore tests/test_q38_server_engine \
		tests/test_q38_server_protocol tests/test_q38_server
	@set -e; \
	./tests/test_q38_directional_steering; \
	./tests/test_q38_json; \
	./tests/test_q38_kvstore; \
	./tests/test_q38_server_engine; \
	./tests/test_q38_server_protocol; \
	./tests/test_q38_server

test-steering: tests/test_q38_directional_steering
	./tests/test_q38_directional_steering

test-steering-cuda: tests/test_q38_directional_steering_cuda
	@set +e; ./tests/test_q38_directional_steering_cuda; status=$$?; \
	if [ $$status -ne 0 ] && [ $$status -ne 2 ]; then exit $$status; fi

tests/test_q38_directional_steering_cuda: tests/test_q38_directional_steering_cuda.o \
		q38_forward_cuda.o q38_directional_steering.o q38_gguf.o \
		q38_moe_cuda.o q38_cuda.o q38_cuda_primitives.o q38_qsa_cuda.o \
		q38_gdn.o q38_cuda_timing.o q38_profile_cuda.o q38_topk_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_q38_directional_steering_cuda.o: tests/test_q38_directional_steering_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

tests/test_q38_directional_steering: tests/test_q38_directional_steering.c \
		q38_directional_steering.o q38_directional_steering.h
	$(CC) $(CFLAGS) -o $@ tests/test_q38_directional_steering.c \
		q38_directional_steering.o -lm

tests/test_q38_json: tests/test_q38_json.c q38_json.o q38_json.h
	$(CC) $(CFLAGS) -o $@ tests/test_q38_json.c q38_json.o

tests/test_q38_kvstore: tests/test_q38_kvstore.c q38_kvstore.o q38_kvstore.h
	$(CC) $(CFLAGS) -o $@ tests/test_q38_kvstore.c q38_kvstore.o -lm

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

GR_FIXTURE_DIR := tests/fixtures/gr
GR_REFERENCE_ARTIFACT := artifacts/perf/subsystems/gr_reference.json
GR_C1_ARTIFACT := artifacts/perf/subsystems/gr_c1_projection.json
GR_BINDING_OBJS := q38_gguf.o q38_weights.o q38_model_config.o q38_ple.o \
	q38_residency.o q38_quant.o q38_qsa.o

tests/gr/test_m3_gr_ref: tests/gr/test_m3_gr_ref.c q38_gr_ref.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

tests/gr/test_m3_gr_binding: tests/gr/test_m3_gr_binding.c $(GR_BINDING_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

tests/gr/test_m3_gr_cuda: tests/gr/test_m3_gr_cuda.o q38_gr.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/gr/test_m3_gr_cuda.o: tests/gr/test_m3_gr_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

tests/gr/gr_reference.o: tests/gr/gr_reference.c tests/gr/gr_reference.h
	$(CC) $(CFLAGS) -c -o $@ $<

tests/gr/gr_extract_fixtures: tests/gr/gr_extract_fixtures.c \
		tests/gr/gr_reference.o $(GR_BINDING_OBJS)
	$(CC) $(CFLAGS) -o $@ tests/gr/gr_extract_fixtures.c \
		tests/gr/gr_reference.o $(GR_BINDING_OBJS) -lm

tests/gr/gr_bench: tests/gr/gr_bench.o tests/gr/gr_reference.o q38_gr.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/gr/gr_bench.o: tests/gr/gr_bench.cu tests/gr/gr_reference.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

tests/gr/gr_c1_bench: tests/gr/gr_c1_bench.o q38_cuda_primitives.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/gr/gr_c1_bench.o: tests/gr/gr_c1_bench.cu q38_cuda_primitives.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

tests/gr/gr_c2_bench: tests/gr/gr_c2_bench.o q38_cuda_primitives.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/gr/gr_c2_bench.o: tests/gr/gr_c2_bench.cu q38_cuda_primitives.h \
		q38_gr_ref.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

tests/gr/gr_c3_bench: tests/gr/gr_c3_bench.o q38_cuda_primitives.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/gr/gr_c3_bench.o: tests/gr/gr_c3_bench.cu q38_cuda_primitives.h \
		q38_gr_ref.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

gr-fixtures: tests/gr/gr_extract_fixtures
	@mkdir -p $(GR_FIXTURE_DIR)
	@./tests/gr/gr_extract_fixtures \
		"$(Q2_CANONICAL_MODEL)" "$(GR_FIXTURE_DIR)"

gr-bench: tests/gr/gr_bench gr-fixtures
	@mkdir -p artifacts/perf/subsystems
	@./tests/gr/gr_bench "$(GR_FIXTURE_DIR)" "$(GR_REFERENCE_ARTIFACT)"

gr-c1-bench: tests/gr/gr_c1_bench gr-fixtures
	@mkdir -p artifacts/perf/subsystems
	@./tests/gr/gr_c1_bench "$(GR_FIXTURE_DIR)" "$(GR_C1_ARTIFACT)"

gr-c2-bench: tests/gr/gr_c2_bench
	@mkdir -p artifacts/perf/subsystems
	@./tests/gr/gr_c2_bench "$(GR_FIXTURE_DIR)" \
		"artifacts/perf/subsystems/gr_c2_bundle.json"

gr-c3-bench: tests/gr/gr_c3_bench
	@mkdir -p artifacts/perf/subsystems
	@./tests/gr/gr_c3_bench "$(GR_FIXTURE_DIR)" \
		"artifacts/perf/subsystems/gr_c3_up_geometry.json"

gr-c3-bundle: tests/gr/gr_c2_bench
	@mkdir -p artifacts/perf/subsystems
	@Q38_GR_C3=1 ./tests/gr/gr_c2_bench "$(GR_FIXTURE_DIR)" \
		"artifacts/perf/subsystems/gr_c3_bundle.json"

gr-c4-bench: tests/gr/gr_c2_bench
	@mkdir -p artifacts/perf/subsystems
	@Q38_GR_C4=1 ./tests/gr/gr_c2_bench "$(GR_FIXTURE_DIR)" \
		"artifacts/perf/subsystems/gr_c4_bundle.json"

test-gr: tests/gr/test_m3_gr_ref tests/gr/test_m3_gr_binding \
		tests/gr/test_m3_gr_cuda gr-bench gr-c1-bench
	@./tests/gr/test_m3_gr_ref
	@./tests/gr/test_m3_gr_binding "$(Q2_CANONICAL_MODEL)"
	@set +e; ./tests/gr/test_m3_gr_cuda; status=$$?; \
	if [ $$status -ne 0 ] && [ $$status -ne 2 ]; then exit $$status; fi

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
	rm -f q38 q38-server q38-server-real q38-server-mock q38-cli $(PRODUCTION_OBJS) q38_session.o \
		$(SERVER_OBJS) q38_server_main.o q38_server_real_main.o \
		q38_server_engine_q38.o q38_cli.o \
		tests/q2_canonical_bench tests/test_q38_directional_steering_cuda \
		tests/test_q38_directional_steering_cuda.o $(TEST_BINS) \
		tests/test_q38_directional_steering tests/test_q38_json tests/test_q38_server_engine \
		tests/test_q38_kvstore \
		tests/test_q38_server_protocol tests/test_q38_server \
		tools/q38_quantize
