# q38 canonical production build and benchmark interface.

CC ?= cc
CFLAGS ?= -O3 -g -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -I. -pthread
MODEL_DIR ?= /home/lvx/q38model
BUILD_DIR ?= build
RELEASE_OBJDIR := $(BUILD_DIR)/release
DIAG_OBJDIR := $(BUILD_DIR)/diag
RELEASE_CFLAGS = $(CFLAGS)
DIAG_CFLAGS = $(CFLAGS) -DQ38_DIAGNOSTICS=1
RELEASE_NVCCFLAGS = $(NVCCFLAGS)
DIAG_NVCCFLAGS = $(NVCCFLAGS) -DQ38_DIAGNOSTICS=1

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
	q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o q38_replay.o \
	q38_residency.o q38_directional_steering.o q38_session.o q38_residency_plan.o
PRODUCTION_CUDA_OBJS := \
	q38_cuda.o q38_forward_cuda.o q38_qsa_cuda.o q38_cuda_primitives.o \
	q38_gdn.o q38_moe_cuda.o q38_cuda_timing.o \
	q38_topk_cuda.o
PRODUCTION_OBJS := $(PRODUCTION_C_OBJS) $(PRODUCTION_CUDA_OBJS)
RELEASE_OBJS := $(addprefix $(RELEASE_OBJDIR)/,$(PRODUCTION_OBJS))
DIAG_OBJS := $(addprefix $(DIAG_OBJDIR)/,$(PRODUCTION_OBJS))

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
S4B_DIAG_C_OBJS := $(addprefix $(DIAG_OBJDIR)/,$(CANONICAL_BENCH_C_OBJS))
S4B_DIAG_CUDA_OBJS := $(addprefix $(DIAG_OBJDIR)/,$(CANONICAL_BENCH_CUDA_OBJS))

TEST_BINS := \
	tests/test_platform tests/test_gguf tests/test_memory \
	tests/test_model_config tests/test_quant_blocks tests/test_residency \
	tests/test_residency_plan

tests/bench_ple_projection_cuda: tests/bench_ple_projection.cu \
		build/release/q38_gdn.o build/release/q38_cuda_primitives.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS) -lm

bench-ple-projection-cuda: tests/bench_ple_projection_cuda

tests/bench_residency_startup: tests/bench_residency_startup.cu \
		q38_residency_plan.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

bench-startup-residency: tests/bench_residency_startup

tests/bench_tokenizer_startup: tests/bench_tokenizer_startup.c \
		q38_tokenizer.o
	$(CC) $(CFLAGS) -Wl,--wrap=malloc -Wl,--wrap=calloc \
		-Wl,--wrap=realloc -Wl,--wrap=strdup -o $@ $^

bench-tokenizer-startup: tests/bench_tokenizer_startup

.PHONY: all q38 q38-diag q38-server q38-server-mock q38-cli q38-dev-worker spark test test-server clean tools \
	q38-server-real \
	bench-ple-projection-cuda \
	bench-startup-residency \
	bench-tokenizer-startup \
	bench-q2-reference-0 bench-q2-decode bench-q2-prefill \
	tests/q2_forward_exclusive_attribution \
	tests/test_s4c_ple_replay \
	test-prod-diag-equivalence \
	check-perf-artifacts test-steering test-steering-cuda \
	gr-fixtures gr-bench gr-c1-bench gr-c2-bench gr-c3-bench \
	gr-c3-bundle gr-c4-bench test-gr test-moe bench-moe \
	gdn-fixtures test-gdn bench-gdn bench-gdn-c1 bench-gdn-c2 bench-gdn-c3 \
	bench-qsa

all: q38

q38: $(RELEASE_OBJS)
	$(NVCC) $(RELEASE_NVCCFLAGS) -o $@ $(RELEASE_OBJS) $(CUDA_LDLIBS)

q38-diag: $(DIAG_OBJS)
	$(NVCC) $(DIAG_NVCCFLAGS) -o $@ $(DIAG_OBJS) $(CUDA_LDLIBS)

SERVER_OBJS := q38_server.o q38_server_protocol.o q38_server_engine_mock.o \
	q38_server_engine.o q38_json.o q38_kvstore.o q38_prompt.o

q38_server.o q38_server_engine.o q38_server_engine_mock.o \
q38_server_protocol.o q38_kvstore.o: q38_server_engine.h
q38_decode.o q38_forward.o q38_session.o: q38_forward.h
q38_decode.o q38_session.o: q38_session.h
q38_forward.o q38_session.o: q38_forward_cuda.h
q38_forward_cuda.o: q38_forward.h q38_forward_cuda.h
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

Q38_WORKER_C_OBJS := $(filter-out q38.o,$(PRODUCTION_C_OBJS))
q38-dev-worker: $(Q38_WORKER_C_OBJS) $(PRODUCTION_CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o q38_dev_worker tests/q38_dev_worker.c \
		$(Q38_WORKER_C_OBJS) $(PRODUCTION_CUDA_OBJS) $(CUDA_LDLIBS) -ldl -lm

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

tests/test_residency_plan: tests/test_residency_plan.c \
		q38_residency_plan.o q38_gguf.h q38_residency_plan.h
	$(CC) $(CFLAGS) -o $@ tests/test_residency_plan.c \
		q38_residency_plan.o

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

MOE_FIXTURE_DIR ?= tests/fixtures/moe
MOE_ARTIFACT := artifacts/perf/subsystems/moe_reference.json
MOE_REFERENCE_OBJS := tests/moe/moe_reference.o q38_quant.o

tests/moe/moe_reference.o: tests/moe/moe_reference.c \
		tests/moe/moe_reference.h q38_quant.h
	$(CC) $(CFLAGS) -c -o $@ $<

tests/moe/test_moe_reference: tests/moe/test_moe_reference.c \
		$(MOE_REFERENCE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

tests/moe/moe_bench: tests/moe/moe_bench.o $(MOE_REFERENCE_OBJS) \
		q38_moe_cuda.o q38_cuda_primitives.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS) -lm

tests/moe/moe_bench.o: tests/moe/moe_bench.cu tests/moe/moe_reference.h \
		q38_moe_cuda.h q38_cuda_primitives.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

test-moe: tests/moe/test_moe_reference
	./tests/moe/test_moe_reference

bench-moe: tests/moe/moe_bench
	@mkdir -p artifacts/perf/subsystems
	@tmp="$(MOE_ARTIFACT).tmp"; rm -f "$$tmp"; \
	./tests/moe/moe_bench "$(MOE_FIXTURE_DIR)" > "$$tmp" && \
	mv "$$tmp" "$(MOE_ARTIFACT)" || { status=$$?; rm -f "$$tmp"; exit $$status; }

GDN_FIXTURE_DIR ?= tests/fixtures/gdn
GDN_ARTIFACT := artifacts/perf/subsystems/gdn_reference.json
GDN_C1_ARTIFACT := artifacts/perf/subsystems/gdn_c1.json

tests/gdn/gdn_reference.o: tests/gdn/gdn_reference.c \
		tests/gdn/gdn_reference.h q38_quant.h
	$(CC) $(CFLAGS) -c -o $@ $<

tests/gdn/gdn_bench: tests/gdn/gdn_bench.o tests/gdn/gdn_reference.o \
		q38_gdn_ref.o q38_quant.o q38_cuda_primitives.o q38_gdn.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS) -lm

tests/gdn/gdn_bench.o: tests/gdn/gdn_bench.cu tests/gdn/gdn_reference.h \
		q38_gdn.h q38_gdn_ref.h q38_cuda_primitives.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

gdn-fixtures: q38-dev-worker
	@mkdir -p $(GDN_FIXTURE_DIR)
	@printf 'CAPTURE_GDN token=220\nQUIT\n' | \
		./q38_dev_worker --model "$(Q2_CANONICAL_MODEL)"

tests/test_m3_gdn_ref: tests/test_m3_gdn_ref.c q38_gdn_ref.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

test-gdn: tests/test_m3_gdn_ref
	./tests/test_m3_gdn_ref

bench-gdn: tests/gdn/gdn_bench
	@test -d "$(GDN_FIXTURE_DIR)/early" && \
		test -d "$(GDN_FIXTURE_DIR)/middle" && \
		test -d "$(GDN_FIXTURE_DIR)/late" || \
		{ echo "GDN fixtures missing; run 'make gdn-fixtures' once"; exit 1; }
	@mkdir -p artifacts/perf/subsystems
	@tmp="$(GDN_ARTIFACT).tmp"; rm -f "$$tmp"; \
	./tests/gdn/gdn_bench "$(GDN_FIXTURE_DIR)" "$$tmp" && \
	mv "$$tmp" "$(GDN_ARTIFACT)" || { status=$$?; rm -f "$$tmp"; exit $$status; }

bench-gdn-c1: tests/gdn/gdn_bench
	@test -d "$(GDN_FIXTURE_DIR)/early" && \
		test -d "$(GDN_FIXTURE_DIR)/middle" && \
		test -d "$(GDN_FIXTURE_DIR)/late" || \
		{ echo "GDN fixtures missing; run 'make gdn-fixtures' once"; exit 1; }
	@mkdir -p artifacts/perf/subsystems
	@tmp="$(GDN_C1_ARTIFACT).tmp"; rm -f "$$tmp"; \
	./tests/gdn/gdn_bench "$(GDN_FIXTURE_DIR)" "$$tmp" gdn_c1 && \
	mv "$$tmp" "$(GDN_C1_ARTIFACT)" || { status=$$?; rm -f "$$tmp"; exit $$status; }

bench-gdn-c2: tests/gdn/gdn_bench
	@test -d "$(GDN_FIXTURE_DIR)/early" && \
		test -d "$(GDN_FIXTURE_DIR)/middle" && \
		test -d "$(GDN_FIXTURE_DIR)/late" || \
		{ echo "GDN fixtures missing; run 'make gdn-fixtures' once"; exit 1; }
	@mkdir -p artifacts/perf/subsystems
	@tmp="artifacts/perf/subsystems/gdn_c2.json.tmp"; rm -f "$$tmp"; \
	./tests/gdn/gdn_bench "$(GDN_FIXTURE_DIR)" "$$tmp" gdn_c2 && \
	mv "$$tmp" artifacts/perf/subsystems/gdn_c2.json || \
		{ status=$$?; rm -f "$$tmp"; exit $$status; }

bench-gdn-c3: tests/gdn/gdn_bench
	@test -d "$(GDN_FIXTURE_DIR)/early" && \
		test -d "$(GDN_FIXTURE_DIR)/middle" && \
		test -d "$(GDN_FIXTURE_DIR)/late" || \
		{ echo "GDN fixtures missing; run 'make gdn-fixtures' once"; exit 1; }
	@mkdir -p artifacts/perf/subsystems
	@tmp="artifacts/perf/subsystems/gdn_c3.json.tmp"; rm -f "$$tmp"; \
	./tests/gdn/gdn_bench "$(GDN_FIXTURE_DIR)" "$$tmp" gdn_c3 && \
	mv "$$tmp" artifacts/perf/subsystems/gdn_c3.json || \
		{ status=$$?; rm -f "$$tmp"; exit $$status; }

tests/qsa/qsa_reference.o: tests/qsa/qsa_reference.c tests/qsa/qsa_reference.h
	$(CC) $(CFLAGS) -c -o $@ $<

tests/qsa/qsa_bench.o: tests/qsa/qsa_bench.cu tests/qsa/qsa_reference.h \
	q38_qsa_cuda.h q38_cuda_primitives.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

tests/qsa/qsa_bench: tests/qsa/qsa_bench.o tests/qsa/qsa_reference.o \
	q38_qsa_cuda.o q38_cuda_primitives.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

bench-qsa: tests/qsa/qsa_bench
	@test -d tests/fixtures/qsa/early && \
	test -d tests/fixtures/qsa/middle && \
	test -d tests/fixtures/qsa/late || \
	{ echo "QSA fixtures missing; run one authorized CAPTURE_QSA load"; exit 1; }
	@mkdir -p artifacts/perf/subsystems
	@tmp="artifacts/perf/subsystems/qsa_reference.json.tmp"; rm -f "$$tmp"; \
	./tests/qsa/qsa_bench tests/fixtures/qsa "$$tmp" && \
	mv "$$tmp" artifacts/perf/subsystems/qsa_reference.json || \
	{ status=$$?; rm -f "$$tmp"; exit $$status; }

tests/q2_canonical_bench: tests/q2_canonical_bench.c \
		$(CANONICAL_BENCH_C_OBJS) $(CANONICAL_BENCH_CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/q2_canonical_bench.c \
		$(CANONICAL_BENCH_C_OBJS) $(CANONICAL_BENCH_CUDA_OBJS) \
		$(CUDA_LDLIBS) -lm

tests/q2_forward_exclusive_attribution: tests/q2_canonical_bench.c \
		$(S4B_DIAG_C_OBJS) $(S4B_DIAG_CUDA_OBJS)
	$(NVCC) $(DIAG_NVCCFLAGS) -o $@ tests/q2_canonical_bench.c \
		$(S4B_DIAG_C_OBJS) $(S4B_DIAG_CUDA_OBJS) \
		$(CUDA_LDLIBS) -lm

tests/test_s4c_ple_replay: tests/test_s4c_ple_replay.c \
		q38_gguf.o q38_ple.o q38_ple_prefetch.o
	$(CC) $(CFLAGS) -o $@ $^ -pthread

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

test-prod-diag-equivalence: q38 q38-diag \
		$(RELEASE_OBJDIR)/q38_gr_ref.o $(DIAG_OBJDIR)/q38_gr_ref.o \
		$(RELEASE_OBJDIR)/q38_gdn_ref.o $(DIAG_OBJDIR)/q38_gdn_ref.o \
		$(RELEASE_OBJDIR)/q38_quant.o $(DIAG_OBJDIR)/q38_quant.o
	@set -e; \
	mkdir -p $(RELEASE_OBJDIR)/equivalence $(DIAG_OBJDIR)/equivalence; \
	$(CC) $(RELEASE_CFLAGS) -o $(RELEASE_OBJDIR)/equivalence/test_m3_gr_ref \
		tests/gr/test_m3_gr_ref.c $(RELEASE_OBJDIR)/q38_gr_ref.o -lm; \
	$(CC) $(DIAG_CFLAGS) -o $(DIAG_OBJDIR)/equivalence/test_m3_gr_ref \
		tests/gr/test_m3_gr_ref.c $(DIAG_OBJDIR)/q38_gr_ref.o -lm; \
	$(CC) $(RELEASE_CFLAGS) -o $(RELEASE_OBJDIR)/equivalence/test_m3_gdn_ref \
		tests/test_m3_gdn_ref.c $(RELEASE_OBJDIR)/q38_gdn_ref.o -lm; \
	$(CC) $(DIAG_CFLAGS) -o $(DIAG_OBJDIR)/equivalence/test_m3_gdn_ref \
		tests/test_m3_gdn_ref.c $(DIAG_OBJDIR)/q38_gdn_ref.o -lm; \
	$(CC) $(RELEASE_CFLAGS) -o $(RELEASE_OBJDIR)/equivalence/test_moe_reference \
		tests/moe/test_moe_reference.c tests/moe/moe_reference.c \
		$(RELEASE_OBJDIR)/q38_quant.o -lm; \
	$(CC) $(DIAG_CFLAGS) -o $(DIAG_OBJDIR)/equivalence/test_moe_reference \
		tests/moe/test_moe_reference.c tests/moe/moe_reference.c \
		$(DIAG_OBJDIR)/q38_quant.o -lm; \
	for test in test_m3_gr_ref test_m3_gdn_ref test_moe_reference; do \
		prod="$$(./$(RELEASE_OBJDIR)/equivalence/$$test)"; \
		diag="$$(./$(DIAG_OBJDIR)/equivalence/$$test)"; \
		test "$$prod" = "$$diag"; \
		printf '%s\n' "$$prod"; \
	done

$(RELEASE_OBJDIR) $(DIAG_OBJDIR):
	mkdir -p $@

$(RELEASE_OBJDIR)/%.o: %.c | $(RELEASE_OBJDIR)
	$(CC) $(RELEASE_CFLAGS) -c -o $@ $<

$(DIAG_OBJDIR)/%.o: %.c | $(DIAG_OBJDIR)
	$(CC) $(DIAG_CFLAGS) -c -o $@ $<

$(RELEASE_OBJDIR)/q38_%.o: cuda/q38_%.cu | $(RELEASE_OBJDIR)
	@echo "q38: release nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(RELEASE_NVCCFLAGS) -c -o $@ $<

$(DIAG_OBJDIR)/q38_%.o: cuda/q38_%.cu | $(DIAG_OBJDIR)
	@echo "q38: diag nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(DIAG_NVCCFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

q38_%.o: cuda/q38_%.cu
	@echo "q38: nvcc arch flags: $(NVCC_ARCH_FLAGS)"
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)
	rm -f q38 q38-diag q38-server q38-server-real q38-server-mock q38-cli $(PRODUCTION_OBJS) q38_session.o \
		$(SERVER_OBJS) q38_server_main.o q38_server_real_main.o \
		q38_server_engine_q38.o q38_cli.o \
		tests/q2_canonical_bench tests/test_q38_directional_steering_cuda \
		tests/test_q38_directional_steering_cuda.o $(TEST_BINS) \
		tests/test_q38_directional_steering tests/test_q38_json tests/test_q38_server_engine \
		tests/test_q38_kvstore \
		tests/test_q38_server_protocol tests/test_q38_server \
		tests/moe/test_moe_reference tests/moe/moe_bench \
		tools/q38_quantize
