/* q38.c — M0 inspection CLI.
 *
 * A single executable that implements --platform, --inspect, --list-tensors,
 * and --memory-plan. No inference path. All output is either human-readable
 * or, with --json, a deterministic JSON document.
 */

#include "q38.h"
#include "q38_cuda.h"
#include "q38_gguf.h"
#include "q38_memory.h"
#include "q38_platform.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *fp) {
    fprintf(fp,
        "usage: q38 <mode> [options]\n"
        "\n"
        "modes:\n"
        "  --platform                 Probe the platform (CUDA + host memory)\n"
        "  --platform-json            Platform probe in JSON format\n"
        "  --inspect <model.gguf>     Print GGUF metadata and tensor summary\n"
        "  --list-tensors <model.gguf> List individual tensors\n"
        "  --memory-plan <model.gguf> Dry-run memory plan (no allocation)\n"
        "\n"
        "options:\n"
        "  --json                     Machine-readable output\n"
        "  --verbose                  Extra diagnostics\n");
}

static void print_bytes_json(uint64_t b) { printf("%" PRIu64, b); }

static void print_platform_human(const q38_platform_info *p) {
    printf("cuda devices:       %d\n", p->cuda_device_count);
    printf("cuda device:        %d\n", p->cuda_device);
    printf("device name:        %s\n", p->device_name);
    printf("compute capability: sm_%d%d\n", p->cc_major, p->cc_minor);
    printf("driver version:     %s\n", p->driver_version[0] ? p->driver_version : "n/a");
    printf("runtime version:    %s\n", p->runtime_version[0] ? p->runtime_version : "n/a");
    printf("cuda total:         ");
    print_bytes_json(p->cuda_total_bytes);
    printf(" bytes\n");
    printf("cuda free:          ");
    print_bytes_json(p->cuda_free_bytes);
    printf(" bytes\n");
    printf("host mem total:     ");
    print_bytes_json(p->mem_total_bytes);
    printf(" bytes\n");
    printf("host mem available: ");
    print_bytes_json(p->mem_available_bytes);
    printf(" bytes\n");
}

static void print_platform_json(const q38_platform_info *p) {
    printf("{\"cuda_device_count\":%d,\"cuda_device\":%d,"
           "\"cc_major\":%d,\"cc_minor\":%d,"
           "\"cuda_total_bytes\":%" PRIu64 ",\"cuda_free_bytes\":%" PRIu64
           ",\"mem_total_bytes\":%" PRIu64 ",\"mem_available_bytes\":%" PRIu64
           ",\"device_name\":\"%s\",\"driver_version\":\"%s\","
           "\"runtime_version\":\"%s\"}\n",
           p->cuda_device_count, p->cuda_device,
           p->cc_major, p->cc_minor,
           p->cuda_total_bytes, p->cuda_free_bytes,
           p->mem_total_bytes, p->mem_available_bytes,
           p->device_name, p->driver_version, p->runtime_version);
}

static int cmd_platform(const q38_options *opt) {
    q38_platform_info p;
    char reason[256];
    if (q38_platform_probe(&p, reason, sizeof(reason)) != 0) {
        fprintf(stderr, "q38: unsupported platform: %s\n", reason);
        return 1;
    }
    if (opt->json) {
        print_platform_json(&p);
    } else {
        print_platform_human(&p);
    }
    return 0;
}

static void model_summary(const q38_gguf *m, uint64_t *tensor_bytes,
                          uint64_t *params) {
    *tensor_bytes = 0;
    *params = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        *tensor_bytes += m->tensors[i].bytes;
        *params += m->tensors[i].elements;
    }
}

static int cmd_inspect(const q38_options *opt) {
    char err[256];
    q38_gguf *m = q38_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "q38: %s\n", err);
        return 1;
    }

    q38_str name = {0}, arch = {0};
    q38_gguf_get_string(m, "general.name", &name);
    q38_gguf_get_string(m, "general.architecture", &arch);

    uint64_t tensor_bytes = 0, params = 0;
    model_summary(m, &tensor_bytes, &params);

    if (opt->json) {
        printf("{\"name\":\"%.*s\",\"architecture\":\"%.*s\","
               "\"version\":%u,\"metadata_keys\":%" PRIu64
               ",\"tensors\":%" PRIu64
               ",\"file_bytes\":%" PRIu64
               ",\"tensor_bytes\":%" PRIu64
               ",\"logical_parameters\":%" PRIu64 "}\n",
               (int)name.len, name.ptr ? name.ptr : "",
               (int)arch.len, arch.ptr ? arch.ptr : "",
               m->version, m->n_kv, m->n_tensors,
               m->size, tensor_bytes, params);
    } else {
        printf("model:     %.*s\n", (int)name.len, name.ptr ? name.ptr : "");
        printf("arch:      %.*s\n", (int)arch.len, arch.ptr ? arch.ptr : "");
        printf("gguf:      v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n",
               m->version, m->n_kv, m->n_tensors);
        printf("file size: %" PRIu64 " bytes\n", m->size);
        printf("tensor bytes: %" PRIu64 "\n", tensor_bytes);
        printf("logical parameters: %" PRIu64 "\n", params);

        printf("tensor types:\n");
        for (uint32_t type = 0; type < 64; type++) {
            uint64_t count = 0, bytes = 0;
            for (uint64_t i = 0; i < m->n_tensors; i++) {
                if (m->tensors[i].type == type) {
                    count++;
                    bytes += m->tensors[i].bytes;
                }
            }
            if (count != 0) {
                printf("  %-8s %5" PRIu64 " tensors, %" PRIu64 " bytes\n",
                       q38_gguf_type_name(type), count, bytes);
            }
        }
    }

    q38_gguf_close(m);
    return 0;
}

static int cmd_list_tensors(const q38_options *opt) {
    char err[256];
    q38_gguf *m = q38_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "q38: %s\n", err);
        return 1;
    }

    if (opt->json) {
        printf("{\"tensors\":[");
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            const q38_tensor *t = &m->tensors[i];
            printf("%s{\"name\":\"%.*s\",\"type\":\"%s\",\"ndim\":%u,"
                   "\"elements\":%" PRIu64 ",\"bytes\":%" PRIu64 "}",
                   i ? "," : "",
                   (int)t->name.len, t->name.ptr,
                   q38_gguf_type_name(t->type),
                   t->ndim, t->elements, t->bytes);
        }
        printf("]}\n");
    } else {
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            const q38_tensor *t = &m->tensors[i];
            printf("%-48.*s %-8s %" PRIu64 " elems %" PRIu64 " bytes\n",
                   (int)t->name.len, t->name.ptr,
                   q38_gguf_type_name(t->type),
                   t->elements, t->bytes);
        }
    }

    q38_gguf_close(m);
    return 0;
}

static int cmd_memory_plan(const q38_options *opt) {
    char err[256];
    q38_gguf *m = q38_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "q38: %s\n", err);
        return 1;
    }

    uint64_t tensor_bytes = 0, params = 0;
    model_summary(m, &tensor_bytes, &params);

    /* Dry run: no cudaMalloc, no host-registration. The mapping is already in
     * place; snapshot RSS + host available + CUDA free/total. */
    q38_memory_tracker tracker;
    q38_memory_tracker_init(&tracker);

    q38_platform_info p;
    char reason[256];
    uint64_t cuda_total = 0, cuda_free = 0;
    if (q38_platform_probe(&p, reason, sizeof(reason)) == 0) {
        cuda_total = p.cuda_total_bytes;
        cuda_free = p.cuda_free_bytes;
    }

    q38_memory_snapshot snap;
    q38_memory_capture(&tracker, "gguf_mapped",
                       m->size, m->size, 0, &snap);
    snap.cuda_total_bytes = cuda_total;
    snap.cuda_free_bytes = cuda_free;

    if (opt->json) {
        char buf[1024];
        q38_memory_snapshot_json(&snap, buf, sizeof(buf));
        printf("%s\n", buf);
    } else {
        printf("model file:        %" PRIu64 " bytes\n", snap.model_file_bytes);
        printf("model mapped:      %" PRIu64 " bytes\n", snap.model_mapped_bytes);
        printf("rss:               %" PRIu64 " bytes\n", snap.rss_bytes);
        printf("mem available:     %" PRIu64 " bytes\n", snap.mem_available_bytes);
        printf("cuda free:         %" PRIu64 " bytes\n", snap.cuda_free_bytes);
        printf("cuda total:        %" PRIu64 " bytes\n", snap.cuda_total_bytes);
        printf("tensor bytes:      %" PRIu64 "\n", tensor_bytes);
        printf("peak internal:     %" PRIu64 " bytes\n", snap.peak_internal_bytes);
    }

    q38_gguf_close(m);
    return 0;
}

int main(int argc, char **argv) {
    q38_options opt;
    memset(&opt, 0, sizeof(opt));

    q38_mode mode = Q38_MODE_NONE;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--platform") == 0 || strcmp(a, "--platform-json") == 0) {
            mode = Q38_MODE_PLATFORM;
            opt.platform = true;
            if (strcmp(a, "--platform-json") == 0) opt.json = true;
        } else if (strcmp(a, "--inspect") == 0) {
            mode = Q38_MODE_INSPECT;
            opt.inspect = true;
            if (i + 1 < argc) opt.model_path = argv[++i];
        } else if (strcmp(a, "--list-tensors") == 0) {
            mode = Q38_MODE_LIST_TENSORS;
            opt.list_tensors = true;
            if (i + 1 < argc) opt.model_path = argv[++i];
        } else if (strcmp(a, "--memory-plan") == 0) {
            mode = Q38_MODE_MEMORY_PLAN;
            opt.memory_plan = true;
            if (i + 1 < argc) opt.model_path = argv[++i];
        } else if (strcmp(a, "--json") == 0) {
            opt.json = true;
        } else if (strcmp(a, "--verbose") == 0) {
            opt.verbose = true;
        } else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "q38: unknown argument '%s'\n", a);
            usage(stderr);
            return 2;
        }
    }

    if (mode == Q38_MODE_NONE) {
        usage(stderr);
        return 2;
    }

    int rc;
    switch (mode) {
    case Q38_MODE_PLATFORM:
        rc = cmd_platform(&opt);
        break;
    case Q38_MODE_INSPECT:
        rc = cmd_inspect(&opt);
        break;
    case Q38_MODE_LIST_TENSORS:
        rc = cmd_list_tensors(&opt);
        break;
    case Q38_MODE_MEMORY_PLAN:
        rc = cmd_memory_plan(&opt);
        break;
    default:
        rc = 2;
        break;
    }

    q38_cuda_cleanup();
    return rc;
}
