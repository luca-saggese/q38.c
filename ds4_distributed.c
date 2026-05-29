#include "ds4_distributed.h"

#include <arpa/inet.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define DS4_DIST_MAGIC 0x44533444u /* DS4D */
#define DS4_DIST_MSG_HELLO 1u
#define DS4_DIST_MSG_ERROR 2u
#define DS4_DIST_MSG_WORK 3u
#define DS4_DIST_MSG_RESULT 4u
#define DS4_DIST_MAX_MODEL_NAME 127u
#define DS4_DIST_WORK_F_INPUT_HC 0x00000001u
#define DS4_DIST_WORK_F_OUTPUT_LOGITS 0x00000002u
#define DS4_DIST_WORK_F_RESET_SESSION 0x00000004u
#define DS4_DIST_WORK_F_ACK_ONLY 0x00000008u
#define DS4_DIST_WORK_F_VALID_MASK \
    (DS4_DIST_WORK_F_INPUT_HC | DS4_DIST_WORK_F_OUTPUT_LOGITS | \
     DS4_DIST_WORK_F_RESET_SESSION | DS4_DIST_WORK_F_ACK_ONLY)
#define DS4_DIST_RESULT_ACK 0u
#define DS4_DIST_RESULT_HIDDEN_STATE 1u
#define DS4_DIST_RESULT_LOGITS 2u
#define DS4_DIST_ACTIVATION_BITS_DEFAULT 32u
#define DS4_DIST_ROUTE_F_OUTPUT_LOGITS 0x00000001u
#define DS4_DIST_ROUTE_RETURN_UPSTREAM 1u
#define DS4_DIST_RECV_TRANSPORT_ERROR 1
#define DS4_DIST_RECV_REMOTE_ERROR 2

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;
} ds4_dist_frame_header;

typedef struct {
    uint32_t model_id;
    uint32_t quant_bits;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t has_output;
    uint32_t has_hidden;
    uint32_t ctx_size;
    uint32_t n_layers;
    uint32_t listen_port;
    uint32_t model_name_len;
} ds4_dist_hello_fixed;

typedef struct {
    uint32_t model_id;
    uint32_t session_hi;
    uint32_t session_lo;
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t prefix_hash_hi;
    uint32_t prefix_hash_lo;
    uint32_t result_hash_hi;
    uint32_t result_hash_lo;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
    uint32_t token_bytes;
    uint32_t input_hc_bytes;
    uint32_t input_hc_bits;
    uint32_t route_count;
    uint32_t route_index;
    uint32_t route_bytes;
} ds4_dist_work_fixed;

typedef struct {
    uint32_t host_len;
    uint32_t port;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
} ds4_dist_route_fixed;

typedef struct {
    uint32_t kind;
    uint32_t host_len;
    uint32_t port;
} ds4_dist_route_return_fixed;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t result_hash_hi;
    uint32_t result_hash_lo;
    uint32_t status;
    uint32_t result_kind;
    uint32_t telemetry_count;
    uint32_t telemetry_bytes;
    uint32_t payload_bytes;
    uint32_t payload_bits;
} ds4_dist_result_fixed;

typedef struct {
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t route_index;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t eval_usec;
    uint32_t downstream_wait_usec;
    uint32_t forward_send_usec;
    uint32_t input_bytes;
    uint32_t output_bytes;
} ds4_dist_telemetry_fixed;

typedef struct ds4_dist_worker_entry {
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    char model_name[DS4_DIST_MAX_MODEL_NAME + 1u];
    uint32_t model_id;
    uint32_t quant_bits;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t has_output;
    uint32_t has_hidden;
    uint32_t ctx_size;
    uint32_t n_layers;
    uint32_t listen_port;
    struct ds4_dist_worker_entry *next;
} ds4_dist_worker_entry;

typedef struct {
    ds4_engine *engine;
    uint32_t model_id;
    uint32_t n_layers;
    uint32_t local_start;
    uint32_t local_end;
    uint32_t ctx_size;
    bool local_has_output;
    bool local_can_output_head;
    bool replay_check;
    bool debug;
    bool use_control_for_work;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    uint64_t generation;
    pthread_mutex_t mu;
    ds4_dist_worker_entry *workers;
} ds4_dist_coordinator_state;

typedef struct {
    ds4_dist_coordinator_state *state;
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
} ds4_dist_client_ctx;

typedef struct {
    ds4_dist_coordinator_state *state;
    int listen_fd;
} ds4_dist_accept_ctx;

typedef struct ds4_dist_worker_session {
    uint64_t session_id;
    uint64_t token_hash;
    bool token_hash_valid;
    ds4_session *session;
    struct ds4_dist_worker_session *next;
} ds4_dist_worker_session;

typedef struct {
    ds4_engine *engine;
    uint32_t model_id;
    uint32_t layer_start;
    uint32_t layer_end;
    bool has_output;
    int ctx_size;
    int listen_fd;
    pthread_mutex_t mu;
    ds4_dist_worker_session *sessions;
} ds4_dist_worker_state;

typedef struct ds4_dist_worker_upstream ds4_dist_worker_upstream;

typedef struct ds4_dist_pending_request {
    uint64_t request_id;
    double downstream_t0;
    ds4_dist_telemetry_fixed telemetry;
    struct ds4_dist_pending_request *next;
} ds4_dist_pending_request;

typedef struct ds4_dist_worker_forwarder {
    ds4_dist_worker_upstream *upstream;
    char host[NI_MAXHOST];
    uint32_t port;
    int fd;
    pthread_t tid;
    bool thread_started;
    pthread_mutex_t send_mu;
    pthread_mutex_t queue_mu;
    pthread_cond_t queue_not_full;
    ds4_dist_pending_request *pending_head;
    ds4_dist_pending_request *pending_tail;
    uint32_t pending_count;
    uint32_t pending_depth;
    bool closing;
    struct ds4_dist_worker_forwarder *next;
} ds4_dist_worker_forwarder;

struct ds4_dist_worker_upstream {
    ds4_dist_worker_state *state;
    int fd;
    pthread_mutex_t write_mu;
    pthread_mutex_t forward_mu;
    ds4_dist_worker_forwarder *forwarders;
};

typedef struct ds4_dist_worker_job {
    void *payload;
    uint32_t bytes;
    struct ds4_dist_worker_job *next;
} ds4_dist_worker_job;

typedef struct {
    ds4_dist_worker_state *state;
    ds4_dist_worker_upstream *upstream;
    pthread_mutex_t mu;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    ds4_dist_worker_job *head;
    ds4_dist_worker_job *tail;
    uint32_t queued;
    uint32_t depth;
    bool closed;
    bool canceled;
    int rc;
} ds4_dist_worker_job_queue;

typedef struct {
    ds4_dist_worker_state *state;
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
} ds4_dist_data_client_ctx;

typedef struct {
    char host[NI_MAXHOST];
    uint32_t port;
    uint32_t kind;
} ds4_dist_route_return;

typedef struct {
    char host[NI_MAXHOST];
    uint32_t port;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
    int fd;
} ds4_dist_route_entry;

typedef struct {
    ds4_dist_route_entry *entry;
    uint32_t count;
    void *blob;
    uint32_t blob_bytes;
} ds4_dist_route_plan;

struct ds4_dist_session {
    ds4_dist_coordinator_state state;
    int listen_fd;
    pthread_t accept_tid;
    bool accept_started;
    ds4_dist_accept_ctx accept_ctx;
    ds4_dist_route_plan plan;
    bool plan_ready;
    uint64_t plan_generation;
    uint64_t session_id;
    uint64_t request_id;
};

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_dist_logprob;

typedef struct {
    ds4_dist_coordinator_state *state;
    int fd;
    ds4_session *progress_session;
    uint64_t first_request_id;
    uint64_t *expected_hashes;
    uint32_t count;
    uint32_t total_tokens;
    uint32_t chunk_cap;
    uint32_t progress_base;
    uint32_t progress_total;
    uint32_t progress_completed;
    bool progress_done;
    uint64_t hc_values;
    bool allow_hidden;
    uint32_t final_kind;
    void *final_payload;
    uint32_t final_payload_bytes;
    int rc;
    char err[256];
    pthread_mutex_t progress_mu;
    pthread_cond_t progress_cv;
} ds4_dist_prefill_result_reader;

typedef struct {
    uint32_t pos;
    uint32_t n_tokens;
    uint32_t hidden_bytes;
    uint64_t request_id;
    uint64_t prefix_hash;
    uint64_t result_hash;
    bool reset_session;
    bool ack_only;
    float *hidden;
} ds4_dist_prefill_send_slot;

typedef struct {
    ds4_dist_coordinator_state *state;
    const ds4_dist_route_plan *plan;
    const ds4_tokens *prompt;
    uint64_t session_id;
    int fd;
    ds4_dist_prefill_send_slot *slots;
    uint32_t slot_count;
    uint32_t head;
    uint32_t tail;
    uint32_t queued;
    bool producer_done;
    bool stop;
    int rc;
    double send_sec;
    uint64_t send_bytes;
    char err[256];
    pthread_mutex_t mu;
    pthread_cond_t can_enqueue;
    pthread_cond_t can_dequeue;
} ds4_dist_prefill_sender;

static uint32_t dist_prefill_send_depth(uint32_t chunk_count) {
    uint32_t depth = 2;
    const char *env = getenv("DS4_DIST_PREFILL_SEND_DEPTH");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 8) {
            depth = (uint32_t)v;
        }
    }
    if (chunk_count != 0 && depth > chunk_count) depth = chunk_count;
    return depth ? depth : 1;
}

static int dist_send_work_frame(
        int fd,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *input_hc,
        const void *route_blob);
static int dist_write_full(int fd, const void *buf, size_t len);

static int dist_worker_handle_work(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes);
static void dist_worker_upstream_init(
        ds4_dist_worker_upstream *upstream,
        ds4_dist_worker_state *state,
        int fd);
static void dist_worker_upstream_destroy(ds4_dist_worker_upstream *upstream);
static int dist_worker_upstream_send_work_error(
        ds4_dist_worker_upstream *upstream,
        uint64_t request_id,
        const char *msg);
static int dist_coordinator_prefill_prompt(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        char *err,
        size_t errlen);
static int dist_validate_options(const ds4_dist_options *opt, char *err, size_t errlen);

static uint32_t dist_resolved_layer_end(const ds4_dist_options *opt, uint32_t n_layers) {
    if (opt->layers.has_output) return n_layers - 1u;
    return opt->layers.end;
}

static const char *dist_role_name(ds4_distributed_role role) {
    switch (role) {
    case DS4_DISTRIBUTED_NONE:        return "none";
    case DS4_DISTRIBUTED_COORDINATOR: return "coordinator";
    case DS4_DISTRIBUTED_WORKER:      return "worker";
    }
    return "unknown";
}

static void dist_sleep_reconnect(void) {
    sleep(1);
}

static double dist_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int dist_socket_buffer_bytes(void) {
    int mb = 128;
    const char *env = getenv("DS4_DIST_SOCKET_BUFFER_MB");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 0 && v <= 512) {
            mb = (int)v;
        }
    }
    return mb > 0 ? mb * 1024 * 1024 : 0;
}

static uint32_t dist_worker_prefetch_depth(void) {
    uint32_t depth = 2;
    const char *env = getenv("DS4_DIST_WORKER_PREFETCH_DEPTH");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 8) {
            depth = (uint32_t)v;
        }
    }
    return depth;
}

static uint32_t dist_worker_forward_window(void) {
    uint32_t depth = 4;
    const char *env = getenv("DS4_DIST_WORKER_FORWARD_WINDOW");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 64) {
            depth = (uint32_t)v;
        }
    }
    return depth;
}

static bool dist_parse_positive_u32(
        const char *s,
        const char *name,
        uint32_t *out,
        char *err,
        size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "%s requires a positive integer", name);
        return false;
    }
    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || s[0] == '\0' || *end != '\0' || v == 0 || v > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "invalid value for %s: %s", name, s);
        return false;
    }
    *out = (uint32_t)v;
    return true;
}

static uint32_t dist_activation_bits_or_default(uint32_t bits) {
    return bits ? bits : DS4_DIST_ACTIVATION_BITS_DEFAULT;
}

static bool dist_activation_bits_valid(uint32_t bits) {
    bits = dist_activation_bits_or_default(bits);
    return bits == 32u || bits == 16u || bits == 8u;
}

static bool dist_activation_wire_bytes(uint32_t bits, uint64_t values, uint32_t *out) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits) || (bits % 8u) != 0) return false;
    const uint64_t bytes = values * (uint64_t)(bits / 8u);
    if (bytes > UINT32_MAX) return false;
    if (out) *out = (uint32_t)bytes;
    return true;
}

static bool dist_activation_values_from_wire_bytes(uint32_t bits, uint32_t bytes, uint64_t *out) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits) || (bits % 8u) != 0) return false;
    const uint32_t bytes_per_value = bits / 8u;
    if (bytes_per_value == 0 || (bytes % bytes_per_value) != 0) return false;
    if (out) *out = bytes / bytes_per_value;
    return true;
}

static bool dist_activation_wire_bytes_from_f32_bytes(uint32_t bits, uint32_t f32_bytes, uint32_t *out) {
    if ((f32_bytes % (uint32_t)sizeof(float)) != 0) return false;
    return dist_activation_wire_bytes(bits, f32_bytes / (uint32_t)sizeof(float), out);
}

static uint16_t dist_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }

    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

static float dist_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t exp = (int32_t)((h >> 10) & 0x1fu);
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint8_t dist_f32_to_f8_e4m3(float f) {
    const uint8_t sign = signbit(f) ? 0x80u : 0u;
    float a = fabsf(f);
    if (a == 0.0f) return sign;
    if (!isfinite(a) || a >= 240.0f) return (uint8_t)(sign | 0x77u);

    if (a < 0.001953125f) {
        int mant = (int)floorf(a * 512.0f + 0.5f);
        if (mant <= 0) return sign;
        if (mant > 7) mant = 7;
        return (uint8_t)(sign | (uint8_t)mant);
    }

    int exp2 = 0;
    (void)frexpf(a, &exp2);
    int exp = exp2 - 1 + 7;
    if (exp <= 0) {
        int mant = (int)floorf(a * 512.0f + 0.5f);
        if (mant <= 0) return sign;
        if (mant > 7) mant = 7;
        return (uint8_t)(sign | (uint8_t)mant);
    }

    float base = ldexpf(1.0f, exp2 - 1);
    int mant = (int)floorf(((a / base) - 1.0f) * 8.0f + 0.5f);
    if (mant >= 8) {
        mant = 0;
        exp++;
    }
    if (exp >= 15) return (uint8_t)(sign | 0x77u);
    return (uint8_t)(sign | (uint8_t)(exp << 3) | (uint8_t)mant);
}

static float dist_f8_e4m3_to_f32(uint8_t h) {
    const float sign = (h & 0x80u) ? -1.0f : 1.0f;
    const uint32_t exp = (h >> 3) & 0x0fu;
    const uint32_t mant = h & 0x07u;
    if (exp == 0) {
        return sign * (float)mant * 0.001953125f;
    }
    if (exp >= 15u) {
        return sign * 240.0f;
    }
    return sign * ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
}

static int dist_write_activation_payload(
        int fd,
        const float *src,
        uint64_t values,
        uint32_t bits) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits)) return -1;
    if (values == 0) return 0;
    if (!src) return -1;
    if (bits == 32u) {
        uint32_t bytes = 0;
        if (!dist_activation_wire_bytes(bits, values, &bytes)) return -1;
        return dist_write_full(fd, src, bytes);
    }

    const uint64_t max_values = 1024u * 1024u;
    uint64_t cap = values < max_values ? values : max_values;
    void *buf = malloc((size_t)cap * (size_t)(bits / 8u));
    if (!buf) return -1;
    uint64_t done = 0;
    int rc = 0;
    while (done < values) {
        uint64_t n = values - done;
        if (n > cap) n = cap;
        if (bits == 16u) {
            uint16_t *dst = buf;
            for (uint64_t i = 0; i < n; i++) dst[i] = dist_f32_to_f16(src[done + i]);
        } else {
            uint8_t *dst = buf;
            for (uint64_t i = 0; i < n; i++) dst[i] = dist_f32_to_f8_e4m3(src[done + i]);
        }
        if (dist_write_full(fd, buf, (size_t)n * (size_t)(bits / 8u)) != 0) {
            rc = -1;
            break;
        }
        done += n;
    }
    free(buf);
    return rc;
}

static int dist_decode_activation_payload(
        const void *wire,
        uint32_t bits,
        uint32_t wire_bytes,
        float **out,
        uint32_t *out_f32_bytes,
        bool *out_uses_wire,
        char *err,
        size_t errlen) {
    if (out) *out = NULL;
    if (out_f32_bytes) *out_f32_bytes = 0;
    if (out_uses_wire) *out_uses_wire = false;
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits)) {
        if (errlen) snprintf(err, errlen, "invalid distributed activation width: %u bits", bits);
        return 1;
    }
    if (wire_bytes != 0 && !wire) {
        if (errlen) snprintf(err, errlen, "missing distributed activation payload");
        return 1;
    }

    uint64_t values = 0;
    if (!dist_activation_values_from_wire_bytes(bits, wire_bytes, &values)) {
        if (errlen) snprintf(err, errlen, "invalid distributed activation payload size");
        return 1;
    }
    const uint64_t f32_bytes64 = values * sizeof(float);
    if (f32_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed activation payload is too large");
        return 1;
    }
    const uint32_t f32_bytes = (uint32_t)f32_bytes64;
    if (bits == 32u) {
        if (out) *out = (float *)(void *)wire;
        if (out_f32_bytes) *out_f32_bytes = f32_bytes;
        if (out_uses_wire) *out_uses_wire = true;
        return 0;
    }

    float *dst = f32_bytes ? malloc(f32_bytes) : NULL;
    if (f32_bytes && !dst) {
        if (errlen) snprintf(err, errlen, "out of memory decoding distributed activations");
        return 1;
    }
    if (bits == 16u) {
        const uint16_t *src = wire;
        for (uint64_t i = 0; i < values; i++) dst[i] = dist_f16_to_f32(src[i]);
    } else {
        const uint8_t *src = wire;
        for (uint64_t i = 0; i < values; i++) dst[i] = dist_f8_e4m3_to_f32(src[i]);
    }
    if (out) *out = dst;
    if (out_f32_bytes) *out_f32_bytes = f32_bytes;
    return 0;
}

static int dist_set_socket_low_latency(int fd) {
    int one = 1;
    int rc = 0;
    int buffer_bytes = dist_socket_buffer_bytes();
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) rc = -1;
    if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one)) != 0) rc = -1;
    if (buffer_bytes > 0 &&
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0) rc = -1;
    if (buffer_bytes > 0 &&
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0) rc = -1;
#ifdef SO_NOSIGPIPE
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) != 0) rc = -1;
#endif
    return rc;
}

#ifdef DS4_DIST_TRACE
#define DIST_DEBUG(...) do { \
    fprintf(stderr, "ds4: distributed debug: " __VA_ARGS__); \
    fputc('\n', stderr); \
} while (0)
#else
#define DIST_DEBUG(...) ((void)0)
#endif

static int dist_write_full(int fd, const void *buf, size_t len) {
    const unsigned char *p = buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int dist_read_full(int fd, void *buf, size_t len) {
    unsigned char *p = buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return 0;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

static int dist_write_frame_header(int fd, uint32_t type, uint32_t bytes) {
    ds4_dist_frame_header h = {
        htonl(DS4_DIST_MAGIC),
        htonl(type),
        htonl(bytes)
    };
    return dist_write_full(fd, &h, sizeof(h));
}

static int dist_read_frame_header(int fd, uint32_t *type, uint32_t *bytes, char *err, size_t errlen) {
    ds4_dist_frame_header h;
    int rc = dist_read_full(fd, &h, sizeof(h));
    if (rc <= 0) return rc;

    uint32_t magic = ntohl(h.magic);
    if (magic != DS4_DIST_MAGIC) {
        if (errlen) snprintf(err, errlen, "bad frame magic 0x%08x", magic);
        return -1;
    }

    *type = ntohl(h.type);
    *bytes = ntohl(h.bytes);
    return 1;
}

static int dist_discard_bytes(int fd, uint32_t bytes) {
    unsigned char buf[4096];
    while (bytes > 0) {
        size_t n = bytes < sizeof(buf) ? bytes : sizeof(buf);
        int rc = dist_read_full(fd, buf, n);
        if (rc <= 0) return rc == 0 ? 0 : -1;
        bytes -= (uint32_t)n;
    }
    return 1;
}

static int dist_send_error(int fd, const char *msg) {
    if (!msg) msg = "distributed protocol error";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    if (dist_write_frame_header(fd, DS4_DIST_MSG_ERROR, (uint32_t)len) != 0) return -1;
    return dist_write_full(fd, msg, len);
}

static void dist_peer_name(int fd, char *host, size_t hostlen, char *port, size_t portlen) {
    if (hostlen) host[0] = '\0';
    if (portlen) port[0] = '\0';

    struct sockaddr_storage ss;
    socklen_t slen = sizeof(ss);
    if (getpeername(fd, (struct sockaddr *)&ss, &slen) == 0) {
        if (getnameinfo((struct sockaddr *)&ss, slen,
                        host, (socklen_t)hostlen,
                        port, (socklen_t)portlen,
                        NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
            return;
        }
    }
    if (hostlen) snprintf(host, hostlen, "unknown");
    if (portlen) snprintf(port, portlen, "0");
}

static int dist_open_listener(const char *host, int port, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    const char *host_display = host ? host : "*";

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, portbuf, &hints, &res);
    if (gai != 0) {
        if (errlen) snprintf(err, errlen, "getaddrinfo(%s:%s): %s", host_display, portbuf, gai_strerror(gai));
        return -1;
    }

    int listen_fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;

        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        dist_set_socket_low_latency(fd);

        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 &&
            listen(fd, 64) == 0) {
            listen_fd = fd;
            break;
        }

        close(fd);
    }

    freeaddrinfo(res);
    if (listen_fd < 0 && errlen) {
        snprintf(err, errlen, "unable to listen on %s:%d: %s", host_display, port, strerror(errno));
    }
    return listen_fd;
}

static int dist_listener_port(int fd) {
    struct sockaddr_storage ss;
    socklen_t slen = sizeof(ss);
    if (getsockname(fd, (struct sockaddr *)&ss, &slen) != 0) return 0;
    char service[NI_MAXSERV];
    if (getnameinfo((struct sockaddr *)&ss, slen,
                    NULL, 0,
                    service, sizeof(service),
                    NI_NUMERICSERV) != 0) {
        return 0;
    }
    char *end = NULL;
    unsigned long v = strtoul(service, &end, 10);
    if (end == service || *end != '\0' || v > 65535ul) return 0;
    return (int)v;
}

static bool dist_connect_errno_retryable(int e) {
    return e == ECONNREFUSED ||
           e == EHOSTUNREACH ||
           e == ENETUNREACH ||
           e == ETIMEDOUT ||
           e == EADDRNOTAVAIL;
}

static int dist_connect_endpoint_once(const char *host, int port, int *last_errno, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    if (last_errno) *last_errno = 0;

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, portbuf, &hints, &res);
    if (gai != 0) {
        if (errlen) snprintf(err, errlen, "getaddrinfo(%s:%s): %s", host, portbuf, gai_strerror(gai));
        if (last_errno) *last_errno = EINVAL;
        return -1;
    }

    int fd = -1;
    int saved_errno = 0;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            saved_errno = errno;
            continue;
        }
        dist_set_socket_low_latency(fd);
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        saved_errno = errno;
        close(fd);
        fd = -1;
    }

    freeaddrinfo(res);
    if (fd < 0 && errlen) {
        if (saved_errno == 0) saved_errno = EIO;
        snprintf(err, errlen, "unable to connect to %s:%d: %s", host, port, strerror(saved_errno));
    }
    if (fd < 0 && last_errno) *last_errno = saved_errno ? saved_errno : EIO;
    return fd;
}

static int dist_connect_endpoint(const char *host, int port, char *err, size_t errlen) {
    int last_errno = 0;
    for (int attempt = 0; attempt < 200; attempt++) {
        int fd = dist_connect_endpoint_once(host, port, &last_errno, err, errlen);
        if (fd >= 0) return fd;
        if (!dist_connect_errno_retryable(last_errno)) break;
        struct timespec ts = {0, 25 * 1000 * 1000};
        nanosleep(&ts, NULL);
    }
    return -1;
}

static void dist_hello_to_wire(ds4_dist_hello_fixed *h) {
    h->model_id = htonl(h->model_id);
    h->quant_bits = htonl(h->quant_bits);
    h->layer_start = htonl(h->layer_start);
    h->layer_end = htonl(h->layer_end);
    h->has_output = htonl(h->has_output);
    h->has_hidden = htonl(h->has_hidden);
    h->ctx_size = htonl(h->ctx_size);
    h->n_layers = htonl(h->n_layers);
    h->listen_port = htonl(h->listen_port);
    h->model_name_len = htonl(h->model_name_len);
}

static void dist_hello_from_wire(ds4_dist_hello_fixed *h) {
    h->model_id = ntohl(h->model_id);
    h->quant_bits = ntohl(h->quant_bits);
    h->layer_start = ntohl(h->layer_start);
    h->layer_end = ntohl(h->layer_end);
    h->has_output = ntohl(h->has_output);
    h->has_hidden = ntohl(h->has_hidden);
    h->ctx_size = ntohl(h->ctx_size);
    h->n_layers = ntohl(h->n_layers);
    h->listen_port = ntohl(h->listen_port);
    h->model_name_len = ntohl(h->model_name_len);
}

static uint64_t dist_u64_from_halves(uint32_t hi, uint32_t lo) {
    return ((uint64_t)hi << 32) | lo;
}

/* FNV-1a over little-endian token IDs.  This is not a security primitive; it is
 * a compact session invariant so distributed workers can reject same-position
 * but different-prefix KV state before doing layer work. */
#define DS4_DIST_TOKEN_HASH_INIT 1469598103934665603ull
#define DS4_DIST_TOKEN_HASH_PRIME 1099511628211ull

static uint64_t dist_token_hash_update(uint64_t h, int token) {
    uint32_t t = (uint32_t)token;
    for (int i = 0; i < 4; i++) {
        h ^= (uint64_t)((t >> (i * 8)) & 0xffu);
        h *= DS4_DIST_TOKEN_HASH_PRIME;
    }
    return h;
}

static uint64_t dist_token_hash_update_span(uint64_t h, const int *tokens, uint32_t n_tokens) {
    for (uint32_t i = 0; i < n_tokens; i++) h = dist_token_hash_update(h, tokens[i]);
    return h;
}

static uint64_t dist_token_hash_prefix(const int *tokens, uint32_t n_tokens) {
    return dist_token_hash_update_span(DS4_DIST_TOKEN_HASH_INIT, tokens, n_tokens);
}

static int dist_session_token_hash_prefix(
        ds4_session *session,
        uint32_t n_tokens,
        uint64_t *hash,
        char *err,
        size_t errlen) {
    const ds4_tokens *checkpoint = ds4_session_tokens(session);
    if (!hash || !checkpoint || checkpoint->len < 0 || (uint32_t)checkpoint->len < n_tokens) {
        if (errlen) snprintf(err, errlen, "distributed session has no %u-token prefix", n_tokens);
        return 1;
    }
    *hash = dist_token_hash_prefix(checkpoint->v, n_tokens);
    return 0;
}

static bool dist_bytes_have_nul(const void *p, uint32_t len) {
    return len != 0 && memchr(p, '\0', len) != NULL;
}

static void dist_u64_to_halves(uint64_t v, uint32_t *hi, uint32_t *lo) {
    *hi = (uint32_t)(v >> 32);
    *lo = (uint32_t)v;
}

static void dist_work_from_wire(ds4_dist_work_fixed *w) {
    w->model_id = ntohl(w->model_id);
    w->session_hi = ntohl(w->session_hi);
    w->session_lo = ntohl(w->session_lo);
    w->request_hi = ntohl(w->request_hi);
    w->request_lo = ntohl(w->request_lo);
    w->prefix_hash_hi = ntohl(w->prefix_hash_hi);
    w->prefix_hash_lo = ntohl(w->prefix_hash_lo);
    w->result_hash_hi = ntohl(w->result_hash_hi);
    w->result_hash_lo = ntohl(w->result_hash_lo);
    w->pos0 = ntohl(w->pos0);
    w->n_tokens = ntohl(w->n_tokens);
    w->layer_start = ntohl(w->layer_start);
    w->layer_end = ntohl(w->layer_end);
    w->flags = ntohl(w->flags);
    w->token_bytes = ntohl(w->token_bytes);
    w->input_hc_bytes = ntohl(w->input_hc_bytes);
    w->input_hc_bits = ntohl(w->input_hc_bits);
    w->route_count = ntohl(w->route_count);
    w->route_index = ntohl(w->route_index);
    w->route_bytes = ntohl(w->route_bytes);
}

static void dist_work_to_wire(ds4_dist_work_fixed *w) {
    w->model_id = htonl(w->model_id);
    w->session_hi = htonl(w->session_hi);
    w->session_lo = htonl(w->session_lo);
    w->request_hi = htonl(w->request_hi);
    w->request_lo = htonl(w->request_lo);
    w->prefix_hash_hi = htonl(w->prefix_hash_hi);
    w->prefix_hash_lo = htonl(w->prefix_hash_lo);
    w->result_hash_hi = htonl(w->result_hash_hi);
    w->result_hash_lo = htonl(w->result_hash_lo);
    w->pos0 = htonl(w->pos0);
    w->n_tokens = htonl(w->n_tokens);
    w->layer_start = htonl(w->layer_start);
    w->layer_end = htonl(w->layer_end);
    w->flags = htonl(w->flags);
    w->token_bytes = htonl(w->token_bytes);
    w->input_hc_bytes = htonl(w->input_hc_bytes);
    w->input_hc_bits = htonl(w->input_hc_bits);
    w->route_count = htonl(w->route_count);
    w->route_index = htonl(w->route_index);
    w->route_bytes = htonl(w->route_bytes);
}

static void dist_route_from_wire(ds4_dist_route_fixed *r) {
    r->host_len = ntohl(r->host_len);
    r->port = ntohl(r->port);
    r->layer_start = ntohl(r->layer_start);
    r->layer_end = ntohl(r->layer_end);
    r->flags = ntohl(r->flags);
}

static void dist_route_to_wire(ds4_dist_route_fixed *r) {
    r->host_len = htonl(r->host_len);
    r->port = htonl(r->port);
    r->layer_start = htonl(r->layer_start);
    r->layer_end = htonl(r->layer_end);
    r->flags = htonl(r->flags);
}

static void dist_route_return_from_wire(ds4_dist_route_return_fixed *r) {
    r->kind = ntohl(r->kind);
    r->host_len = ntohl(r->host_len);
    r->port = ntohl(r->port);
}

static void dist_route_return_to_wire(ds4_dist_route_return_fixed *r) {
    r->kind = htonl(r->kind);
    r->host_len = htonl(r->host_len);
    r->port = htonl(r->port);
}

static void dist_result_to_wire(ds4_dist_result_fixed *r) {
    r->request_hi = htonl(r->request_hi);
    r->request_lo = htonl(r->request_lo);
    r->result_hash_hi = htonl(r->result_hash_hi);
    r->result_hash_lo = htonl(r->result_hash_lo);
    r->status = htonl(r->status);
    r->result_kind = htonl(r->result_kind);
    r->telemetry_count = htonl(r->telemetry_count);
    r->telemetry_bytes = htonl(r->telemetry_bytes);
    r->payload_bytes = htonl(r->payload_bytes);
    r->payload_bits = htonl(r->payload_bits);
}

static void dist_result_from_wire(ds4_dist_result_fixed *r) {
    r->request_hi = ntohl(r->request_hi);
    r->request_lo = ntohl(r->request_lo);
    r->result_hash_hi = ntohl(r->result_hash_hi);
    r->result_hash_lo = ntohl(r->result_hash_lo);
    r->status = ntohl(r->status);
    r->result_kind = ntohl(r->result_kind);
    r->telemetry_count = ntohl(r->telemetry_count);
    r->telemetry_bytes = ntohl(r->telemetry_bytes);
    r->payload_bytes = ntohl(r->payload_bytes);
    r->payload_bits = ntohl(r->payload_bits);
}

static void dist_telemetry_to_wire(ds4_dist_telemetry_fixed *t) {
    t->layer_start = htonl(t->layer_start);
    t->layer_end = htonl(t->layer_end);
    t->route_index = htonl(t->route_index);
    t->pos0 = htonl(t->pos0);
    t->n_tokens = htonl(t->n_tokens);
    t->eval_usec = htonl(t->eval_usec);
    t->downstream_wait_usec = htonl(t->downstream_wait_usec);
    t->forward_send_usec = htonl(t->forward_send_usec);
    t->input_bytes = htonl(t->input_bytes);
    t->output_bytes = htonl(t->output_bytes);
}

static void dist_telemetry_from_wire(ds4_dist_telemetry_fixed *t) {
    t->layer_start = ntohl(t->layer_start);
    t->layer_end = ntohl(t->layer_end);
    t->route_index = ntohl(t->route_index);
    t->pos0 = ntohl(t->pos0);
    t->n_tokens = ntohl(t->n_tokens);
    t->eval_usec = ntohl(t->eval_usec);
    t->downstream_wait_usec = ntohl(t->downstream_wait_usec);
    t->forward_send_usec = ntohl(t->forward_send_usec);
    t->input_bytes = ntohl(t->input_bytes);
    t->output_bytes = ntohl(t->output_bytes);
}

static uint32_t dist_usec_since(double t0, double t1) {
    if (t1 <= t0) return 0;
    const double usec = (t1 - t0) * 1000000.0;
    if (usec >= (double)UINT32_MAX) return UINT32_MAX;
    return (uint32_t)(usec + 0.5);
}

static int dist_send_hello(ds4_engine *engine, const ds4_dist_options *opt, int ctx_size, uint32_t listen_port, int fd) {
    uint32_t n_layers = (uint32_t)ds4_engine_layer_count(engine);
    const char *model_name = ds4_engine_model_name(engine);
    if (!model_name) model_name = "unknown";
    size_t model_name_len = strlen(model_name);
    if (model_name_len > DS4_DIST_MAX_MODEL_NAME) model_name_len = DS4_DIST_MAX_MODEL_NAME;

    ds4_dist_hello_fixed h = {
        (uint32_t)ds4_engine_model_id(engine),
        (uint32_t)ds4_engine_routed_quant_bits(engine),
        opt->layers.start,
        dist_resolved_layer_end(opt, n_layers),
        opt->layers.has_output ? 1u : 0u,
        1u,
        ctx_size > 0 ? (uint32_t)ctx_size : 0u,
        n_layers,
        listen_port,
        (uint32_t)model_name_len
    };
    ds4_dist_hello_fixed wire = h;
    dist_hello_to_wire(&wire);

    uint32_t bytes = (uint32_t)sizeof(wire) + (uint32_t)model_name_len;
    if (dist_write_frame_header(fd, DS4_DIST_MSG_HELLO, bytes) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    if (model_name_len && dist_write_full(fd, model_name, model_name_len) != 0) return -1;
    return 0;
}

static int dist_recv_hello(int fd, ds4_dist_hello_fixed *hello, char *model_name, size_t model_name_cap, char *err, size_t errlen) {
    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) return rc;
    if (type != DS4_DIST_MSG_HELLO) {
        if (errlen) snprintf(err, errlen, "expected HELLO frame, got type %u", type);
        dist_discard_bytes(fd, bytes);
        return -1;
    }
    if (bytes < sizeof(*hello) || bytes > sizeof(*hello) + DS4_DIST_MAX_MODEL_NAME) {
        if (errlen) snprintf(err, errlen, "invalid HELLO payload length %u", bytes);
        dist_discard_bytes(fd, bytes);
        return -1;
    }

    ds4_dist_hello_fixed wire;
    rc = dist_read_full(fd, &wire, sizeof(wire));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    dist_hello_from_wire(&wire);

    uint32_t remaining = bytes - (uint32_t)sizeof(wire);
    if (wire.model_name_len != remaining || wire.model_name_len > DS4_DIST_MAX_MODEL_NAME) {
        if (errlen) snprintf(err, errlen, "invalid HELLO model name length %u", wire.model_name_len);
        dist_discard_bytes(fd, remaining);
        return -1;
    }

    if (model_name_cap) model_name[0] = '\0';
    if (wire.model_name_len) {
        char tmp[DS4_DIST_MAX_MODEL_NAME + 1u];
        rc = dist_read_full(fd, tmp, wire.model_name_len);
        if (rc <= 0) return rc == 0 ? 0 : -1;
        if (dist_bytes_have_nul(tmp, wire.model_name_len)) {
            if (errlen) snprintf(err, errlen, "HELLO model family contains NUL bytes");
            return -1;
        }
        tmp[wire.model_name_len] = '\0';
        if (model_name_cap) {
            snprintf(model_name, model_name_cap, "%s", tmp);
        }
    }

    *hello = wire;
    return 1;
}

static void dist_coordinator_report_plan(ds4_dist_coordinator_state *state);

static bool dist_coordinator_debug_enabled(const ds4_dist_coordinator_state *state) {
    return state && state->debug;
}

#define DIST_COORD_DEBUG(state, ...) do { \
    if (dist_coordinator_debug_enabled(state)) fprintf(stderr, __VA_ARGS__); \
} while (0)

static void dist_coordinator_add_worker(
        ds4_dist_coordinator_state *state,
        int fd,
        const char *peer_host,
        const char *peer_port,
        const ds4_dist_hello_fixed *hello,
        const char *model_name) {
    ds4_dist_worker_entry *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: out of memory while registering worker\n");
        return;
    }

    entry->fd = fd;
    snprintf(entry->peer_host, sizeof(entry->peer_host), "%s", peer_host);
    snprintf(entry->peer_port, sizeof(entry->peer_port), "%s", peer_port);
    snprintf(entry->model_name, sizeof(entry->model_name), "%s", model_name ? model_name : "unknown");
    entry->model_id = hello->model_id;
    entry->quant_bits = hello->quant_bits;
    entry->layer_start = hello->layer_start;
    entry->layer_end = hello->layer_end;
    entry->has_output = hello->has_output;
    entry->has_hidden = hello->has_hidden;
    entry->ctx_size = hello->ctx_size;
    entry->n_layers = hello->n_layers;
    entry->listen_port = hello->listen_port;

    pthread_mutex_lock(&state->mu);
    ds4_dist_worker_entry **link = &state->workers;
    while (*link) {
        ds4_dist_worker_entry *old = *link;
        if (strcmp(old->peer_host, peer_host) == 0 &&
            old->model_id == hello->model_id &&
            old->layer_start == hello->layer_start &&
            old->layer_end == hello->layer_end &&
            old->has_output == hello->has_output)
        {
            *link = old->next;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: dropped stale worker %s:%s layers=%u:%u%s\n",
                             old->peer_host,
                             old->peer_port,
                             old->layer_start,
                             old->layer_end,
                             old->has_output ? "+output" : "");
            free(old);
            continue;
        }
        link = &old->next;
    }
    entry->next = state->workers;
    state->workers = entry;
    state->generation++;
    pthread_mutex_unlock(&state->mu);

    char layer_end[32];
    if (entry->has_output) snprintf(layer_end, sizeof(layer_end), "output");
    else snprintf(layer_end, sizeof(layer_end), "%u", entry->layer_end);
    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: registered worker %s:%s data_port=%u model_id=%u quant=Q%u layers=%u:%s hidden=%u ctx=%u\n",
                     entry->peer_host,
                     entry->peer_port,
                     entry->listen_port,
                     entry->model_id,
                     entry->quant_bits,
                     entry->layer_start,
                     layer_end,
                     entry->has_hidden,
                     entry->ctx_size);
    if (dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
}

static int dist_worker_route_cmp(const void *a, const void *b) {
    const ds4_dist_worker_entry *ea = *(const ds4_dist_worker_entry * const *)a;
    const ds4_dist_worker_entry *eb = *(const ds4_dist_worker_entry * const *)b;
    if (ea->layer_start < eb->layer_start) return -1;
    if (ea->layer_start > eb->layer_start) return 1;
    if (ea->has_output != eb->has_output) return ea->has_output ? -1 : 1;
    if (ea->layer_end > eb->layer_end) return -1;
    if (ea->layer_end < eb->layer_end) return 1;
    return 0;
}

static bool dist_worker_route_candidate_ok(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_worker_entry *w,
        uint32_t last) {
    const bool needs_hidden = w->layer_end < last || !w->has_output;
    if (needs_hidden && !w->has_hidden) return false;
    if (w->layer_end >= last && !w->has_output && !state->local_can_output_head) return false;
    return true;
}

static bool dist_route_search_workers(
        const ds4_dist_coordinator_state *state,
        ds4_dist_worker_entry **workers,
        uint32_t n,
        uint32_t next,
        uint32_t last,
        ds4_dist_worker_entry **path,
        uint32_t *path_len,
        uint32_t *missing_layer) {
    bool saw_start = false;
    for (uint32_t i = 0; i < n; i++) {
        ds4_dist_worker_entry *w = workers[i];
        if (w->layer_start < next) continue;
        if (w->layer_start > next) break;
        saw_start = true;
        if (!dist_worker_route_candidate_ok(state, w, last)) continue;

        path[(*path_len)++] = w;
        if (w->layer_end >= last) return true;
        uint32_t child_missing = w->layer_end + 1u;
        if (dist_route_search_workers(state,
                                      workers,
                                      n,
                                      child_missing,
                                      last,
                                      path,
                                      path_len,
                                      &child_missing)) {
            return true;
        }
        if (child_missing > *missing_layer) *missing_layer = child_missing;
        (*path_len)--;
    }
    if (!saw_start && next > *missing_layer) *missing_layer = next;
    return false;
}

static void dist_coordinator_report_plan(ds4_dist_coordinator_state *state) {
    if (!dist_coordinator_debug_enabled(state)) return;
    pthread_mutex_lock(&state->mu);
    uint32_t n = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) n++;
    ds4_dist_worker_entry **workers = n ? calloc(n, sizeof(workers[0])) : NULL;
    ds4_dist_worker_entry **path = n ? calloc(n, sizeof(path[0])) : NULL;
    if ((n && !workers) || (n && !path)) {
        free(workers);
        free(path);
        pthread_mutex_unlock(&state->mu);
        fprintf(stderr, "ds4: distributed coordinator: out of memory building route plan\n");
        return;
    }
    uint32_t i = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) workers[i++] = it;
    qsort(workers, n, sizeof(workers[0]), dist_worker_route_cmp);

    const uint32_t last = state->n_layers - 1u;
    bool complete = state->local_start == 0;
    bool has_output = state->local_end == last &&
                      (state->local_has_output || state->local_can_output_head);
    uint32_t next = state->local_end + 1u;
    if (state->local_end >= last) next = state->n_layers;
    uint32_t path_len = 0;
    uint32_t missing = next;
    if (complete && !has_output) {
        complete = dist_route_search_workers(state,
                                             workers,
                                             n,
                                             next,
                                             last,
                                             path,
                                             &path_len,
                                             &missing);
        if (complete && path_len != 0) {
            ds4_dist_worker_entry *final = path[path_len - 1u];
            has_output = final->has_output || state->local_can_output_head;
            next = state->n_layers;
        }
    }

    char plan[1024];
    size_t used = 0;
    char local_end[32];
    if (state->local_has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", state->local_end);
    used += (size_t)snprintf(plan + used, used < sizeof(plan) ? sizeof(plan) - used : 0,
                             "local %u:%s",
                             state->local_start,
                             local_end);
    for (i = 0; i < path_len; i++) {
        ds4_dist_worker_entry *w = path[i];
        if (used < sizeof(plan)) {
            char end[32];
            if (w->has_output) snprintf(end, sizeof(end), "output");
            else snprintf(end, sizeof(end), "%u", w->layer_end);
            used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                     " -> %s:%u Q%u %u:%s",
                                     w->peer_host,
                                     w->listen_port,
                                     w->quant_bits,
                                     w->layer_start,
                                     end);
        }
    }
    if (complete && path_len != 0 &&
        !path[path_len - 1u]->has_output && state->local_can_output_head &&
        used < sizeof(plan)) {
        used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                 " -> local output");
    }
    if (complete && path_len == 0 &&
        state->local_end == last && !state->local_has_output &&
        state->local_can_output_head && used < sizeof(plan)) {
        used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                 " -> local output");
    }
    complete = complete && has_output && next == state->n_layers;
    pthread_mutex_unlock(&state->mu);

    if (complete) {
        fprintf(stderr, "ds4: distributed coordinator: complete route ready: %s\n", plan);
    } else {
        fprintf(stderr, "ds4: distributed coordinator: route incomplete; next needed layer %u\n", missing);
    }
    free(path);
    free(workers);
}

static void dist_route_plan_free(ds4_dist_route_plan *plan) {
    if (!plan) return;
    for (uint32_t i = 0; i < plan->count; i++) {
        if (plan->entry[i].fd >= 0) close(plan->entry[i].fd);
    }
    free(plan->entry);
    free(plan->blob);
    memset(plan, 0, sizeof(*plan));
}

static bool dist_route_entry_matches_worker(
        const ds4_dist_route_entry *route,
        const ds4_dist_worker_entry *worker) {
    const bool route_has_output = (route->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0;
    return route->port == worker->listen_port &&
           strcmp(route->host, worker->peer_host) == 0 &&
           route->layer_start == worker->layer_start &&
           route->layer_end == worker->layer_end &&
           route_has_output == (worker->has_output != 0);
}

static void dist_coordinator_forget_route_workers(
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan) {
    bool removed_any = false;
    pthread_mutex_lock(&state->mu);
    for (uint32_t i = 0; i < plan->count; i++) {
        ds4_dist_worker_entry **link = &state->workers;
        while (*link) {
            ds4_dist_worker_entry *entry = *link;
            if (!dist_route_entry_matches_worker(&plan->entry[i], entry)) {
                link = &entry->next;
                continue;
            }
            *link = entry->next;
            close(entry->fd);
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: forgot failed route worker %s:%u layers=%u:%u%s\n",
                             plan->entry[i].host,
                             plan->entry[i].port,
                             entry->layer_start,
                             entry->layer_end,
                             entry->has_output ? "+output" : "");
            free(entry);
            removed_any = true;
            break;
        }
    }
    if (removed_any) state->generation++;
    pthread_mutex_unlock(&state->mu);

    if (removed_any && dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
}

static bool dist_route_plan_append_blob(
        ds4_dist_route_plan *plan,
        const ds4_dist_route_entry *entry,
        char *err,
        size_t errlen) {
    const size_t host_len = strlen(entry->host);
    if (host_len == 0 || host_len >= NI_MAXHOST) {
        if (errlen) snprintf(err, errlen, "invalid route host");
        return false;
    }
    const uint64_t add = sizeof(ds4_dist_route_fixed) + host_len;
    if (add > UINT32_MAX || plan->blob_bytes > UINT32_MAX - (uint32_t)add) {
        if (errlen) snprintf(err, errlen, "route payload is too large");
        return false;
    }
    uint32_t old_bytes = plan->blob_bytes;
    uint32_t new_bytes = old_bytes + (uint32_t)add;
    void *new_blob = realloc(plan->blob, new_bytes);
    if (!new_blob) {
        if (errlen) snprintf(err, errlen, "out of memory building route payload");
        return false;
    }
    plan->blob = new_blob;
    uint8_t *p = (uint8_t *)plan->blob + old_bytes;
    ds4_dist_route_fixed fixed = {
        (uint32_t)host_len,
        entry->port,
        entry->layer_start,
        entry->layer_end,
        entry->flags,
    };
    dist_route_to_wire(&fixed);
    memcpy(p, &fixed, sizeof(fixed));
    memcpy(p + sizeof(fixed), entry->host, host_len);
    plan->blob_bytes = new_bytes;
    return true;
}

static bool dist_route_plan_append_return_upstream(
        ds4_dist_route_plan *plan,
        char *err,
        size_t errlen) {
    const uint64_t add = sizeof(ds4_dist_route_return_fixed);
    if (plan->blob_bytes > UINT32_MAX - (uint32_t)add) {
        if (errlen) snprintf(err, errlen, "route payload is too large");
        return false;
    }
    const uint32_t old_bytes = plan->blob_bytes;
    const uint32_t new_bytes = old_bytes + (uint32_t)add;
    void *new_blob = realloc(plan->blob, new_bytes);
    if (!new_blob) {
        if (errlen) snprintf(err, errlen, "out of memory building route payload");
        return false;
    }
    plan->blob = new_blob;
    ds4_dist_route_return_fixed fixed = {
        DS4_DIST_ROUTE_RETURN_UPSTREAM,
        0,
        0,
    };
    dist_route_return_to_wire(&fixed);
    memcpy((uint8_t *)plan->blob + old_bytes, &fixed, sizeof(fixed));
    plan->blob_bytes = new_bytes;
    return true;
}

static bool dist_coordinator_build_route_plan(
        ds4_dist_coordinator_state *state,
        ds4_dist_route_plan *plan,
        uint64_t *generation,
        char *err,
        size_t errlen) {
    memset(plan, 0, sizeof(*plan));
    if (generation) *generation = 0;

    pthread_mutex_lock(&state->mu);
    uint32_t n = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) n++;
    ds4_dist_worker_entry **workers = n ? calloc(n, sizeof(workers[0])) : NULL;
    ds4_dist_worker_entry **path = n ? calloc(n, sizeof(path[0])) : NULL;
    if ((n && !workers) || (n && !path)) {
        free(workers);
        free(path);
        pthread_mutex_unlock(&state->mu);
        if (errlen) snprintf(err, errlen, "out of memory building route");
        return false;
    }
    uint32_t i = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) workers[i++] = it;
    qsort(workers, n, sizeof(workers[0]), dist_worker_route_cmp);

    const uint32_t last = state->n_layers - 1u;
    if (state->local_start != 0) {
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        if (errlen) snprintf(err, errlen, "coordinator route does not start at layer 0");
        return false;
    }
    if (state->local_end == last &&
        (state->local_has_output || state->local_can_output_head)) {
        if (generation) *generation = state->generation;
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        return true;
    }

    uint32_t next = state->local_end + 1u;
    uint32_t path_len = 0;
    uint32_t missing = next;
    if (!dist_route_search_workers(state,
                                   workers,
                                   n,
                                   next,
                                   last,
                                   path,
                                   &path_len,
                                   &missing)) {
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        if (errlen) snprintf(err, errlen, "distributed route incomplete: missing layer %u", missing);
        return false;
    }

    for (i = 0; i < path_len; i++) {
        ds4_dist_worker_entry *w = path[i];
        ds4_dist_route_entry entry;
        memset(&entry, 0, sizeof(entry));
        entry.fd = -1;
        snprintf(entry.host, sizeof(entry.host), "%s", w->peer_host);
        entry.port = w->listen_port;
        entry.layer_start = w->layer_start;
        entry.layer_end = w->layer_end;
        entry.flags = w->has_output ? DS4_DIST_ROUTE_F_OUTPUT_LOGITS : 0u;
        if (state->use_control_for_work && plan->count == 0) {
            entry.fd = dup(w->fd);
            if (entry.fd < 0) {
                pthread_mutex_unlock(&state->mu);
                free(workers);
                free(path);
                dist_route_plan_free(plan);
                if (errlen) snprintf(err, errlen, "failed to duplicate first-hop worker connection: %s", strerror(errno));
                return false;
            }
            dist_set_socket_low_latency(entry.fd);
        }

        ds4_dist_route_entry *new_entries = realloc(plan->entry, (size_t)(plan->count + 1u) * sizeof(plan->entry[0]));
        if (!new_entries) {
            pthread_mutex_unlock(&state->mu);
            free(workers);
            free(path);
            if (entry.fd >= 0) close(entry.fd);
            dist_route_plan_free(plan);
            if (errlen) snprintf(err, errlen, "out of memory building route entries");
            return false;
        }
        plan->entry = new_entries;
        plan->entry[plan->count++] = entry;
        if (!dist_route_plan_append_blob(plan, &entry, err, errlen)) {
            pthread_mutex_unlock(&state->mu);
            free(workers);
            free(path);
            dist_route_plan_free(plan);
            return false;
        }
    }
    if (generation) *generation = state->generation;
    pthread_mutex_unlock(&state->mu);
    free(workers);
    free(path);
    if (plan->count != 0 && !dist_route_plan_append_return_upstream(plan, err, errlen)) {
        dist_route_plan_free(plan);
        return false;
    }
    return true;
}

static int dist_logits_argmax(const float *logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

static bool dist_coordinator_ensure_route(
        ds4_dist_coordinator_state *state,
        ds4_dist_route_plan *plan,
        uint64_t *generation,
        char *err,
        size_t errlen) {
    return dist_coordinator_build_route_plan(state, plan, generation, err, errlen);
}

static uint64_t dist_coordinator_generation(ds4_dist_coordinator_state *state) {
    if (!state) return 0;
    pthread_mutex_lock(&state->mu);
    uint64_t generation = state->generation;
    pthread_mutex_unlock(&state->mu);
    return generation;
}

static int dist_recv_result_alloc(
        int fd,
        const ds4_dist_coordinator_state *state,
        uint64_t request_id,
        uint32_t *kind,
        uint64_t *result_hash,
        void **payload,
        uint32_t *payload_bytes,
        char *err,
        size_t errlen) {
    *payload = NULL;
    *payload_bytes = 0;
    *kind = 0;
    if (result_hash) *result_hash = 0;

    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed connection");
        return 1;
    }
    if (type != DS4_DIST_MSG_RESULT || bytes < sizeof(ds4_dist_result_fixed)) {
        dist_discard_bytes(fd, bytes);
        if (errlen) snprintf(err, errlen, "distributed worker returned invalid frame");
        return 1;
    }

    ds4_dist_result_fixed result;
    rc = dist_read_full(fd, &result, sizeof(result));
    if (rc <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read distributed result");
        return 1;
    }
    dist_result_from_wire(&result);
    const uint64_t got_request = dist_u64_from_halves(result.request_hi, result.request_lo);
    const uint64_t got_hash = dist_u64_from_halves(result.result_hash_hi,
                                                  result.result_hash_lo);
    const uint32_t body_bytes = bytes - (uint32_t)sizeof(result);
    if (result.telemetry_bytes % (uint32_t)sizeof(ds4_dist_telemetry_fixed) != 0 ||
        result.telemetry_count != result.telemetry_bytes / (uint32_t)sizeof(ds4_dist_telemetry_fixed) ||
        result.telemetry_bytes > body_bytes ||
        result.payload_bytes != body_bytes - result.telemetry_bytes) {
        dist_discard_bytes(fd, body_bytes);
        if (errlen) snprintf(err, errlen, "distributed result telemetry metadata mismatch");
        return 1;
    }
    if (got_request != request_id) {
        dist_discard_bytes(fd, bytes - (uint32_t)sizeof(result));
        if (errlen) snprintf(err, errlen, "distributed result metadata mismatch");
        return 1;
    }

    if (result.telemetry_bytes != 0) {
        if (dist_coordinator_debug_enabled(state)) {
            ds4_dist_telemetry_fixed *telemetry = malloc(result.telemetry_bytes);
            if (!telemetry) {
                dist_discard_bytes(fd, result.telemetry_bytes);
                if (errlen) snprintf(err, errlen, "out of memory reading distributed telemetry");
                return 1;
            }
            rc = dist_read_full(fd, telemetry, result.telemetry_bytes);
            if (rc <= 0) {
                free(telemetry);
                if (errlen) snprintf(err, errlen, "failed to read distributed result telemetry");
                return 1;
            }
            for (uint32_t i = 0; i < result.telemetry_count; i++) {
                dist_telemetry_from_wire(&telemetry[i]);
                DIST_COORD_DEBUG(state,
                                 "ds4: distributed telemetry: request=%llu hop=%u layers=%u:%u route=%u pos=%u tokens=%u eval=%.3fms downstream_wait=%.3fms forward_send=%.3fms input=%.2fMiB output=%.2fMiB\n",
                                 (unsigned long long)got_request,
                                 i,
                                 telemetry[i].layer_start,
                                 telemetry[i].layer_end,
                                 telemetry[i].route_index,
                                 telemetry[i].pos0,
                                 telemetry[i].n_tokens,
                                 (double)telemetry[i].eval_usec / 1000.0,
                                 (double)telemetry[i].downstream_wait_usec / 1000.0,
                                 (double)telemetry[i].forward_send_usec / 1000.0,
                                 (double)telemetry[i].input_bytes / (1024.0 * 1024.0),
                                 (double)telemetry[i].output_bytes / (1024.0 * 1024.0));
            }
            free(telemetry);
        } else if (dist_discard_bytes(fd, result.telemetry_bytes) <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed result telemetry");
            return 1;
        }
    }

    void *buf = NULL;
    if (result.payload_bytes != 0) {
        buf = malloc(result.payload_bytes);
        if (!buf) {
            dist_discard_bytes(fd, result.payload_bytes);
            if (errlen) snprintf(err, errlen, "out of memory reading distributed result");
            return 1;
        }
        rc = dist_read_full(fd, buf, result.payload_bytes);
        if (rc <= 0) {
            free(buf);
            if (errlen) snprintf(err, errlen, "failed to read distributed result payload");
            return 1;
        }
    }

    if (result.status != 0) {
        if (errlen) {
            if (buf && result.payload_bytes) {
                size_t n = result.payload_bytes < errlen - 1 ? result.payload_bytes : errlen - 1;
                memcpy(err, buf, n);
                err[n] = '\0';
            } else {
                snprintf(err, errlen, "distributed worker returned an error");
            }
        }
        free(buf);
        return DS4_DIST_RECV_REMOTE_ERROR;
    }

    if (result.result_kind == DS4_DIST_RESULT_HIDDEN_STATE && result.payload_bytes != 0) {
        float *decoded = NULL;
        uint32_t decoded_bytes = 0;
        bool uses_wire = false;
        if (dist_decode_activation_payload(buf,
                                           result.payload_bits,
                                           result.payload_bytes,
                                           &decoded,
                                           &decoded_bytes,
                                           &uses_wire,
                                           err,
                                           errlen) != 0) {
            free(buf);
            return 1;
        }
        if (!uses_wire) {
            free(buf);
            buf = decoded;
        }
        result.payload_bytes = decoded_bytes;
    }

    *kind = result.result_kind;
    if (result_hash) *result_hash = got_hash;
    *payload = buf;
    *payload_bytes = result.payload_bytes;
    return 0;
}

static int dist_coordinator_send_remote_work_on_fd(
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        int fd,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        uint64_t prefix_hash,
        uint64_t result_hash,
        bool reset_session,
        bool ack_only,
        const float *hidden_hc,
        uint32_t hidden_hc_bytes,
        char *err,
        size_t errlen) {
    if (plan->count == 0) {
        if (errlen) snprintf(err, errlen, "distributed route has no remote worker");
        return 1;
    }
    const ds4_dist_route_entry *first = &plan->entry[0];

    ds4_dist_work_fixed work;
    memset(&work, 0, sizeof(work));
    work.model_id = state->model_id;
    dist_u64_to_halves(session_id, &work.session_hi, &work.session_lo);
    dist_u64_to_halves(request_id, &work.request_hi, &work.request_lo);
    dist_u64_to_halves(prefix_hash, &work.prefix_hash_hi, &work.prefix_hash_lo);
    dist_u64_to_halves(result_hash, &work.result_hash_hi, &work.result_hash_lo);
    work.pos0 = pos0;
    work.n_tokens = n_tokens;
    work.layer_start = first->layer_start;
    work.layer_end = first->layer_end;
    work.flags = DS4_DIST_WORK_F_INPUT_HC;
    if (reset_session) work.flags |= DS4_DIST_WORK_F_RESET_SESSION;
    if (ack_only) work.flags |= DS4_DIST_WORK_F_ACK_ONLY;
    if ((first->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0) {
        work.flags |= DS4_DIST_WORK_F_OUTPUT_LOGITS;
    }
    uint32_t wire_hidden_hc_bytes = 0;
    if (!dist_activation_wire_bytes_from_f32_bytes(state->activation_bits,
                                                   hidden_hc_bytes,
                                                   &wire_hidden_hc_bytes)) {
        if (errlen) snprintf(err, errlen, "invalid distributed hidden-state size");
        return 1;
    }
    work.token_bytes = n_tokens * sizeof(uint32_t);
    work.input_hc_bytes = wire_hidden_hc_bytes;
    work.input_hc_bits = state->activation_bits;
    work.route_count = plan->count;
    work.route_index = 0;
    work.route_bytes = plan->blob_bytes;

    if (dist_send_work_frame(fd, &work, tokens, hidden_hc, plan->blob) != 0) {
        if (errlen) snprintf(err, errlen, "failed to send distributed work");
        return 1;
    }
    return 0;
}

static int dist_coordinator_eval_remote_on_fd(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        int fd,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        uint64_t prefix_hash,
        uint64_t expected_result_hash,
        bool reset_session,
        const float *hidden_hc,
        uint32_t hidden_hc_bytes,
        float *logits,
        char *err,
        size_t errlen) {
    int rc = dist_coordinator_send_remote_work_on_fd(state,
                                                     plan,
                                                     fd,
                                                     tokens,
                                                     n_tokens,
                                                     pos0,
                                                     session_id,
                                                     request_id,
                                                     prefix_hash,
                                                     expected_result_hash,
                                                     reset_session,
                                                     false,
                                                     hidden_hc,
                                                     hidden_hc_bytes,
                                                     err,
                                                     errlen);
    uint32_t kind = 0, payload_bytes = 0;
    uint64_t result_hash = 0;
    void *payload = NULL;
    if (rc == 0) {
        rc = dist_recv_result_alloc(fd,
                                    state,
                                    request_id,
                                    &kind,
                                    &result_hash,
                                    &payload,
                                    &payload_bytes,
                                    err,
                                    errlen);
    }
    if (rc != 0) return rc;
    if (result_hash != expected_result_hash) {
        free(payload);
        if (errlen) snprintf(err, errlen, "distributed result prefix hash mismatch");
        return 1;
    }

    const uint32_t logits_bytes = (uint32_t)((uint64_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    if (kind == DS4_DIST_RESULT_LOGITS && payload_bytes == logits_bytes) {
        memcpy(logits, payload, logits_bytes);
        free(payload);
        return 0;
    }
    if (kind == DS4_DIST_RESULT_HIDDEN_STATE && payload_bytes == hidden_hc_bytes) {
        int head_rc = ds4_session_eval_output_head_from_hc(session,
                                                           payload,
                                                           n_tokens,
                                                           logits,
                                                           err,
                                                           errlen);
        free(payload);
        return head_rc;
    }
    if (kind == DS4_DIST_RESULT_HIDDEN_STATE) {
        free(payload);
        if (errlen) snprintf(err, errlen, "distributed route returned invalid hidden-state size");
        return 1;
    }
    free(payload);
    if (errlen) snprintf(err, errlen, "distributed route did not return logits or hidden-state");
    return 1;
}

static int dist_coordinator_eval_span(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        bool reset_session,
        float *logits,
        char *err,
        size_t errlen) {
    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t hidden_bytes64 = (uint64_t)n_tokens * hc_values * sizeof(float);
    if (hidden_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed coordinator hidden-state chunk is too large");
        return 1;
    }
    uint64_t prefix_hash = DS4_DIST_TOKEN_HASH_INIT;
    if (reset_session) {
        if (pos0 != 0) {
            if (errlen) snprintf(err, errlen, "distributed reset span must start at position 0");
            return 1;
        }
    } else if (dist_session_token_hash_prefix(session,
                                              pos0,
                                              &prefix_hash,
                                              err,
                                              errlen) != 0) {
        return 1;
    }
    const uint64_t result_hash = dist_token_hash_update_span(prefix_hash, tokens, n_tokens);
    const uint32_t hidden_bytes = (uint32_t)hidden_bytes64;
    float *hidden = NULL;
    if (plan->count != 0) {
        hidden = malloc(hidden_bytes);
        if (!hidden) {
            if (errlen) snprintf(err, errlen, "out of memory allocating coordinator hidden-state");
            return 1;
        }
    }
    if (reset_session &&
        ds4_session_layer_slice_reset(session, err, errlen) != 0) {
        free(hidden);
        return 1;
    }

    const bool local_logits = plan->count == 0;
    int remote_fd = -1;
    if (plan->count != 0) {
        const ds4_dist_route_entry *first = &plan->entry[0];
        remote_fd = first->fd;
        if (remote_fd < 0) {
            if (errlen) snprintf(err, errlen, "distributed route has no live first-hop connection");
            free(hidden);
            return 1;
        }
    }

    int rc = ds4_session_eval_layer_slice(session,
                                          tokens,
                                          n_tokens,
                                          pos0,
                                          state->local_start,
                                          state->local_end,
                                          NULL,
                                          local_logits ? NULL : hidden,
                                          local_logits,
                                          local_logits ? logits : NULL,
                                          err,
                                          errlen);
    if (rc == 0 && plan->count != 0) {
        rc = dist_coordinator_eval_remote_on_fd(state,
                                                session,
                                                plan,
                                                remote_fd,
                                                tokens,
                                                n_tokens,
                                                pos0,
                                                session_id,
                                                request_id,
                                                prefix_hash,
                                                result_hash,
                                                reset_session,
                                                hidden,
                                                hidden_bytes,
                                                logits,
                                                err,
                                                errlen);
    }
    free(hidden);
    return rc;
}

static bool dist_prompt_is_rendered_chat(const char *prompt) {
    const char *bos = "<｜begin▁of▁sentence｜>";
    return prompt && strncmp(prompt, bos, strlen(bos)) == 0;
}

static bool dist_json_utf8_valid(const char *s, size_t n) {
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            i++;
            continue;
        }
        int need = 0;
        if ((c & 0xe0) == 0xc0) need = 2;
        else if ((c & 0xf0) == 0xe0) need = 3;
        else if ((c & 0xf8) == 0xf0) need = 4;
        else return false;
        if (i + (size_t)need > n) return false;
        unsigned char c1 = (unsigned char)s[i + 1u];
        if ((c1 & 0xc0) != 0x80) return false;
        if (need == 2 && c < 0xc2) return false;
        if (need == 3 && c == 0xe0 && c1 < 0xa0) return false;
        if (need == 3 && c == 0xed && c1 >= 0xa0) return false;
        if (need == 4 && c == 0xf0 && c1 < 0x90) return false;
        if (need == 4 && c == 0xf4 && c1 >= 0x90) return false;
        for (int j = 2; j < need; j++) {
            if ((((unsigned char)s[i + (size_t)j]) & 0xc0) != 0x80) return false;
        }
        i += (size_t)need;
    }
    return true;
}

static void dist_json_write_string(FILE *fp, const char *s, size_t n) {
    const bool valid_utf8 = dist_json_utf8_valid(s, n);
    fputc('"', fp);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else if (!valid_utf8 && c >= 0x80) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void dist_json_write_token(FILE *fp, ds4_engine *engine, int token) {
    size_t n = 0;
    char *text = ds4_token_text(engine, token, &n);
    fprintf(fp, "{\"id\":%d,\"text\":", token);
    dist_json_write_string(fp, text ? text : "", text ? n : 0);
    fputs(",\"bytes\":[", fp);
    if (text) {
        for (size_t i = 0; i < n; i++) {
            if (i) fputc(',', fp);
            fprintf(fp, "%u", (unsigned)(unsigned char)text[i]);
        }
    }
    fputc(']', fp);
    fputc('}', fp);
    free(text);
}

static int dist_write_logits_dump(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen,
        const ds4_tokens *prompt,
        const ds4_dist_route_plan *plan,
        const float *logits) {
    FILE *fp = fopen(gen->dump_logits_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open distributed --dump-logits file: %s\n",
                gen->dump_logits_path);
        return 1;
    }

    const int vocab = ds4_engine_vocab_size(state->engine);
    const int argmax = dist_logits_argmax(logits, vocab);
    fprintf(fp,
            "{\n"
            "  \"source\":\"ds4-distributed\",\n"
            "  \"quant_bits\":%d,\n"
            "  \"prompt_tokens\":%d,\n"
            "  \"ctx\":%d,\n"
            "  \"vocab\":%d,\n"
            "  \"route_count\":%u,\n"
            "  \"argmax_token\":",
            ds4_engine_routed_quant_bits(state->engine),
            prompt->len,
            gen->ctx_size,
            vocab,
            plan->count);
    dist_json_write_token(fp, state->engine, argmax);
    fprintf(fp, ",\n  \"argmax_logit\":%.9g,\n  \"logits\":[", logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close distributed --dump-logits file: %s\n",
                gen->dump_logits_path);
        return 1;
    }
    return 0;
}

static int dist_logits_top_logprobs(const float *logits, int vocab, ds4_dist_logprob *scores, int k) {
    if (k <= 0) return 0;
    for (int i = 0; i < k; i++) {
        scores[i].id = -1;
        scores[i].logit = -FLT_MAX;
        scores[i].logprob = -FLT_MAX;
    }

    float max_logit = -FLT_MAX;
    for (int i = 0; i < vocab; i++) {
        if (isfinite(logits[i]) && logits[i] > max_logit) max_logit = logits[i];
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (int i = 0; i < vocab; i++) {
        if (isfinite(logits[i])) sum += exp((double)logits[i] - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);

    int n = 0;
    for (int i = 0; i < vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == k && v <= scores[k - 1].logit) continue;
        int pos = n < k ? n++ : k - 1;
        while (pos > 0 && v > scores[pos - 1].logit) {
            scores[pos] = scores[pos - 1];
            pos--;
        }
        scores[pos].id = i;
        scores[pos].logit = v;
        scores[pos].logprob = (float)((double)v - logsum);
    }
    return n;
}

static int dist_coordinator_rebuild_from_transcript(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        ds4_dist_route_plan *plan,
        const ds4_tokens *transcript,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        uint64_t *plan_generation,
        bool forget_route,
        char *err,
        size_t errlen) {
    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: replaying %d tokens after distributed %s\n",
                     transcript->len,
                     forget_route ? "route failure" : "KV mismatch");
    if (forget_route) {
        dist_coordinator_forget_route_workers(state, plan);
        dist_route_plan_free(plan);
        uint64_t generation = 0;
        if (!dist_coordinator_ensure_route(state, plan, &generation, err, errlen)) return 1;
        if (plan_generation) *plan_generation = generation;
    } else if (plan->count == 0) {
        uint64_t generation = 0;
        if (!dist_coordinator_ensure_route(state, plan, &generation, err, errlen)) return 1;
        if (plan_generation) *plan_generation = generation;
    }
    if (dist_coordinator_prefill_prompt(state,
                                        session,
                                        plan,
                                        transcript,
                                        session_id,
                                        request_id,
                                        logits,
                                        err,
                                        errlen) != 0) {
        return 1;
    }
    return 0;
}

static int dist_write_logprobs_dump(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen,
        const ds4_tokens *prompt,
        ds4_dist_route_plan *plan,
        ds4_session *session,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits) {
    FILE *fp = fopen(gen->dump_logprobs_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open distributed --dump-logprobs file: %s\n",
                gen->dump_logprobs_path);
        return 1;
    }

    int k = gen->dump_logprobs_top_k > 0 ? gen->dump_logprobs_top_k : 20;
    if (k > 128) k = 128;
    ds4_dist_logprob *scores = calloc((size_t)k, sizeof(scores[0]));
    if (!scores) {
        fclose(fp);
        return 1;
    }

    ds4_tokens transcript = {0};
    ds4_tokens_copy(&transcript, prompt);

    int max_tokens = gen->n_predict;
    int room = gen->ctx_size - prompt->len;
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    fprintf(fp,
            "{\n"
            "  \"source\":\"ds4-distributed\",\n"
            "  \"prompt_tokens\":%d,\n"
            "  \"ctx\":%d,\n"
            "  \"top_k\":%d,\n"
            "  \"route_count\":%u,\n"
            "  \"steps\":[\n",
            prompt->len,
            gen->ctx_size,
            k,
            plan->count);

    char err[256];
    int rc = 0;
    const int eos = ds4_token_eos(state->engine);
    for (int generated = 0; generated < max_tokens; generated++) {
        const int n = dist_logits_top_logprobs(logits, ds4_engine_vocab_size(state->engine), scores, k);
        const int token = dist_logits_argmax(logits, ds4_engine_vocab_size(state->engine));
        if (generated) fputs(",\n", fp);
        fprintf(fp, "    {\"step\":%d,\"selected\":", generated);
        dist_json_write_token(fp, state->engine, token);
        fputs(",\"top_logprobs\":[", fp);
        for (int i = 0; i < n; i++) {
            if (i) fputc(',', fp);
            fputs("{\"token\":", fp);
            dist_json_write_token(fp, state->engine, scores[i].id);
            fprintf(fp, ",\"logit\":%.9g,\"logprob\":%.9g}", scores[i].logit, scores[i].logprob);
        }
        fputs("]}", fp);

        if (token == eos) break;
        const uint32_t token_pos = (uint32_t)prompt->len + (uint32_t)generated;
        ds4_tokens_push(&transcript, token);
        if (dist_coordinator_eval_span(state, session, plan,
                                       &token, 1, token_pos,
                                       session_id, (*request_id)++,
                                       false, logits, err, sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: distributed decode failed while dumping logprobs: %s\n",
                    err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         plan,
                                                         &transcript,
                                                         session_id,
                                                         request_id,
                                                         logits,
                                                         NULL,
                                                         true,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr,
                        "ds4: distributed recovery failed while dumping logprobs: %s\n",
                        err);
                rc = 1;
                break;
            }
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close distributed --dump-logprobs file: %s\n",
                gen->dump_logprobs_path);
        rc = 1;
    }
    ds4_tokens_free(&transcript);
    free(scores);
    return rc;
}

static int dist_prefill_sender_init(
        ds4_dist_prefill_sender *sender,
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint64_t session_id,
        int fd,
        uint32_t chunk_count,
        uint32_t max_hidden_bytes,
        char *err,
        size_t errlen) {
    memset(sender, 0, sizeof(*sender));
    sender->state = state;
    sender->plan = plan;
    sender->prompt = prompt;
    sender->session_id = session_id;
    sender->fd = fd;
    sender->slot_count = dist_prefill_send_depth(chunk_count);
    pthread_mutex_init(&sender->mu, NULL);
    pthread_cond_init(&sender->can_enqueue, NULL);
    pthread_cond_init(&sender->can_dequeue, NULL);

    sender->slots = calloc(sender->slot_count, sizeof(sender->slots[0]));
    if (!sender->slots) {
        if (errlen) snprintf(err, errlen, "out of memory allocating prefill sender slots");
        return 1;
    }
    for (uint32_t i = 0; i < sender->slot_count; i++) {
        sender->slots[i].hidden = malloc(max_hidden_bytes);
        if (!sender->slots[i].hidden) {
            if (errlen) snprintf(err, errlen, "out of memory allocating prefill sender hidden-state buffers");
            return 1;
        }
    }
    return 0;
}

static void dist_prefill_sender_destroy(ds4_dist_prefill_sender *sender) {
    if (!sender) return;
    if (sender->slots) {
        for (uint32_t i = 0; i < sender->slot_count; i++) {
            free(sender->slots[i].hidden);
        }
        free(sender->slots);
    }
    pthread_cond_destroy(&sender->can_dequeue);
    pthread_cond_destroy(&sender->can_enqueue);
    pthread_mutex_destroy(&sender->mu);
}

static ds4_dist_prefill_send_slot *dist_prefill_sender_acquire_slot(
        ds4_dist_prefill_sender *sender,
        char *err,
        size_t errlen) {
    pthread_mutex_lock(&sender->mu);
    while (!sender->stop && sender->queued == sender->slot_count) {
        pthread_cond_wait(&sender->can_enqueue, &sender->mu);
    }
    if (sender->stop || sender->rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender->err[0] ? sender->err : "distributed prefill sender stopped");
        pthread_mutex_unlock(&sender->mu);
        return NULL;
    }
    ds4_dist_prefill_send_slot *slot = &sender->slots[sender->tail];
    pthread_mutex_unlock(&sender->mu);
    return slot;
}

static int dist_prefill_sender_enqueue_slot(
        ds4_dist_prefill_sender *sender,
        char *err,
        size_t errlen) {
    pthread_mutex_lock(&sender->mu);
    if (sender->stop || sender->rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender->err[0] ? sender->err : "distributed prefill sender stopped");
        pthread_mutex_unlock(&sender->mu);
        return 1;
    }
    sender->tail = (sender->tail + 1u) % sender->slot_count;
    sender->queued++;
    pthread_cond_signal(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
    return 0;
}

static void dist_prefill_sender_finish(ds4_dist_prefill_sender *sender) {
    pthread_mutex_lock(&sender->mu);
    sender->producer_done = true;
    pthread_cond_signal(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
}

static void dist_prefill_sender_cancel(ds4_dist_prefill_sender *sender) {
    pthread_mutex_lock(&sender->mu);
    sender->producer_done = true;
    sender->stop = true;
    pthread_cond_broadcast(&sender->can_enqueue);
    pthread_cond_broadcast(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
    shutdown(sender->fd, SHUT_RDWR);
}

static void *dist_prefill_sender_main(void *arg) {
    ds4_dist_prefill_sender *sender = arg;
    for (;;) {
        pthread_mutex_lock(&sender->mu);
        while (!sender->stop && sender->queued == 0 && !sender->producer_done) {
            pthread_cond_wait(&sender->can_dequeue, &sender->mu);
        }
        if (sender->stop || (sender->queued == 0 && sender->producer_done)) {
            pthread_mutex_unlock(&sender->mu);
            break;
        }
        ds4_dist_prefill_send_slot *slot = &sender->slots[sender->head];
        pthread_mutex_unlock(&sender->mu);

        char send_err[256];
        const double send_t0 = dist_now_sec();
        int rc = dist_coordinator_send_remote_work_on_fd(sender->state,
                                                         sender->plan,
                                                         sender->fd,
                                                         sender->prompt->v + slot->pos,
                                                         slot->n_tokens,
                                                         slot->pos,
                                                         sender->session_id,
                                                         slot->request_id,
                                                         slot->prefix_hash,
                                                         slot->result_hash,
                                                         slot->reset_session,
                                                         slot->ack_only,
                                                         slot->hidden,
                                                         slot->hidden_bytes,
                                                         send_err,
                                                         sizeof(send_err));
        const double send_t1 = dist_now_sec();

        pthread_mutex_lock(&sender->mu);
        uint32_t slot_hidden_wire_bytes = slot->hidden_bytes;
        (void)dist_activation_wire_bytes_from_f32_bytes(sender->state->activation_bits,
                                                        slot->hidden_bytes,
                                                        &slot_hidden_wire_bytes);
        sender->send_sec += send_t1 - send_t0;
        sender->send_bytes += (uint64_t)sizeof(ds4_dist_work_fixed) +
                              (uint64_t)slot->n_tokens * sizeof(uint32_t) +
                              (uint64_t)slot_hidden_wire_bytes +
                              sender->plan->blob_bytes;
        if (rc != 0) {
            sender->rc = 1;
            snprintf(sender->err, sizeof(sender->err), "%s", send_err);
            sender->stop = true;
            pthread_cond_broadcast(&sender->can_enqueue);
            pthread_cond_broadcast(&sender->can_dequeue);
            pthread_mutex_unlock(&sender->mu);
            shutdown(sender->fd, SHUT_RDWR);
            break;
        }
        sender->head = (sender->head + 1u) % sender->slot_count;
        sender->queued--;
        pthread_cond_signal(&sender->can_enqueue);
        pthread_mutex_unlock(&sender->mu);
    }
    return NULL;
}

static void dist_prefill_reader_signal_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t completed,
        bool done) {
    pthread_mutex_lock(&reader->progress_mu);
    if (completed > reader->progress_completed) reader->progress_completed = completed;
    if (done) reader->progress_done = true;
    pthread_cond_broadcast(&reader->progress_cv);
    pthread_mutex_unlock(&reader->progress_mu);
}

static void dist_prefill_reader_emit_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t *reported) {
    if (!reader || !reported || !reader->progress_session) return;

    pthread_mutex_lock(&reader->progress_mu);
    uint32_t completed = reader->progress_completed;
    pthread_mutex_unlock(&reader->progress_mu);

    while (*reported < completed) {
        (*reported)++;
        uint32_t rel = (*reported) * reader->chunk_cap;
        if (rel > reader->total_tokens) rel = reader->total_tokens;
        uint32_t current = reader->progress_base + rel;
        if (current > reader->progress_total) current = reader->progress_total;
        ds4_session_report_progress(reader->progress_session,
                                    "prefill_chunk",
                                    (int)current,
                                    (int)reader->progress_total);
    }
}

static bool dist_prefill_reader_wait_emit_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t *reported) {
    if (!reader || !reported) return true;

    pthread_mutex_lock(&reader->progress_mu);
    while (reader->progress_completed <= *reported && !reader->progress_done) {
        pthread_cond_wait(&reader->progress_cv, &reader->progress_mu);
    }
    const bool done = reader->progress_done;
    pthread_mutex_unlock(&reader->progress_mu);

    dist_prefill_reader_emit_progress(reader, reported);
    pthread_mutex_lock(&reader->progress_mu);
    const bool finished = done && *reported >= reader->progress_completed;
    pthread_mutex_unlock(&reader->progress_mu);
    return finished;
}

static bool dist_prefill_reader_wait_flow_window(
        ds4_dist_prefill_result_reader *reader,
        uint32_t submitted,
        uint32_t window,
        uint32_t *reported) {
    if (!reader || window == 0) return true;

    for (;;) {
        pthread_mutex_lock(&reader->progress_mu);
        const uint32_t completed = reader->progress_completed;
        const bool done = reader->progress_done;
        const bool has_room = submitted < completed + window;
        if (done || has_room) {
            pthread_mutex_unlock(&reader->progress_mu);
            dist_prefill_reader_emit_progress(reader, reported);
            return !done && has_room;
        }
        pthread_cond_wait(&reader->progress_cv, &reader->progress_mu);
        pthread_mutex_unlock(&reader->progress_mu);
        dist_prefill_reader_emit_progress(reader, reported);
    }
}

static void *dist_prefill_result_reader_main(void *arg) {
    ds4_dist_prefill_result_reader *reader = arg;
    reader->rc = 0;
    reader->err[0] = '\0';
    reader->final_kind = 0;
    reader->final_payload = NULL;
    reader->final_payload_bytes = 0;

    const uint32_t logits_bytes =
        (uint32_t)((uint64_t)ds4_engine_vocab_size(reader->state->engine) * sizeof(float));
    for (uint32_t i = 0; i < reader->count; i++) {
        const uint64_t request_id = reader->first_request_id + (uint64_t)i;
        uint32_t kind = 0;
        uint32_t payload_bytes = 0;
        uint64_t result_hash = 0;
        void *payload = NULL;
        int recv_rc = dist_recv_result_alloc(reader->fd,
                                             reader->state,
                                             request_id,
                                             &kind,
                                             &result_hash,
                                             &payload,
                                             &payload_bytes,
                                             reader->err,
                                             sizeof(reader->err));
        if (recv_rc != 0) {
            reader->rc = recv_rc;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        if (reader->expected_hashes && result_hash != reader->expected_hashes[i]) {
            snprintf(reader->err,
                     sizeof(reader->err),
                     "distributed pipelined prefill prefix hash mismatch");
            reader->rc = 1;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        const uint32_t pos0 = i * reader->chunk_cap;
        const uint32_t remaining = reader->total_tokens - pos0;
        const uint32_t chunk = remaining < reader->chunk_cap ? remaining : reader->chunk_cap;
        const uint64_t hidden_bytes64 = (uint64_t)chunk * reader->hc_values * sizeof(float);
        const bool final_chunk = i + 1u == reader->count;
        const bool valid_ack = !final_chunk &&
                               kind == DS4_DIST_RESULT_ACK &&
                               payload_bytes == 0;
        const bool valid_logits = kind == DS4_DIST_RESULT_LOGITS && payload_bytes == logits_bytes;
        const bool valid_hidden = reader->allow_hidden &&
                                  hidden_bytes64 <= UINT32_MAX &&
                                  kind == DS4_DIST_RESULT_HIDDEN_STATE &&
                                  payload_bytes == (uint32_t)hidden_bytes64;
        if (!valid_ack && !valid_logits && !valid_hidden) {
            snprintf(reader->err,
                     sizeof(reader->err),
                     "distributed pipelined prefill returned invalid result");
            reader->rc = 1;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        if (final_chunk) {
            reader->final_kind = kind;
            reader->final_payload = payload;
            reader->final_payload_bytes = payload_bytes;
            payload = NULL;
        }
        free(payload);
        dist_prefill_reader_signal_progress(reader, i + 1u, final_chunk);
    }
    dist_prefill_reader_signal_progress(reader, reader->count, true);
    return NULL;
}

static bool dist_coordinator_can_pipeline_prefill(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        ds4_session *session,
        uint32_t n_tokens,
        uint32_t chunk_cap) {
    if (getenv("DS4_DIST_DISABLE_PREFILL_PIPELINE")) return false;
    if (!state || !plan) return false;
    (void)session;
    if (chunk_cap == 0 || n_tokens <= chunk_cap) return false;
    if (plan->count == 0) return false;
    if (plan->entry[0].fd < 0) return false;
    const ds4_dist_route_entry *final = &plan->entry[plan->count - 1u];
    if ((final->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) == 0) {
        return final->layer_end + 1u == state->n_layers &&
               state->local_can_output_head;
    }
    return true;
}

static int dist_coordinator_prefill_chunk_cap(
        const ds4_dist_coordinator_state *state,
        ds4_session *session,
        uint32_t *chunk_cap,
        char *err,
        size_t errlen) {
    if (!chunk_cap) return 1;
    const int prefill_cap_i = ds4_session_prefill_cap(session);
    if (prefill_cap_i <= 0) {
        if (errlen) snprintf(err, errlen, "distributed coordinator has no prefill capacity");
        return 1;
    }
    const uint32_t prefill_cap = (uint32_t)prefill_cap_i;
    uint32_t requested = state ? state->prefill_chunk : 0u;
    const char *env = getenv("DS4_DIST_PREFILL_CHUNK");
    if (requested == 0 && env && env[0]) {
        if (!dist_parse_positive_u32(env, "DS4_DIST_PREFILL_CHUNK", &requested, err, errlen)) {
            return 1;
        }
    }
    if (requested == 0) requested = prefill_cap;
    if (requested > prefill_cap) {
        if (errlen) {
            snprintf(err,
                     errlen,
                     "distributed prefill chunk %u exceeds session prefill cap %u",
                     requested,
                     prefill_cap);
        }
        return 1;
    }
    *chunk_cap = requested;
    return 0;
}

static int dist_coordinator_prefill_window(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        uint32_t chunk_count,
        uint32_t *window,
        char *err,
        size_t errlen) {
    if (!window) return 1;
    uint32_t requested = state ? state->prefill_window : 0u;
    const char *env = getenv("DS4_DIST_PREFILL_WINDOW");
    if (requested == 0 && env && env[0]) {
        if (!dist_parse_positive_u32(env, "DS4_DIST_PREFILL_WINDOW", &requested, err, errlen)) {
            return 1;
        }
    }
    if (requested > 64u) {
        if (errlen) snprintf(err, errlen, "distributed prefill window %u exceeds limit 64", requested);
        return 1;
    }
    if (requested == 0) {
        const uint32_t remote_stages = plan ? plan->count : 0u;
        requested = remote_stages + 2u;
        if (requested < 2u) requested = 2u;
        if (requested > 8u) requested = 8u;
    }
    if (chunk_count != 0 && requested > chunk_count) requested = chunk_count;
    *window = requested ? requested : 1u;
    return 0;
}

static void dist_report_prefill_progress(ds4_session *session, uint32_t current, uint32_t total) {
    if (!session) return;
    if (current > (uint32_t)INT_MAX) current = (uint32_t)INT_MAX;
    if (total > (uint32_t)INT_MAX) total = (uint32_t)INT_MAX;
    ds4_session_report_progress(session, "prefill_chunk", (int)current, (int)total);
}

static int dist_coordinator_prefill_prompt_pipelined(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint32_t span_start,
        uint32_t n_tokens,
        bool reset_first_chunk,
        uint32_t chunk_cap,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        char *err,
        size_t errlen) {
    const uint32_t total = n_tokens;
    if (!prompt ||
        span_start > (uint32_t)prompt->len ||
        n_tokens == 0 ||
        n_tokens > (uint32_t)prompt->len - span_start) {
        if (errlen) snprintf(err, errlen, "invalid distributed pipelined prefill span");
        return 1;
    }
    const uint32_t span_end = span_start + n_tokens;
    const uint32_t chunk_count = (total + chunk_cap - 1u) / chunk_cap;
    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t max_hidden_bytes64 = (uint64_t)chunk_cap * hc_values * sizeof(float);
    if (max_hidden_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed coordinator hidden-state chunk is too large");
        return 1;
    }
    const uint32_t max_hidden_bytes = (uint32_t)max_hidden_bytes64;
    uint32_t flow_window = 0;
    if (dist_coordinator_prefill_window(state,
                                        plan,
                                        chunk_count,
                                        &flow_window,
                                        err,
                                        errlen) != 0) {
        return 1;
    }

    ds4_dist_prefill_sender sender;
    if (dist_prefill_sender_init(&sender,
                                 state,
                                 plan,
                                 prompt,
                                 session_id,
                                 plan->entry[0].fd,
                                 chunk_count,
                                 max_hidden_bytes,
                                 err,
                                 errlen) != 0) {
        dist_prefill_sender_destroy(&sender);
        return 1;
    }

    ds4_dist_prefill_result_reader reader;
    memset(&reader, 0, sizeof(reader));
    reader.state = state;
    reader.fd = plan->entry[0].fd;
    reader.progress_session = session;
    reader.first_request_id = *request_id;
    reader.count = chunk_count;
    reader.total_tokens = total;
    reader.chunk_cap = chunk_cap;
    reader.progress_base = span_start;
    reader.progress_total = (uint32_t)prompt->len;
    reader.hc_values = hc_values;
    reader.allow_hidden =
        (plan->entry[plan->count - 1u].flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) == 0;
    pthread_mutex_init(&reader.progress_mu, NULL);
    pthread_cond_init(&reader.progress_cv, NULL);
    reader.expected_hashes = calloc(chunk_count, sizeof(reader.expected_hashes[0]));
    if (!reader.expected_hashes) {
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "out of memory allocating distributed prefill hashes");
        return 1;
    }
    uint64_t chunk_prefix_hash = dist_token_hash_prefix(prompt->v, span_start);
    for (uint32_t i = 0, hash_pos = span_start; i < chunk_count; i++) {
        const uint32_t remaining = span_end - hash_pos;
        const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        chunk_prefix_hash = dist_token_hash_update_span(chunk_prefix_hash,
                                                        prompt->v + hash_pos,
                                                        chunk);
        reader.expected_hashes[i] = chunk_prefix_hash;
        hash_pos += chunk;
    }

    pthread_t reader_tid;
    if (pthread_create(&reader_tid, NULL, dist_prefill_result_reader_main, &reader) != 0) {
        free(reader.expected_hashes);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "failed to start distributed prefill result reader");
        return 1;
    }
    pthread_t sender_tid;
    if (pthread_create(&sender_tid, NULL, dist_prefill_sender_main, &sender) != 0) {
        dist_prefill_sender_cancel(&sender);
        pthread_join(reader_tid, NULL);
        free(reader.expected_hashes);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "failed to start distributed prefill sender");
        return 1;
    }

    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: pipelined prefill %u chunks of up to %u tokens through %u worker%s, first hop %s:%u, send depth %u, flow window %u\n",
                     chunk_count,
                     chunk_cap,
                     plan->count,
                     plan->count == 1u ? "" : "s",
                     plan->entry[0].host,
                     plan->entry[0].port,
                     sender.slot_count,
                     flow_window);

    int rc = 0;
    double local_eval_sec = 0.0;
    const double pipeline_t0 = dist_now_sec();
    uint32_t pos = span_start;
    uint64_t next_prefix_hash = dist_token_hash_prefix(prompt->v, span_start);
    uint32_t reported_chunks = 0;
    uint32_t submitted_chunks = 0;
    while (pos < span_end) {
        if (!dist_prefill_reader_wait_flow_window(&reader,
                                                  submitted_chunks,
                                                  flow_window,
                                                  &reported_chunks)) {
            if (errlen) snprintf(err, errlen, "distributed prefill result reader stopped");
            rc = 1;
            break;
        }
        const uint32_t remaining = span_end - pos;
        const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        const uint64_t hidden_bytes64 = (uint64_t)chunk * hc_values * sizeof(float);
        const uint32_t hidden_bytes = (uint32_t)hidden_bytes64;
        ds4_dist_prefill_send_slot *slot =
            dist_prefill_sender_acquire_slot(&sender, err, errlen);
        if (!slot) {
            rc = 1;
            break;
        }
        if (pos == span_start &&
            reset_first_chunk &&
            ds4_session_layer_slice_reset(session, err, errlen) != 0) {
            rc = 1;
            break;
        }
        const double local_t0 = dist_now_sec();
        rc = ds4_session_eval_layer_slice(session,
                                          prompt->v + pos,
                                          chunk,
                                          pos,
                                          state->local_start,
                                          state->local_end,
                                          NULL,
                                          slot->hidden,
                                          false,
                                          NULL,
                                          err,
                                          errlen);
        const double local_t1 = dist_now_sec();
        local_eval_sec += local_t1 - local_t0;
        if (rc != 0) break;

        slot->pos = pos;
        slot->n_tokens = chunk;
        slot->hidden_bytes = hidden_bytes;
        slot->request_id = *request_id;
        slot->prefix_hash = next_prefix_hash;
        slot->result_hash = reader.expected_hashes[submitted_chunks];
        slot->reset_session = reset_first_chunk && pos == span_start;
        slot->ack_only = !getenv("DS4_DIST_DISABLE_PREFILL_ACK_ONLY") &&
                         pos + chunk < span_end;
        rc = dist_prefill_sender_enqueue_slot(&sender, err, errlen);
        if (rc != 0) break;

        dist_prefill_reader_emit_progress(&reader, &reported_chunks);
        (*request_id)++;
        submitted_chunks++;
        next_prefix_hash = slot->result_hash;
        pos += chunk;
    }

    if (rc == 0) dist_prefill_sender_finish(&sender);
    else dist_prefill_sender_cancel(&sender);
    pthread_join(sender_tid, NULL);
    if (rc == 0 && sender.rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender.err[0] ? sender.err : "distributed prefill sender failed");
        rc = 1;
    }
    if (rc != 0) {
        shutdown(plan->entry[0].fd, SHUT_RDWR);
    }
    if (rc == 0) {
        while (!dist_prefill_reader_wait_emit_progress(&reader, &reported_chunks)) {
            ;
        }
    }
    pthread_join(reader_tid, NULL);
    const double pipeline_t1 = dist_now_sec();
    if (rc == 0 && reader.rc == 0) {
        const double total_sec = pipeline_t1 - pipeline_t0;
        DIST_COORD_DEBUG(state,
                         "ds4: distributed coordinator: pipelined prefill done tokens=%u chunks=%u total=%.3fs %.2f t/s local=%.3fs send=%.3fs %.2f MiB/s\n",
                         total,
                         chunk_count,
                         total_sec,
                         total_sec > 0.0 ? (double)total / total_sec : 0.0,
                         local_eval_sec,
                         sender.send_sec,
                         sender.send_sec > 0.0
                             ? ((double)sender.send_bytes / (1024.0 * 1024.0)) / sender.send_sec
                             : 0.0);
    }
    if (reader.rc != 0) {
        if (errlen) snprintf(err, errlen, "%s", reader.err[0] ? reader.err : "distributed pipelined prefill failed");
        int reader_rc = reader.rc;
        free(reader.final_payload);
        free(reader.expected_hashes);
        dist_prefill_sender_destroy(&sender);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        return reader_rc;
    }
    dist_prefill_sender_destroy(&sender);
    free(reader.expected_hashes);
    pthread_cond_destroy(&reader.progress_cv);
    pthread_mutex_destroy(&reader.progress_mu);
    if (rc != 0) {
        free(reader.final_payload);
        return 1;
    }
    const uint32_t logits_bytes =
        (uint32_t)((uint64_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    if (reader.final_kind == DS4_DIST_RESULT_LOGITS &&
        reader.final_payload_bytes == logits_bytes) {
        memcpy(logits, reader.final_payload, logits_bytes);
        free(reader.final_payload);
        return 0;
    }
    if (reader.final_kind == DS4_DIST_RESULT_HIDDEN_STATE &&
        reader.final_payload) {
        const uint32_t last_pos = (chunk_count - 1u) * chunk_cap;
        const uint32_t last_tokens = total - last_pos;
        int head_rc = ds4_session_eval_output_head_from_hc(session,
                                                           reader.final_payload,
                                                           last_tokens,
                                                           logits,
                                                           err,
                                                           errlen);
        free(reader.final_payload);
        return head_rc;
    }
    free(reader.final_payload);
    if (errlen) snprintf(err, errlen, "distributed pipelined prefill did not return a final result");
    return 1;
}

static int dist_coordinator_prefill_prompt(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        char *err,
        size_t errlen) {
    uint32_t chunk_cap = 0;
    if (dist_coordinator_prefill_chunk_cap(state, session, &chunk_cap, err, errlen) != 0) {
        return 1;
    }
    const uint32_t prompt_len = (uint32_t)prompt->len;
    if (dist_coordinator_can_pipeline_prefill(state, plan, session, prompt_len, chunk_cap)) {
        return dist_coordinator_prefill_prompt_pipelined(state,
                                                        session,
                                                        plan,
                                                        prompt,
                                                        0,
                                                        prompt_len,
                                                        true,
                                                        chunk_cap,
                                                        session_id,
                                                        request_id,
                                                        logits,
                                                        err,
                                                        errlen);
    }

    uint32_t pos = 0;
    while (pos < prompt_len) {
        uint32_t remaining = prompt_len - pos;
        uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        int eval_rc = dist_coordinator_eval_span(state, session, plan,
                                                 prompt->v + pos, chunk, pos,
                                                 session_id, (*request_id)++,
                                                 pos == 0, logits, err, errlen);
        if (eval_rc != 0) {
            return eval_rc;
        }
        pos += chunk;
        dist_report_prefill_progress(session, pos, prompt_len);
    }
    return 0;
}

static int dist_replay_check_logits(
        ds4_dist_coordinator_state *state,
        const float *before,
        const float *after) {
    const int vocab = ds4_engine_vocab_size(state->engine);
    float max_abs = 0.0f;
    int max_i = 0;
    uint32_t mismatches = 0;
    for (int i = 0; i < vocab; i++) {
        if (before[i] != after[i]) {
            const float d = fabsf(before[i] - after[i]);
            if (d > max_abs) {
                max_abs = d;
                max_i = i;
            }
            mismatches++;
        }
    }
    if (mismatches != 0) {
        fprintf(stderr,
                "ds4: distributed replay check failed: mismatches=%u max_abs=%g token=%d before=%g after=%g\n",
                mismatches,
                max_abs,
                max_i,
                before[max_i],
                after[max_i]);
        return 1;
    }
    fprintf(stderr, "ds4: distributed replay check passed: logits exact match across reset/replay\n");
    return 0;
}

static int dist_run_coordinator_generation(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen) {
    char err[256];
    ds4_dist_route_plan plan;
    uint64_t plan_generation = 0;
    if (!dist_coordinator_ensure_route(state, &plan, &plan_generation, err, sizeof(err))) {
        fprintf(stderr, "ds4: distributed coordinator: %s\n", err);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, state->engine, gen->ctx_size) != 0) {
        fprintf(stderr, "ds4: distributed coordinator: failed to create local session\n");
        dist_route_plan_free(&plan);
        return 1;
    }

    ds4_tokens prompt = {0};
    if (dist_prompt_is_rendered_chat(gen->prompt)) {
        ds4_tokenize_rendered_chat(state->engine, gen->prompt, &prompt);
    } else {
        ds4_encode_chat_prompt(state->engine, gen->system, gen->prompt, gen->think_mode, &prompt);
    }
    if (prompt.len <= 0) {
        fprintf(stderr, "ds4: distributed coordinator: empty prompt\n");
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return 1;
    }

    const uint64_t session_id = ((uint64_t)(uint32_t)time(NULL) << 32) ^ (uint64_t)getpid();
    uint64_t request_id = 1;
    float *logits = malloc((size_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    if (!logits) {
        fprintf(stderr, "ds4: distributed coordinator: out of memory allocating logits\n");
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return 1;
    }

    int prefill_rc = dist_coordinator_prefill_prompt(state,
                                                     session,
                                                     &plan,
                                                     &prompt,
                                                     session_id,
                                                     &request_id,
                                                     logits,
                                                     err,
                                                     sizeof(err));
    if (prefill_rc != 0) {
        fprintf(stderr,
                "ds4: distributed prompt processing failed: %s\n",
                err);
        if (dist_coordinator_rebuild_from_transcript(state,
                                                     session,
                                                     &plan,
                                                     &prompt,
                                                     session_id,
                                                     &request_id,
                                                     logits,
                                                     NULL,
                                                     prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: distributed prompt recovery failed: %s\n",
                    err);
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
    }

    if (state->replay_check) {
        const size_t logits_bytes = (size_t)ds4_engine_vocab_size(state->engine) * sizeof(logits[0]);
        float *before = malloc(logits_bytes);
        if (!before) {
            fprintf(stderr, "ds4: distributed replay check: out of memory allocating logits copy\n");
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
        memcpy(before, logits, logits_bytes);
        int replay_prefill_rc = dist_coordinator_prefill_prompt(state,
                                                                session,
                                                                &plan,
                                                                &prompt,
                                                                session_id,
                                                                &request_id,
                                                                logits,
                                                                err,
                                                                sizeof(err));
        if (replay_prefill_rc != 0) {
            fprintf(stderr,
                    "ds4: distributed replay prompt processing failed: %s\n",
                    err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         &plan,
                                                         &prompt,
                                                         session_id,
                                                         &request_id,
                                                         logits,
                                                         NULL,
                                                         replay_prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr,
                        "ds4: distributed replay recovery failed: %s\n",
                        err);
                free(before);
                free(logits);
                ds4_tokens_free(&prompt);
                ds4_session_free(session);
                dist_route_plan_free(&plan);
                return 1;
            }
        }
        int replay_rc = dist_replay_check_logits(state, before, logits);
        free(before);
        if (replay_rc != 0) {
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
    }

    if (gen->dump_logits_path) {
        int rc = dist_write_logits_dump(state, gen, &prompt, &plan, logits);
        free(logits);
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return rc;
    }
    if (gen->dump_logprobs_path) {
        int rc = dist_write_logprobs_dump(state,
                                          gen,
                                          &prompt,
                                          &plan,
                                          session,
                                          session_id,
                                          &request_id,
                                          logits);
        free(logits);
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return rc;
    }

    int generated = 0;
    int max_tokens = gen->n_predict > 0 ? gen->n_predict : 1;
    int room = gen->ctx_size - prompt.len;
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    uint64_t rng = gen->seed ? gen->seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    const int eos = ds4_token_eos(state->engine);
    ds4_tokens transcript = {0};
    ds4_tokens_copy(&transcript, &prompt);
    while (generated < max_tokens) {
        int token = ds4_sample_logits(logits,
                                      ds4_engine_vocab_size(state->engine),
                                      gen->temperature,
                                      0,
                                      gen->top_p,
                                      gen->min_p,
                                      &rng);
        if (token == eos) break;

        size_t len = 0;
        char *text = ds4_token_text(state->engine, token, &len);
        if (len) fwrite(text, 1, len, stdout);
        fflush(stdout);
        free(text);

        uint32_t token_pos = (uint32_t)prompt.len + (uint32_t)generated;
        generated++;
        ds4_tokens_push(&transcript, token);
        if (generated >= max_tokens) break;

        int decode_rc = dist_coordinator_eval_span(state, session, &plan,
                                                   &token, 1, token_pos,
                                                   session_id, request_id++,
                                                   false, logits, err, sizeof(err));
        if (decode_rc != 0) {
            fprintf(stderr, "\nds4: distributed decode failed: %s\n", err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         &plan,
                                                         &transcript,
                                                         session_id,
                                                         &request_id,
                                                         logits,
                                                         NULL,
                                                         decode_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr, "ds4: distributed decode recovery failed: %s\n", err);
                ds4_tokens_free(&transcript);
                free(logits);
                ds4_tokens_free(&prompt);
                ds4_session_free(session);
                dist_route_plan_free(&plan);
                return 1;
            }
        }
    }
    fputc('\n', stdout);

    ds4_tokens_free(&transcript);
    free(logits);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    dist_route_plan_free(&plan);
    return 0;
}

static void dist_coordinator_remove_worker(ds4_dist_coordinator_state *state, int fd) {
    pthread_mutex_lock(&state->mu);
    ds4_dist_worker_entry **link = &state->workers;
    while (*link) {
        ds4_dist_worker_entry *entry = *link;
        if (entry->fd == fd) {
            *link = entry->next;
            state->generation++;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: removed worker %s:%s layers=%u:%u%s\n",
                             entry->peer_host,
                             entry->peer_port,
                             entry->layer_start,
                             entry->layer_end,
                             entry->has_output ? "+output" : "");
            pthread_mutex_unlock(&state->mu);
            free(entry);
            if (dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
            return;
        }
        link = &entry->next;
    }
    pthread_mutex_unlock(&state->mu);
}

static void dist_coordinator_monitor_worker_fd(
        ds4_dist_coordinator_state *state,
        int fd,
        const char *peer_host,
        const char *peer_port) {
    for (;;) {
        struct pollfd pfd = {
            .fd = fd,
            .events = 0,
            .revents = 0,
        };
        int rc = poll(&pfd, 1, 1000);
        if (rc < 0) {
            if (errno == EINTR) continue;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: worker %s:%s poll failed: %s\n",
                             peer_host,
                             peer_port,
                             strerror(errno));
            break;
        }
        if (rc == 0) continue;
        if ((pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) break;
    }
}

static void *dist_coordinator_client_main(void *arg) {
    ds4_dist_client_ctx *ctx = arg;
    int fd = ctx->fd;
    ds4_dist_coordinator_state *state = ctx->state;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    snprintf(peer_host, sizeof(peer_host), "%s", ctx->peer_host);
    snprintf(peer_port, sizeof(peer_port), "%s", ctx->peer_port);
    free(ctx);

    ds4_dist_hello_fixed hello;
    char model_name[DS4_DIST_MAX_MODEL_NAME + 1u];
    char err[256];
    int rc = dist_recv_hello(fd, &hello, model_name, sizeof(model_name), err, sizeof(err));
    if (rc <= 0) {
        if (rc < 0) DIST_COORD_DEBUG(state, "ds4: distributed coordinator: bad HELLO from %s:%s: %s\n", peer_host, peer_port, err);
        close(fd);
        return NULL;
    }

    if (hello.model_id != state->model_id) {
        snprintf(err, sizeof(err), "model id mismatch: worker=%u coordinator=%u", hello.model_id, state->model_id);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    const char *expected_model_name = ds4_engine_model_name(state->engine);
    if (!expected_model_name) expected_model_name = "unknown";
    if (strcmp(model_name, expected_model_name) != 0) {
        snprintf(err,
                 sizeof(err),
                 "model family mismatch: worker=%s coordinator=%s",
                 model_name,
                 expected_model_name);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.n_layers != state->n_layers) {
        snprintf(err, sizeof(err), "layer count mismatch: worker=%u coordinator=%u", hello.n_layers, state->n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.quant_bits != 2u && hello.quant_bits != 4u) {
        snprintf(err, sizeof(err), "unsupported worker quant profile Q%u", hello.quant_bits);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_output > 1u) {
        snprintf(err, sizeof(err), "invalid worker output-head flag %u", hello.has_output);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_hidden > 1u) {
        snprintf(err, sizeof(err), "invalid worker hidden-state flag %u", hello.has_hidden);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.layer_start >= hello.n_layers || hello.layer_end >= hello.n_layers || hello.layer_end < hello.layer_start) {
        snprintf(err, sizeof(err), "invalid worker layer range %u:%u for %u layers", hello.layer_start, hello.layer_end, hello.n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_output && hello.layer_end + 1u != hello.n_layers) {
        snprintf(err,
                 sizeof(err),
                 "worker output head requires final layer: range=%u:%u layers=%u",
                 hello.layer_start,
                 hello.layer_end,
                 hello.n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (state->ctx_size != 0 && hello.ctx_size < state->ctx_size) {
        snprintf(err,
                 sizeof(err),
                 "worker context too small: worker=%u coordinator=%u",
                 hello.ctx_size,
                 state->ctx_size);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.listen_port == 0 || hello.listen_port > 65535u) {
        snprintf(err, sizeof(err), "invalid worker data listen port %u", hello.listen_port);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }

    dist_coordinator_add_worker(state, fd, peer_host, peer_port, &hello, model_name);

    if (state->use_control_for_work) {
        dist_coordinator_monitor_worker_fd(state, fd, peer_host, peer_port);
    } else {
        for (;;) {
            uint32_t type = 0, bytes = 0;
            rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
            if (rc == 0) break;
            if (rc < 0) {
                DIST_COORD_DEBUG(state, "ds4: distributed coordinator: worker %s:%s protocol error: %s\n", peer_host, peer_port, err);
                break;
            }
            if (type == DS4_DIST_MSG_HELLO) {
                DIST_COORD_DEBUG(state, "ds4: distributed coordinator: worker %s:%s sent duplicate HELLO\n", peer_host, peer_port);
                dist_discard_bytes(fd, bytes);
                break;
            }
            rc = dist_discard_bytes(fd, bytes);
            if (rc <= 0) break;
        }
    }

    dist_coordinator_remove_worker(state, fd);
    close(fd);
    return NULL;
}

static void *dist_coordinator_accept_main(void *arg) {
    ds4_dist_accept_ctx *accept_ctx = arg;
    int listen_fd = accept_ctx->listen_fd;
    ds4_dist_coordinator_state *state = accept_ctx->state;

    for (;;) {
        struct sockaddr_storage ss;
        socklen_t slen = sizeof(ss);
        int fd = accept(listen_fd, (struct sockaddr *)&ss, &slen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (errno == EBADF || errno == EINVAL) break;
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: accept failed: %s\n", strerror(errno));
            continue;
        }
        dist_set_socket_low_latency(fd);

        ds4_dist_client_ctx *ctx = calloc(1, sizeof(*ctx));
        if (!ctx) {
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: out of memory accepting worker\n");
            close(fd);
            continue;
        }
        ctx->state = state;
        ctx->fd = fd;
        if (getnameinfo((struct sockaddr *)&ss, slen,
                        ctx->peer_host, sizeof(ctx->peer_host),
                        ctx->peer_port, sizeof(ctx->peer_port),
                        NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
            snprintf(ctx->peer_host, sizeof(ctx->peer_host), "unknown");
            snprintf(ctx->peer_port, sizeof(ctx->peer_port), "0");
        }

        pthread_t tid;
        if (pthread_create(&tid, NULL, dist_coordinator_client_main, ctx) != 0) {
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: pthread_create failed\n");
            close(fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }
    return NULL;
}

static uint64_t dist_make_session_id(const void *ptr) {
    uint64_t id = ((uint64_t)(uint32_t)time(NULL) << 32) ^ (uint64_t)getpid();
    id ^= ((uint64_t)(uintptr_t)ptr << 17) ^ (uint64_t)(uintptr_t)ptr;
    id ^= (uint64_t)clock();
    return id ? id : 1u;
}

static int dist_session_ensure_route(ds4_dist_session *d, char *err, size_t errlen) {
    if (!d) {
        if (errlen) snprintf(err, errlen, "missing distributed session");
        return 1;
    }
    uint64_t generation = dist_coordinator_generation(&d->state);
    if (d->plan_ready && d->plan_generation == generation) return 0;
    dist_route_plan_free(&d->plan);
    if (!dist_coordinator_ensure_route(&d->state, &d->plan, &generation, err, errlen)) {
        d->plan_ready = false;
        d->plan_generation = 0;
        return 1;
    }
    d->plan_ready = true;
    d->plan_generation = generation;
    return 0;
}

int ds4_dist_session_create(
        ds4_dist_session **out,
        ds4_engine *engine,
        const ds4_dist_options *opt,
        ds4_session *owner,
        int ctx_size,
        char *err,
        size_t errlen) {
    (void)owner;
    if (!out || !engine || !opt) {
        if (errlen) snprintf(err, errlen, "missing distributed session parameters");
        return 1;
    }
    *out = NULL;
    if (opt->role != DS4_DISTRIBUTED_COORDINATOR) {
        if (errlen) snprintf(err, errlen, "distributed session requires coordinator role");
        return 1;
    }
    if (dist_validate_options(opt, err, errlen) != 0) return 1;

    int listen_fd = dist_open_listener(opt->listen_host, opt->listen_port, err, errlen);
    if (listen_fd < 0) return 1;

    ds4_dist_session *d = calloc(1, sizeof(*d));
    if (!d) {
        close(listen_fd);
        if (errlen) snprintf(err, errlen, "out of memory creating distributed session");
        return 1;
    }

    d->listen_fd = listen_fd;
    d->state.engine = engine;
    d->state.model_id = (uint32_t)ds4_engine_model_id(engine);
    d->state.n_layers = (uint32_t)ds4_engine_layer_count(engine);
    d->state.local_start = opt->layers.start;
    d->state.local_end = dist_resolved_layer_end(opt, d->state.n_layers);
    d->state.ctx_size = ctx_size > 0 ? (uint32_t)ctx_size : 0u;
    d->state.local_has_output = opt->layers.has_output;
    d->state.local_can_output_head = true;
    d->state.replay_check = opt->replay_check;
    d->state.debug = opt->debug;
    d->state.use_control_for_work = true;
    d->state.prefill_chunk = opt->prefill_chunk;
    d->state.prefill_window = opt->prefill_window;
    d->state.activation_bits = dist_activation_bits_or_default(opt->activation_bits);
    pthread_mutex_init(&d->state.mu, NULL);
    d->session_id = dist_make_session_id(d);
    d->request_id = 1;

    char local_end[32];
    if (opt->layers.has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", opt->layers.end);
    DIST_COORD_DEBUG(&d->state,
                     "ds4: distributed coordinator API: listening on %s:%d model_id=%u layers=%u local=%u:%s activation_bits=%u\n",
                     opt->listen_host,
                     opt->listen_port,
                     d->state.model_id,
                     d->state.n_layers,
                     opt->layers.start,
                     local_end,
                     d->state.activation_bits);

    d->accept_ctx.state = &d->state;
    d->accept_ctx.listen_fd = listen_fd;
    if (pthread_create(&d->accept_tid, NULL, dist_coordinator_accept_main, &d->accept_ctx) != 0) {
        close(listen_fd);
        pthread_mutex_destroy(&d->state.mu);
        free(d);
        if (errlen) snprintf(err, errlen, "failed to start distributed coordinator accept loop");
        return 1;
    }
    pthread_detach(d->accept_tid);
    d->accept_started = true;
    *out = d;
    return 0;
}

void ds4_dist_session_free(ds4_dist_session *d) {
    if (!d) return;
    if (d->listen_fd >= 0) {
        close(d->listen_fd);
        d->listen_fd = -1;
    }
    dist_route_plan_free(&d->plan);
    pthread_mutex_lock(&d->state.mu);
    for (ds4_dist_worker_entry *it = d->state.workers; it; it = it->next) {
        if (it->fd >= 0) {
            close(it->fd);
            it->fd = -1;
        }
    }
    pthread_mutex_unlock(&d->state.mu);
    /* Client threads are detached and remove their registry entries after the
     * socket closes. Keep this small coordinator object process-lifetime to
     * avoid racing those threads during application shutdown. */
}

int ds4_dist_session_route_ready(ds4_dist_session *d, char *err, size_t errlen) {
    if (!d) {
        if (errlen) snprintf(err, errlen, "missing distributed session");
        return -1;
    }

    ds4_dist_route_plan probe = {0};
    if (!dist_coordinator_build_route_plan(&d->state, &probe, NULL, err, errlen)) {
        return 0;
    }
    dist_route_plan_free(&probe);
    if (errlen) err[0] = '\0';
    return 1;
}

int ds4_dist_session_sync(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        const ds4_tokens *prompt,
        float *logits,
        char *err,
        size_t errlen) {
    if (!d || !owner || !prompt || prompt->len <= 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid distributed sync request");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;

    if (checkpoint &&
        checkpoint->len >= 0 &&
        checkpoint->len <= prompt->len &&
        ds4_tokens_starts_with(prompt, checkpoint))
    {
        if (checkpoint->len == prompt->len) return 0;

        uint32_t chunk_cap = 0;
        if (dist_coordinator_prefill_chunk_cap(&d->state, owner, &chunk_cap, err, errlen) != 0) {
            return 1;
        }
        const uint32_t pos0 = (uint32_t)checkpoint->len;
        const uint32_t suffix = (uint32_t)prompt->len - pos0;
        if (dist_coordinator_can_pipeline_prefill(&d->state, &d->plan, owner, suffix, chunk_cap)) {
            int prefill_rc = dist_coordinator_prefill_prompt_pipelined(&d->state,
                                                                       owner,
                                                                       &d->plan,
                                                                       prompt,
                                                                       pos0,
                                                                       suffix,
                                                                       false,
                                                                       chunk_cap,
                                                                       d->session_id,
                                                                       &d->request_id,
                                                                       logits,
                                                                       err,
                                                                       errlen);
            if (prefill_rc != 0) {
                if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                             owner,
                                                             &d->plan,
                                                             prompt,
                                                             d->session_id,
                                                             &d->request_id,
                                                             logits,
                                                             &d->plan_generation,
                                                             prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                             err,
                                                             errlen) != 0) {
                    d->plan_ready = false;
                    d->plan_generation = 0;
                    return 1;
                }
                d->plan_ready = true;
            }
            return 0;
        }

        uint32_t pos = pos0;
        while (pos < (uint32_t)prompt->len) {
            const uint32_t remaining = (uint32_t)prompt->len - pos;
            const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
            int eval_rc = dist_coordinator_eval_span(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt->v + pos,
                                                     chunk,
                                                     pos,
                                                     d->session_id,
                                                     d->request_id++,
                                                     false,
                                                     logits,
                                                     err,
                                                     errlen);
            if (eval_rc != 0) {
                if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                             owner,
                                                             &d->plan,
                                                             prompt,
                                                             d->session_id,
                                                             &d->request_id,
                                                             logits,
                                                             &d->plan_generation,
                                                             eval_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                             err,
                                                             errlen) != 0) {
                    d->plan_ready = false;
                    d->plan_generation = 0;
                    return 1;
                }
                d->plan_ready = true;
                return 0;
            }
            pos += chunk;
            dist_report_prefill_progress(owner, pos, (uint32_t)prompt->len);
        }
        return 0;
    }

    int prefill_rc = dist_coordinator_prefill_prompt(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     err,
                                                     errlen);
    if (prefill_rc != 0) {
        if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     &d->plan_generation,
                                                     prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     errlen) != 0) {
            d->plan_ready = false;
            d->plan_generation = 0;
            return 1;
        }
        d->plan_ready = true;
    }
    return 0;
}

int ds4_dist_session_eval(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        int token,
        float *logits,
        char *err,
        size_t errlen) {
    if (!d || !owner || !checkpoint || checkpoint->len < 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid distributed decode request");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;

    ds4_tokens transcript = {0};
    ds4_tokens_copy(&transcript, checkpoint);
    ds4_tokens_push(&transcript, token);

    int rc = dist_coordinator_eval_span(&d->state,
                                        owner,
                                        &d->plan,
                                        &token,
                                        1,
                                        (uint32_t)checkpoint->len,
                                        d->session_id,
                                        d->request_id++,
                                        false,
                                        logits,
                                        err,
                                        errlen);
    if (rc != 0) {
        if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     &transcript,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     &d->plan_generation,
                                                     rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     errlen) != 0) {
            d->plan_ready = false;
            d->plan_generation = 0;
            ds4_tokens_free(&transcript);
            return 1;
        }
        d->plan_ready = true;
        rc = 0;
    }
    ds4_tokens_free(&transcript);
    return rc;
}

static int dist_run_coordinator(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen) {
    char err[256];
    int listen_fd = dist_open_listener(opt->listen_host, opt->listen_port, err, sizeof(err));
    if (listen_fd < 0) {
        fprintf(stderr, "ds4: distributed coordinator: %s\n", err);
        return 1;
    }

    ds4_dist_coordinator_state state;
    memset(&state, 0, sizeof(state));
    state.engine = engine;
    state.model_id = (uint32_t)ds4_engine_model_id(engine);
    state.n_layers = (uint32_t)ds4_engine_layer_count(engine);
    state.local_start = opt->layers.start;
    state.local_end = dist_resolved_layer_end(opt, state.n_layers);
    state.ctx_size = gen && gen->ctx_size > 0 ? (uint32_t)gen->ctx_size : 0u;
    state.local_has_output = opt->layers.has_output;
    state.local_can_output_head = true;
    state.replay_check = opt->replay_check;
    state.debug = opt->debug;
    state.use_control_for_work = gen && gen->prompt;
    state.prefill_chunk = opt->prefill_chunk;
    state.prefill_window = opt->prefill_window;
    state.activation_bits = dist_activation_bits_or_default(opt->activation_bits);
    pthread_mutex_init(&state.mu, NULL);

    char local_end[32];
    if (opt->layers.has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", opt->layers.end);
    DIST_COORD_DEBUG(&state,
                     "ds4: distributed coordinator: listening on %s:%d model_id=%u layers=%u local=%u:%s activation_bits=%u\n",
                     opt->listen_host,
                     opt->listen_port,
                     state.model_id,
                     state.n_layers,
                     opt->layers.start,
                     local_end,
                     state.activation_bits);

    ds4_dist_accept_ctx accept_ctx = {
        .state = &state,
        .listen_fd = listen_fd,
    };
    if (!gen || !gen->prompt) {
        dist_coordinator_accept_main(&accept_ctx);
        return 0;
    }

    pthread_t accept_tid;
    if (pthread_create(&accept_tid, NULL, dist_coordinator_accept_main, &accept_ctx) != 0) {
        fprintf(stderr, "ds4: distributed coordinator: pthread_create failed for accept loop\n");
        close(listen_fd);
        return 1;
    }
    pthread_detach(accept_tid);

    return dist_run_coordinator_generation(&state, gen);
}

static int dist_worker_read_loop(ds4_dist_worker_state *state, int fd) {
    ds4_dist_worker_upstream upstream;
    dist_worker_upstream_init(&upstream, state, fd);
    int loop_rc = 0;

    for (;;) {
        uint32_t type = 0, bytes = 0;
        char err[256];
        int rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
        if (rc == 0) break;
        if (rc < 0) {
            fprintf(stderr, "ds4: distributed worker: protocol error: %s\n", err);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_ERROR) {
            char msg[512];
            uint32_t n = bytes < sizeof(msg) - 1u ? bytes : (uint32_t)sizeof(msg) - 1u;
            rc = dist_read_full(fd, msg, n);
            if (rc <= 0) {
                loop_rc = 1;
                break;
            }
            msg[n] = '\0';
            if (bytes > n) dist_discard_bytes(fd, bytes - n);
            fprintf(stderr, "ds4: distributed worker: coordinator error: %s\n", msg);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_WORK) {
            rc = dist_worker_handle_work(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            continue;
        }
        rc = dist_discard_bytes(fd, bytes);
        if (rc <= 0) {
            loop_rc = rc == 0 ? 0 : 1;
            break;
        }
        pthread_mutex_lock(&upstream.write_mu);
        dist_send_error(fd, "unsupported distributed worker frame");
        pthread_mutex_unlock(&upstream.write_mu);
        fprintf(stderr, "ds4: distributed worker: rejected unsupported frame type %u\n", type);
        loop_rc = 1;
        break;
    }

    dist_worker_upstream_destroy(&upstream);
    return loop_rc;
}

static int dist_send_work_result(
        int fd,
        uint64_t request_id,
        uint64_t result_hash,
        uint32_t status,
        uint32_t result_kind,
        uint32_t payload_bits,
        const ds4_dist_telemetry_fixed *telemetry,
        uint32_t telemetry_count,
        const void *payload,
        uint32_t payload_bytes) {
    if (payload_bytes != 0 && !payload) return -1;
    if (telemetry_count != 0 && !telemetry) return -1;
    uint32_t wire_payload_bytes = payload_bytes;
    uint64_t hidden_values = 0;
    if (status == 0 && result_kind == DS4_DIST_RESULT_HIDDEN_STATE) {
        payload_bits = dist_activation_bits_or_default(payload_bits);
        if (!dist_activation_bits_valid(payload_bits) ||
            (payload_bytes % (uint32_t)sizeof(float)) != 0)
            return -1;
        hidden_values = payload_bytes / (uint32_t)sizeof(float);
        if (!dist_activation_wire_bytes(payload_bits, hidden_values, &wire_payload_bytes))
            return -1;
    } else if (status == 0 && result_kind == DS4_DIST_RESULT_LOGITS) {
        payload_bits = 32u;
    } else {
        payload_bits = 0;
    }
    const uint64_t telemetry_bytes64 =
        (uint64_t)telemetry_count * sizeof(ds4_dist_telemetry_fixed);
    if (telemetry_bytes64 > UINT32_MAX) return -1;
    const uint32_t telemetry_bytes = (uint32_t)telemetry_bytes64;
    const uint64_t frame_bytes = sizeof(ds4_dist_result_fixed) +
                                 telemetry_bytes64 +
                                 (uint64_t)wire_payload_bytes;
    if (frame_bytes > UINT32_MAX) return -1;

    ds4_dist_result_fixed r;
    dist_u64_to_halves(request_id, &r.request_hi, &r.request_lo);
    dist_u64_to_halves(status == 0 ? result_hash : 0,
                       &r.result_hash_hi,
                       &r.result_hash_lo);
    r.status = status;
    r.result_kind = result_kind;
    r.telemetry_count = telemetry_count;
    r.telemetry_bytes = telemetry_bytes;
    r.payload_bytes = wire_payload_bytes;
    r.payload_bits = payload_bits;

    ds4_dist_result_fixed wire = r;
    dist_result_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_RESULT, (uint32_t)frame_bytes) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < telemetry_count; i++) {
        ds4_dist_telemetry_fixed tw = telemetry[i];
        dist_telemetry_to_wire(&tw);
        if (dist_write_full(fd, &tw, sizeof(tw)) != 0) return -1;
    }
    if (status == 0 && result_kind == DS4_DIST_RESULT_HIDDEN_STATE && wire_payload_bytes != 0) {
        if (dist_write_activation_payload(fd, payload, hidden_values, payload_bits) != 0) return -1;
    } else if (payload_bytes && payload && dist_write_full(fd, payload, payload_bytes) != 0) {
        return -1;
    }
    return 1;
}

static int dist_send_work_error(int fd, uint64_t request_id, const char *msg) {
    if (!msg) msg = "distributed work failed";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    return dist_send_work_result(fd, request_id, 0, 1, 0, 0, NULL, 0, msg, (uint32_t)len);
}

static int dist_send_work_frame(
        int fd,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *input_hc,
        const void *route_blob);

static bool dist_route_get_entry(
        const void *route_blob,
        uint32_t route_bytes,
        uint32_t route_count,
        uint32_t target_index,
        ds4_dist_route_entry *out,
        char *err,
        size_t errlen) {
    if (!route_blob || !out || target_index >= route_count) {
        if (errlen) snprintf(err, errlen, "invalid route entry index");
        return false;
    }
    const uint8_t *p = route_blob;
    uint32_t remaining = route_bytes;
    for (uint32_t i = 0; i < route_count; i++) {
        if (remaining < sizeof(ds4_dist_route_fixed)) {
            if (errlen) snprintf(err, errlen, "truncated route entry");
            return false;
        }
        ds4_dist_route_fixed fixed;
        memcpy(&fixed, p, sizeof(fixed));
        dist_route_from_wire(&fixed);
        p += sizeof(fixed);
        remaining -= (uint32_t)sizeof(fixed);
        if (fixed.host_len == 0 || fixed.host_len >= NI_MAXHOST || fixed.host_len > remaining) {
            if (errlen) snprintf(err, errlen, "invalid route host length");
            return false;
        }
        if (dist_bytes_have_nul(p, fixed.host_len)) {
            if (errlen) snprintf(err, errlen, "route host contains NUL bytes");
            return false;
        }
        if (i == target_index) {
            memcpy(out->host, p, fixed.host_len);
            out->host[fixed.host_len] = '\0';
            out->port = fixed.port;
            out->layer_start = fixed.layer_start;
            out->layer_end = fixed.layer_end;
            out->flags = fixed.flags;
            out->fd = -1;
            return true;
        }
        p += fixed.host_len;
        remaining -= fixed.host_len;
    }
    if (remaining != 0) {
        if (errlen) snprintf(err, errlen, "route payload has trailing bytes");
        return false;
    }
    if (errlen) snprintf(err, errlen, "route entry not found");
    return false;
}

static bool dist_route_get_return_target(
        const void *route_blob,
        uint32_t route_bytes,
        uint32_t route_count,
        ds4_dist_route_return *out,
        char *err,
        size_t errlen) {
    if (!route_blob || !out || route_count == 0) {
        if (errlen) snprintf(err, errlen, "invalid route final destination");
        return false;
    }
    const uint8_t *p = route_blob;
    uint32_t remaining = route_bytes;
    for (uint32_t i = 0; i < route_count; i++) {
        if (remaining < sizeof(ds4_dist_route_fixed)) {
            if (errlen) snprintf(err, errlen, "truncated route entry");
            return false;
        }
        ds4_dist_route_fixed fixed;
        memcpy(&fixed, p, sizeof(fixed));
        dist_route_from_wire(&fixed);
        p += sizeof(fixed);
        remaining -= (uint32_t)sizeof(fixed);
        if (fixed.host_len > remaining) {
            if (errlen) snprintf(err, errlen, "invalid route host length");
            return false;
        }
        if (dist_bytes_have_nul(p, fixed.host_len)) {
            if (errlen) snprintf(err, errlen, "route host contains NUL bytes");
            return false;
        }
        p += fixed.host_len;
        remaining -= fixed.host_len;
    }
    if (remaining < sizeof(ds4_dist_route_return_fixed)) {
        if (errlen) snprintf(err, errlen, "route payload missing final destination");
        return false;
    }
    ds4_dist_route_return_fixed fixed;
    memcpy(&fixed, p, sizeof(fixed));
    dist_route_return_from_wire(&fixed);
    p += sizeof(fixed);
    remaining -= (uint32_t)sizeof(fixed);
    if (fixed.host_len >= NI_MAXHOST || fixed.host_len > remaining) {
        if (errlen) snprintf(err, errlen, "invalid route final destination host length");
        return false;
    }
    if (dist_bytes_have_nul(p, fixed.host_len)) {
        if (errlen) snprintf(err, errlen, "route final destination host contains NUL bytes");
        return false;
    }
    memset(out, 0, sizeof(*out));
    out->kind = fixed.kind;
    out->port = fixed.port;
    if (fixed.host_len) {
        memcpy(out->host, p, fixed.host_len);
        out->host[fixed.host_len] = '\0';
    }
    p += fixed.host_len;
    remaining -= fixed.host_len;
    if (remaining != 0) {
        if (errlen) snprintf(err, errlen, "route payload has trailing bytes");
        return false;
    }
    return true;
}

static bool dist_route_validate_blob(
        const void *route_blob,
        uint32_t route_bytes,
        uint32_t route_count,
        uint32_t n_layers,
        char *err,
        size_t errlen) {
    if (route_count == 0) {
        if (route_bytes == 0) return true;
        if (errlen) snprintf(err, errlen, "route payload has entries without a route count");
        return false;
    }
    if (!route_blob) {
        if (errlen) snprintf(err, errlen, "route payload is missing");
        return false;
    }

    const uint8_t *p = route_blob;
    uint32_t remaining = route_bytes;
    uint32_t prev_end = UINT32_MAX;
    for (uint32_t i = 0; i < route_count; i++) {
        if (remaining < sizeof(ds4_dist_route_fixed)) {
            if (errlen) snprintf(err, errlen, "truncated route entry");
            return false;
        }
        ds4_dist_route_fixed fixed;
        memcpy(&fixed, p, sizeof(fixed));
        dist_route_from_wire(&fixed);
        p += sizeof(fixed);
        remaining -= (uint32_t)sizeof(fixed);

        if (fixed.host_len == 0 || fixed.host_len >= NI_MAXHOST || fixed.host_len > remaining) {
            if (errlen) snprintf(err, errlen, "invalid route host length");
            return false;
        }
        if (dist_bytes_have_nul(p, fixed.host_len)) {
            if (errlen) snprintf(err, errlen, "route host contains NUL bytes");
            return false;
        }
        if (fixed.port == 0 || fixed.port > 65535u) {
            if (errlen) snprintf(err, errlen, "invalid route port");
            return false;
        }
        if (fixed.layer_start >= n_layers || fixed.layer_end >= n_layers ||
            fixed.layer_end < fixed.layer_start) {
            if (errlen) snprintf(err, errlen, "invalid route layer range");
            return false;
        }
        if ((fixed.flags & ~DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0) {
            if (errlen) snprintf(err, errlen, "invalid route flags");
            return false;
        }
        if ((fixed.flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0 &&
            fixed.layer_end + 1u != n_layers) {
            if (errlen) snprintf(err, errlen, "route logits require final layer");
            return false;
        }
        if (i != 0 && fixed.layer_start != prev_end + 1u) {
            if (errlen) snprintf(err, errlen, "route layer ranges are not contiguous");
            return false;
        }

        p += fixed.host_len;
        remaining -= fixed.host_len;
        prev_end = fixed.layer_end;
    }
    if (remaining < sizeof(ds4_dist_route_return_fixed)) {
        if (errlen) snprintf(err, errlen, "route payload missing final destination");
        return false;
    }
    ds4_dist_route_return_fixed ret;
    memcpy(&ret, p, sizeof(ret));
    dist_route_return_from_wire(&ret);
    p += sizeof(ret);
    remaining -= (uint32_t)sizeof(ret);
    if (ret.host_len >= NI_MAXHOST || ret.host_len > remaining) {
        if (errlen) snprintf(err, errlen, "invalid route final destination host length");
        return false;
    }
    if (dist_bytes_have_nul(p, ret.host_len)) {
        if (errlen) snprintf(err, errlen, "route final destination host contains NUL bytes");
        return false;
    }
    if (ret.kind != DS4_DIST_ROUTE_RETURN_UPSTREAM) {
        if (errlen) snprintf(err, errlen, "unsupported route final destination");
        return false;
    }
    if (ret.host_len != 0 || ret.port != 0) {
        if (errlen) snprintf(err, errlen, "invalid upstream route final destination");
        return false;
    }
    p += ret.host_len;
    remaining -= ret.host_len;
    if (remaining != 0) {
        if (errlen) snprintf(err, errlen, "route payload has trailing bytes");
        return false;
    }
    return true;
}

static int dist_send_work_frame(
        int fd,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *input_hc,
        const void *route_blob) {
    if (!work || !tokens || work->n_tokens == 0) return -1;
    const uint64_t token_bytes = (uint64_t)work->n_tokens * sizeof(uint32_t);
    if (token_bytes > UINT32_MAX || work->token_bytes != (uint32_t)token_bytes) return -1;
    if (work->input_hc_bytes != 0 && !input_hc) return -1;
    if (work->route_bytes != 0 && !route_blob) return -1;
    uint64_t input_hc_values = 0;
    if (work->input_hc_bytes != 0 &&
        !dist_activation_values_from_wire_bytes(work->input_hc_bits,
                                                work->input_hc_bytes,
                                                &input_hc_values))
        return -1;
    const uint64_t frame_bytes = sizeof(ds4_dist_work_fixed) +
                                 (uint64_t)work->token_bytes +
                                 work->input_hc_bytes +
                                 work->route_bytes;
    if (frame_bytes > UINT32_MAX) return -1;

    ds4_dist_work_fixed wire = *work;
    dist_work_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_WORK, (uint32_t)frame_bytes) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < work->n_tokens; i++) {
        uint32_t t = htonl((uint32_t)tokens[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    if (work->input_hc_bytes &&
        dist_write_activation_payload(fd,
                                      input_hc,
                                      input_hc_values,
                                      work->input_hc_bits) != 0)
        return -1;
    if (work->route_bytes && dist_write_full(fd, route_blob, work->route_bytes) != 0) return -1;
    return 0;
}

static int dist_worker_upstream_send_work_result(
        ds4_dist_worker_upstream *upstream,
        uint64_t request_id,
        uint64_t result_hash,
        uint32_t status,
        uint32_t result_kind,
        uint32_t payload_bits,
        const ds4_dist_telemetry_fixed *telemetry,
        uint32_t telemetry_count,
        const void *payload,
        uint32_t payload_bytes) {
    pthread_mutex_lock(&upstream->write_mu);
    int rc = dist_send_work_result(upstream->fd,
                                   request_id,
                                   result_hash,
                                   status,
                                   result_kind,
                                   payload_bits,
                                   telemetry,
                                   telemetry_count,
                                   payload,
                                   payload_bytes);
    pthread_mutex_unlock(&upstream->write_mu);
    return rc;
}

static int dist_worker_upstream_send_work_error(
        ds4_dist_worker_upstream *upstream,
        uint64_t request_id,
        const char *msg) {
    pthread_mutex_lock(&upstream->write_mu);
    int rc = dist_send_work_error(upstream->fd, request_id, msg);
    pthread_mutex_unlock(&upstream->write_mu);
    return rc;
}

static void dist_worker_upstream_init(
        ds4_dist_worker_upstream *upstream,
        ds4_dist_worker_state *state,
        int fd) {
    memset(upstream, 0, sizeof(*upstream));
    upstream->state = state;
    upstream->fd = fd;
    pthread_mutex_init(&upstream->write_mu, NULL);
    pthread_mutex_init(&upstream->forward_mu, NULL);
}

static bool dist_worker_forwarder_enqueue_request(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t request_id,
        const ds4_dist_telemetry_fixed *telemetry,
        double downstream_t0) {
    pthread_mutex_lock(&forwarder->queue_mu);
    while (!forwarder->closing &&
           forwarder->pending_depth != 0 &&
           forwarder->pending_count >= forwarder->pending_depth) {
        pthread_cond_wait(&forwarder->queue_not_full, &forwarder->queue_mu);
    }
    if (forwarder->closing) {
        pthread_mutex_unlock(&forwarder->queue_mu);
        return false;
    }
    ds4_dist_pending_request *node = calloc(1, sizeof(*node));
    if (!node) {
        pthread_mutex_unlock(&forwarder->queue_mu);
        return false;
    }
    node->request_id = request_id;
    node->downstream_t0 = downstream_t0;
    if (telemetry) node->telemetry = *telemetry;
    if (forwarder->pending_tail) forwarder->pending_tail->next = node;
    else forwarder->pending_head = node;
    forwarder->pending_tail = node;
    forwarder->pending_count++;
    pthread_mutex_unlock(&forwarder->queue_mu);
    return true;
}

static bool dist_worker_forwarder_pop_request(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t *request_id,
        ds4_dist_telemetry_fixed *telemetry,
        double *downstream_t0) {
    pthread_mutex_lock(&forwarder->queue_mu);
    ds4_dist_pending_request *node = forwarder->pending_head;
    if (!node) {
        pthread_mutex_unlock(&forwarder->queue_mu);
        return false;
    }
    forwarder->pending_head = node->next;
    if (!forwarder->pending_head) forwarder->pending_tail = NULL;
    if (forwarder->pending_count != 0) forwarder->pending_count--;
    pthread_cond_signal(&forwarder->queue_not_full);
    pthread_mutex_unlock(&forwarder->queue_mu);

    if (request_id) *request_id = node->request_id;
    if (telemetry) *telemetry = node->telemetry;
    if (downstream_t0) *downstream_t0 = node->downstream_t0;
    free(node);
    return true;
}

static bool dist_worker_forwarder_remove_request(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t request_id) {
    pthread_mutex_lock(&forwarder->queue_mu);
    ds4_dist_pending_request **link = &forwarder->pending_head;
    ds4_dist_pending_request *prev = NULL;
    while (*link) {
        ds4_dist_pending_request *node = *link;
        if (node->request_id == request_id) {
            *link = node->next;
            if (forwarder->pending_tail == node) forwarder->pending_tail = prev;
            if (forwarder->pending_count != 0) forwarder->pending_count--;
            pthread_cond_signal(&forwarder->queue_not_full);
            pthread_mutex_unlock(&forwarder->queue_mu);
            free(node);
            return true;
        }
        prev = node;
        link = &node->next;
    }
    pthread_mutex_unlock(&forwarder->queue_mu);
    return false;
}

static void dist_worker_forwarder_note_send_done(
        ds4_dist_worker_forwarder *forwarder,
        uint64_t request_id,
        uint32_t forward_send_usec,
        double downstream_t0) {
    pthread_mutex_lock(&forwarder->queue_mu);
    for (ds4_dist_pending_request *node = forwarder->pending_head; node; node = node->next) {
        if (node->request_id == request_id) {
            node->telemetry.forward_send_usec = forward_send_usec;
            node->downstream_t0 = downstream_t0;
            break;
        }
    }
    pthread_mutex_unlock(&forwarder->queue_mu);
}

static void dist_worker_forwarder_clear_requests(ds4_dist_worker_forwarder *forwarder) {
    pthread_mutex_lock(&forwarder->queue_mu);
    ds4_dist_pending_request *it = forwarder->pending_head;
    forwarder->pending_head = NULL;
    forwarder->pending_tail = NULL;
    forwarder->pending_count = 0;
    pthread_cond_broadcast(&forwarder->queue_not_full);
    pthread_mutex_unlock(&forwarder->queue_mu);

    while (it) {
        ds4_dist_pending_request *next = it->next;
        free(it);
        it = next;
    }
}

static void dist_worker_forwarder_close_queue(ds4_dist_worker_forwarder *forwarder) {
    pthread_mutex_lock(&forwarder->queue_mu);
    forwarder->closing = true;
    pthread_cond_broadcast(&forwarder->queue_not_full);
    pthread_mutex_unlock(&forwarder->queue_mu);
}

static void *dist_worker_forwarder_relay_main(void *arg) {
    ds4_dist_worker_forwarder *forwarder = arg;
    ds4_dist_worker_upstream *upstream = forwarder->upstream;
    int fd = forwarder->fd;
    uint8_t *buf = malloc(1024 * 1024);
    if (!buf) {
        shutdown(upstream->fd, SHUT_RDWR);
        return NULL;
    }
    DIST_DEBUG("relay start downstream=%s:%u fd=%d upstream_fd=%d",
               forwarder->host,
               forwarder->port,
               fd,
               upstream->fd);

    for (;;) {
        uint32_t type = 0, bytes = 0;
        char err[256];
        int rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
        if (rc <= 0) {
            uint64_t pending_request = 0;
            if (dist_worker_forwarder_pop_request(forwarder, &pending_request, NULL, NULL)) {
                dist_worker_upstream_send_work_error(upstream,
                                                     pending_request,
                                                     "next worker closed connection");
            }
            DIST_DEBUG("relay read header end downstream=%s:%u rc=%d err=%s",
                       forwarder->host,
                       forwarder->port,
                       rc,
                       err);
            break;
        }
        DIST_DEBUG("relay got frame downstream=%s:%u type=%u bytes=%u",
                   forwarder->host,
                   forwarder->port,
                   type,
                   bytes);
        uint64_t expected_request = 0;
        if (type != DS4_DIST_MSG_RESULT || bytes < sizeof(ds4_dist_result_fixed)) {
            dist_discard_bytes(fd, bytes);
            if (dist_worker_forwarder_pop_request(forwarder, &expected_request, NULL, NULL)) {
                dist_worker_upstream_send_work_error(upstream,
                                                     expected_request,
                                                     "next worker did not return valid RESULT");
            }
            DIST_DEBUG("relay invalid frame downstream=%s:%u", forwarder->host, forwarder->port);
            break;
        }

        ds4_dist_result_fixed wire_result;
        rc = dist_read_full(fd, &wire_result, sizeof(wire_result));
        if (rc <= 0) {
            if (dist_worker_forwarder_pop_request(forwarder, &expected_request, NULL, NULL)) {
                dist_worker_upstream_send_work_error(upstream,
                                                     expected_request,
                                                     "next worker closed while returning RESULT");
            }
            DIST_DEBUG("relay read result fixed failed downstream=%s:%u rc=%d",
                       forwarder->host,
                       forwarder->port,
                       rc);
            break;
        }

        ds4_dist_result_fixed result = wire_result;
        dist_result_from_wire(&result);
        const uint64_t got_request = dist_u64_from_halves(result.request_hi, result.request_lo);
        const uint32_t body_bytes = bytes - (uint32_t)sizeof(wire_result);
        if (result.telemetry_bytes % (uint32_t)sizeof(ds4_dist_telemetry_fixed) != 0 ||
            result.telemetry_count != result.telemetry_bytes / (uint32_t)sizeof(ds4_dist_telemetry_fixed) ||
            result.telemetry_bytes > body_bytes ||
            result.payload_bytes != body_bytes - result.telemetry_bytes) {
            dist_discard_bytes(fd, body_bytes);
            if (dist_worker_forwarder_pop_request(forwarder, &expected_request, NULL, NULL)) {
                dist_worker_upstream_send_work_error(upstream,
                                                     expected_request,
                                                     "next worker RESULT metadata mismatch");
            }
            DIST_DEBUG("relay result metadata mismatch downstream=%s:%u telemetry=%u payload=%u frame=%u",
                       forwarder->host,
                       forwarder->port,
                       result.telemetry_bytes,
                       result.payload_bytes,
                       body_bytes);
            break;
        }
        ds4_dist_telemetry_fixed local_telemetry;
        double downstream_t0 = 0.0;
        if (!dist_worker_forwarder_pop_request(forwarder, &expected_request, &local_telemetry, &downstream_t0)) {
            dist_discard_bytes(fd, body_bytes);
            DIST_DEBUG("relay got unexpected result request=%llu with no pending request",
                       (unsigned long long)got_request);
            break;
        }
        if (got_request != expected_request) {
            dist_discard_bytes(fd, body_bytes);
            dist_worker_upstream_send_work_error(upstream,
                                                 expected_request,
                                                 "next worker RESULT metadata mismatch");
            DIST_DEBUG("relay request mismatch expected=%llu got=%llu",
                       (unsigned long long)expected_request,
                       (unsigned long long)got_request);
            break;
        }
        local_telemetry.downstream_wait_usec = dist_usec_since(downstream_t0, dist_now_sec());
        const uint64_t out_telemetry_bytes64 =
            (uint64_t)result.telemetry_bytes + sizeof(ds4_dist_telemetry_fixed);
        const uint32_t out_telemetry_count = result.telemetry_count + 1u;
        if (out_telemetry_bytes64 > UINT32_MAX || out_telemetry_count == 0) {
            dist_discard_bytes(fd, body_bytes);
            dist_worker_upstream_send_work_error(upstream,
                                                 expected_request,
                                                 "distributed telemetry chain is too large");
            break;
        }
        const uint32_t out_telemetry_bytes = (uint32_t)out_telemetry_bytes64;
        const uint64_t out_frame_bytes64 = sizeof(ds4_dist_result_fixed) +
                                           out_telemetry_bytes64 +
                                           (uint64_t)result.payload_bytes;
        if (out_frame_bytes64 > UINT32_MAX) {
            dist_discard_bytes(fd, body_bytes);
            dist_worker_upstream_send_work_error(upstream,
                                                 expected_request,
                                                 "distributed RESULT frame is too large");
            break;
        }
        DIST_DEBUG("relay result request=%llu status=%u kind=%u telemetry=%u payload=%u",
                   (unsigned long long)got_request,
                   result.status,
                   result.result_kind,
                   out_telemetry_count,
                   result.payload_bytes);

        pthread_mutex_lock(&upstream->write_mu);
        result.telemetry_count = out_telemetry_count;
        result.telemetry_bytes = out_telemetry_bytes;
        ds4_dist_result_fixed out_wire_result = result;
        dist_result_to_wire(&out_wire_result);
        int write_rc = dist_write_frame_header(upstream->fd,
                                               DS4_DIST_MSG_RESULT,
                                               (uint32_t)out_frame_bytes64);
        if (write_rc == 0) write_rc = dist_write_full(upstream->fd, &out_wire_result, sizeof(out_wire_result));

        uint32_t remaining = result.telemetry_bytes - (uint32_t)sizeof(ds4_dist_telemetry_fixed);
        while (write_rc == 0 && remaining > 0) {
            uint32_t n = remaining < 1024u * 1024u ? remaining : 1024u * 1024u;
            rc = dist_read_full(fd, buf, n);
            if (rc <= 0) {
                write_rc = -1;
                break;
            }
            if (dist_write_full(upstream->fd, buf, n) != 0) {
                write_rc = -1;
                break;
            }
            remaining -= n;
        }
        if (write_rc == 0) {
            ds4_dist_telemetry_fixed local_wire = local_telemetry;
            dist_telemetry_to_wire(&local_wire);
            if (dist_write_full(upstream->fd, &local_wire, sizeof(local_wire)) != 0) {
                write_rc = -1;
            }
        }

        remaining = result.payload_bytes;
        while (write_rc == 0 && remaining > 0) {
            uint32_t n = remaining < 1024u * 1024u ? remaining : 1024u * 1024u;
            rc = dist_read_full(fd, buf, n);
            if (rc <= 0) {
                write_rc = -1;
                break;
            }
            if (dist_write_full(upstream->fd, buf, n) != 0) {
                write_rc = -1;
                break;
            }
            remaining -= n;
        }
        pthread_mutex_unlock(&upstream->write_mu);

        DIST_DEBUG("relay wrote result request=%llu write_rc=%d remaining=%u",
                   (unsigned long long)got_request,
                   write_rc,
                   remaining);
        if (write_rc != 0) break;
        if (remaining != 0) break;
    }

    DIST_DEBUG("relay closing upstream_fd=%d", upstream->fd);
    dist_worker_forwarder_close_queue(forwarder);
    shutdown(upstream->fd, SHUT_RDWR);
    free(buf);
    return NULL;
}

static ds4_dist_worker_forwarder *dist_worker_get_forwarder(
        ds4_dist_worker_upstream *upstream,
        const char *host,
        uint32_t port,
        char *err,
        size_t errlen) {
    pthread_mutex_lock(&upstream->forward_mu);
    for (ds4_dist_worker_forwarder *it = upstream->forwarders; it; it = it->next) {
        if (it->port == port && !strcmp(it->host, host)) {
            pthread_mutex_unlock(&upstream->forward_mu);
            return it;
        }
    }

    int fd = dist_connect_endpoint(host, (int)port, err, errlen);
    if (fd < 0) {
        pthread_mutex_unlock(&upstream->forward_mu);
        return NULL;
    }

    ds4_dist_worker_forwarder *forwarder = calloc(1, sizeof(*forwarder));
    if (!forwarder) {
        close(fd);
        pthread_mutex_unlock(&upstream->forward_mu);
        if (errlen) snprintf(err, errlen, "out of memory creating worker-to-worker forwarder");
        return NULL;
    }
    forwarder->upstream = upstream;
    snprintf(forwarder->host, sizeof(forwarder->host), "%s", host);
    forwarder->port = port;
    forwarder->fd = fd;
    forwarder->pending_depth = dist_worker_forward_window();
    pthread_mutex_init(&forwarder->send_mu, NULL);
    pthread_mutex_init(&forwarder->queue_mu, NULL);
    pthread_cond_init(&forwarder->queue_not_full, NULL);
    if (pthread_create(&forwarder->tid, NULL, dist_worker_forwarder_relay_main, forwarder) != 0) {
        pthread_cond_destroy(&forwarder->queue_not_full);
        pthread_mutex_destroy(&forwarder->queue_mu);
        pthread_mutex_destroy(&forwarder->send_mu);
        close(fd);
        free(forwarder);
        pthread_mutex_unlock(&upstream->forward_mu);
        if (errlen) snprintf(err, errlen, "failed to start worker-to-worker relay thread");
        return NULL;
    }
    forwarder->thread_started = true;
    forwarder->next = upstream->forwarders;
    upstream->forwarders = forwarder;
    pthread_mutex_unlock(&upstream->forward_mu);

    fprintf(stderr,
            "ds4: distributed worker: opened pipelined worker-to-worker connection to %s:%u (window %u)\n",
            host,
            port,
            forwarder->pending_depth);
    return forwarder;
}

static void dist_worker_upstream_destroy(ds4_dist_worker_upstream *upstream) {
    pthread_mutex_lock(&upstream->forward_mu);
    ds4_dist_worker_forwarder *forwarders = upstream->forwarders;
    upstream->forwarders = NULL;
    pthread_mutex_unlock(&upstream->forward_mu);

    for (ds4_dist_worker_forwarder *it = forwarders; it; it = it->next) {
        dist_worker_forwarder_close_queue(it);
        if (it->fd >= 0) shutdown(it->fd, SHUT_RDWR);
    }
    while (forwarders) {
        ds4_dist_worker_forwarder *next = forwarders->next;
        if (forwarders->thread_started) pthread_join(forwarders->tid, NULL);
        if (forwarders->fd >= 0) close(forwarders->fd);
        dist_worker_forwarder_clear_requests(forwarders);
        pthread_cond_destroy(&forwarders->queue_not_full);
        pthread_mutex_destroy(&forwarders->queue_mu);
        pthread_mutex_destroy(&forwarders->send_mu);
        free(forwarders);
        forwarders = next;
    }

    pthread_mutex_destroy(&upstream->forward_mu);
    pthread_mutex_destroy(&upstream->write_mu);
}

static int dist_forward_work_to_next(
        ds4_dist_worker_upstream *upstream,
        const ds4_dist_route_entry *next,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *hidden_hc,
        uint32_t hidden_hc_bytes,
        const ds4_dist_telemetry_fixed *telemetry,
        const void *route_blob) {
    char err[256];
    ds4_dist_worker_forwarder *forwarder =
        dist_worker_get_forwarder(upstream, next->host, next->port, err, sizeof(err));
    const uint64_t request_id = dist_u64_from_halves(work->request_hi, work->request_lo);
    if (!forwarder) {
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }

    ds4_dist_work_fixed forwarded = *work;
    forwarded.layer_start = next->layer_start;
    forwarded.layer_end = next->layer_end;
    forwarded.route_index = work->route_index + 1u;
    forwarded.flags |= DS4_DIST_WORK_F_INPUT_HC;
    forwarded.input_hc_bits = dist_activation_bits_or_default(work->input_hc_bits);
    if (!dist_activation_wire_bytes_from_f32_bytes(forwarded.input_hc_bits,
                                                   hidden_hc_bytes,
                                                   &forwarded.input_hc_bytes)) {
        return dist_worker_upstream_send_work_error(upstream,
                                                    request_id,
                                                    "invalid forwarded hidden-state size");
    }
    if ((next->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0) {
        forwarded.flags |= DS4_DIST_WORK_F_OUTPUT_LOGITS;
    } else {
        forwarded.flags &= ~DS4_DIST_WORK_F_OUTPUT_LOGITS;
    }

    pthread_mutex_lock(&forwarder->send_mu);
    const double send_t0 = dist_now_sec();
    if (!dist_worker_forwarder_enqueue_request(forwarder, request_id, telemetry, send_t0)) {
        pthread_mutex_unlock(&forwarder->send_mu);
        return dist_worker_upstream_send_work_error(upstream,
                                                    request_id,
                                                    "out of memory tracking forwarded request");
    }
    DIST_DEBUG("forward send request=%llu to %s:%u route_index=%u tokens=%u pos=%u bytes=%u",
               (unsigned long long)request_id,
               next->host,
               next->port,
               forwarded.route_index,
               forwarded.n_tokens,
               forwarded.pos0,
               forwarded.input_hc_bytes);
    int rc = dist_send_work_frame(forwarder->fd, &forwarded, tokens, hidden_hc, route_blob);
    const double send_t1 = dist_now_sec();
    dist_worker_forwarder_note_send_done(forwarder,
                                         request_id,
                                         dist_usec_since(send_t0, send_t1),
                                         send_t1);
    pthread_mutex_unlock(&forwarder->send_mu);
    if (rc != 0) {
        DIST_DEBUG("forward send failed request=%llu to %s:%u",
                   (unsigned long long)request_id,
                   next->host,
                   next->port);
        dist_worker_forwarder_remove_request(forwarder, request_id);
        shutdown(forwarder->fd, SHUT_RDWR);
        int err_rc = dist_worker_upstream_send_work_error(upstream,
                                                          request_id,
                                                          "failed to forward distributed work");
        shutdown(upstream->fd, SHUT_RDWR);
        return err_rc;
    }
    DIST_DEBUG("forward send ok request=%llu", (unsigned long long)request_id);
    return 1;
}

static ds4_dist_worker_session *dist_worker_get_session_locked(
        ds4_dist_worker_state *state,
        uint64_t session_id,
        char *err,
        size_t errlen) {
    for (ds4_dist_worker_session *it = state->sessions; it; it = it->next) {
        if (it->session_id == session_id) return it;
    }

    ds4_dist_worker_session *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        if (errlen) snprintf(err, errlen, "out of memory creating distributed session");
        return NULL;
    }
    if (ds4_session_create(&entry->session, state->engine, state->ctx_size) != 0) {
        free(entry);
        if (errlen) snprintf(err, errlen, "failed to create distributed worker session");
        return NULL;
    }
    entry->session_id = session_id;
    entry->next = state->sessions;
    state->sessions = entry;
    return entry;
}

static uint32_t dist_worker_clear_sessions(ds4_dist_worker_state *state) {
    uint32_t n = 0;
    pthread_mutex_lock(&state->mu);
    ds4_dist_worker_session *it = state->sessions;
    state->sessions = NULL;
    pthread_mutex_unlock(&state->mu);

    while (it) {
        ds4_dist_worker_session *next = it->next;
        ds4_session_free(it->session);
        free(it);
        it = next;
        n++;
    }
    return n;
}

typedef struct {
    const uint8_t *p;
    uint32_t remaining;
} ds4_dist_mem_reader;

static int dist_mem_read(ds4_dist_mem_reader *r, void *dst, uint32_t len) {
    if (!r || len > r->remaining) return -1;
    if (len != 0) memcpy(dst, r->p, len);
    r->p += len;
    r->remaining -= len;
    return 1;
}

static int dist_worker_process_work_payload(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        const void *payload,
        uint32_t bytes) {
    uint64_t request_id = 0;
    char err[256];
    if (bytes < sizeof(ds4_dist_work_fixed)) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "truncated distributed WORK frame");
    }

    ds4_dist_mem_reader reader = {
        .p = payload,
        .remaining = bytes,
    };
    ds4_dist_work_fixed work;
    int rc = dist_mem_read(&reader, &work, (uint32_t)sizeof(work));
    if (rc <= 0) return -1;
    dist_work_from_wire(&work);
    const uint64_t session_id = dist_u64_from_halves(work.session_hi, work.session_lo);
    request_id = dist_u64_from_halves(work.request_hi, work.request_lo);
    const uint64_t work_prefix_hash = dist_u64_from_halves(work.prefix_hash_hi,
                                                           work.prefix_hash_lo);
    const uint64_t work_result_hash = dist_u64_from_halves(work.result_hash_hi,
                                                           work.result_hash_lo);
    DIST_DEBUG("worker work request=%llu layers=%u:%u tokens=%u pos=%u flags=0x%x token_bytes=%u input_hc=%u/%ub route_count=%u route_index=%u route_bytes=%u",
               (unsigned long long)request_id,
               work.layer_start,
               work.layer_end,
               work.n_tokens,
               work.pos0,
               work.flags,
               work.token_bytes,
               work.input_hc_bytes,
               work.input_hc_bits,
               work.route_count,
               work.route_index,
               work.route_bytes);

    const uint32_t remaining = bytes - (uint32_t)sizeof(work);
    const uint64_t token_bytes_expected = (uint64_t)work.n_tokens * sizeof(uint32_t);
    const uint64_t payload_bytes_expected =
        (uint64_t)work.token_bytes + work.input_hc_bytes + work.route_bytes;
    if ((uint64_t)work.token_bytes != token_bytes_expected ||
        payload_bytes_expected != remaining) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed WORK payload sizes");
    }
    if (work.route_count == 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK frame is missing distributed route");
    }
    if (work.route_index >= work.route_count) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed WORK route metadata");
    }
    if (work.model_id != state->model_id) {
        snprintf(err, sizeof(err), "model id mismatch: work=%u worker=%u", work.model_id, state->model_id);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if (work.layer_start != state->layer_start || work.layer_end != state->layer_end) {
        snprintf(err, sizeof(err), "worker is assigned layers %u:%u but request asked for %u:%u",
                 state->layer_start, state->layer_end, work.layer_start, work.layer_end);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if ((work.flags & ~DS4_DIST_WORK_F_VALID_MASK) != 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed WORK flags");
    }
    if (work.n_tokens == 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK frame has no tokens");
    }
    if (work.pos0 > (uint32_t)state->ctx_size ||
        work.n_tokens > (uint32_t)state->ctx_size - work.pos0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK token span exceeds worker context");
    }

    const bool output_logits = (work.flags & DS4_DIST_WORK_F_OUTPUT_LOGITS) != 0;
    const bool input_hc_present = (work.flags & DS4_DIST_WORK_F_INPUT_HC) != 0;
    const bool ack_only = (work.flags & DS4_DIST_WORK_F_ACK_ONLY) != 0;
    if (input_hc_present && work.layer_start == 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "layer 0 WORK must not provide input hidden-state");
    }
    if (!input_hc_present && work.layer_start != 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "nonzero layer WORK requires input hidden-state");
    }
    if (output_logits && !state->has_output) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "worker was not assigned the output head");
    }
    const uint32_t n_layers = (uint32_t)ds4_engine_layer_count(state->engine);
    if (output_logits && work.layer_end + 1u != n_layers) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK logits require final transformer layer");
    }

    int *tokens = malloc((size_t)work.n_tokens * sizeof(tokens[0]));
    if (!tokens) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "out of memory reading WORK tokens");
    }
    for (uint32_t i = 0; i < work.n_tokens; i++) {
        uint32_t wire_token = 0;
        rc = dist_mem_read(&reader, &wire_token, (uint32_t)sizeof(wire_token));
        if (rc <= 0) {
            free(tokens);
            return -1;
        }
        uint32_t token = ntohl(wire_token);
        if (token > (uint32_t)INT_MAX || token >= (uint32_t)ds4_engine_vocab_size(state->engine)) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "WORK token id is outside the model vocabulary");
        }
        tokens[i] = (int)token;
    }
    if (dist_token_hash_update_span(work_prefix_hash, tokens, work.n_tokens) !=
        work_result_hash) {
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK token prefix hash metadata mismatch");
    }

    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t expected_hc_values = (uint64_t)work.n_tokens * hc_values;
    const uint64_t expected_hc_bytes64 = expected_hc_values * sizeof(float);
    if (expected_hc_bytes64 > UINT32_MAX) {
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "distributed hidden-state payload is too large");
    }
    const uint32_t expected_hc_bytes = (uint32_t)expected_hc_bytes64;
    const uint32_t input_hc_bits = dist_activation_bits_or_default(work.input_hc_bits);

    float *input_hc = NULL;
    const void *input_hc_wire = NULL;
    if (input_hc_present) {
        uint32_t expected_hc_wire_bytes = 0;
        if (!dist_activation_bits_valid(input_hc_bits) ||
            !dist_activation_wire_bytes(input_hc_bits,
                                        expected_hc_values,
                                        &expected_hc_wire_bytes)) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed activation width");
        }
        if (work.input_hc_bytes != expected_hc_wire_bytes) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "input hidden-state size does not match token span");
        }
        if (work.input_hc_bytes > reader.remaining) {
            DIST_DEBUG("worker input hidden read failed request=%llu rc=%d bytes=%u",
                       (unsigned long long)request_id,
                       -1,
                       work.input_hc_bytes);
            free(tokens);
            return -1;
        }
        input_hc_wire = reader.p;
        reader.p += work.input_hc_bytes;
        reader.remaining -= work.input_hc_bytes;
        DIST_DEBUG("worker input hidden read ok request=%llu bytes=%u bits=%u",
                   (unsigned long long)request_id,
                   work.input_hc_bytes,
                   input_hc_bits);
    } else if (work.input_hc_bytes != 0) {
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK frame has hidden bytes without input flag");
    }
    void *route_blob = NULL;
    if (work.route_bytes != 0) {
        route_blob = malloc(work.route_bytes);
        if (!route_blob) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "out of memory reading distributed route");
        }
        rc = dist_mem_read(&reader, route_blob, work.route_bytes);
        if (rc <= 0) {
            DIST_DEBUG("worker route read failed request=%llu rc=%d bytes=%u",
                       (unsigned long long)request_id,
                       rc,
                       work.route_bytes);
            free(route_blob);
            free(tokens);
            return -1;
        }
        DIST_DEBUG("worker route read ok request=%llu bytes=%u",
                   (unsigned long long)request_id,
                   work.route_bytes);
    }

    ds4_dist_route_entry current_route;
    ds4_dist_route_entry next_route;
    const bool has_route = work.route_count != 0;
    const bool has_next = has_route && work.route_index + 1u < work.route_count;
    if (has_route) {
        if (!dist_route_validate_blob(route_blob, work.route_bytes, work.route_count,
                                      n_layers,
                                      err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
        if (!dist_route_get_entry(route_blob, work.route_bytes, work.route_count,
                                  work.route_index, &current_route, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
        if (current_route.layer_start != work.layer_start ||
            current_route.layer_end != work.layer_end) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "WORK layer range does not match route entry");
        }
        const bool route_output_logits = (current_route.flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0;
        if (route_output_logits != output_logits) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "WORK logits flag does not match route entry");
        }
        if (has_next &&
            !dist_route_get_entry(route_blob, work.route_bytes, work.route_count,
                                  work.route_index + 1u, &next_route, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
    }
    if (has_next && output_logits) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "non-final route entry requested logits");
    }
    if (has_route && !has_next) {
        ds4_dist_route_return ret;
        if (!dist_route_get_return_target(route_blob, work.route_bytes, work.route_count,
                                          &ret, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
        if (ret.kind != DS4_DIST_ROUTE_RETURN_UPSTREAM) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "unsupported final result destination");
        }
    }

    const bool final_ack_only = ack_only && !has_next;
    const bool local_output_logits = output_logits && !has_next && !final_ack_only;
    const bool produce_hidden = !local_output_logits && !final_ack_only;
    const uint32_t result_kind = final_ack_only
        ? DS4_DIST_RESULT_ACK
        : (local_output_logits ? DS4_DIST_RESULT_LOGITS : DS4_DIST_RESULT_HIDDEN_STATE);
    const uint32_t result_bytes = final_ack_only
        ? 0u
        : (local_output_logits
            ? (uint32_t)((uint64_t)ds4_engine_vocab_size(state->engine) * sizeof(float))
            : expected_hc_bytes);
    float *result = result_bytes ? malloc(result_bytes) : NULL;
    if (result_bytes && !result) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "out of memory allocating distributed result");
    }

    bool input_hc_uses_wire = false;
    uint32_t input_hc_decoded_bytes = 0;
    if (input_hc_present &&
        dist_decode_activation_payload(input_hc_wire,
                                       input_hc_bits,
                                       work.input_hc_bytes,
                                       &input_hc,
                                       &input_hc_decoded_bytes,
                                       &input_hc_uses_wire,
                                       err,
                                       sizeof(err)) != 0) {
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if (input_hc_present && input_hc_decoded_bytes != expected_hc_bytes) {
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "decoded input hidden-state size does not match token span");
    }

    pthread_mutex_lock(&state->mu);
    ds4_dist_worker_session *session = dist_worker_get_session_locked(state, session_id, err, sizeof(err));
    if (!session) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if ((work.flags & DS4_DIST_WORK_F_RESET_SESSION) != 0 &&
        ds4_session_layer_slice_reset(session->session, err, sizeof(err)) != 0) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if ((work.flags & DS4_DIST_WORK_F_RESET_SESSION) != 0) {
        session->token_hash = DS4_DIST_TOKEN_HASH_INIT;
        session->token_hash_valid = true;
    } else if (!session->token_hash_valid) {
        const ds4_tokens *timeline = ds4_session_tokens(session->session);
        if (!timeline || timeline->len < 0) {
            pthread_mutex_unlock(&state->mu);
            if (!input_hc_uses_wire) free(input_hc);
            free(result);
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "worker session has no token timeline");
        }
        session->token_hash = dist_token_hash_prefix(timeline->v, (uint32_t)timeline->len);
        session->token_hash_valid = true;
    }
    if (session->token_hash != work_prefix_hash) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "worker KV prefix hash mismatch");
    }
    const double eval_t0 = dist_now_sec();
    int eval_rc = ds4_session_eval_layer_slice(session->session,
                                               tokens,
                                               work.n_tokens,
                                               work.pos0,
                                               work.layer_start,
                                               work.layer_end,
                                               input_hc,
                                               produce_hidden ? result : NULL,
                                               local_output_logits,
                                               local_output_logits ? result : NULL,
                                               err,
                                               sizeof(err));
    const double eval_t1 = dist_now_sec();
    if (eval_rc == 0) {
        session->token_hash = work_result_hash;
        session->token_hash_valid = true;
    } else {
        session->token_hash_valid = false;
    }
    pthread_mutex_unlock(&state->mu);
    DIST_DEBUG("worker eval request=%llu layers=%u:%u tokens=%u pos=%u has_next=%d output=%d rc=%d",
               (unsigned long long)request_id,
               work.layer_start,
               work.layer_end,
               work.n_tokens,
               work.pos0,
               has_next ? 1 : 0,
               local_output_logits ? 1 : 0,
               eval_rc);

    if (eval_rc != 0) {
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }

    uint32_t result_wire_bytes = result_bytes;
    if (result_kind == DS4_DIST_RESULT_HIDDEN_STATE &&
        !dist_activation_wire_bytes_from_f32_bytes(input_hc_bits,
                                                   result_bytes,
                                                   &result_wire_bytes)) {
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid output hidden-state size");
    }

    ds4_dist_telemetry_fixed telemetry = {
        .layer_start = work.layer_start,
        .layer_end = work.layer_end,
        .route_index = work.route_index,
        .pos0 = work.pos0,
        .n_tokens = work.n_tokens,
        .eval_usec = dist_usec_since(eval_t0, eval_t1),
        .downstream_wait_usec = 0,
        .forward_send_usec = 0,
        .input_bytes = work.token_bytes + work.input_hc_bytes,
        .output_bytes = result_wire_bytes,
    };

    int send_rc;
    if (has_next) {
        send_rc = dist_forward_work_to_next(upstream,
                                            &next_route,
                                            &work,
                                            tokens,
                                            result,
                                            result_bytes,
                                            &telemetry,
                                            route_blob);
    } else {
        send_rc = dist_worker_upstream_send_work_result(upstream,
                                                        request_id,
                                                        work_result_hash,
                                                        0,
                                                        result_kind,
                                                        result_kind == DS4_DIST_RESULT_HIDDEN_STATE ? input_hc_bits : 32u,
                                                        &telemetry,
                                                        1,
                                                        result,
                                                        result_bytes);
    }
    DIST_DEBUG("worker send complete request=%llu has_next=%d send_rc=%d",
               (unsigned long long)request_id,
               has_next ? 1 : 0,
               send_rc);
    if (!input_hc_uses_wire) free(input_hc);
    free(result);
    free(route_blob);
    free(tokens);
    return send_rc;
}

static int dist_worker_handle_work(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes) {
    void *payload = malloc(bytes);
    if (!payload) {
        dist_discard_bytes(upstream->fd, bytes);
        return dist_worker_upstream_send_work_error(upstream, 0, "out of memory reading distributed WORK frame");
    }
    int rc = dist_read_full(upstream->fd, payload, bytes);
    if (rc <= 0) {
        free(payload);
        return rc == 0 ? 0 : -1;
    }
    rc = dist_worker_process_work_payload(state, upstream, payload, bytes);
    free(payload);
    return rc;
}

static void dist_worker_job_free(ds4_dist_worker_job *job) {
    if (!job) return;
    free(job->payload);
    free(job);
}

static void dist_worker_job_queue_init(
        ds4_dist_worker_job_queue *q,
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream) {
    memset(q, 0, sizeof(*q));
    q->state = state;
    q->upstream = upstream;
    q->depth = dist_worker_prefetch_depth();
    pthread_mutex_init(&q->mu, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static void dist_worker_job_queue_clear_locked(ds4_dist_worker_job_queue *q) {
    ds4_dist_worker_job *it = q->head;
    q->head = NULL;
    q->tail = NULL;
    q->queued = 0;
    while (it) {
        ds4_dist_worker_job *next = it->next;
        dist_worker_job_free(it);
        it = next;
    }
}

static void dist_worker_job_queue_destroy(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    dist_worker_job_queue_clear_locked(q);
    pthread_mutex_unlock(&q->mu);
    pthread_cond_destroy(&q->not_full);
    pthread_cond_destroy(&q->not_empty);
    pthread_mutex_destroy(&q->mu);
}

static void dist_worker_job_queue_finish(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    q->closed = true;
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mu);
}

static void dist_worker_job_queue_cancel(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    q->closed = true;
    q->canceled = true;
    dist_worker_job_queue_clear_locked(q);
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mu);
}

static bool dist_worker_job_queue_enqueue(
        ds4_dist_worker_job_queue *q,
        ds4_dist_worker_job *job) {
    pthread_mutex_lock(&q->mu);
    while (!q->closed && !q->canceled && q->queued >= q->depth) {
        pthread_cond_wait(&q->not_full, &q->mu);
    }
    if (q->closed || q->canceled) {
        pthread_mutex_unlock(&q->mu);
        return false;
    }
    if (q->tail) q->tail->next = job;
    else q->head = job;
    q->tail = job;
    q->queued++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mu);
    return true;
}

static ds4_dist_worker_job *dist_worker_job_queue_pop(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    while (!q->head && !q->closed && !q->canceled) {
        pthread_cond_wait(&q->not_empty, &q->mu);
    }
    if (q->canceled || !q->head) {
        pthread_mutex_unlock(&q->mu);
        return NULL;
    }
    ds4_dist_worker_job *job = q->head;
    q->head = job->next;
    if (!q->head) q->tail = NULL;
    q->queued--;
    job->next = NULL;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mu);
    return job;
}

static void *dist_worker_prefetch_eval_main(void *arg) {
    ds4_dist_worker_job_queue *q = arg;
    for (;;) {
        ds4_dist_worker_job *job = dist_worker_job_queue_pop(q);
        if (!job) break;
        int rc = dist_worker_process_work_payload(q->state,
                                                  q->upstream,
                                                  job->payload,
                                                  job->bytes);
        dist_worker_job_free(job);
        if (rc <= 0) {
            pthread_mutex_lock(&q->mu);
            q->rc = rc == 0 ? 0 : 1;
            pthread_mutex_unlock(&q->mu);
            dist_worker_job_queue_cancel(q);
            shutdown(q->upstream->fd, SHUT_RDWR);
            break;
        }
    }
    return NULL;
}

static int dist_worker_read_loop_prefetch(ds4_dist_worker_state *state, int fd) {
    ds4_dist_worker_upstream upstream;
    dist_worker_upstream_init(&upstream, state, fd);

    ds4_dist_worker_job_queue queue;
    dist_worker_job_queue_init(&queue, state, &upstream);

    pthread_t eval_tid;
    if (pthread_create(&eval_tid, NULL, dist_worker_prefetch_eval_main, &queue) != 0) {
        dist_worker_job_queue_destroy(&queue);
        dist_worker_upstream_destroy(&upstream);
        return 1;
    }

    int loop_rc = 0;
    fprintf(stderr,
            "ds4: distributed worker: receive prefetch depth %u enabled\n",
            queue.depth);

    for (;;) {
        uint32_t type = 0, bytes = 0;
        char err[256];
        int rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
        if (rc == 0) break;
        if (rc < 0) {
            fprintf(stderr, "ds4: distributed worker: protocol error: %s\n", err);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_ERROR) {
            char msg[512];
            uint32_t n = bytes < sizeof(msg) - 1u ? bytes : (uint32_t)sizeof(msg) - 1u;
            rc = dist_read_full(fd, msg, n);
            if (rc <= 0) {
                loop_rc = 1;
                break;
            }
            msg[n] = '\0';
            if (bytes > n) dist_discard_bytes(fd, bytes - n);
            fprintf(stderr, "ds4: distributed worker: coordinator error: %s\n", msg);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_WORK) {
            ds4_dist_worker_job *job = calloc(1, sizeof(*job));
            if (!job) {
                dist_discard_bytes(fd, bytes);
                dist_worker_upstream_send_work_error(&upstream, 0, "out of memory queueing distributed WORK");
                loop_rc = 1;
                break;
            }
            job->payload = malloc(bytes);
            job->bytes = bytes;
            if (!job->payload) {
                dist_worker_job_free(job);
                dist_discard_bytes(fd, bytes);
                dist_worker_upstream_send_work_error(&upstream, 0, "out of memory reading distributed WORK frame");
                loop_rc = 1;
                break;
            }
            rc = dist_read_full(fd, job->payload, bytes);
            if (rc <= 0) {
                dist_worker_job_free(job);
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            if (!dist_worker_job_queue_enqueue(&queue, job)) {
                dist_worker_job_free(job);
                loop_rc = 1;
                break;
            }
            continue;
        }
        rc = dist_discard_bytes(fd, bytes);
        if (rc <= 0) {
            loop_rc = rc == 0 ? 0 : 1;
            break;
        }
        pthread_mutex_lock(&upstream.write_mu);
        dist_send_error(fd, "unsupported distributed worker frame");
        pthread_mutex_unlock(&upstream.write_mu);
        fprintf(stderr, "ds4: distributed worker: rejected unsupported frame type %u\n", type);
        loop_rc = 1;
        break;
    }

    if (loop_rc == 0) dist_worker_job_queue_finish(&queue);
    else dist_worker_job_queue_cancel(&queue);
    pthread_join(eval_tid, NULL);
    if (loop_rc == 0 && queue.rc != 0) loop_rc = 1;
    dist_worker_job_queue_destroy(&queue);
    dist_worker_upstream_destroy(&upstream);
    return loop_rc;
}

static void *dist_worker_data_client_main(void *arg) {
    ds4_dist_data_client_ctx *ctx = arg;
    ds4_dist_worker_state *state = ctx->state;
    int fd = ctx->fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    snprintf(peer_host, sizeof(peer_host), "%s", ctx->peer_host);
    snprintf(peer_port, sizeof(peer_port), "%s", ctx->peer_port);
    free(ctx);

    int rc = getenv("DS4_DIST_DISABLE_WORKER_PREFETCH")
        ? dist_worker_read_loop(state, fd)
        : dist_worker_read_loop_prefetch(state, fd);
    if (rc != 0) {
        fprintf(stderr,
                "ds4: distributed worker: data connection %s:%s closed after error\n",
                peer_host,
                peer_port);
    }

    close(fd);
    return NULL;
}

static void *dist_worker_data_listener_main(void *arg) {
    ds4_dist_worker_state *state = arg;
    int listen_fd = state->listen_fd;
    for (;;) {
        struct sockaddr_storage ss;
        socklen_t slen = sizeof(ss);
        int fd = accept(listen_fd, (struct sockaddr *)&ss, &slen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "ds4: distributed worker: data accept failed: %s\n", strerror(errno));
            continue;
        }
        dist_set_socket_low_latency(fd);

        ds4_dist_data_client_ctx *ctx = calloc(1, sizeof(*ctx));
        if (!ctx) {
            fprintf(stderr, "ds4: distributed worker: out of memory accepting data connection\n");
            close(fd);
            continue;
        }
        ctx->state = state;
        ctx->fd = fd;
        if (getnameinfo((struct sockaddr *)&ss, slen,
                        ctx->peer_host, sizeof(ctx->peer_host),
                        ctx->peer_port, sizeof(ctx->peer_port),
                        NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
            snprintf(ctx->peer_host, sizeof(ctx->peer_host), "unknown");
            snprintf(ctx->peer_port, sizeof(ctx->peer_port), "0");
        }

        pthread_t tid;
        if (pthread_create(&tid, NULL, dist_worker_data_client_main, ctx) != 0) {
            fprintf(stderr, "ds4: distributed worker: pthread_create failed for data connection\n");
            close(fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }
}

static int dist_run_worker(ds4_engine *engine, const ds4_dist_options *opt, int ctx_size) {
    char layer_end[32];
    if (opt->layers.has_output) snprintf(layer_end, sizeof(layer_end), "output");
    else snprintf(layer_end, sizeof(layer_end), "%u", opt->layers.end);

    char err[256];
    const char *listen_host = opt->listen_host;
    int requested_port = opt->listen_port > 0 ? opt->listen_port : 0;
    int listen_fd = dist_open_listener(listen_host, requested_port, err, sizeof(err));
    if (listen_fd < 0) {
        fprintf(stderr, "ds4: distributed worker: %s\n", err);
        return 1;
    }
    int listen_port_i = dist_listener_port(listen_fd);
    if (listen_port_i <= 0) {
        fprintf(stderr, "ds4: distributed worker: could not determine data listener port\n");
        close(listen_fd);
        return 1;
    }
    const uint32_t listen_port = (uint32_t)listen_port_i;

    ds4_dist_worker_state state;
    memset(&state, 0, sizeof(state));
    state.engine = engine;
    state.model_id = (uint32_t)ds4_engine_model_id(engine);
    state.layer_start = opt->layers.start;
    state.layer_end = dist_resolved_layer_end(opt, (uint32_t)ds4_engine_layer_count(engine));
    state.has_output = opt->layers.has_output;
    state.ctx_size = ctx_size;
    state.listen_fd = listen_fd;
    pthread_mutex_init(&state.mu, NULL);

    pthread_t data_tid;
    if (pthread_create(&data_tid, NULL, dist_worker_data_listener_main, &state) != 0) {
        fprintf(stderr, "ds4: distributed worker: pthread_create failed for data listener\n");
        close(listen_fd);
        return 1;
    }
    pthread_detach(data_tid);

    fprintf(stderr,
            "ds4: distributed worker: layers %u:%s model_id=%d data_listen=%s:%u connecting to coordinator %s:%d\n",
            opt->layers.start,
            layer_end,
            ds4_engine_model_id(engine),
            listen_host ? listen_host : "*",
            listen_port,
            opt->coordinator_host,
            opt->coordinator_port);

    for (;;) {
        int fd = dist_connect_endpoint(opt->coordinator_host, opt->coordinator_port, err, sizeof(err));
        if (fd < 0) {
            fprintf(stderr, "ds4: distributed worker: %s; retrying\n", err);
            dist_sleep_reconnect();
            continue;
        }

        char peer_host[NI_MAXHOST], peer_port[NI_MAXSERV];
        dist_peer_name(fd, peer_host, sizeof(peer_host), peer_port, sizeof(peer_port));
        fprintf(stderr, "ds4: distributed worker: connected to coordinator %s:%s\n", peer_host, peer_port);

        if (dist_send_hello(engine, opt, ctx_size, listen_port, fd) != 0) {
            fprintf(stderr, "ds4: distributed worker: failed to send HELLO: %s\n", strerror(errno));
            close(fd);
            dist_sleep_reconnect();
            continue;
        }

        int rc = getenv("DS4_DIST_DISABLE_WORKER_PREFETCH")
            ? dist_worker_read_loop(&state, fd)
            : dist_worker_read_loop_prefetch(&state, fd);
        close(fd);
        uint32_t dropped_sessions = dist_worker_clear_sessions(&state);
        if (dropped_sessions) {
            fprintf(stderr,
                    "ds4: distributed worker: cleared %u sessions after coordinator disconnect\n",
                    dropped_sessions);
        }
        fprintf(stderr, "ds4: distributed worker: coordinator disconnected%s; reconnecting\n",
                rc ? " after error" : "");
        dist_sleep_reconnect();
    }
}

static bool dist_parse_role(const char *s, ds4_distributed_role *out) {
    if (!s || !out) return false;
    if (!strcmp(s, "none")) {
        *out = DS4_DISTRIBUTED_NONE;
        return true;
    }
    if (!strcmp(s, "coordinator")) {
        *out = DS4_DISTRIBUTED_COORDINATOR;
        return true;
    }
    if (!strcmp(s, "worker")) {
        *out = DS4_DISTRIBUTED_WORKER;
        return true;
    }
    return false;
}

static bool dist_parse_u32_component(const char *p, size_t len, uint32_t *out) {
    if (!p || len == 0 || !out) return false;
    char buf[32];
    if (len >= sizeof(buf)) return false;
    memcpy(buf, p, len);
    buf[len] = '\0';

    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(buf, &end, 10);
    if (errno != 0 || end == buf || *end != '\0' || v > UINT32_MAX) return false;
    *out = (uint32_t)v;
    return true;
}

static bool dist_parse_layers(const char *s, ds4_distributed_layers *out, char *err, size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "missing layer range");
        return false;
    }

    const char *colon = strchr(s, ':');
    if (!colon || colon == s || colon[1] == '\0') {
        if (errlen) snprintf(err, errlen, "expected A:B or A:output");
        return false;
    }
    if (strchr(colon + 1, ':')) {
        if (errlen) snprintf(err, errlen, "layer range has too many ':' separators");
        return false;
    }

    ds4_distributed_layers parsed = {0};
    if (!dist_parse_u32_component(s, (size_t)(colon - s), &parsed.start)) {
        if (errlen) snprintf(err, errlen, "invalid start layer in %s", s);
        return false;
    }

    const char *end = colon + 1;
    if (!strcmp(end, "output")) {
        parsed.end = UINT32_MAX;
        parsed.has_output = true;
    } else {
        if (!dist_parse_u32_component(end, strlen(end), &parsed.end)) {
            if (errlen) snprintf(err, errlen, "invalid end layer in %s", s);
            return false;
        }
        if (parsed.end < parsed.start) {
            if (errlen) snprintf(err, errlen, "layer range end precedes start in %s", s);
            return false;
        }
    }

    parsed.set = true;
    *out = parsed;
    return true;
}

static const char *dist_cli_need_arg(
        int *index,
        int argc,
        char **argv,
        const char *arg,
        char *err,
        size_t errlen) {
    if (!index || !argv || *index + 1 >= argc) {
        if (errlen) snprintf(err, errlen, "%s requires an argument", arg);
        return NULL;
    }
    return argv[++*index];
}

static bool dist_cli_parse_port(const char *s, const char *arg, int *out, char *err, size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "%s requires a TCP port", arg);
        return false;
    }
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno != 0 || s[0] == '\0' || *end != '\0' || v <= 0 || v > 65535) {
        if (errlen) snprintf(err, errlen, "invalid value for %s: %s", arg, s);
        return false;
    }
    *out = (int)v;
    return true;
}

bool ds4_dist_enabled(const ds4_dist_options *opt) {
    return opt && opt->role != DS4_DISTRIBUTED_NONE;
}

ds4_dist_options *ds4_dist_options_create(void) {
    return calloc(1, sizeof(ds4_dist_options));
}

void ds4_dist_options_free(ds4_dist_options *opt) {
    free(opt);
}

void ds4_dist_usage(FILE *fp) {
    fprintf(fp,
        "  --role ROLE\n"
        "      Distributed role: coordinator or worker.\n"
        "  --layers A:B\n"
        "      Inclusive distributed layer slice, e.g. 10:20 or 21:output.\n"
        "  --listen HOST PORT\n"
        "      Coordinator TCP listen address. Workers may later use it to force their data listener.\n"
        "  --coordinator HOST PORT\n"
        "      Coordinator TCP address for --role worker.\n"
        "  --dist-prefill-chunk N\n"
        "      Coordinator prefill pipeline chunk size. Default: session cap, normally 4096.\n"
        "      Non-default values are experimental and can change logits unless validated.\n"
        "  --dist-prefill-window N\n"
        "      Coordinator max end-to-end prefill chunks in flight. Default: workers+2, capped at 8.\n"
        "  --dist-activation-bits N\n"
        "      Coordinator hidden-state transport width: 32, 16, or 8. Default: 32.\n"
        "  --dist-replay-check\n"
        "      Coordinator diagnostic: reset and replay the prompt, then compare logits.\n"
        "  --debug\n"
        "      Print coordinator route/debug logs. Workers keep their normal logs without this.\n"
    );
}

ds4_dist_cli_parse_result ds4_dist_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_dist_options *opt,
        char *err,
        size_t errlen) {
    if (!arg) return DS4_DIST_CLI_NOT_MATCHED;
    if (!strcmp(arg, "--role")) {
        const char *role = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!role) return DS4_DIST_CLI_ERROR;
        if (!opt || !dist_parse_role(role, &opt->role)) {
            if (errlen) snprintf(err, errlen,
                                 "invalid distributed role: %s (valid roles: none, coordinator, worker)",
                                 role);
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--layers")) {
        const char *layers = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!layers) return DS4_DIST_CLI_ERROR;
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (!dist_parse_layers(layers, &opt->layers, err, errlen)) {
            char detail[160];
            if (errlen && err[0] != '\0') {
                snprintf(detail, sizeof(detail), "%s", err);
                snprintf(err, errlen, "invalid --layers %s: %s", layers, detail);
            }
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--listen")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->listen_host || opt->listen_port) {
            if (errlen) snprintf(err, errlen, "specify --listen only once");
            return DS4_DIST_CLI_ERROR;
        }
        const char *host = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!host) return DS4_DIST_CLI_ERROR;
        const char *port = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!port) return DS4_DIST_CLI_ERROR;
        if (!dist_cli_parse_port(port, arg, &opt->listen_port, err, errlen)) return DS4_DIST_CLI_ERROR;
        opt->listen_host = host;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--coordinator")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->coordinator_host || opt->coordinator_port) {
            if (errlen) snprintf(err, errlen, "specify --coordinator only once");
            return DS4_DIST_CLI_ERROR;
        }
        const char *host = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!host) return DS4_DIST_CLI_ERROR;
        const char *port = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!port) return DS4_DIST_CLI_ERROR;
        if (!dist_cli_parse_port(port, arg, &opt->coordinator_port, err, errlen)) return DS4_DIST_CLI_ERROR;
        opt->coordinator_host = host;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-prefill-chunk")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!dist_parse_positive_u32(value, arg, &opt->prefill_chunk, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-prefill-window")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!dist_parse_positive_u32(value, arg, &opt->prefill_window, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->prefill_window > 64u) {
            if (errlen) snprintf(err, errlen, "%s must be <= 64", arg);
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-activation-bits")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        uint32_t bits = 0;
        if (!dist_parse_positive_u32(value, arg, &bits, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        if (!dist_activation_bits_valid(bits)) {
            if (errlen) snprintf(err, errlen, "%s must be 32, 16, or 8", arg);
            return DS4_DIST_CLI_ERROR;
        }
        opt->activation_bits = bits;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-replay-check")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        opt->replay_check = true;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--debug")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        opt->debug = true;
        return DS4_DIST_CLI_MATCHED;
    }
    return DS4_DIST_CLI_NOT_MATCHED;
}

static int dist_validate_options(const ds4_dist_options *opt, char *err, size_t errlen) {
    if (!opt) {
        if (errlen) snprintf(err, errlen, "missing distributed options");
        return 1;
    }

    if (opt->role == DS4_DISTRIBUTED_NONE) {
        if (opt->layers.set || opt->listen_host || opt->listen_port ||
            opt->coordinator_host || opt->coordinator_port ||
            opt->prefill_chunk != 0 || opt->prefill_window != 0 ||
            opt->activation_bits != 0) {
            if (errlen) snprintf(err, errlen, "distributed options require --role coordinator or --role worker");
            return 1;
        }
        return 0;
    }

    if (!opt->layers.set) {
        if (errlen) snprintf(err, errlen, "--role %s requires --layers", dist_role_name(opt->role));
        return 1;
    }
    if (opt->prefill_window > 64u) {
        if (errlen) snprintf(err, errlen, "--dist-prefill-window must be <= 64");
        return 1;
    }
    if (opt->activation_bits != 0 && !dist_activation_bits_valid(opt->activation_bits)) {
        if (errlen) snprintf(err, errlen, "--dist-activation-bits must be 32, 16, or 8");
        return 1;
    }

    if (opt->role == DS4_DISTRIBUTED_COORDINATOR) {
        if (!opt->listen_host || opt->listen_port <= 0) {
            if (errlen) snprintf(err, errlen, "--role coordinator requires --listen HOST PORT");
            return 1;
        }
        if (opt->coordinator_host || opt->coordinator_port) {
            if (errlen) snprintf(err, errlen, "--role coordinator must not use --coordinator");
            return 1;
        }
        return 0;
    }

    if (opt->role == DS4_DISTRIBUTED_WORKER) {
        if (!opt->coordinator_host || opt->coordinator_port <= 0) {
            if (errlen) snprintf(err, errlen, "--role worker requires --coordinator HOST PORT");
            return 1;
        }
        if (opt->prefill_chunk != 0) {
            if (errlen) snprintf(err, errlen, "--dist-prefill-chunk requires --role coordinator");
            return 1;
        }
        if (opt->prefill_window != 0) {
            if (errlen) snprintf(err, errlen, "--dist-prefill-window requires --role coordinator");
            return 1;
        }
        if (opt->activation_bits != 0) {
            if (errlen) snprintf(err, errlen, "--dist-activation-bits requires --role coordinator");
            return 1;
        }
        return 0;
    }

    if (errlen) snprintf(err, errlen, "invalid distributed role");
    return 1;
}

int ds4_dist_prepare_engine_options(
        const ds4_dist_options *opt,
        ds4_engine_options *engine,
        char *err,
        size_t errlen) {
    if (dist_validate_options(opt, err, errlen) != 0) return 1;
    if (opt && opt->replay_check && opt->role != DS4_DISTRIBUTED_COORDINATOR) {
        if (errlen) snprintf(err, errlen, "--dist-replay-check requires --role coordinator");
        return 1;
    }
    if (engine && opt) {
        engine->distributed = *opt;
        if (ds4_dist_enabled(opt)) {
            engine->load_slice = true;
            engine->load_layer_start = opt->layers.start;
            engine->load_layer_end = opt->layers.has_output ? UINT32_MAX : opt->layers.end;
            engine->load_output = opt->layers.has_output || opt->role == DS4_DISTRIBUTED_COORDINATOR;
        }
    }
    return 0;
}

static int dist_validate_layers_for_model(const ds4_dist_options *opt, uint32_t n_layers, char *err, size_t errlen) {
    if (!opt || opt->role == DS4_DISTRIBUTED_NONE || !opt->layers.set) return 0;
    if (n_layers == 0) {
        if (errlen) snprintf(err, errlen, "model reports no layers");
        return 1;
    }

    const uint32_t last = n_layers - 1u;
    if (opt->layers.start > last) {
        if (errlen) snprintf(err, errlen, "layer range starts past final model layer %u", last);
        return 1;
    }
    if (!opt->layers.has_output && opt->layers.end > last) {
        if (errlen) snprintf(err, errlen, "layer range ends past final model layer %u", last);
        return 1;
    }
    if (opt->role == DS4_DISTRIBUTED_COORDINATOR && opt->layers.start != 0) {
        if (errlen) snprintf(err, errlen, "coordinator layer range must start at layer 0");
        return 1;
    }
    return 0;
}

int ds4_dist_run(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen) {
    if (!engine || !opt) {
        fprintf(stderr, "ds4: distributed runtime requires an open engine and options\n");
        return 1;
    }
    char err[256];
    if (dist_validate_options(opt, err, sizeof(err)) != 0 ||
        dist_validate_layers_for_model(opt, (uint32_t)ds4_engine_layer_count(engine), err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: %s\n", err);
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);

    if (opt->role == DS4_DISTRIBUTED_COORDINATOR) {
        return dist_run_coordinator(engine, opt, gen);
    }
    if (opt->role == DS4_DISTRIBUTED_WORKER) {
        return dist_run_worker(engine, opt, gen ? gen->ctx_size : 0);
    }

    fprintf(stderr, "ds4: distributed runtime requested without a distributed role\n");
    return 1;
}
