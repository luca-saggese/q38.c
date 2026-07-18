/* Tensor-parallel transport and lockstep protocol.  See ds4_tp.h and
 * misc/METAL_TENSOR_PARALLELISM.md for the design.
 *
 * Wire notes: both ranks are identical Apple Silicon machines by
 * definition, so the wire format is host little-endian; the hello magic
 * doubles as a byte-order check.  The control socket is a plain blocking
 * TCP stream carrying framed commands.  Gate traffic goes over RDMA
 * (Thunderbolt UC queue pair, two-sided send/recv — see the driver quirks
 * note at ds4_tp_rdma) or over a dedicated full-duplex TCP socket at 16KB
 * per direction as the fallback. */

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <stdarg.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ds4_tp.h"

#if defined(__APPLE__) && defined(__has_include)
#if __has_include(<infiniband/verbs.h>)
#include <infiniband/verbs.h>
#include <dlfcn.h>
#define DS4_TP_HAVE_VERBS 1
#endif
#endif

#define DS4_TP_MAGIC UINT32_C(0x44533454) /* "DS4T" */
#define DS4_TP_BATCH_MAGIC UINT32_C(0x44533442) /* "DS4B" */
#define DS4_TP_PROTOCOL_VERSION 1u

/* Default gate timeout is generous: the first gate after a sync waits for
 * the peer's whole (possibly cold page cache) prefill. */
#define DS4_TP_DEFAULT_TIMEOUT_SEC 300

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;
} ds4_tp_frame_header;

typedef struct {
    uint32_t magic;      /* also detects byte-order mismatch */
    uint32_t version;
    uint32_t role;
    uint32_t rdma_ok;    /* this side has a usable verbs device */
    uint32_t null_split; /* DS4_TP_NULL_SPLIT bring-up mode, must match */
    uint32_t pad;
    uint64_t gguf_bytes;
    uint32_t model_id;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t quant_bits;
    uint32_t ctx_size;
} ds4_tp_hello_fixed;

typedef struct {
    uint64_t slab_base;
    uint32_t rkey;
    uint32_t qpn;
    uint32_t psn;
    uint32_t mtu;
    uint16_t lid;
    uint8_t gid[16];
    uint8_t link_layer;
} ds4_tp_rdma_info;

/* TCP gate frames carry a small header so a desynchronized pair fails loudly
 * instead of silently mixing partials. */
typedef struct {
    uint32_t magic;
    uint16_t layer;
    uint16_t gate;
    uint64_t seq;
} ds4_tp_gate_header;

#ifdef DS4_TP_HAVE_VERBS
/* librdma is loaded at runtime so builds and machines without the RDMA
 * stack (or with it disabled) fall back to TCP with no link-time cost.
 * ibv_post_send()/ibv_poll_cq() are header inlines over context->ops, so
 * only the setup entry points need dlsym. */
typedef struct {
    void *handle;
    struct ibv_device **(*get_device_list)(int *);
    void (*free_device_list)(struct ibv_device **);
    const char *(*get_device_name)(struct ibv_device *);
    struct ibv_context *(*open_device)(struct ibv_device *);
    int (*close_device)(struct ibv_context *);
    int (*query_device)(struct ibv_context *, struct ibv_device_attr *);
    int (*query_port)(struct ibv_context *, uint8_t, struct ibv_port_attr *);
    int (*query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *);
    int (*dealloc_pd)(struct ibv_pd *);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int);
    int (*dereg_mr)(struct ibv_mr *);
    struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *, struct ibv_comp_channel *, int);
    int (*destroy_cq)(struct ibv_cq *);
    struct ibv_qp *(*create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
    int (*destroy_qp)(struct ibv_qp *);
    int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
} ds4_tp_verbs_api;

/* AppleThunderboltRDMA quirks (validated with scratchpad probes,
 * 2026-07-06): only UC queue pairs exist (RC/UD: ENOTSUP); RDMA WRITE work
 * requests are accepted but never execute, so the data plane is two-sided
 * SEND/RECV like Apple's own JACCL; messages above 16KB are not delivered;
 * RTR requires GRH addressing with the IPv4-mapped GID that appears only
 * once the Thunderbolt member interface has an IPv4 address of its own.
 * UC delivery is in-order and the gate sequence is globally deterministic
 * (86 gates per token, fixed order), so receives are pre-posted purely by
 * sequence number: recv for seq s lands in the slab in-slot (s-1) % slots
 * and its completion IS the arrival signal. */
#define DS4_TP_RDMA_MAX_MSG 16384
#define DS4_TP_RDMA_RECV_WINDOW 16

typedef struct {
    ds4_tp_verbs_api api;
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    struct ibv_port_attr port;
    union ibv_gid gid;
    int gid_index;
    uint32_t max_inline;
    ds4_tp_rdma_info peer;
    uint32_t send_outstanding;  /* signaled sends not yet reaped */
    uint64_t recv_done;         /* highest gate seq whose recv completed */
    uint64_t recv_posted;       /* highest gate seq with a posted recv */
    pthread_mutex_t post_lock;
} ds4_tp_rdma;
#endif

struct ds4_tp {
    ds4_tp_options opt;
    int rank;                   /* 0 leader, 1 worker */
    int control_fd;
    int data_fd;                /* TCP gate socket (batch gates under RDMA too) */
    bool rdma_active;
    uint32_t peer_ctx;
    uint32_t n_layer;
    uint32_t n_embd;
    uint64_t vec_bytes;
    uint32_t n_slots;
    uint8_t *slab;
    uint64_t slab_bytes;
    /* Slab regions, see ds4_tp.h layout comment. */
    uint64_t out_off;
    uint64_t in_off;
    uint64_t in_flags_off;
    uint64_t token_off;
    uint64_t out_flags_off;     /* local staging for RDMA flag writes */
    uint64_t gpu_flags_off;     /* GPU-written gate-ready flags (u32/slot) */
    uint64_t batch_out_off;     /* [layer][row] verify-block local partials */
    uint64_t batch_in_off;      /* [layer][row] verify-block peer partials */
    uint64_t timeout_sec;
#ifdef DS4_TP_HAVE_VERBS
    ds4_tp_rdma rdma;
#endif
};

/* ------------------------------------------------------------------------
 * Small socket helpers (same conventions as ds4_distributed.c).
 * --------------------------------------------------------------------- */

static double tp_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void tp_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || !errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static int tp_write_full(int fd, const void *buf, size_t len) {
    const char *p = buf;
    while (len) {
        ssize_t w = write(fd, p, len);
        if (w < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (w == 0) return 0;
        p += w;
        len -= (size_t)w;
    }
    return 1;
}

static int tp_read_full(int fd, void *buf, size_t len) {
    char *p = buf;
    while (len) {
        ssize_t r = read(fd, p, len);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (r == 0) return 0;
        p += r;
        len -= (size_t)r;
    }
    return 1;
}

static void tp_socket_tune(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    /* Gate exchanges are latency-critical 16KB messages; large socket
     * buffers only matter for the TCP fallback's pipelining. */
    int sz = 4 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
}

static int tp_listen(const char *host, int port, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    int rc = getaddrinfo(host && host[0] ? host : NULL, portbuf, &hints, &res);
    if (rc != 0) {
        tp_set_err(err, errlen, "tp listen resolve %s:%d: %s", host, port, gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(fd, 2) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) tp_set_err(err, errlen, "tp listen %s:%d: %s", host, port, strerror(errno));
    return fd;
}

static int tp_dial(const char *host, int port, double timeout_sec, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    double deadline = tp_now_sec() + timeout_sec;
    int last_errno = 0;
    uint32_t attempts = 0;
    do {
        struct addrinfo hints = {0}, *res = NULL;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        int gai = getaddrinfo(host, portbuf, &hints, &res);
        if (gai == 0) {
            for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
                int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
                if (fd < 0) continue;
                if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
                    freeaddrinfo(res);
                    return fd;
                }
                last_errno = errno;
                close(fd);
            }
            freeaddrinfo(res);
        }
        /* Retrying is normal while the peer loads its model; still say why
         * every ~10s so a wrong address or a policy block is visible. */
        if (attempts++ % 50 == 0) {
            fprintf(stderr, "ds4-tp: connecting to %s:%d ... (%s)\n", host, port,
                    gai != 0 ? gai_strerror(gai) :
                    last_errno ? strerror(last_errno) : "no address worked");
        }
        usleep(200 * 1000);
    } while (tp_now_sec() < deadline);
    tp_set_err(err, errlen, "tp connect %s:%d: %s", host, port,
               last_errno ? strerror(last_errno) : "unreachable");
    return -1;
}

static int tp_send_frame(int fd, uint32_t type, const void *payload, uint32_t bytes) {
    ds4_tp_frame_header h = { DS4_TP_MAGIC, type, bytes };
    if (!tp_write_full(fd, &h, sizeof(h))) return 0;
    if (bytes && !tp_write_full(fd, payload, bytes)) return 0;
    return 1;
}

static int tp_read_frame_header(int fd, uint32_t *type, uint32_t *bytes) {
    ds4_tp_frame_header h;
    if (!tp_read_full(fd, &h, sizeof(h))) return 0;
    if (h.magic != DS4_TP_MAGIC) return 0;
    *type = h.type;
    *bytes = h.bytes;
    return 1;
}

/* ------------------------------------------------------------------------
 * Options and CLI.
 * --------------------------------------------------------------------- */

bool ds4_tp_enabled(const ds4_tp_options *opt) {
    return opt && opt->role != DS4_TP_NONE;
}

void ds4_tp_usage(FILE *fp) {
    fprintf(fp,
        "Tensor parallelism (two identical machines, full model on both):\n"
        "  --tp-lead <port>            Lead a TP pair: listen for the worker,\n"
        "                              then run the normal CLI flow.\n"
        "  --tp-lead-host <host>       Leader listen address (default 0.0.0.0).\n"
        "  --tp-worker <host> <port>   Dial the leader and mirror its session.\n"
        "  --tp-transport <auto|rdma|tcp>  Gate transport (default auto).\n"
        "  --tp-debug-hash <n>         Cross-check hidden state every n tokens.\n");
}

int ds4_tp_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_tp_options *opt,
        char *err,
        size_t errlen)
{
    int i = *index;
    if (!strcmp(arg, "--tp-lead")) {
        if (i + 1 >= argc) goto missing;
        opt->role = DS4_TP_LEADER;
        opt->listen_port = atoi(argv[++i]);
        if (opt->listen_port <= 0 || opt->listen_port > 65535) {
            tp_set_err(err, errlen, "invalid --tp-lead port");
            return DS4_TP_CLI_ERROR;
        }
        if (!opt->listen_host) opt->listen_host = "0.0.0.0";
    } else if (!strcmp(arg, "--tp-lead-host")) {
        if (i + 1 >= argc) goto missing;
        opt->listen_host = argv[++i];
    } else if (!strcmp(arg, "--tp-worker")) {
        if (i + 2 >= argc) goto missing;
        opt->role = DS4_TP_WORKER;
        opt->leader_host = argv[++i];
        opt->leader_port = atoi(argv[++i]);
        if (opt->leader_port <= 0 || opt->leader_port > 65535) {
            tp_set_err(err, errlen, "invalid --tp-worker port");
            return DS4_TP_CLI_ERROR;
        }
    } else if (!strcmp(arg, "--tp-transport")) {
        if (i + 1 >= argc) goto missing;
        const char *v = argv[++i];
        if (!strcmp(v, "auto")) opt->transport = DS4_TP_TRANSPORT_AUTO;
        else if (!strcmp(v, "rdma")) opt->transport = DS4_TP_TRANSPORT_RDMA;
        else if (!strcmp(v, "tcp")) opt->transport = DS4_TP_TRANSPORT_TCP;
        else {
            tp_set_err(err, errlen, "invalid --tp-transport %s", v);
            return DS4_TP_CLI_ERROR;
        }
    } else if (!strcmp(arg, "--tp-debug-hash")) {
        if (i + 1 >= argc) goto missing;
        opt->debug_hash = atoi(argv[++i]);
    } else {
        return DS4_TP_CLI_NOT_MATCHED;
    }
    *index = i;
    return DS4_TP_CLI_MATCHED;
missing:
    tp_set_err(err, errlen, "%s requires an argument", arg);
    return DS4_TP_CLI_ERROR;
}

int ds4_tp_validate_engine_options(
        const ds4_engine_options *opt,
        char *err,
        size_t errlen)
{
    if (!ds4_tp_enabled(&opt->tp)) return 1;
    if (opt->backend != DS4_BACKEND_METAL) {
        tp_set_err(err, errlen, "tensor parallelism requires the Metal backend");
        return 0;
    }
    if (opt->ssd_streaming) {
        tp_set_err(err, errlen, "tensor parallelism requires resident weights (no --ssd-streaming)");
        return 0;
    }
    if (opt->distributed.role != DS4_DISTRIBUTED_NONE) {
        tp_set_err(err, errlen, "tensor parallelism and --role distributed modes are exclusive");
        return 0;
    }
    /* Speculative drafting (DSpark/MTP) is allowed on the leader: the
     * verify block is mirrored to the worker via DS4_TP_FRAME_VERIFY and
     * the legacy MTP path falls back to per-token decode under TP. */
    if (opt->load_slice) {
        tp_set_err(err, errlen, "tensor parallelism loads the full model on both ranks");
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Slab layout.
 * --------------------------------------------------------------------- */

uint64_t ds4_tp_slab_bytes(uint32_t n_layer, uint32_t n_embd) {
    uint64_t vec = (uint64_t)n_embd * sizeof(float);
    uint64_t slots = (uint64_t)n_layer * DS4_TP_GATES_PER_LAYER;
    return slots * vec * 2 +    /* out + in vectors */
           slots * 8 * 2 +      /* in flags + out flag staging */
           16 +                 /* token slot */
           slots * 4 +          /* GPU-written gate-ready flags */
           (uint64_t)n_layer * DS4_TP_BATCH_MAX_ROWS * vec * 2; /* batch out+in */
}

static void tp_slab_layout(ds4_tp *tp) {
    uint64_t vec = tp->vec_bytes;
    uint64_t slots = tp->n_slots;
    tp->out_off = 0;
    tp->in_off = slots * vec;
    tp->in_flags_off = tp->in_off + slots * vec;
    tp->token_off = tp->in_flags_off + slots * 8;
    tp->out_flags_off = tp->token_off + 16;
    tp->gpu_flags_off = tp->out_flags_off + slots * 8;
    tp->batch_out_off = tp->gpu_flags_off + slots * 4;
    tp->batch_in_off = tp->batch_out_off +
                       (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * vec;
    tp->slab_bytes = tp->batch_in_off +
                     (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * vec;
}

uint64_t ds4_tp_slab_gpu_flags_offset(const ds4_tp *tp) {
    return tp->gpu_flags_off;
}

static uint32_t tp_slot(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    (void)tp;
    return layer * DS4_TP_GATES_PER_LAYER + gate;
}

uint64_t ds4_tp_slab_out_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    return tp->out_off + (uint64_t)tp_slot(tp, layer, gate) * tp->vec_bytes;
}

uint64_t ds4_tp_slab_in_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    return tp->in_off + (uint64_t)tp_slot(tp, layer, gate) * tp->vec_bytes;
}

uint64_t ds4_tp_slab_batch_out_offset(const ds4_tp *tp, uint32_t layer) {
    return tp->batch_out_off +
           (uint64_t)layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
}

uint64_t ds4_tp_slab_batch_in_offset(const ds4_tp *tp, uint32_t layer) {
    return tp->batch_in_off +
           (uint64_t)layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
}

/* ------------------------------------------------------------------------
 * RDMA path.
 * --------------------------------------------------------------------- */

#ifdef DS4_TP_HAVE_VERBS

static int tp_rdma_load_api(ds4_tp_verbs_api *api) {
    if (api->handle) return 1;
    void *h = dlopen("/usr/lib/librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) return 0;
#define TP_SYM(field, name) \
    do { \
        api->field = (__typeof__(api->field))dlsym(h, name); \
        if (!api->field) { dlclose(h); return 0; } \
    } while (0)
    TP_SYM(get_device_list, "ibv_get_device_list");
    TP_SYM(free_device_list, "ibv_free_device_list");
    TP_SYM(get_device_name, "ibv_get_device_name");
    TP_SYM(open_device, "ibv_open_device");
    TP_SYM(close_device, "ibv_close_device");
    TP_SYM(query_device, "ibv_query_device");
    TP_SYM(query_port, "ibv_query_port");
    TP_SYM(query_gid, "ibv_query_gid");
    TP_SYM(alloc_pd, "ibv_alloc_pd");
    TP_SYM(dealloc_pd, "ibv_dealloc_pd");
    TP_SYM(reg_mr, "ibv_reg_mr");
    TP_SYM(dereg_mr, "ibv_dereg_mr");
    TP_SYM(create_cq, "ibv_create_cq");
    TP_SYM(destroy_cq, "ibv_destroy_cq");
    TP_SYM(create_qp, "ibv_create_qp");
    TP_SYM(destroy_qp, "ibv_destroy_qp");
    TP_SYM(modify_qp, "ibv_modify_qp");
#undef TP_SYM
    api->handle = h;
    return 1;
}

/* Probe only: does this machine expose a verbs device right now? */
static int tp_rdma_probe(ds4_tp_verbs_api *api) {
    if (!tp_rdma_load_api(api)) return 0;
    int num = 0;
    struct ibv_device **devs = api->get_device_list(&num);
    if (!devs) return 0;
    api->free_device_list(devs);
    return num > 0;
}

static int tp_rdma_open(ds4_tp *tp, char *err, size_t errlen) {
    ds4_tp_rdma *r = &tp->rdma;
    int num = 0;
    struct ibv_device **devs = r->api.get_device_list(&num);
    if (!devs || num == 0) {
        tp_set_err(err, errlen, "tp rdma: no verbs devices");
        if (devs) r->api.free_device_list(devs);
        return 0;
    }
    /* One verbs device per Thunderbolt port (rdma_enN); pick the one whose
     * port is up — that is the cabled link.  DS4_TP_RDMA_DEV overrides. */
    const char *want_name = getenv("DS4_TP_RDMA_DEV");
    char states[256] = "";
    for (int i = 0; i < num && !r->ctx; i++) {
        const char *name = r->api.get_device_name(devs[i]);
        if (want_name && strcmp(want_name, name) != 0) continue;
        struct ibv_context *ctx = r->api.open_device(devs[i]);
        if (!ctx) continue;
        struct ibv_port_attr pa;
        if (r->api.query_port(ctx, 1, &pa) == 0 &&
            (pa.state == IBV_PORT_ACTIVE || want_name)) {
            r->ctx = ctx;
            r->port = pa;
            fprintf(stderr, "ds4-tp: rdma device %s (port state %d)\n", name, (int)pa.state);
            break;
        }
        size_t off = strlen(states);
        snprintf(states + off, sizeof(states) - off, "%s%s=%d",
                 off ? ", " : "", name, (int)pa.state);
        r->api.close_device(ctx);
    }
    r->api.free_device_list(devs);
    if (!r->ctx) {
        tp_set_err(err, errlen,
                   "tp rdma: no device with an active port (%s); is the peer up "
                   "and rdma_ctl enabled on both machines?", states);
        return 0;
    }
    /* The driver only connects through the IPv4-mapped GID
     * (::ffff:a.b.c.d), which exists only when the Thunderbolt member
     * interface carries an IPv4 address (the bridge's address does not
     * count).  DS4_TP_GID_INDEX overrides the search. */
    r->gid_index = -1;
    const char *gid_env = getenv("DS4_TP_GID_INDEX");
    if (gid_env) {
        r->gid_index = atoi(gid_env);
        if (r->api.query_gid(r->ctx, 1, r->gid_index, &r->gid) != 0) {
            tp_set_err(err, errlen, "tp rdma: query_gid(%d): %s",
                       r->gid_index, strerror(errno));
            return 0;
        }
    } else {
        for (int i = 0; i < r->port.gid_tbl_len; i++) {
            union ibv_gid tmp;
            if (r->api.query_gid(r->ctx, 1, i, &tmp) != 0) continue;
            uint64_t hi;
            uint16_t mid, v4tag;
            memcpy(&hi, &tmp.raw[0], 8);
            memcpy(&mid, &tmp.raw[8], 2);
            memcpy(&v4tag, &tmp.raw[10], 2);
            if (hi == 0 && mid == 0 && v4tag == 0xffff) {
                r->gid = tmp;
                r->gid_index = i;
                break;
            }
        }
        if (r->gid_index < 0) {
            tp_set_err(err, errlen,
                       "tp rdma: no IPv4-mapped GID on the active port; give the "
                       "Thunderbolt interface its own IPv4 (e.g. sudo ifconfig en1 "
                       "inet 10.99.0.2/30 alias) on both machines");
            return 0;
        }
    }
    r->pd = r->api.alloc_pd(r->ctx);
    if (!r->pd) {
        tp_set_err(err, errlen, "tp rdma: alloc_pd failed");
        return 0;
    }
    r->cq = r->api.create_cq(r->ctx, 512, NULL, NULL, 0);
    if (!r->cq) {
        tp_set_err(err, errlen, "tp rdma: create_cq failed");
        return 0;
    }
    struct ibv_qp_init_attr qia = {0};
    qia.send_cq = r->cq;
    qia.recv_cq = r->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr = 256;
    qia.cap.max_recv_wr = 64;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 0;
    r->qp = r->api.create_qp(r->pd, &qia);
    if (!r->qp) {
        tp_set_err(err, errlen, "tp rdma: create_qp(UC): %s", strerror(errno));
        return 0;
    }
    r->max_inline = qia.cap.max_inline_data;
    pthread_mutex_init(&r->post_lock, NULL);
    return 1;
}

static int tp_rdma_post_gate_recv(ds4_tp *tp, uint64_t seq);

static int tp_rdma_register_and_exchange(ds4_tp *tp, char *err, size_t errlen) {
    ds4_tp_rdma *r = &tp->rdma;
    r->mr = r->api.reg_mr(r->pd, tp->slab, tp->slab_bytes,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_WRITE);
    if (!r->mr) {
        tp_set_err(err, errlen, "tp rdma: reg_mr(%llu bytes): %s",
                   (unsigned long long)tp->slab_bytes, strerror(errno));
        return 0;
    }
    ds4_tp_rdma_info mine = {0};
    mine.slab_base = (uint64_t)(uintptr_t)tp->slab;
    mine.rkey = r->mr->rkey;
    mine.qpn = r->qp->qp_num;
    mine.psn = (uint32_t)(getpid() ^ (uintptr_t)tp) & 0xffffff;
    mine.mtu = (uint32_t)r->port.active_mtu;
    mine.lid = r->port.lid;
    memcpy(mine.gid, r->gid.raw, 16);
    mine.link_layer = r->port.link_layer;
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_RDMA_INFO, &mine, sizeof(mine))) {
        tp_set_err(err, errlen, "tp rdma: info send failed");
        return 0;
    }
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_RDMA_INFO || bytes != sizeof(r->peer) ||
        !tp_read_full(tp->control_fd, &r->peer, sizeof(r->peer))) {
        tp_set_err(err, errlen, "tp rdma: info recv failed");
        return 0;
    }

    /* INIT -> RTR -> RTS with the exact recipe the driver accepts (same as
     * JACCL): MTU 1024 and GRH via the IPv4-mapped GID. */
    struct ibv_qp_attr a = {0};
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                        IBV_ACCESS_REMOTE_WRITE;
    if (r->api.modify_qp(r->qp, &a,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify INIT: %s", strerror(errno));
        return 0;
    }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = IBV_MTU_1024;
    a.dest_qp_num = r->peer.qpn;
    a.rq_psn = r->peer.psn;
    a.ah_attr.dlid = (uint16_t)r->peer.lid;
    a.ah_attr.port_num = 1;
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, r->peer.gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t)r->gid_index;
    a.ah_attr.grh.hop_limit = 1;
    if (r->api.modify_qp(r->qp, &a,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTR: %s", strerror(errno));
        return 0;
    }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = mine.psn;
    if (r->api.modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTS: %s", strerror(errno));
        return 0;
    }
    if (tp->vec_bytes > DS4_TP_RDMA_MAX_MSG) {
        tp_set_err(err, errlen,
                   "tp rdma: gate vector %llu bytes exceeds the driver's %u "
                   "message limit (needs chunking)",
                   (unsigned long long)tp->vec_bytes, DS4_TP_RDMA_MAX_MSG);
        return 0;
    }
    /* Arm the initial recv window, then barrier on the control socket so
     * neither side can send a gate before the peer's recvs exist. */
    for (uint64_t s = 1; s <= DS4_TP_RDMA_RECV_WINDOW; s++) {
        if (!tp_rdma_post_gate_recv(tp, s)) {
            tp_set_err(err, errlen, "tp rdma: initial recv post failed");
            return 0;
        }
    }
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_RDMA_READY, NULL, 0)) {
        tp_set_err(err, errlen, "tp rdma: ready send failed");
        return 0;
    }
    uint32_t rtype = 0, rbytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &rtype, &rbytes) ||
        rtype != DS4_TP_FRAME_RDMA_READY || rbytes != 0) {
        tp_set_err(err, errlen, "tp rdma: ready barrier failed");
        return 0;
    }
    return 1;
}

/* ibv_wc_status_str lives in librdma; resolve lazily to keep the dlopen-only
 * linkage discipline. */
static const char *tp_wc_status_str(int status) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "wc status %d", status);
    return buf;
}

/* Reap completions: send CQEs free send-queue slots, recv CQEs advance the
 * arrival watermark (UC is in-order, so gate seq recv completions arrive
 * monotonically).  Returns 0 on any completion error. */
static int tp_rdma_drain_cq(ds4_tp *tp) {
    ds4_tp_rdma *r = &tp->rdma;
    struct ibv_wc wc[16];
    int n = ibv_poll_cq(r->cq, 16, wc);
    if (n < 0) return 0;
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            fprintf(stderr, "ds4-tp: rdma completion error: %s (wr_id %llu)\n",
                    tp_wc_status_str(wc[i].status),
                    (unsigned long long)wc[i].wr_id);
            return 0;
        }
        if (wc[i].opcode & IBV_WC_RECV) {
            if (wc[i].wr_id > r->recv_done) r->recv_done = wc[i].wr_id;
        } else if (r->send_outstanding > 0) {
            r->send_outstanding--;
        }
    }
    return 1;
}

/* Arm the receive for gate seq: UC delivery order pairs the peer's seq'th
 * send with our seq'th posted recv, landing it in the in-slot the combine
 * kernel reads. */
static int tp_rdma_post_gate_recv(ds4_tp *tp, uint64_t seq) {
    ds4_tp_rdma *r = &tp->rdma;
    const uint32_t slot = (uint32_t)((seq - 1) % tp->n_slots);
    struct ibv_sge sge;
    struct ibv_recv_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    sge.addr = (uintptr_t)(tp->slab + tp->in_off + (uint64_t)slot * tp->vec_bytes);
    sge.length = (uint32_t)tp->vec_bytes;
    sge.lkey = r->mr->lkey;
    wr.wr_id = seq;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    if (ibv_post_recv(r->qp, &wr, &bad) != 0) {
        fprintf(stderr, "ds4-tp: rdma post_recv(seq %llu): %s\n",
                (unsigned long long)seq, strerror(errno));
        return 0;
    }
    if (seq > r->recv_posted) r->recv_posted = seq;
    return 1;
}

/* One gate: send our partial, wait for the peer's recv completion, keep the
 * recv window armed. */
static int tp_rdma_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
    ds4_tp_rdma *r = &tp->rdma;
    const uint32_t slot = layer * DS4_TP_GATES_PER_LAYER + gate;
    if (slot != (uint32_t)((seq - 1) % tp->n_slots)) {
        fprintf(stderr, "ds4-tp: gate order broke: layer %u gate %u vs seq %llu\n",
                layer, gate, (unsigned long long)seq);
        return 0;
    }
    struct ibv_sge sge;
    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    sge.addr = (uintptr_t)(tp->slab + tp->out_off + (uint64_t)slot * tp->vec_bytes);
    sge.length = (uint32_t)tp->vec_bytes;
    sge.lkey = r->mr->lkey;
    wr.wr_id = seq;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    pthread_mutex_lock(&r->post_lock);
    int ok = ibv_post_send(r->qp, &wr, &bad) == 0;
    if (ok) r->send_outstanding++;
    else fprintf(stderr, "ds4-tp: rdma post_send: %s\n", strerror(errno));

    double deadline = 0.0;
    while (ok && r->recv_done < seq) {
        ok = tp_rdma_drain_cq(tp);
        if (deadline == 0.0) deadline = tp_now_sec() + (double)tp->timeout_sec;
        else if (tp_now_sec() > deadline) {
            fprintf(stderr, "ds4-tp: timeout waiting gate seq %llu (recv_done %llu)\n",
                    (unsigned long long)seq, (unsigned long long)r->recv_done);
            ok = 0;
        }
    }
    if (ok) ok = tp_rdma_post_gate_recv(tp, seq + DS4_TP_RDMA_RECV_WINDOW);
    pthread_mutex_unlock(&r->post_lock);
    return ok;
}

static void tp_rdma_close(ds4_tp *tp) {
    ds4_tp_rdma *r = &tp->rdma;
    if (r->qp) r->api.destroy_qp(r->qp);
    if (r->mr) r->api.dereg_mr(r->mr);
    if (r->cq) r->api.destroy_cq(r->cq);
    if (r->pd) r->api.dealloc_pd(r->pd);
    if (r->ctx) r->api.close_device(r->ctx);
    r->qp = NULL; r->mr = NULL; r->cq = NULL; r->pd = NULL; r->ctx = NULL;
}

#endif /* DS4_TP_HAVE_VERBS */

/* ------------------------------------------------------------------------
 * Bring-up.
 * --------------------------------------------------------------------- */

static int tp_hello_exchange(ds4_tp *tp, const ds4_tp_identity *id, int rdma_ok,
                             char *err, size_t errlen) {
    ds4_tp_hello_fixed mine = {
        .magic = DS4_TP_MAGIC,
        .version = DS4_TP_PROTOCOL_VERSION,
        .role = (uint32_t)tp->opt.role,
        .rdma_ok = (uint32_t)rdma_ok,
        .null_split = getenv("DS4_TP_NULL_SPLIT") != NULL,
        .gguf_bytes = id->gguf_bytes,
        .model_id = id->model_id,
        .n_layer = id->n_layer,
        .n_embd = id->n_embd,
        .n_vocab = id->n_vocab,
        .quant_bits = id->quant_bits,
        .ctx_size = id->ctx_size,
    };
    ds4_tp_hello_fixed theirs;
    if (!tp_write_full(tp->control_fd, &mine, sizeof(mine)) ||
        !tp_read_full(tp->control_fd, &theirs, sizeof(theirs))) {
        tp_set_err(err, errlen, "tp hello exchange failed");
        return 0;
    }
    if (theirs.magic != DS4_TP_MAGIC) {
        tp_set_err(err, errlen, "tp hello: bad magic (mixed byte order or wrong peer?)");
        return 0;
    }
    if (theirs.version != DS4_TP_PROTOCOL_VERSION) {
        tp_set_err(err, errlen, "tp hello: protocol version %u != %u",
                   theirs.version, DS4_TP_PROTOCOL_VERSION);
        return 0;
    }
    if (theirs.role == mine.role) {
        tp_set_err(err, errlen, "tp hello: both sides claim role %u", mine.role);
        return 0;
    }
    if (theirs.null_split != mine.null_split) {
        tp_set_err(err, errlen, "tp hello: DS4_TP_NULL_SPLIT set on one side only");
        return 0;
    }
    if (theirs.gguf_bytes != mine.gguf_bytes || theirs.model_id != mine.model_id ||
        theirs.n_layer != mine.n_layer || theirs.n_embd != mine.n_embd ||
        theirs.n_vocab != mine.n_vocab || theirs.quant_bits != mine.quant_bits) {
        tp_set_err(err, errlen,
                   "tp hello: model mismatch (peer gguf=%llu id=%u layers=%u embd=%u "
                   "vocab=%u qbits=%u)",
                   (unsigned long long)theirs.gguf_bytes, theirs.model_id,
                   theirs.n_layer, theirs.n_embd, theirs.n_vocab, theirs.quant_bits);
        return 0;
    }
    tp->peer_ctx = theirs.ctx_size;
    tp->n_layer = id->n_layer;
    tp->n_embd = id->n_embd;
    tp->vec_bytes = (uint64_t)id->n_embd * sizeof(float);
    tp->n_slots = id->n_layer * DS4_TP_GATES_PER_LAYER;
    tp_slab_layout(tp);
    /* Transport decision: RDMA only when both sides can. */
    int want_rdma = tp->opt.transport != DS4_TP_TRANSPORT_TCP;
    tp->rdma_active = want_rdma && rdma_ok && theirs.rdma_ok;
    if (tp->opt.transport == DS4_TP_TRANSPORT_RDMA && !tp->rdma_active) {
        tp_set_err(err, errlen, "tp: --tp-transport rdma but %s side has no active device",
                   rdma_ok ? "the peer" : "this");
        return 0;
    }
    return 1;
}

int ds4_tp_create(
        ds4_tp **out,
        const ds4_tp_options *opt,
        const ds4_tp_identity *id,
        char *err,
        size_t errlen)
{
    *out = NULL;
    ds4_tp *tp = calloc(1, sizeof(*tp));
    if (!tp) {
        tp_set_err(err, errlen, "tp: out of memory");
        return 0;
    }
    tp->opt = *opt;
    tp->rank = opt->role == DS4_TP_LEADER ? 0 : 1;
    tp->control_fd = -1;
    tp->data_fd = -1;
    tp->timeout_sec = DS4_TP_DEFAULT_TIMEOUT_SEC;
    const char *tmo = getenv("DS4_TP_TIMEOUT_SEC");
    if (tmo) tp->timeout_sec = (uint64_t)atoi(tmo);

    int rdma_ok = 0;
#ifdef DS4_TP_HAVE_VERBS
    if (opt->transport != DS4_TP_TRANSPORT_TCP)
        rdma_ok = tp_rdma_probe(&tp->rdma.api);
#endif

    int listener = -1;
    if (tp->rank == 0) {
        listener = tp_listen(opt->listen_host, opt->listen_port, err, errlen);
        if (listener < 0) goto fail;
        fprintf(stderr, "ds4-tp: waiting for worker on %s:%d ...\n",
                opt->listen_host ? opt->listen_host : "0.0.0.0", opt->listen_port);
        tp->control_fd = accept(listener, NULL, NULL);
        if (tp->control_fd < 0) {
            tp_set_err(err, errlen, "tp accept: %s", strerror(errno));
            goto fail;
        }
    } else {
        tp->control_fd = tp_dial(opt->leader_host, opt->leader_port,
                                 (double)tp->timeout_sec, err, errlen);
        if (tp->control_fd < 0) goto fail;
    }
    tp_socket_tune(tp->control_fd);

    if (!tp_hello_exchange(tp, id, rdma_ok, err, errlen)) goto fail;

#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) {
        if (!tp_rdma_open(tp, err, errlen)) goto fail;
    }
#endif
    {
        /* Second socket dedicated to gate traffic so control frames never
         * interleave with 16KB gate payloads.  Created under RDMA too: the
         * verify-block batch gates always run over TCP. */
        if (tp->rank == 0) {
            tp->data_fd = accept(listener, NULL, NULL);
            if (tp->data_fd < 0) {
                tp_set_err(err, errlen, "tp data accept: %s", strerror(errno));
                goto fail;
            }
        } else {
            tp->data_fd = tp_dial(opt->leader_host, opt->leader_port,
                                  (double)tp->timeout_sec, err, errlen);
            if (tp->data_fd < 0) goto fail;
        }
        tp_socket_tune(tp->data_fd);
    }
    if (listener >= 0) close(listener);
    fprintf(stderr, "ds4-tp: %s connected, transport=%s\n",
            tp->rank == 0 ? "worker" : "leader",
            tp->rdma_active ? "rdma" : "tcp");
    *out = tp;
    return 1;
fail:
    if (listener >= 0) close(listener);
    ds4_tp_free(tp);
    return 0;
}

int ds4_tp_attach_slab(ds4_tp *tp, void *base, char *err, size_t errlen) {
    tp->slab = base;
    memset(tp->slab + tp->in_flags_off, 0, (uint64_t)tp->n_slots * 8);
    memset(tp->slab + tp->token_off, 0, 16);
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) return tp_rdma_register_and_exchange(tp, err, errlen);
#endif
    (void)err; (void)errlen;
    return 1;
}

void ds4_tp_free(ds4_tp *tp) {
    if (!tp) return;
#ifdef DS4_TP_HAVE_VERBS
    tp_rdma_close(tp);
#endif
    if (tp->control_fd >= 0) close(tp->control_fd);
    if (tp->data_fd >= 0) close(tp->data_fd);
    free(tp);
}

int ds4_tp_rank(const ds4_tp *tp) { return tp->rank; }
bool ds4_tp_is_rdma(const ds4_tp *tp) { return tp->rdma_active; }
uint32_t ds4_tp_peer_ctx(const ds4_tp *tp) { return tp->peer_ctx; }

/* ------------------------------------------------------------------------
 * Gate exchange.
 * --------------------------------------------------------------------- */

int ds4_tp_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) return tp_rdma_gate_exchange(tp, layer, gate, seq);
#endif
    /* TCP: both sides write their partial then read the peer's.  16KB per
     * direction fits comfortably in the socket buffers, so the symmetric
     * write-then-read cannot deadlock.  Header and payload go out in one
     * writev so NODELAY does not split them into two segments. */
    ds4_tp_gate_header h = { DS4_TP_MAGIC, (uint16_t)layer, (uint16_t)gate, seq };
    struct iovec iov[2] = {
        { &h, sizeof(h) },
        { tp->slab + ds4_tp_slab_out_offset(tp, layer, gate), tp->vec_bytes },
    };
    size_t want = sizeof(h) + tp->vec_bytes;
    ssize_t w = writev(tp->data_fd, iov, 2);
    if (w < 0 || (size_t)w != want) {
        /* Short writev: finish with the plain path. */
        if (w < 0) return 0;
        size_t done = (size_t)w;
        if (done < sizeof(h)) {
            if (!tp_write_full(tp->data_fd, (char *)&h + done, sizeof(h) - done)) return 0;
            done = sizeof(h);
        }
        uint64_t payload_done = done - sizeof(h);
        if (!tp_write_full(tp->data_fd,
                           tp->slab + ds4_tp_slab_out_offset(tp, layer, gate) + payload_done,
                           tp->vec_bytes - payload_done))
            return 0;
    }
    ds4_tp_gate_header ph;
    if (!tp_read_full(tp->data_fd, &ph, sizeof(ph))) return 0;
    if (ph.magic != DS4_TP_MAGIC || ph.layer != layer || ph.gate != gate || ph.seq != seq) {
        fprintf(stderr, "ds4-tp: gate desync: got l=%u g=%u seq=%llu, want l=%u g=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, gate, (unsigned long long)seq);
        return 0;
    }
    if (!tp_read_full(tp->data_fd, tp->slab + ds4_tp_slab_in_offset(tp, layer, gate),
                      tp->vec_bytes))
        return 0;
    return 1;
}

/* Verify-block batch gate: one exchange per layer moving all block rows at
 * once.  Runs over the TCP data socket even when the row gates use RDMA —
 * it is off the per-token latency path (43 exchanges per verify block) and
 * the RDMA engine's 16KB/seq accounting stays untouched.  Symmetric
 * write-then-read like the TCP row gate; 4MB socket buffers absorb the
 * <=128KB payloads without deadlock. */
int ds4_tp_batch_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t rows,
                               uint64_t seq) {
    if (tp->data_fd < 0 || rows == 0 || rows > DS4_TP_BATCH_MAX_ROWS) return 0;
    const uint64_t bytes = (uint64_t)rows * tp->vec_bytes;
    ds4_tp_gate_header h = { DS4_TP_BATCH_MAGIC, (uint16_t)layer,
                             (uint16_t)rows, seq };
    struct iovec iov[2] = {
        { &h, sizeof(h) },
        { tp->slab + ds4_tp_slab_batch_out_offset(tp, layer), bytes },
    };
    size_t want = sizeof(h) + bytes;
    ssize_t w = writev(tp->data_fd, iov, 2);
    if (w < 0) return 0;
    if ((size_t)w != want) {
        size_t done = (size_t)w;
        if (done < sizeof(h)) {
            if (!tp_write_full(tp->data_fd, (char *)&h + done, sizeof(h) - done))
                return 0;
            done = sizeof(h);
        }
        uint64_t payload_done = done - sizeof(h);
        if (!tp_write_full(tp->data_fd,
                           tp->slab + ds4_tp_slab_batch_out_offset(tp, layer) +
                               payload_done,
                           bytes - payload_done))
            return 0;
    }
    ds4_tp_gate_header ph;
    if (!tp_read_full(tp->data_fd, &ph, sizeof(ph))) return 0;
    if (ph.magic != DS4_TP_BATCH_MAGIC || ph.layer != layer ||
        ph.gate != rows || ph.seq != seq) {
        fprintf(stderr,
                "ds4-tp: batch gate desync: got l=%u rows=%u seq=%llu, "
                "want l=%u rows=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, rows, (unsigned long long)seq);
        return 0;
    }
    return tp_read_full(tp->data_fd,
                        tp->slab + ds4_tp_slab_batch_in_offset(tp, layer),
                        bytes);
}

/* ------------------------------------------------------------------------
 * Lockstep control plane.
 * --------------------------------------------------------------------- */

int ds4_tp_send_sync(ds4_tp *tp, const int *tokens, uint32_t n_tokens) {
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_SYNC,
                         tokens, n_tokens * sizeof(int));
}

int ds4_tp_send_sync_ack(ds4_tp *tp) {
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_SYNC_ACK, NULL, 0);
}

int ds4_tp_wait_sync_ack(ds4_tp *tp, char *err, size_t errlen) {
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_SYNC_ACK || bytes != 0) {
        tp_set_err(err, errlen, "tp: worker failed during prefill sync");
        return 0;
    }
    return 1;
}

int ds4_tp_send_eval(ds4_tp *tp, uint64_t seq, int token) {
    struct { uint64_t seq; int32_t token; } msg = { seq, token };
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_EVAL, &msg, sizeof(msg));
}

int ds4_tp_send_rewind(ds4_tp *tp, int pos) {
    int32_t v = pos;
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_REWIND, &v, sizeof(v));
}

int ds4_tp_send_invalidate(ds4_tp *tp) {
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_INVALIDATE, NULL, 0);
}

int ds4_tp_send_stop(ds4_tp *tp) {
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_STOP, NULL, 0);
}

int ds4_tp_recv_command(
        ds4_tp *tp,
        ds4_tp_frame_type *type,
        int **tokens,
        uint32_t *n_tokens,
        uint64_t *seq,
        int *token,
        char *err,
        size_t errlen)
{
    *type = DS4_TP_FRAME_ERROR;
    *tokens = NULL;
    *n_tokens = 0;
    uint32_t ftype = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &ftype, &bytes)) {
        tp_set_err(err, errlen, "tp: control channel closed");
        return 0;
    }
    switch (ftype) {
    case DS4_TP_FRAME_SYNC: {
        uint32_t count = bytes / sizeof(int);
        int *arr = malloc(bytes ? bytes : 4);
        if (!arr || (bytes && !tp_read_full(tp->control_fd, arr, bytes))) {
            free(arr);
            tp_set_err(err, errlen, "tp: truncated sync frame");
            return 0;
        }
        *tokens = arr;
        *n_tokens = count;
        *type = DS4_TP_FRAME_SYNC;
        return 1;
    }
    case DS4_TP_FRAME_EVAL: {
        struct { uint64_t seq; int32_t token; } msg;
        if (bytes != sizeof(msg) || !tp_read_full(tp->control_fd, &msg, sizeof(msg))) {
            tp_set_err(err, errlen, "tp: truncated eval frame");
            return 0;
        }
        *seq = msg.seq;
        *token = msg.token;
        *type = DS4_TP_FRAME_EVAL;
        return 1;
    }
    case DS4_TP_FRAME_REWIND: {
        int32_t v;
        if (bytes != sizeof(v) || !tp_read_full(tp->control_fd, &v, sizeof(v))) {
            tp_set_err(err, errlen, "tp: truncated rewind frame");
            return 0;
        }
        *token = v;
        *type = DS4_TP_FRAME_REWIND;
        return 1;
    }
    case DS4_TP_FRAME_VERIFY: {
        uint32_t count = bytes / sizeof(int);
        int *arr = malloc(bytes ? bytes : 4);
        if (!arr || (bytes && !tp_read_full(tp->control_fd, arr, bytes))) {
            free(arr);
            tp_set_err(err, errlen, "tp: truncated verify frame");
            return 0;
        }
        *tokens = arr;
        *n_tokens = count;
        *type = DS4_TP_FRAME_VERIFY;
        return 1;
    }
    case DS4_TP_FRAME_INVALIDATE:
        *type = DS4_TP_FRAME_INVALIDATE;
        return 1;
    case DS4_TP_FRAME_STOP:
        *type = DS4_TP_FRAME_STOP;
        return 1;
    default:
        tp_set_err(err, errlen, "tp: unexpected frame type %u", ftype);
        return 0;
    }
}

int ds4_tp_send_logits_half(ds4_tp *tp, const float *half, uint32_t count) {
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_LOGITS,
                         half, count * sizeof(float));
}

int ds4_tp_recv_logits_half(ds4_tp *tp, float *half, uint32_t count) {
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_LOGITS || bytes != count * sizeof(float)) {
        fprintf(stderr, "ds4-tp: bad logits frame (type %u bytes %u)\n", type, bytes);
        return 0;
    }
    return tp_read_full(tp->control_fd, half, bytes);
}

int ds4_tp_send_verify(ds4_tp *tp, const int *drafts, uint32_t n) {
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_VERIFY,
                         drafts, n * sizeof(int));
}

int ds4_tp_send_verify_commit(ds4_tp *tp, int32_t full_accept, int32_t replay_n) {
    struct { int32_t full; int32_t replay; } msg = { full_accept, replay_n };
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_VERIFY_COMMIT,
                         &msg, sizeof(msg));
}

int ds4_tp_recv_verify_commit(ds4_tp *tp, int32_t *full_accept, int32_t *replay_n) {
    uint32_t type = 0, bytes = 0;
    struct { int32_t full; int32_t replay; } msg;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_VERIFY_COMMIT || bytes != sizeof(msg) ||
        !tp_read_full(tp->control_fd, &msg, sizeof(msg))) {
        fprintf(stderr, "ds4-tp: bad verify-commit frame (type %u bytes %u)\n",
                type, bytes);
        return 0;
    }
    *full_accept = msg.full;
    *replay_n = msg.replay;
    return 1;
}

int ds4_tp_hash_check(ds4_tp *tp, uint64_t seq, uint64_t hash, char *err, size_t errlen) {
    struct { uint64_t seq; uint64_t hash; } mine = { seq, hash }, theirs;
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_HASH, &mine, sizeof(mine))) {
        tp_set_err(err, errlen, "tp: hash send failed");
        return 0;
    }
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_HASH || bytes != sizeof(theirs) ||
        !tp_read_full(tp->control_fd, &theirs, sizeof(theirs))) {
        tp_set_err(err, errlen, "tp: hash recv failed");
        return 0;
    }
    if (theirs.seq != seq || theirs.hash != hash) {
        tp_set_err(err, errlen,
                   "tp: LOCKSTEP DIVERGENCE at seq %llu: local %016llx peer %016llx",
                   (unsigned long long)seq,
                   (unsigned long long)hash, (unsigned long long)theirs.hash);
        return -1;
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Worker main loop.
 * --------------------------------------------------------------------- */

int ds4_tp_worker_run(ds4_engine *engine, const ds4_tp_options *opt) {
    char err[256] = "";
    ds4_tp_identity id = {
        .gguf_bytes = ds4_engine_model_bytes(engine),
        .model_id = (uint32_t)ds4_engine_model_id(engine),
        .n_layer = (uint32_t)ds4_engine_layer_count(engine),
        .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
        .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
        .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
        .ctx_size = 0, /* adopt the leader's */
    };

    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, opt, &id, err, sizeof(err))) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
        return 1;
    }
    if (!ds4_engine_tp_bind(engine, tp, err, sizeof(err))) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
        ds4_tp_free(tp);
        return 1;
    }
    ds4_session *session = NULL;
    int ctx = (int)ds4_tp_peer_ctx(tp);
    if (ctx <= 0) ctx = 8192;
    if (ds4_session_create(&session, engine, ctx) != 0) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: session create failed");
        ds4_tp_free(tp);
        return 1;
    }
    ds4_log(stderr, DS4_LOG_OK, "tp worker ready (ctx %d)", ctx);

    int rc = 0;
    ds4_tokens prompt = {0};
    while (1) {
        ds4_tp_frame_type type;
        int *tokens = NULL;
        uint32_t n_tokens = 0;
        uint64_t seq = 0;
        int token = 0;
        if (!ds4_tp_recv_command(tp, &type, &tokens, &n_tokens, &seq, &token,
                                 err, sizeof(err))) {
            ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
            rc = 1;
            break;
        }
        if (type == DS4_TP_FRAME_STOP) {
            ds4_log(stderr, DS4_LOG_DEFAULT, "tp worker: leader finished");
            break;
        } else if (type == DS4_TP_FRAME_SYNC) {
            prompt.len = 0;
            for (uint32_t i = 0; i < n_tokens; i++) ds4_tokens_push(&prompt, tokens[i]);
            free(tokens);
            int sync_rc = ds4_session_sync(session, &prompt, err, sizeof(err));
            if (sync_rc != 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker sync: %s", err);
                rc = 1;
                break;
            }
            if (!ds4_tp_send_sync_ack(tp)) {
                rc = 1;
                break;
            }
            if (getenv("DS4_TP_NULL_SPLIT") == NULL) {
                const int vocab = ds4_engine_vocab_size(engine);
                const uint32_t vhalf = (uint32_t)vocab / 2u;
                float *lg = malloc((size_t)vocab * sizeof(float));
                int sent = lg &&
                           ds4_session_copy_logits(session, lg, vocab) == vocab &&
                           ds4_tp_send_logits_half(tp, lg + vhalf, vhalf);
                free(lg);
                if (!sent) {
                    rc = 1;
                    break;
                }
            }
        } else if (type == DS4_TP_FRAME_EVAL) {
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker eval: %s", err);
                rc = 1;
                break;
            }
        } else if (type == DS4_TP_FRAME_VERIFY) {
            int spec_rc = ds4_session_tp_spec_cycle(session, tokens,
                                                    (int)n_tokens,
                                                    err, sizeof(err));
            free(tokens);
            if (spec_rc != 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker verify: %s", err);
                rc = 1;
                break;
            }
        } else if (type == DS4_TP_FRAME_REWIND) {
            ds4_session_rewind(session, token);
        } else if (type == DS4_TP_FRAME_INVALIDATE) {
            ds4_session_invalidate(session);
        } else {
            ds4_log(stderr, DS4_LOG_ERROR, "tp worker: unexpected frame %d", (int)type);
            rc = 1;
            break;
        }
    }
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_tp_free(tp);
    return rc;
}
