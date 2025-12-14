/* scheduler.c
   Modes:
    - pot   : Classic Power-of-Two (queue length only)
    - aipot : AI-PoT (Ridge regression + expected_finish + node_scale + pred_err)
    - rr    : Round Robin (static)
*/

/* Define POSIX source for clock_gettime, pread, pwrite, etc. */
#define _POSIX_C_SOURCE 200809L

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>   /* memcpy, memset, strcmp, strncpy, etc. */
#include <time.h>     /* clock_gettime, time */
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <limits.h>   /* INT_MAX */

#define TAG_TASK 100
#define TAG_DONE 101
#define TAG_SHUT 102
#define TAG_CAL  105

#define MAX_WORKERS 64
#define MAX_TASKS   200000

/* ---------- Utility: time in ms ---------- */
static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---------- Logging ---------- */
static void write_log_header(void) {
    FILE *f = fopen("logs/run_tasks.csv", "w");
    if (!f) return;
    fprintf(f, "req_id,t,size,op,seq,key,pred_ms,assigned,actual_ms,node_scale_before,node_scale_after\n");
    fclose(f);
}
static void log_row(int req_id, double t, long size, int op, int seq, int key,
                    double pred_ms, int assigned, double actual_ms,
                    double ns_before, double ns_after) {
    FILE *f = fopen("logs/run_tasks.csv", "a");
    if (!f) return;
    fprintf(f, "%d,%.3f,%ld,%d,%d,%d,%.3f,%d,%.3f,%.3f,%.3f\n",
            req_id, t, size, op, seq, key, pred_ms, assigned,
            actual_ms, ns_before, ns_after);
    fclose(f);
}

/* ---------- Ridge Regression model (intercept + 5 coefs) ---------- */
static double rr_intercept = 0.0;
static double rr_coef[5]   = {0,0,0,0,0};

static int read_coeffs(const char *path, double *intercept, double coef[5]) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int r = fscanf(f, "%lf %lf %lf %lf %lf %lf",
                   intercept, &coef[0], &coef[1], &coef[2], &coef[3], &coef[4]);
    fclose(f);
    return (r == 6);
}

/* Classic log-space Ridge model:
   features: log_size, op, seq, op*log_size, seq*log_size
*/
static double predict_ms(double size, int op, int seq) {
    double x0 = log(size + 1.0);
    double x1 = (double)op;
    double x2 = (double)seq;
    double x3 = x1 * x0;
    double x4 = x2 * x0;

    double logp = rr_intercept
        + rr_coef[0]*x0 + rr_coef[1]*x1 + rr_coef[2]*x2
        + rr_coef[3]*x3 + rr_coef[4]*x4;

    double p = exp(logp);
    if (p < 0.1) p = 0.1;
    return p;
}

/* ---------- CPU / IO simulation ---------- */
static void do_cpu_work_simple(size_t bytes) {
    const size_t buf_size = 64 * 1024;
    unsigned char *buf = (unsigned char*)malloc(buf_size);
    unsigned char *dst = (unsigned char*)malloc(buf_size);
    if (!buf || !dst) { free(buf); free(dst); return; }
    memset(buf, 0x5a, buf_size);
    size_t loops = (bytes + buf_size - 1) / buf_size;
    for (size_t i = 0; i < loops; i++) {
        memcpy(dst, buf, buf_size);
        for (int j = 0; j < 16; j++) dst[j] ^= (unsigned char)j;
    }
    free(buf); free(dst);
}

static double do_file_io_simple(int fd, off_t *next_offset, size_t size,
                                int is_write, int is_sequential, size_t file_size) {
    double t0 = now_ms();
    size_t remaining = size;
    const size_t chunk = 64 * 1024;
    static unsigned char *buf = NULL;
    if (!buf) {
        buf = (unsigned char*)malloc(128 * 1024);
        if (buf) memset(buf, 0x5b, 128 * 1024);
    }
    while (remaining > 0) {
        size_t to = remaining < chunk ? remaining : chunk;
        off_t offset;
        if (is_sequential) {
            offset = *next_offset; *next_offset += to;
            if (*next_offset >= (off_t)file_size) *next_offset = 0;
        } else {
            off_t max_off = (off_t)file_size - (off_t)to;
            if (max_off < 0) max_off = 0;
            offset = (off_t)((rand() % (max_off / chunk + 1)) * chunk);
        }
        ssize_t rc;
        if (is_write) rc = pwrite(fd, buf, to, offset);
        else          rc = pread(fd,  buf, to, offset);
        (void)rc;
        remaining -= to;
    }
    return now_ms() - t0;
}

/* -------------------- MAIN -------------------- */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank=0, np=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (np < 3) {
        if (rank==0) fprintf(stderr,"Need at least 3 processes (1 master + 2 workers)\n");
        MPI_Finalize();
        return 1;
    }

    char mode[16] = "pot";
    int total_tasks = 1000;
    int seed = (int)time(NULL);

    for (int i=1;i<argc;i++) {
        if (strcmp(argv[i],"--mode")==0 && i+1<argc) {
            strncpy(mode, argv[++i], 15); mode[15]=0;
        } else if (strcmp(argv[i],"--tasks")==0 && i+1<argc) {
            total_tasks = atoi(argv[++i]);
        } else if (strcmp(argv[i],"--seed")==0 && i+1<argc) {
            seed = atoi(argv[++i]);
        }
    }

    srand(seed + 101);

    int use_ai = (strcmp(mode,"aipot")==0);

    if (rank==0) {
        system("mkdir -p logs");
        write_log_header();
    }

    /* Load Ridge regression coefficients for AI-PoT */
    if (use_ai && rank==0) {
        if (!read_coeffs("model/coeffs.txt", &rr_intercept, rr_coef)) {
            fprintf(stderr,
                    "Master: Ridge coeffs file model/coeffs.txt not found or invalid. AI-PoT will not work.\n");
        } else {
            printf("Master: Loaded Ridge model: intercept=%f coefs=%f %f %f %f %f\n",
                   rr_intercept, rr_coef[0], rr_coef[1], rr_coef[2], rr_coef[3], rr_coef[4]);
        }
    }

    /* Shared structures */
    const int nworkers = np - 1;
    double expected_finish[MAX_WORKERS];   /* sum of pending predicted times per worker */
    int    queue_len[MAX_WORKERS];         /* integer queue length */
    double node_scale[MAX_WORKERS];        /* EMA multiplier for each worker */
    double pred_map[MAX_TASKS+5];          /* per-task predicted times */
    double reservation_map[MAX_TASKS+5];   /* reserved predicted time (for expected_finish) */
    double pred_err[MAX_WORKERS];          /* EMA of relative prediction error per worker */

    static long   feat_size[MAX_TASKS+5];
    static int    feat_op[MAX_TASKS+5];
    static int    feat_seq[MAX_TASKS+5];
    static int    feat_key[MAX_TASKS+5];
    static double feat_ns_before[MAX_TASKS+5];
    static int    feat_assigned_worker[MAX_TASKS+5];

    for (int i=0;i<MAX_WORKERS;i++) {
        expected_finish[i]=0.0;
        queue_len[i]=0;
        node_scale[i]=1.0;
        pred_err[i]=0.0;
    }

    if (rank==0) {
        printf("Master: mode=%s total_tasks=%d seed=%d workers=%d\n",
               mode, total_tasks, seed, nworkers);

        /* Calibration: measure baseline time per worker */
        int K = 3;
        for (int w=1; w<=nworkers; w++)
            MPI_Send(&K, 1, MPI_INT, w, TAG_CAL, MPI_COMM_WORLD);

        for (int w=1; w<=nworkers; w++) {
            double res=0.0;
            MPI_Recv(&res, 1, MPI_DOUBLE, w, TAG_CAL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double baseline = 1.0 + 0.6 * log2(8192 + 1) - 2.0;
            if (baseline < 0.1) baseline = 0.1;
            node_scale[w] = res / baseline;
            if (node_scale[w] < 0.5) node_scale[w] = 0.5;
            if (node_scale[w] > 8.0) node_scale[w] = 8.0;
            printf("Calibrated worker %d: avg=%.3f baseline=%.3f scale=%.3f\n",
                   w, res, baseline, node_scale[w]);
        }

        int issued=0, completed=0;
        double start_time = now_ms();
        static int rr_idx = 1;   /* RR pointer */

        while (completed < total_tasks) {
            /* Handle completions (non-blocking) */
            int flag=0;
            MPI_Status st;
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &flag, &st);
            while (flag) {
                double donebuf[2];
                MPI_Recv(donebuf, 2, MPI_DOUBLE, st.MPI_SOURCE, TAG_DONE,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int rid = (int)donebuf[0];
                double actual_ms = donebuf[1];
                int src = st.MPI_SOURCE;

                /* Update expected_finish: subtract reserved predicted time */
                double reserved = reservation_map[rid];
                if (reserved <= 0)
                    reserved = (pred_map[rid] > 0 ? pred_map[rid] : actual_ms);
                expected_finish[src] -= reserved;
                if (expected_finish[src] < 0) expected_finish[src] = 0.0;

                /* Queue length update */
                if (queue_len[src] > 0) queue_len[src]--;

                /* Prediction error tracking for AI confidence */
                double pred_ms = (pred_map[rid] > 0) ? pred_map[rid] : 1.0;
                double rel_err = fabs(actual_ms - pred_ms) / (pred_ms + 1e-6);
                pred_err[src] = 0.05 * rel_err + 0.95 * pred_err[src];

                /* Node scale EMA update */
                double s_inst = actual_ms / pred_ms;
                double ns_before = node_scale[src];
                node_scale[src] = 0.05 * s_inst + 0.95 * node_scale[src];
                double ns_after = node_scale[src];

                log_row(rid, now_ms(),
                        feat_size[rid], feat_op[rid], feat_seq[rid], feat_key[rid],
                        pred_ms, feat_assigned_worker[rid], actual_ms,
                        feat_ns_before[rid], ns_after);

                completed++;
                MPI_Iprobe(MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &flag, &st);
            }

            /* Issue new task */
            if (issued < total_tasks) {
                int tid = ++issued;

                /* Bimodal task distribution: 90% "mice", 10% "elephants" */
                double u = (double)rand()/RAND_MAX;
                double size;
                if (u < 0.90) {
                    size = 1024 + (rand() % 4096);  /* 1KB..5KB */
                } else {
                    size = (double)(1024*1024) * (50.0 + (rand()%50)); /* 50..99 MB */
                }
                if (size < 1.0) size = 1.0;

                int op  = (rand()%100) < 50 ? 1 : 0;
                int seq = (rand()%100) < 50 ? 1 : 0;
                int key = rand()%1000;

                /* Snapshot of queue lengths if needed for debugging/analysis */
                int qsnapshot[MAX_WORKERS];
                for (int w=1; w<=nworkers; w++) qsnapshot[w] = queue_len[w];

                double pred_ms = 0.0;
                if (!use_ai) {
                    /* Analytic heuristic when NOT using AI */
                    pred_ms = 1.0 + 0.6 * log2(size + 1) + (op ? 5.0 : 0.0) - (seq ? 2.0 : 0.0);
                    if (pred_ms < 0.1) pred_ms = 0.1;
                }

                int    chosen  = -1;
                double reserve = 0.0;

                if (strcmp(mode,"rr")==0) {
                    /* ---------- Round Robin ---------- */
                    chosen = rr_idx;
                    rr_idx = (rr_idx % nworkers) + 1;
                    reservation_map[tid] = 0.0;
                    pred_map[tid]        = 0.0;
                }
                else if (strcmp(mode,"aipot")==0) {
                    /* ---------- AI-PoT ---------- */
                    int a = (rand()%nworkers)+1;
                    int b = (rand()%nworkers)+1;
                    while (b==a) b = (rand()%nworkers)+1;

                    double cand_pred_a = predict_ms(size, op, seq);
                    double cand_pred_b = predict_ms(size, op, seq);

                    double threshold = 0.6; /* 60% relative error threshold */
                    double penalty_a = pred_err[a] > threshold ? 1.5 : 1.0;
                    double penalty_b = pred_err[b] > threshold ? 1.5 : 1.0;

                    double score_a = expected_finish[a]
                                   + ((double)queue_len[a] * node_scale[a])
                                   + penalty_a * cand_pred_a;
                    double score_b = expected_finish[b]
                                   + ((double)queue_len[b] * node_scale[b])
                                   + penalty_b * cand_pred_b;

                    chosen = (score_a <= score_b) ? a : b;

                    pred_ms = (chosen==a) ? cand_pred_a : cand_pred_b;
                    reserve = fmin(pred_ms, 20000.0);

                    expected_finish[chosen] += reserve;
                    reservation_map[tid] = reserve;
                    pred_map[tid]        = pred_ms;
                }
                else {
                    /* ---------- Classic PoT (default) ---------- */
                    int a = (rand()%nworkers)+1;
                    int b = (rand()%nworkers)+1;
                    while (b==a) b=(rand()%nworkers)+1;
                    int qa = queue_len[a];
                    int qb = queue_len[b];
                    chosen = (qa <= qb) ? a : b;
                    reservation_map[tid] = 0.0;
                    pred_map[tid]        = 0.0;
                }

                if (chosen < 1 || chosen > nworkers) chosen = 1;
                queue_len[chosen]++;

                feat_size[tid]       = (long)size;
                feat_op[tid]         = op;
                feat_seq[tid]        = seq;
                feat_key[tid]        = key;
                feat_ns_before[tid]  = node_scale[chosen];
                feat_assigned_worker[tid] = chosen;

                double buf[6] = { (double)tid, size, (double)op, (double)seq, (double)key, pred_ms };
                MPI_Send(buf, 6, MPI_DOUBLE, chosen, TAG_TASK, MPI_COMM_WORLD);
            } else {
                usleep(1000);
            }
        } /* end master loop */

        double total_ms = now_ms() - start_time;
        printf("All tasks completed. Total time: %.3f ms\n", total_ms);

        for (int w=1; w<=nworkers; w++) {
            double s = -1.0;
            MPI_Send(&s, 1, MPI_DOUBLE, w, TAG_SHUT, MPI_COMM_WORLD);
        }

    } else {
        /* ======================= WORKER ======================= */
        const int wrank = rank;
        double slow_factor = 1.0;
        if (wrank==2) slow_factor = 8.0;  /* strong straggler */
        if (wrank==3) slow_factor = 4.0;  /* slow node */
        if (wrank==4) slow_factor = 0.5;  /* faster node */

        char path[256];
        snprintf(path, sizeof(path), "worker_data/worker%d.dat", wrank);
        int fd = open(path, O_RDWR);
        if (fd < 0) perror("open worker file");

#ifdef __APPLE__
#ifdef F_NOCACHE
        {
            int val = 1;
            (void)fcntl(fd, F_NOCACHE, val);
        }
#endif
#endif

        struct stat stbuf;
        fstat(fd, &stbuf);
        size_t file_size = (size_t)stbuf.st_size;
        off_t  next_offset = 0;

        /* Calibration */
        MPI_Status stt;
        int k=0;
        MPI_Recv(&k, 1, MPI_INT, 0, TAG_CAL, MPI_COMM_WORLD, &stt);
        double sumt=0.0;
        for (int i=0;i<k;i++)
            sumt += do_file_io_simple(fd, &next_offset, 8192, 0, 1, file_size);
        double mean = sumt / (k>0?k:1);
        MPI_Send(&mean, 1, MPI_DOUBLE, 0, TAG_CAL, MPI_COMM_WORLD);

        /* Main worker loop */
        while (1) {
            MPI_Status status;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_SHUT) {
                double tmp;
                MPI_Recv(&tmp, 1, MPI_DOUBLE, 0, TAG_SHUT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            } else if (status.MPI_TAG == TAG_TASK) {
                double buf[6];
                MPI_Recv(buf, 6, MPI_DOUBLE, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int  tid  = (int)buf[0];
                long size = (long)buf[1];
                int  op   = (int)buf[2];
                int  seq  = (int)buf[3];
                (void)buf[4]; (void)buf[5];

                double t0 = now_ms();
                if ((size_t)size <= 16*1024) {
                    do_cpu_work_simple((size_t)size);
                } else {
                    (void)do_file_io_simple(fd, &next_offset, (size_t)size, op==1, seq==1, file_size);
                }
                double work_time = now_ms() - t0;

                double jitter = (double)(rand()%10); /* 0..9ms noise */
                if (jitter>0) usleep((useconds_t)(jitter*1000.0));
                double actual_ms = work_time + jitter;

                if (slow_factor != 1.0) {
                    double extra_ms = (slow_factor - 1.0) * actual_ms;
                    if (extra_ms > 0) usleep((useconds_t)(extra_ms * 1000.0));
                    if (slow_factor > 0) actual_ms *= slow_factor;
                }

                double done[2] = { (double)tid, actual_ms };
                MPI_Send(done, 2, MPI_DOUBLE, 0, TAG_DONE, MPI_COMM_WORLD);
            } else {
                double tmp;
                MPI_Recv(&tmp, 1, MPI_DOUBLE, 0, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        if (fd>=0) close(fd);
    }

    MPI_Finalize();
    return 0;
}
