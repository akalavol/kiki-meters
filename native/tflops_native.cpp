// Module natif de TFLOPS Meter : calcul flottant CPU (AVX2/FMA) et mémoire (débit, latence).
// Compilé en DLL (voir build_native.cmd) et appelé depuis Python via ctypes.
//
// Toutes les fonctions sont bloquantes : Python les appelle depuis un thread de travail.
// tm_cancel(1) demande l'arrêt d'une mesure en cours (elle renvoie alors -1).

#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#define EXPORT extern "C" __declspec(dllexport)
static void* aligned_alloc64(size_t n) { return _aligned_malloc(n, 64); }
static void aligned_free64(void* p) { _aligned_free(p); }
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
static void* aligned_alloc64(size_t n) { void* p = nullptr; return posix_memalign(&p, 64, n) ? nullptr : p; }
static void aligned_free64(void* p) { free(p); }
#endif

using Clock = std::chrono::steady_clock;
static std::atomic<int> g_cancel{0};

static double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

EXPORT int tm_version() { return 4; }
EXPORT void tm_cancel(int v) { g_cancel.store(v); }

// Bits : 1 = AVX2, 2 = FMA, 4 = AVX-512F
EXPORT int tm_cpu_features() {
    __builtin_cpu_init();
    int f = 0;
    if (__builtin_cpu_supports("avx2")) f |= 1;
    if (__builtin_cpu_supports("fma")) f |= 2;
    if (__builtin_cpu_supports("avx512f")) f |= 4;
    return f;
}

static bool has_avx2_fma() { return (tm_cpu_features() & 3) == 3; }

// --------------------------------------------------------------------------- calcul flottant
// 12 accumulateurs indépendants : de quoi saturer 2 unités FMA de latence 4 cycles.

static volatile double g_sink;
static const uint64_t CHUNK = 1u << 18;

__attribute__((target("avx2,fma")))
static uint64_t fma_fp32_avx2(const std::atomic<bool>& stop) {
    __m256 a = _mm256_set1_ps(0.9999999f), b = _mm256_set1_ps(1e-7f);
    __m256 x0 = _mm256_set1_ps(0.10f), x1 = _mm256_set1_ps(0.11f), x2 = _mm256_set1_ps(0.12f),
           x3 = _mm256_set1_ps(0.13f), x4 = _mm256_set1_ps(0.14f), x5 = _mm256_set1_ps(0.15f),
           x6 = _mm256_set1_ps(0.16f), x7 = _mm256_set1_ps(0.17f), x8 = _mm256_set1_ps(0.18f),
           x9 = _mm256_set1_ps(0.19f), x10 = _mm256_set1_ps(0.20f), x11 = _mm256_set1_ps(0.21f);
    uint64_t n = 0;
    while (!stop.load(std::memory_order_relaxed)) {
        for (uint64_t i = 0; i < CHUNK; i++) {
            x0 = _mm256_fmadd_ps(x0, a, b); x1 = _mm256_fmadd_ps(x1, a, b);
            x2 = _mm256_fmadd_ps(x2, a, b); x3 = _mm256_fmadd_ps(x3, a, b);
            x4 = _mm256_fmadd_ps(x4, a, b); x5 = _mm256_fmadd_ps(x5, a, b);
            x6 = _mm256_fmadd_ps(x6, a, b); x7 = _mm256_fmadd_ps(x7, a, b);
            x8 = _mm256_fmadd_ps(x8, a, b); x9 = _mm256_fmadd_ps(x9, a, b);
            x10 = _mm256_fmadd_ps(x10, a, b); x11 = _mm256_fmadd_ps(x11, a, b);
        }
        n += CHUNK;
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3)),
                             _mm256_add_ps(_mm256_add_ps(x4, x5), _mm256_add_ps(x6, x7)));
    s = _mm256_add_ps(s, _mm256_add_ps(_mm256_add_ps(x8, x9), _mm256_add_ps(x10, x11)));
    float out[8]; _mm256_storeu_ps(out, s); g_sink = out[0];
    return n * 12 * 16;  // 12 FMA x 8 voies x 2 flops
}

__attribute__((target("avx2,fma")))
static uint64_t fma_fp64_avx2(const std::atomic<bool>& stop) {
    __m256d a = _mm256_set1_pd(0.9999999), b = _mm256_set1_pd(1e-7);
    __m256d x0 = _mm256_set1_pd(0.10), x1 = _mm256_set1_pd(0.11), x2 = _mm256_set1_pd(0.12),
            x3 = _mm256_set1_pd(0.13), x4 = _mm256_set1_pd(0.14), x5 = _mm256_set1_pd(0.15),
            x6 = _mm256_set1_pd(0.16), x7 = _mm256_set1_pd(0.17), x8 = _mm256_set1_pd(0.18),
            x9 = _mm256_set1_pd(0.19), x10 = _mm256_set1_pd(0.20), x11 = _mm256_set1_pd(0.21);
    uint64_t n = 0;
    while (!stop.load(std::memory_order_relaxed)) {
        for (uint64_t i = 0; i < CHUNK; i++) {
            x0 = _mm256_fmadd_pd(x0, a, b); x1 = _mm256_fmadd_pd(x1, a, b);
            x2 = _mm256_fmadd_pd(x2, a, b); x3 = _mm256_fmadd_pd(x3, a, b);
            x4 = _mm256_fmadd_pd(x4, a, b); x5 = _mm256_fmadd_pd(x5, a, b);
            x6 = _mm256_fmadd_pd(x6, a, b); x7 = _mm256_fmadd_pd(x7, a, b);
            x8 = _mm256_fmadd_pd(x8, a, b); x9 = _mm256_fmadd_pd(x9, a, b);
            x10 = _mm256_fmadd_pd(x10, a, b); x11 = _mm256_fmadd_pd(x11, a, b);
        }
        n += CHUNK;
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0, x1), _mm256_add_pd(x2, x3)),
                              _mm256_add_pd(_mm256_add_pd(x4, x5), _mm256_add_pd(x6, x7)));
    s = _mm256_add_pd(s, _mm256_add_pd(_mm256_add_pd(x8, x9), _mm256_add_pd(x10, x11)));
    double out[4]; _mm256_storeu_pd(out, s); g_sink = out[0];
    return n * 12 * 8;  // 12 FMA x 4 voies x 2 flops
}

// Repli SSE2 (processeurs sans AVX2/FMA) : multiplication + addition séparées.
static uint64_t mad_fp32_sse(const std::atomic<bool>& stop) {
    __m128 a = _mm_set1_ps(0.9999999f), b = _mm_set1_ps(1e-7f);
    __m128 x[8];
    for (int k = 0; k < 8; k++) x[k] = _mm_set1_ps(0.1f + 0.01f * k);
    uint64_t n = 0;
    while (!stop.load(std::memory_order_relaxed)) {
        for (uint64_t i = 0; i < CHUNK; i++)
            for (int k = 0; k < 8; k++) x[k] = _mm_add_ps(_mm_mul_ps(x[k], a), b);
        n += CHUNK;
    }
    float out[4]; _mm_storeu_ps(out, _mm_add_ps(x[0], x[7])); g_sink = out[0];
    return n * 8 * 8;
}

static uint64_t mad_fp64_sse(const std::atomic<bool>& stop) {
    __m128d a = _mm_set1_pd(0.9999999), b = _mm_set1_pd(1e-7);
    __m128d x[8];
    for (int k = 0; k < 8; k++) x[k] = _mm_set1_pd(0.1 + 0.01 * k);
    uint64_t n = 0;
    while (!stop.load(std::memory_order_relaxed)) {
        for (uint64_t i = 0; i < CHUNK; i++)
            for (int k = 0; k < 8; k++) x[k] = _mm_add_pd(_mm_mul_pd(x[k], a), b);
        n += CHUNK;
    }
    double out[2]; _mm_storeu_pd(out, _mm_add_pd(x[0], x[7])); g_sink = out[0];
    return n * 8 * 4;
}

// precision : 0 = FP32, 1 = FP64. Renvoie des GFLOPS (-1 si annulé).
EXPORT double tm_cpu_flops(int precision, int threads, double seconds) {
    g_cancel.store(0);
    if (threads < 1) threads = (int)std::max(1u, std::thread::hardware_concurrency());
    const bool avx = has_avx2_fma();
    using Fn = uint64_t (*)(const std::atomic<bool>&);
    Fn fn = precision == 1 ? (avx ? fma_fp64_avx2 : mad_fp64_sse) : (avx ? fma_fp32_avx2 : mad_fp32_sse);

    std::atomic<bool> stop{false};
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<uint64_t> flops(threads, 0);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&, t] {
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            flops[t] = fn(stop);
        });
    while (ready.load() < threads) std::this_thread::yield();
    auto t0 = Clock::now();
    go.store(true);
    while (seconds_since(t0) < seconds && !g_cancel.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    stop.store(true);
    for (auto& th : pool) th.join();
    double dt = seconds_since(t0);
    if (g_cancel.load()) return -1.0;
    uint64_t total = 0;
    for (auto f : flops) total += f;
    return (double)total / dt / 1e9;
}

// --------------------------------------------------------------------------- débit mémoire

__attribute__((target("avx2")))
static double read_avx2(const char* p, size_t n) {
    const __m256i* v = (const __m256i*)p;
    size_t count = n / 32;
    __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        s0 = _mm256_add_epi64(s0, _mm256_load_si256(v + i));
        s1 = _mm256_add_epi64(s1, _mm256_load_si256(v + i + 1));
        s2 = _mm256_add_epi64(s2, _mm256_load_si256(v + i + 2));
        s3 = _mm256_add_epi64(s3, _mm256_load_si256(v + i + 3));
    }
    __m256i s = _mm256_add_epi64(_mm256_add_epi64(s0, s1), _mm256_add_epi64(s2, s3));
    long long out[4]; _mm256_storeu_si256((__m256i*)out, s);
    return (double)(out[0] + out[1] + out[2] + out[3]);
}

static double read_sse(const char* p, size_t n) {
    const __m128i* v = (const __m128i*)p;
    size_t count = n / 16;
    __m128i s0 = _mm_setzero_si128(), s1 = s0;
    for (size_t i = 0; i + 2 <= count; i += 2) {
        s0 = _mm_add_epi64(s0, _mm_load_si128(v + i));
        s1 = _mm_add_epi64(s1, _mm_load_si128(v + i + 1));
    }
    long long out[2]; _mm_storeu_si128((__m128i*)out, _mm_add_epi64(s0, s1));
    return (double)(out[0] + out[1]);
}

__attribute__((target("avx2")))
static void write_avx2(char* p, size_t n) {
    __m256i z = _mm256_set1_epi32(0x5A5A5A5A);
    __m256i* v = (__m256i*)p;
    size_t count = n / 32;
    for (size_t i = 0; i < count; i++) _mm256_stream_si256(v + i, z);
    _mm_sfence();
}

static void write_sse(char* p, size_t n) {
    __m128i z = _mm_set1_epi32(0x5A5A5A5A);
    __m128i* v = (__m128i*)p;
    size_t count = n / 16;
    for (size_t i = 0; i < count; i++) _mm_stream_si128(v + i, z);
    _mm_sfence();
}

// mode : 0 = lecture, 1 = écriture, 2 = copie. Renvoie des Go/s (1 Go = 1e9 octets), -1 si annulé/erreur.
// Pour la copie, le débit compte les octets copiés (convention des outils grand public).
EXPORT double tm_mem_bandwidth(int mode, double size_mb, int threads, double seconds) {
    g_cancel.store(0);
    if (threads < 1) threads = (int)std::max(1u, std::thread::hardware_concurrency());
    size_t bytes = (size_t)(size_mb * 1024 * 1024);
    bytes -= bytes % (64 * (size_t)threads * 4);
    if (bytes == 0) return -1.0;
    char* src = (char*)aligned_alloc64(bytes);
    char* dst = mode == 2 ? (char*)aligned_alloc64(bytes) : nullptr;
    if (!src || (mode == 2 && !dst)) { aligned_free64(src); aligned_free64(dst); return -1.0; }

    const size_t slice = bytes / threads;
    const bool avx = (tm_cpu_features() & 1) != 0;
    auto parallel = [&](auto&& body) {
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; t++) pool.emplace_back([&, t] { body(t); });
        for (auto& th : pool) th.join();
    };
    // Premier contact en parallèle : les pages sont réparties comme pendant la mesure.
    parallel([&](int t) {
        memset(src + t * slice, t + 1, slice);
        if (dst) memset(dst + t * slice, 0, slice);
    });

    std::vector<double> sinks(threads);
    double best = 0.0;
    auto t_start = Clock::now();
    int passes = 0;
    while ((passes < 3 || seconds_since(t_start) < seconds) && passes < 10000) {
        if (g_cancel.load()) { best = -1.0; break; }
        auto t0 = Clock::now();
        parallel([&](int t) {
            char* s = src + t * slice;
            if (mode == 0) sinks[t] += avx ? read_avx2(s, slice) : read_sse(s, slice);
            else if (mode == 1) { if (avx) write_avx2(s, slice); else write_sse(s, slice); }
            else memcpy(dst + t * slice, s, slice);
        });
        double dt = seconds_since(t0);
        best = std::max(best, (double)bytes / dt / 1e9);
        passes++;
    }
    for (double s : sinks) g_sink += s;
    aligned_free64(src);
    aligned_free64(dst);
    return best;
}

// --------------------------------------------------------------------------- latence mémoire
// Parcours de pointeurs dans un cycle aléatoire (une ligne de cache par élément) :
// chaque accès dépend du précédent, le temps mesuré est donc la latence réelle.

EXPORT double tm_mem_latency(double size_mb, double seconds) {
    g_cancel.store(0);
    const size_t line = 64;
    size_t n = (size_t)(size_mb * 1024 * 1024) / line;
    if (n < 16) return -1.0;
    char* buf = (char*)aligned_alloc64(n * line);
    if (!buf) return -1.0;

    std::vector<uint32_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = (uint32_t)i;
    std::mt19937_64 rng(12345);
    for (size_t i = n - 1; i > 0; i--) {  // algorithme de Sattolo : un seul grand cycle
        std::uniform_int_distribution<size_t> d(0, i - 1);
        std::swap(order[i], order[d(rng)]);
    }
    for (size_t i = 0; i < n; i++)
        *(void**)(buf + (size_t)order[i] * line) = buf + (size_t)order[(i + 1) % n] * line;
    std::vector<uint32_t>().swap(order);

    void** p = (void**)buf;
    for (size_t i = 0; i < n; i++) p = (void**)*p;  // échauffement (TLB, caches)

    const uint64_t steps = 1u << 20;
    double best = 1e30;
    auto t_start = Clock::now();
    int rounds = 0;
    while ((rounds < 3 || seconds_since(t_start) < seconds) && rounds < 100000) {
        if (g_cancel.load()) { best = -1.0; break; }
        auto t0 = Clock::now();
        for (uint64_t i = 0; i < steps; i += 8) {
            p = (void**)*p; p = (void**)*p; p = (void**)*p; p = (void**)*p;
            p = (void**)*p; p = (void**)*p; p = (void**)*p; p = (void**)*p;
        }
        best = std::min(best, seconds_since(t0) * 1e9 / (double)steps);
        rounds++;
    }
    g_sink = (double)(uintptr_t)p;
    aligned_free64(buf);
    return best;
}

// --------------------------------------------------------------------------- stockage (Windows)
// E/S directes (FILE_FLAG_NO_BUFFERING | WRITE_THROUGH) : le cache de Windows est contourné,
// on mesure le disque et non la RAM. Les données écrites sont aléatoires (pas de compression possible).
//
// mode : 0 = préparation (création + écriture séquentielle complète, 1 thread) — renvoie aussi des Mo/s
//        1 = écriture séquentielle   2 = lecture séquentielle
//        3 = lecture aléatoire       4 = écriture aléatoire
// threads = nombre d'E/S simultanées (profondeur de file). Renvoie des Mo/s (1 Mo = 1e6 octets),
// -1 si annulé, -2 si erreur (code Windows via tm_last_error).

#ifdef _WIN32
static std::atomic<unsigned long> g_last_error{0};
EXPORT unsigned long tm_last_error() { return g_last_error.load(); }

EXPORT double tm_disk(const wchar_t* path, double size_mb, int mode, int block_kb, int threads, double seconds) {
    g_cancel.store(0);
    g_last_error.store(0);
    if (threads < 1) threads = 1;
    if (mode == 0) threads = 1;
    const uint64_t block = (uint64_t)block_kb * 1024;
    uint64_t size = (uint64_t)(size_mb * 1024 * 1024);
    size -= size % (block * (uint64_t)threads);
    if (size == 0 || block % 4096) return -2.0;

    if (mode == 0) {
        HANDLE h = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) { g_last_error.store(GetLastError()); return -2.0; }
        CloseHandle(h);
    } else {
        WIN32_FILE_ATTRIBUTE_DATA fa;
        if (!GetFileAttributesExW(path, GetFileExInfoStandard, &fa)) { g_last_error.store(GetLastError()); return -2.0; }
        uint64_t existing = ((uint64_t)fa.nFileSizeHigh << 32) | fa.nFileSizeLow;
        size = std::min(size, existing - existing % (block * (uint64_t)threads));
        if (size == 0) return -2.0;
    }

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> total{0};
    std::atomic<int> failed{0};
    const bool write = (mode == 0 || mode == 1 || mode == 4);
    const bool random_io = (mode == 3 || mode == 4);
    const uint64_t region = size / threads;

    auto worker = [&](int t) {
        HANDLE h = CreateFileW(path, GENERIC_READ | GENERIC_WRITE,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
        if (h == INVALID_HANDLE_VALUE) { g_last_error.store(GetLastError()); failed.store(1); return; }
        unsigned char* buf = (unsigned char*)aligned_alloc64(block + 4096);
        unsigned char* io = (unsigned char*)(((uintptr_t)buf + 4095) & ~(uintptr_t)4095);
        std::mt19937_64 rng(0x9E3779B97F4A7C15ull ^ (uint64_t)(t + 1) * 0x2545F4914F6CDD1Dull);
        for (uint64_t i = 0; i + 8 <= block; i += 8) *(uint64_t*)(io + i) = rng();
        const uint64_t blocks_total = size / block;
        uint64_t pos = (uint64_t)t * region, done = 0;
        while (!stop.load(std::memory_order_relaxed) && !g_cancel.load()) {
            uint64_t off;
            if (random_io) {
                off = (rng() % blocks_total) * block;
            } else {
                if (pos >= (uint64_t)(t + 1) * region) {
                    if (mode == 0) break;          // préparation : une seule passe
                    pos = (uint64_t)t * region;
                }
                off = pos;
                pos += block;
            }
            OVERLAPPED ov = {};
            ov.Offset = (DWORD)(off & 0xFFFFFFFFu);
            ov.OffsetHigh = (DWORD)(off >> 32);
            DWORD n = 0;
            BOOL ok = write ? WriteFile(h, io, (DWORD)block, &n, &ov) : ReadFile(h, io, (DWORD)block, &n, &ov);
            if (!ok || n != block) { g_last_error.store(GetLastError()); failed.store(1); break; }
            done += block;
            if (write && (done & ((1u << 26) - 1)) == 0)  // varie les données toutes les 64 Mo
                *(uint64_t*)io ^= rng();
        }
        total.fetch_add(done);
        aligned_free64(buf);
        CloseHandle(h);
    };

    auto t0 = Clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back(worker, t);
    if (mode != 0) {
        while (seconds_since(t0) < seconds && !g_cancel.load() && !failed.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        stop.store(true);
    }
    for (auto& th : pool) th.join();
    double dt = seconds_since(t0);
    if (g_cancel.load()) return -1.0;
    if (failed.load() && total.load() == 0) return -2.0;
    return (double)total.load() / dt / 1e6;
}

EXPORT int tm_disk_delete(const wchar_t* path) { return DeleteFileW(path) ? 1 : 0; }
#endif
