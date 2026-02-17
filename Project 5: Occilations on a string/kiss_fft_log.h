/*
 *  KISS FFT Logging Header Stub
 *  This is a minimal stub to satisfy the include requirement in _kiss_fft_guts.h
 */

#ifndef _kiss_fft_log_h
#define _kiss_fft_log_h

#include <stdio.h>
#include <stdlib.h>

#ifdef KISS_FFT_DEBUG_ENABLED
#define KISS_FFT_DEBUG(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#define KISS_FFT_WARNING(fmt, ...) fprintf(stderr, "WARNING: " fmt, ##__VA_ARGS__)
#define KISS_FFT_ERROR(fmt, ...)                       \
    do {                                               \
        fprintf(stderr, "ERROR: " fmt, ##__VA_ARGS__); \
        exit(1);                                       \
    } while (0)
#else
#define KISS_FFT_DEBUG(fmt, ...) ((void)0)
#define KISS_FFT_WARNING(fmt, ...) ((void)0)
#define KISS_FFT_ERROR(fmt, ...)                            \
    do {                                                    \
        fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__); \
        exit(1);                                            \
    } while (0)
#endif

#endif /* _kiss_fft_log_h */
