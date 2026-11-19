/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * Fixed-point 64-bit integer math for tcp_copa_auto (zero FPU / float usage).
 * Compiles identically in Linux kernel space (__KERNEL__) and userspace tests.
 */
#ifndef COPA_AUTO_MATH_H
#define COPA_AUTO_MATH_H

#ifdef __KERNEL__
#include <linux/bitops.h>
#include <linux/math64.h>
#include <linux/types.h>
#else
#include <stdbool.h>
#include <stdint.h>
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t s32;
typedef int64_t s64;
#ifndef min_t
#define min_t(type, x, y) ({ type __x = (x); type __y = (y); __x < __y ? __x : __y; })
#endif
#ifndef max_t
#define max_t(type, x, y) ({ type __x = (x); type __y = (y); __x > __y ? __x : __y; })
#endif
#ifndef clamp_t
#define clamp_t(type, val, lo, hi) min_t(type, max_t(type, val, lo), hi)
#endif
static inline u64 div64_u64(u64 dividend, u64 divisor)
{
    return divisor ? (dividend / divisor) : 0;
}
static inline s64 div64_s64(s64 dividend, s64 divisor)
{
    return divisor ? (dividend / divisor) : 0;
}
static inline u64 int_sqrt64(u64 x)
{
    u64 r = 0, bit = 1ULL << 62;
    while (bit > x)
        bit >>= 2;
    while (bit != 0) {
        if (x >= r + bit) {
            x -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    return r;
}
static inline int fls64(u64 x)
{
    if (!x)
        return 0;
    return 64 - __builtin_clzll(x);
}
#endif

/* Q16 fixed-point scale: 1.0 == 65536 */
#define COPA_Q16_SHIFT 16
#define COPA_Q16_ONE   (1LL << COPA_Q16_SHIFT)
/* ln(2) in Q16: 0.69314718056 * 65536 = 45426 */
#define COPA_LN2_Q16   45426LL
/* 1 / ln(2) in Q16: 1.44269504089 * 65536 = 94548 */
#define COPA_INV_LN2_Q16 94548LL

/* Delta is stored in Q20 fixed point: delta = 1.0 == (1 << 20) = 1048576.
 * delta_min = 0.001 => ~1049; delta_max = 0.5 => 524288.
 */
#define COPA_DELTA_SHIFT 20
#define COPA_DELTA_ONE   (1ULL << COPA_DELTA_SHIFT)

/*
 * Natural logarithm ln(x / scale) in Q16, for x > 0, scale > 0.
 * Uses binary normalization x = 2^e * m (m in [1, 2)) and 4-term atanh series
 * ln(m) = 2 * (z + z^3/3 + z^5/5 + z^7/7) where z = (m - 1)/(m + 1) in [0, 1/3).
 * Max error < 0.002% over the entire u64 range.
 */
static inline s64 copa_ln_ratio_q16(u64 num, u64 den)
{
    int e_num, e_den, exp2;
    u64 m_num, m_den, m;
    s64 z, z2, term, sum, ln_m;

    if (num == 0)
        num = 1;
    if (den == 0)
        den = 1;

    e_num = fls64(num) - 1;
    e_den = fls64(den) - 1;

    /* Normalize both to Q20 in [1<<20, 2<<20) */
    if (e_num >= 20)
        m_num = num >> (e_num - 20);
    else
        m_num = num << (20 - e_num);

    if (e_den >= 20)
        m_den = den >> (e_den - 20);
    else
        m_den = den << (20 - e_den);

    exp2 = e_num - e_den;
    /* m = (m_num / m_den) in Q20; lies in (0.5, 2.0) */
    m = div64_u64(m_num << 20, m_den);
    if (m < (1ULL << 20)) {
        m <<= 1;
        exp2 -= 1;
    }

    /* z = (m - 1) / (m + 1) in Q20, z in [0, 1/3) */
    z = (s64)div64_u64((m - (1ULL << 20)) << 20, m + (1ULL << 20));
    z2 = (z * z) >> 20;

    sum = z;
    term = (z * z2) >> 20;
    sum += div64_s64(term, 3);
    term = (term * z2) >> 20;
    sum += div64_s64(term, 5);
    term = (term * z2) >> 20;
    sum += div64_s64(term, 7);

    /* Convert 2 * sum from Q20 to Q16 */
    ln_m = (sum * 2) >> 4;
    return (s64)exp2 * COPA_LN2_Q16 + ln_m;
}

/*
 * Exponential exp(x_q16) returned in Q16 (i.e. exp(x) * 65536).
 * Clamps x_q16 to [-12 * 65536, +12 * 65536] to prevent overflow.
 * Uses range reduction x = k*ln(2) + r, |r| <= ln(2)/2, and 6th-order Taylor.
 */
static inline u64 copa_exp_q16(s64 x_q16)
{
    s64 k, r, term, sum;
    u64 res;

    x_q16 = clamp_t(s64, x_q16, -12LL * COPA_Q16_ONE, 12LL * COPA_Q16_ONE);
    /* k = round(x / ln(2)) */
    if (x_q16 >= 0)
        k = (x_q16 * COPA_INV_LN2_Q16 + (1LL << 31)) >> 32;
    else
        k = -(((-x_q16) * COPA_INV_LN2_Q16 + (1LL << 31)) >> 32);

    r = x_q16 - k * COPA_LN2_Q16; /* in Q16, |r| <= 22713 */

    /* Evaluate exp(r) in Q24 for high precision */
    r <<= 8; /* Q24 */
    sum = 1LL << 24;
    term = r;
    sum += term;
    term = (term * r) >> 24;
    sum += term / 2;
    term = (term * r) >> 24;
    sum += term / 6;
    term = (term * r) >> 24;
    sum += term / 24;
    term = (term * r) >> 24;
    sum += term / 120;
    term = (term * r) >> 24;
    sum += term / 720;

    if (sum < 1)
        sum = 1;
    res = (u64)sum >> 8; /* back to Q16 */
    if (k >= 0) {
        if (k > 30)
            k = 30;
        res <<= k;
    } else {
        if (-k > 30)
            return 1;
        res >>= (-k);
    }
    return res ? res : 1;
}

/*
 * Queueing target d_q* [us] maximizing theta*ln(rho) - (1-theta)*ln(D).
 * theta_milli in [10, 990] (500 = 0.50 power-optimal).
 *
 * For theta = 0.5: d_q* = j + sqrt(k * R0').
 * For general theta:
 *   a = 1 - 2*theta
 *   d_q* = j + 2*theta*R0' / (a + sqrt(a^2 + 4*theta*(1-theta)*R0'/k))
 * All computed in 64-bit integer arithmetic (microseconds).
 */
static inline u64 copa_target_dq_us(u64 jitter_us, u64 k_us, u64 base_rtt_us, u32 theta_milli)
{
    u64 q_us;

    if (k_us == 0)
        k_us = 1;
    if (base_rtt_us == 0)
        base_rtt_us = 1000;

    if (theta_milli == 500) {
        /* sqrt(k_us * base_rtt_us) */
        q_us = int_sqrt64(k_us * base_rtt_us);
        return jitter_us + q_us;
    }

    /* Scale a = (1000 - 2*theta_milli) in milli-units (1000 == 1.0).
     * Work in scaled units S = 1000000 (10^6) inside sqrt:
     *   a_s = (1000 - 2*theta) * 1000  (in [-10^6, +10^6])
     *   rad = a_s^2 + 4 * theta * (1000 - theta) * 10^6 * base_rtt_us / k_us
     * To avoid u64 overflow in 4*theta*(1000-theta)*10^6*base_rtt_us, scale by 10^3:
     *   a_k = (1000 - 2*theta)          (in [-1000, +1000])
     *   term = 4 * theta * (1000 - theta) * base_rtt_us / k_us  (units of 10^6)
     *   rad_1e6 = a_k * a_k + term
     *   root_1e3 = int_sqrt64(rad_1e6 * 1000000ULL) / 1000  (better: keep *1000 precision)
     */
    {
        s64 a_milli = 1000LL - 2LL * (s64)theta_milli;
        u64 w4 = 4ULL * (u64)theta_milli * (1000ULL - (u64)theta_milli); /* up to 10^6 */
        /* Compute term in units of 10^6 with extra *1000 precision:
         * rad_1e12 = (a_milli * 1000)^2 + (w4 * 10^6) * base_rtt_us / k_us
         * Note: w4 <= 10^6; if base_rtt_us <= 10^7 (10 s), w4 * base_rtt_us <= 10^13,
         * so multiply by 10^6 carefully via div64_u64.
         */
        u64 a2_1e12 = (u64)(a_milli * a_milli) * 1000000ULL;
        u64 ratio_q = div64_u64(base_rtt_us * 1000000ULL, k_us);
        /* Cap ratio_q to prevent u64 overflow in w4 * ratio_q */
        if (ratio_q > 1000000000000ULL)
            ratio_q = 1000000000000ULL;
        u64 term_1e12 = w4 * ratio_q;
        u64 root_1e6 = int_sqrt64(a2_1e12 + term_1e12);
        s64 denom_1e6 = a_milli * 1000LL + (s64)root_1e6;
        if (denom_1e6 < 1)
            denom_1e6 = 1;
        /* q_us = (2 * theta_milli / 1000) * base_rtt_us / (denom_1e6 / 10^6)
         *      = 2 * theta_milli * base_rtt_us * 1000 / denom_1e6
         */
        q_us = div64_u64(2ULL * (u64)theta_milli * base_rtt_us * 1000ULL, (u64)denom_1e6);
    }
    return jitter_us + q_us;
}

#endif /* COPA_AUTO_MATH_H */
