/* Userspace unit tests for copa_auto_math.h and copa_auto_core.h */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "copa_auto_core.h"

static void test_math_ln_exp(void)
{
	double ratios[] = { 0.05, 0.2, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 16.0, 50.0 };
	size_t i;

	for (i = 0; i < sizeof(ratios) / sizeof(ratios[0]); i++) {
		u64 b = 1000000ULL;
		u64 a = (u64)llround(ratios[i] * (double)b);
		s64 got_q16 = copa_ln_ratio_q16(a, b);
		double got = (double)got_q16 / 65536.0;
		double expected = log(ratios[i]);
		double err = fabs(got - expected);
		if (err > 0.005) {
			fprintf(stderr, "FAIL ln(%.3f): got=%.5f exp=%.5f err=%.5f\n",
				ratios[i], got, expected, err);
			exit(1);
		}
	}

	double xs[] = { -3.5, -2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 3.5 };
	for (i = 0; i < sizeof(xs) / sizeof(xs[0]); i++) {
		s64 x_q16 = (s64)llround(xs[i] * 65536.0);
		u64 got_q16 = copa_exp_q16(x_q16);
		double got = (double)got_q16 / 65536.0;
		double expected = exp(xs[i]);
		double rel = fabs(got - expected) / expected;
		if (rel > 0.01) {
			fprintf(stderr, "FAIL exp(%.3f): got=%.5f exp=%.5f rel=%.5f\n",
				xs[i], got, expected, rel);
			exit(1);
		}
	}
	printf("[PASS] test_math_ln_exp\n");
}

static void test_target_dq(void)
{
	u32 thetas[] = { 200, 500, 900, 980 };
	u64 ks[] = { 50, 500, 5000, 25000 };
	u64 js[] = { 0, 2000, 10000 };
	u64 base_rtt = 50000;
	size_t ti, ki, ji;

	for (ti = 0; ti < 4; ti++) {
		double th = (double)thetas[ti] / 1000.0;
		for (ki = 0; ki < 4; ki++) {
			double k = (double)ks[ki] / 1e6;
			for (ji = 0; ji < 3; ji++) {
				double j = (double)js[ji] / 1e6;
				double rtt = (double)base_rtt / 1e6;
				double a = 1.0 - 2.0 * th;
				double exp_dq_s = j + (2.0 * th * rtt) / (a + sqrt(a * a + 4.0 * th * (1.0 - th) * rtt / k));
				u64 got_us = copa_target_dq_us(js[ji], ks[ki], base_rtt, thetas[ti]);
				double got_s = (double)got_us / 1e6;
				double rel = fabs(got_s - exp_dq_s) / fmax(exp_dq_s, 1e-6);
				if (rel > 0.005) {
					fprintf(stderr, "FAIL target_dq th=%.2f k=%llu j=%llu: got=%.6f exp=%.6f\n",
						th, (unsigned long long)ks[ki], (unsigned long long)js[ji],
						got_s, exp_dq_s);
					exit(1);
				}
			}
		}
	}
	printf("[PASS] test_target_dq\n");
}

static void test_slow_start_and_steady(void)
{
	struct copa_auto_state ca;
	struct copa_probe_board board = { 0 };
	struct copa_auto_cfg cfg = {
		.adaptive = true,
		.mode_switching = true,
		.theta_milli = 500,
		.init_delta_q20 = COPA_DELTA_INIT_Q20,
	};
	u64 now = 1000000ULL; /* 1 s */
	u32 cwnd = 10;
	int i;

	copa_state_init(&ca, cfg, &board, now, cwnd);

	/* 50 ACKs with zero queueing: slow start should grow cwnd monotonically */
	for (i = 0; i < 50; i++) {
		now += 1000ULL;
		cwnd = copa_on_ack(&ca, now, 30000ULL, 30000ULL, 1, 1448, 12500000ULL, false);
	}
	assert(ca.slow_start == true);
	assert(cwnd >= 50);

	/* Sustained 25 ms queueing (> jguard = 15 ms) should exit slow start */
	for (i = 0; i < 40; i++) {
		now += 2000ULL;
		cwnd = copa_on_ack(&ca, now, 55000ULL, 40000ULL, 1, 1448, 12500000ULL, false);
	}
	assert(ca.slow_start == false);
	assert(cwnd >= COPA_MIN_CWND_PKTS);

	/* Verify pacing rate is non-zero and reasonable */
	u64 pace = copa_pacing_rate_bps(&ca, 1448, now);
	assert(pace > 0);

	printf("[PASS] test_slow_start_and_steady (cwnd=%u pace=%llu B/s)\n",
	       cwnd, (unsigned long long)pace);
}

static void test_restart_from_idle_and_k_estimation(void)
{
	struct copa_auto_state ca;
	struct copa_probe_board board = { 0 };
	struct copa_auto_cfg cfg = {
		.adaptive = true,
		.mode_switching = true,
		.theta_milli = 980,
		.init_delta_q20 = COPA_DELTA_INIT_Q20,
	};
	u64 now = 1000000ULL;
	u32 cwnd;
	int i;

	copa_state_init(&ca, cfg, &board, now, 10);

	/* Warm up: 100 Mbps, 30 ms RTT (~258 pkts BDP) */
	for (i = 0; i < 200; i++) {
		u64 rtt = 30000ULL + ((i % 10 == 0) ? 0ULL : 4000ULL);
		now += 5000ULL;
		cwnd = copa_on_ack(&ca, now, rtt, 32000ULL, 2, 1448, 12500000ULL, false);
	}
	(void)cwnd;
	/* Force exit slow start */
	for (i = 0; i < 30; i++) {
		now += 5000ULL;
		copa_on_ack(&ca, now, 60000ULL, 45000ULL, 2, 1448, 12500000ULL, false);
	}
	assert(!ca.slow_start);

	/* Run several 8-round segments so AutoDelta computes ratio_us and delta_run_q20 */
	for (i = 0; i < 600; i++) {
		u64 rtt = 30000ULL + ((i % 6 == 0) ? 200ULL : 6000ULL);
		now += 15000ULL;
		copa_on_ack(&ca, now, rtt, 34000ULL, 2, 1448, 12500000ULL, false);
	}
	assert(ca.ratio_us > 0);
	assert(ca.dq_target_us > 0);
	assert(ca.delta_run_q20 >= COPA_DELTA_MIN_Q20);
	assert(ca.delta_run_q20 <= COPA_DELTA_MAX_Q20);

	/* App-limited then 500 ms idle -> RestartFromIdle should re-enter slow start */
	now += 10000ULL;
	copa_on_ack(&ca, now, 30000ULL, 30000ULL, 1, 1448, 12500000ULL, true);
	now += 500000ULL;
	cwnd = copa_on_ack(&ca, now, 30000ULL, 30000ULL, 1, 1448, 12500000ULL, false);
	assert(ca.slow_start == true);
	assert(cwnd >= 10);

	printf("[PASS] test_restart_from_idle_and_k_estimation (ratio=%llu us, dq_target=%llu us, delta=%.4f)\n",
	       (unsigned long long)ca.ratio_us,
	       (unsigned long long)ca.dq_target_us,
	       (double)ca.delta_run_q20 / 1048576.0);
}

static void test_copa_orig_fixed_delta(void)
{
	struct copa_auto_state ca;
	struct copa_probe_board board = { 0 };
	struct copa_auto_cfg cfg = {
		.adaptive = false,
		.mode_switching = true,
		.theta_milli = 500,
		.init_delta_q20 = COPA_DELTA_MAX_Q20,
	};
	u64 now = 1000000ULL;
	int i;

	copa_state_init(&ca, cfg, &board, now, 10);
	for (i = 0; i < 100; i++) {
		now += 10000ULL;
		copa_on_ack(&ca, now, 30000ULL + (i % 5) * 1000ULL, 32000ULL, 1, 1448, 1250000ULL, false);
	}
	assert(ca.delta_q20 == COPA_DELTA_MAX_Q20);
	assert(ca.delta_run_q20 == 0);
	printf("[PASS] test_copa_orig_fixed_delta\n");
}

int main(void)
{
	test_math_ln_exp();
	test_target_dq();
	test_slow_start_and_steady();
	test_restart_from_idle_and_k_estimation();
	test_copa_orig_fixed_delta();
	printf("All tcp_copa_auto core tests passed!\n");
	return 0;
}
