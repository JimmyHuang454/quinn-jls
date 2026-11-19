// SPDX-License-Identifier: GPL-2.0-only
/*
 * Linux Kernel TCP congestion control module for Copa-Auto and variants.
 *
 * Registers 6 TCP CCAs in a single module:
 *   - copa_auto     : adaptive delta (theta = 0.50), mode switching ON
 *   - copa_auto_nc  : adaptive delta (theta = 0.50), mode switching OFF
 *   - copa_auto_t90 : adaptive delta (theta = 0.90), mode switching ON
 *   - copa_auto_t98 : adaptive delta (theta = 0.98), mode switching ON
 *   - copa_auto_t   : adaptive delta (theta = theta_milli/1000), mode switching ON
 *   - copa_orig     : original fixed-delta Copa (delta = 0.50), mode switching ON
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/version.h>
#include <linux/ktime.h>
#include <net/tcp.h>

#include "copa_auto_core.h"

static unsigned int theta_milli = 980;
module_param(theta_milli, uint, 0644);
MODULE_PARM_DESC(theta_milli,
		 "Target throughput weight theta*1000 for copa_auto_t (10..995)");

struct tcp_copa_priv {
	struct copa_auto_state *ca;
};

static struct copa_probe_board g_copa_board;

static inline u64 tcp_copa_now_us(const struct tcp_sock *tp)
{
	if (tp && tp->tcp_mstamp)
		return tp->tcp_mstamp;
	return ktime_to_us(ktime_get());
}

static inline u32 tcp_copa_get_cwnd(const struct tcp_sock *tp)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 18, 0)
	return tcp_snd_cwnd(tp);
#else
	return tp->snd_cwnd;
#endif
}

static inline void tcp_copa_set_cwnd(struct tcp_sock *tp, u32 val)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 18, 0)
	tcp_snd_cwnd_set(tp, val);
#else
	tp->snd_cwnd = val;
#endif
}

static void tcp_copa_init_common(struct sock *sk, struct copa_auto_cfg cfg)
{
	struct tcp_copa_priv *priv = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);
	struct copa_auto_state *ca;
	u32 init_cwnd;

	BUILD_BUG_ON(sizeof(struct tcp_copa_priv) > ICSK_CA_PRIV_SIZE);

	ca = kzalloc(sizeof(*ca), GFP_ATOMIC);
	priv->ca = ca;
	if (!ca)
		return;

	init_cwnd = max_t(u32, tcp_copa_get_cwnd(tp), 10U);
	copa_state_init(ca, cfg, &g_copa_board, tcp_copa_now_us(tp), init_cwnd);

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 13, 0)
	cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);
#endif
}

static void tcp_copa_init_auto(struct sock *sk)
{
	struct copa_auto_cfg cfg = {
		.adaptive = true,
		.mode_switching = true,
		.theta_milli = 500,
		.init_delta_q20 = COPA_DELTA_INIT_Q20,
	};
	tcp_copa_init_common(sk, cfg);
}

static void tcp_copa_init_auto_nc(struct sock *sk)
{
	struct copa_auto_cfg cfg = {
		.adaptive = true,
		.mode_switching = false,
		.theta_milli = 500,
		.init_delta_q20 = COPA_DELTA_INIT_Q20,
	};
	tcp_copa_init_common(sk, cfg);
}

static void tcp_copa_init_auto_t90(struct sock *sk)
{
	struct copa_auto_cfg cfg = {
		.adaptive = true,
		.mode_switching = true,
		.theta_milli = 900,
		.init_delta_q20 = COPA_DELTA_INIT_Q20,
	};
	tcp_copa_init_common(sk, cfg);
}

static void tcp_copa_init_auto_t98(struct sock *sk)
{
	struct copa_auto_cfg cfg = {
		.adaptive = true,
		.mode_switching = true,
		.theta_milli = 980,
		.init_delta_q20 = COPA_DELTA_INIT_Q20,
	};
	tcp_copa_init_common(sk, cfg);
}

static void tcp_copa_init_auto_t(struct sock *sk)
{
	u32 th = clamp_t(u32, READ_ONCE(theta_milli), 10U, 995U);
	struct copa_auto_cfg cfg = {
		.adaptive = true,
		.mode_switching = true,
		.theta_milli = th,
		.init_delta_q20 = COPA_DELTA_INIT_Q20,
	};
	tcp_copa_init_common(sk, cfg);
}

static void tcp_copa_init_orig(struct sock *sk)
{
	struct copa_auto_cfg cfg = {
		.adaptive = false,
		.mode_switching = true,
		.theta_milli = 500,
		.init_delta_q20 = COPA_DELTA_MAX_Q20, /* 0.50 */
	};
	tcp_copa_init_common(sk, cfg);
}

static void tcp_copa_release(struct sock *sk)
{
	struct tcp_copa_priv *priv = inet_csk_ca(sk);

	if (priv->ca) {
		kfree(priv->ca);
		priv->ca = NULL;
	}
}

static u32 tcp_copa_ssthresh(struct sock *sk)
{
	return max_t(u32, tcp_copa_get_cwnd(tcp_sk(sk)), 2U);
}

static u32 tcp_copa_undo_cwnd(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);

	return max_t(u32, tcp_copa_get_cwnd(tp), tp->prior_cwnd);
}

static void tcp_copa_set_state(struct sock *sk, u8 new_state)
{
	struct tcp_copa_priv *priv = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);

	if (!priv->ca)
		return;

	if (new_state == TCP_CA_Loss || new_state == TCP_CA_Recovery) {
		u64 now_us = tcp_copa_now_us(tp);

		copa_on_loss(priv->ca, now_us, new_state == TCP_CA_Loss);
		if (new_state == TCP_CA_Loss) {
			u32 cwnd = max_t(u32, (u32)(priv->ca->cwnd_q10 >> 10),
					 COPA_MIN_CWND_PKTS);
			tcp_copa_set_cwnd(tp, min_t(u32, cwnd, tp->snd_cwnd_clamp));
		}
	}
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 10, 0)
static void tcp_copa_cong_control(struct sock *sk, u32 ack, int flag,
				  const struct rate_sample *rs)
#else
static void tcp_copa_cong_control(struct sock *sk,
				  const struct rate_sample *rs)
#endif
{
	struct tcp_copa_priv *priv = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);
	struct copa_auto_state *ca = priv->ca;
	u64 now_us, srtt_us, rtt_sample_us, rate_bps = 0, pacing_rate;
	u32 mss, acked_pkts, new_cwnd;
	bool app_limited;

	if (!ca)
		return;

	now_us = tcp_copa_now_us(tp);
	srtt_us = tp->srtt_us >> 3;
	rtt_sample_us = (rs && rs->rtt_us > 0) ? (u64)rs->rtt_us : srtt_us;
	if (rtt_sample_us == 0)
		return;
	if (srtt_us == 0)
		srtt_us = rtt_sample_us;

	mss = max_t(u32, tp->mss_cache, 536U);
	acked_pkts = (rs && rs->acked_sacked > 0) ? (u32)rs->acked_sacked : 1U;
	if (rs && rs->delivered > 0 && rs->interval_us > 0) {
		rate_bps = div64_u64((u64)rs->delivered * (u64)mss * 1000000ULL,
				     (u64)rs->interval_us);
	}
	app_limited = (rs && rs->is_app_limited) || (tp->app_limited != 0);

	new_cwnd = copa_on_ack(ca, now_us, rtt_sample_us, srtt_us,
			       acked_pkts, mss, rate_bps, app_limited);
	new_cwnd = clamp_t(u32, new_cwnd, COPA_MIN_CWND_PKTS, tp->snd_cwnd_clamp);
	tcp_copa_set_cwnd(tp, new_cwnd);

	pacing_rate = copa_pacing_rate_bps(ca, mss, now_us);
	if (pacing_rate > 0) {
		unsigned long max_rate = READ_ONCE(sk->sk_max_pacing_rate);
		if (pacing_rate > max_rate)
			pacing_rate = max_rate;
		WRITE_ONCE(sk->sk_pacing_rate, (unsigned long)pacing_rate);
	}
}

#define DEFINE_COPA_OPS(sym, cc_name, init_fn)			\
static struct tcp_congestion_ops sym __read_mostly = {		\
	.flags		= TCP_CONG_NON_RESTRICTED,		\
	.name		= cc_name,				\
	.owner		= THIS_MODULE,				\
	.init		= init_fn,				\
	.release	= tcp_copa_release,			\
	.cong_control	= tcp_copa_cong_control,		\
	.ssthresh	= tcp_copa_ssthresh,			\
	.undo_cwnd	= tcp_copa_undo_cwnd,			\
	.set_state	= tcp_copa_set_state,			\
}

DEFINE_COPA_OPS(tcp_copa_auto_ops,     "copa_auto",     tcp_copa_init_auto);
DEFINE_COPA_OPS(tcp_copa_auto_nc_ops,  "copa_auto_nc",  tcp_copa_init_auto_nc);
DEFINE_COPA_OPS(tcp_copa_auto_t90_ops, "copa_auto_t90", tcp_copa_init_auto_t90);
DEFINE_COPA_OPS(tcp_copa_auto_t98_ops, "copa_auto_t98", tcp_copa_init_auto_t98);
DEFINE_COPA_OPS(tcp_copa_auto_t_ops,   "copa_auto_t",   tcp_copa_init_auto_t);
DEFINE_COPA_OPS(tcp_copa_orig_ops,     "copa_orig",     tcp_copa_init_orig);

static struct tcp_congestion_ops * const copa_ops_list[] = {
	&tcp_copa_auto_ops,
	&tcp_copa_auto_nc_ops,
	&tcp_copa_auto_t90_ops,
	&tcp_copa_auto_t98_ops,
	&tcp_copa_auto_t_ops,
	&tcp_copa_orig_ops,
};

static int __init tcp_copa_auto_register(void)
{
	int i, ret;

	for (i = 0; i < ARRAY_SIZE(copa_ops_list); i++) {
		ret = tcp_register_congestion_control(copa_ops_list[i]);
		if (ret) {
			while (--i >= 0)
				tcp_unregister_congestion_control(copa_ops_list[i]);
			return ret;
		}
	}
	pr_info("tcp_copa_auto: registered copa_auto, copa_auto_nc, copa_auto_t90, copa_auto_t98, copa_auto_t, copa_orig\n");
	return 0;
}

static void __exit tcp_copa_auto_unregister(void)
{
	int i;

	for (i = ARRAY_SIZE(copa_ops_list) - 1; i >= 0; i--)
		tcp_unregister_congestion_control(copa_ops_list[i]);
	pr_info("tcp_copa_auto: unregistered\n");
}

module_init(tcp_copa_auto_register);
module_exit(tcp_copa_auto_unregister);

MODULE_AUTHOR("quinn-jls / copa-auto");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("TCP Copa-Auto (adaptive delta + DrainController + RCT) congestion control");
