//! Rate-based queue drain and gentle jitter-calibration state machine for `copa-auto`
//!
//! Instead of cutting `cwnd` abruptly to `0.5 * BDP` (which freezes packet transmission while
//! `in_flight > cwnd`), [`DrainController`] paces slightly below the estimated bottleneck rate
//! (`0.78..0.85 * bw_est`) while keeping `cwnd` soft-capped so packets continue to flow smoothly.
//! Both `RefreshRttMin` drains and `MeasureJitter` probes share this single mechanism, differing
//! only in whether they exit immediately once the queue empties or stay briefly to sample `j`.

use crate::{Duration, Instant};

/// Gain applied to `bw_est` during an `RTT_min`-refresh drain
const DRAIN_PACING_GAIN: f64 = 0.85;
/// Deeper gain used when a thick queue (`dq > 20 ms`) needs faster draining
const DEEP_DRAIN_PACING_GAIN: f64 = 0.75;
/// Gain applied to `bw_est` during a `MeasureJitter` calibration
const JITTER_PROBE_PACING_GAIN: f64 = 0.78;
/// Soft `cwnd` target fraction of estimated BDP during draining
const SOFT_CWND_BDP_FRACTION: f64 = 0.90;
/// Maximum slot-multiple backoff when consecutive drains do not lower `RTT_min`
const MAX_BACKOFF_SLOTS: u64 = 8;

/// Purpose of a [`DrainController`] episode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DrainKind {
    /// Short drain to clear bottleneck queue and refresh `RTT_min`; exits as soon as idle/plateau
    RefreshRttMin,
    /// Low-load calibration to measure the jitter floor `j` (and refresh `RTT_min` in one pass)
    MeasureJitter,
}

/// Two-bucket rotating max filter over `2..4 * srtt` for bottleneck delivery rate
#[derive(Debug, Clone, Default)]
struct BandwidthEstimator {
    window_start: Option<Instant>,
    last_ack: Option<Instant>,
    delivered: u64,
    bucket_start: Option<Instant>,
    bucket_updated: [Option<Instant>; 2],
    buckets: [f64; 2],
    active: usize,
}

impl BandwidthEstimator {
    fn reset_sample(&mut self) {
        self.window_start = None;
        self.delivered = 0;
    }

    fn on_ack(
        &mut self,
        now: Instant,
        bytes: u64,
        srtt: Duration,
        window: f64,
        app_limited: bool,
        pause_updates: bool,
    ) {
        if pause_updates {
            self.reset_sample();
            self.last_ack = Some(now);
            return;
        }
        let min_win = srtt.max(Duration::from_millis(5));
        // Ignore silence gaps so idle pauses never dilute rate = delivered / elapsed
        if self
            .last_ack
            .is_some_and(|prev| now.saturating_duration_since(prev) > 2 * min_win)
        {
            self.window_start = Some(now);
            self.delivered = 0;
        }
        self.last_ack = Some(now);

        let start = *self.window_start.get_or_insert(now);
        self.delivered = self.delivered.saturating_add(bytes);
        let elapsed = now.saturating_duration_since(start);
        if elapsed < min_win || self.delivered == 0 {
            return;
        }
        // Require a meaningful fraction of cwnd unless 4*srtt passed (e.g. sharp capacity drop)
        if (self.delivered as f64) < 0.5 * window && elapsed < 4 * min_win {
            return;
        }

        let rate = self.delivered as f64 / elapsed.as_secs_f64();
        let b_start = *self.bucket_start.get_or_insert(now);
        if !app_limited && now.saturating_duration_since(b_start) >= 2 * min_win {
            self.active ^= 1;
            self.buckets[self.active] = 0.0;
            self.bucket_updated[self.active] = None;
            self.bucket_start = Some(now);
        }

        let current_est = self.estimate(now, srtt).unwrap_or(0.0);
        if (!app_limited || (current_est > 0.0 && rate > current_est))
            && rate >= self.buckets[self.active]
        {
            self.buckets[self.active] = rate;
            self.bucket_updated[self.active] = Some(now);
        }
        self.window_start = Some(now);
        self.delivered = 0;
    }

    fn estimate(&self, now: Instant, srtt: Duration) -> Option<f64> {
        let max_age = (8 * srtt).max(Duration::from_secs(1));
        let mut best = 0.0_f64;
        for i in 0..2 {
            if self.bucket_updated[i]
                .is_some_and(|t| now.saturating_duration_since(t) <= max_age)
            {
                best = best.max(self.buckets[i]);
            }
        }
        (best > 0.0).then_some(best)
    }
}

#[derive(Debug, Clone, Copy)]
struct ActiveDrain {
    kind: DrainKind,
    slot: Option<u64>,
    started_at: Instant,
    deadline: Instant,
    idle_reached_at: Option<Instant>,
    measure_until: Option<Instant>,
    saved_window: f64,
    saved_rate: f64,
    soft_target_window: f64,
    drain_rate: f64,
    rtt_min_at_start: Duration,
    trusted_at_start: Duration,
    trusted_verified: bool,
    hold_full_deadline: bool,
    drain_min_rtt: Duration,
    round_start: Instant,
    prev_drain_min_rtt: Option<Duration>,
    flat_rounds: u32,
    congested: bool,
    probe_dq_sum: f64,
    probe_acks: u64,
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    Normal,
    Draining(ActiveDrain),
    Recovering { until: Instant, rate_cap: f64 },
}

/// Outcome returned when a drain/calibration episode finishes
#[derive(Debug, Clone, Copy)]
pub(super) struct DrainFinished {
    pub(super) kind: DrainKind,
    pub(super) slot: Option<u64>,
    pub(super) saved_window: f64,
    pub(super) rtt_min_at_start: Duration,
    pub(super) drain_min_rtt: Duration,
    pub(super) congested: bool,
    pub(super) probe_dq: Option<f64>,
    pub(super) bdp_est: Option<f64>,
    pub(super) plateau_converged: bool,
}

/// Manages smooth rate-based queue drains, jitter calibration holds, and recovery pacing
#[derive(Debug, Clone)]
pub(super) struct DrainController {
    bw: BandwidthEstimator,
    phase: Phase,
    /// Conservative baseline RTT updated at the end of drains/probes and lowered by smaller RTTs
    trusted_min: Option<Duration>,
    /// Last time the connection observed near-idle conditions via `app_limited`
    last_app_idle: Option<Instant>,
    /// Last aligned slot evaluated for `RefreshRttMin`
    evaluated_slot: Option<u64>,
    /// Exponentially backed-off interval (in 20 s slots) when drains find no RTT_min drift
    backoff_slots: u64,
    /// Earliest aligned slot eligible for the next `RefreshRttMin` drain
    next_drain_slot: u64,
}

impl DrainController {
    pub(super) fn new() -> Self {
        Self {
            bw: BandwidthEstimator::default(),
            phase: Phase::Normal,
            trusted_min: None,
            last_app_idle: None,
            evaluated_slot: None,
            backoff_slots: 1,
            next_drain_slot: 0,
        }
    }

    #[cfg(test)]
    pub(super) fn trusted_min(&self) -> Option<Duration> {
        self.trusted_min
    }

    pub(super) fn set_trusted_min(&mut self, value: Duration) {
        self.trusted_min = Some(value);
    }

    pub(super) fn observe_sample(&mut self, now: Instant, sample: Duration, dq: Duration, app_limited: bool) {
        if let Some(trusted) = &mut self.trusted_min {
            *trusted = (*trusted).min(sample);
            if app_limited && dq <= Duration::from_millis(2).max(trusted.mul_f64(0.05)) {
                self.last_app_idle = Some(now);
            }
        }
    }

    pub(super) fn on_ack_bw(
        &mut self,
        now: Instant,
        bytes: u64,
        srtt: Duration,
        window: f64,
        app_limited: bool,
    ) {
        let pause = !matches!(self.phase, Phase::Normal);
        self.bw.on_ack(now, bytes, srtt, window, app_limited, pause);
    }

    pub(super) fn is_draining(&self) -> bool {
        matches!(self.phase, Phase::Draining(_))
    }

    pub(super) fn is_recovering(&self) -> bool {
        matches!(self.phase, Phase::Recovering { .. })
    }

    pub(super) fn is_normal(&self) -> bool {
        matches!(self.phase, Phase::Normal)
    }

    #[cfg(test)]
    pub(super) fn active_kind(&self) -> Option<DrainKind> {
        match self.phase {
            Phase::Draining(d) => Some(d.kind),
            _ => None,
        }
    }

    pub(super) fn drain_pacing_rate(&self) -> Option<f64> {
        match self.phase {
            Phase::Draining(d) => Some(d.drain_rate),
            _ => None,
        }
    }

    pub(super) fn recovery_pace_cap(&self, now: Instant) -> Option<f64> {
        match self.phase {
            Phase::Recovering { until, rate_cap } if now < until => Some(rate_cap),
            _ => None,
        }
    }

    /// Expire `Recovering` (or a timed-out `Draining`) without requiring another ACK
    pub(super) fn poll_expiry(
        &mut self,
        now: Instant,
        srtt: Duration,
        rtt_min: Option<Duration>,
    ) -> Option<DrainFinished> {
        match self.phase {
            Phase::Recovering { until, .. } if now >= until => {
                self.phase = Phase::Normal;
                None
            }
            Phase::Draining(d) if now >= d.deadline => {
                let rtt_min = rtt_min.unwrap_or(d.rtt_min_at_start);
                Some(self.transition_to_recovering(now, srtt, rtt_min, d))
            }
            _ => None,
        }
    }

    /// Abort any active drain/recovery (e.g. when entering competitive mode)
    pub(super) fn abort(&mut self) {
        self.phase = Phase::Normal;
    }

    pub(super) fn mark_congestion(&mut self, persistent: bool, mode_switching: bool) {
        if let Phase::Draining(d) = &mut self.phase {
            // In non-competitive mode (`copa-auto-nc`), non-persistent random loss is ignored by
            // Copa and must not cancel fast recovery.
            if persistent || mode_switching {
                d.congested = true;
            }
        }
    }

    pub(super) fn evaluated_slot(&self) -> Option<u64> {
        self.evaluated_slot
    }

    pub(super) fn mark_slot_evaluated(&mut self, slot: u64) {
        self.evaluated_slot = Some(slot);
    }

    /// Whether a `RefreshRttMin` drain is warranted for `slot`
    pub(super) fn should_drain_slot(
        &mut self,
        now: Instant,
        slot: u64,
        rtt_min: Duration,
        skip_window: Duration,
        board_requested: bool,
    ) -> bool {
        self.evaluated_slot = Some(slot);
        if board_requested {
            return true;
        }
        if slot < self.next_drain_slot {
            return false;
        }
        if self
            .last_app_idle
            .is_some_and(|t| now.saturating_duration_since(t) < skip_window)
        {
            return false;
        }
        let Some(trusted) = self.trusted_min else {
            return true;
        };
        let drift_margin = Duration::from_millis(1).max(trusted.mul_f64(0.02));
        rtt_min > trusted.saturating_add(drift_margin)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn start(
        &mut self,
        kind: DrainKind,
        slot: Option<u64>,
        now: Instant,
        srtt: Duration,
        rtt_min: Duration,
        standing: Duration,
        dq: Duration,
        window: f64,
        min_window: f64,
        pacing_rate: f64,
        max_duration: Duration,
        hold_full_deadline: bool,
    ) {
        let trusted_verified = self.trusted_min.is_some();
        let trusted = self.trusted_min.unwrap_or(rtt_min).min(rtt_min);
        let standing_s = standing.max(trusted).as_secs_f64().max(1e-9);
        let cwnd_rate = window / standing_s;
        let bw_est = self.bw.estimate(now, srtt).unwrap_or(cwnd_rate);
        // Bound upward against a stale pre-drop bandwidth estimate
        let effective_bw = bw_est.min(1.5 * cwnd_rate).max(min_window / standing_s);

        let gain = match kind {
            DrainKind::RefreshRttMin => {
                if dq > Duration::from_millis(20) {
                    DEEP_DRAIN_PACING_GAIN
                } else {
                    DRAIN_PACING_GAIN
                }
            }
            DrainKind::MeasureJitter => JITTER_PROBE_PACING_GAIN,
        };
        let min_rate = min_window / srtt.max(Duration::from_millis(10)).as_secs_f64();
        let drain_rate = (gain * effective_bw).max(min_rate);
        let bdp_est = effective_bw * trusted.as_secs_f64();
        let soft_target_window = (SOFT_CWND_BDP_FRACTION * bdp_est)
            .max(min_window)
            .min(window);

        let deadline_dur = match kind {
            DrainKind::RefreshRttMin => {
                let floor = max_duration.max(Duration::from_millis(150));
                floor.max(4 * srtt).max(2 * dq + srtt)
            }
            DrainKind::MeasureJitter => max_duration.max(4 * srtt),
        };

        self.phase = Phase::Draining(ActiveDrain {
            kind,
            slot,
            started_at: now,
            deadline: now + deadline_dur,
            idle_reached_at: None,
            measure_until: None,
            saved_window: window,
            saved_rate: pacing_rate,
            soft_target_window,
            drain_rate,
            rtt_min_at_start: rtt_min,
            trusted_at_start: trusted,
            trusted_verified,
            hold_full_deadline,
            drain_min_rtt: Duration::MAX,
            round_start: now,
            prev_drain_min_rtt: None,
            flat_rounds: 0,
            congested: false,
            probe_dq_sum: 0.0,
            probe_acks: 0,
        });
    }

    /// Advance the active drain on an ACK; returns `Some(DrainFinished)` when draining ends
    #[allow(clippy::too_many_arguments)]
    pub(super) fn on_ack(
        &mut self,
        now: Instant,
        sample: Duration,
        _bytes: u64,
        rtt_min: Duration,
        standing: Duration,
        dq: Duration,
        srtt: Duration,
        jitter_floor: Option<f64>,
    ) -> Option<DrainFinished> {
        let mut d = match self.phase {
            Phase::Draining(d) => d,
            Phase::Recovering { until, .. } => {
                if now >= until {
                    self.phase = Phase::Normal;
                }
                return None;
            }
            Phase::Normal => return None,
        };

        d.drain_min_rtt = d.drain_min_rtt.min(sample);

        let round_len = srtt.max(Duration::from_millis(10));
        let round_elapsed = now.saturating_duration_since(d.round_start);
        if round_elapsed >= round_len {
            if let Some(prev) = d.prev_drain_min_rtt {
                let flat_tol = Duration::from_millis(1).max(srtt.mul_f64(0.02));
                if prev.saturating_sub(d.drain_min_rtt) <= flat_tol {
                    d.flat_rounds += 1;
                } else {
                    d.flat_rounds = 0;
                }
            }
            d.prev_drain_min_rtt = Some(d.drain_min_rtt);
            d.round_start = now;
        }

        let elapsed = now.saturating_duration_since(d.started_at);
        let min_hold = srtt.max(Duration::from_millis(20));

        if d.idle_reached_at.is_none() && elapsed >= min_hold {
            let j_dur = Duration::from_secs_f64(jitter_floor.unwrap_or(0.0).max(0.0));
            let base_tol = Duration::from_millis(1).max(d.trusted_at_start.mul_f64(0.05));
            let back_to_baseline = d.trusted_verified
                && standing <= d.trusted_at_start.saturating_add(j_dur).saturating_add(base_tol);
            let plateau_converged = elapsed >= 2 * srtt && d.flat_rounds >= 2;
            if back_to_baseline || plateau_converged {
                d.idle_reached_at = Some(now);
                if d.kind == DrainKind::MeasureJitter && !d.hold_full_deadline {
                    // Sample jitter floor over a short clean post-idle window (~5 SRTT, within
                    // the remaining slot budget) rather than stalling for 2 s.
                    let sample_len = (5 * srtt).clamp(
                        Duration::from_millis(100),
                        Duration::from_millis(600),
                    );
                    d.measure_until = Some((now + sample_len).min(d.deadline));
                }
            }
        }

        // For MeasureJitter, also allow sampling in the second half of a long board-aligned slot
        // if another flow holds the slot open and baseline wasn't matched early.
        let total_budget = d.deadline.saturating_duration_since(d.started_at);
        let sampling = d.kind == DrainKind::MeasureJitter
            && (d.idle_reached_at.is_some() || elapsed >= total_budget / 2);
        if sampling {
            d.probe_dq_sum += dq.as_secs_f64();
            d.probe_acks += 1;
        }

        let done = match d.kind {
            DrainKind::RefreshRttMin => d.idle_reached_at.is_some() || now >= d.deadline,
            DrainKind::MeasureJitter => {
                now >= d.deadline || d.measure_until.is_some_and(|until| now >= until)
            }
        };

        if done {
            Some(self.transition_to_recovering(now, srtt, rtt_min, d))
        } else {
            self.phase = Phase::Draining(d);
            None
        }
    }

    /// Smoothly ramped `cwnd` during `Draining`: starts at `saved_window` and glides to
    /// `soft_target_window` over `1 * srtt` while `pacing_rate` already caps sending rate, so
    /// `in_flight` never triggers a zero-send freeze.
    pub(super) fn current_window(&self, now: Instant, srtt: Duration, min_window: f64) -> Option<f64> {
        let Phase::Draining(d) = self.phase else {
            return None;
        };
        let ramp = srtt.max(Duration::from_millis(10)).as_secs_f64();
        let frac = (now.saturating_duration_since(d.started_at).as_secs_f64() / ramp)
            .clamp(0.0, 1.0);
        let w = d.saved_window * (1.0 - frac) + d.soft_target_window * frac;
        Some(w.max(min_window))
    }

    pub(super) fn estimated_bdp(&self, now: Instant, srtt: Duration, base_rtt: Duration) -> Option<f64> {
        self.bw.estimate(now, srtt).map(|bw| bw * base_rtt.as_secs_f64())
    }

    pub(super) fn start_recovery_cap(&mut self, until: Instant, rate_cap: f64) {
        if rate_cap > 0.0 {
            self.phase = Phase::Recovering { until, rate_cap };
        }
    }

    #[cfg(test)]
    pub(super) fn set_recovery_cap_for_test(&mut self, until: Instant, rate_cap: f64) {
        self.phase = Phase::Recovering { until, rate_cap };
    }

    fn transition_to_recovering(
        &mut self,
        now: Instant,
        srtt: Duration,
        rtt_min: Duration,
        d: ActiveDrain,
    ) -> DrainFinished {
        let plateau_converged = d.flat_rounds >= 2
            && !d.congested
            && d.kind == DrainKind::RefreshRttMin
            && d.slot.is_none()
            && d.drain_min_rtt < Duration::MAX;
        let baseline_rose = plateau_converged
            && d.drain_min_rtt > rtt_min
            && d.drain_min_rtt.saturating_sub(rtt_min)
                > Duration::from_millis(5).max(rtt_min.mul_f64(0.15));
        let new_base = if baseline_rose {
            d.drain_min_rtt
        } else if d.drain_min_rtt < Duration::MAX {
            rtt_min.min(d.drain_min_rtt)
        } else {
            rtt_min
        };
        self.trusted_min = Some(new_base);

        // Adapt `RefreshRttMin` cooldown: reset when RTT_min improved, back off when stable
        if d.kind == DrainKind::RefreshRttMin {
            let improve_tol = Duration::from_millis(1).max(d.rtt_min_at_start.mul_f64(0.02));
            if d.rtt_min_at_start.saturating_sub(new_base) >= improve_tol {
                self.backoff_slots = 1;
            } else {
                self.backoff_slots = (self.backoff_slots * 2).min(MAX_BACKOFF_SLOTS);
            }
            if let Some(slot) = d.slot {
                self.next_drain_slot = slot + self.backoff_slots;
            }
        } else {
            self.backoff_slots = 1;
        }

        let rate_cap = d.saved_rate;
        if rate_cap > 0.0 && !srtt.is_zero() {
            self.phase = Phase::Recovering {
                until: now + srtt,
                rate_cap,
            };
        } else {
            self.phase = Phase::Normal;
        }

        let probe_dq = (d.kind == DrainKind::MeasureJitter && d.probe_acks > 0)
            .then(|| d.probe_dq_sum / d.probe_acks as f64);
        let bdp_est = self.estimated_bdp(now, srtt, new_base);

        DrainFinished {
            kind: d.kind,
            slot: d.slot,
            saved_window: d.saved_window,
            rtt_min_at_start: d.rtt_min_at_start,
            drain_min_rtt: new_base,
            congested: d.congested,
            probe_dq,
            bdp_est,
            plateau_converged,
        }
    }
}
