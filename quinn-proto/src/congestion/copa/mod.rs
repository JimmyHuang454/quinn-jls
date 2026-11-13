//! Copa: practical delay-based congestion control
//!
//! Arun and Balakrishnan, "Copa: Practical Delay-Based Congestion Control for the Internet",
//! NSDI 2018. Optionally with copa-auto, which chooses delta online (see [`CopaAutoConfig`]).
//!
//! Control law: the queueing delay is `d_q = RTT_standing - RTT_min`, where `RTT_min` is the
//! minimum RTT over 10 s and `RTT_standing` the minimum over the last half smoothed RTT. The
//! target rate is `1 / (delta * d_q)` packets per second and the current rate
//! `cwnd / RTT_standing`. On every acknowledged packet the window moves towards the target by
//! `v / (delta * cwnd)` packets, where the velocity `v` doubles every RTT once the window has
//! moved in the same direction for three RTTs. Packets are paced at `2 * cwnd / RTT_standing`.
//! At equilibrium `lambda * d_q = 1 / delta`.
//!
//! The competitive mode (enabled by default, see [`CopaConfig::mode_switching`]) handles
//! buffer-filling competitors: when the queue has not been nearly empty for five RTTs, `1/delta`
//! grows by one packet per RTT and is halved on loss. Naming in the delta trade-off paper:
//! `Copa` / `Copa-NC` use a fixed delta with the competitive mode on / off, `copa-auto` /
//! `copa-auto-NC` choose delta online with the competitive mode on / off.

use std::any::Any;
use std::sync::Arc;

use super::{BASE_DATAGRAM_SIZE, Controller, ControllerFactory, ControllerMetrics};
use crate::connection::RttEstimator;
use crate::packet::SpaceId;
use crate::{Duration, Instant};

mod auto;
mod filter;
#[cfg(test)]
mod scenarios;
#[cfg(test)]
mod sim;
#[cfg(test)]
mod tests;

pub use auto::CopaAutoEstimate;
use auto::{AutoDelta, ProbeBoard, Sample};
use filter::{Kind, WindowedFilter};

/// Minimum congestion window in packets (as in the ns-3 implementation)
const MIN_WINDOW_PACKETS: u64 = 4;

/// Upper bound of the window, to keep the floating-point arithmetic sane
const MAX_WINDOW: f64 = (1u64 << 40) as f64;

/// Direction in which the window moved during the last RTT
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    None,
    Up,
    Down,
}

/// How the velocity parameter is updated
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopaVelocity {
    /// Rule of the paper: once per window, compare cwnd with its value one window earlier;
    /// reset to 1 when the direction changes, double each window after three in the same
    /// direction
    Paper,
    /// Rule of the authors' genericCC implementation: count the increase/decrease decisions of
    /// the ACKs of each RTT, double while the majority keeps the direction, reset otherwise
    GenericCc,
}

/// Copa congestion controller
#[derive(Debug, Clone)]
pub struct Copa {
    config: Arc<CopaConfig>,
    current_mtu: u64,
    /// Congestion window in bytes, fractional so that small steps on large windows add up
    window: f64,
    slow_start: bool,

    rtt_min: WindowedFilter,
    rtt_standing: WindowedFilter,
    rtt_max: WindowedFilter,
    srtt: Duration,
    standing: Option<Duration>,
    dq: Duration,

    delta: f64,
    competitive: bool,
    last_nearly_empty: Instant,
    last_delta_decrease: Option<Instant>,
    delta_before_decrease: Option<f64>,

    velocity: u64,
    direction: Direction,
    same_direction: u32,
    last_recorded_window: Option<(Instant, f64)>,
    ack_votes: i64,
    prev_vote_dir: i64,

    auto: Option<AutoDelta>,
    /// Window and RTT_min when the current aligned probe began, and whether a loss was signalled
    /// since (fast recovery)
    probe_saved: Option<ProbeSaved>,
    /// Drain in progress until this time, the window held meanwhile, and the last slot drained
    drain_until: Option<Instant>,
    drain_window: f64,
    drained_slot: Option<u64>,
    /// Last packet that found the bottleneck idle: RTT_min is fresh without a drain
    last_idle: Option<Instant>,
    /// RTT_min measured at the end of the last drain or probe, lowered by any smaller sample.
    /// The idle test for skipping drains compares against it rather than against RTT_min, which
    /// drifts up with a busy queue and would then make every packet look idle.
    trusted_min: Option<Duration>,
    /// After a restore, pace at the rate from before the probe until this time
    pace_cap: Option<(Instant, f64)>,
    /// Drains started and skipped (RTT_min was fresh)
    #[cfg(test)]
    drains: (u64, u64),
}

#[derive(Debug, Clone, Copy)]
struct ProbeSaved {
    window: f64,
    /// Pacing rate (bytes/s) when the probe began
    rate: f64,
    rtt_min: Duration,
    congested: bool,
    /// A drain (window held at its minimum) rather than a probe at delta_max: always restored
    /// in full, since the window was cut on purpose
    drain: bool,
}

impl Copa {
    /// Construct a state using the given `config` and current time `now`
    pub fn new(config: Arc<CopaConfig>, now: Instant, current_mtu: u16) -> Self {
        let auto = config
            .adaptive
            .clone()
            .map(|auto| AutoDelta::new(Arc::new(auto), now));
        Self {
            current_mtu: current_mtu as u64,
            window: config.initial_window as f64,
            slow_start: true,
            rtt_min: WindowedFilter::new(Kind::Min),
            rtt_standing: WindowedFilter::new(Kind::Min),
            rtt_max: WindowedFilter::new(Kind::Max),
            srtt: Duration::ZERO,
            standing: None,
            dq: Duration::ZERO,
            delta: config.delta,
            competitive: false,
            last_nearly_empty: now,
            last_delta_decrease: None,
            delta_before_decrease: None,
            velocity: 1,
            direction: Direction::None,
            same_direction: 0,
            last_recorded_window: None,
            ack_votes: 0,
            prev_vote_dir: 1,
            auto,
            probe_saved: None,
            drain_until: None,
            drain_window: 0.0,
            drained_slot: None,
            last_idle: None,
            trusted_min: None,
            pace_cap: None,
            #[cfg(test)]
            drains: (0, 0),
            config,
        }
    }

    /// Delta currently used by the control law
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Current velocity parameter
    pub fn velocity(&self) -> u64 {
        self.velocity
    }

    /// Whether the competitive mode is active
    pub fn is_competitive(&self) -> bool {
        self.competitive
    }

    /// Whether the controller is still in its initial slow start
    pub fn in_slow_start(&self) -> bool {
        self.slow_start
    }

    /// Last measured queueing delay `RTT_standing - RTT_min`
    pub fn queueing_delay(&self) -> Duration {
        self.dq
    }

    /// Minimum RTT over the configured window, `None` before the first sample
    pub fn min_rtt(&self) -> Option<Duration> {
        self.rtt_min.get()
    }

    /// Minimum RTT over the last half smoothed RTT, `None` before the first sample
    pub fn standing_rtt(&self) -> Option<Duration> {
        self.standing
    }

    /// Estimates of copa-auto, `None` when it is disabled
    pub fn adaptive_estimate(&self) -> Option<CopaAutoEstimate> {
        self.auto.as_ref().map(|a| a.estimate())
    }

    fn minimum_window(&self) -> f64 {
        (MIN_WINDOW_PACKETS * self.current_mtu) as f64
    }

    fn window_packets(&self) -> f64 {
        (self.window / self.current_mtu as f64).max(1.0)
    }

    /// Switch between the default and the competitive mode
    fn update_mode(&mut self, now: Instant, dq: Duration, rtt_min: Duration) {
        if !self.config.mode_switching {
            return;
        }
        let rtt_max = self.rtt_max.get().unwrap_or(rtt_min);
        // "Nearly empty": queueing delay below 10% of the RTT oscillation of the last four RTTs
        if dq.as_secs_f64() < 0.1 * rtt_max.saturating_sub(rtt_min).as_secs_f64() {
            self.last_nearly_empty = now;
        }
        let competitive = now.saturating_duration_since(self.last_nearly_empty) > 5 * self.srtt;
        if competitive != self.competitive {
            self.competitive = competitive;
            // Both modes start from the configured delta
            self.delta = self.config.delta;
            self.last_delta_decrease = None;
            self.delta_before_decrease = None;
            tracing::debug!(competitive, "copa mode switch");
        }
    }

    /// Velocity rule of the paper, evaluated once per window
    fn update_velocity_paper(&mut self, now: Instant) {
        let Some((recorded_at, recorded_window)) = self.last_recorded_window else {
            self.last_recorded_window = Some((now, self.window));
            return;
        };
        if now.saturating_duration_since(recorded_at) < self.srtt {
            return;
        }
        let direction = if self.window > recorded_window {
            Direction::Up
        } else {
            Direction::Down
        };
        if direction != self.direction {
            self.velocity = 1;
            self.same_direction = 0;
        } else {
            self.same_direction += 1;
            if self.same_direction >= 3 {
                self.velocity = self.velocity.saturating_mul(2);
            }
        }
        self.cap_velocity();
        self.direction = direction;
        self.last_recorded_window = Some((now, self.window));
    }

    /// Velocity rule of genericCC: votes of the ACKs during the last RTT
    fn update_velocity_generic_cc(&mut self, now: Instant, increase: bool, rtt: Duration) {
        let recorded_at = self.last_recorded_window.map_or(now, |(t, _)| t);
        if self.last_recorded_window.is_none() {
            self.last_recorded_window = Some((now, self.window));
        }
        if recorded_at + rtt < now {
            if self.prev_vote_dir * self.ack_votes > 0 {
                self.velocity = self.velocity.saturating_mul(2);
            } else {
                self.velocity = 1;
                self.prev_vote_dir = -self.prev_vote_dir;
            }
            self.last_recorded_window = Some((now, self.window));
            self.ack_votes = 0;
        }
        if self.velocity as f64 > self.delta * self.window_packets() {
            self.velocity /= 2;
        }
        self.velocity = self.velocity.max(1);
        self.ack_votes += if increase { 1 } else { -1 };
        self.direction = if increase {
            Direction::Up
        } else {
            Direction::Down
        };
    }

    /// As in genericCC, a step of `v / (delta * cwnd)` must not exceed one packet per ACK
    fn cap_velocity(&mut self) {
        let limit = self.delta * self.window_packets();
        while self.velocity > 1 && self.velocity as f64 > limit {
            self.velocity /= 2;
        }
    }

    /// Fast recovery around copa-auto's aligned probes, and the short drains between them.
    ///
    /// A probe sets delta to its maximum, so the window collapses (to a few packets on a jittery
    /// link); regrowing it through the control law takes seconds. The capacity has not changed
    /// meanwhile, so the window saved at the start of the probe is restored at its end, as BBR
    /// does after PROBE_RTT, and for one smoothed RTT packets are paced at the rate from before
    /// the probe instead of `2 cwnd / RTT_standing`, whose short standing RTT right after the
    /// probe would send the whole window as a burst. Skipped after a loss or when RTT_min moved
    /// notably (the path may have changed).
    ///
    /// A drain keeps RTT_min at the propagation delay when several flows keep the queue busy. It
    /// is skipped when a packet found the bottleneck idle recently (RTT_min is then fresh, e.g.
    /// after another flow's drain, which also lines the drains of flows up as in BBR), holds the
    /// window at `drain_window` times the estimated BDP (`cwnd RTT_min / RTT_standing`) for
    /// `2 d_q + srtt` (the queue empties at half the rate, then one RTT to see it), and is then
    /// restored in full. Returns whether a drain is in progress.
    fn fast_recovery(&mut self, now: Instant, rtt_min: Duration) -> bool {
        let Some(auto) = &self.auto else { return false };
        if let Some(slot) = auto.drain_slot(now, self.slow_start)
            && self.drained_slot != Some(slot)
        {
            self.drained_slot = Some(slot);
            let fresh = self
                .last_idle
                .is_some_and(|t| now.saturating_duration_since(t) < auto.drain_skip());
            #[cfg(test)]
            {
                if fresh {
                    self.drains.1 += 1;
                } else {
                    self.drains.0 += 1;
                }
            }
            if !fresh {
                let base = self.trusted_min.unwrap_or(rtt_min).min(rtt_min);
                let standing = self.standing.unwrap_or(rtt_min).max(base);
                let bdp = self.window * base.as_secs_f64() / standing.as_secs_f64().max(1e-9);
                self.drain_window = (auto.drain_window() * bdp).max(self.minimum_window());
                let hold = auto.drain_min().max(2 * self.dq + self.srtt);
                self.drain_until = Some(now + hold);
            }
        }
        let draining = self.drain_until.is_some_and(|until| now < until);
        if !draining {
            self.drain_until = None;
        }
        let probing = draining || auto.in_aligned_probe(now, self.slow_start);
        match (probing, self.probe_saved) {
            (true, None) => {
                self.probe_saved = Some(ProbeSaved {
                    window: self.window,
                    rate: self.pacing_rate().map_or(0.0, |r| r as f64),
                    rtt_min,
                    congested: false,
                    drain: draining,
                });
            }
            (false, Some(saved)) => {
                self.probe_saved = None;
                // The queue was emptied: RTT_min is now the propagation delay
                self.trusted_min = Some(rtt_min);
                let ratio = if saved.drain { 1.0 } else { auto.recovery() };
                let moved =
                    auto.rtt_min_changed(saved.rtt_min.as_secs_f64(), rtt_min.as_secs_f64());
                if ratio > 0.0 && !saved.congested && (saved.drain || !moved) {
                    self.window = self.window.max(ratio * saved.window).min(MAX_WINDOW);
                    self.reset_velocity();
                    if saved.rate > 0.0 {
                        self.pace_cap = Some((now + self.srtt, saved.rate));
                    }
                }
            }
            _ => {}
        }
        draining
    }

    fn reset_velocity(&mut self) {
        self.velocity = 1;
        self.direction = Direction::None;
        self.same_direction = 0;
        self.last_recorded_window = None;
        self.ack_votes = 0;
        self.prev_vote_dir = 1;
    }
}

impl Copa {
    /// Sending rate in bytes per second Copa wants packets paced at
    ///
    /// Reported to the connection as [`ControllerMetrics::pacing_rate`], which drives the pacer.
    pub(super) fn pacing_rate(&self) -> Option<u64> {
        // Twice cwnd / RTT_standing, to accommodate pacing imperfections (section 2.1)
        let standing = self.standing?.as_secs_f64();
        if standing <= 0.0 {
            return None;
        }
        let rate = 2.0 * self.window / standing;
        let rate = self.pace_cap.map_or(rate, |(_, cap)| rate.min(cap));
        Some(rate as u64)
    }
}

impl Controller for Copa {
    fn on_ack(
        &mut self,
        now: Instant,
        sent: Instant,
        bytes: u64,
        _pn: u64,
        _space: SpaceId,
        app_limited: bool,
        rtt: &RttEstimator,
    ) {
        let sample = now.saturating_duration_since(sent);
        // The connection updates its RTT estimator after the congestion controller has seen the
        // acknowledged packets, so the very first samples fall back to the initial RTT
        self.srtt = rtt.get();
        let srtt = self.srtt;

        self.rtt_min.update(
            now,
            sample,
            self.config.min_rtt_window.max(Duration::from_micros(1)),
        );
        self.rtt_standing
            .update(now, sample, (srtt / 2).max(Duration::from_micros(1)));
        self.rtt_max
            .update(now, sample, (4 * srtt).max(Duration::from_micros(1)));
        let (Some(rtt_min), Some(standing)) = (self.rtt_min.get(), self.rtt_standing.get()) else {
            return;
        };
        if standing.is_zero() {
            return;
        }
        self.standing = Some(standing);
        // eq. (2) of the paper
        let dq = standing.saturating_sub(rtt_min);
        self.dq = dq;
        self.update_mode(now, dq, rtt_min);

        let mss = self.current_mtu as f64;
        let acked_packets = bytes as f64 / mss;
        if self.pace_cap.is_some_and(|(until, _)| now >= until) {
            self.pace_cap = None;
        }
        if !self.competitive {
            // The default mode uses the configured delta, or the one chosen by copa-auto
            self.delta = self.config.delta;
            let excluded =
                self.drain_until.is_some_and(|until| now < until) || self.pace_cap.is_some();
            if let Some(auto) = &mut self.auto {
                if !self.slow_start && self.trusted_min.is_none() && !auto.probing_jitter() {
                    self.trusted_min = Some(rtt_min);
                }
                if let Some(trusted) = &mut self.trusted_min {
                    *trusted = (*trusted).min(sample);
                    if auto.is_idle(sample, *trusted) {
                        self.last_idle = Some(now);
                    }
                }
                auto.on_ack(
                    Sample {
                        now,
                        rtt: sample,
                        rtt_min,
                        dq,
                        srtt,
                        slow_start: self.slow_start,
                        competitive: false,
                        app_limited,
                        excluded,
                    },
                    self.delta,
                );
                self.delta = auto.delta(now, self.slow_start, self.config.delta);
                if self.fast_recovery(now, rtt_min) {
                    // Draining: the window is held below the BDP, the control law is paused
                    self.window = self.drain_window;
                    return;
                }
            }
        } else {
            // A probe interrupted by the competitive mode is not restored
            self.probe_saved = None;
            self.drain_until = None;
            self.pace_cap = None;
            if let Some(auto) = &mut self.auto {
                // Keeps the round clock running; competitive half-periods are discarded
                auto.on_ack(
                    Sample {
                        now,
                        rtt: sample,
                        rtt_min,
                        dq,
                        srtt,
                        slow_start: self.slow_start,
                        competitive: true,
                        app_limited,
                        excluded: true,
                    },
                    self.delta,
                );
            }
            // Additive increase of 1/delta: one packet per RTT
            self.delta = 1.0 / (1.0 / self.delta + acked_packets / self.window_packets());
        }

        let increase = if dq.is_zero() {
            true
        } else {
            // Target rate 1 / (delta dq) against the current rate cwnd / RTT_standing, both in
            // packets per second
            let target = 1.0 / (self.delta * dq.as_secs_f64());
            let current = self.window / mss / standing.as_secs_f64();
            target >= current
        };

        if app_limited && increase {
            // The window was not the limit: growing it would not be backed by any measurement
            return;
        }

        if !(increase && self.slow_start) {
            match self.config.velocity {
                CopaVelocity::Paper => self.update_velocity_paper(now),
                CopaVelocity::GenericCc => self.update_velocity_generic_cc(now, increase, sample),
            }
        }

        if increase {
            if self.slow_start {
                // One packet per acknowledged packet: cwnd doubles per RTT, clocked by the ACKs
                self.window += bytes as f64;
            } else {
                self.window += acked_packets * mss * self.velocity as f64
                    / (self.delta * self.window_packets());
            }
        } else {
            self.slow_start = false;
            let step =
                acked_packets * mss * self.velocity as f64 / (self.delta * self.window_packets());
            self.window -= step;
        }
        self.window = self.window.clamp(self.minimum_window(), MAX_WINDOW);
    }

    fn on_congestion_event(
        &mut self,
        now: Instant,
        _sent: Instant,
        is_persistent_congestion: bool,
        _is_ecn: bool,
        _lost_bytes: u64,
        _largest_lost: u64,
        _space: SpaceId,
    ) {
        if let Some(saved) = &mut self.probe_saved {
            saved.congested = true;
        }
        if is_persistent_congestion {
            // Comparable to a TCP retransmission timeout, which the paper does not cover: as TCP
            // does (RFC 5681), restart from the minimum window in slow start
            self.window = self.minimum_window();
            self.slow_start = true;
            self.reset_velocity();
            return;
        }
        // The default mode ignores loss, which makes Copa robust to non-congestive losses. The
        // competitive mode halves 1/delta at most once per RTT, like NewReno halves its window.
        if !self.competitive {
            return;
        }
        if self
            .last_delta_decrease
            .is_some_and(|last| now.saturating_duration_since(last) < self.srtt)
        {
            return;
        }
        self.delta_before_decrease = Some(self.delta);
        self.delta = self.config.delta.min(2.0 * self.delta);
        self.last_delta_decrease = Some(now);
    }

    fn on_spurious_congestion_event(&mut self) {
        if let (true, Some(delta)) = (self.competitive, self.delta_before_decrease.take()) {
            self.delta = delta;
        }
    }

    fn on_mtu_update(&mut self, new_mtu: u16) {
        self.current_mtu = new_mtu as u64;
        self.window = self.window.max(self.minimum_window());
    }

    fn window(&self) -> u64 {
        self.window as u64
    }

    fn metrics(&self) -> ControllerMetrics {
        ControllerMetrics {
            congestion_window: self.window(),
            ssthresh: None,
            pacing_rate: self.pacing_rate(),
            bandwidth_estimate: None,
            send_quantum: None,
        }
    }

    fn clone_box(&self) -> Box<dyn Controller> {
        Box::new(self.clone())
    }

    fn initial_window(&self) -> u64 {
        self.config.initial_window
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// Configuration for the [`Copa`] congestion controller
///
/// The defaults are those of the paper (delta 0.5, competitive mode on) and of the ns-3
/// implementation used by the delta trade-off paper.
#[derive(Debug, Clone)]
pub struct CopaConfig {
    initial_window: u64,
    delta: f64,
    mode_switching: bool,
    min_rtt_window: Duration,
    velocity: CopaVelocity,
    adaptive: Option<CopaAutoConfig>,
}

impl CopaConfig {
    /// Copa with the competitive mode off (Copa-NC)
    pub fn copa_nc() -> Self {
        let mut config = Self::default();
        config.mode_switching(false);
        config
    }

    /// copa-auto: delta chosen online, competitive mode on
    pub fn copa_auto() -> Self {
        let mut config = Self::default();
        config.adaptive(Some(CopaAutoConfig::default()));
        config
    }

    /// copa-auto-NC: delta chosen online, competitive mode off
    pub fn copa_auto_nc() -> Self {
        let mut config = Self::copa_auto();
        config.mode_switching(false);
        config
    }

    /// Default limit on the amount of outstanding data in bytes
    pub fn initial_window(&mut self, value: u64) -> &mut Self {
        self.initial_window = value;
        self
    }

    /// Latency factor delta of the default mode (0.5 in the paper)
    ///
    /// Smaller values tolerate more queueing for more throughput. With copa-auto, it is only
    /// used until the first estimate. Panics unless positive and finite.
    pub fn delta(&mut self, value: f64) -> &mut Self {
        assert!(value > 0.0 && value.is_finite(), "delta must be positive");
        self.delta = value;
        self
    }

    /// Whether Copa switches to the competitive mode when a buffer-filling flow keeps the queue
    /// from draining (on by default, as in the paper)
    pub fn mode_switching(&mut self, value: bool) -> &mut Self {
        self.mode_switching = value;
        self
    }

    /// Window of the minimum RTT filter (10 s in the paper)
    pub fn min_rtt_window(&mut self, value: Duration) -> &mut Self {
        self.min_rtt_window = value;
        self
    }

    /// Velocity update rule
    pub fn velocity(&mut self, value: CopaVelocity) -> &mut Self {
        self.velocity = value;
        self
    }

    /// Choose delta online (copa-auto); `None` keeps it fixed
    pub fn adaptive(&mut self, value: Option<CopaAutoConfig>) -> &mut Self {
        self.adaptive = value;
        self
    }
}

impl Default for CopaConfig {
    fn default() -> Self {
        Self {
            initial_window: 14720.clamp(2 * BASE_DATAGRAM_SIZE, 10 * BASE_DATAGRAM_SIZE),
            delta: 0.5,
            mode_switching: true,
            min_rtt_window: Duration::from_secs(10),
            velocity: CopaVelocity::Paper,
            adaptive: None,
        }
    }
}

impl ControllerFactory for CopaConfig {
    fn build(self: Arc<Self>, now: Instant, current_mtu: u16) -> Box<dyn Controller> {
        Box::new(Copa::new(self, now, current_mtu))
    }
}

/// Parameters of copa-auto, which chooses delta online in Copa's default mode
///
/// Defaults are those of the ns-3 implementation (`TcpCopa` `Adaptive*` attributes).
#[derive(Debug, Clone)]
pub struct CopaAutoConfig {
    delta_min: f64,
    delta_max: f64,
    half_rounds: u32,
    settle_rounds: u32,
    probe_every: u32,
    weight: f64,
    gain: f64,
    gain_far: f64,
    far_ratio: f64,
    fair_increase: f64,
    probe_period: Duration,
    probe_duration: Duration,
    idle_margin: Duration,
    probe_epoch: Option<Instant>,
    adaptive_probes: bool,
    stable_probes: u32,
    stable_tolerance: (Duration, f64),
    reprobe_rtt_change: (Duration, f64),
    fallback_period: Duration,
    recovery: f64,
    drain_min: Duration,
    probe_rtts: u32,
    probe_duration_min: Duration,
    fallback_max: Duration,
    drain_window: f64,
    drain_skip: Duration,
    drain_enabled: bool,
    jitter_floor_ms: f64,
    board: Arc<ProbeBoard>,
}

impl CopaAutoConfig {
    /// Lower and upper bounds of delta (0.001 and 0.5); the upper one is also the delta of the
    /// low-load probes
    pub fn delta_range(&mut self, min: f64, max: f64) -> &mut Self {
        assert!(
            0.0 < min && min <= max && max.is_finite(),
            "invalid delta range"
        );
        self.delta_min = min;
        self.delta_max = max;
        self
    }

    /// Rounds (smoothed RTTs) per half-period (8), of which the first `settle` (3) are discarded
    pub fn half_period(&mut self, rounds: u32, settle: u32) -> &mut Self {
        assert!(rounds > 0 && settle < rounds, "invalid half-period");
        self.half_rounds = rounds;
        self.settle_rounds = settle;
        self
    }

    /// Without aligned probes, two low-load half-periods in every `value` half-periods (120)
    pub fn probe_every(&mut self, value: u32) -> &mut Self {
        self.probe_every = value.max(2);
        self
    }

    /// Weight of a new `k` sample in its log-domain moving average (0.3)
    pub fn weight(&mut self, value: f64) -> &mut Self {
        self.weight = value.clamp(0.0, 1.0);
        self
    }

    /// Exponent eta of the multiplicative update (0.25 near the target, 0.25 when `d_q / d_q*`
    /// or its inverse exceeds `far_ratio` = 1.5)
    pub fn gain(&mut self, near: f64, far: f64, far_ratio: f64) -> &mut Self {
        assert!(far_ratio >= 1.0, "far_ratio must be at least 1");
        self.gain = near;
        self.gain_far = far;
        self.far_ratio = far_ratio;
        self
    }

    /// Additive increase `a` of `1/delta`, in packets per unit of `ln(d_q* / d_q)` (10). Zero
    /// updates delta multiplicatively in both directions.
    pub fn fair_increase(&mut self, value: f64) -> &mut Self {
        self.fair_increase = value.max(0.0);
        self
    }

    /// Period (20 s) and longest length (2 s) of the low-load probes aligned on the wall clock.
    /// A zero period uses `probe_every` half-periods of each flow instead.
    pub fn probe_timing(&mut self, period: Duration, duration: Duration) -> &mut Self {
        self.probe_period = period;
        self.probe_duration = duration;
        self
    }

    /// A packet found the bottleneck idle when its `RTT - RTT_min` is at most the jitter floor
    /// plus this margin (1 ms)
    pub fn idle_margin(&mut self, value: Duration) -> &mut Self {
        self.idle_margin = value;
        self
    }

    /// Probe only while the jitter floor is unsettled (true), or in every aligned slot (false,
    /// the original schedule)
    ///
    /// With `true`, aligned probes run until `stable_probes` consecutive probes agree on `j`;
    /// then they stop, except when `RTT_min` moves by more than the re-probe threshold and in the
    /// slots that are a multiple of the fallback period, which stay aligned on the wall clock so
    /// that the flows of a host still probe together.
    pub fn adaptive_probes(&mut self, value: bool) -> &mut Self {
        self.adaptive_probes = value;
        self
    }

    /// Number of consecutive probes that must agree on `j` (3), and how close they must be: within
    /// `max(abs, rel * j)` (0.5 ms, 10%)
    pub fn stable_probes(&mut self, count: u32, abs: Duration, rel: f64) -> &mut Self {
        self.stable_probes = count.max(1);
        self.stable_tolerance = (abs, rel.max(0.0));
        self
    }

    /// A change of `RTT_min` by more than `max(abs, rel * RTT_min)` (5 ms, 20%) since the last
    /// probe, in either direction, starts a new estimate of `j`; so does a fallback probe that
    /// finds a higher floor. Fallback probes run every `fallback` (120 s) once `j` is settled,
    /// and the period doubles up to `fallback_max` (600 s) while they agree; slots that are
    /// multiples of `fallback_max` (wall-clock aligned) always probe.
    pub fn reprobe(
        &mut self,
        abs: Duration,
        rel: f64,
        fallback: Duration,
        fallback_max: Duration,
    ) -> &mut Self {
        self.reprobe_rtt_change = (abs, rel.max(0.0));
        self.fallback_period = fallback;
        self.fallback_max = fallback_max.max(fallback);
        self
    }

    /// Probe length in smoothed RTTs (16: eight to settle, eight measured), within
    /// `[min, probe_duration]` (300 ms to 2 s). The flows of a board use the longest length
    /// registered for the slot. Zero uses the fixed `probe_duration`.
    pub fn probe_rtts(&mut self, rtts: u32, min: Duration) -> &mut Self {
        self.probe_rtts = rtts;
        self.probe_duration_min = min;
        self
    }

    /// Fast recovery after an aligned probe: restore the window to `ratio` (0.5) of its value
    /// when the probe began, unless a loss was signalled or `RTT_min` moved notably meanwhile,
    /// and pace at the pre-probe rate for one smoothed RTT. 0 disables it (the window then
    /// regrows through the control law).
    pub fn probe_recovery(&mut self, ratio: f64) -> &mut Self {
        self.recovery = ratio.clamp(0.0, 1.0);
        self
    }

    /// Drains in the aligned slots that do not measure `j`: the window is held at `window` (0.5)
    /// times the estimated BDP for `max(min, 2 d_q + srtt)` (`min` 50 ms), unless a packet found
    /// the bottleneck idle within `skip` (10 s, the RTT_min window). A zero `min` disables the
    /// drains, which lets `RTT_min` drift up when several flows keep the queue busy.
    pub fn drain(&mut self, min: Duration, window: f64, skip: Duration) -> &mut Self {
        self.drain_min = min;
        self.drain_window = window.clamp(0.0, 1.0);
        self.drain_skip = skip;
        self
    }

    /// Enable or disable the on-demand queue drains in aligned slots that do not measure `j`
    /// (`true` by default)
    pub fn drain_enabled(&mut self, value: bool) -> &mut Self {
        self.drain_enabled = value;
        self
    }

    /// Fixed jitter floor in milliseconds, or negative to probe online (`-1.0` by default)
    ///
    /// A negative value enables low-load probing; `0.0` pauses probing and fixes `j = 0`;
    /// a positive value pauses probing and fixes `j` to `value` ms.
    pub fn jitter_floor_ms(&mut self, value: f64) -> &mut Self {
        assert!(value.is_finite(), "jitter_floor_ms must be finite");
        self.jitter_floor_ms = value;
        self
    }

    /// The original copa-auto: a probe in every aligned slot and no fast recovery
    pub fn periodic_probes(&mut self) -> &mut Self {
        self.adaptive_probes(false)
            .probe_recovery(0.0)
            .probe_rtts(0, Duration::ZERO)
    }

    /// Align the probes on `epoch` instead of the wall clock (for simulations)
    #[doc(hidden)]
    pub fn probe_epoch(&mut self, epoch: Option<Instant>) -> &mut Self {
        self.probe_epoch = epoch;
        self
    }
}

impl Default for CopaAutoConfig {
    fn default() -> Self {
        Self {
            delta_min: 0.001,
            delta_max: 0.5,
            half_rounds: 8,
            settle_rounds: 3,
            probe_every: 120,
            weight: 0.3,
            gain: 0.25,
            gain_far: 0.25,
            far_ratio: 1.5,
            fair_increase: 10.0,
            probe_period: Duration::from_secs(20),
            probe_duration: Duration::from_secs(2),
            idle_margin: Duration::from_millis(1),
            probe_epoch: None,
            adaptive_probes: true,
            stable_probes: 3,
            stable_tolerance: (Duration::from_micros(500), 0.1),
            reprobe_rtt_change: (Duration::from_millis(5), 0.2),
            fallback_period: Duration::from_secs(120),
            recovery: 0.5,
            drain_min: Duration::from_millis(50),
            probe_rtts: 16,
            probe_duration_min: Duration::from_millis(300),
            fallback_max: Duration::from_secs(600),
            drain_window: 0.5,
            drain_skip: Duration::from_secs(10),
            drain_enabled: true,
            jitter_floor_ms: -1.0,
            board: Arc::default(),
        }
    }
}
