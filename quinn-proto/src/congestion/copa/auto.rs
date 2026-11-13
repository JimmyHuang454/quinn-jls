//! copa-auto: online choice of Copa's delta in the default mode
//!
//! Every smoothed RTT is a round; `half_rounds` rounds form a half-period, of which the first
//! `settle_rounds` are discarded while Copa settles. At the end of each valid half-period:
//!
//! * the utilization `rho` is one minus the fraction of packets that found the bottleneck idle,
//!   i.e. whose `RTT - RTT_min <= j + idle_margin`;
//! * the jitter floor `j` is the lowest mean `d_q` measured during low-load probes, in which every
//!   flow of the host uses `delta_max`. Probes are aligned on the wall clock (20 s slots), and
//!   the flows built from one [`CopaAutoConfig`] share a probe board: a flow that needs a probe
//!   registers it for the next slot, and every flow probes in a registered slot, so no flow
//!   measures while another one refills the queue. A probe lasts `probe_rtts` smoothed RTTs
//!   (the longest one registered for the slot, within `[probe_duration_min, probe_duration]`)
//!   and its second half is measured. With adaptive probes (the default) a flow asks for probes
//!   only until three agree on `j`, then again when `RTT_min` moves notably or in the fallback
//!   slots, whose period doubles from 120 s to 600 s while the probes keep agreeing; slots that
//!   are multiples of the longest fallback period always probe, which also aligns flows that do
//!   not share a board. A fallback probe that finds a higher floor, or an `RTT_min` change,
//!   starts a new estimate, so `j` can also grow;
//! * the ratio `k = (d_q - j)(1 - rho) / rho` is averaged in the log domain;
//! * the queueing target is `d_q* = j + sqrt(k * R0')`, with `R0' = RTT_min + j`;
//! * delta is moved towards `delta * d_q / d_q*`: multiplicatively when `d_q >= d_q*`, and by an
//!   additive increase of `1/delta` when `d_q < d_q*`, so that the deltas of flows sharing a
//!   bottleneck converge (AIMD on `1/delta`).
//!
//! This is the algorithm of the delta trade-off paper (section "在线算法 copa-auto",
//! eq:rho-est, eq:k-est, eq:k-ewma, eq:dq-target, eq:delta-update) and of the ns-3 `TcpCopa`
//! `AdaptiveDelta` attribute; defaults are identical to ns-3.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::CopaAutoConfig;
use crate::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Upper bound of the jitter floor, in seconds
const MAX_JITTER: f64 = 0.02;

/// Largest `|ln(d_q / d_q*)|` taken into one delta update (a factor of 4)
const MAX_STEP: f64 = std::f64::consts::LN_2 * 2.0;

/// Probe requests shared by the flows built from one [`CopaAutoConfig`]
///
/// Small rings indexed by slot. `requests` holds `slot + 1` for a slot that some flow wants to
/// probe; `lengths` holds `(slot + 1) << 24 | ms`, the longest probe any flow of the board
/// would need in that slot (every flow publishes, so that a long-RTT flow that only joins still
/// gets a probe it can settle in). `fetch_max` keeps the latest slot and, within a slot, the
/// longest length, without a lock.
#[derive(Debug, Default)]
pub(super) struct ProbeBoard {
    requests: [AtomicU64; 4],
    lengths: [AtomicU64; 4],
}

impl ProbeBoard {
    const LENGTH_BITS: u32 = 24;

    fn publish(&self, slot: u64, length: Duration, request: bool) {
        let i = (slot % 4) as usize;
        if request {
            self.requests[i].fetch_max(slot + 1, Ordering::Relaxed);
        }
        let ms = (length.as_millis() as u64).clamp(1, (1 << Self::LENGTH_BITS) - 1);
        self.lengths[i].fetch_max((slot + 1) << Self::LENGTH_BITS | ms, Ordering::Relaxed);
    }

    fn requested(&self, slot: u64) -> bool {
        self.requests[(slot % 4) as usize].load(Ordering::Relaxed) == slot + 1
    }

    /// Longest probe published for `slot`
    fn length(&self, slot: u64) -> Option<Duration> {
        let value = self.lengths[(slot % 4) as usize].load(Ordering::Relaxed);
        (value >> Self::LENGTH_BITS == slot + 1)
            .then(|| Duration::from_millis(value & ((1 << Self::LENGTH_BITS) - 1)))
    }
}

/// Inputs of one acknowledged packet
#[derive(Debug, Clone, Copy)]
pub(super) struct Sample {
    pub(super) now: Instant,
    /// RTT of this packet
    pub(super) rtt: Duration,
    pub(super) rtt_min: Duration,
    /// Queueing delay `RTT_standing - RTT_min`
    pub(super) dq: Duration,
    pub(super) srtt: Duration,
    pub(super) slow_start: bool,
    pub(super) competitive: bool,
    pub(super) app_limited: bool,
    /// The controller is draining or recovering from a probe: not an equilibrium sample
    pub(super) excluded: bool,
}

/// Estimates published after each half-period, for tests and logging
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
pub struct CopaAutoEstimate {
    /// Jitter floor `j` in seconds, `None` before the first probe ends
    pub jitter: Option<f64>,
    /// Ratio `k = c / mu` in seconds
    pub ratio: f64,
    /// Total utilization of the bottleneck
    pub rho: f64,
    /// `R0' = RTT_min + j` in seconds
    pub base_rtt: f64,
    /// Target queueing delay `d_q*` in seconds
    pub dq_target: f64,
    /// Delta chosen for the default mode, `None` until the first update
    pub delta: Option<f64>,
}

#[derive(Debug, Clone)]
pub(super) struct AutoDelta {
    config: Arc<CopaAutoConfig>,
    /// Phase (seconds, in `[0, probe_period)`) of the aligned probes at `anchor`
    anchor: Instant,
    anchor_phase: f64,

    est: CopaAutoEstimate,

    last_ack: Option<Instant>,
    round_start: Option<Instant>,
    round: Accumulator,
    rounds: u64,
    seg: Accumulator,
    seg_probe: bool,
    seg_app_limited: bool,
    /// Half-periods still to discard after a probe, while Copa leaves the probe's equilibrium
    settle_halves: u32,

    first_probe_done: bool,
    aligned_probe_seen: bool,
    last_probe_dq: f64,
    probe_dq_sum: f64,
    probe_acks: u64,

    /// Wall-clock index of the aligned slot that contains `anchor`
    anchor_slot: u64,
    /// Slot whose probe decision was taken, whether it probes, and the probe length in seconds
    decided_slot: Option<u64>,
    probing_slot: Option<u64>,
    slot_duration: f64,
    /// Fallback probes double their period up to `fallback_max` while they agree
    fallback_level: u32,
    /// Consecutive aligned probes that agreed with the estimate of `j`
    agreeing: u32,
    stable: bool,
    /// The next aligned probe replaces `j` instead of lowering it
    reset_next: bool,
    rtt_min_at_probe: Option<f64>,
    /// Highest probe result since the estimate was (re)started: probes above the minimum are
    /// normal on a jittery link, only one above this range means the floor rose
    probe_hi: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Accumulator {
    dq_sum: f64,
    acks: u64,
    idle: u64,
    max_gap: f64,
}

impl Accumulator {
    fn add(&mut self, other: &Self) {
        self.dq_sum += other.dq_sum;
        self.acks += other.acks;
        self.idle += other.idle;
        self.max_gap = self.max_gap.max(other.max_gap);
    }
}

impl AutoDelta {
    pub(super) fn new(config: Arc<CopaAutoConfig>, now: Instant) -> Self {
        let period = config.probe_period.as_secs_f64();
        let mut anchor_slot = 0;
        let anchor_phase = if period <= 0.0 {
            0.0
        } else {
            // Probes are aligned on absolute time so that all flows of a host probe together
            // and the queue empties: the phase is the wall clock modulo the period, unless a
            // test fixes the epoch.
            let since_epoch = match config.probe_epoch {
                Some(epoch) => now.saturating_duration_since(epoch).as_secs_f64(),
                None => SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0),
            };
            anchor_slot = (since_epoch / period).floor() as u64;
            since_epoch.rem_euclid(period)
        };
        let fixed_jitter = (config.jitter_floor_ms >= 0.0).then(|| config.jitter_floor_ms * 1e-3);
        let est = CopaAutoEstimate {
            jitter: fixed_jitter,
            ..CopaAutoEstimate::default()
        };
        Self {
            config,
            anchor: now,
            anchor_phase,
            est,
            last_ack: None,
            round_start: None,
            round: Accumulator::default(),
            rounds: 0,
            seg: Accumulator::default(),
            seg_probe: false,
            seg_app_limited: false,
            settle_halves: 0,
            first_probe_done: fixed_jitter.is_some(),
            aligned_probe_seen: fixed_jitter.is_some(),
            last_probe_dq: 0.0,
            probe_dq_sum: 0.0,
            probe_acks: 0,
            anchor_slot,
            decided_slot: None,
            probing_slot: None,
            slot_duration: 0.0,
            fallback_level: 0,
            agreeing: 0,
            stable: fixed_jitter.is_some(),
            reset_next: false,
            rtt_min_at_probe: None,
            probe_hi: 0.0,
        }
    }

    pub(super) fn estimate(&self) -> CopaAutoEstimate {
        self.est
    }

    pub(super) fn probing_jitter(&self) -> bool {
        self.config.jitter_floor_ms < 0.0
    }

    fn aligned(&self) -> bool {
        !self.config.probe_period.is_zero()
    }

    fn half_index(&self) -> u64 {
        self.rounds / u64::from(self.config.half_rounds.max(1))
    }

    /// Round-based low-load probe: two half-periods in every `probe_every`, or with aligned
    /// probes only the first probe after the slow start, until the queue it left has drained
    fn in_round_probe(&self) -> bool {
        if !self.probing_jitter() {
            return false;
        }
        if self.aligned() {
            return !self.first_probe_done;
        }
        self.half_index() % u64::from(self.config.probe_every.max(1)) <= 1
    }

    fn phase(&self, now: Instant) -> f64 {
        let period = self.config.probe_period.as_secs_f64();
        let elapsed = now.saturating_duration_since(self.anchor).as_secs_f64();
        (self.anchor_phase + elapsed).rem_euclid(period)
    }

    /// Wall-clock index of the aligned slot containing `now`
    fn slot(&self, now: Instant) -> u64 {
        let period = self.config.probe_period.as_secs_f64();
        let elapsed = now.saturating_duration_since(self.anchor).as_secs_f64();
        self.anchor_slot + ((self.anchor_phase + elapsed) / period).floor() as u64
    }

    fn in_timed_probe(&self, now: Instant, slow_start: bool) -> bool {
        if !self.probing_jitter() || !self.aligned() || slow_start {
            return false;
        }
        if !self.config.adaptive_probes {
            return self.phase(now) < self.config.probe_duration.as_secs_f64();
        }
        self.probing_slot == Some(self.slot(now)) && self.phase(now) < self.slot_duration
    }

    /// Length of the probe this flow wants, in seconds: `probe_rtts` smoothed RTTs within
    /// `[probe_duration_min, probe_duration]`, or the fixed `probe_duration`
    fn wanted_duration(&self, srtt: Duration) -> Duration {
        let max = self.config.probe_duration;
        if self.config.probe_rtts == 0 {
            return max;
        }
        (srtt * self.config.probe_rtts).clamp(self.config.probe_duration_min.min(max), max)
    }

    /// Slots per fallback period at the given level
    fn fallback_every(&self, level: u32) -> u64 {
        let period = self.config.probe_period.as_secs_f64();
        let base = self.config.fallback_period.as_secs_f64();
        let max = self.config.fallback_max.as_secs_f64().max(base);
        let seconds = (base * 2f64.powi(level as i32)).min(max);
        (seconds / period).round().max(1.0) as u64
    }

    /// Slots in which every flow probes, board or not: multiples of the longest fallback period
    fn forced(&self, slot: u64) -> bool {
        slot.is_multiple_of(self.fallback_every(u32::MAX >> 1))
    }

    fn max_fallback_level(&self) -> u32 {
        let base = self.config.fallback_period.as_secs_f64().max(1e-9);
        let max = self.config.fallback_max.as_secs_f64().max(base);
        (max / base).log2().ceil().max(0.0) as u32
    }

    /// An aligned probe after the first one is running: the controller saves its window for the
    /// fast recovery
    pub(super) fn in_aligned_probe(&self, now: Instant, slow_start: bool) -> bool {
        self.first_probe_done && self.in_timed_probe(now, slow_start)
    }

    pub(super) fn first_probe_done(&self) -> bool {
        self.first_probe_done
    }

    /// Whether an aligned low-load probe should gently hold the window at `drain_window * BDP`
    /// (`0.5 * BDP`) rather than jumping `delta` to `delta_max`
    pub(super) fn in_gentle_probe(&self, now: Instant, slow_start: bool) -> bool {
        self.probing_jitter()
            && self.aligned()
            && self.config.gentle_probe
            && !slow_start
            && (!self.first_probe_done || self.in_timed_probe(now, slow_start))
    }

    pub(super) fn recovery(&self) -> f64 {
        self.config.recovery
    }

    /// With adaptive probes, an aligned slot that does not measure `j` is a short drain instead
    /// (like BBR's PROBE_RTT): the window drops to its minimum for about one RTT so that the
    /// queue empties and `RTT_min` (a 10 s minimum) stays the propagation delay. Without it,
    /// several flows keep the queue busy and `RTT_min` creeps up. Returns the slot to drain.
    pub(super) fn drain_slot(&self, now: Instant, slow_start: bool) -> Option<u64> {
        if !self.config.adaptive_probes
            || !self.config.drain_enabled
            || self.config.drain_min.is_zero()
            || !self.aligned()
            || !self.first_probe_done
            || slow_start
        {
            return None;
        }
        let slot = self.slot(now);
        if self.decided_slot == Some(slot) && self.phase(now) >= self.slot_duration {
            return None;
        }
        (self.decided_slot == Some(slot) && self.probing_slot != Some(slot)).then_some(slot)
    }

    pub(super) fn drain_min(&self) -> Duration {
        self.config.drain_min
    }

    /// Whether `RTT_min` moved notably between two values, in seconds
    pub(super) fn rtt_min_changed(&self, a: f64, b: f64) -> bool {
        let (abs, rel) = self.config.reprobe_rtt_change;
        (a - b).abs() > abs.as_secs_f64().max(rel * a.max(b))
    }

    /// Whether the aligned slot starting now should probe: always with the original schedule
    /// and until `j` is settled; afterwards only in the fallback slots (wall-clock multiples of
    /// the fallback period, shared by the flows of a host) or after a notable `RTT_min` change
    fn wants_probe(&self, slot: u64) -> bool {
        if !self.probing_jitter() {
            return false;
        }
        if !self.config.adaptive_probes || !self.first_probe_done || !self.stable {
            return true;
        }
        slot.is_multiple_of(self.fallback_every(self.fallback_level))
    }

    /// Start a new estimate of `j` with the next aligned probe
    fn unsettle(&mut self) {
        self.stable = false;
        self.agreeing = 0;
        self.reset_next = true;
        self.fallback_level = 0;
    }

    /// A packet that found the bottleneck idle: its RTT is within the jitter floor plus the idle
    /// margin of `reference`
    pub(super) fn is_idle(&self, rtt: Duration, reference: Duration) -> bool {
        let j = self.est.jitter.unwrap_or(0.0);
        rtt.as_secs_f64() - reference.as_secs_f64() <= j + self.config.idle_margin.as_secs_f64()
    }

    pub(super) fn drain_window(&self) -> f64 {
        self.config.drain_window
    }

    pub(super) fn drain_skip(&self) -> Duration {
        self.config.drain_skip
    }

    /// Delta that the default mode should use now
    ///
    /// `fallback` is the configured delta, used until the first update.
    pub(super) fn delta(&self, now: Instant, slow_start: bool, fallback: f64) -> f64 {
        if !slow_start && (self.in_round_probe() || self.in_timed_probe(now, slow_start)) {
            if self.aligned() && self.config.gentle_probe {
                return self.est.delta.unwrap_or(fallback);
            }
            return self.config.delta_max;
        }
        self.est.delta.unwrap_or(fallback)
    }

    /// Account one acknowledged packet; `delta` is the delta in use in the default mode
    pub(super) fn on_ack(&mut self, s: Sample, delta: f64) {
        let dq = s.dq.as_secs_f64();
        let round_start = *self.round_start.get_or_insert(s.now);
        if let Some(last) = self.last_ack {
            let gap = s.now.saturating_duration_since(last).as_secs_f64();
            self.round.max_gap = self.round.max_gap.max(gap);
        }
        self.last_ack = Some(s.now);
        self.seg_app_limited |= s.app_limited;
        self.seg_probe |= s.excluded;

        if self.config.adaptive_probes && self.aligned() {
            // A path change shows as a jump of RTT_min: the floor measured before may not hold
            if self.probing_jitter()
                && let Some(r0) = self.rtt_min_at_probe
                && self.stable
                && self.rtt_min_changed(r0, s.rtt_min.as_secs_f64())
            {
                self.unsettle();
            }
            if !s.slow_start {
                let slot = self.slot(s.now);
                let own = self.wanted_duration(s.srtt);
                // Register ahead, once this slot's probe window is over (so that the outcome of a
                // probe in this slot is known), for the flows deciding at the next boundary
                if self.probing_jitter()
                    && self.first_probe_done
                    && self.phase(s.now) >= self.config.probe_duration.as_secs_f64()
                {
                    let request = self.wants_probe(slot + 1);
                    self.config.board.publish(slot + 1, own, request);
                }
                // Decide once per slot, at its first ACK, so the probe keeps its length
                if self.decided_slot != Some(slot)
                    && self.phase(s.now) < self.config.probe_duration.as_secs_f64()
                {
                    self.decided_slot = Some(slot);
                    if self.probing_jitter() {
                        // Late registration still reaches the flows whose first ACK comes later
                        let board = &self.config.board;
                        board.publish(slot, own, self.wants_probe(slot));
                        let length = board.length(slot).unwrap_or(own).max(own);
                        self.slot_duration = length.as_secs_f64();
                        if board.requested(slot) || self.forced(slot) {
                            self.probing_slot = Some(slot);
                        }
                    } else {
                        self.slot_duration = own.as_secs_f64();
                    }
                }
            }
        }

        if self.in_timed_probe(s.now, s.slow_start) {
            self.seg_probe = true;
            // A probe that overlaps the first one still holds the slow-start queue; only the
            // second half of a probe is measured
            let duration = if self.config.adaptive_probes {
                self.slot_duration
            } else {
                self.config.probe_duration.as_secs_f64()
            };
            if self.first_probe_done && self.phase(s.now) >= 0.5 * duration {
                self.probe_dq_sum += dq;
                self.probe_acks += 1;
            }
        } else if self.probe_acks > 0 {
            // End of an aligned probe: every flow of the host was at low load, so the mean dq of
            // its second half is the jitter floor. The first aligned probe replaces the estimate
            // of the unaligned first probe, so that all flows hold values from the same probes.
            let probe_dq = self.probe_dq_sum / self.probe_acks as f64;
            let fresh = !self.aligned_probe_seen || self.reset_next;
            let old = self.est.jitter;
            let j = match (fresh, old) {
                (false, Some(j)) => j.min(probe_dq),
                _ => probe_dq,
            };
            if self.config.adaptive_probes {
                // Stable once `stable_probes` probes in a row leave the minimum where it was
                let (abs, rel) = self.config.stable_tolerance;
                let tol = |x: f64| abs.as_secs_f64().max(rel * x);
                match old {
                    // A settled estimate whose fallback probe lands well above every probe seen
                    // while settling (1.5 times the highest): the link got worse, and the minimum
                    // alone could never grow back. Single probes scatter on a jittery link, so a
                    // margin within that range would restart the estimate for nothing.
                    Some(_)
                        if !fresh
                            && self.stable
                            && probe_dq
                                > (1.5 * self.probe_hi).max(self.probe_hi + tol(self.probe_hi)) =>
                    {
                        self.unsettle();
                    }
                    Some(old) if !fresh && old - j <= tol(old) => self.agreeing += 1,
                    _ => self.agreeing = 0,
                }
                if fresh {
                    // This probe started the new estimate
                    self.reset_next = false;
                    self.probe_hi = probe_dq;
                } else if !self.reset_next {
                    self.probe_hi = self.probe_hi.max(probe_dq);
                }
                let was_stable = self.stable;
                self.stable =
                    !self.reset_next && self.agreeing + 1 >= self.config.stable_probes.max(1);
                if was_stable && self.stable {
                    // A fallback probe agreed: wait twice as long for the next one
                    self.fallback_level = (self.fallback_level + 1).min(self.max_fallback_level());
                }
                self.rtt_min_at_probe = Some(s.rtt_min.as_secs_f64());
            }
            self.est.jitter = Some(j.min(MAX_JITTER));
            self.aligned_probe_seen = true;
            tracing::debug!(probe_dq, jitter = j, "copa-auto aligned probe");
            self.probe_dq_sum = 0.0;
            self.probe_acks = 0;
        }

        self.round.dq_sum += dq;
        self.round.acks += 1;
        // The packet found the bottleneck idle: its RTT is the propagation delay plus jitter
        if let Some(j) = self.est.jitter {
            let pkt_delay = s.rtt.as_secs_f64() - s.rtt_min.as_secs_f64();
            if pkt_delay <= j + self.config.idle_margin.as_secs_f64() {
                self.round.idle += 1;
            }
        }

        let round_time = s.now.saturating_duration_since(round_start);
        if round_time < s.srtt || self.round.acks < 2 {
            return;
        }
        if !s.slow_start {
            self.end_round(&s, delta);
        }
        self.round_start = Some(s.now);
        self.round = Accumulator::default();
    }

    fn end_round(&mut self, s: &Sample, delta: f64) {
        let cfg = &self.config;
        let half_rounds = u64::from(cfg.half_rounds.max(1));
        let probe_every = u64::from(cfg.probe_every.max(1));
        let probe = self.in_round_probe();
        // The first half-period of a low-load probe only lets Copa settle
        let settling = probe && self.half_index().is_multiple_of(probe_every);
        if !settling && self.rounds % half_rounds >= u64::from(cfg.settle_rounds) {
            let round = self.round;
            self.seg.add(&round);
        }
        self.rounds += 1;
        if !self.rounds.is_multiple_of(half_rounds) || settling {
            return;
        }

        // A half-period with an outage (handover, radio link failure), in the competitive mode
        // or limited by the application is not an equilibrium point
        let srtt = s.srtt.as_secs_f64();
        let settling_after_probe = self.settle_halves > 0 && !probe;
        if settling_after_probe {
            self.settle_halves -= 1;
        }
        let valid = self.seg.acks > 0
            && !settling_after_probe
            && !s.competitive
            && !self.seg_probe
            && !self.seg_app_limited
            && self.seg.max_gap < srtt.max(0.05);
        let seg_dq = if valid {
            self.seg.dq_sum / self.seg.acks as f64
        } else {
            0.0
        };

        if self.aligned() && !self.first_probe_done {
            // The first probe lasts until dq stops falling (at most eight half-periods): earlier
            // samples still hold the slow-start queue
            if (valid && self.last_probe_dq > 0.0 && seg_dq >= 0.9 * self.last_probe_dq)
                || self.half_index() >= 8
            {
                self.first_probe_done = true;
                self.settle_halves = 1;
            }
            if valid {
                self.last_probe_dq = seg_dq;
            }
        }

        if valid && probe {
            // Jitter floor: minimum over the low-load probes. Other flows keep the queue busy
            // during a probe, so a probe can only overestimate it; the minimum cannot drift up.
            let j = self.est.jitter.map_or(seg_dq, |j| j.min(seg_dq));
            self.est.jitter = Some(j.min(MAX_JITTER));
        } else if let (true, Some(j)) = (valid, self.est.jitter) {
            self.update_delta(s, seg_dq, j, delta);
        }

        self.seg = Accumulator::default();
        self.seg_probe = false;
        self.seg_app_limited = false;
    }

    fn update_delta(&mut self, s: &Sample, seg_dq: f64, j: f64, delta: f64) {
        let cfg = &self.config;
        // eq:rho-est: total utilization from the idle fraction
        let rho = (1.0 - self.seg.idle as f64 / self.seg.acks as f64).clamp(0.05, 0.98);
        // eq:k-est: k = c / mu from dq = j + k rho / (1 - rho)
        let queueing = (seg_dq - j).max(1e-5);
        let k = queueing * (1.0 - rho) / rho;
        // eq:k-ewma, in the log domain
        let ratio = if self.est.ratio <= 0.0 {
            k
        } else {
            ((1.0 - cfg.weight) * self.est.ratio.ln() + cfg.weight * k.ln()).exp()
        };
        // eq:dq-target
        let base_rtt = s.rtt_min.as_secs_f64() + j;
        let dq_target = j + (ratio * base_rtt).sqrt();

        // eq:delta-update. At equilibrium delta* = delta dq / dq*. Large steps far from the
        // target, small ones near it where measurement noise dominates.
        let base = self.est.delta.unwrap_or(delta);
        // A half-period with an empty queue would give ln(0); and one far off the target (Copa
        // still moving after a probe) must not throw delta to its bound in one step
        let err = (seg_dq.max(1e-5) / dq_target)
            .ln()
            .clamp(-MAX_STEP, MAX_STEP);
        let gain = if err.abs() > cfg.far_ratio.ln() {
            cfg.gain_far
        } else {
            cfg.gain
        };
        let mut next = base * (gain * err).exp();
        if cfg.fair_increase > 0.0 && err < 0.0 {
            // dq below the target: the same additive step of 1/delta for every flow, so that
            // the differences between flows shrink with each multiplicative decrease
            next = 1.0 / (1.0 / base - cfg.fair_increase * err);
        }
        let next = next.clamp(cfg.delta_min, cfg.delta_max);

        self.est.rho = rho;
        self.est.ratio = ratio;
        self.est.base_rtt = base_rtt;
        self.est.dq_target = dq_target;
        self.est.delta = Some(next);
        tracing::debug!(
            dq = seg_dq,
            rho,
            jitter = j,
            k = ratio,
            dq_target,
            delta = next,
            "copa-auto half-period"
        );
    }

    #[cfg(test)]
    pub(super) fn rounds(&self) -> u64 {
        self.rounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// Round-based probes (no wall-clock alignment), so tests are deterministic
    fn round_based() -> Arc<CopaAutoConfig> {
        let mut c = CopaAutoConfig::default();
        c.probe_timing(Duration::ZERO, Duration::ZERO);
        Arc::new(c)
    }

    fn sample(now: Instant, rtt: Duration, dq: Duration) -> Sample {
        Sample {
            now,
            rtt,
            rtt_min: ms(40),
            dq,
            srtt: ms(10),
            slow_start: false,
            competitive: false,
            app_limited: false,
            excluded: false,
        }
    }

    /// Feed one ACK per millisecond for `duration`; `idle(i)` says whether ACK `i` found the
    /// bottleneck idle
    fn feed(
        a: &mut AutoDelta,
        t: &mut Instant,
        duration: Duration,
        dq: Duration,
        idle: impl Fn(u64) -> bool,
        tweak: impl Fn(&mut Sample),
    ) {
        let end = *t + duration;
        let mut i = 0;
        while *t < end {
            let extra = if idle(i) { Duration::ZERO } else { dq + ms(5) };
            let mut s = sample(*t, ms(40) + extra, dq);
            tweak(&mut s);
            let delta = a.delta(*t, s.slow_start, 0.5);
            a.on_ack(s, delta);
            *t += ms(1);
            i += 1;
        }
    }

    #[test]
    fn jitter_from_first_probe_then_rho_k_and_target() {
        let mut a = AutoDelta::new(round_based(), Instant::now());
        let mut t = Instant::now();
        // Half-periods 0 (settling) and 1 (measured) of the first probe: dq = 2 ms. Rounds are
        // 10 ms long; the ACK at 160 ms closes the 16th round.
        feed(&mut a, &mut t, ms(161), ms(2), |_| false, |_| {});
        let est = a.estimate();
        let j = est.jitter.expect("jitter after the first probe");
        assert!((j - 0.002).abs() < 1e-9, "j = {j}");
        assert_eq!(est.delta, None);
        assert_eq!(a.delta(t, false, 0.5), 0.5);

        // Next half-period: dq = 8 ms, 30% of the packets find the queue idle
        feed(&mut a, &mut t, ms(80), ms(8), |i| i % 10 < 3, |_| {});
        let est = a.estimate();
        assert!((est.rho - 0.7).abs() < 0.05, "rho = {}", est.rho);
        // eq:k-est with the estimated rho; the first sample initializes the average
        let k = (0.008 - j) * (1.0 - est.rho) / est.rho;
        assert!((est.ratio - k).abs() < 1e-12);
        // eq:dq-target with R0' = RTT_min + j
        assert!((est.base_rtt - (0.040 + j)).abs() < 1e-12);
        assert!((est.dq_target - (j + (k * (0.040 + j)).sqrt())).abs() < 1e-12);
        assert!(est.delta.is_some());
    }

    #[test]
    fn probes_use_delta_max_and_slow_start_uses_fallback() {
        let mut c = CopaAutoConfig::default();
        c.probe_timing(Duration::ZERO, Duration::ZERO)
            .delta_range(0.01, 0.4);
        let a = AutoDelta::new(Arc::new(c), Instant::now());
        let now = Instant::now();
        assert_eq!(a.delta(now, true, 0.5), 0.5);
        assert_eq!(a.delta(now, false, 0.5), 0.4);
    }

    /// 98% utilization
    fn with_jitter(j: f64) -> (AutoDelta, Sample) {
        let mut a = AutoDelta::new(round_based(), Instant::now());
        a.est.jitter = Some(j);
        a.seg = Accumulator {
            dq_sum: 0.0,
            acks: 100,
            idle: 2,
            max_gap: 0.0,
        };
        (a, sample(Instant::now(), ms(40), ms(0)))
    }

    #[test]
    fn multiplicative_decrease_when_above_target() {
        let (mut a, s) = with_jitter(0.001);
        a.update_delta(&s, 0.004, 0.001, 0.1);
        let est = a.estimate();
        assert!(est.dq_target < 0.004);
        let expected = 0.1 * (0.004 / est.dq_target).powf(0.25);
        assert!((est.delta.unwrap() - expected).abs() < 1e-12);
        assert!(est.delta.unwrap() > 0.1);
    }

    #[test]
    fn additive_increase_of_inverse_delta_when_below_target() {
        let (mut a, s) = with_jitter(0.001);
        a.update_delta(&s, 0.0011, 0.001, 0.1);
        let est = a.estimate();
        assert!(est.dq_target > 0.0011);
        let expected = 1.0 / (10.0 + 10.0 * (est.dq_target / 0.0011).ln());
        assert!((est.delta.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn delta_is_clamped() {
        let (mut a, s) = with_jitter(0.001);
        // Far below target, from a small delta: 1/delta would exceed 1/delta_min
        a.est.delta = Some(0.00101);
        a.update_delta(&s, 0.0001, 0.001, 0.5);
        assert_eq!(a.estimate().delta, Some(0.001));
        let (mut a, s) = with_jitter(0.001);
        a.update_delta(&s, 10.0, 0.001, 0.5);
        assert_eq!(a.estimate().delta, Some(0.5));
    }

    #[test]
    fn aimd_on_inverse_delta_brings_flows_together() {
        // Two flows see the same dq / dq*, first below the target then above
        let (mut a, s) = with_jitter(0.001);
        let (mut b, _) = with_jitter(0.001);
        a.est.delta = Some(1.0 / 4.0);
        b.est.delta = Some(1.0 / 40.0);
        let ratio = |a: &AutoDelta, b: &AutoDelta| a.est.delta.unwrap() / b.est.delta.unwrap();
        let before = ratio(&a, &b);
        for dq in [0.0012, 0.01, 0.0012, 0.01] {
            a.seg = Accumulator {
                dq_sum: 0.0,
                acks: 100,
                idle: 2,
                max_gap: 0.0,
            };
            b.seg = a.seg;
            a.update_delta(&s, dq, 0.001, 0.5);
            b.update_delta(&s, dq, 0.001, 0.5);
        }
        let after = ratio(&a, &b);
        assert!(after < before / 2.0, "ratio {before} -> {after}");
    }

    #[test]
    fn invalid_half_periods_do_not_update_delta() {
        for (name, tweak) in [
            (
                "competitive",
                (|s: &mut Sample| s.competitive = true) as fn(&mut Sample),
            ),
            ("app limited", |s: &mut Sample| s.app_limited = true),
        ] {
            let mut a = AutoDelta::new(round_based(), Instant::now());
            let mut t = Instant::now();
            feed(&mut a, &mut t, ms(161), ms(2), |_| false, |_| {});
            assert!(a.estimate().jitter.is_some());
            feed(&mut a, &mut t, ms(400), ms(8), |i| i % 2 == 0, tweak);
            assert_eq!(a.estimate().delta, None, "{name}");
        }
        // An outage (no ACK for longer than max(50 ms, srtt)) invalidates its half-period
        let mut a = AutoDelta::new(round_based(), Instant::now());
        let mut t = Instant::now();
        feed(&mut a, &mut t, ms(161), ms(2), |_| false, |_| {});
        feed(&mut a, &mut t, ms(30), ms(8), |i| i % 2 == 0, |_| {});
        t += ms(60);
        feed(&mut a, &mut t, ms(60), ms(8), |i| i % 2 == 0, |_| {});
        assert_eq!(a.estimate().delta, None);
    }

    #[test]
    fn slow_start_does_not_count_rounds() {
        let mut a = AutoDelta::new(round_based(), Instant::now());
        let mut t = Instant::now();
        feed(
            &mut a,
            &mut t,
            ms(200),
            ms(2),
            |_| false,
            |s| s.slow_start = true,
        );
        assert_eq!(a.rounds(), 0);
        assert_eq!(a.estimate().jitter, None);
    }

    fn aligned(epoch: Instant) -> Arc<CopaAutoConfig> {
        let mut c = CopaAutoConfig::default();
        c.probe_epoch(Some(epoch));
        Arc::new(c)
    }

    #[test]
    fn probes_are_aligned_on_the_epoch() {
        let epoch = Instant::now();
        let a = AutoDelta::new(aligned(epoch), epoch + ms(1234));
        let b = AutoDelta::new(aligned(epoch), epoch + Duration::from_secs(7));
        for offset in [0u64, 500, 1999, 2000, 2500, 19_999, 20_000, 21_000, 43_000] {
            let now = epoch + Duration::from_secs(8) + ms(offset);
            assert!((a.phase(now) - b.phase(now)).abs() < 1e-9);
            let in_probe = (8000 + offset) % 20_000 < 2000;
            assert_eq!(a.in_timed_probe(now, false), in_probe, "offset {offset}");
            assert!(!a.in_timed_probe(now, true));
        }
    }

    /// Feed one aligned probe (phases 0..2.2 s of the period starting at `period_start`)
    fn aligned_probe(a: &mut AutoDelta, period_start: Instant, dq: Duration) {
        let mut t = period_start;
        while t < period_start + ms(2200) {
            a.on_ack(sample(t, ms(40) + dq, dq), 0.5);
            t += ms(1);
        }
    }

    #[test]
    fn jitter_floor_is_a_minimum_over_aligned_probes() {
        let epoch = Instant::now();
        let mut a = AutoDelta::new(aligned(epoch), epoch);
        a.first_probe_done = true;
        // An unaligned estimate is replaced by the first aligned probe, even if larger
        a.est.jitter = Some(0.001);
        aligned_probe(&mut a, epoch, ms(5));
        assert!((a.estimate().jitter.unwrap() - 0.005).abs() < 1e-9);
        aligned_probe(&mut a, epoch + Duration::from_secs(20), ms(8));
        assert!((a.estimate().jitter.unwrap() - 0.005).abs() < 1e-9);
        aligned_probe(&mut a, epoch + Duration::from_secs(40), ms(3));
        assert!((a.estimate().jitter.unwrap() - 0.003).abs() < 1e-9);

        let mut b = AutoDelta::new(aligned(epoch), epoch);
        b.first_probe_done = true;
        aligned_probe(&mut b, epoch, ms(50));
        assert_eq!(b.estimate().jitter, Some(MAX_JITTER));
    }

    #[test]
    fn first_aligned_probe_ends_when_dq_stops_falling() {
        let epoch = Instant::now();
        // Start after the aligned probe window so that only the round-based first probe runs
        let start = epoch + Duration::from_secs(3);
        let mut a = AutoDelta::new(aligned(epoch), start);
        let mut t = start;
        assert!(!a.first_probe_done());
        // dq falls from 20 ms by 30% per half-period (80 ms), then stays flat
        while t < start + ms(2000) {
            let half = t.saturating_duration_since(start).as_millis() as i32 / 80;
            let dq = Duration::from_secs_f64(0.020 * 0.7f64.powi(half.min(3)));
            a.on_ack(sample(t, ms(40) + dq, dq), 0.5);
            t += ms(1);
        }
        assert!(a.first_probe_done());
        assert!(a.estimate().jitter.is_some());
    }

    /// Feed one aligned period (20 s, one ACK per millisecond) starting at `start`, with the given
    /// dq during the probe window and RTT_min; returns whether its slot probed and drained
    fn period(a: &mut AutoDelta, start: Instant, dq: Duration, rtt_min: Duration) -> (bool, bool) {
        let mut t = start;
        let (mut probed, mut drained) = (false, false);
        while t < start + Duration::from_secs(20) {
            let mut s = sample(t, rtt_min + dq, dq);
            s.rtt_min = rtt_min;
            a.on_ack(s, 0.5);
            probed |= a.in_timed_probe(t, false);
            drained |= a.drain_slot(t, false).is_some();
            t += ms(1);
        }
        (probed, drained)
    }

    fn settled(epoch: Instant) -> AutoDelta {
        settled_with(aligned(epoch), epoch)
    }

    fn settled_with(config: Arc<CopaAutoConfig>, epoch: Instant) -> AutoDelta {
        let mut a = AutoDelta::new(config, epoch);
        a.first_probe_done = true;
        // Three probes leave the minimum where it was: j is settled
        for (i, dq) in [5.0, 5.2, 5.1].into_iter().enumerate() {
            let start = epoch + Duration::from_secs(20 * i as u64);
            let (probed, _) = period(&mut a, start, Duration::from_secs_f64(dq / 1e3), ms(40));
            assert!(probed, "probe {i}");
        }
        assert!(a.stable);
        a
    }

    #[test]
    fn adaptive_probes_stop_once_the_floor_settles_and_drain_instead() {
        let epoch = Instant::now();
        let mut a = settled(epoch);
        assert!((a.estimate().jitter.unwrap() - 0.005).abs() < 1e-9);
        // Slots 3, 4, 5 do not probe (delta stays as chosen) but drain; slot 6 is a 120 s
        // fallback slot and probes
        for slot in 3..6u64 {
            let (probed, drained) = period(
                &mut a,
                epoch + Duration::from_secs(20 * slot),
                ms(5),
                ms(40),
            );
            assert!(!probed && drained, "slot {slot}");
            assert_ne!(
                a.delta(epoch + Duration::from_secs(20 * slot) + ms(500), false, 0.1),
                0.5
            );
        }
        let (probed, drained) = period(&mut a, epoch + Duration::from_secs(120), ms(5), ms(40));
        assert!(probed && !drained);
        assert!(a.stable);
    }

    #[test]
    fn periodic_schedule_probes_in_every_slot() {
        let epoch = Instant::now();
        let mut c = CopaAutoConfig::default();
        c.probe_epoch(Some(epoch)).periodic_probes();
        let mut a = AutoDelta::new(Arc::new(c), epoch);
        a.first_probe_done = true;
        for slot in 0..8u64 {
            let (probed, drained) = period(
                &mut a,
                epoch + Duration::from_secs(20 * slot),
                ms(5),
                ms(40),
            );
            assert!(probed && !drained, "slot {slot}");
        }
    }

    #[test]
    fn rtt_min_change_restarts_the_estimate_and_j_can_grow() {
        let epoch = Instant::now();
        let mut a = settled(epoch);
        // The path changes: RTT_min 40 -> 70 ms, and the floor is now 9 ms
        let (probed, _) = period(&mut a, epoch + Duration::from_secs(60), ms(9), ms(70));
        assert!(probed, "a notable RTT_min change probes in the next slot");
        assert!(
            (a.estimate().jitter.unwrap() - 0.009).abs() < 1e-9,
            "the new estimate replaces j instead of keeping the old minimum"
        );
        assert!(!a.stable);
    }

    #[test]
    fn fallback_probe_well_above_the_settled_range_restarts_the_estimate() {
        let epoch = Instant::now();
        let mut a = settled(epoch);
        for slot in 3..6u64 {
            period(
                &mut a,
                epoch + Duration::from_secs(20 * slot),
                ms(5),
                ms(40),
            );
        }
        // A fallback probe within the scatter seen while settling keeps the estimate
        let mut b = a.clone();
        period(&mut b, epoch + Duration::from_secs(120), ms(6), ms(40));
        assert!(b.stable);
        // One at more than 1.5 times the highest settling probe: the floor rose
        period(&mut a, epoch + Duration::from_secs(120), ms(12), ms(40));
        assert!(!a.stable);
        let (probed, _) = period(&mut a, epoch + Duration::from_secs(140), ms(12), ms(40));
        assert!(probed);
        assert!(
            (a.estimate().jitter.unwrap() - 0.012).abs() < 1e-9,
            "j grew to the new floor"
        );
    }

    #[test]
    fn flows_sharing_a_board_probe_together_with_the_longest_length() {
        let epoch = Instant::now();
        let config = aligned(epoch);
        // A settled flow and a new one built from the same configuration
        let mut old = settled_with(config.clone(), epoch);
        let mut new = AutoDelta::new(config, epoch);
        new.first_probe_done = true;
        let start = epoch + Duration::from_secs(60);
        let mut t = start;
        let (mut old_probed, mut new_probed) = (false, false);
        while t < start + Duration::from_secs(40) {
            let dq = ms(5);
            new.on_ack(sample(t, ms(40) + dq, dq), 0.5);
            let mut s = sample(t, ms(40) + dq, dq);
            s.srtt = ms(100);
            old.on_ack(s, 0.5);
            if t >= start + Duration::from_secs(20) {
                old_probed |= old.in_timed_probe(t, false);
                new_probed |= new.in_timed_probe(t, false);
            }
            t += ms(1);
        }
        // The new flow registered slot 4 during slot 3; the settled one joined it
        assert!(new_probed && old_probed);
        // The settled flow needs 16 x 100 ms, longer than the new flow's 300 ms floor
        assert!(
            (old.slot_duration - 1.6).abs() < 1e-9,
            "{}",
            old.slot_duration
        );
        assert!(
            (new.slot_duration - 1.6).abs() < 1e-9,
            "{}",
            new.slot_duration
        );
    }

    #[test]
    fn probe_length_follows_the_rtt_within_bounds() {
        let a = AutoDelta::new(aligned(Instant::now()), Instant::now());
        assert_eq!(a.wanted_duration(ms(10)), ms(300));
        assert_eq!(a.wanted_duration(ms(50)), ms(800));
        assert_eq!(a.wanted_duration(ms(500)), ms(2000));
        let mut c = CopaAutoConfig::default();
        c.probe_rtts(0, Duration::ZERO);
        let fixed = AutoDelta::new(Arc::new(c), Instant::now());
        assert_eq!(fixed.wanted_duration(ms(10)), ms(2000));
    }

    #[test]
    fn fallback_probes_back_off_while_they_agree() {
        let epoch = Instant::now();
        let mut a = settled(epoch);
        let mut probed = vec![];
        for slot in 3..32u64 {
            let (p, _) = period(
                &mut a,
                epoch + Duration::from_secs(20 * slot),
                ms(5),
                ms(40),
            );
            if p {
                probed.push(slot);
            }
        }
        // 120 s (slot 6), then 240 s (12), 480 s (24), capped at 600 s (30, also a forced slot)
        assert_eq!(probed, vec![6, 12, 24, 30]);
        assert!(a.stable);
        // A restart goes back to the shortest period
        a.unsettle();
        assert_eq!(a.fallback_level, 0);
    }

    #[test]
    fn excluded_samples_invalidate_their_half_period() {
        let mut a = AutoDelta::new(round_based(), Instant::now());
        let mut t = Instant::now();
        feed(&mut a, &mut t, ms(161), ms(2), |_| false, |_| {});
        assert!(a.estimate().jitter.is_some());
        feed(
            &mut a,
            &mut t,
            ms(400),
            ms(8),
            |i| i % 2 == 0,
            |s| s.excluded = true,
        );
        assert_eq!(a.estimate().delta, None);
    }

    #[test]
    fn one_update_moves_delta_by_a_bounded_step() {
        let (mut a, s) = with_jitter(0.001);
        // An empty queue (dq = 0) would otherwise throw 1/delta to infinity
        a.update_delta(&s, 0.0, 0.001, 0.1);
        let inv = 1.0 / a.estimate().delta.unwrap();
        assert!(
            (inv - (10.0 + 10.0 * MAX_STEP)).abs() < 1e-9,
            "1/delta = {inv}"
        );
    }

    #[test]
    fn fixed_jitter_floor_skips_probes_and_respects_drain_toggle() {
        let epoch = Instant::now();
        for (value_ms, expected_j, drain_enabled) in
            [(0.0, 0.0, true), (10.0, 0.01, true), (10.0, 0.01, false)]
        {
            let mut c = CopaAutoConfig::default();
            c.probe_epoch(Some(epoch))
                .jitter_floor_ms(value_ms)
                .drain_enabled(drain_enabled);
            let mut a = AutoDelta::new(Arc::new(c), epoch);
            assert!(a.first_probe_done());
            assert_eq!(a.estimate().jitter, Some(expected_j));
            // Even across a forced 600 s slot and an RTT_min jump, no j probe runs
            for (slot, rtt_min) in [(0u64, ms(40)), (1, ms(70)), (30, ms(70))] {
                let (probed, drained) =
                    period(&mut a, epoch + Duration::from_secs(20 * slot), ms(12), rtt_min);
                assert!(!probed, "value_ms={value_ms}, slot={slot}");
                assert_eq!(drained, drain_enabled, "value_ms={value_ms}, slot={slot}");
            }
            assert_eq!(a.estimate().jitter, Some(expected_j));
            assert!((a.estimate().base_rtt - (0.070 + expected_j)).abs() < 1e-9);
            assert!(a.estimate().delta.is_some());
        }
    }
}
