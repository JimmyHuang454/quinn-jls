//! A small deterministic discrete-event simulator of flows sharing one bottleneck
//!
//! It drives any [`Controller`] through the same calls as a quinn connection (`on_sent`,
//! `on_ack` per packet, `on_end_acks`, then the RTT estimator update, `on_congestion_event`), so
//! Copa can be compared with NewReno, Cubic or BBR. Time is virtual; a scenario of tens of
//! seconds runs in well under a second.
//!
//! Model: senders attach directly to a FIFO bottleneck of rate `mu` (piecewise constant over
//! time) and buffer `buffer` bytes (tail drop). After the bottleneck a packet takes half the base
//! RTT plus an optional random jitter to reach the receiver, which acknowledges every packet, or
//! every second packet with a 25 ms timer (delayed ACKs). ACKs take the other half of the base
//! RTT. A dropped packet is reported lost one base RTT plus the queueing delay after it was sent,
//! roughly when packet-threshold loss detection would notice it.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, VecDeque};
use std::sync::Arc;

use super::{Copa, CopaConfig};
use crate::congestion::{Controller, ControllerFactory};
use crate::connection::RttEstimator;
use crate::packet::SpaceId;
use crate::{Duration, Instant};

pub(super) const MTU: u64 = 1200;

/// How senders pace
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pacing {
    /// Use `ControllerMetrics::pacing_rate`, falling back to quinn's `1.25 * window / srtt` (what quinn
    /// does with the pacing hook)
    Controller,
    /// Always `1.25 * window / srtt` (quinn without the pacing hook)
    Quinn,
}

#[derive(Debug, Clone)]
pub(super) struct Link {
    /// Bottleneck rate in bits per second from the given offset on; the first entry must be at 0.
    /// A zero rate is an outage during which every packet is dropped.
    pub(super) rate: Vec<(Duration, f64)>,
    pub(super) base_rtt: Duration,
    pub(super) buffer: u64,
    /// Maximum extra one-way delay after the bottleneck, uniformly distributed
    pub(super) jitter: Duration,
    /// When set, the jitter is not drawn per packet but held for this long (piecewise
    /// constant), like a cellular scheduler that delays bursts of packets together; Copa's
    /// standing-RTT filter cannot remove such jitter
    pub(super) jitter_hold: Option<Duration>,
    pub(super) delayed_ack: bool,
    pub(super) pacing: Pacing,
    pub(super) seed: u64,
}

impl Link {
    pub(super) fn new(mbps: f64, base_rtt: Duration) -> Self {
        // Two bandwidth-delay products
        let bdp = mbps * 1e6 / 8.0 * base_rtt.as_secs_f64();
        Self {
            rate: vec![(Duration::ZERO, mbps * 1e6)],
            base_rtt,
            buffer: (2.0 * bdp).max(8.0 * MTU as f64) as u64,
            jitter: Duration::ZERO,
            jitter_hold: None,
            delayed_ack: false,
            pacing: Pacing::Controller,
            seed: 1,
        }
    }
}

pub(super) enum Source {
    Copa(Box<Copa>),
    Other(Box<dyn Controller>),
    /// Unresponsive constant bit rate traffic (bits per second), on and off periods
    Cbr {
        bps: f64,
        on: Duration,
        off: Duration,
    },
}

impl Source {
    pub(super) fn copa(config: CopaConfig) -> Self {
        Self::copa_at(config, Instant::now())
    }

    fn copa_at(config: CopaConfig, now: Instant) -> Self {
        Self::Copa(Box::new(Copa::new(Arc::new(config), now, MTU as u16)))
    }

    pub(super) fn other(factory: impl ControllerFactory + 'static) -> Self {
        Self::Other(Arc::new(factory).build(Instant::now(), MTU as u16))
    }

    fn cc(&mut self) -> Option<&mut dyn Controller> {
        match self {
            Self::Copa(c) => Some(&mut **c),
            Self::Other(c) => Some(&mut **c),
            Self::Cbr { .. } => None,
        }
    }

    pub(super) fn as_copa(&self) -> Option<&Copa> {
        match self {
            Self::Copa(c) => Some(c),
            _ => None,
        }
    }
}

pub(super) struct Flow {
    pub(super) source: Source,
    pub(super) start: Duration,
}

/// Measurements over `[warmup, duration)`
#[derive(Debug, Clone)]
pub(super) struct Report {
    /// Goodput of each flow in bits per second
    pub(super) throughput: Vec<f64>,
    /// Mean queueing delay at the bottleneck (seconds) seen by each flow's packets
    pub(super) queueing: Vec<f64>,
    /// Mean of Copa's own `d_q` estimate (seconds), sampled per ACK; 0 for other flows
    pub(super) copa_dq: Vec<f64>,
    /// Mean delta of each Copa flow, sampled per ACK
    pub(super) copa_delta: Vec<f64>,
    /// Fraction of ACKs during which each Copa flow was in the competitive mode
    pub(super) competitive: Vec<f64>,
    /// Packets dropped at the bottleneck, per flow
    pub(super) drops: Vec<u64>,
    /// Delivered bits over the capacity offered during the window
    pub(super) utilization: f64,
    /// Mean queueing delay over all packets (seconds)
    pub(super) mean_queueing: f64,
    pub(super) max_window: Vec<u64>,
    /// Persistent congestion events signalled to each flow
    pub(super) persistent: Vec<u64>,
    /// Goodput of each flow in each second of the window, bits per second
    pub(super) per_second: Vec<Vec<f64>>,
}

impl Report {
    /// Jain's fairness index over the given flows
    /// Throughput over base RTT plus mean queueing delay (Kleinrock's power, in bits/s/s)
    pub(super) fn power(&self, i: usize, base_rtt: Duration) -> f64 {
        self.throughput[i] / (base_rtt.as_secs_f64() + self.queueing[i])
    }

    pub(super) fn jain(&self, flows: impl IntoIterator<Item = usize>) -> f64 {
        let x: Vec<f64> = flows.into_iter().map(|i| self.throughput[i]).collect();
        let sum: f64 = x.iter().sum();
        let sq: f64 = x.iter().map(|v| v * v).sum();
        sum * sum / (x.len() as f64 * sq)
    }
}

#[derive(Debug)]
enum Event {
    Wake(usize),
    Deliver {
        flow: usize,
        pn: u64,
    },
    AckTimer {
        flow: usize,
        generation: u64,
    },
    Ack {
        flow: usize,
        pns: Vec<u64>,
        ack_delay: Duration,
    },
    Lost {
        flow: usize,
        pn: u64,
    },
}

#[derive(Default)]
struct Sender {
    /// Bytes acknowledged in each second of the measurement window
    per_second: Vec<u64>,
    rtt: Option<RttEstimator>,
    next_pn: u64,
    in_flight: u64,
    sent: BTreeMap<u64, (Instant, u64)>,
    next_send: Option<Instant>,
    wake_at: Option<Instant>,
    /// First loss since the last ACK, for persistent congestion
    loss_run_start: Option<Instant>,
    // Receiver
    pending: Vec<u64>,
    pending_since: Option<Instant>,
    generation: u64,
    // Statistics
    delivered: u64,
    queueing_sum: f64,
    queueing_n: u64,
    dq_sum: f64,
    delta_sum: f64,
    competitive_n: u64,
    acks_n: u64,
    drops: u64,
    persistent: u64,
    max_window: u64,
}

struct Bottleneck {
    last_departure: Instant,
    /// (departure time, size) of queued packets
    queue: VecDeque<(Instant, u64)>,
    queued: u64,
}

pub(super) struct Sim {
    pub(super) t0: Instant,
    now: Instant,
    link: Link,
    pub(super) flows: Vec<Flow>,
    senders: Vec<Sender>,
    events: BinaryHeap<Reverse<(Instant, u64)>>,
    payloads: HashMap<u64, Event>,
    seq: u64,
    bottleneck: Bottleneck,
    rng: u64,
    warmup: Duration,
    end: Instant,
    capacity_bits: f64,
    delivered_bits: f64,
}

impl Sim {
    pub(super) fn new(link: Link, flows: Vec<Flow>) -> Self {
        let t0 = Instant::now();
        let n = flows.len();
        let rng = link.seed.max(1);
        Self {
            t0,
            now: t0,
            link,
            flows,
            senders: (0..n).map(|_| Sender::default()).collect(),
            events: BinaryHeap::new(),
            payloads: HashMap::new(),
            seq: 0,
            bottleneck: Bottleneck {
                last_departure: t0,
                queue: VecDeque::new(),
                queued: 0,
            },
            rng,
            warmup: Duration::ZERO,
            end: t0,
            capacity_bits: 0.0,
            delivered_bits: 0.0,
        }
    }

    /// The epoch to give copa-auto's probes so that they are deterministic
    pub(super) fn epoch(&self) -> Instant {
        self.t0
    }

    fn schedule(&mut self, at: Instant, event: Event) {
        self.seq += 1;
        self.events.push(Reverse((at, self.seq)));
        self.payloads.insert(self.seq, event);
    }

    fn random(&mut self) -> f64 {
        // xorshift64
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 11) as f64 / (1u64 << 53) as f64
    }

    fn rate_at(&self, t: Instant) -> f64 {
        let offset = t.saturating_duration_since(self.t0);
        self.link
            .rate
            .iter()
            .rev()
            .find(|(start, _)| *start <= offset)
            .map_or(self.link.rate[0].1, |&(_, r)| r)
    }

    fn measuring(&self, t: Instant) -> bool {
        t.saturating_duration_since(self.t0) >= self.warmup
    }

    /// Run until `duration`, measuring after `warmup`
    pub(super) fn run(&mut self, duration: Duration, warmup: Duration) -> Report {
        self.warmup = warmup;
        let end = self.t0 + duration;
        self.end = end;
        for i in 0..self.flows.len() {
            let start = self.t0 + self.flows[i].start;
            self.schedule(start, Event::Wake(i));
        }
        // Offered capacity, integrated over the measurement window
        let step = Duration::from_millis(1);
        let mut t = self.t0 + warmup;
        while t < end {
            self.capacity_bits += self.rate_at(t) * step.as_secs_f64();
            t += step;
        }

        while let Some(Reverse((at, seq))) = self.events.pop() {
            if at >= end {
                break;
            }
            self.now = at;
            let event = self.payloads.remove(&seq).unwrap();
            match event {
                Event::Wake(i) => {
                    if self.senders[i].wake_at == Some(at) || self.senders[i].wake_at.is_none() {
                        self.senders[i].wake_at = None;
                        self.try_send(i);
                    }
                }
                Event::Deliver { flow, pn } => self.on_deliver(flow, pn),
                Event::AckTimer { flow, generation } => {
                    if self.senders[flow].generation == generation {
                        self.flush_acks(flow);
                    }
                }
                Event::Ack {
                    flow,
                    pns,
                    ack_delay,
                } => self.on_ack(flow, pns, ack_delay),
                Event::Lost { flow, pn } => self.on_lost(flow, pn),
            }
        }
        self.report(duration, warmup)
    }

    fn report(&self, duration: Duration, warmup: Duration) -> Report {
        let window = (duration - warmup).as_secs_f64();
        let mean = |sum: f64, n: u64| if n == 0 { 0.0 } else { sum / n as f64 };
        let s = &self.senders;
        let total_q: f64 = s.iter().map(|s| s.queueing_sum).sum();
        let total_n: u64 = s.iter().map(|s| s.queueing_n).sum();
        Report {
            throughput: s
                .iter()
                .map(|s| s.delivered as f64 * 8.0 / window)
                .collect(),
            queueing: s
                .iter()
                .map(|s| mean(s.queueing_sum, s.queueing_n))
                .collect(),
            copa_dq: s.iter().map(|s| mean(s.dq_sum, s.acks_n)).collect(),
            copa_delta: s.iter().map(|s| mean(s.delta_sum, s.acks_n)).collect(),
            competitive: s
                .iter()
                .map(|s| mean(s.competitive_n as f64, s.acks_n))
                .collect(),
            drops: s.iter().map(|s| s.drops).collect(),
            utilization: self.delivered_bits / self.capacity_bits,
            mean_queueing: mean(total_q, total_n),
            max_window: s.iter().map(|s| s.max_window).collect(),
            persistent: s.iter().map(|s| s.persistent).collect(),
            per_second: s
                .iter()
                .map(|s| {
                    let n = window.floor() as usize;
                    (0..n)
                        .map(|i| s.per_second.get(i).copied().unwrap_or(0) as f64 * 8.0)
                        .collect()
                })
                .collect(),
        }
    }

    fn try_send(&mut self, i: usize) {
        let now = self.now;
        if let Source::Cbr { bps, on, off } = self.flows[i].source {
            let cycle = (on + off).as_secs_f64();
            let phase = now
                .saturating_duration_since(self.t0 + self.flows[i].start)
                .as_secs_f64()
                % cycle;
            if phase < on.as_secs_f64() {
                self.transmit(i, MTU);
                let next = now + Duration::from_secs_f64(MTU as f64 * 8.0 / bps);
                self.schedule(next, Event::Wake(i));
            } else {
                let next = now + Duration::from_secs_f64(cycle - phase);
                self.schedule(next, Event::Wake(i));
            }
            return;
        }

        let base_rtt = self.link.base_rtt;
        loop {
            let pacing = self.link.pacing;
            let sender = &mut self.senders[i];
            let rtt = *sender
                .rtt
                .get_or_insert_with(|| RttEstimator::new(base_rtt * 2));
            let cc = self.flows[i].source.cc().unwrap();
            let window = cc.window();
            if sender.in_flight + MTU > window {
                return;
            }
            if let Some(next) = sender.next_send.filter(|&next| next > now) {
                if sender.wake_at.is_none_or(|w| w > next) {
                    sender.wake_at = Some(next);
                    self.schedule(next, Event::Wake(i));
                }
                return;
            }
            let default_rate = 1.25 * window as f64 / rtt.get().as_secs_f64();
            let rate = match pacing {
                Pacing::Controller => cc.metrics().pacing_rate.map_or(default_rate, |r| r as f64),
                Pacing::Quinn => default_rate,
            };
            let pn = sender.next_pn;
            sender.next_pn += 1;
            sender.in_flight += MTU;
            sender.sent.insert(pn, (now, MTU));
            let gap = Duration::from_secs_f64(MTU as f64 / rate.max(1.0));
            sender.next_send = Some(sender.next_send.map_or(now, |n| n.max(now)) + gap);
            cc.on_sent(now, MTU, pn);
            self.transmit_packet(i, pn, MTU);
        }
    }

    fn transmit(&mut self, i: usize, size: u64) {
        let pn = self.senders[i].next_pn;
        self.senders[i].next_pn += 1;
        self.transmit_packet(i, pn, size);
    }

    /// Enqueue at the bottleneck, or drop
    fn transmit_packet(&mut self, i: usize, pn: u64, size: u64) {
        let now = self.now;
        let b = &mut self.bottleneck;
        while let Some(&(departure, s)) = b.queue.front() {
            if departure > now {
                break;
            }
            b.queue.pop_front();
            b.queued -= s;
        }
        let queue_delay = b.last_departure.saturating_duration_since(now);
        // A zero rate is an outage: the link drops everything
        let rate = self.rate_at(now);
        if rate <= 0.0 || self.bottleneck.queued + size > self.link.buffer {
            let is_cbr = matches!(self.flows[i].source, Source::Cbr { .. });
            self.senders[i].drops += 1;
            if !is_cbr {
                let at = now + self.link.base_rtt + queue_delay + Duration::from_millis(1);
                self.schedule(at, Event::Lost { flow: i, pn });
            }
            return;
        }
        let b = &mut self.bottleneck;
        let service = Duration::from_secs_f64(size as f64 * 8.0 / rate);
        let departure = b.last_departure.max(now) + service;
        b.last_departure = departure;
        b.queue.push_back((departure, size));
        b.queued += size;

        if self.measuring(departure) && departure < self.end {
            self.delivered_bits += size as f64 * 8.0;
            let is_cbr = matches!(self.flows[i].source, Source::Cbr { .. });
            let s = &mut self.senders[i];
            if is_cbr {
                s.delivered += size;
            }
            s.queueing_sum += queue_delay.as_secs_f64();
            s.queueing_n += 1;
        }
        if matches!(self.flows[i].source, Source::Cbr { .. }) {
            return;
        }
        let jitter = match self.link.jitter_hold {
            Some(hold) if !hold.is_zero() => {
                // Same value for every packet leaving in one hold slot
                let slot = departure.saturating_duration_since(self.t0).as_nanos()
                    / hold.as_nanos().max(1);
                let mut x = (slot as u64 ^ self.link.seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                x ^= x >> 33;
                x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
                x ^= x >> 33;
                self.link
                    .jitter
                    .mul_f64((x >> 11) as f64 / (1u64 << 53) as f64)
            }
            _ => self.link.jitter.mul_f64(self.random()),
        };
        let at = departure + self.link.base_rtt / 2 + jitter;
        self.schedule(at, Event::Deliver { flow: i, pn });
    }

    fn on_deliver(&mut self, i: usize, pn: u64) {
        let now = self.now;
        let s = &mut self.senders[i];
        s.pending.push(pn);
        if !self.link.delayed_ack || s.pending.len() >= 2 {
            self.flush_acks(i);
        } else if s.pending.len() == 1 {
            s.pending_since = Some(now);
            s.generation += 1;
            let generation = s.generation;
            self.schedule(
                now + Duration::from_millis(25),
                Event::AckTimer {
                    flow: i,
                    generation,
                },
            );
        }
    }

    fn flush_acks(&mut self, i: usize) {
        let now = self.now;
        let s = &mut self.senders[i];
        if s.pending.is_empty() {
            return;
        }
        let pns = std::mem::take(&mut s.pending);
        let ack_delay = s
            .pending_since
            .take()
            .map_or(Duration::ZERO, |t| now.saturating_duration_since(t));
        s.generation += 1;
        let at = now + self.link.base_rtt / 2;
        self.schedule(
            at,
            Event::Ack {
                flow: i,
                pns,
                ack_delay,
            },
        );
    }

    fn on_ack(&mut self, i: usize, pns: Vec<u64>, ack_delay: Duration) {
        let now = self.now;
        let measuring = self.measuring(now);
        let s = &mut self.senders[i];
        let rtt = s.rtt.expect("sent before acked");
        let cc = self.flows[i].source.cc().unwrap();
        let mut largest = None;
        for pn in pns {
            let Some((sent, size)) = s.sent.remove(&pn) else {
                continue;
            };
            s.in_flight -= size;
            cc.on_ack(now, sent, size, pn, SpaceId::Data, false, &rtt);
            if measuring {
                s.delivered += size;
                let sec = now
                    .saturating_duration_since(self.t0 + self.warmup)
                    .as_secs() as usize;
                if s.per_second.len() <= sec {
                    s.per_second.resize(sec + 1, 0);
                }
                s.per_second[sec] += size;
            }
            largest = Some((pn, sent));
        }
        let Some((largest, largest_sent)) = largest else {
            return;
        };
        cc.on_end_acks(now, s.in_flight, false, Some(largest), SpaceId::Data);
        let mut rtt = rtt;
        rtt.update(ack_delay, now.saturating_duration_since(largest_sent));
        s.rtt = Some(rtt);
        s.loss_run_start = None;
        s.max_window = s.max_window.max(cc.window());
        if let (true, Some(c)) = (measuring, self.flows[i].source.as_copa()) {
            s.dq_sum += c.queueing_delay().as_secs_f64();
            s.delta_sum += c.delta();
            s.competitive_n += u64::from(c.is_competitive());
            s.acks_n += 1;
        }
        self.try_send(i);
    }

    fn on_lost(&mut self, i: usize, pn: u64) {
        let now = self.now;
        let s = &mut self.senders[i];
        let Some((sent, size)) = s.sent.remove(&pn) else {
            return;
        };
        s.in_flight -= size;
        // Persistent congestion: losses only, for longer than three smoothed RTTs
        let run_start = *s.loss_run_start.get_or_insert(sent);
        let srtt = s.rtt.map_or(self.link.base_rtt, |r| r.get());
        let persistent = sent.saturating_duration_since(run_start) > 3 * srtt;
        if persistent {
            s.loss_run_start = None;
            s.persistent += 1;
        }
        let cc = self.flows[i].source.cc().unwrap();
        cc.on_congestion_event(now, sent, persistent, false, size, pn, SpaceId::Data);
        self.try_send(i);
    }
}

/// Convenience: flows that all start at 0
pub(super) fn flows(sources: Vec<Source>) -> Vec<Flow> {
    sources
        .into_iter()
        .map(|source| Flow {
            source,
            start: Duration::ZERO,
        })
        .collect()
}

/// A copa-auto(-NC) configuration whose probes are aligned on `epoch`
pub(super) fn copa_auto(epoch: Instant, mode_switching: bool) -> CopaConfig {
    let mut auto = super::CopaAutoConfig::default();
    auto.probe_epoch(Some(epoch));
    let mut config = CopaConfig::default();
    config.adaptive(Some(auto)).mode_switching(mode_switching);
    config
}

impl Sim {
    /// Replace flow `i` by a Copa whose clock starts at the simulation start
    pub(super) fn set_copa(&mut self, i: usize, config: CopaConfig) {
        self.flows[i].source = Source::copa_at(config, self.t0);
    }
}
