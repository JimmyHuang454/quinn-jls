//! Scenario tests on the bottleneck simulator
//!
//! Variants follow the delta trade-off paper: `Copa` / `Copa-NC` have a fixed delta with the
//! competitive mode on / off; `copa-auto` / `copa-auto-NC` choose delta online. Bounds leave a
//! margin around the values the simulator gives today, so they catch regressions without being
//! tied to the exact numbers.

use super::sim::*;
use super::*;
use crate::congestion::NewRenoConfig;

fn secs(n: u64) -> Duration {
    Duration::from_secs(n)
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

const BASE_RTT: Duration = Duration::from_millis(40);

fn copa_nc(delta: f64) -> CopaConfig {
    let mut c = CopaConfig::copa_nc();
    c.delta(delta);
    c
}

/// Packets per second of a throughput in bits per second
fn packets(bps: f64) -> f64 {
    bps / 8.0 / MTU as f64
}

fn run_one(link: Link, config: CopaConfig, duration: Duration, warmup: Duration) -> Report {
    Sim::new(link, flows(vec![Source::copa(config)])).run(duration, warmup)
}

/// eq:copa-fp: at equilibrium lambda * d_q = 1 / delta, with Copa's own d_q estimate
#[test]
fn s1_equilibrium_matches_fixed_point() {
    for (mbps, delta) in [
        (2.4, 0.1),
        (2.4, 0.25),
        (2.4, 0.5),
        (12.0, 0.1),
        (12.0, 0.5),
    ] {
        let r = run_one(Link::new(mbps, BASE_RTT), copa_nc(delta), secs(30), secs(5));
        let fixed_point = packets(r.throughput[0]) * r.copa_dq[0] * delta;
        assert!(r.utilization > 0.98, "{mbps} Mbit/s, delta {delta}: {r:?}");
        assert!(
            (0.75..1.25).contains(&fixed_point),
            "{mbps} Mbit/s, delta {delta}: lambda dq delta = {fixed_point}"
        );
    }
}

/// delta is the trade-off knob: smaller delta, more queueing
#[test]
fn s2_queueing_grows_as_delta_shrinks() {
    let queue =
        |delta| run_one(Link::new(12.0, BASE_RTT), copa_nc(delta), secs(30), secs(5)).mean_queueing;
    let q: Vec<f64> = [1.0, 0.5, 0.25, 0.1].into_iter().map(queue).collect();
    assert!(q.windows(2).all(|w| w[0] < w[1]), "{q:?}");
}

/// n Copa flows: equal shares, and a standing queue of about n / delta packets
#[test]
fn s3_flows_share_fairly() {
    for n in [2usize, 4] {
        let sources = (0..n)
            .map(|_| Source::copa(CopaConfig::copa_nc()))
            .collect();
        let r = Sim::new(Link::new(12.0, BASE_RTT), flows(sources)).run(secs(30), secs(10));
        assert!(r.jain(0..n) > 0.99, "n = {n}: {r:?}");
        assert!(r.utilization > 0.98, "n = {n}: {r:?}");
        let total: f64 = r.throughput.iter().map(|&t| packets(t)).sum();
        let fixed_point = total * r.copa_dq[0] * 0.5 / n as f64;
        assert!(
            (0.75..1.25).contains(&fixed_point),
            "n = {n}: {fixed_point}"
        );
    }

    // A late flow with the competitive mode on still gets its share
    let mut f = flows(vec![
        Source::copa(CopaConfig::default()),
        Source::copa(CopaConfig::default()),
    ]);
    f[1].start = secs(5);
    let r = Sim::new(Link::new(12.0, BASE_RTT), f).run(secs(40), secs(15));
    assert!(r.jain(0..2) > 0.95, "{r:?}");
}

/// Against a buffer-filling NewReno, the competitive mode keeps Copa from starving
#[test]
fn s4_competitive_mode_against_newreno() {
    let share = |config: CopaConfig| {
        let sources = vec![
            Source::copa(config),
            Source::other(NewRenoConfig::default()),
        ];
        let r = Sim::new(Link::new(12.0, BASE_RTT), flows(sources)).run(secs(60), secs(10));
        (r.throughput[0] / (r.throughput[0] + r.throughput[1]), r)
    };
    let (on, r) = share(CopaConfig::default());
    assert!(on > 0.3, "with mode switching: share {on}, {r:?}");
    assert!(r.competitive[0] > 0.5, "{r:?}");
    let (off, r) = share(CopaConfig::copa_nc());
    assert!(off < 0.1, "without mode switching: share {off}, {r:?}");
    assert_eq!(r.competitive[0], 0.0);
}

/// Capacity falls from 10 to 2 Mbit/s: Copa-NC converges to the new rate with a short queue
///
/// With the competitive mode on, the queue left by the drop keeps Copa in the competitive mode
/// (see the paper's conclusion on mode-switching misfires), so only Copa-NC is checked here.
#[test]
fn s5_capacity_drop() {
    let mut link = Link::new(10.0, BASE_RTT);
    link.rate.push((secs(15), 2e6));
    let r = run_one(link, CopaConfig::copa_nc(), secs(30), secs(20));
    assert!(r.utilization > 0.95, "{r:?}");
    assert!(r.mean_queueing < 0.02, "{r:?}");
}

/// A one-second outage: persistent congestion, slow start, then full rate again
#[test]
fn s6_recovers_from_outage() {
    for config in [CopaConfig::default(), CopaConfig::copa_nc()] {
        let mut link = Link::new(10.0, BASE_RTT);
        link.rate.push((secs(10), 0.0));
        link.rate.push((secs(11), 10e6));
        let mut sim = Sim::new(link, flows(vec![Source::copa(config)]));
        let r = sim.run(secs(20), secs(13));
        assert!(r.persistent[0] > 0, "{r:?}");
        assert!(r.utilization > 0.95, "{r:?}");
        // No window blow-up: 10 Mbit/s * 40 ms is about 42 packets
        assert!(r.max_window[0] < 200 * MTU, "{r:?}");
        let copa = sim.flows[0].source.as_copa().unwrap();
        assert!(!copa.in_slow_start());
    }
}

/// Delayed ACKs (every second packet, 25 ms timer) inflate per-packet RTT samples; the
/// RTT_standing filter must keep the queueing estimate and the rate unaffected
#[test]
fn s7_delayed_acks() {
    let run = |delayed_ack| {
        let mut link = Link::new(12.0, BASE_RTT);
        link.delayed_ack = delayed_ack;
        run_one(link, CopaConfig::default(), secs(30), secs(5))
    };
    let (plain, delayed) = (run(false), run(true));
    assert!(delayed.utilization > 0.97, "{delayed:?}");
    assert!(
        delayed.mean_queueing < 2.0 * plain.mean_queueing + 0.001,
        "{plain:?} {delayed:?}"
    );
}

/// Copa's own pacing rate (2 cwnd / RTT_standing) against quinn's default (1.25 cwnd / srtt)
#[test]
fn s8_pacing_rules_agree() {
    let run = |pacing| {
        let mut link = Link::new(12.0, BASE_RTT);
        link.pacing = pacing;
        run_one(link, CopaConfig::default(), secs(30), secs(5))
    };
    let (copa, quinn) = (run(Pacing::Controller), run(Pacing::Quinn));
    assert!(copa.utilization > 0.98 && quinn.utilization > 0.98);
    assert!((copa.mean_queueing - quinn.mean_queueing).abs() < 0.001);
}

/// Best power among fixed deltas with the competitive mode off
fn best_fixed_power(link: &dyn Fn() -> Link) -> f64 {
    [0.5, 0.25, 0.1, 0.05]
        .into_iter()
        .map(|delta| run_one(link(), copa_nc(delta), secs(80), secs(30)).power(0, BASE_RTT))
        .fold(0.0, f64::max)
}

fn run_auto(link: Link, mode_switching: bool) -> (Report, CopaAutoEstimate) {
    let mut sim = Sim::new(link, flows(vec![Source::copa(CopaConfig::default())]));
    let epoch = sim.epoch();
    sim.set_copa(0, copa_auto(epoch, mode_switching));
    let r = sim.run(secs(80), secs(30));
    let est = sim.flows[0]
        .source
        .as_copa()
        .unwrap()
        .adaptive_estimate()
        .unwrap();
    (r, est)
}

/// Capacity changing every 500 ms, uniformly within +-40% (sigma 0.4)
fn variable_link() -> Link {
    let mut link = Link::new(12.0, BASE_RTT);
    link.rate.clear();
    let mut x = 7u64;
    for i in 0..200 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let u = (x >> 11) as f64 / (1u64 << 53) as f64;
        let factor = 1.0 + 0.4 * 3f64.sqrt() * (2.0 * u - 1.0);
        link.rate.push((ms(500 * i), 12e6 * factor));
    }
    link
}

/// copa-auto on one flow: sane estimates, d_q tracking d_q*, and power close to the best fixed
/// delta chosen after the fact
#[test]
fn c1_copa_auto_single_flow() {
    // Constant capacity, no jitter: delta 0.5 is already near-optimal
    let plain = || Link::new(12.0, BASE_RTT);
    // 10 ms of random one-way jitter
    let jitter = || {
        let mut link = Link::new(12.0, BASE_RTT);
        link.jitter = ms(10);
        link
    };
    // During the short [30, 80s] window, two gentle 0.5 BDP probes run on `variable` while
    // capacity swings +-69% every 500 ms, trimming utilization from ~0.94 to ~0.91.
    for (name, link, min_util, min_ratio) in [
        ("constant", &plain as &dyn Fn() -> Link, 0.93, 0.8),
        ("jitter", &jitter, 0.93, 0.95),
        ("variable", &variable_link, 0.90, 0.85),
    ] {
        let best = best_fixed_power(link);
        for mode_switching in [true, false] {
            let (r, est) = run_auto(link(), mode_switching);
            let ctx = format!("{name}, mode switching {mode_switching}: {r:?} {est:?}");
            let j = est.jitter.expect("jitter floor");
            assert!((0.0..0.02).contains(&j), "{ctx}");
            assert!(est.rho > 0.0 && est.rho < 1.0, "{ctx}");
            assert!(est.dq_target > j, "{ctx}");
            let delta = est.delta.expect("delta");
            assert!((0.001..=0.5).contains(&delta), "{ctx}");
            assert!(r.utilization > min_util, "{ctx}");
            let ratio = r.power(0, BASE_RTT) / best;
            assert!(ratio > min_ratio, "{ctx}: power ratio {ratio}");
            if name != "variable" {
                let tracking = r.copa_dq[0] / est.dq_target;
                assert!(
                    (0.5..2.0).contains(&tracking),
                    "{ctx}: dq / dq* = {tracking}"
                );
            }
        }
    }
}

/// copa-auto-NC with 2 and 4 flows: aligned probes give every flow the same jitter floor, and the
/// AIMD on 1/delta keeps the shares fair
#[test]
fn c2_copa_auto_multi_flow() {
    for n in [2usize, 4] {
        let sources = (0..n)
            .map(|_| Source::copa(CopaConfig::default()))
            .collect();
        let mut sim = Sim::new(Link::new(12.0, BASE_RTT), flows(sources));
        let epoch = sim.epoch();
        for i in 0..n {
            sim.set_copa(i, copa_auto(epoch, false));
        }
        let r = sim.run(secs(80), secs(30));
        assert!(r.utilization > 0.98, "n = {n}: {r:?}");
        // Early shares only. Since fast recovery restores each flow's own window after a probe,
        // a probe no longer resets the shares, and 1/delta still swings between flows during the
        // first minute: over 30-80 s Jain is 0.83 for two flows and 0.73 for four (above 0.9
        // with the original copa-auto). c6 checks the fairness over 300 s.
        let bound = if n == 2 { 0.8 } else { 0.7 };
        assert!(r.jain(0..n) > bound, "n = {n}: {r:?}");
        let floors: Vec<f64> = (0..n)
            .map(|i| {
                let est = sim.flows[i].source.as_copa().unwrap().adaptive_estimate();
                est.unwrap().jitter.unwrap()
            })
            .collect();
        let spread = floors.iter().cloned().fold(f64::MIN, f64::max)
            - floors.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread < 0.001, "n = {n}: jitter floors {floors:?}");
    }
}

/// Unresponsive on/off background traffic: the idle fraction counts it in rho, and copa-auto
/// fills the capacity the background leaves
#[test]
fn c3_copa_auto_with_background() {
    for mode_switching in [true, false] {
        let sources = vec![
            Source::copa(CopaConfig::default()),
            Source::Cbr {
                bps: 4e6,
                on: secs(1),
                off: secs(1),
            },
        ];
        let mut sim = Sim::new(Link::new(12.0, BASE_RTT), flows(sources));
        let epoch = sim.epoch();
        sim.set_copa(0, copa_auto(epoch, mode_switching));
        let r = sim.run(secs(80), secs(30));
        assert!(r.utilization > 0.95, "{r:?}");
        // The background sends 2 Mbit/s on average; Copa gets nearly all of the rest
        assert!(r.throughput[0] > 9e6, "{r:?}");
        let est = sim.flows[0].source.as_copa().unwrap().adaptive_estimate();
        assert!(est.unwrap().delta.is_some());
    }
}

/// copa-auto against NewReno: the competitive mode takes over, and copa-auto keeps its share
#[test]
fn c4_copa_auto_against_newreno() {
    let sources = vec![
        Source::copa(CopaConfig::default()),
        Source::other(NewRenoConfig::default()),
    ];
    let mut sim = Sim::new(Link::new(12.0, BASE_RTT), flows(sources));
    let epoch = sim.epoch();
    sim.set_copa(0, copa_auto(epoch, true));
    let r = sim.run(secs(80), secs(30));
    assert!(r.competitive[0] > 0.5, "{r:?}");
    let share = r.throughput[0] / (r.throughput[0] + r.throughput[1]);
    assert!(share > 0.3, "share {share}: {r:?}");
}

/// On a 5G-like link (jitter that Copa's filters keep), every probe at delta_max is a dip to a
/// few Mbit/s. Adaptive probes stop once the floor settles; drains (with fast recovery) keep
/// RTT_min fresh without such dips.
#[test]
fn c5_adaptive_probes_cut_dips_on_a_jittery_link() {
    let run = |policy| {
        let mut sim = Sim::new(
            jittery_5g(),
            flows(vec![Source::copa(CopaConfig::default())]),
        );
        let epoch = sim.epoch();
        sim.set_copa(0, copa_auto_policy(epoch, false, policy));
        sim.run(secs(300), secs(30))
    };
    let dips = |r: &Report| {
        let s = &r.per_second[0];
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        s.iter().filter(|&&x| x < 0.3 * mean).count()
    };
    let power = |r: &Report| r.utilization / ((0.05 + r.mean_queueing) / 0.05);
    let old = run(ProbePolicy::Periodic);
    let new = run(ProbePolicy::Both);
    assert!(
        dips(&new) + 4 <= dips(&old),
        "dips {} -> {}",
        dips(&old),
        dips(&new)
    );
    assert!(power(&new) > 0.97 * power(&old), "{old:?}\n{new:?}");
}

/// With adaptive probes several flows keep the queue busy between probes; the drains keep
/// RTT_min at the propagation delay and the flows fair
#[test]
fn c6_drains_keep_rtt_min_with_several_flows() {
    for n in [2usize, 4] {
        let sources = (0..n)
            .map(|_| Source::copa(CopaConfig::default()))
            .collect();
        let mut sim = Sim::new(Link::new(12.0, BASE_RTT), flows(sources));
        let epoch = sim.epoch();
        for i in 0..n {
            sim.set_copa(i, copa_auto(epoch, false));
        }
        let r = sim.run(secs(300), secs(30));
        assert!(r.jain(0..n) > 0.9, "n = {n}: {r:?}");
        for i in 0..n {
            let rtt_min = sim.flows[i].source.as_copa().unwrap().min_rtt().unwrap();
            assert!(
                rtt_min < BASE_RTT + ms(6),
                "n = {n}, flow {i}: RTT_min {rtt_min:?}"
            );
        }
    }
}

/// Drains only when RTT_min is stale: a single flow empties the queue by itself and skips almost
/// all of them; four flows sharing a lower-BDP bottleneck keep the queue busy even at `delta = 0.5`,
/// so they drain (`gentle_probe` keeps `j = 0` accurate on `12 Mbit/s` so 4 flows there swing idle)
#[test]
fn c7_drains_are_skipped_when_rtt_min_is_fresh() {
    for (n, mbps, mostly_skipped) in [(1usize, 12.0, true), (4, 3.0, false)] {
        let sources = (0..n)
            .map(|_| Source::copa(CopaConfig::default()))
            .collect();
        let mut sim = Sim::new(Link::new(mbps, BASE_RTT), flows(sources));
        let epoch = sim.epoch();
        for i in 0..n {
            sim.set_copa(i, copa_auto(epoch, false));
        }
        sim.run(secs(300), secs(30));
        let (started, skipped) = sim.flows[0].source.as_copa().unwrap().drains;
        assert!(
            started + skipped > 5,
            "n = {n}: {started} started, {skipped} skipped"
        );
        if mostly_skipped {
            assert!(
                skipped > 3 * started,
                "n = {n}: {started} started, {skipped} skipped"
            );
        } else {
            assert!(started > 0, "n = {n}: {started} started, {skipped} skipped");
        }
    }
}

#[test]
fn c8_fixed_jitter_and_drain_toggle() {
    for drain_enabled in [false, true] {
        let n = 4usize;
        let sources = (0..n)
            .map(|_| Source::copa(CopaConfig::default()))
            .collect();
        let mut sim = Sim::new(Link::new(12.0, BASE_RTT), flows(sources));
        let epoch = sim.epoch();
        for i in 0..n {
            let mut auto = CopaAutoConfig::default();
            auto.probe_epoch(Some(epoch))
                .jitter_floor_ms(1.0)
                .drain_enabled(drain_enabled);
            let mut cfg = CopaConfig::copa_nc();
            cfg.adaptive(Some(auto));
            sim.set_copa(i, cfg);
        }
        sim.run(secs(120), secs(20));
        let copa = sim.flows[0].source.as_copa().unwrap();
        assert_eq!(copa.adaptive_estimate().unwrap().jitter, Some(0.001));
        let (started, skipped) = copa.drains;
        if drain_enabled {
            assert!(started > 0, "drain enabled: {started} started, {skipped} skipped");
            assert!(copa.min_rtt().unwrap() < BASE_RTT + ms(6));
        } else {
            assert_eq!((started, skipped), (0, 0));
        }
    }
}

/// Gentle BDP-based `j` probes avoid the 20 s near-zero stalls caused by `delta_max` probes under
/// heavy jitter even when probes fire in every slot
#[test]
fn c9_gentle_probe_avoids_zero_dips() {
    let run = |gentle: bool| {
        let mut sim = Sim::new(
            jittery_5g(),
            flows(vec![Source::copa(CopaConfig::copa_nc())]),
        );
        let epoch = sim.epoch();
        let mut auto = CopaAutoConfig::default();
        auto.probe_epoch(Some(epoch))
            .adaptive_probes(false)
            .gentle_probe(gentle);
        let mut cfg = CopaConfig::copa_nc();
        cfg.adaptive(Some(auto));
        sim.set_copa(0, cfg);
        let r = sim.run(secs(140), secs(20));
        let est = sim.flows[0]
            .source
            .as_copa()
            .unwrap()
            .adaptive_estimate()
            .unwrap();
        let min1s = r.per_second[0].iter().cloned().fold(f64::MAX, f64::min);
        (r, est, min1s)
    };
    let (legacy_r, _, legacy_min1s) = run(false);
    let (gentle_r, gentle_est, gentle_min1s) = run(true);
    assert!(
        gentle_min1s > 2.0 * legacy_min1s && gentle_min1s > 8e6,
        "gentle min1s {gentle_min1s:.0} should avoid legacy stalls ({legacy_min1s:.0}): gentle={gentle_r:?}, legacy={legacy_r:?}"
    );
    let j = gentle_est.jitter.expect("jitter floor");
    assert!((0.001..0.015).contains(&j), "gentle jitter={j}");
}

/// Probe schedules compared by `probe_policy_comparison`
#[derive(Debug, Clone, Copy)]
enum ProbePolicy {
    /// A probe in every aligned slot, no fast recovery (the original copa-auto)
    Periodic,
    /// Adaptive probes only
    Adaptive,
    /// Fast recovery only
    Recovery,
    /// Adaptive probes and fast recovery (the default)
    Both,
}

fn copa_auto_policy(epoch: Instant, mode_switching: bool, policy: ProbePolicy) -> CopaConfig {
    let mut auto = CopaAutoConfig::default();
    auto.probe_epoch(Some(epoch));
    match policy {
        ProbePolicy::Periodic => {
            auto.periodic_probes();
        }
        ProbePolicy::Adaptive => {
            auto.probe_recovery(0.0);
        }
        ProbePolicy::Recovery => {
            auto.adaptive_probes(false);
        }
        ProbePolicy::Both => {}
    }
    let mut config = CopaConfig::default();
    config.adaptive(Some(auto)).mode_switching(mode_switching);
    config
}

/// A 5G-like uplink: 40 Mbit/s, 50 ms base RTT, up to 30 ms of extra one-way delay that changes
/// every 100 ms (scheduling and radio conditions)
fn jittery_5g() -> Link {
    let mut link = Link::new(40.0, ms(50));
    link.jitter = ms(30);
    // Held for 100 ms: longer than half an RTT, so Copa's standing RTT keeps it
    link.jitter_hold = Some(ms(100));
    link
}

/// Prints throughput, queueing, power, jitter floor and the 1 s dips of each probe policy.
/// Run with `cargo test -p quinn-proto --lib probe_policy_comparison -- --ignored --nocapture`.
#[test]
#[ignore]
fn probe_policy_comparison() {
    let duration = secs(300);
    let warmup = secs(30);
    let links: [(&str, &dyn Fn() -> Link, usize); 6] = [
        ("constant 12M", &|| Link::new(12.0, BASE_RTT), 1),
        (
            "jitter 10ms",
            &|| {
                let mut l = Link::new(12.0, BASE_RTT);
                l.jitter = ms(10);
                l
            },
            1,
        ),
        ("variable", &variable_link, 1),
        ("5G-like 40M", &jittery_5g, 1),
        ("2 flows", &|| Link::new(12.0, BASE_RTT), 2),
        ("4 flows", &|| Link::new(12.0, BASE_RTT), 4),
    ];
    println!(
        "{:<13} {:<9} {:>6} {:>5} {:>7} {:>6} {:>6} {:>7} {:>5} {:>6} {:>7}",
        "link",
        "policy",
        "Mbps",
        "util",
        "q ms",
        "power",
        "j ms",
        "min1s",
        "dips",
        "jain",
        "rttmin"
    );
    for (name, link, n) in links {
        for policy in [
            ProbePolicy::Periodic,
            ProbePolicy::Adaptive,
            ProbePolicy::Recovery,
            ProbePolicy::Both,
        ] {
            let l = link();
            let base = l.base_rtt;
            let sources = (0..n)
                .map(|_| Source::copa(CopaConfig::default()))
                .collect();
            let mut sim = Sim::new(l, flows(sources));
            let epoch = sim.epoch();
            for i in 0..n {
                sim.set_copa(i, copa_auto_policy(epoch, false, policy));
            }
            let r = sim.run(duration, warmup);
            // Total goodput per second across flows; a dip is a second under half the mean
            let secs_n = r.per_second[0].len();
            let total: Vec<f64> = (0..secs_n)
                .map(|t| r.per_second.iter().map(|f| f[t]).sum())
                .collect();
            let mean = total.iter().sum::<f64>() / secs_n as f64;
            let min = total.iter().cloned().fold(f64::MAX, f64::min);
            let dips = total.iter().filter(|&&x| x < 0.5 * mean).count();
            // The paper's normalised power: utilization / (mean RTT / 50 ms)
            let rtt = base.as_secs_f64() + r.mean_queueing;
            let power = r.utilization / (rtt / 0.05);
            let j = sim.flows[0]
                .source
                .as_copa()
                .and_then(|c| c.adaptive_estimate())
                .and_then(|e| e.jitter)
                .unwrap_or(f64::NAN);
            let rtt_min = sim.flows[0]
                .source
                .as_copa()
                .and_then(|c| c.min_rtt())
                .map_or(f64::NAN, |d| d.as_secs_f64() * 1e3);
            println!(
                "{:<13} {:<9} {:>6.2} {:>5.3} {:>7.2} {:>6.3} {:>6.2} {:>7.2} {:>5} {:>6.3} {:>7.2}",
                name,
                format!("{policy:?}"),
                r.throughput.iter().sum::<f64>() / 1e6,
                r.utilization,
                r.mean_queueing * 1e3,
                power,
                j * 1e3,
                min / 1e6,
                dips,
                r.jain(0..n),
                rtt_min,
            );
        }
    }
}

/// Per-second goodput on the 5G-like link, for looking at the dips by eye.
#[test]
#[ignore]
fn probe_policy_series() {
    for policy in [ProbePolicy::Periodic, ProbePolicy::Both] {
        let mut sim = Sim::new(
            jittery_5g(),
            flows(vec![Source::copa(CopaConfig::default())]),
        );
        let epoch = sim.epoch();
        sim.set_copa(0, copa_auto_policy(epoch, false, policy));
        let r = sim.run(secs(300), secs(0));
        let est = sim.flows[0].source.as_copa().unwrap().adaptive_estimate();
        let line: Vec<String> = r.per_second[0]
            .iter()
            .map(|b| format!("{:.0}", b / 1e6))
            .collect();
        println!("{policy:?} {est:?}\n{}", line.join(" "));
    }
}

/// Fast-recovery ratio sweep with adaptive probes.
#[test]
#[ignore]
fn probe_recovery_sweep() {
    let links: [(&str, &dyn Fn() -> Link, usize); 3] = [
        ("constant 12M", &|| Link::new(12.0, BASE_RTT), 1),
        ("5G-like 40M", &jittery_5g, 1),
        ("4 flows", &|| Link::new(12.0, BASE_RTT), 4),
    ];
    for (name, link, n) in links {
        for ratio in [0.0, 0.5, 0.7, 0.9, 1.0] {
            let l = link();
            let base = l.base_rtt;
            let sources = (0..n)
                .map(|_| Source::copa(CopaConfig::default()))
                .collect();
            let mut sim = Sim::new(l, flows(sources));
            let epoch = sim.epoch();
            for i in 0..n {
                let mut auto = CopaAutoConfig::default();
                auto.probe_epoch(Some(epoch)).probe_recovery(ratio);
                let mut config = CopaConfig::default();
                config.adaptive(Some(auto)).mode_switching(false);
                sim.set_copa(i, config);
            }
            let r = sim.run(secs(300), secs(30));
            let rtt = base.as_secs_f64() + r.mean_queueing;
            println!(
                "{name:<13} ratio {ratio:.1}: util {:.3} q {:.2} ms power {:.3} jain {:.3}",
                r.utilization,
                r.mean_queueing * 1e3,
                r.utilization / (rtt / 0.05),
                r.jain(0..n)
            );
        }
    }
}
