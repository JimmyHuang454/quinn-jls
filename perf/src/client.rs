#[cfg(feature = "json-output")]
use std::path::PathBuf;
use std::{
    fs,
    io::{BufWriter, Write},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use bytes::Bytes;
use clap::Parser;
use quinn::{crypto::rustls::QuicClientConfig, TokioRuntime};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, error, info};

use crate::{
    noprotection::NoProtectionClientConfig,
    parse_byte_size,
    stats::{OpenStreamStats, Stats},
    CommonOpt, PERF_CIPHER_SUITES,
};

/// Connects to a QUIC perf server and maintains a specified pattern of requests until interrupted
#[derive(Parser)]
#[clap(name = "client")]
pub struct Opt {
    /// Host to connect to
    #[clap(default_value = "localhost:4433")]
    host: String,
    /// Override DNS resolution for host
    #[clap(long)]
    ip: Option<IpAddr>,
    /// Specify the local socket address
    #[clap(long)]
    local_addr: Option<SocketAddr>,
    /// Number of unidirectional requests to maintain concurrently
    #[clap(long, default_value = "0")]
    uni_requests: u64,
    /// Number of bidirectional requests to maintain concurrently
    #[clap(long, default_value = "1")]
    bi_requests: u64,
    /// Number of bytes to request
    ///
    /// This can use SI suffixes for sizes. For example, 1M will transfer
    /// 1MiB, 10G will transfer 10GiB.
    #[clap(long, default_value = "1M", value_parser = parse_byte_size)]
    download_size: u64,
    /// Number of bytes to transmit, in addition to the request header
    ///
    /// This can use SI suffixes for sizes. For example, 1M will transfer
    /// 1MiB, 10G will transfer 10GiB.
    #[clap(long, default_value = "1M", value_parser = parse_byte_size)]
    upload_size: u64,
    /// The time to run in seconds
    #[clap(long, default_value = "60")]
    duration: u64,
    /// The interval in seconds at which stats are reported
    #[clap(long, default_value = "1")]
    interval: u64,
    /// Mortise-format workload trace (`<gap_ms> <size_bytes>` per line); replaces fixed uni/bi loops
    #[clap(long)]
    workload: Option<PathBuf>,
    /// Output CSV path for per-request RCT records (`id,size,client_send,server_recv,client_recv,rct`)
    #[clap(long)]
    rct_csv: Option<PathBuf>,
    /// Output JSON path for connection-level path statistics
    #[cfg(feature = "json-output")]
    #[clap(long)]
    summary_json: Option<PathBuf>,
    /// File path to output JSON statistics to. If the file is '-', stdout will be used
    #[cfg(feature = "json-output")]
    #[clap(long)]
    json: Option<PathBuf>,
    /// Common options
    #[command(flatten)]
    common: CommonOpt,
}

pub async fn run(opt: Opt) -> Result<()> {
    let mut host_parts = opt.host.split(':');
    let host_name = host_parts.next().unwrap();
    let host_port = host_parts
        .next()
        .map_or(Ok(443), |x| x.parse())
        .context("parsing port")?;
    let addr = match opt.ip {
        None => tokio::net::lookup_host(&opt.host)
            .await
            .context("resolving host")?
            .next()
            .unwrap(),
        Some(ip) => SocketAddr::new(ip, host_port),
    };

    info!("connecting to {} at {}", host_name, addr);

    let bind_addr = opt.local_addr.unwrap_or_else(|| {
        let unspec = if addr.is_ipv4() {
            Ipv4Addr::UNSPECIFIED.into()
        } else {
            Ipv6Addr::UNSPECIFIED.into()
        };
        SocketAddr::new(unspec, 0)
    });

    info!("local addr {:?}", bind_addr);

    let socket = opt.common.bind_socket(bind_addr)?;

    let mut endpoint_cfg = quinn::EndpointConfig::default();
    endpoint_cfg.max_udp_payload_size(opt.common.max_udp_payload_size)?;

    let endpoint = quinn::Endpoint::new(endpoint_cfg, None, socket, Arc::new(TokioRuntime))?;

    let default_provider = rustls::crypto::ring::default_provider();
    let provider = Arc::new(rustls::crypto::CryptoProvider {
        cipher_suites: PERF_CIPHER_SUITES.into(),
        ..default_provider
    });

    let mut crypto = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .unwrap()
        .dangerous()
        .with_custom_certificate_verifier(SkipServerVerification::new(provider))
        .with_no_client_auth();
    crypto.alpn_protocols = vec![b"perf".to_vec()];

    if opt.common.keylog {
        crypto.key_log = Arc::new(rustls::KeyLogFile::new());
    }

    let transport = opt.common.build_transport_config(
        #[cfg(feature = "qlog")]
        "perf-client",
    )?;

    let crypto = Arc::new(QuicClientConfig::try_from(crypto)?);
    let mut config = quinn::ClientConfig::new(match opt.common.no_protection {
        true => Arc::new(NoProtectionClientConfig::new(crypto)),
        false => crypto,
    });
    config.transport_config(Arc::new(transport));

    let stream_stats = OpenStreamStats::default();

    let connection = endpoint
        .connect_with(config, addr, host_name)?
        .await
        .context("connecting")?;

    info!("established");

    let workload_items = match &opt.workload {
        Some(path) => Some(load_workload(path)?),
        None => None,
    };
    let is_workload = workload_items.is_some();
    let rct_records = Arc::new(Mutex::new(Vec::<RctRecord>::new()));
    let run_start = Instant::now();

    let drive_fut = async {
        if let Some(items) = workload_items {
            drive_workload(
                connection.clone(),
                stream_stats.clone(),
                items,
                rct_records.clone(),
            )
            .await
        } else {
            tokio::try_join!(
                drive_uni(
                    connection.clone(),
                    stream_stats.clone(),
                    opt.uni_requests,
                    opt.upload_size,
                    opt.download_size
                ),
                drive_bi(
                    connection.clone(),
                    stream_stats.clone(),
                    opt.bi_requests,
                    opt.upload_size,
                    opt.download_size
                )
            )
            .map(|_| ())
        }
    };

    let mut stats = Stats::default();

    let stats_fut = async {
        let interval_duration = Duration::from_secs(opt.interval);

        #[cfg(feature = "json-output")]
        let allow_table_output = opt.json.clone().is_none_or(|path| path != Path::new("-"));
        #[cfg(not(feature = "json-output"))]
        let allow_table_output = true;

        loop {
            let start = Instant::now();
            tokio::time::sleep(interval_duration).await;
            {
                stats.on_interval(start, &stream_stats);

                if allow_table_output {
                    stats.print();
                    if opt.common.conn_stats {
                        println!("{:?}\n", connection.stats());
                    }
                }
            }
        }
    };

    tokio::select! {
        res = drive_fut => {
            if let Err(e) = res {
                error!("drive failed: {:#}", e);
            }
            if is_workload {
                info!("workload finished, shutting down");
                connection.close(0u32.into(), b"done");
            }
        }
        _ = stats_fut => {}
        _ = tokio::signal::ctrl_c() => {
            info!("shutting down");
            connection.close(0u32.into(), b"interrupted");
        }
        // Add a small duration so the final interval can be reported
        _ = tokio::time::sleep(Duration::from_secs(opt.duration) + Duration::from_millis(200)) => {
            info!("shutting down");
            connection.close(0u32.into(), b"done");
        }
    }

    let final_stats = connection.stats();
    let elapsed_s = run_start.elapsed().as_secs_f64();
    endpoint.wait_idle().await;

    if let Some(path) = &opt.rct_csv {
        let mut records = rct_records.lock().unwrap().clone();
        records.sort_by_key(|r| r.id);
        write_rct_csv(path, &records, final_stats.path.lost_packets)?;
    }

    #[cfg(feature = "json-output")]
    if let Some(path) = &opt.summary_json {
        let completed = rct_records.lock().unwrap().len();
        write_summary_json(path, &final_stats, elapsed_s, completed)?;
    }

    #[cfg(feature = "json-output")]
    if let Some(path) = opt.json {
        stats.print_json(path.as_path())?;
    }

    Ok(())
}

async fn drain_stream(
    mut stream: quinn::RecvStream,
    download: u64,
    stream_stats: OpenStreamStats,
) -> Result<()> {
    if download == 0 {
        return Ok(());
    }

    #[rustfmt::skip]
    let mut bufs = [
        Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
        Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
        Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
        Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
        Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
        Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
        Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
        Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
    ];
    let download_start = Instant::now();
    let recv_stream_stats = stream_stats.new_receiver(&stream, download);

    let mut first_byte = true;

    while let Some(size) = stream.read_chunks(&mut bufs[..]).await? {
        if first_byte {
            recv_stream_stats.on_first_byte(download_start.elapsed());
            first_byte = false;
        }
        let bytes_received = bufs[..size].iter().map(|b| b.len()).sum();
        recv_stream_stats.on_bytes(bytes_received);
    }

    if first_byte {
        recv_stream_stats.on_first_byte(download_start.elapsed());
    }
    recv_stream_stats.finish(download_start.elapsed());

    debug!("response finished on {}", stream.id());
    Ok(())
}

async fn drive_uni(
    connection: quinn::Connection,
    stream_stats: OpenStreamStats,
    concurrency: u64,
    upload: u64,
    download: u64,
) -> Result<()> {
    if concurrency == 0 {
        return Ok(());
    }

    let sem = Arc::new(Semaphore::new(concurrency as usize));

    loop {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let send = connection.open_uni().await?;
        let stream_stats = stream_stats.clone();

        debug!("sending request on {}", send.id());
        let connection = connection.clone();
        tokio::spawn(async move {
            if let Err(e) = request_uni(send, connection, upload, download, stream_stats).await {
                error!("sending request failed: {:#}", e);
            }

            drop(permit);
        });
    }
}

async fn request_uni(
    send: quinn::SendStream,
    conn: quinn::Connection,
    upload: u64,
    download: u64,
    stream_stats: OpenStreamStats,
) -> Result<()> {
    request(send, upload, download, stream_stats.clone()).await?;
    let recv = conn.accept_uni().await?;
    drain_stream(recv, download, stream_stats).await?;
    Ok(())
}

async fn request(
    mut send: quinn::SendStream,
    mut upload: u64,
    download: u64,
    stream_stats: OpenStreamStats,
) -> Result<()> {
    let upload_start = Instant::now();
    send.write_all(&download.to_be_bytes()).await?;
    if upload == 0 {
        send.finish().unwrap();
        return Ok(());
    }

    let send_stream_stats = stream_stats.new_sender(&send, upload);

    static DATA: [u8; 1024 * 1024] = [42; 1024 * 1024];
    while upload > 0 {
        let chunk_len = upload.min(DATA.len() as u64);
        send.write_chunk(Bytes::from_static(&DATA[..chunk_len as usize]))
            .await
            .context("sending response")?;
        send_stream_stats.on_bytes(chunk_len as usize);
        upload -= chunk_len;
    }
    send.finish().unwrap();
    // Wait for stream to close
    _ = send.stopped().await;
    send_stream_stats.finish(upload_start.elapsed());

    debug!("upload finished on {}", send.id());
    Ok(())
}

async fn drive_bi(
    connection: quinn::Connection,
    stream_stats: OpenStreamStats,
    concurrency: u64,
    upload: u64,
    download: u64,
) -> Result<()> {
    if concurrency == 0 {
        return Ok(());
    }

    let sem = Arc::new(Semaphore::new(concurrency as usize));

    loop {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let (send, recv) = connection.open_bi().await?;
        let stream_stats = stream_stats.clone();

        debug!("sending request on {}", send.id());
        tokio::spawn(async move {
            if let Err(e) = request_bi(send, recv, upload, download, stream_stats).await {
                error!("request failed: {:#}", e);
            }

            drop(permit);
        });
    }
}

async fn request_bi(
    send: quinn::SendStream,
    recv: quinn::RecvStream,
    upload: u64,
    download: u64,
    stream_stats: OpenStreamStats,
) -> Result<()> {
    request(send, upload, download, stream_stats.clone()).await?;
    drain_stream(recv, download, stream_stats).await?;
    Ok(())
}

#[derive(Debug)]
struct SkipServerVerification(Arc<rustls::crypto::CryptoProvider>);

impl SkipServerVerification {
    fn new(provider: Arc<rustls::crypto::CryptoProvider>) -> Arc<Self> {
        Arc::new(Self(provider))
    }
}

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        _now: UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

#[derive(Clone, Debug)]
struct RctRecord {
    id: usize,
    size: u64,
    client_send_ns: u128,
    client_recv_ns: u128,
    rct_ms: u64,
}

fn parse_workload_str(content: &str) -> Result<Vec<(Duration, u64)>> {
    let mut items = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let gap_ms = parts
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .with_context(|| format!("invalid gap_ms at workload line {}", idx + 1))?;
        let size = parts
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .with_context(|| format!("invalid size_bytes at workload line {}", idx + 1))?;
        items.push((Duration::from_millis(gap_ms), size));
    }
    Ok(items)
}

fn load_workload(path: &Path) -> Result<Vec<(Duration, u64)>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading workload {}", path.display()))?;
    parse_workload_str(&content)
}

fn now_epoch_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

async fn drive_workload(
    connection: quinn::Connection,
    stream_stats: OpenStreamStats,
    items: Vec<(Duration, u64)>,
    rct_records: Arc<Mutex<Vec<RctRecord>>>,
) -> Result<()> {
    let mut scheduled = Instant::now();
    let mut tasks = JoinSet::new();

    for (idx, (gap, size)) in items.into_iter().enumerate() {
        if !gap.is_zero() {
            scheduled += gap;
            tokio::time::sleep_until(scheduled.into()).await;
        }
        let id = idx + 1;
        let conn = connection.clone();
        let stream_stats = stream_stats.clone();
        let rct_records = rct_records.clone();
        tasks.spawn(async move {
            let client_send_ns = now_epoch_ns();
            let req_start = Instant::now();
            let (send, recv) = conn.open_bi().await?;
            request_bi(send, recv, 0, size, stream_stats).await?;
            let rct_ms = req_start.elapsed().as_millis() as u64;
            let client_recv_ns = now_epoch_ns();
            rct_records.lock().unwrap().push(RctRecord {
                id,
                size,
                client_send_ns,
                client_recv_ns,
                rct_ms,
            });
            Ok::<(), anyhow::Error>(())
        });
    }

    while let Some(res) = tasks.join_next().await {
        res.context("workload request task panicked")??;
    }

    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn write_rct_csv(path: &Path, records: &[RctRecord], lost_packets: u64) -> Result<()> {
    ensure_parent_dir(path)?;
    let file = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(w, "id,size,client_send,server_recv,client_recv,rct")?;
    for r in records {
        writeln!(
            w,
            "{},{},{},0,{},{}",
            r.id, r.size, r.client_send_ns, r.client_recv_ns, r.rct_ms
        )?;
    }
    writeln!(w, "0,0,0,{lost_packets},0,0")?;
    w.flush()?;
    Ok(())
}

#[cfg(feature = "json-output")]
fn write_summary_json(
    path: &Path,
    stats: &quinn::ConnectionStats,
    elapsed_s: f64,
    requests_completed: usize,
) -> Result<()> {
    ensure_parent_dir(path)?;
    let summary = serde_json::json!({
        "elapsed_s": elapsed_s,
        "requests_completed": requests_completed,
        "udp_tx_bytes": stats.udp_tx.bytes,
        "udp_rx_bytes": stats.udp_rx.bytes,
        "sent_packets": stats.path.sent_packets,
        "lost_packets": stats.path.lost_packets,
        "lost_bytes": stats.path.lost_bytes,
        "congestion_events": stats.path.congestion_events,
        "rtt_ms": stats.path.rtt.as_secs_f64() * 1000.0,
        "min_rtt_ms": stats.path.min_rtt.as_secs_f64() * 1000.0,
    });
    let file = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    serde_json::to_writer_pretty(file, &summary)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mortise_workload() {
        let input = "0\t359246\n4485\t161520\n# comment\n\n24 33164\n";
        let parsed = parse_workload_str(input).unwrap();
        assert_eq!(
            parsed,
            vec![
                (Duration::from_millis(0), 359246),
                (Duration::from_millis(4485), 161520),
                (Duration::from_millis(24), 33164),
            ]
        );
    }
}
