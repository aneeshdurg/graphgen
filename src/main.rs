use std::env;
use std::fs::{File, OpenOptions};
use std::io::{prelude::*, BufReader, BufWriter, SeekFrom};
use std::path::PathBuf;
use std::process::Command;
use std::thread;

use clap::{Parser, ValueEnum};
use rand::distributions::uniform::UniformFloat;
use rand::Rng;
use rand_distr::uniform::UniformSampler;
use rand_distr::{Distribution, Exp, Normal};
use tqdm::{pbar, tqdm};

#[derive(Clone, Debug, ValueEnum, PartialEq)]
enum Dist {
    /// Do not generate any values
    None,
    /// Generate values with a uniform distibution
    Uniform,
    /// Generate values with a normal distibution
    Normal,
    /// Generate values with a exponential distibution
    Exp,
}

/// Generate graphs with different edge and property distributions
#[derive(Clone, Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Number of nodes
    n_nodes: usize,
    /// Minimum property size (bytes)
    min_prop_size: usize,
    /// Maximum property size (bytes)
    max_prop_size: usize,

    /// Edge distribution
    #[arg(long)]
    edge_dist: Dist,

    /// Node property size distribution
    #[arg(long)]
    node_prop_dist: Dist,

    /// Edge property size distribution
    #[arg(long, default_value = "none")]
    edge_prop_dist: Dist,

    /// Output directory
    #[arg(long, default_value = ".")]
    outdir: PathBuf,

    /// Number of processes
    #[arg(long, default_value = "8")]
    nprocs: usize,

    /// Generate chunks that can be used to incrementally build the graph - 1 chunk per thread
    #[arg(long)]
    generatechunks: bool,
}

fn create_dir(dir: &PathBuf) {
    let cmd = format!("mkdir -p {}", &dir.to_string_lossy());
    Command::new("sh")
        .args(["-c", &cmd])
        .output()
        .expect("failed to create outdir");
}

fn get_prop<R>(
    rng: &mut R,
    data_source: &mut File,
    prop_dist: &Dist,
    min_prop_size: usize,
    prop_range: usize,
) -> String
where
    R: Rng,
{
    let f: f64 = match prop_dist {
        Dist::None => 0.,
        Dist::Uniform => UniformFloat::<f64>::new(0.0, 1.0).sample(rng),
        Dist::Normal => {
            let n = Normal::new(0.5, 0.5).unwrap().sample(rng);
            if n > 1. {
                1.
            } else if n < 0. {
                0.
            } else {
                n
            }
        }
        Dist::Exp => Exp::new(0.5).unwrap().sample(rng),
    };
    let mut buf = vec![0u8; (min_prop_size + ((f * prop_range as f64) as usize)) / 3];
    data_source
        .read_exact(&mut buf)
        .expect("Failed to read from urandom");
    urlencoding::encode_binary(&buf).to_string()
}

fn get_n_edges<R>(rng: &mut R, edge_dist: &Dist) -> usize
where
    R: Rng,
{
    let f: f64 = match edge_dist {
        Dist::None => 0.,
        Dist::Uniform => UniformFloat::<f64>::new(0.0, 1.0).sample(rng),
        Dist::Normal => {
            let n = Normal::new(0.5, 0.5).unwrap().sample(rng);
            if n > 1. {
                1.
            } else if n < 0. {
                0.
            } else {
                n
            }
        }
        Dist::Exp => {
            let e = Exp::new(0.5).unwrap().sample(rng);
            e * e
        }
    };

    (f * 10000.) as usize
}

fn generate_chunk(args: Args, id: usize) {
    let mut nodefile = BufWriter::new(
        File::create(format!("nodes_{}.csv", id)).expect("Failed to create thread-local nodes.csv"),
    );
    let mut edgefile = BufWriter::new(
        File::create(format!("edges_{}.csv", id)).expect("Failed to create thread-local edges.csv"),
    );
    let mut statsfile = BufWriter::new(
        File::create(format!("stats_{}.txt", id)).expect("Failed to create thread-local stats.txt"),
    );

    let prop_range = args.max_prop_size - args.min_prop_size;
    let mut data_source = File::open("/dev/urandom").expect("Failed to open urandom");

    let chunksize = args.n_nodes / args.nprocs;
    let start = chunksize * id;
    let end = std::cmp::min(start + chunksize, args.n_nodes);

    let mut rng = rand::thread_rng();

    let has_node_props = args.node_prop_dist != Dist::None;
    let has_edge_props = args.edge_prop_dist != Dist::None;

    let mut node_line = String::with_capacity(
        format!("{}\n", args.n_nodes).len()
            + if has_node_props {
                // If we are adding node properties, reserve space for them. 1 bytes for '|' plus
                // the actual property
                1 + args.max_prop_size * 4
            } else {
                0
            },
    );
    let two_ids_len = format!("{}|{}\n", args.n_nodes, args.n_nodes).len();
    let mut edge_line = String::with_capacity(
        two_ids_len
            + if has_edge_props {
                // If we are adding edge properties, reserve space for them. 1 bytes for '|' plus
                // the actual property
                1 + args.max_prop_size * 4
            } else {
                0
            },
    );
    let mut stats_line = String::with_capacity(two_ids_len);

    for nid in tqdm(start..end) {
        let nid_str = &nid.to_string();

        node_line.push_str(&nid_str);
        if has_node_props {
            node_line.push_str("|");
            node_line.push_str(&get_prop(
                &mut rng,
                &mut data_source,
                &args.node_prop_dist,
                args.min_prop_size,
                prop_range,
            ));
        }
        node_line.push_str("\n");
        nodefile
            .write_all(node_line.as_bytes())
            .expect("Failed to write node");
        node_line.clear();

        let n_edges = get_n_edges(&mut rng, &args.edge_dist);
        stats_line.push_str(&nid_str);
        stats_line.push_str(" ");
        stats_line.push_str(&n_edges.to_string());
        stats_line.push_str("\n");
        statsfile
            .write_all(stats_line.as_bytes())
            .expect("Failed to write stats");
        stats_line.clear();

        edge_line.push_str(&nid_str);
        edge_line.push_str("|");
        let prefix_len = edge_line.len();
        for _ in 0..n_edges {
            // If we are generating individual chunks, we must ensure that each chunk only refers to
            // nodes that exist within this chunk, or any prior chunks.
            let maxnid = if args.generatechunks {
                end
            } else {
                args.n_nodes
            };
            let dst = rand::thread_rng().gen_range(0..maxnid);
            edge_line.push_str(&dst.to_string());
            if has_edge_props {
                edge_line.push_str("|");
                edge_line.push_str(&get_prop(
                    &mut rng,
                    &mut data_source,
                    &args.edge_prop_dist,
                    args.min_prop_size,
                    prop_range,
                ));
            }
            edge_line.push_str("\n");
            edgefile
                .write_all(edge_line.as_bytes())
                .expect("Failed to write edge");
            edge_line.truncate(prefix_len);
        }
        edge_line.clear();
    }
}

fn combine_chunks(args: &Args) {
    let nodefile = OpenOptions::new()
        .read(true)
        .write(true)
        .open("nodes.csv")
        .expect("Failed to create nodes.csv");
    let edgefile = OpenOptions::new()
        .read(true)
        .write(true)
        .open("edges.csv")
        .expect("Failed to create edges.csv");

    // Determine how big each chunk was and compute a prefix sum of the lengths (note that the
    // output files already have headers, so we need to account for those in the offsets)
    let mut nodefiles = vec![];
    let mut nodeoffsets = vec![nodefile.metadata().unwrap().len()];
    let mut edgefiles = vec![];
    let mut edgeoffsets = vec![edgefile.metadata().unwrap().len()];
    for i in tqdm(0..args.nprocs) {
        let childnodefile = File::open(format!("nodes_{}.csv", i)).unwrap();
        nodeoffsets.push(nodeoffsets[i] + childnodefile.metadata().unwrap().len());
        nodefiles.push(childnodefile);

        let childedgefile = File::open(format!("edges_{}.csv", i)).unwrap();
        edgeoffsets.push(edgeoffsets[i] + childedgefile.metadata().unwrap().len());
        edgefiles.push(childedgefile);
    }
    // Set the node/edge files to be as big as the sum of all chunks
    nodefile
        .set_len(*nodeoffsets.last().unwrap())
        .expect("Failed to set length of node file");
    edgefile
        .set_len(*edgeoffsets.last().unwrap())
        .expect("Failed to set length of edge file");
    drop(nodefile);
    drop(edgefile);

    // For each chunk, create a thread and open the output files at an offset from the prefix sum,
    // so that each thread has a unique range to copy a chunk into.
    let mut children = vec![];
    for i in 0..args.nprocs {
        let nstart = nodeoffsets[i];
        let estart = edgeoffsets[i];

        let total_bytes = nodeoffsets[i + 1] - nstart + edgeoffsets[i + 1] - estart;
        children.push(thread::spawn(move || {
            let mut nodefile = OpenOptions::new()
                .write(true)
                .open("nodes.csv")
                .expect("Failed to create nodes.csv");
            nodefile
                .seek(SeekFrom::Start(nstart))
                .expect("Failed to seek nodes.csv");

            let mut edgefile = OpenOptions::new()
                .write(true)
                .open("edges.csv")
                .expect("Failed to create edges.csv");
            edgefile
                .seek(SeekFrom::Start(estart))
                .expect("Failed to seek edges.csv");

            let childnodes_name = format!("nodes_{}.csv", i);
            let childedges_name = format!("edges_{}.csv", i);

            let childnodefile =
                BufReader::new(File::open(&childnodes_name).expect("Failed to open node file"));
            let childedgefile =
                BufReader::new(File::open(&childedges_name).expect("Failed to open edge file"));

            let mut nodefile = BufWriter::new(nodefile);
            let mut edgefile = BufWriter::new(edgefile);

            let mut pbar = pbar(Some(total_bytes as usize));

            for line in childnodefile.lines().flatten() {
                nodefile
                    .write_all(line.as_bytes())
                    .expect("Failed to write to nodefile");

                nodefile
                    .write_all(b"\n")
                    .expect("Failed to write to nodefile");
                pbar.update(line.as_bytes().len() + 1).expect("");
            }
            std::fs::remove_file(childnodes_name).expect("failed to remove child node file");

            for line in childedgefile.lines().flatten() {
                edgefile
                    .write_all(line.as_bytes())
                    .expect("Failed to write to edgefile");

                edgefile
                    .write_all(b"\n")
                    .expect("Failed to write to edgefile");
                pbar.update(line.as_bytes().len() + 1).expect("");
            }
            std::fs::remove_file(childedges_name).expect("failed to remove child edge file");
        }));
    }
    for child in children {
        child.join().expect("Failed to join thread");
    }
}

fn generate(args: Args) {
    create_dir(&args.outdir);
    assert!(env::set_current_dir(&args.outdir).is_ok());

    {
        let mut nodefile = File::create("nodes.csv").expect("Failed to create nodes.csv");
        nodefile
            .write_all(b"NodeID|data\n")
            .expect("Failed to write node header");
        let mut edgefile = File::create("edges.csv").expect("Failed to create edges.csv");
        edgefile
            .write_all(b"SrcID|DstID\n")
            .expect("Failed to write edge header");
    }

    let mut children = vec![];
    for i in 0..args.nprocs {
        let args_copy = args.clone();
        children.push(thread::spawn(move || {
            generate_chunk(args_copy, i);
        }));
    }

    for child in children {
        child.join().expect("Failed to join thread");
    }

    // If we're generating chunks, we can just exit here since we don't need to combine all the
    // files at the end.
    if args.generatechunks {
        return;
    }
    combine_chunks(&args);
}

fn main() {
    let args = Args::parse();
    generate(args);
}
