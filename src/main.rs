use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::path::PathBuf;
use std::process::Command;
use std::thread;

use clap::{Parser, ValueEnum};
use rand::distributions::uniform::UniformFloat;
use rand::Rng;
use rand_distr::uniform::UniformSampler;
use rand_distr::{Distribution, Exp, Normal};
use tqdm::tqdm;

#[derive(Clone, Debug, ValueEnum)]
enum Dist {
    Uniform,
    Normal,
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

    /// Edge distribution
    #[arg(long)]
    prop_dist: Dist,

    /// Output directory
    #[arg(long, default_value = ".")]
    outdir: PathBuf,

    /// Number of processes
    #[arg(long, default_value = "8")]
    nprocs: usize,
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
    let mut buf = vec![0u8; min_prop_size + ((f * prop_range as f64) as usize)];
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
    //let mut buf = vec![0u8; bytes_to_read];
    // reader.read_exact(&mut buf)?;
    let chunksize = args.n_nodes / args.nprocs;
    let start = chunksize * id;
    let end = std::cmp::min(start + chunksize, args.n_nodes);

    let mut rng = rand::thread_rng();

    for nid in tqdm(start..end) {
        let prop = get_prop(
            &mut rng,
            &mut data_source,
            &args.prop_dist,
            args.min_prop_size,
            prop_range,
        );
        let node = format!("{}|{}\n", nid, prop);
        nodefile
            .write_all(node.as_bytes())
            .expect("Failed to write node");

        let n_edges = get_n_edges(&mut rng, &args.edge_dist);
        let stats = format!("{} {}\n", nid, n_edges);
        statsfile
            .write_all(stats.as_bytes())
            .expect("Failed to write stats");
        for _ in 0..n_edges {
            let dst = rand::thread_rng().gen_range(0..args.n_nodes);
            let edge = format!("{}|{}\n", nid, dst);
            edgefile
                .write_all(edge.as_bytes())
                .expect("Failed to write edge");
        }
    }
}

fn generate(args: Args) {
    create_dir(&args.outdir);
    assert!(env::set_current_dir(&args.outdir).is_ok());

    {
        let mut nodefile =
            BufWriter::new(File::create("nodes.csv").expect("Failed to create nodes.csv"));
        nodefile
            .write_all(b"NodeID|data\n")
            .expect("Failed to write node header");
    }

    {
        let mut edgefile =
            BufWriter::new(File::create("edges.csv").expect("Failed to create edges.csv"));
        edgefile
            .write_all(b"SrcID|DstID\n")
            .expect("Failed to write edge header");
    }

    // Make a vector to hold the children which are spawned.
    let mut children = vec![];

    for i in 0..args.nprocs {
        let args_copy = args.clone();
        // Spin up another thread
        children.push(thread::spawn(move || {
            generate_chunk(args_copy, i);
        }));
    }

    let mut i = 0;
    for child in tqdm(children) {
        let _ = child.join();
        let childnodes = format!("nodes_{}.csv", i);
        let childedges = format!("edges_{}.csv", i);
        Command::new("dd")
            .arg(format!("if={}", childnodes))
            .arg("bs=4k")
            .arg("of=nodes.csv")
            .arg("oflag=append")
            .output()
            .expect("concat'ing node files failed");
        Command::new("dd")
            .arg(format!("if={}", childedges))
            .arg("bs=4k")
            .arg("of=edges.csv")
            .arg("oflag=append")
            .output()
            .expect("concat'ing node files failed");

        std::fs::remove_file(childnodes).expect("failed to remove child node file");
        std::fs::remove_file(childedges).expect("failed to remove child node file");
        i += 1;
    }
}

fn main() {
    let args = Args::parse();
    generate(args);
}
