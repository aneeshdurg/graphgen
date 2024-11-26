use clap::{Parser, ValueEnum};
use rand;
use rand::distributions::uniform::UniformFloat;
use rand::Rng;
use rand_distr::uniform::UniformSampler;
use rand_distr::{Distribution, Exp, Normal};
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufRead};
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::thread;
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

fn get_prop(
    data_source: &mut File,
    prop_dist: &Dist,
    min_prop_size: usize,
    prop_range: usize,
) -> String {
    let mut rng = rand::thread_rng();

    let f: f64 = match prop_dist {
        Dist::Uniform => UniformFloat::<f64>::new(0.0, 1.0).sample(&mut rng),
        Dist::Normal => {
            let n = Normal::new(0.5, 0.5).unwrap().sample(&mut rng);
            if n > 1. {
                1.
            } else if n < 0. {
                0.
            } else {
                n
            }
        }
        Dist::Exp => Exp::new(0.5).unwrap().sample(&mut rng),
    };
    let mut buf = vec![0u8; min_prop_size + ((f * prop_range as f64) as usize)];
    data_source
        .read_exact(&mut buf)
        .expect("Failed to read from urandom");
    urlencoding::encode_binary(&buf).to_string()
}

fn get_n_edges(edge_dist: &Dist) -> usize {
    let mut rng = rand::thread_rng();

    let f: f64 = match edge_dist {
        Dist::Uniform => UniformFloat::<f64>::new(0.0, 1.0).sample(&mut rng),
        Dist::Normal => {
            let n = Normal::new(0.5, 0.5).unwrap().sample(&mut rng);
            if n > 1. {
                1.
            } else if n < 0. {
                0.
            } else {
                n
            }
        }
        Dist::Exp => {
            let e = Exp::new(0.5).unwrap().sample(&mut rng);
            e * e
        }
    };

    (f * 10000.) as usize
}

fn generate_chunk(args: Args, id: usize) {
    let mut nodefile =
        File::create(format!("nodes_{}.csv", id)).expect("Failed to create thread-local nodes.csv");
    let mut edgefile =
        File::create(format!("edges_{}.csv", id)).expect("Failed to create thread-local edges.csv");
    let mut statsfile =
        File::create(format!("stats_{}.txt", id)).expect("Failed to create thread-local stats.txt");

    let prop_range = args.max_prop_size - args.min_prop_size;
    let mut data_source = File::open("/dev/urandom").expect("Failed to open urandom");
    //let mut buf = vec![0u8; bytes_to_read];
    // reader.read_exact(&mut buf)?;
    let chunksize = args.n_nodes / args.nprocs;
    let start = chunksize * id;
    let end = std::cmp::min(start + chunksize, args.n_nodes);

    for nid in tqdm(start..end) {
        let prop = get_prop(
            &mut data_source,
            &args.prop_dist,
            args.min_prop_size,
            prop_range,
        );
        let node = format!("{}|{}\n", nid, prop);
        nodefile
            .write_all(node.as_bytes())
            .expect("Failed to write node");

        let n_edges = get_n_edges(&args.edge_dist);
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

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn generate(args: Args) {
    create_dir(&args.outdir);
    assert!(env::set_current_dir(&args.outdir).is_ok());

    let mut nodefile = File::create("nodes.csv").expect("Failed to create nodes.csv");
    nodefile
        .write_all(b"NodeID|data\n")
        .expect("Failed to write node header");

    let mut edgefile = File::create("edges.csv").expect("Failed to create edges.csv");
    edgefile
        .write_all(b"SrcID|DstID\n")
        .expect("Failed to write edge header");

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
        if let Ok(lines) = read_lines(&childnodes) {
            // Consumes the iterator, returns an (Optional) String
            for line in lines.flatten() {
                nodefile
                    .write_all(line.as_bytes())
                    .expect("Failed to write to nodefile");

                nodefile
                    .write_all(b"\n")
                    .expect("Failed to write to nodefile");
            }
        }
        Command::new("rm")
            .arg(childnodes)
            .output()
            .expect("failed to remove child node file");

        let childedges = format!("edges_{}.csv", i);
        if let Ok(lines) = read_lines(&childedges) {
            // Consumes the iterator, returns an (Optional) String
            for line in lines.flatten() {
                edgefile
                    .write_all(line.as_bytes())
                    .expect("Failed to write to edgefile");

                edgefile
                    .write_all(b"\n")
                    .expect("Failed to write to edgefile");
            }
        }
        Command::new("rm")
            .arg(childedges)
            .output()
            .expect("Failed to remove child edge file");
        i += 1;
    }
}

fn main() {
    let args = Args::parse();
    generate(args);
}
