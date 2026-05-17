#[path = "../web/warp_server.rs"]
mod shared;

fn main() -> anyhow::Result<()> {
    shared::main()
}
