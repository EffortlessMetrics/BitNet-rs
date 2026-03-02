$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
cargo check -p xtask 2>&1
