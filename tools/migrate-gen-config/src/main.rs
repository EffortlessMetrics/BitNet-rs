use quote::quote;
use std::{fs, path::Path};
use syn::{visit_mut::VisitMut, *};
use walkdir::WalkDir;

fn is_test_rs(path: &Path) -> bool {
    let s = path.display().to_string();
    s.ends_with(".rs")
        && (s.contains("/tests/")
            || s.contains("/test/")
            || s.contains("\\tests\\")
            || s.contains("\\test\\"))
}

struct GenConfigRewriter;

impl VisitMut for GenConfigRewriter {
    fn visit_expr_mut(&mut self, node: &mut Expr) {
        // First, recursively visit children
        syn::visit_mut::visit_expr_mut(self, node);

        // Then check if this is a GenerationConfig struct literal
        if let Expr::Struct(expr_struct) = node
            && let Some(path) = expr_struct.path.segments.last()
            && path.ident == "GenerationConfig"
        {
            // Collect fields into builder calls
            let mut builder_calls = Vec::<proc_macro2::TokenStream>::new();

            for field in &expr_struct.fields {
                let field_name = match &field.member {
                    Member::Named(ident) => ident.to_string(),
                    Member::Unnamed(_) => continue,
                };
                let expr = &field.expr;

                let call = match field_name.as_str() {
                    "max_new_tokens" => {
                        quote!( .with_max_tokens(#expr) )
                    }
                    "temperature" => {
                        quote!( .with_temperature(#expr) )
                    }
                    "top_k" => {
                        quote!( .with_top_k(#expr) )
                    }
                    "top_p" => {
                        quote!( .with_top_p(#expr) )
                    }
                    "repetition_penalty" => {
                        quote!( .with_repetition_penalty(#expr) )
                    }
                    "stop_sequences" => {
                        quote!( .with_stop_sequences(#expr) )
                    }
                    "stop_token_ids" => {
                        quote!( .with_stop_token_ids(#expr) )
                    }
                    "stop_string_window" => {
                        quote!( .with_stop_string_window(#expr) )
                    }
                    "seed" => {
                        // Handle Option<u64> - if it's Some(x), unwrap; if None, skip
                        quote!( .with_seed(#expr.unwrap_or(42)) )
                    }
                    "skip_special_tokens" => {
                        quote!( .with_skip_special_tokens(#expr) )
                    }
                    "eos_token_id" => {
                        quote!( .with_eos_token_id(#expr) )
                    }
                    "logits_tap_steps" => {
                        quote!( .with_logits_tap_steps(#expr) )
                    }
                    "logits_topk" => {
                        quote!( .with_logits_topk(#expr) )
                    }
                    "add_bos" => {
                        quote!( .with_add_bos(#expr) )
                    }
                    "logits_cb" => {
                        quote!( .with_logits_cb(#expr) )
                    }
                    other => {
                        eprintln!("NOTE: unknown GenerationConfig field `{}`, skipping", other);
                        continue;
                    }
                };
                builder_calls.push(call);
            }

            // Create the builder chain
            let chain: proc_macro2::TokenStream = if builder_calls.is_empty() {
                quote!(GenerationConfig::greedy())
            } else {
                quote! {
                    GenerationConfig::greedy()
                    #(#builder_calls)*
                }
            };

            // Replace struct literal with the chained builder expression
            match syn::parse2::<Expr>(chain) {
                Ok(new_expr) => {
                    *node = new_expr;
                }
                Err(e) => {
                    eprintln!("ERROR: Failed to parse builder chain: {}", e);
                }
            }
        }
    }
}

fn main() {
    let root = std::env::var("ROOT").unwrap_or_else(|_| ".".into());
    let root = Path::new(&root);

    let mut files_rewritten = 0;
    let mut total_files_checked = 0;

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if !is_test_rs(path) {
            continue;
        }
        // Only touch inference tests by default
        if !path.display().to_string().contains("bitnet-inference") {
            continue;
        }

        total_files_checked += 1;

        let src = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("WARNING: Failed to read {}: {}", path.display(), e);
                continue;
            }
        };

        let file: File = match syn::parse_file(&src) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("WARNING: Failed to parse {}: {}", path.display(), e);
                continue;
            }
        };

        let mut file = file;
        let mut rewriter = GenConfigRewriter;
        rewriter.visit_file_mut(&mut file);

        let new_src = prettyplease::unparse(&file);
        if new_src != src {
            match fs::write(path, new_src) {
                Ok(_) => {
                    println!("✓ Rewrote {}", path.display());
                    files_rewritten += 1;
                }
                Err(e) => {
                    eprintln!("ERROR: Failed to write {}: {}", path.display(), e);
                }
            }
        }
    }

    println!("\n=== Migration Summary ===");
    println!("Files checked: {}", total_files_checked);
    println!("Files rewritten: {}", files_rewritten);
    if files_rewritten > 0 {
        println!("\nRun 'cargo fmt --all' to format the rewritten files.");
    }
}
