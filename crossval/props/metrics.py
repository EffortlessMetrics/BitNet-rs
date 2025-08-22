"""
Hard-to-game metrics for comparing model outputs.
These metrics capture different aspects of similarity that are difficult to manipulate.
"""
import math
import re
import json as _json
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter


def levenshtein(a: List[int], b: List[int]) -> int:
    """
    Compute Levenshtein (edit) distance between two sequences.
    Classic dynamic programming implementation at token level.
    
    Lower is better (0 = identical).
    """
    if not a:
        return len(b)
    if not b:
        return len(a)
    
    # DP table
    dp = [list(range(len(b) + 1))]
    
    for i, token_a in enumerate(a, 1):
        row = [i] + [0] * len(b)
        for j, token_b in enumerate(b, 1):
            cost = 0 if token_a == token_b else 1
            row[j] = min(
                dp[-1][j] + 1,      # deletion
                row[j-1] + 1,       # insertion
                dp[-1][j-1] + cost  # substitution
            )
        dp.append(row)
    
    return dp[-1][-1]


def prefix_match_len(a: List[int], b: List[int]) -> int:
    """
    Length of longest common prefix between two sequences.
    
    Higher is better (max = min(len(a), len(b))).
    """
    i = 0
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return i


def suffix_match_len(a: List[int], b: List[int]) -> int:
    """
    Length of longest common suffix between two sequences.
    
    Higher is better.
    """
    if not a or not b:
        return 0
    
    i = 0
    max_len = min(len(a), len(b))
    while i < max_len and a[-(i+1)] == b[-(i+1)]:
        i += 1
    return i


def ngram_f1(a_tokens: List[str], b_tokens: List[str], n: int = 2) -> float:
    """
    N-gram F1 score between two sequences.
    Captures local structure similarity.
    
    Range: [0, 1], higher is better.
    """
    def get_ngrams(tokens):
        if len(tokens) < n:
            # For sequences shorter than n, use the sequence itself
            return Counter([tuple(tokens)]) if tokens else Counter()
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    A = get_ngrams(a_tokens)
    B = get_ngrams(b_tokens)
    
    if not A and not B:
        return 1.0  # Both empty = perfect match
    if not A or not B:
        return 0.0  # One empty = no match
    
    # Intersection
    intersection = sum((A & B).values())
    
    # Precision and recall
    precision = intersection / (sum(A.values()) or 1)
    recall = intersection / (sum(B.values()) or 1)
    
    # F1
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def token_ids(text: str) -> List[str]:
    """
    Simple whitespace tokenization for language-agnostic comparison.
    Returns list of non-whitespace tokens.
    """
    import re
    return re.findall(r"\S+", text)


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    """
    Jaccard similarity coefficient (set-based).
    Good for catching vocabulary differences.
    
    Range: [0, 1], higher is better.
    """
    if not a and not b:
        return 1.0
    
    set_a = set(a)
    set_b = set(b)
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def longest_common_subsequence(a: List[int], b: List[int]) -> int:
    """
    Length of longest common subsequence (not necessarily contiguous).
    Captures overall structural similarity.
    
    Higher is better.
    """
    if not a or not b:
        return 0
    
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def basic_text_metrics(a: str, b: str) -> Dict[str, float]:
    """
    Compute standard text similarity metrics.
    Returns dict with multiple metrics for comprehensive comparison.
    """
    A = token_ids(a)
    B = token_ids(b)
    
    # Convert to int IDs for sequence metrics
    vocab = list(set(A + B))
    vocab_map = {token: i for i, token in enumerate(vocab)}
    A_ids = [vocab_map[t] for t in A]
    B_ids = [vocab_map[t] for t in B]
    
    metrics = {
        "levenshtein": levenshtein(A_ids, B_ids),
        "prefix_match": prefix_match_len(A_ids, B_ids),
        "suffix_match": suffix_match_len(A_ids, B_ids),
        "bigram_f1": ngram_f1(A, B, n=2),
        "trigram_f1": ngram_f1(A, B, n=3),
        "jaccard": jaccard_similarity(A, B),
        "lcs_len": longest_common_subsequence(A_ids, B_ids),
        "len_a": len(A),
        "len_b": len(B),
        "len_diff": abs(len(A) - len(B)),
    }
    
    # Normalized metrics
    max_len = max(len(A), len(B))
    if max_len > 0:
        metrics["levenshtein_norm"] = metrics["levenshtein"] / max_len
        metrics["prefix_match_norm"] = metrics["prefix_match"] / max_len
        metrics["lcs_norm"] = metrics["lcs_len"] / max_len
    else:
        metrics["levenshtein_norm"] = 0.0
        metrics["prefix_match_norm"] = 1.0
        metrics["lcs_norm"] = 1.0
    
    return metrics


def logit_rank_correlation(logits_a: List[List[Tuple[int, float]]], 
                           logits_b: List[List[Tuple[int, float]]]) -> Optional[float]:
    """
    Compute rank correlation between top-k logits at each step.
    Input: list of steps, each containing list of (token_id, logit) tuples.
    
    Returns Kendall's tau correlation averaged over steps.
    Range: [-1, 1], higher is better (1 = perfect agreement).
    """
    if not logits_a or not logits_b:
        return None
    
    from scipy.stats import kendalltau
    
    correlations = []
    min_steps = min(len(logits_a), len(logits_b))
    
    for step in range(min_steps):
        # Get token IDs and ranks
        tokens_a = {tok_id: rank for rank, (tok_id, _) in enumerate(logits_a[step])}
        tokens_b = {tok_id: rank for rank, (tok_id, _) in enumerate(logits_b[step])}
        
        # Find common tokens
        common = set(tokens_a.keys()) & set(tokens_b.keys())
        if len(common) < 2:
            continue
        
        # Get ranks for common tokens
        ranks_a = [tokens_a[t] for t in sorted(common)]
        ranks_b = [tokens_b[t] for t in sorted(common)]
        
        # Compute correlation
        tau, _ = kendalltau(ranks_a, ranks_b)
        if not math.isnan(tau):
            correlations.append(tau)
    
    if not correlations:
        return None
    
    return sum(correlations) / len(correlations)


def perplexity_from_logits(logits: List[float], targets: List[int]) -> float:
    """
    Compute perplexity from logits and target token IDs.
    Lower is better (1 = perfect prediction).
    """
    if not logits or not targets:
        return float('inf')
    
    total_loss = 0.0
    count = 0
    
    for logit_vec, target in zip(logits, targets):
        # Softmax
        exp_logits = np.exp(logit_vec - np.max(logit_vec))
        probs = exp_logits / np.sum(exp_logits)
        
        # NLL for target
        if target < len(probs):
            prob = max(probs[target], 1e-10)  # Avoid log(0)
            total_loss -= math.log(prob)
            count += 1
    
    if count == 0:
        return float('inf')
    
    return math.exp(total_loss / count)


def combined_similarity_score(metrics: Dict[str, float]) -> float:
    """
    Combine multiple metrics into a single similarity score.
    Designed to be hard to game - requires matching multiple aspects.
    
    Range: [0, 1], higher is better.
    """
    # Weight different aspects
    weights = {
        "prefix_match_norm": 0.25,  # Early agreement important
        "bigram_f1": 0.20,          # Local structure
        "levenshtein_norm": 0.20,   # Overall edit distance (inverted)
        "lcs_norm": 0.15,           # Global structure
        "jaccard": 0.10,            # Vocabulary overlap
        "trigram_f1": 0.10,         # Longer local patterns
    }
    
    score = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            value = metrics[metric]
            
            # Invert distance metrics
            if "levenshtein" in metric:
                value = 1.0 - value
            
            score += weight * value
    
    return score


# JSON extraction and validation
JSON_BLOCK = re.compile(r"(\{.*?\}|\[.*?\])", re.DOTALL)
CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[Any]:
    """
    Extract and validate JSON from text.
    Handles common wrapping patterns (code fences, prose).
    
    Returns parsed JSON object/array or None if invalid.
    """
    if not text:
        return None
    
    # Try direct parse first
    try:
        return _json.loads(text.strip())
    except:
        pass
    
    # Try removing code fences
    fence_match = CODE_FENCE.search(text)
    if fence_match:
        try:
            return _json.loads(fence_match.group(1).strip())
        except:
            pass
    
    # Try extracting JSON block from prose
    json_match = JSON_BLOCK.search(text)
    if json_match:
        try:
            return _json.loads(json_match.group(1))
        except:
            pass
    
    # Last resort: try to find balanced braces/brackets
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
            
            if char == "\\":
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return _json.loads(text[start_idx:i+1])
                    except:
                        break
    
    return None


def validate_json_schema(obj: Any, expected_keys: List[str]) -> bool:
    """
    Check if JSON object has expected keys.
    """
    if not isinstance(obj, dict):
        return False
    return all(key in obj for key in expected_keys)


def relative_metrics(metrics: Dict[str, float], ref_length: int) -> Dict[str, float]:
    """
    Add relative (length-normalized) versions of absolute metrics.
    """
    rel = {}
    ref_len = max(1, ref_length)
    
    if "levenshtein" in metrics:
        rel["levenshtein_rel"] = metrics["levenshtein"] / ref_len
    
    if "prefix_match" in metrics:
        rel["prefix_match_rel"] = metrics["prefix_match"] / ref_len
    
    if "lcs_len" in metrics:
        rel["lcs_rel"] = metrics["lcs_len"] / ref_len
    
    return rel


def kendalls_tau(topk_a_ids: List[int], topk_b_ids: List[int], variant: str = "b") -> float:
    """
    Kendall's tau over the intersection of token ids.
    Input lists are in descending logit order (rank 0 is best).
    
    Args:
        topk_a_ids: List of token IDs from system A (descending by logit)
        topk_b_ids: List of token IDs from system B (descending by logit)
        variant: "a" (ignores ties) or "b" (tie-aware, default)
    
    Returns tau in [-1, 1] where:
    - 1.0 = perfect agreement
    - 0.0 = no correlation  
    - -1.0 = perfect disagreement
    """
    # Build rank dictionaries
    A = {tid: r for r, tid in enumerate(topk_a_ids)}
    B = {tid: r for r, tid in enumerate(topk_b_ids)}
    
    # Find common tokens
    common = [tid for tid in A if tid in B]
    n = len(common)
    
    if n < 2:
        return 0.0
    
    # Get ranks for common tokens
    ranks = [(A[t], B[t]) for t in common]
    
    # Count concordant, discordant, and ties
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    ties_both = 0
    
    for i in range(n):
        ai, bi = ranks[i]
        for j in range(i + 1, n):
            aj, bj = ranks[j]
            
            # Check relationship in A and B
            diff_a = ai - aj
            diff_b = bi - bj
            
            if diff_a == 0 and diff_b == 0:
                ties_both += 1
            elif diff_a == 0:
                ties_a += 1
            elif diff_b == 0:
                ties_b += 1
            elif (diff_a > 0 and diff_b > 0) or (diff_a < 0 and diff_b < 0):
                concordant += 1
            else:
                discordant += 1
    
    # Compute tau based on variant
    if variant == "a":
        # Tau-a: ignores ties
        denominator = n * (n - 1) / 2
        if denominator > 0:
            return (concordant - discordant) / denominator
    else:  # variant == "b"
        # Tau-b: tie-aware
        n0 = n * (n - 1) / 2
        n1 = n0 - ties_a - ties_both
        n2 = n0 - ties_b - ties_both
        
        if n1 > 0 and n2 > 0:
            return (concordant - discordant) / ((n1 * n2) ** 0.5)
    
    return 0.0