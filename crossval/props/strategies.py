"""
Adversarial prompt generation strategies using Hypothesis.
Generates diverse, challenging prompts that stress-test model behavior.
"""
from hypothesis import strategies as st
import string
import random


# Safe punctuation (exclude braces to avoid JSON confusion in prompts)
PUNCT = "".join(ch for ch in string.punctuation if ch not in "{}")


def s_whitespace():
    """Generate various whitespace patterns."""
    return st.text(alphabet=" \t\n\r", min_size=0, max_size=10)


def s_noise():
    """Generate Unicode noise (excluding invalid categories)."""
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs", "Cc", "Cf")),
        min_size=0,
        max_size=60
    )


def s_lang():
    """Multilingual instruction prefixes testing Unicode handling."""
    return st.sampled_from([
        "Explain in English:",
        "Explique en français:",
        "Explique en español:",
        "説明してください:",
        "请解释：",
        "Поясните:",
        "شرح باللغة العربية:",
        "Erklären Sie auf Deutsch:",
        "Spiega in italiano:",
    ])


def s_json_task():
    """Tasks requesting structured JSON output."""
    return st.sampled_from([
        'Return a valid JSON object with keys "title" and "items" (array of 3 strings).',
        'Respond ONLY with JSON: {"answer": <integer>, "reason": <string>}.',
        'Output JSON with {"lang": "en", "summary": <string of <= 25 words>}.',
        'Generate JSON: {"status": "ok"|"error", "data": <any>}.',
        'Create a JSON array of 5 numbers between 1 and 100.',
    ])


def s_code_task():
    """Programming tasks across languages."""
    return st.sampled_from([
        "Write a Rust function that reverses a string and include a short example.",
        "Given an array of integers, show a Python snippet to compute the median.",
        "Provide a tiny JSON Schema for an object {id: string, score: number}.",
        "Create a JavaScript arrow function to filter even numbers from an array.",
        "Write a SQL query to find duplicate emails in a users table.",
        "Show a bash one-liner to count lines in all .txt files.",
        "Implement FizzBuzz in Go for numbers 1 to 20.",
    ])


def s_math_task():
    """Mathematical tasks with verifiable answers."""
    return st.sampled_from([
        "Compute 12345 * 678 and explain briefly.",
        "What is the derivative of x^3 + 2x at x = 5? Show steps.",
        "Simplify (3/4 + 5/6) as a fraction.",
        "Find the GCD of 48 and 18 using Euclidean algorithm.",
        "Calculate 2^10 without using a calculator.",
        "What is the sum of integers from 1 to 100?",
        "Solve for x: 2x + 7 = 23",
    ])


def s_instruction_task():
    """Step-by-step instruction tasks."""
    return st.sampled_from([
        "List 3 steps to make a peanut butter sandwich.",
        "How do you change a flat tire? Give 5 key steps.",
        "Explain how to tie a shoelace in 4 steps.",
        "Describe the water cycle in 3 phases.",
        "What are the stages of the software development lifecycle?",
    ])


def s_edge_cases():
    """Edge case prompts that stress tokenization and handling."""
    return st.sampled_from([
        "",  # Empty prompt
        " " * 50,  # Only whitespace
        "a" * 200,  # Repetitive
        "😀" * 10,  # Only emoji
        "\n\n\n\n\n",  # Only newlines
        "." * 30,  # Only punctuation
        "NaN undefined null None nil NULL",  # Special values
        "0 1 -1 3.14159 -273.15 1e308",  # Numbers
        "\\n\\t\\r\\x00\\u0000",  # Escape sequences
    ])


def s_wrap(content):
    """Wrap content with random whitespace and punctuation."""
    ws_before = st.text(alphabet=" \t\n", min_size=0, max_size=10)
    ws_after = st.text(alphabet=" \t\n", min_size=0, max_size=10)
    punct_before = st.text(alphabet=PUNCT, min_size=0, max_size=5)
    punct_after = st.text(alphabet=PUNCT, min_size=0, max_size=5)

    return st.tuples(
        ws_before, punct_before,
        content,
        punct_after, ws_after
    ).map(lambda t: "".join(t))


def s_adversarial_unicode():
    """Unicode edge cases that stress normalization."""
    return st.sampled_from([
        "Zürich café naïve résumé",  # Diacritics
        "🔥💯✨ Test 测试 テスト",  # Mixed scripts
        "ﬁle ﬀort",  # Ligatures
        "𝓗𝓮𝓵𝓵𝓸 𝕎𝕠𝕣𝕝𝕕",  # Mathematical alphanumeric
        "⓵⓶⓷ ➊➋➌",  # Circled numbers
        "𐀀𐀁𐀂",  # Linear B
        "‏RTL‏ ‎LTR‎ mixing",  # Direction marks
        "​Zero​Width​Spaces​",  # Zero-width spaces
    ])


def s_length_varied():
    """Vary prompt lengths to test buffer handling."""
    short = st.text(string.ascii_letters + string.digits, min_size=1, max_size=10)
    medium = st.text(string.printable, min_size=50, max_size=200)
    long = st.text(string.printable, min_size=500, max_size=1000)

    return st.one_of(short, medium, long)


def prompt_strategy():
    """
    Main prompt generation strategy.
    Combines various sub-strategies to create diverse, adversarial prompts.
    """
    # Core task types
    base_task = st.one_of(
        s_json_task(),
        s_code_task(),
        s_math_task(),
        s_instruction_task(),
    )

    # Optional modifiers
    lang_prefix = st.one_of(st.just(""), s_lang())
    noise_suffix = st.one_of(st.just(""), s_noise())

    # Combine with wrapping
    normal_prompt = st.tuples(lang_prefix, base_task, noise_suffix).map(
        lambda t: " ".join(x for x in t if x)
    )

    # Mix in edge cases and unicode stress tests
    edge_prompt = s_edge_cases()
    unicode_prompt = s_adversarial_unicode()
    length_prompt = s_length_varied()

    # Weighted combination (70% normal, 30% edge/adversarial)
    return st.one_of(
        normal_prompt,
        normal_prompt,
        normal_prompt,
        normal_prompt,
        normal_prompt,
        normal_prompt,
        normal_prompt,  # 70% normal
        edge_prompt,     # 10% edge
        unicode_prompt,  # 10% unicode
        length_prompt,   # 10% length stress
    )


def reproducible_prompt_set(seed: int = 42, count: int = 100):
    """
    Generate a fixed set of prompts for reproducible testing.
    Useful for regression testing and benchmarking.
    """
    random.seed(seed)
    prompts = []

    # Ensure diversity
    categories = [
        s_json_task(),
        s_code_task(),
        s_math_task(),
        s_instruction_task(),
        s_edge_cases(),
        s_adversarial_unicode(),
    ]

    for i in range(count):
        cat = categories[i % len(categories)]
        prompt = cat.example()
        prompts.append(prompt)

    return prompts
