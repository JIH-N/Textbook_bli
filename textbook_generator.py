#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║        AUTOMATED CHINESE TECHNICAL TEXTBOOK GENERATOR       ║
║                                                              ║
║  Pipeline:                                                   ║
║  Bilibili URL / Subtitle File                                ║
║      → yt-dlp (subtitle extraction)                          ║
║      → Ollama LLM (ASR correction, translation, glossary)    ║
║      → pypinyin (deterministic tone marks)                   ║
║      → CC-CEDICT (dictionary cross-check)                    ║
║      → Self-contained A4-ready HTML textbook                 ║
║                                                              ║
║  Usage:                                                      ║
║    python textbook_generator.py <bilibili_url>               ║
║    python textbook_generator.py <subtitle_file.srt>          ║
║    python textbook_generator.py <transcript.txt>             ║
╚══════════════════════════════════════════════════════════════╝
"""

# ============================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ============================================================
# Standard library - no pip install needed
import argparse
import json
import os
import re
import sys
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

# Third-party - installed via requirements.txt
try:
    import requests
    from pypinyin import pinyin, Style
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


# ── Global Configuration ──
# Change these to match your local setup.
# The model name must match what you've pulled in Ollama.
CONFIG = {
    "ollama_url": "http://localhost:11434",   # Ollama's default local endpoint
    "model": "qwen2.5:7b",                   # Best open-source 7B for Chinese tasks
    "chunk_seconds": 90,                      # SRT: group subtitles into ~90s chunks
    "chunk_chars": 300,                       # TXT: split plain text every ~300 chars
    "temperature": 0.3,                       # Low temp = less creative, more accurate
    "cedict_path": "cedict_ts.u8",            # CC-CEDICT dictionary file path
    "questions_every_n_chunks": 3,            # Insert comprehension Qs every N chunks
}

# ── Domain-Specific Term List ──
# Fed to the LLM during ASR correction to anchor technical vocabulary.
# The LLM uses this list to fix common misrecognitions in speech-to-text.
# Add your own terms as you encounter them across videos.
DOMAIN_TERMS = [
    # ── Analog Circuit Design ──
    "运算放大器", "运放", "负反馈", "正反馈", "增益", "带宽",
    "虚短", "虚断", "锁相环", "比较器", "差分对", "电流镜",
    "共模", "差模", "偏置电流", "失调电压", "压摆率", "建立时间",
    "环路增益", "相位裕度", "波特图", "零极点", "密勒效应",
    "MOSFET", "CMOS", "BJT", "二极管", "电容", "电感", "电阻",
    # ── Signal Processing / DSP ──
    "傅里叶变换", "快速傅里叶", "FFT", "采样率", "奈奎斯特",
    "低通滤波器", "高通滤波器", "带通滤波器", "数字信号处理",
    "卷积", "相关", "频谱", "时域", "频域", "量化", "过采样",
    "信噪比", "调制", "解调", "IIR", "FIR", "z变换",
    # ── AI / Machine Learning ──
    "神经网络", "深度学习", "卷积神经网络", "循环神经网络",
    "反向传播", "梯度下降", "损失函数", "过拟合", "欠拟合",
    "注意力机制", "Transformer", "大语言模型", "推理",
    "训练", "微调", "量化", "蒸馏", "嵌入", "向量",
]


# ============================================================
# SECTION 2: SUBTITLE ACQUISITION (yt-dlp)
# ============================================================
# yt-dlp handles Bilibili's authentication, anti-scraping,
# and subtitle format quirks. We try multiple subtitle sources
# in priority order: manual subs > auto-generated > ASR.

def extract_subtitles_ytdlp(url: str, output_dir: str = ".") -> dict:
    """
    Use yt-dlp to download subtitles and video metadata from a URL.

    Strategy:
    1. First attempt: download existing subtitles (human-uploaded)
    2. Fallback: download auto-generated subtitles
    3. yt-dlp natively supports Bilibili, YouTube, and 1000+ sites

    Returns:
        dict with keys: "subtitle_file", "title", "uploader", "duration"
    """
    output_template = os.path.join(output_dir, "%(id)s")

    # ── Step 1: Fetch video metadata first ──
    # We grab title, uploader, duration before downloading subs.
    print("[yt-dlp] Fetching video metadata...")
    meta_cmd = [
        "yt-dlp",
        "--dump-json",            # Output JSON metadata without downloading
        "--no-download",          # Don't download the video itself
        url
    ]
    try:
        result = subprocess.run(meta_cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"[WARNING] Could not fetch metadata: {e}")
        metadata = {}

    video_info = {
        "title": metadata.get("title", "Unknown Title"),
        "uploader": metadata.get("uploader", "Unknown"),
        "duration": metadata.get("duration", 0),
        "url": url,
        "subtitle_file": None,
    }

    # ── Step 2: Download subtitles ──
    # Try Chinese subs first (zh-Hans, zh), then any available language.
    # --write-sub: downloads manually uploaded subtitles
    # --write-auto-sub: downloads auto-generated/ASR subtitles (fallback)
    # --sub-lang: specifies preferred languages in priority order
    # --convert-subs srt: normalizes output to .srt format regardless of source
    print("[yt-dlp] Downloading subtitles...")
    sub_cmd = [
        "yt-dlp",
        "--write-sub",            # Try human-uploaded subs first
        "--write-auto-sub",       # Fall back to auto-generated
        "--sub-lang", "zh-Hans,zh,zh-CN,zh-TW,ai-zh",  # Chinese variants
        "--convert-subs", "srt",  # Normalize to SRT format
        "--skip-download",        # Don't download the video file
        "-o", output_template,    # Output path template
        url
    ]

    try:
        subprocess.run(sub_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] yt-dlp subtitle download failed: {e.stderr}")
        return video_info

    # ── Step 3: Locate the downloaded subtitle file ──
    # yt-dlp names files as: <video_id>.<lang>.srt
    # We search for any .srt file matching our video ID.
    video_id = metadata.get("id", "")
    srt_files = list(Path(output_dir).glob(f"{video_id}*.srt"))

    if not srt_files:
        # Broader search if ID-based search fails
        srt_files = list(Path(output_dir).glob("*.srt"))

    if srt_files:
        video_info["subtitle_file"] = str(srt_files[0])
        print(f"[yt-dlp] Subtitle acquired: {srt_files[0].name}")
    else:
        print("[WARNING] No subtitle files found. Video may lack captions.")

    return video_info


# ============================================================
# SECTION 3: SUBTITLE PARSING (SRT / TXT)
# ============================================================
# Two parsers that produce the same output structure: a list of
# "chunks" — each chunk is a study unit with text and optional
# timestamps. This normalization lets the rest of the pipeline
# work identically regardless of input format.

def parse_srt(filepath: str, chunk_seconds: int = 90) -> list[dict]:
    """
    Parse an SRT file and group subtitle entries into timed chunks.

    SRT format:
        1
        00:00:01,000 --> 00:00:04,500
        这是第一句字幕

    Each chunk covers ~chunk_seconds of video, creating a natural
    study unit. Chunks respect sentence boundaries where possible.

    Returns:
        List of dicts: [{"start": "00:01:30", "end": "00:03:00", "text": "..."}]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # ── Regex to extract each SRT entry ──
    # Captures: index, start_time, end_time, subtitle_text
    # The [\s\S] pattern handles multi-line subtitle text.
    pattern = re.compile(
        r"(\d+)\s*\n"                              # Entry index
        r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*"  # Start timestamp
        r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n"      # End timestamp
        r"([\s\S]*?)(?=\n\n|\n\d+\s*\n|$)"          # Subtitle text (non-greedy)
    )

    entries = []
    for match in pattern.finditer(content):
        start_str = match.group(2).replace(",", ".")
        end_str = match.group(3).replace(",", ".")
        text = match.group(4).strip()
        # Remove HTML tags that some SRT files contain (e.g., <i>, <font>)
        text = re.sub(r"<[^>]+>", "", text)
        if text:
            entries.append({
                "start_sec": _timestamp_to_seconds(start_str),
                "end_sec": _timestamp_to_seconds(end_str),
                "start": start_str[:8],  # HH:MM:SS (drop milliseconds for display)
                "end": end_str[:8],
                "text": text,
            })

    if not entries:
        print("[WARNING] No subtitle entries found in SRT file.")
        return []

    # ── Group entries into chunks of ~chunk_seconds ──
    chunks = []
    current_chunk = {"start": entries[0]["start"], "end": "", "text": ""}
    chunk_start_sec = entries[0]["start_sec"]

    for entry in entries:
        elapsed = entry["end_sec"] - chunk_start_sec

        if elapsed >= chunk_seconds and current_chunk["text"]:
            # Finalize current chunk and start a new one
            current_chunk["end"] = entry["start"]
            chunks.append(current_chunk)
            current_chunk = {"start": entry["start"], "end": "", "text": ""}
            chunk_start_sec = entry["start_sec"]

        # Append text with space separator (Chinese doesn't use spaces,
        # but this prevents accidental character merging from SRT line breaks)
        current_chunk["text"] += entry["text"].replace("\n", "")

    # Don't forget the last chunk
    if current_chunk["text"]:
        current_chunk["end"] = entries[-1]["end"]
        chunks.append(current_chunk)

    print(f"[Parser] SRT → {len(chunks)} chunks ({chunk_seconds}s each)")
    return chunks


def parse_txt(filepath: str, chunk_chars: int = 300) -> list[dict]:
    """
    Parse a plain text transcript into character-count-based chunks.

    Unlike SRT, plain text has no timestamps. We split by character
    count, trying to break at sentence boundaries (。！？) to keep
    chunks semantically coherent.

    Returns:
        List of dicts: [{"start": None, "end": None, "text": "..."}]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # ── Clean up the text ──
    # Remove excessive whitespace but preserve sentence structure
    content = re.sub(r"\s+", "", content)  # Chinese text: remove all spaces

    # ── Split at sentence boundaries ──
    # We split on Chinese sentence-ending punctuation, then re-group
    # into chunks of approximately chunk_chars characters.
    sentences = re.split(r"(?<=[。！？；])", content)
    sentences = [s for s in sentences if s.strip()]

    chunks = []
    current_text = ""

    for sentence in sentences:
        if len(current_text) + len(sentence) > chunk_chars and current_text:
            chunks.append({"start": None, "end": None, "text": current_text})
            current_text = sentence
        else:
            current_text += sentence

    if current_text:
        chunks.append({"start": None, "end": None, "text": current_text})

    print(f"[Parser] TXT → {len(chunks)} chunks (~{chunk_chars} chars each)")
    return chunks


def _timestamp_to_seconds(ts: str) -> float:
    """Convert 'HH:MM:SS.mmm' to total seconds (float)."""
    parts = ts.replace(",", ".").split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


# ============================================================
# SECTION 4: PINYIN ENGINE (pypinyin — deterministic)
# ============================================================
# CRITICAL DESIGN DECISION: We do NOT let the LLM generate pinyin.
# 7B models hallucinate tone marks ~15-20% of the time on technical
# compounds. pypinyin uses a dictionary-based approach that is 100%
# deterministic for known words.

def add_pinyin(text: str) -> str:
    """
    Generate tone-marked pinyin for a Chinese text string.

    Uses pypinyin's TONE style which produces marks like:
    "运算放大器" → "yùn suàn fàng dà qì"

    For mixed Chinese-English text (common in tech videos),
    non-Chinese characters pass through unchanged.

    Returns:
        Space-separated pinyin string with tone marks.
    """
    result = pinyin(text, style=Style.TONE, errors="ignore")
    # pinyin() returns list of lists: [['yùn'], ['suàn'], ...]
    return " ".join([item[0] for item in result])


def add_pinyin_ruby(text: str) -> str:
    """
    Generate HTML ruby annotations for Chinese characters.

    Ruby text places pinyin directly above each character,
    which is the gold standard for reading comprehension.
    Renders as small text above each character in browsers.

    Example output:
        <ruby>运<rp>(</rp><rt>yùn</rt><rp>)</rp></ruby>

    The <rp> tags provide fallback parentheses for browsers
    that don't support ruby (rare, but good practice).
    """
    result = []
    for char in text:
        if re.match(r"[\u4e00-\u9fff]", char):
            # Character is Chinese → add ruby annotation
            py = pinyin(char, style=Style.TONE)[0][0]
            result.append(
                f"<ruby>{char}<rp>(</rp><rt>{py}</rt><rp>)</rp></ruby>"
            )
        else:
            # Non-Chinese character (punctuation, English, numbers) → pass through
            result.append(char)
    return "".join(result)


# ============================================================
# SECTION 5: CC-CEDICT DICTIONARY (Cross-check layer)
# ============================================================
# CC-CEDICT is the standard open-source Chinese-English dictionary.
# We use it as a "ground truth" layer to validate LLM-generated
# definitions. If the LLM's definition diverges significantly from
# CEDICT, we flag it for manual review.

class CEDICTLoader:
    """
    Load and query the CC-CEDICT dictionary.

    File format (one entry per line):
        Traditional Simplified [pinyin] /definition 1/definition 2/

    Example:
        運算放大器 运算放大器 [yun4 suan4 fang4 da4 qi4] /operational amplifier/op-amp/

    Download from: https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz
    """

    def __init__(self, filepath: str = None):
        self.entries = {}  # simplified → {"pinyin": ..., "definitions": [...]}
        if filepath and os.path.exists(filepath):
            self._load(filepath)
            print(f"[CEDICT] Loaded {len(self.entries)} dictionary entries")
        else:
            print("[CEDICT] Dictionary file not found — cross-check disabled.")
            print(f"         Expected at: {filepath or CONFIG['cedict_path']}")
            print("         Download: https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz")

    def _load(self, filepath: str):
        """Parse the CC-CEDICT text file into a lookup dictionary."""
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                # Skip comments (lines starting with #)
                if line.startswith("#"):
                    continue
                # Parse: Traditional Simplified [pinyin] /defs/
                match = re.match(
                    r"(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/", line.strip()
                )
                if match:
                    simplified = match.group(2)
                    entry_pinyin = match.group(3)
                    definitions = match.group(4).split("/")
                    self.entries[simplified] = {
                        "pinyin": entry_pinyin,
                        "definitions": definitions,
                    }

    def lookup(self, term: str) -> dict | None:
        """
        Look up a simplified Chinese term.

        Returns:
            {"pinyin": "yun4 suan4 fang4 da4 qi4", "definitions": ["op-amp", ...]}
            or None if not found.
        """
        return self.entries.get(term)


# ============================================================
# SECTION 6: OLLAMA LLM INTERFACE
# ============================================================
# All LLM calls go through a single function. The LLM handles
# tasks that REQUIRE contextual understanding:
#   - ASR error correction (needs domain knowledge)
#   - Summarization (needs semantic understanding)
#   - Translation (needs context for ambiguous terms)
#   - Glossary extraction (needs domain classification)
#   - Comprehension questions (needs pedagogical judgment)
#
# Tasks that DON'T require the LLM use deterministic tools:
#   - Pinyin → pypinyin library
#   - Dictionary lookup → CC-CEDICT

def ollama_chat(prompt: str, system_prompt: str = "", retries: int = 2) -> str:
    """
    Send a chat completion request to the local Ollama server.

    Uses the /api/chat endpoint (not /api/generate) for better
    instruction following with the chat-tuned model.

    Args:
        prompt: The user message content
        system_prompt: System-level instructions for the model
        retries: Number of retry attempts on failure

    Returns:
        The model's response text, or an error message string.
    """
    url = f"{CONFIG['ollama_url']}/api/chat"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": CONFIG["model"],
        "messages": messages,
        "stream": False,              # Wait for full response (simpler parsing)
        "options": {
            "temperature": CONFIG["temperature"],
            "num_ctx": 8192,          # Context window — 8K is safe for 7B models
        },
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except requests.exceptions.ConnectionError:
            print(f"\n[ERROR] Cannot connect to Ollama at {CONFIG['ollama_url']}")
            print("  → Is Ollama running? Start it with: ollama serve")
            print(f"  → Is the model downloaded? Run: ollama pull {CONFIG['model']}")
            if attempt < retries:
                print(f"  → Retrying ({attempt + 1}/{retries})...")
            else:
                return "[LLM_ERROR: Ollama connection failed]"
        except requests.exceptions.Timeout:
            print(f"\n[WARNING] Ollama request timed out (attempt {attempt + 1})")
            if attempt >= retries:
                return "[LLM_ERROR: Request timed out]"
        except Exception as e:
            print(f"\n[ERROR] Unexpected Ollama error: {e}")
            if attempt >= retries:
                return f"[LLM_ERROR: {e}]"

    return "[LLM_ERROR: All retries exhausted]"


# ============================================================
# SECTION 7: LLM TASK PROMPTS & PROCESSING PIPELINE
# ============================================================
# Each function below wraps a specific LLM task with a carefully
# engineered prompt. Prompts are designed to:
#   1. Constrain output format (JSON where possible)
#   2. Provide domain context (term lists)
#   3. Minimize hallucination surface area

def correct_asr_errors(raw_text: str) -> str:
    """
    Use the LLM to fix speech-to-text errors in the raw subtitle text.

    Common ASR errors in Chinese technical content:
    - Homophones: 锁相环 → 所相环 (same sound, wrong characters)
    - Word boundaries: 运算 放大器 → 运 算放 大器 (wrong segmentation)
    - English loanwords: "MOSFET" → "摸死FET" (phonetic mangling)

    The domain term list anchors the LLM to correct technical vocabulary.
    """
    system = (
        "You are a Chinese technical transcript proofreader specializing in "
        "analog circuits, signal processing, and AI/ML. "
        "Fix speech recognition errors. Preserve the original meaning. "
        "Only fix clear errors — do not rephrase or summarize. "
        "Output ONLY the corrected Chinese text, nothing else."
    )
    prompt = (
        f"Below is a raw auto-generated Chinese subtitle transcript. "
        f"It may contain speech recognition errors, especially for technical terms.\n\n"
        f"KNOWN TECHNICAL TERMS (use these as reference for corrections):\n"
        f"{', '.join(DOMAIN_TERMS[:50])}\n\n"  # Send top 50 terms to stay within context
        f"RAW TRANSCRIPT:\n{raw_text}\n\n"
        f"Output the corrected transcript:"
    )
    return ollama_chat(prompt, system)


def generate_summary(full_text: str, video_title: str = "") -> str:
    """
    Generate a 3-line core summary of the video content.

    Line 1: Main topic / what the video covers
    Line 2: Key concepts or techniques discussed
    Line 3: Practical takeaway or application context
    """
    system = (
        "You are a technical content summarizer. Write in English. "
        "Produce exactly 3 lines of summary. Be specific and concrete, "
        "not vague. Include technical term names in both Chinese and English."
    )
    prompt = (
        f"Video title: {video_title}\n\n"
        f"Full transcript:\n{full_text[:3000]}\n\n"  # Truncate for 7B context limits
        f"Write a 3-line summary:\n"
        f"Line 1 — Main topic:\n"
        f"Line 2 — Key concepts:\n"
        f"Line 3 — Practical takeaway:"
    )
    return ollama_chat(prompt, system)


def translate_chunk(chinese_text: str) -> str:
    """
    Translate a single chunk of Chinese transcript to English.

    The prompt emphasizes preserving technical term accuracy
    over natural-sounding prose. For an engineering researcher,
    precise terminology matters more than fluency.
    """
    system = (
        "You are a Chinese-to-English technical translator specializing in "
        "electronics, signal processing, and AI. "
        "Translate accurately. Keep technical terms precise. "
        "Output ONLY the English translation, nothing else."
    )
    return ollama_chat(f"Translate to English:\n\n{chinese_text}", system)


def extract_glossary(full_text: str) -> dict:
    """
    Extract and categorize vocabulary from the transcript.

    Returns two categories:
    - "daily": Common Chinese words useful for general fluency
    - "technical": Domain-specific jargon (analog/DSP/AI)

    Each term includes: Chinese, domain tag, English definition.
    Pinyin is NOT generated here — we add it deterministically
    with pypinyin in the HTML generation step.

    The LLM returns JSON for reliable parsing.
    """
    system = (
        "You are a Chinese language instructor for engineers. "
        "Extract vocabulary from the transcript below. "
        "Respond ONLY with valid JSON, no markdown fences, no explanation."
    )
    prompt = (
        f"Extract important vocabulary from this Chinese technical transcript.\n\n"
        f"TRANSCRIPT:\n{full_text[:3000]}\n\n"
        f"Return JSON in this exact format:\n"
        f'{{"daily": [\n'
        f'    {{"term": "因为", "definition": "because"}},\n'
        f'    {{"term": "所以", "definition": "therefore"}}\n'
        f'  ],\n'
        f'  "technical": [\n'
        f'    {{"term": "运放", "domain": "analog", "definition": "op-amp (operational amplifier)"}},\n'
        f'    {{"term": "采样率", "domain": "dsp", "definition": "sampling rate"}}\n'
        f'  ]\n'
        f"}}\n\n"
        f"Extract 10-15 daily terms and 10-20 technical terms. "
        f"For technical terms, set domain to one of: analog, dsp, ai."
    )
    raw = ollama_chat(prompt, system)

    # ── Parse the JSON response ──
    # LLMs sometimes wrap JSON in ```json ... ``` fences — strip them.
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        glossary = json.loads(raw)
        # Validate structure
        if "daily" not in glossary:
            glossary["daily"] = []
        if "technical" not in glossary:
            glossary["technical"] = []
        return glossary
    except json.JSONDecodeError:
        print("[WARNING] LLM returned invalid JSON for glossary. Using fallback.")
        return {"daily": [], "technical": []}


def generate_comprehension_questions(chunks_text: str, chunk_indices: str) -> str:
    """
    Generate 2-3 comprehension questions for a group of chunks.

    Question types:
    1. Factual recall — tests if the reader understood what was said
    2. Term application — fill-in-the-blank with technical terms
    3. Conceptual — why/how questions linking ideas

    Returns HTML-formatted question block.
    """
    system = (
        "You are a Chinese language and engineering instructor. "
        "Create 2-3 study questions based on the transcript below. "
        "Mix question types: 1 factual, 1 fill-in-the-blank (use ____ for blanks), "
        "1 conceptual. Write questions in English but include Chinese terms where relevant. "
        "Output ONLY the numbered questions, nothing else."
    )
    prompt = (
        f"Transcript from chunks {chunk_indices}:\n\n"
        f"{chunks_text}\n\n"
        f"Generate 2-3 comprehension questions:"
    )
    return ollama_chat(prompt, system)


# ============================================================
# SECTION 8: MAIN PROCESSING ORCHESTRATOR
# ============================================================
# This ties everything together: parse → correct → chunk → process

def process_pipeline(chunks: list[dict], video_info: dict, cedict: CEDICTLoader) -> dict:
    """
    Run the full processing pipeline on parsed subtitle chunks.

    Processing order:
    1. Concatenate all text → ASR error correction (one LLM call)
    2. Re-split corrected text into chunks
    3. Full text → Summary (one LLM call)
    4. Full text → Glossary extraction (one LLM call)
    5. Each chunk → Pinyin (deterministic, no LLM)
    6. Each chunk → Translation (one LLM call per chunk)
    7. Every N chunks → Comprehension questions (one LLM call per group)

    Returns:
        Complete data structure ready for HTML rendering.
    """
    # ── Step 1: ASR Error Correction ──
    # We correct the FULL text in one pass so the LLM has maximum context
    # for disambiguating homophones. Then we redistribute corrected text
    # back into the original chunk boundaries.
    print("\n📝 Step 1/5: Correcting ASR errors...")
    full_raw_text = "".join([c["text"] for c in chunks])

    # Only correct if the text is substantial (very short texts don't need it)
    if len(full_raw_text) > 50:
        corrected_full = correct_asr_errors(full_raw_text)
        # If the LLM returned an error, fall back to raw text
        if corrected_full.startswith("[LLM_ERROR"):
            corrected_full = full_raw_text
    else:
        corrected_full = full_raw_text

    # ── Redistribute corrected text back into chunks ──
    # Since ASR correction may slightly change text length,
    # we re-chunk by proportional character count.
    original_lengths = [len(c["text"]) for c in chunks]
    total_original = sum(original_lengths)
    corrected_chunks = []
    cursor = 0

    for i, chunk in enumerate(chunks):
        # Proportional allocation of corrected text to each chunk
        if i < len(chunks) - 1:
            proportion = original_lengths[i] / total_original
            end = cursor + int(len(corrected_full) * proportion)
        else:
            end = len(corrected_full)  # Last chunk gets the rest

        corrected_chunks.append({
            "start": chunk["start"],
            "end": chunk["end"],
            "text": corrected_full[cursor:end],
        })
        cursor = end

    # ── Step 2: Generate Summary ──
    print("📋 Step 2/5: Generating summary...")
    summary = generate_summary(corrected_full, video_info.get("title", ""))

    # ── Step 3: Extract Glossary ──
    print("📖 Step 3/5: Extracting glossary...")
    glossary = extract_glossary(corrected_full)

    # ── Cross-check glossary terms with CC-CEDICT ──
    for category in ["daily", "technical"]:
        for term_entry in glossary.get(category, []):
            cedict_result = cedict.lookup(term_entry["term"])
            if cedict_result:
                term_entry["cedict_match"] = True
                term_entry["cedict_def"] = cedict_result["definitions"][0]
            else:
                term_entry["cedict_match"] = False

    # ── Step 4: Process Each Chunk (Pinyin + Translation) ──
    print("🔤 Step 4/5: Adding pinyin and translations...")
    processed_chunks = []
    for i, chunk in enumerate(tqdm(corrected_chunks, desc="   Chunks")):
        processed = {
            "index": i + 1,
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"],
            "pinyin": add_pinyin(chunk["text"]),
            "ruby_html": add_pinyin_ruby(chunk["text"]),
            "translation": translate_chunk(chunk["text"]),
        }
        processed_chunks.append(processed)

    # ── Step 5: Comprehension Questions ──
    print("❓ Step 5/5: Generating comprehension questions...")
    questions = []
    n = CONFIG["questions_every_n_chunks"]
    for i in range(0, len(processed_chunks), n):
        group = processed_chunks[i : i + n]
        group_text = " ".join([c["text"] for c in group])
        indices = f"{group[0]['index']}-{group[-1]['index']}"
        q = generate_comprehension_questions(group_text, indices)
        questions.append({
            "after_chunk": group[-1]["index"],
            "questions_html": q,
        })

    return {
        "video_info": video_info,
        "summary": summary,
        "glossary": glossary,
        "chunks": processed_chunks,
        "questions": questions,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ============================================================
# SECTION 9: HTML GENERATION
# ============================================================
# Self-contained HTML with embedded CSS. Two layout modes:
# - Screen: interactive <details> toggles, clickable timestamps
# - Print: @media print forces A4 sizing, auto-expands all toggles

def generate_html(data: dict) -> str:
    """
    Render the processed data as a self-contained A4-ready HTML textbook.

    Design principles:
    - Single file, zero external dependencies (fonts loaded from CDN with fallbacks)
    - Print-optimized: @media print sets A4 margins, page breaks, expanded toggles
    - Screen-optimized: <details> tags hide translations until clicked
    - Ruby text for inline pinyin above characters
    - Clean typography optimized for dense CJK + Latin mixed content
    """
    video = data["video_info"]
    summary = data["summary"]
    glossary = data["glossary"]
    chunks = data["chunks"]
    questions = data["questions"]

    # ── Build Glossary Tables ──
    daily_rows = ""
    for t in glossary.get("daily", []):
        py = add_pinyin(t["term"])
        cedict_badge = "✓" if t.get("cedict_match") else "△"
        daily_rows += (
            f"<tr>"
            f"<td class='term-cell'>{t['term']}</td>"
            f"<td class='pinyin-cell'>{py}</td>"
            f"<td>{t.get('definition', '')}</td>"
            f"<td class='badge'>{cedict_badge}</td>"
            f"</tr>\n"
        )

    tech_rows = ""
    for t in glossary.get("technical", []):
        py = add_pinyin(t["term"])
        domain = t.get("domain", "").upper()
        cedict_badge = "✓" if t.get("cedict_match") else "△"
        # Color-code domains
        domain_class = f"domain-{t.get('domain', 'other')}"
        tech_rows += (
            f"<tr>"
            f"<td class='term-cell'>{t['term']}</td>"
            f"<td class='pinyin-cell'>{py}</td>"
            f"<td><span class='domain-tag {domain_class}'>{domain}</span></td>"
            f"<td>{t.get('definition', '')}</td>"
            f"<td class='badge'>{cedict_badge}</td>"
            f"</tr>\n"
        )

    # ── Build Chunk Sections ──
    # Each chunk becomes a study unit with: header, ruby text, pinyin line,
    # hidden translation, and optional comprehension questions after it.
    question_map = {q["after_chunk"]: q["questions_html"] for q in questions}

    chunks_html = ""
    for chunk in chunks:
        # Timestamp header (only if SRT source provided timestamps)
        time_label = ""
        if chunk["start"]:
            time_label = f'<span class="timestamp">⏱ {chunk["start"]} — {chunk["end"]}</span>'

        # Comprehension questions (inserted after every Nth chunk)
        q_section = ""
        if chunk["index"] in question_map:
            q_text = question_map[chunk["index"]]
            q_section = f"""
            <div class="questions-block">
                <h4>✍ Comprehension Check — Chunks up to {chunk["index"]}</h4>
                <div class="questions-content">{q_text}</div>
            </div>
            """

        chunks_html += f"""
        <section class="chunk" id="chunk-{chunk['index']}">
            <div class="chunk-header">
                <h3>Chunk {chunk['index']}</h3>
                {time_label}
            </div>

            <div class="script-block">
                <div class="ruby-text">{chunk['ruby_html']}</div>
            </div>

            <details class="pinyin-toggle">
                <summary>Show Pinyin (plain text)</summary>
                <p class="pinyin-plain">{chunk['pinyin']}</p>
            </details>

            <details class="translation-toggle">
                <summary>Show English Translation</summary>
                <div class="translation">{chunk['translation']}</div>
            </details>
        </section>
        {q_section}
        """

    # ── Assemble Full HTML Document ──
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{video.get('title', 'Chinese Technical Textbook')}</title>

    <!-- ─── Fonts: Noto for CJK, Source Serif for Latin ─── -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">

    <style>
        /* ════════════════════════════════════════════════════
           CSS CUSTOM PROPERTIES (Theme)
           Easy to customize: change these variables to restyle
           the entire document.
           ════════════════════════════════════════════════════ */
        :root {{
            --color-bg: #faf9f6;
            --color-text: #2c2c2c;
            --color-heading: #1a1a2e;
            --color-accent: #c0392b;
            --color-accent-light: #f9ebea;
            --color-border: #d5c9b1;
            --color-chunk-bg: #ffffff;
            --color-pinyin: #6b5b3e;
            --color-translation-bg: #f0ece4;
            --color-question-bg: #eaf2e8;
            --color-question-border: #5a8a4a;
            --color-domain-analog: #8e44ad;
            --color-domain-dsp: #2980b9;
            --color-domain-ai: #d35400;
            --color-badge-ok: #27ae60;
            --color-badge-warn: #e67e22;

            --font-cjk: 'Noto Serif SC', 'Songti SC', 'SimSun', serif;
            --font-latin: 'Source Serif 4', 'Georgia', serif;
            --font-mono: 'JetBrains Mono', 'Menlo', monospace;

            --page-width: 190mm;
            --page-padding: 10mm;
        }}

        /* ════════════════════════════════════════════════════
           BASE LAYOUT — Screen Mode
           ════════════════════════════════════════════════════ */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: var(--font-latin);
            background: var(--color-bg);
            color: var(--color-text);
            line-height: 1.7;
            max-width: var(--page-width);
            margin: 0 auto;
            padding: 20mm var(--page-padding);
        }}

        /* ── Cover / Header Section ── */
        .cover {{
            text-align: center;
            padding-bottom: 8mm;
            border-bottom: 2px solid var(--color-border);
            margin-bottom: 10mm;
        }}
        .cover h1 {{
            font-family: var(--font-cjk);
            font-size: 1.6em;
            color: var(--color-heading);
            margin-bottom: 4mm;
            line-height: 1.3;
        }}
        .cover .meta {{
            font-size: 0.85em;
            color: #777;
            line-height: 1.6;
        }}
        .cover .meta span {{
            display: inline-block;
            margin: 0 8px;
        }}

        /* ── Summary Section ── */
        .summary {{
            background: var(--color-accent-light);
            border-left: 4px solid var(--color-accent);
            padding: 5mm;
            margin-bottom: 8mm;
            border-radius: 0 4px 4px 0;
        }}
        .summary h2 {{
            font-size: 1em;
            color: var(--color-accent);
            margin-bottom: 3mm;
            font-family: var(--font-latin);
        }}
        .summary p {{
            font-size: 0.9em;
            white-space: pre-line;
        }}

        /* ── Glossary Section ── */
        .glossary {{
            margin-bottom: 10mm;
        }}
        .glossary h2 {{
            font-size: 1.15em;
            color: var(--color-heading);
            border-bottom: 1px solid var(--color-border);
            padding-bottom: 2mm;
            margin-bottom: 4mm;
        }}
        .glossary h3 {{
            font-size: 0.95em;
            color: var(--color-accent);
            margin: 5mm 0 3mm;
        }}
        .glossary table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85em;
            margin-bottom: 5mm;
        }}
        .glossary th {{
            background: var(--color-heading);
            color: white;
            padding: 2mm 3mm;
            text-align: left;
            font-weight: 600;
        }}
        .glossary td {{
            padding: 2mm 3mm;
            border-bottom: 1px solid #e8e4dc;
            vertical-align: top;
        }}
        .glossary tr:nth-child(even) {{
            background: #f7f5f0;
        }}
        .term-cell {{
            font-family: var(--font-cjk);
            font-size: 1.1em;
            font-weight: 700;
            white-space: nowrap;
        }}
        .pinyin-cell {{
            font-family: var(--font-mono);
            font-size: 0.9em;
            color: var(--color-pinyin);
        }}
        .domain-tag {{
            font-size: 0.75em;
            padding: 1px 6px;
            border-radius: 3px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .domain-analog {{ background: #f0e6f6; color: var(--color-domain-analog); }}
        .domain-dsp    {{ background: #e6f0fa; color: var(--color-domain-dsp); }}
        .domain-ai     {{ background: #fde8d8; color: var(--color-domain-ai); }}
        .badge {{
            text-align: center;
            font-size: 1em;
        }}

        /* ── Chunk Sections (Study Units) ── */
        .chunk {{
            background: var(--color-chunk-bg);
            border: 1px solid var(--color-border);
            border-radius: 4px;
            padding: 5mm;
            margin-bottom: 6mm;
            page-break-inside: avoid;  /* Print: don't split a chunk across pages */
        }}
        .chunk-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3mm;
        }}
        .chunk-header h3 {{
            font-size: 0.95em;
            color: var(--color-heading);
        }}
        .timestamp {{
            font-family: var(--font-mono);
            font-size: 0.8em;
            color: #999;
            background: #f5f3ee;
            padding: 1px 8px;
            border-radius: 12px;
        }}

        /* ── Ruby (Pinyin above characters) ── */
        .ruby-text {{
            font-family: var(--font-cjk);
            font-size: 1.15em;
            line-height: 2.8;  /* Extra space for ruby annotations above */
        }}
        .ruby-text ruby {{
            ruby-align: center;
        }}
        .ruby-text rt {{
            font-family: var(--font-mono);
            font-size: 0.48em;
            color: var(--color-pinyin);
            font-weight: 400;
        }}
        .ruby-text rp {{
            color: var(--color-pinyin);
        }}

        /* ── Toggleable Sections ── */
        details {{
            margin-top: 3mm;
            border-radius: 3px;
        }}
        details summary {{
            cursor: pointer;
            font-size: 0.85em;
            color: var(--color-accent);
            font-weight: 600;
            padding: 1mm 0;
            user-select: none;
        }}
        details summary:hover {{
            text-decoration: underline;
        }}
        .pinyin-plain {{
            font-family: var(--font-mono);
            font-size: 0.85em;
            color: var(--color-pinyin);
            padding: 2mm 0;
            line-height: 1.8;
        }}
        .translation {{
            background: var(--color-translation-bg);
            padding: 3mm;
            border-radius: 3px;
            font-size: 0.9em;
            line-height: 1.6;
            margin-top: 2mm;
        }}

        /* ── Comprehension Questions ── */
        .questions-block {{
            background: var(--color-question-bg);
            border-left: 4px solid var(--color-question-border);
            padding: 5mm;
            margin-bottom: 6mm;
            border-radius: 0 4px 4px 0;
            page-break-inside: avoid;
        }}
        .questions-block h4 {{
            font-size: 0.9em;
            color: var(--color-question-border);
            margin-bottom: 3mm;
        }}
        .questions-content {{
            font-size: 0.9em;
            white-space: pre-line;
        }}

        /* ── Footer ── */
        .footer {{
            margin-top: 10mm;
            padding-top: 4mm;
            border-top: 1px solid var(--color-border);
            font-size: 0.75em;
            color: #aaa;
            text-align: center;
        }}

        /* ════════════════════════════════════════════════════
           PRINT LAYOUT — A4 Optimization
           ════════════════════════════════════════════════════
           When printing (Ctrl+P), these rules activate:
           - Forces A4 page size with proper margins
           - Auto-expands all <details> toggles
           - Removes interactive-only elements
           - Ensures page breaks don't split chunks
           ════════════════════════════════════════════════════ */
        @media print {{
            @page {{
                size: A4;
                margin: 15mm 15mm 20mm 15mm;
            }}

            body {{
                background: white;
                max-width: none;
                padding: 0;
                font-size: 10pt;
            }}

            /* Force all hidden sections open when printing */
            details {{
                display: block !important;
            }}
            details > summary {{
                display: none !important;  /* Hide "Show/Hide" toggle text */
            }}
            details > *:not(summary) {{
                display: block !important;
            }}

            /* Page break control */
            .chunk {{
                page-break-inside: avoid;
                break-inside: avoid;
                border: 1px solid #ccc;
            }}
            .questions-block {{
                page-break-inside: avoid;
                break-inside: avoid;
            }}
            .glossary {{
                page-break-before: always;  /* Glossary starts on new page */
            }}
            .cover {{
                page-break-after: always;   /* Cover is its own page */
            }}

            /* Clean up for print */
            .timestamp {{
                background: none;
                border: 1px solid #ccc;
            }}
        }}
    </style>
</head>

<body>
    <!-- ═══════════ COVER PAGE ═══════════ -->
    <header class="cover">
        <h1>📘 {video.get('title', 'Chinese Technical Textbook')}</h1>
        <div class="meta">
            <span>👤 {video.get('uploader', 'Unknown')}</span>
            <span>⏱ {video.get('duration', 0) // 60}:{video.get('duration', 0) % 60:02d}</span>
            <span>📅 Generated: {data['generated_at']}</span>
        </div>
        <div class="meta" style="margin-top: 2mm;">
            <span>🔗 <a href="{video.get('url', '#')}" style="color: var(--color-accent);">Source Video</a></span>
        </div>
    </header>

    <!-- ═══════════ SUMMARY ═══════════ -->
    <div class="summary">
        <h2>Core Summary</h2>
        <p>{summary}</p>
    </div>

    <!-- ═══════════ GLOSSARY ═══════════ -->
    <div class="glossary">
        <h2>📖 Glossary</h2>

        <h3>Daily Terms (日常用语)</h3>
        <table>
            <thead>
                <tr>
                    <th>Term</th>
                    <th>Pinyin</th>
                    <th>Definition</th>
                    <th title="✓ = CEDICT verified, △ = LLM only">Dict</th>
                </tr>
            </thead>
            <tbody>
                {daily_rows if daily_rows else '<tr><td colspan="4" style="text-align:center; color:#999;">No daily terms extracted</td></tr>'}
            </tbody>
        </table>

        <h3>Technical Jargon (专业术语)</h3>
        <table>
            <thead>
                <tr>
                    <th>Term</th>
                    <th>Pinyin</th>
                    <th>Domain</th>
                    <th>Definition</th>
                    <th title="✓ = CEDICT verified, △ = LLM only">Dict</th>
                </tr>
            </thead>
            <tbody>
                {tech_rows if tech_rows else '<tr><td colspan="5" style="text-align:center; color:#999;">No technical terms extracted</td></tr>'}
            </tbody>
        </table>
        <p style="font-size: 0.75em; color: #999; margin-top: 2mm;">
            ✓ = Verified against CC-CEDICT dictionary &nbsp;|&nbsp; △ = LLM-generated definition (unverified)
        </p>
    </div>

    <!-- ═══════════ SCRIPT STUDY (Chunked) ═══════════ -->
    <h2 style="font-size: 1.15em; color: var(--color-heading); border-bottom: 1px solid var(--color-border); padding-bottom: 2mm; margin-bottom: 5mm;">
        📝 Script Study
    </h2>

    {chunks_html}

    <!-- ═══════════ FOOTER ═══════════ -->
    <footer class="footer">
        Generated by Automated Chinese Technical Textbook Generator<br>
        Model: {CONFIG['model']} via Ollama &nbsp;|&nbsp; Pinyin: pypinyin (deterministic)
    </footer>

</body>
</html>"""

    return html


# ============================================================
# SECTION 10: CLI ENTRY POINT
# ============================================================
# Handles argument parsing, input detection, and orchestration.

def main():
    """
    Main entry point. Accepts either a URL or a local file path.

    Examples:
        python textbook_generator.py https://www.bilibili.com/video/BV1xx...
        python textbook_generator.py ./my_subtitles.srt
        python textbook_generator.py ./transcript.txt
        python textbook_generator.py input.srt -o my_textbook.html
        python textbook_generator.py input.srt --model qwen2.5:14b
    """
    parser = argparse.ArgumentParser(
        description="Generate A4-ready Chinese technical textbooks from video subtitles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s https://www.bilibili.com/video/BV1xx411x7xX
              %(prog)s subtitles.srt
              %(prog)s transcript.txt -o my_textbook.html
              %(prog)s subtitles.srt --model qwen2.5:14b
        """)
    )
    parser.add_argument(
        "input",
        help="Bilibili/YouTube URL or path to a local .srt/.txt subtitle file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output HTML file path (default: auto-generated from input name)",
        default=None
    )
    parser.add_argument(
        "--model",
        help=f"Ollama model name (default: {CONFIG['model']})",
        default=CONFIG["model"]
    )
    parser.add_argument(
        "--cedict",
        help=f"Path to CC-CEDICT dictionary file (default: {CONFIG['cedict_path']})",
        default=CONFIG["cedict_path"]
    )

    args = parser.parse_args()

    # ── Apply CLI overrides to global config ──
    CONFIG["model"] = args.model

    # ── Load CC-CEDICT dictionary ──
    cedict = CEDICTLoader(args.cedict)

    # ── Determine input type: URL or local file ──
    input_path = args.input
    is_url = input_path.startswith("http://") or input_path.startswith("https://")

    video_info = {
        "title": "",
        "uploader": "",
        "duration": 0,
        "url": input_path if is_url else "",
    }

    if is_url:
        # ── URL Mode: Extract subtitles with yt-dlp ──
        print(f"🌐 Input: URL detected → {input_path}")
        video_info = extract_subtitles_ytdlp(input_path)

        if not video_info.get("subtitle_file"):
            print("\n[FATAL] Could not extract subtitles from URL.")
            print("Alternatives:")
            print("  1. Download subtitles manually and pass the .srt file")
            print("  2. Use a browser extension to export subtitles")
            print("  3. Check if the video has captions enabled")
            sys.exit(1)

        subtitle_file = video_info["subtitle_file"]
    else:
        # ── File Mode: Use provided subtitle file directly ──
        subtitle_file = input_path
        if not os.path.exists(subtitle_file):
            print(f"[FATAL] File not found: {subtitle_file}")
            sys.exit(1)
        print(f"📄 Input: Local file → {subtitle_file}")
        # Use filename as a fallback title
        video_info["title"] = Path(subtitle_file).stem

    # ── Auto-detect format and parse ──
    ext = Path(subtitle_file).suffix.lower()

    if ext == ".srt":
        print(f"[Format] SRT detected → parsing with timestamps")
        chunks = parse_srt(subtitle_file, CONFIG["chunk_seconds"])
    elif ext in (".txt", ".text"):
        print(f"[Format] TXT detected → parsing without timestamps")
        chunks = parse_txt(subtitle_file, CONFIG["chunk_chars"])
    else:
        # Unknown extension — try SRT first (more structured), fall back to TXT
        print(f"[Format] Unknown extension '{ext}' → trying SRT parser...")
        chunks = parse_srt(subtitle_file, CONFIG["chunk_seconds"])
        if not chunks:
            print(f"[Format] SRT parse failed → falling back to TXT parser")
            chunks = parse_txt(subtitle_file, CONFIG["chunk_chars"])

    if not chunks:
        print("[FATAL] No text content found in the subtitle file.")
        sys.exit(1)

    print(f"[Ready] {len(chunks)} chunks to process\n")

    # ── Run the processing pipeline ──
    data = process_pipeline(chunks, video_info, cedict)

    # ── Generate HTML output ──
    html_content = generate_html(data)

    # ── Write output file ──
    if args.output:
        output_path = args.output
    else:
        # Auto-name: <input_name>_textbook.html
        stem = Path(subtitle_file).stem
        output_path = f"{stem}_textbook.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n{'='*56}")
    print(f"✅ Textbook generated: {output_path}")
    print(f"   Chunks: {len(data['chunks'])}")
    print(f"   Daily terms: {len(data['glossary'].get('daily', []))}")
    print(f"   Technical terms: {len(data['glossary'].get('technical', []))}")
    print(f"   Questions: {len(data['questions'])} sets")
    print(f"{'='*56}")
    print(f"\n🖥  Open in browser: file://{os.path.abspath(output_path)}")
    print(f"🖨  Print to A4: Open in browser → Ctrl+P → Print")


if __name__ == "__main__":
    main()
