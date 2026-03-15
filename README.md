# 📘 Automated Chinese Technical Textbook Generator

Generate A4-ready, print-friendly HTML textbooks from Bilibili video subtitles.
Designed for engineers learning technical Chinese through real-world content.

## Pipeline

```
Bilibili URL ──→ yt-dlp ──→ Ollama (Qwen2.5) ──→ pypinyin ──→ HTML Textbook
     or                      ASR correction          Tone        A4 print-ready
Local .srt/.txt              Translation             marks       Screen-viewable
                             Glossary
                             Comprehension Qs
```

## Setup (One-time)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install yt-dlp (subtitle extraction)

```bash
# macOS
brew install yt-dlp

# Linux
sudo apt install yt-dlp
# or
pip install yt-dlp

# Windows
winget install yt-dlp
```

### 3. Install & start Ollama

```bash
# Install: https://ollama.com/download
# Then pull the model:
ollama pull qwen2.5:7b

# Start the server (if not auto-started):
ollama serve
```

### 4. (Optional) Download CC-CEDICT dictionary

For dictionary cross-checking of LLM-generated definitions:

```bash
# Download and extract
wget https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz
gunzip cedict_1_0_ts_utf-8_mdbg.txt.gz
mv cedict_1_0_ts_utf-8_mdbg.txt cedict_ts.u8
```

The script works without it — you'll just lose the ✓/△ verification badges.

### 5. (For Bilibili) Export browser cookies

Some Bilibili videos require login. Set up yt-dlp cookies:

```bash
# If yt-dlp can't find subtitles, try:
yt-dlp --cookies-from-browser chrome <URL>
```

## Usage

### From a Bilibili/YouTube URL

```bash
python textbook_generator.py https://www.bilibili.com/video/BV1xx411x7xX
```

### From a local subtitle file

```bash
python textbook_generator.py subtitles.srt
python textbook_generator.py transcript.txt
```

### With options

```bash
# Custom output filename
python textbook_generator.py input.srt -o my_textbook.html

# Use a larger model for better quality
python textbook_generator.py input.srt --model qwen2.5:14b

# Specify CC-CEDICT path
python textbook_generator.py input.srt --cedict /path/to/cedict_ts.u8
```

## Output

A single self-contained `.html` file with:

- **Cover**: Video title, uploader, duration, source link
- **Summary**: 3-line core summary
- **Glossary**: Daily terms + Technical jargon (Analog/DSP/AI), CEDICT-verified
- **Script Study**: Chunked transcript with ruby pinyin, hidden translations
- **Comprehension Checks**: Questions every 3 chunks

### Viewing

- **Screen**: Open in any browser. Click toggles to reveal translations.
- **Print**: `Ctrl+P` in browser. A4 layout auto-activates, all toggles expand.

## Customization

Edit the `CONFIG` dict at the top of `textbook_generator.py`:

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `qwen2.5:7b` | Ollama model name |
| `chunk_seconds` | `90` | SRT chunk duration |
| `chunk_chars` | `300` | TXT chunk character count |
| `temperature` | `0.3` | LLM creativity (lower = more accurate) |
| `questions_every_n_chunks` | `3` | Frequency of comprehension questions |

### Adding domain terms

Edit the `DOMAIN_TERMS` list to add vocabulary specific to your research area.
These terms help the LLM correct ASR errors more accurately.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ConnectionError: Ollama` | Run `ollama serve` in a terminal |
| `Model not found` | Run `ollama pull qwen2.5:7b` |
| No subtitles from URL | Try `--cookies-from-browser chrome` with yt-dlp |
| Bad pinyin on rare terms | Add terms to `DOMAIN_TERMS` list |
| Glossary JSON parse error | Retry — 7B models occasionally malformat JSON |

## Architecture Notes

- **Pinyin**: 100% deterministic via `pypinyin` — never LLM-generated
- **Dictionary**: CC-CEDICT cross-check catches LLM definition errors
- **LLM tasks**: Only used where context understanding is required
  (ASR correction, translation, summarization, glossary extraction)
- **HTML**: Zero external JS dependencies, works fully offline after generation
