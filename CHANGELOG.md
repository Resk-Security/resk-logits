# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] - 2026-10-25

### Fixed
- `resklogits generate` no longer crashes: it read `args.synonyms` while the
  parser defines `--no-synonyms`.
- `device` now defaults to `cpu` in `ShadowBanProcessor`,
  `MultiLevelShadowBanProcessor` and `VectorizedAhoCorasick`; pass
  `device="cuda"` explicitly when the model runs on GPU.

## [0.2.1] - 2026-10-24

### Security
- Require `torch>=2.13.0` and `filelock>=4.2.1` (PYSEC-2025-194 and the
  filelock advisories fixed in 4.2.1).
- Minimum Python is now 3.11 (filelock 4.2.1 requires it; Python 3.10 is EOL).
- CI now runs `pip-audit` on every supported Python version.

## [0.2.0] - 2026-09-11

### Added
- Streaming helpers: `stream()` context manager and `stream_generate()`.
- 8 utility logits processors (length control, forced ending, MCQ restriction,
  token bans, trigger phrases, citation grounding, etc.).
- vLLM adapter (`to_vllm()`, `to_vllm_list()`, `VLLMWrapper`).
- YAML processor configuration and symbolic rule builder.
- Support for Python 3.11 through 3.14 (metadata previously required 3.13).
- Explicit PyPI license metadata (Apache-2.0).

### Fixed
- `transformers` v5 `batch_encode_plus` compatibility.
- Ruff/Black issues in the streaming and processor code.

## [0.1.0] - 2024-11-13

### Added
- Initial release of ReskLogits
- GPU-accelerated vectorized Aho-Corasick pattern matcher
- `ShadowBanProcessor` for single-level filtering with configurable penalties
- `MultiLevelShadowBanProcessor` for tiered severity-based filtering
- Comprehensive banned phrases dataset (400+ phrases across 20 categories)
- Support for HuggingFace Transformers integration
- GPU-accelerated operations (see BENCHMARKS.md for reproducible CPU numbers)
- Example scripts (demo, benchmark, simple usage)
- Full test suite
- Documentation and quick start guide

### Features
- Shadow ban approach (soft penalties vs hard blocks)
- Stateful pattern matching to catch jailbreak attempts
- Automatic EOS forcing on complete matches
- Batch processing support
- CPU and CUDA device support
- Scalable to 1000+ patterns with minimal overhead

### Performance
- Build time: ~0.5s for 1000 patterns
- Processing overhead: see BENCHMARKS.md; the 0.001 ms GPU figure is a vendor measurement and is not reproduced in this repository
- Generation overhead: depends on model step time; measure with benchmarks/benchmark_processor.py
- Memory: ~10MB for danger mask

