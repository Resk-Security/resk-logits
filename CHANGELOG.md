# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2024-11-13

### Added
- Initial release of ReskLogits
- GPU-accelerated vectorized Aho-Corasick pattern matcher
- `ShadowBanProcessor` for single-level filtering with configurable penalties
- `MultiLevelShadowBanProcessor` for tiered severity-based filtering
- Comprehensive banned phrases dataset (400+ phrases across 20 categories)
- Support for HuggingFace Transformers integration
- Zero-latency GPU operations (~0.001ms per token)
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
- Processing overhead: ~0.001ms per token (GPU)
- Generation overhead: <5%
- Memory: ~10MB for danger mask

