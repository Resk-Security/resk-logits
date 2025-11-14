"""
Cache System for Rule Generation

Provides hash-based caching to avoid regenerating patterns.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class RuleCache:
    """
    Hash-based cache for generated patterns.
    
    Cache structure:
    .resklogits_cache/
    ├── index.json              # Cache metadata index
    ├── <hash>_patterns.json    # Generated patterns
    └── <hash>_meta.json        # Generation metadata
    """

    def __init__(self, cache_dir: str = ".resklogits_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load cache index from disk."""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"entries": {}, "version": "1.0"}

    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2)

    def compute_hash(self, content: str) -> str:
        """
        Compute SHA256 hash of content.
        
        Args:
            content: Content to hash (usually YAML rules)
            
        Returns:
            Hex hash string
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def exists(self, rule_hash: str) -> bool:
        """Check if cache entry exists for given hash."""
        pattern_file = self.cache_dir / f"{rule_hash}_patterns.json"
        return pattern_file.exists()

    def load(self, rule_hash: str) -> List[str]:
        """
        Load patterns from cache.
        
        Args:
            rule_hash: Hash of rules
            
        Returns:
            List of cached patterns
            
        Raises:
            FileNotFoundError: If cache entry doesn't exist
        """
        pattern_file = self.cache_dir / f"{rule_hash}_patterns.json"

        if not pattern_file.exists():
            raise FileNotFoundError(f"Cache entry not found: {rule_hash}")

        with open(pattern_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Update access time in index
        if rule_hash in self.index["entries"]:
            self.index["entries"][rule_hash]["last_accessed"] = time.time()
            self._save_index()

        return data.get("patterns", [])

    def save(self, rule_hash: str, patterns: List[str], metadata: Dict[str, Any] = None):
        """
        Save patterns to cache.
        
        Args:
            rule_hash: Hash of rules
            patterns: Generated patterns
            metadata: Optional metadata about generation
        """
        # Save patterns
        pattern_file = self.cache_dir / f"{rule_hash}_patterns.json"
        with open(pattern_file, 'w', encoding='utf-8') as f:
            json.dump({
                "patterns": patterns,
                "count": len(patterns),
                "created": time.time()
            }, f, indent=2)

        # Save metadata
        if metadata:
            meta_file = self.cache_dir / f"{rule_hash}_meta.json"
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

        # Update index
        self.index["entries"][rule_hash] = {
            "created": time.time(),
            "last_accessed": time.time(),
            "pattern_count": len(patterns),
            "has_metadata": metadata is not None
        }
        self._save_index()

    def get_metadata(self, rule_hash: str) -> Optional[Dict[str, Any]]:
        """Load metadata for cached entry."""
        meta_file = self.cache_dir / f"{rule_hash}_meta.json"

        if not meta_file.exists():
            return None

        with open(meta_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata."""
        entries = []
        for hash_key, info in self.index["entries"].items():
            entry = {
                "hash": hash_key,
                **info,
                "created_str": datetime.fromtimestamp(info["created"]).isoformat(),
                "last_accessed_str": datetime.fromtimestamp(info["last_accessed"]).isoformat()
            }
            entries.append(entry)

        # Sort by last accessed (most recent first)
        entries.sort(key=lambda x: x["last_accessed"], reverse=True)
        return entries

    def clear(self, rule_hash: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            rule_hash: Specific hash to clear (None = clear all)
        """
        if rule_hash:
            # Clear specific entry
            pattern_file = self.cache_dir / f"{rule_hash}_patterns.json"
            meta_file = self.cache_dir / f"{rule_hash}_meta.json"

            if pattern_file.exists():
                pattern_file.unlink()
            if meta_file.exists():
                meta_file.unlink()

            # Remove from index
            if rule_hash in self.index["entries"]:
                del self.index["entries"][rule_hash]
                self._save_index()
        else:
            # Clear all entries
            for file in self.cache_dir.glob("*_patterns.json"):
                file.unlink()
            for file in self.cache_dir.glob("*_meta.json"):
                file.unlink()

            self.index["entries"] = {}
            self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.index["entries"])
        total_patterns = sum(
            entry.get("pattern_count", 0)
            for entry in self.index["entries"].values()
        )

        # Calculate total size
        total_size = 0
        for file in self.cache_dir.glob("*.json"):
            total_size += file.stat().st_size

        return {
            "total_entries": total_entries,
            "total_patterns": total_patterns,
            "cache_size_bytes": total_size,
            "cache_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }

    def cleanup_old(self, max_age_days: int = 30):
        """
        Remove cache entries older than specified age.
        
        Args:
            max_age_days: Maximum age in days
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        to_remove = []
        for hash_key, info in self.index["entries"].items():
            age = current_time - info["last_accessed"]
            if age > max_age_seconds:
                to_remove.append(hash_key)

        for hash_key in to_remove:
            self.clear(hash_key)

    def __repr__(self):
        stats = self.get_stats()
        return f"RuleCache({stats['total_entries']} entries, {stats['total_patterns']} patterns)"


class CachedGenerator:
    """Wrapper that adds caching to any generator function."""

    def __init__(self, generator_func, cache: RuleCache = None):
        """
        Initialize cached generator.
        
        Args:
            generator_func: Function that generates patterns
            cache: RuleCache instance (creates default if None)
        """
        self.generator_func = generator_func
        self.cache = cache or RuleCache()

    def generate(self, rules_content: str, force: bool = False, **kwargs) -> List[str]:
        """
        Generate patterns with caching.
        
        Args:
            rules_content: Rule content to hash
            force: Force regeneration even if cached
            **kwargs: Additional arguments for generator function
            
        Returns:
            Generated patterns
        """
        # Compute hash
        rule_hash = self.cache.compute_hash(rules_content)

        # Check cache unless force regeneration
        if not force and self.cache.exists(rule_hash):
            return self.cache.load(rule_hash)

        # Generate patterns
        patterns = self.generator_func(rules_content, **kwargs)

        # Save to cache
        metadata = {
            "generator": self.generator_func.__name__,
            "kwargs": kwargs,
            "timestamp": time.time()
        }
        self.cache.save(rule_hash, patterns, metadata)

        return patterns

    def __repr__(self):
        return f"CachedGenerator({self.generator_func.__name__}, {self.cache})"

