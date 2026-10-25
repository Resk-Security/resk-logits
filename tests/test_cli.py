import inspect
import json
from types import SimpleNamespace

from resklogits import ShadowBanProcessor
from resklogits.cli import cmd_generate


def test_generate_command_writes_patterns(tmp_path):
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """
rules:
  toxicity:
    exact:
      - badword
    synonyms:
      - [badword, bad-word]
""",
        encoding="utf-8",
    )
    output = tmp_path / "patterns.json"

    cmd_generate(
        SimpleNamespace(
            rules=str(rules),
            output=str(output),
            no_synonyms=False,
            no_cache=True,
            force=True,
            by_category=True,
            preview=5,
            verbose=False,
        )
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    assert "toxicity" in data
    assert "badword" in data["toxicity"]


def test_processor_defaults_to_cpu():
    default = inspect.signature(ShadowBanProcessor).parameters["device"].default
    assert default == "cpu"
