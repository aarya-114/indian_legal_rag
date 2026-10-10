import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ingestion.text_cleaner import process_all_judgments


class TextCleanerEncodingTests(unittest.TestCase):
    def test_cleaned_judgment_json_is_utf8_on_windows_defaults(self):
        text = "\u201f" + (" judgment text" * 60)
        dataset = {
            "train": [
                {
                    "Titles": "Example v. State",
                    "Text": text,
                    "Court_Name": "Example Court",
                    "Court_Type": "High_Court",
                    "Case_Type": "Civil",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "judgments_clean.json"
            original_open = builtins.open

            def windows_default_open(file, mode="r", *args, **kwargs):
                if "b" not in mode:
                    kwargs.setdefault("encoding", "cp1252")
                return original_open(file, mode, *args, **kwargs)

            with patch("builtins.open", side_effect=windows_default_open):
                process_all_judgments(dataset, str(output_path))

            contents = output_path.read_bytes().decode("utf-8")
            self.assertIn("\u201f", contents)
            self.assertEqual(json.loads(contents)[0]["cleaned_text"], text)


if __name__ == "__main__":
    unittest.main()
