"""Small corpus fixtures verify the upstream TAR wrapper without huge downloads."""

import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_data as data


def tar_gzip(files, directory=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.GNU_FORMAT) as archive:
        if directory:
            member = tarfile.TarInfo("nested")
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
        for name, contents in files:
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    return gzip.compress(stream.getvalue())


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.records = b'{"id":"0","contents":"title\\nfirst passage"}\n{"id":"1","contents":"second passage"}\n'

    def tearDown(self):
        self.temp.cleanup()

    def test_plain_gzip_and_upstream_tar_gzip_produce_identical_jsonl(self):
        for compressed in [gzip.compress(self.records),
                           tar_gzip([("data00/nested/wiki_dump.jsonl", self.records)], directory=True)]:
            path = self.root / "wiki-18.jsonl.gz"
            path.write_bytes(compressed)
            self.assertEqual(b"".join(data.iter_corpus_lines(path)), self.records)
        self.assertFalse((self.root / "data00").exists(), "Archive paths must never be extracted")

    def test_archive_requires_exactly_one_jsonl_payload(self):
        fixtures = [tar_gzip([], directory=True),
                    tar_gzip([("first.jsonl", self.records), ("second.jsonl", self.records)]),
                    tar_gzip([("unexpected.txt", self.records)])]
        for compressed in fixtures:
            path = self.root / "wiki-18.jsonl.gz"
            path.write_bytes(compressed)
            with self.assertRaises(ValueError):
                list(data.iter_corpus_lines(path))

    def run_pipeline(self, payload):
        blobs = {"part_aa": b"first-index-part", "part_ab": b"second-index-part",
                 "wiki-18.jsonl.gz": tar_gzip([("data00/nested/wiki_dump.jsonl", payload)])}

        def repository(endpoint, name):
            names = ["part_aa", "part_ab"] if name.endswith("index") else ["wiki-18.jsonl.gz"]
            return "fixture-revision", {name: {"size": len(blobs[name]),
                                               "lfs": {"oid": hashlib.sha256(blobs[name]).hexdigest()}}
                                        for name in names}

        def download(endpoint, repository, revision, name, destination, metadata):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(blobs[name])
            return {"file": name, "sha256": data.verify_download(destination, metadata)}

        with patch.object(data, "repository", repository), patch.object(data, "download", download), \
                patch.object(data.shutil, "disk_usage", return_value=SimpleNamespace(free=200_000_000_000)):
            data.prepare_retrieval(SimpleNamespace(data_dir=self.root, endpoint="fixture-only"))

    def test_real_pipeline_extracts_tar_payload_and_records_actual_bytes(self):
        self.run_pipeline(self.records)
        corpus = self.root / "corpus"
        self.assertEqual((corpus / "wiki-18.jsonl").read_bytes(), self.records)
        self.assertEqual((corpus / "e5_Flat.index").read_bytes(), b"first-index-partsecond-index-part")
        manifest = json.loads((corpus / "retrieval_manifest.json").read_text())
        self.assertEqual(manifest["corpus"]["passages"], 2)
        self.assertEqual(manifest["corpus"]["bytes"], len(self.records))
        self.assertEqual(manifest["corpus"]["sha256"], hashlib.sha256(self.records).hexdigest())

    def test_bad_payload_is_not_published_as_final_corpus(self):
        with self.assertRaises(ValueError):
            self.run_pipeline(b'{"id":"0","no_contents":"bad document"}\n')
        self.assertFalse((self.root / "corpus/wiki-18.jsonl").exists())
        self.assertFalse((self.root / "corpus/retrieval_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
