import hashlib
import io
import json
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from scripts.data import download_librispeech as downloader


class DownloadLibriSpeechTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.subset = "dev-clean"
        self.range_requests = []
        self.payload = self.make_archive()
        self.ignore_range = False
        test = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requested_range = self.headers.get("Range")
                test.range_requests.append(requested_range)
                start = int(requested_range.removeprefix("bytes=").removesuffix("-")) if requested_range else 0
                if requested_range and not test.ignore_range:
                    if start >= len(test.payload):
                        self.send_response(416)
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(test.payload) - 1}/{len(test.payload)}")
                    body = test.payload[start:]
                else:
                    self.send_response(200)
                    body = test.payload
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base_url = f"http://127.0.0.1:{server.server_port}/"
        self.checksums = patch.dict(downloader.CHECKSUMS, {self.subset: hashlib.sha256(self.payload).hexdigest()})
        self.counts = patch.dict(downloader.AUDIO_COUNTS, {self.subset: 2})
        self.checksums.start()
        self.counts.start()
        self.addCleanup(self.checksums.stop)
        self.addCleanup(self.counts.stop)

    def make_archive(self, extra=None):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for utterance in ("1-2-0000", "1-2-0001"):
                content = b"fLaC fixture"
                member = tarfile.TarInfo(f"LibriSpeech/{self.subset}/1/2/{utterance}.flac")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            if extra is not None:
                archive.addfile(extra, io.BytesIO(b"x") if extra.isfile() else None)
        return buffer.getvalue()

    @property
    def target(self):
        return self.root / "LibriSpeech" / self.subset

    @property
    def archive(self):
        return self.root / ".downloads" / f"{self.subset}.tar.gz"

    def download(self):
        return downloader.download_librispeech(self.root, (self.subset,), self.base_url)

    def test_success_marker_and_existing_subset_are_validated(self):
        result = self.download()
        self.assertTrue(result["subsets"][self.subset]["archive_sha256_verified"])
        marker = json.loads((self.target / downloader.MARKER_NAME).read_text())
        self.assertEqual(marker["audio_files"], 2)
        self.assertEqual(len(list(self.target.rglob("*.flac"))), 2)
        self.download()
        self.assertEqual(len(self.range_requests), 1)
        (self.target / "1/2/1-2-0000.flac").unlink()
        with self.assertRaisesRegex(ValueError, "found 1 FLAC files, expected 2"):
            self.download()

    def test_partial_transfer_is_resumed_or_restarted_when_range_ignored(self):
        for ignore_range in (False, True):
            with self.subTest(ignore_range=ignore_range):
                root = self.root / str(ignore_range)
                archive = root / ".downloads" / f"{self.subset}.tar.gz"
                archive.parent.mkdir(parents=True)
                partial = archive.with_name(archive.name + ".part")
                partial.write_bytes(self.payload[:35])
                self.ignore_range = ignore_range
                downloader.download_librispeech(root, (self.subset,), self.base_url)
                self.assertEqual(self.range_requests[-1], "bytes=35-")
                self.assertEqual(archive.read_bytes(), self.payload)
                self.assertFalse(partial.exists())

    def test_complete_partial_with_http_416_is_verified_and_published(self):
        self.archive.parent.mkdir(parents=True)
        self.archive.with_name(self.archive.name + ".part").write_bytes(self.payload)
        self.download()
        self.assertTrue(self.target.is_dir())
        self.assertEqual(self.archive.read_bytes(), self.payload)

    def test_bad_hash_rejects_campus_html_without_publishing(self):
        self.payload = b"<html>Please sign in to campus Wi-Fi</html>"
        with self.assertRaisesRegex(ValueError, "non-gzip response.*campus/network login"):
            self.download()
        self.assertFalse(self.target.exists())
        self.assertFalse(self.archive.exists())
        self.assertFalse(self.archive.with_name(self.archive.name + ".part").exists())

    def test_campus_redirect_preserves_an_existing_partial_download(self):
        self.archive.parent.mkdir(parents=True)
        partial = self.archive.with_name(self.archive.name + ".part")
        original = self.payload[:35]
        partial.write_bytes(original)
        self.payload = b"<html>Campus login required</html>"
        self.ignore_range = True
        with self.assertRaisesRegex(ValueError, "existing partial download was preserved"):
            self.download()
        self.assertEqual(partial.read_bytes(), original)
        self.assertFalse(self.target.exists())

    def test_official_top_level_metadata_is_preserved(self):
        member = tarfile.TarInfo("LibriSpeech/LICENSE.TXT")
        member.size = 1
        self.payload = self.make_archive(member)
        downloader.CHECKSUMS[self.subset] = hashlib.sha256(self.payload).hexdigest()
        self.download()
        self.assertEqual((self.root / "LibriSpeech/LICENSE.TXT").read_bytes(), b"x")

    def test_gzip_with_wrong_checksum_is_not_published(self):
        downloader.CHECKSUMS[self.subset] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.download()
        self.assertFalse(self.target.exists())
        self.assertFalse(self.archive.exists())

    def test_path_traversal_and_links_never_publish_subset(self):
        for name, member_type in (("LibriSpeech/../../escape", tarfile.REGTYPE),
                                  ("/absolute/escape", tarfile.REGTYPE),
                                  ("LibriSpeech/dev-clean/link", tarfile.SYMTYPE),
                                  ("LibriSpeech/dev-clean/link", tarfile.LNKTYPE)):
            with self.subTest(name=name, member_type=member_type):
                member = tarfile.TarInfo(name)
                member.type = member_type
                member.size = 1 if member.isfile() else 0
                member.linkname = "../../escape"
                self.payload = self.make_archive(member)
                downloader.CHECKSUMS[self.subset] = hashlib.sha256(self.payload).hexdigest()
                self.archive.unlink(missing_ok=True)
                with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                    self.download()
                self.assertFalse(self.target.exists())
                self.assertFalse((self.root / "escape").exists())
                self.assertFalse(list(self.root.glob(f".{self.subset}-extract-*")))

    def test_existing_partial_directory_is_preserved_and_refused(self):
        audio = self.target / "1/2/1-2-0000.flac"
        audio.parent.mkdir(parents=True)
        audio.write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "Incomplete subset.*existing data was preserved"):
            self.download()
        self.assertEqual(audio.read_bytes(), b"existing")
        self.assertFalse(self.range_requests)
        self.assertFalse((self.target / downloader.MARKER_NAME).exists())

    def test_archive_with_incomplete_audio_count_is_not_published(self):
        downloader.AUDIO_COUNTS[self.subset] = 3
        with self.assertRaisesRegex(ValueError, "found 2 FLAC files, expected 3"):
            self.download()
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
