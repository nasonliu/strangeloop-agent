import hashlib
import io
import json
import struct
import unittest

from strangeloop.media import (
    BasicMetadataPerceptor,
    MAX_MEDIA_BYTES,
    MediaArtifact,
    MediaSpan,
    NullMediaPerceptor,
    Percept,
    inspect_media,
    ingest_media,
    media_from_payload,
    media_payload,
    sha256_stream,
)


def png(width=2, height=3):
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"


def jpeg(width=7, height=5):
    return b"\xff\xd8\xff\xc0\x00\x11\x08" + struct.pack(">HH", height, width) + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"


def wav(sample_rate=8000, channels=1, frames=8):
    block_align = channels * 2
    body = b"\x00" * (frames * block_align)
    fmt = struct.pack("<HHIIHH", 1, channels, sample_rate, sample_rate * block_align, block_align, 16)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(body)) + body
    return b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks


class MediaTests(unittest.TestCase):
    def test_png_jpeg_and_wav_metadata(self):
        image = inspect_media(io.BytesIO(png()))
        self.assertEqual((image.modality, image.mime_type, image.width, image.height), ("image", "image/png", 2, 3))
        self.assertNotIn("path", image.to_payload())
        photo = inspect_media(io.BytesIO(jpeg()))
        self.assertEqual((photo.mime_type, photo.width, photo.height), ("image/jpeg", 7, 5))
        sound = inspect_media(io.BytesIO(wav()))
        self.assertEqual((sound.modality, sound.sample_rate_hz, sound.channels, sound.frame_count, sound.duration_ms), ("audio", 8000, 1, 8, 1))

    def test_stream_hash_is_calculated_incrementally_and_mime_must_match(self):
        data = png()
        class ChunkBoundedStream(io.BytesIO):
            def read(self, size=-1):
                if size < 0 or size > 64 * 1024:
                    raise AssertionError("hashing must use bounded reads")
                return super().read(size)

        digest, length, parsed = sha256_stream(ChunkBoundedStream(data))
        self.assertEqual((digest, length, parsed), (hashlib.sha256(data).hexdigest(), len(data), data))
        with self.assertRaises(ValueError):
            inspect_media(io.BytesIO(data), declared_mime_type="audio/wav")

    def test_malformed_unknown_and_over_budget_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            inspect_media(io.BytesIO(b"not media"))
        with self.assertRaises(ValueError):
            inspect_media(io.BytesIO(b"\x89PNG\r\n\x1a\n"))
        with self.assertRaises(ValueError):
            inspect_media(io.BytesIO(b"RIFF\x00\x00\x00\x00WAVE"))
        with self.assertRaises(ValueError):
            sha256_stream(io.BytesIO(b"x" * 8), byte_limit=7)
        with self.assertRaises(ValueError):
            inspect_media(io.BytesIO(png(width=20000, height=1)))
        malformed_wav = bytearray(wav())
        malformed_wav[4] = 0
        with self.assertRaises(ValueError):
            inspect_media(io.BytesIO(bytes(malformed_wav)))

    def test_strict_artifact_span_and_percept_validation(self):
        with self.assertRaises(ValueError):
            MediaArtifact(modality="image", mime_type="image/png", byte_length=1,
                          sha256="A" * 64, width=1, height=1)
        with self.assertRaises(ValueError):
            MediaSpan(kind="time", start_ms=4, end_ms=4)
        with self.assertRaises(ValueError):
            MediaSpan(kind="region", x=0, y=0, width=0, height=1)
        artifact = inspect_media(io.BytesIO(png()))
        with self.assertRaises(ValueError):
            Percept(artifact_id=artifact.artifact_id, modality="image", summary="x", labels=(),
                    confidence=1.0, perceptor_id="test", spans=(MediaSpan(kind="time", start_ms=0, end_ms=1),))

    def test_payload_round_trip_rejects_unrecognized_fields_and_cross_artifact_percepts(self):
        artifact = inspect_media(io.BytesIO(png()))
        percept = Percept(artifact_id=artifact.artifact_id, modality="image", summary="fixture",
                          labels=("fixture",), confidence=.5, perceptor_id="fake",
                          spans=(MediaSpan(kind="region", x=0, y=0, width=1, height=1),))
        payload = media_payload(artifact, (percept,))
        restored, restored_percepts = media_from_payload(payload)
        self.assertEqual(restored.to_payload(), artifact.to_payload())
        self.assertEqual(restored_percepts[0].to_payload(), percept.to_payload())
        payload["artifact"]["path"] = "/private/input.png"
        with self.assertRaises(ValueError):
            media_from_payload(payload)
        with self.assertRaises(ValueError):
            media_payload(artifact, (Percept(artifact_id="other", modality="image", summary="x",
                                             labels=(), confidence=0.5, perceptor_id="fake"),))

    def test_perceptor_protocol_implementations_are_bounded_and_public(self):
        artifact, null_result = ingest_media(io.BytesIO(wav()), NullMediaPerceptor())
        self.assertEqual(null_result, ())
        artifact, result = ingest_media(io.BytesIO(wav()), BasicMetadataPerceptor())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].artifact_id, artifact.artifact_id)
        self.assertIn("8000 Hz", result[0].summary)

    def test_fake_adapter_reads_transient_png_and_wav_bytes_but_payload_has_no_bytes_or_path(self):
        class FakeAdapter:
            perceptor_id = "fixture-adapter/v1"

            def __init__(self):
                self.stream = None

            def perceive(self, artifact, stream):
                self.stream = stream
                raw = stream.read()
                if artifact.modality == "image":
                    if raw[:8] != b"\x89PNG\r\n\x1a\n":
                        raise AssertionError("adapter did not receive PNG bytes")
                    span = MediaSpan(kind="region", x=0, y=0, width=1, height=1)
                else:
                    if raw[8:12] != b"WAVE":
                        raise AssertionError("adapter did not receive WAV bytes")
                    span = MediaSpan(kind="time", start_ms=0, end_ms=1)
                return (Percept(artifact_id=artifact.artifact_id, modality=artifact.modality,
                                summary="fixture annotation", labels=("fixture",),
                                confidence=0.25, perceptor_id=self.perceptor_id, spans=(span,)),)

        fake = FakeAdapter()
        image_bytes = png() + b"SENSITIVE_IMAGE_BYTES"
        artifact, percepts = ingest_media(io.BytesIO(image_bytes), fake)
        self.assertTrue(fake.stream.closed)
        payload = media_payload(artifact, percepts)
        self.assertEqual(payload["percepts"][0]["perceptor_id"], "fixture-adapter/v1")
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn("SENSITIVE_IMAGE_BYTES", serialized)
        self.assertNotIn("path", serialized)
        sound, sound_percepts = ingest_media(io.BytesIO(wav()), FakeAdapter())
        self.assertEqual(sound_percepts[0].spans[0].end_ms, 1)
        self.assertEqual(sound.modality, "audio")

    def test_adapter_results_are_validated_and_stream_is_read_only(self):
        class InvalidAdapter:
            perceptor_id = "invalid/v1"

            def perceive(self, artifact, stream):
                try:
                    stream.write(b"overwrite")
                except io.UnsupportedOperation:
                    pass
                else:
                    raise AssertionError("adapter stream was unexpectedly writable")
                return (Percept(artifact_id=artifact.artifact_id, modality="image", summary="bad",
                                labels=(), confidence=0.5, perceptor_id="other/v1",
                                spans=(MediaSpan(kind="region", x=0, y=0, width=99, height=1),)),)

        with self.assertRaises(ValueError):
            ingest_media(io.BytesIO(png()), InvalidAdapter())

        class OutOfBoundsAdapter:
            perceptor_id = "out-of-bounds/v1"

            def perceive(self, artifact, stream):
                del stream
                if artifact.modality == "image":
                    span = MediaSpan(kind="region", x=2, y=0, width=1, height=1)
                else:
                    span = MediaSpan(kind="time", start_ms=0, end_ms=2)
                return (Percept(artifact_id=artifact.artifact_id, modality=artifact.modality,
                                summary="bad bounds", labels=(), confidence=0.5,
                                perceptor_id=self.perceptor_id, spans=(span,)),)

        with self.assertRaises(ValueError):
            ingest_media(io.BytesIO(png(width=2, height=3)), OutOfBoundsAdapter())
        with self.assertRaises(ValueError):
            ingest_media(io.BytesIO(wav(frames=8)), OutOfBoundsAdapter())


if __name__ == "__main__":
    unittest.main()
