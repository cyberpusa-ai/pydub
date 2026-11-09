from functools import partial
import errno
import os
import sys
import unittest
from tempfile import (
    NamedTemporaryFile,
    mkdtemp,
    gettempdir
)
import tempfile
import struct
import subprocess
import warnings
from io import BytesIO

from pydub import AudioSegment
from pydub import audio_segment
from pydub.audio_segment import extract_wav_headers
from pydub.utils import (
    db_to_float,
    ratio_to_db,
    make_chunks,
    mediainfo,
    mediainfo_json,
    which,
    get_encoder_name,
    get_supported_decoders,
    get_supported_encoders,
    stereo_to_ms,
    ms_to_stereo,
)
from pydub.exceptions import (
    InvalidTag,
    InvalidID3TagVersion,
    InvalidDuration,
    CouldntDecodeError,
    CouldntEncodeError,
    MissingAudioParameter,
)
from pydub.silence import (
    detect_silence,
    split_on_silence,
)
from pydub.generators import (
    Sine,
    Square,
    Pulse,
    Triangle,
    Sawtooth,
    WhiteNoise,
)

data_dir = os.path.join(os.path.dirname(__file__), 'data')


def has_encoder(name):
    """
    get_supported_encoders() lists codecs (e.g. "vorbis"), not the encoder
    libraries that implement them (e.g. "libvorbis"), so ask ffmpeg directly.
    """
    try:
        p = subprocess.Popen([get_encoder_name(), '-encoders'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return False
    output = p.communicate()[0].decode('utf-8', 'ignore')
    return any(line.split()[1:2] == [name] for line in output.splitlines())


HAS_LIBVORBIS = has_encoder('libvorbis')


class UtilityTests(unittest.TestCase):

    def test_db_float_conversions(self):
        self.assertEqual(db_to_float(20), 10)
        self.assertEqual(db_to_float(10, using_amplitude=False), 10)
        self.assertEqual(db_to_float(0), 1)
        self.assertEqual(ratio_to_db(1), 0)
        self.assertEqual(ratio_to_db(10), 20)
        self.assertEqual(ratio_to_db(10, using_amplitude=False), 10)
        self.assertEqual(3, db_to_float(ratio_to_db(3)))
        self.assertEqual(12, ratio_to_db(db_to_float(12)))
        self.assertEqual(3, db_to_float(ratio_to_db(3, using_amplitude=False), using_amplitude=False))
        self.assertEqual(12, ratio_to_db(db_to_float(12, using_amplitude=False), using_amplitude=False))

    def test_stereo_to_ms_and_back(self):
        left = Sine(440).to_audio_segment(duration=500, volume=-20)
        right = Sine(660).to_audio_segment(duration=500, volume=-20)
        stereo = AudioSegment.from_mono_audiosegments(left, right)

        mid_side = stereo_to_ms(stereo)
        self.assertTrue(isinstance(mid_side, AudioSegment))
        self.assertEqual(mid_side.channels, 2)
        self.assertEqual(len(mid_side), len(stereo))

        back = ms_to_stereo(mid_side)
        self.assertTrue(isinstance(back, AudioSegment))
        self.assertEqual(back.channels, 2)
        self.assertEqual(len(back), len(stereo))
        new_left, new_right = back.split_to_mono()
        self.assertAlmostEqual(new_left.dBFS - left.dBFS, 3, places=0)
        self.assertAlmostEqual(new_right.dBFS - right.dBFS, 3, places=0)

    def test_stereo_to_ms_identical_channels_has_silent_side(self):
        mono = Sine(440).to_audio_segment(duration=500, volume=-20)
        stereo = AudioSegment.from_mono_audiosegments(mono, mono)
        mid, side = stereo_to_ms(stereo).split_to_mono()
        self.assertEqual(side.rms, 0)
        self.assertTrue(mid.dBFS > mono.dBFS)


if sys.version_info >= (3, 6):
    class PathLikeObjectTests(unittest.TestCase):

        class MyPathLike:
            def __init__(self, path):
                self.path = path

            def __fspath__(self):
                return self.path

        def setUp(self):
            self.mp3_path_str = os.path.join(data_dir, 'test1.mp3')

            from pathlib import Path
            self.mp3_pathlib_path = Path(self.mp3_path_str)

            self.mp3_path_like_str = self.MyPathLike(self.mp3_path_str)
            self.mp3_path_like_bytes = self.MyPathLike(bytes(self.mp3_path_str, sys.getdefaultencoding()))

        def test_audio_segment_from_pathlib_path(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            seg2 = AudioSegment.from_file(self.mp3_pathlib_path)

            self.assertEqual(len(seg1), len(seg2))
            self.assertEqual(seg1._data, seg2._data)
            self.assertTrue(len(seg1) > 0)

        def test_audio_segment_from_path_like_str(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            seg2 = AudioSegment.from_file(self.mp3_path_like_str)

            self.assertEqual(len(seg1), len(seg2))
            self.assertEqual(seg1._data, seg2._data)
            self.assertTrue(len(seg1) > 0)

        def test_audio_segment_from_path_like_bytes(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            seg2 = AudioSegment.from_file(self.mp3_path_like_bytes)

            self.assertEqual(len(seg1), len(seg2))
            self.assertEqual(seg1._data, seg2._data)
            self.assertTrue(len(seg1) > 0)

        def test_non_existant_pathlib_path(self):
            from pathlib import Path
            path = Path('this/path/should/not/exist/do/not/make/this/exist')
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

            path = Path('')
            # On Unicies this will raise a IsADirectoryError, on Windows this
            # will result in a PermissionError. Both of these are subclasses of
            # OSError. We aren't so much worried about the specific exception
            # here, just that reading a file from an empty path is an error.
            with self.assertRaises(OSError):
                _ = AudioSegment.from_file(path)

        def test_non_existant_path_like_str(self):
            path = self.MyPathLike('this/path/should/not/exist/do/not/make/this/exist')
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

            path = self.MyPathLike('')
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

        def test_non_existant_path_like_bytes(self):
            path = self.MyPathLike(bytes('this/path/should/not/exist/do/not/make/this/exist', sys.getdefaultencoding()))
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

            path = self.MyPathLike(bytes('', sys.getdefaultencoding()))
            with self.assertRaises(FileNotFoundError):
                _ = AudioSegment.from_file(path)

        def assertWithinRange(self, val, lower_bound, upper_bound):
            self.assertTrue(lower_bound <= val <= upper_bound,
                            "%s is not in the acceptable range: %s - %s" %
                            (val, lower_bound, upper_bound))

        def assertWithinTolerance(self, val, expected, tolerance=None,
                                  percentage=None):
            if percentage is not None:
                tolerance = expected * percentage
            lower_bound = expected - tolerance
            upper_bound = expected + tolerance
            self.assertWithinRange(val, lower_bound, upper_bound)

        def test_export_pathlib_path(self):
            seg1 = AudioSegment.from_file(self.mp3_path_str)
            from pathlib import Path
            path = Path(tempfile.gettempdir()) / 'pydub-test-export-8ajds.mp3'
            out_f = None
            try:
                out_f = seg1.export(path, format='mp3')
                seg2 = AudioSegment.from_file(path, format='mp3')

                self.assertTrue(len(seg1) > 0)
                self.assertWithinTolerance(len(seg1),
                                           len(seg2),
                                           percentage=0.01)
            finally:
                if out_f:
                    out_f.close()
                os.unlink(path)


class FileAccessTests(unittest.TestCase):

    def setUp(self):
        self.mp3_path = os.path.join(data_dir, 'test1.mp3')

    def test_audio_segment_from_mp3(self):
        seg1 = AudioSegment.from_mp3(os.path.join(data_dir, 'test1.mp3'))

        mp3_file = open(os.path.join(data_dir, 'test1.mp3'), 'rb')
        seg2 = AudioSegment.from_mp3(mp3_file)

        self.assertEqual(len(seg1), len(seg2))
        self.assertTrue(seg1._data == seg2._data)
        self.assertTrue(len(seg1) > 0)

    def _in_dir_with_dash_named_copy(self, test):
        # a relative filename starting with "-" must not be parsed as an option
        import shutil
        orig_cwd = os.getcwd()
        tmp_dir = mkdtemp()
        try:
            shutil.copy(self.mp3_path, os.path.join(tmp_dir, '-x.mp3'))
            os.chdir(tmp_dir)
            test('-x.mp3')
        finally:
            os.chdir(orig_cwd)
            shutil.rmtree(tmp_dir)

    def test_mediainfo_filename_starting_with_dash(self):
        def test(filename):
            info = mediainfo_json(filename)
            self.assertEqual(info['streams'][0]['codec_type'], 'audio')
            self.assertEqual(mediainfo(filename)['codec_name'], 'mp3')
        self._in_dir_with_dash_named_copy(test)

    def test_from_file_filename_starting_with_dash(self):
        expected = AudioSegment.from_file(self.mp3_path)

        def test(filename):
            seg = AudioSegment.from_file(filename)
            self.assertEqual(len(seg), len(expected))
            self.assertEqual(seg._data, expected._data)
            seg = AudioSegment.from_file_using_temporary_files(filename)
            self.assertEqual(len(seg), len(expected))
        self._in_dir_with_dash_named_copy(test)


test1wav = test4wav = test1 = test2 = test3 = testparty = testdcoffset = None


class AudioSegmentTests(unittest.TestCase):

    def setUp(self):
        global test1, test2, test3, testparty, testdcoffset
        if not test1:
            a = os.path.join(data_dir, 'test1.mp3')
            test1 = AudioSegment.from_mp3(os.path.join(data_dir, 'test1.mp3'))
            test2 = AudioSegment.from_mp3(os.path.join(data_dir, 'test2.mp3'))
            test3 = AudioSegment.from_mp3(os.path.join(data_dir, 'test3.mp3'))
            testdcoffset = AudioSegment.from_mp3(
                os.path.join(data_dir, 'test-dc_offset.wav'))
            testparty = AudioSegment.from_mp3(
                os.path.join(data_dir, 'party.mp3'))

        self.seg1 = test1
        self.seg2 = test2
        self.seg3 = test3
        self.mp3_seg_party = testparty
        self.seg_dc_offset = testdcoffset

        self.ogg_file_path = os.path.join(data_dir, 'bach.ogg')
        self.mp4_file_path = os.path.join(data_dir, 'creative_common.mp4')
        self.mp3_file_path = os.path.join(data_dir, 'party.mp3')
        self.webm_file_path = os.path.join(data_dir, 'test5.webm')

        self.jpg_cover_path = os.path.join(data_dir, 'cover.jpg')
        self.png_cover_path = os.path.join(data_dir, 'cover.png')

    def assertWithinRange(self, val, lower_bound, upper_bound):
        self.assertTrue(lower_bound <= val <= upper_bound,
                        "%s is not in the acceptable range: %s - %s" %
                        (val, lower_bound, upper_bound))

    def assertWithinTolerance(self, val, expected, tolerance=None,
                              percentage=None):
        if percentage is not None:
            tolerance = expected * percentage
        lower_bound = expected - tolerance
        upper_bound = expected + tolerance
        self.assertWithinRange(val, lower_bound, upper_bound)

    def test_direct_instantiation_with_bytes(self):
        seg = AudioSegment(
            b'RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00\x00}\x00\x00\x00\xf4\x01\x00\x04\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00')
        self.assertEqual(seg.frame_count(), 1)
        self.assertEqual(seg.channels, 2)
        self.assertEqual(seg.sample_width, 2)
        self.assertEqual(seg.frame_rate, 32000)

    def test_24_bit_audio(self):
        path24 = os.path.join(data_dir, 'test1-24bit.wav')
        seg24 = AudioSegment._from_safe_wav(path24)
        # The data length lies at bytes 40-44
        with open(path24, 'rb') as f:
            raw24 = f.read()
        len24 = struct.unpack("<L", raw24[40:44])[0]

        # should have been converted to 32 bit
        self.assertEqual(seg24.sample_width, 4)
        # the data length should have grown by exactly 4:3 (24 bits turn into 32 bits)
        self.assertEqual(len(seg24.raw_data) * 3, len24 * 4)

    def test_export_24_bit_wav(self):
        # 24-bit audio is held as 32-bit, so a plain wav export is 32-bit.
        # codec='pcm_s24le' writes a real 24-bit wav without losing samples.
        seg24 = AudioSegment.from_file(os.path.join(data_dir, 'test1-24bit.wav'))
        self.assertEqual(seg24.sample_width, 4)

        import shutil
        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, 'out24.wav')
            seg24.export(path, format='wav', codec='pcm_s24le').close()

            info = mediainfo(path)
            self.assertEqual(info['codec_name'], 'pcm_s24le')
            self.assertEqual(info['bits_per_sample'], '24')

            reloaded = AudioSegment.from_file(path)
            self.assertEqual(reloaded.sample_width, 4)
            self.assertEqual(reloaded.frame_rate, seg24.frame_rate)
            self.assertEqual(reloaded.channels, seg24.channels)
            self.assertEqual(reloaded.raw_data, seg24.raw_data)
        finally:
            shutil.rmtree(tmp_dir)

    def test_8_bit_audio(self):
        original_path = os.path.join(data_dir,'test1.wav')
        original_segment = AudioSegment.from_file(original_path)
        target_rms = original_segment.rms//2**8

        path_with_8bits = os.path.join(data_dir,'test1-8bit.wav')

        def check_8bit_segment(segment):
            self.assertWithinTolerance(segment.rms,target_rms,tolerance=0)

        # check reading directly
        check_8bit_segment(AudioSegment.from_file(path_with_8bits))

        # check using ffmpeg on it
        with open(path_with_8bits,'rb') as file_8bit:
            check_8bit_segment(AudioSegment.from_file(file_8bit))

        # check conversion from higher-width sample
        check_8bit_segment(AudioSegment.from_file(original_path).set_sample_width(1))

        # check audio export
        with NamedTemporaryFile('w+b', suffix='.wav') as tmp_file:
            original_segment.set_sample_width(1).export(tmp_file,format='wav')
            tmp_file.seek(0)
            check_8bit_segment(AudioSegment.from_file(tmp_file))

    def test_192khz_audio(self):
        test_files = [('test-192khz-16bit.wav', 16),
                      ('test-192khz-24bit.wav', 32),
                      ('test-192khz-32bit.flac', 32),
                      ('test-192khz-32bit.wav', 32),
                      ('test-192khz-64bit.wav', 64)]
        base_file, bit_depth = test_files[0]
        path = os.path.join(data_dir, base_file)
        base = AudioSegment.from_file(path)

        with open(path, 'rb') as f:
            headers = extract_wav_headers(f.read())
        data16_size = headers[-1].size
        self.assertEqual(len(base.raw_data), data16_size)
        self.assertEqual(base.frame_rate, 192000)
        self.assertEqual(base.sample_width, bit_depth / 8)

        for test_file, bit_depth in test_files[1:]:
            path = os.path.join(data_dir, test_file)
            seg = AudioSegment.from_file(path)
            self.assertEqual(seg.sample_width, bit_depth / 8)
            self.assertEqual(seg.frame_rate, 192000)
            self.assertEqual(len(seg.raw_data), len(base.raw_data) *
                             seg.sample_width / base.sample_width)
            self.assertEqual(seg.frame_rate, 192000)

    def test_concat(self):
        catted_audio = self.seg1 + self.seg2

        expected = len(self.seg1) + len(self.seg2)
        self.assertWithinTolerance(len(catted_audio), expected, 1)

    def test_append(self):
        merged1 = self.seg3.append(self.seg1, crossfade=100)
        merged2 = self.seg3.append(self.seg2, crossfade=100)

        self.assertEqual(len(merged1), len(self.seg1) + len(self.seg3) - 100)
        self.assertEqual(len(merged2), len(self.seg2) + len(self.seg3) - 100)

    def test_too_long_crossfade(self):
        s1 = self.seg1[:1000]
        s2 = self.seg2[:500]
        self.assertRaises(ValueError, lambda: s1.append(s2, crossfade=len(s1) + 10))

    def test_sum(self):
        def gen():
            yield self.seg1
            yield self.seg2
            yield self.seg3

        try:
            summed = sum(gen())
        except TypeError as e:
            if "unsupported operand" in str(e):
                self.fail("Could not sum() audio segments.")
            else:
                raise

    def test_volume_with_add_sub(self):
        quieter = self.seg1 - 6
        self.assertAlmostEqual(ratio_to_db(quieter.rms, self.seg1.rms),
                               -6,
                               places=2)

        louder = quieter + 2.5
        self.assertAlmostEqual(ratio_to_db(louder.rms, quieter.rms),
                               2.5,
                               places=2)

    def test_repeat_with_multiply(self):
        seg = self.seg1 * 3
        expected = len(self.seg1) * 3
        expected = (expected - 2, expected + 2)
        self.assertTrue(expected[0] < len(seg) < expected[1])

    def test_overlay(self):
        seg_mult = self.seg1[:5000] * self.seg2[:3000]
        seg_over = self.seg1[:5000].overlay(self.seg2[:3000], loop=True)

        self.assertEqual(len(seg_mult), len(seg_over))
        self.assertTrue(seg_mult._data == seg_over._data)

        self.assertEqual(len(seg_mult), 5000)
        self.assertEqual(len(seg_over), 5000)

    def test_overlay_times(self):
        # infinite
        seg_mult = self.seg1[:5000] * self.seg2[:3000]
        seg_over = self.seg1[:5000].overlay(self.seg2[:3000], times=99999999)
        self.assertEqual(len(seg_mult), len(seg_over))
        self.assertEqual(len(seg_over), 5000)
        self.assertTrue(seg_mult._data == seg_over._data)

        # no times, no-op
        piece = self.seg2[:1000]
        seg_manual = self.seg1[:4000]
        seg_over = self.seg1[:4000].overlay(piece, times=0)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 1 loop
        seg_manual = self.seg1[:4000].overlay(piece, position=500)
        seg_over = self.seg1[:4000].overlay(piece, times=1)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 2 loops
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500)
        seg_over = self.seg1[:4000].overlay(piece, times=2)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 3 loops
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500).overlay(piece, position=2500)
        seg_over = self.seg1[:4000].overlay(piece, times=3)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 4 loops (last will pass end)
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500).overlay(piece, position=2500) \
            .overlay(piece, position=3500)
        seg_over = self.seg1[:4000].overlay(piece, times=4)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # 5 loops (last won't happen b/c past end)
        seg_manual = self.seg1[:4000].overlay(piece, position=500) \
            .overlay(piece, position=1500).overlay(piece, position=2500) \
            .overlay(piece, position=3500).overlay(piece, position=3500)
        seg_over = self.seg1[:4000].overlay(piece, times=5)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

        # ~infinite, same (as 4 and 5 really)
        seg_over = self.seg1[:4000].overlay(piece, times=999999999)
        self.assertEqual(len(seg_manual), len(seg_over))
        self.assertEqual(len(seg_over), 4000)
        self.assertFalse(seg_mult._data == seg_over._data)

    def test_overlay_with_gain_change(self):
        # Use overlay silence with volume change
        seg_one = self.seg1[:5000]
        seg_silent = AudioSegment.silent(duration=2000)
        seg_over = seg_one.overlay(seg_silent, gain_during_overlay=-7)

        # Manually lower first segment
        seg_one_lower = seg_one - 7
        seg_manual = seg_one_lower[:2000] + seg_one[2000:]

        self.assertEqual(len(seg_over), len(seg_manual))
        self.assertAlmostEqual(seg_over.dBFS, seg_manual.dBFS)
        self.assertEqual(len(seg_manual), 5000)
        self.assertEqual(len(seg_over), 5000)

    def test_slicing(self):
        empty = self.seg1[:0]
        second_long_slice = self.seg1[:1000]
        remainder = self.seg1[1000:]

        self.assertEqual(len(empty), 0)
        self.assertEqual(len(second_long_slice), 1000)
        self.assertEqual(len(remainder), len(self.seg1) - 1000)

        last_5_seconds = self.seg1[-5000:]
        before = self.seg1[:-5000]

        self.assertEqual(len(last_5_seconds), 5000)
        self.assertEqual(len(before), len(self.seg1) - 5000)

        past_end = second_long_slice[:1500]
        self.assertTrue(second_long_slice._data == past_end._data)

    def test_slicing_by_step(self):
        audio = self.seg1[:10000]
        chunks = audio[:0]

        for chunk in audio[::1000]:
            self.assertTrue(isinstance(chunk, AudioSegment))
            self.assertEqual(len(chunk), 1000)
            chunks += chunk

        self.assertEqual(len(audio), len(chunks))

    def test_indexing(self):
        short = self.seg1[:100]

        rebuilt1 = self.seg1[:0]
        for part in short:
            rebuilt1 += part

        rebuilt2 = sum([part for part in short])

        self.assertTrue(short._data == rebuilt1._data)
        self.assertTrue(short._data == rebuilt2._data)

    def test_set_channels(self):
        mono = self.seg1.set_channels(1)
        stereo = mono.set_channels(2)

        self.assertEqual(len(self.seg1), len(mono))
        self.assertEqual(len(self.seg1), len(stereo))

        mono = self.seg2.set_channels(1)
        mono = mono.set_frame_rate(22050)

        self.assertEqual(len(mono), len(self.seg2))

        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_file:
            if sys.platform == 'win32':
                tmp_file.close()

            out_f = mono.export(tmp_file.name, 'mp3')
            out_f.close()
            monomp3 = AudioSegment.from_mp3(tmp_file.name)

            self.assertWithinTolerance(
                len(monomp3),
                len(self.seg2),
                tolerance=105
            )

            if sys.platform == 'win32':
                os.remove(tmp_file.name)

        merged = mono.append(stereo, crossfade=100)
        self.assertWithinTolerance(len(merged),
                                   len(self.seg1) + len(self.seg2) - 100,
                                   tolerance=1)

    def test_split_to_mono(self):
        seg = self.seg1
        mono_segments = seg.split_to_mono()
        seg_lchannel = mono_segments[0]
        seg_rchannel = mono_segments[1]

        self.assertEqual(len(seg_lchannel), len(seg))
        self.assertEqual(len(seg_rchannel), len(seg))

        self.assertEqual(seg_lchannel.frame_rate, seg.frame_rate)
        self.assertEqual(seg_rchannel.frame_rate, seg.frame_rate)

        self.assertEqual(seg_lchannel.frame_count(), seg.frame_count())
        self.assertEqual(seg_rchannel.frame_count(), seg.frame_count())

    def test_apply_gain_stereo(self):
        seg = self.seg1

        orig_l, orig_r = seg.split_to_mono()
        orig_dbfs_l = orig_l.dBFS
        orig_dbfs_r = orig_r.dBFS

        # for readability: infinity
        inf = float("inf")

        def assertAlmostEqual(v1, v2, **kwargs):
            if v1 in (inf, -inf):
                self.assertEqual(v1, v2)
            else:
                self.assertAlmostEqual(v1, v2, **kwargs)

        def check_stereo_gain(left_dbfs_change, right_dbfs_change):
            panned = seg.apply_gain_stereo(left_dbfs_change, right_dbfs_change)
            self.assertEqual(panned.channels, 2)

            l, r = panned.split_to_mono()
            assertAlmostEqual(l.dBFS, orig_dbfs_l + left_dbfs_change, places=2)
            assertAlmostEqual(r.dBFS, orig_dbfs_r + right_dbfs_change, places=2)

        # hard left
        check_stereo_gain(0.0, -inf)
        check_stereo_gain(0.0, -6.0)
        check_stereo_gain(0.0, 0.0)
        check_stereo_gain(-6.0, 0.0)
        check_stereo_gain(-inf, 0.0)

    def test_pan(self):
        seg = self.seg1

        orig_l, orig_r = seg.split_to_mono()
        orig_dbfs_l = orig_l.dBFS
        orig_dbfs_r = orig_r.dBFS

        # for readability: infinity
        inf = float("inf")

        def assertAlmostEqual(v1, v2, **kwargs):
            if v1 in (inf, -inf):
                self.assertEqual(v1, v2)
            else:
                self.assertAlmostEqual(v1, v2, **kwargs)

        def check_pan(pan, left_dbfs_change, right_dbfs_change):
            panned = seg.pan(pan)

            l, r = panned.split_to_mono()
            assertAlmostEqual(l.dBFS, orig_dbfs_l + left_dbfs_change, places=1)
            assertAlmostEqual(r.dBFS, orig_dbfs_r + right_dbfs_change, places=1)

        check_pan(-1.0, 3.0, -inf)
        check_pan(-0.5, 1.5, -4.65)
        check_pan(0.0, 0.0, 0.0)
        check_pan(0.5, -4.65, 1.5)
        check_pan(1.0, -inf, 3.0)

    def test_export_as_mp3(self):
        seg = self.seg1
        exported_mp3 = seg.export()
        seg_exported_mp3 = AudioSegment.from_mp3(exported_mp3)

        self.assertWithinTolerance(len(seg_exported_mp3),
                                   len(seg),
                                   percentage=0.01)

    def test_export_with_int_bitrate(self):
        seg = self.seg1[:2000]
        exported_mp3 = seg.export(format="mp3", bitrate=128000)
        seg_exported_mp3 = AudioSegment.from_mp3(exported_mp3)

        self.assertWithinTolerance(len(seg_exported_mp3),
                                   len(seg),
                                   percentage=0.01)

        # an int bitrate behaves the same as the equivalent string
        int_size = len(seg.export(format="mp3", bitrate=64000).read())
        str_size = len(seg.export(format="mp3", bitrate="64000").read())
        self.assertEqual(int_size, str_size)

    def test_export_as_wav(self):
        seg = self.seg1
        exported_wav = seg.export(format='wav')
        seg_exported_wav = AudioSegment.from_wav(exported_wav)

        self.assertWithinTolerance(len(seg_exported_wav),
                                   len(seg),
                                   percentage=0.01)

    def test_export_as_wav_with_codec(self):
        seg = self.seg1
        exported_wav = seg.export(format='wav', codec='pcm_s32le')
        seg_exported_wav = AudioSegment.from_wav(exported_wav)

        self.assertWithinTolerance(len(seg_exported_wav),
                                   len(seg),
                                   percentage=0.01)
        self.assertEqual(seg_exported_wav.sample_width, 4)

    def test_export_as_wav_with_parameters(self):
        seg = self.seg1
        exported_wav = seg.export(format='wav', parameters=['-ar', '16000', '-ac', '1'])
        seg_exported_wav = AudioSegment.from_wav(exported_wav)

        self.assertWithinTolerance(len(seg_exported_wav),
                                   len(seg),
                                   percentage=0.01)
        self.assertEqual(seg_exported_wav.frame_rate, 16000)
        self.assertEqual(seg_exported_wav.channels, 1)

    def test_from_file_with_parameters(self):
        # parameters must be output options (before the output target), not
        # trailing options that ffmpeg ignores
        orig = AudioSegment.from_file(self.mp3_file_path)
        self.assertNotEqual(orig.frame_rate, 8000)
        self.assertEqual(orig.channels, 1)
        seg = AudioSegment.from_file(self.mp3_file_path,
                                     parameters=['-ar', '8000', '-ac', '2'])
        self.assertEqual(seg.frame_rate, 8000)
        self.assertEqual(seg.channels, 2)
        self.assertWithinTolerance(len(seg), len(orig), percentage=0.01)

        with open(self.mp3_file_path, 'rb') as mp3_file:
            seg = AudioSegment.from_file(mp3_file, format='mp3',
                                         parameters=['-ar', '8000'])
        self.assertEqual(seg.frame_rate, 8000)

    def test_from_file_using_temporary_files_with_parameters(self):
        orig = AudioSegment.from_file(self.mp3_file_path)
        seg = AudioSegment.from_file_using_temporary_files(
            self.mp3_file_path, parameters=['-ar', '8000', '-ac', '2'])
        self.assertEqual(seg.frame_rate, 8000)
        self.assertEqual(seg.channels, 2)
        self.assertWithinTolerance(len(seg), len(orig), percentage=0.01)

    def test_export_as_raw(self):
        seg = self.seg1
        exported_raw = seg.export(format='raw')
        seg_exported_raw = AudioSegment.from_raw(exported_raw, sample_width=seg.sample_width, frame_rate=seg.frame_rate,
                                                 channels=seg.channels)

        self.assertWithinTolerance(len(seg_exported_raw),
                                   len(seg),
                                   percentage=0.01)

    def test_export_as_raw_with_codec(self):
        seg = self.seg1
        with self.assertRaises(AttributeError):
            seg.export(format='raw', codec='pcm_s32le')

    def test_export_as_raw_with_parameters(self):
        seg = self.seg1
        with self.assertRaises(AttributeError):
            seg.export(format='raw', parameters=['-ar', '16000', '-ac', '1'])

    @unittest.skipUnless(HAS_LIBVORBIS,
                         "Unsupported codecs")
    def test_export_as_ogg(self):
        seg = self.seg1
        exported_ogg = seg.export(format='ogg')
        seg_exported_ogg = AudioSegment.from_ogg(exported_ogg)

        self.assertWithinTolerance(len(seg_exported_ogg),
                                   len(seg),
                                   percentage=0.01)

    @unittest.skipUnless(HAS_LIBVORBIS,
                         "Unsupported codecs")
    def test_export_forced_codec(self):
        seg = self.seg1 + self.seg2

        with NamedTemporaryFile('w+b', suffix='.ogg') as tmp_file:
            if sys.platform == 'win32':
                tmp_file.close()

            seg.export(tmp_file.name, 'ogg', codec='libvorbis')
            exported = AudioSegment.from_ogg(tmp_file.name)
            self.assertWithinTolerance(len(exported),
                                       len(seg),
                                       percentage=0.01)
            if sys.platform == 'win32':
                os.remove(tmp_file.name)

    def test_fades(self):
        seg = self.seg1[:10000]

        # 1 ms difference in the position of the end of the fade out
        inf_end = seg.fade(start=0, end=float('inf'), to_gain=-120)
        negative_end = seg.fade(start=0, end=-1, to_gain=-120)

        self.assertWithinTolerance(inf_end.rms, negative_end.rms,
                                   percentage=0.001)
        self.assertTrue(negative_end.rms <= inf_end.rms)
        self.assertTrue(inf_end.rms < seg.rms)

        self.assertEqual(len(inf_end), len(seg))

        self.assertTrue(-6 < ratio_to_db(inf_end.rms, seg.rms) < -5)

        # use a slice out of the middle to make sure there is audio
        seg = self.seg2[2000:8000]
        fade_out = seg.fade_out(1000)
        fade_in = seg.fade_in(1000)

        self.assertTrue(0 < fade_out.rms < seg.rms)
        self.assertTrue(0 < fade_in.rms < seg.rms)

        self.assertEqual(len(fade_out), len(seg))
        self.assertEqual(len(fade_in), len(seg))

        db_at_beginning = ratio_to_db(fade_in[:1000].rms, seg[:1000].rms)
        db_at_end = ratio_to_db(fade_in[-1000:].rms, seg[-1000:].rms)
        self.assertTrue(db_at_beginning < db_at_end)

        db_at_beginning = ratio_to_db(fade_out[:1000].rms, seg[:1000].rms)
        db_at_end = ratio_to_db(fade_out[-1000:].rms, seg[-1000:].rms)
        self.assertTrue(db_at_end < db_at_beginning)

    def test_reverse(self):
        seg = self.seg1
        rseg = seg.reverse()

        # the reversed audio should be exactly equal in playback duration
        self.assertEqual(len(seg), len(rseg))

        r2seg = rseg.reverse()

        # if you reverse it twice you should get an identical AudioSegment
        self.assertEqual(seg, r2seg)

    def test_normalize(self):
        seg = self.seg1
        normalized = seg.normalize(0.0)

        self.assertEqual(len(normalized), len(seg))
        self.assertTrue(normalized.rms > seg.rms)
        self.assertWithinTolerance(
            normalized.max,
            normalized.max_possible_amplitude,
            percentage=0.0001
        )

    def test_for_accidental_shortening(self):
        seg = self.mp3_seg_party
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            if sys.platform == 'win32':
                tmp_mp3_file.close()

            fd = seg.export(tmp_mp3_file.name)
            fd.close()

            for i in range(3):
                fd = AudioSegment.from_mp3(tmp_mp3_file.name).export(tmp_mp3_file.name, "mp3")
                fd.close()

            tmp_seg = AudioSegment.from_mp3(tmp_mp3_file.name)
            self.assertAlmostEqual(len(tmp_seg), len(seg), places=1)

            if sys.platform == 'win32':
                os.remove(tmp_mp3_file.name)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_formats(self):
        seg_m4a = AudioSegment.from_file(
            os.path.join(data_dir, 'format_test.m4a'), "m4a")
        self.assertTrue(len(seg_m4a))

    def test_equal_and_not_equal(self):
        wav_file = self.seg1.export(format='wav')
        wav = AudioSegment.from_wav(wav_file)
        self.assertTrue(self.seg1 == wav)
        self.assertFalse(self.seg1 != wav)

    def test_duration(self):
        self.assertEqual(int(self.seg1.duration_seconds), 10)

        wav_file = self.seg1.export(format='wav')
        wav = AudioSegment.from_wav(wav_file)
        self.assertEqual(wav.duration_seconds, self.seg1.duration_seconds)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_autodetect_format(self):
        aac_path = os.path.join(data_dir, 'wrong_extension.aac')
        fn = partial(AudioSegment.from_file, aac_path, "aac")
        self.assertRaises(CouldntDecodeError, fn)

        # Trying to auto detect input file format
        aac_file = AudioSegment.from_file(
            os.path.join(data_dir, 'wrong_extension.aac'))
        self.assertEqual(int(aac_file.duration_seconds), 9)

    def test_export_ogg_as_mp3(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            AudioSegment.from_file(self.ogg_file_path).export(tmp_mp3_file,
                                                              format="mp3")

    @unittest.skipUnless(HAS_LIBVORBIS,
                         "Unsupported codecs")
    def test_export_mp3_as_ogg(self):
        with NamedTemporaryFile('w+b', suffix='.ogg') as tmp_ogg_file:
            AudioSegment.from_file(self.mp3_file_path).export(tmp_ogg_file,
                                                              format="ogg")

    def test_export_webm_as_mp3(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            AudioSegment.from_file(
                self.webm_file_path,
                codec="opus"
            ).export(tmp_mp3_file, format="mp3")

    def test_export_mp3_as_webm(self):
        with NamedTemporaryFile('w+b', suffix='.webm') as tmp_webm_file:
            AudioSegment.from_file(self.mp3_file_path).export(tmp_webm_file,
                                                              format="webm")

    @unittest.skipUnless('aac' in get_supported_decoders() and HAS_LIBVORBIS,
                         "Unsupported codecs")
    def test_export_mp4_as_ogg(self):
        with NamedTemporaryFile('w+b', suffix='.ogg') as tmp_ogg_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_ogg_file,
                                                              format="ogg")

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_mp3_file,
                                                              format="mp3")

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_wav(self):
        with NamedTemporaryFile('w+b', suffix='.wav') as tmp_wav_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_wav_file,
                                                              format="mp3")

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3_with_tags(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            tags_dict = {
                'title': "The Title You Want",
                'artist': "Artist's name",
                'album': "Name of the Album"
            }
            AudioSegment.from_file(self.mp4_file_path).export(tmp_mp3_file,
                                                              format="mp3",
                                                              tags=tags_dict)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3_with_tags_raises_exception_when_tags_are_not_a_dictionary(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            json = '{"title": "The Title You Want", "album": "Name of the Album", "artist": "Artist\'s name"}'
            func = partial(
                AudioSegment.from_file(self.mp4_file_path).export, tmp_mp3_file,
                format="mp3", tags=json)
            self.assertRaises(InvalidTag, func)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp4_as_mp3_with_tags_raises_exception_when_id3version_is_wrong(self):
        tags = {'artist': 'Artist', 'title': 'Title'}
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            func = partial(
                AudioSegment.from_file(self.mp4_file_path).export,
                tmp_mp3_file,
                format="mp3",
                tags=tags,
                id3v2_version='BAD VERSION'
            )
            self.assertRaises(InvalidID3TagVersion, func)

    @unittest.skipUnless('aac' in get_supported_decoders(),
                         "Unsupported codecs")
    def test_export_mp3_with_tags(self):
        tags = {'artist': 'Mozart', 'title': 'The Magic Flute'}

        delete = sys.platform != 'win32'

        with NamedTemporaryFile('w+b', suffix='.mp3', delete=delete) as tmp_mp3_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_mp3_file, format="mp3", tags=tags)

            if sys.platform == 'win32':
                tmp_mp3_file.close()

            info = mediainfo(filepath=tmp_mp3_file.name)
            info_tags = info["TAG"]

            self.assertEqual(info_tags["artist"], "Mozart")
            self.assertEqual(info_tags["title"], "The Magic Flute")

            if sys.platform == 'win32':
                os.remove(tmp_mp3_file.name)

    def test_mp3_with_jpg_cover_img(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            outf = self.seg1.export(tmp_mp3_file, format="mp3", cover=self.jpg_cover_path)
            testseg = AudioSegment.from_file(outf, format="mp3")

            # should be within a 150ms and 1.5dB (not perfectly equal due to codecs)
            self.assertWithinTolerance(len(self.seg1), len(testseg), 150)
            self.assertWithinTolerance(self.seg1.dBFS, testseg.dBFS, 1.5)

    def test_mp3_with_png_cover_img(self):
        with NamedTemporaryFile('w+b', suffix='.mp3') as tmp_mp3_file:
            outf = self.seg1.export(tmp_mp3_file, format="mp3", cover=self.png_cover_path)
            testseg = AudioSegment.from_file(outf, format="mp3")

            # should be within a 150ms and 1.5dB (not perfectly equal due to codecs)
            self.assertWithinTolerance(len(self.seg1), len(testseg), 150)
            self.assertWithinTolerance(self.seg1.dBFS, testseg.dBFS, 1.5)

    def test_fade_raises_exception_when_duration_start_end_are_none(self):
        seg = self.seg1
        func = partial(seg.fade, start=1, end=1, duration=1)
        self.assertRaises(TypeError, func)

    def test_silent(self):
        seg = AudioSegment.silent(len(self.seg1))
        self.assertEqual(len(self.seg1), len(seg))
        self.assertEqual(seg.rms, 0)
        self.assertEqual(seg.frame_width, 2)

        seg_8bit = seg.set_sample_width(1)
        self.assertEqual(seg_8bit.sample_width, 1)
        self.assertEqual(seg_8bit.frame_width, 1)
        self.assertEqual(seg_8bit.rms, 0)

        seg *= self.seg1
        self.assertEqual(seg.rms, self.seg1.rms)
        self.assertEqual(len(seg), len(self.seg1))
        self.assertEqual(seg.frame_width, self.seg1.frame_width)
        self.assertEqual(seg.frame_rate, self.seg1.frame_rate)

    def test_from_mono_audiosegments(self):
        monoseg1 = self.seg1.set_channels(1)
        monoseg2 = monoseg1.reverse()
        stereo_sound = AudioSegment.from_mono_audiosegments(monoseg1, monoseg2)

        self.assertEqual(stereo_sound.channels, 2)
        self.assertEqual(stereo_sound.dBFS, monoseg1.dBFS)
        self.assertEqual(len(stereo_sound), len(monoseg1))

    def test_from_mono_audiosegments_three_channels(self):
        left = Sine(440).to_audio_segment(duration=500)
        center = Sine(660).to_audio_segment(duration=500)
        right = Sine(880).to_audio_segment(duration=500)
        seg = AudioSegment.from_mono_audiosegments(left, center, right)

        self.assertEqual(seg.channels, 3)
        self.assertEqual(seg.frame_count(), left.frame_count())
        samples = seg.get_array_of_samples()
        self.assertEqual(samples[0::3], left.get_array_of_samples())
        self.assertEqual(samples[1::3], center.get_array_of_samples())
        self.assertEqual(samples[2::3], right.get_array_of_samples())

    def test_from_mono_audiosegments_rejects_different_lengths(self):
        long_seg = Sine(440).to_audio_segment(duration=1000)
        short_seg = Sine(440).to_audio_segment(duration=500)

        for segs in [(long_seg, short_seg), (short_seg, long_seg)]:
            with self.assertRaises(ValueError) as cm:
                AudioSegment.from_mono_audiosegments(*segs)
            self.assertIn("same length", str(cm.exception))
            self.assertIn("fill_up", str(cm.exception))

    def test_from_mono_audiosegments_rejects_non_mono(self):
        mono = Sine(440).to_audio_segment(duration=500)
        stereo = mono.set_channels(2)

        for segs in [(mono, stereo), (stereo, mono), (stereo,)]:
            with self.assertRaises(ValueError) as cm:
                AudioSegment.from_mono_audiosegments(*segs)
            self.assertIn("mono", str(cm.exception))

    def test_from_mono_audiosegments_requires_a_segment(self):
        self.assertRaises(ValueError, AudioSegment.from_mono_audiosegments)

    def test_from_mono_audiosegments_rejects_unknown_kwargs(self):
        mono = Sine(440).to_audio_segment(duration=500)
        self.assertRaises(TypeError, AudioSegment.from_mono_audiosegments, mono, mono, fill=True)

    def test_from_mono_audiosegments_fill_up(self):
        long_seg = Sine(440).to_audio_segment(duration=1000)
        short_seg = Sine(880).to_audio_segment(duration=400)
        n = int(short_seg.frame_count())

        seg = AudioSegment.from_mono_audiosegments(long_seg, short_seg, fill_up=True)
        self.assertEqual(seg.channels, 2)
        self.assertEqual(seg.frame_count(), long_seg.frame_count())
        left, right = seg.split_to_mono()
        self.assertEqual(left.raw_data, long_seg.raw_data)
        self.assertEqual(right[:400].raw_data, short_seg.raw_data)
        padding = right.get_array_of_samples()[n:]
        self.assertEqual(len(padding), int(long_seg.frame_count()) - n)
        self.assertTrue(all(sample == 0 for sample in padding))

        # padding goes on whichever channel is shorter
        seg = AudioSegment.from_mono_audiosegments(short_seg, long_seg, fill_up=True)
        left, right = seg.split_to_mono()
        self.assertEqual(right.raw_data, long_seg.raw_data)
        self.assertEqual(left[:400].raw_data, short_seg.raw_data)
        self.assertEqual(left[400:].rms, 0)

    def test_from_mono_audiosegments_fill_up_keeps_equal_lengths_unchanged(self):
        left = Sine(440).to_audio_segment(duration=500)
        right = Sine(880).to_audio_segment(duration=500)
        self.assertEqual(
            AudioSegment.from_mono_audiosegments(left, right, fill_up=True),
            AudioSegment.from_mono_audiosegments(left, right))

    def test_from_mono_audiosegments_syncs_frame_rate_and_sample_width(self):
        hi = Sine(440, sample_rate=44100, bit_depth=16).to_audio_segment(duration=500)
        lo = Sine(440, sample_rate=22050, bit_depth=8).to_audio_segment(duration=500)
        seg = AudioSegment.from_mono_audiosegments(hi, lo, fill_up=True)

        self.assertEqual(seg.channels, 2)
        self.assertEqual(seg.frame_rate, 44100)
        self.assertEqual(seg.sample_width, 2)
        self.assertEqual(seg.frame_count(), hi.frame_count())

    def test_fade_raises_exception_when_duration_is_negative(self):
        seg = self.seg1
        func = partial(seg.fade,
                       to_gain=1,
                       from_gain=1,
                       start=None,
                       end=None,
                       duration=-1)
        self.assertRaises(InvalidDuration, func)

    def test_make_chunks(self):
        seg = self.seg1
        chunks = make_chunks(seg, 100)
        seg2 = chunks[0]
        for chunk in chunks[1:]:
            seg2 += chunk
        self.assertEqual(len(seg), len(seg2))

    def test_empty(self):
        self.assertEqual(len(self.seg1), len(self.seg1 + AudioSegment.empty()))
        self.assertEqual(len(self.seg2), len(self.seg2 + AudioSegment.empty()))
        self.assertEqual(len(self.seg3), len(self.seg3 + AudioSegment.empty()))

    def test_speedup(self):
        speedup_seg = self.seg1.speedup(2.0)

        self.assertWithinTolerance(
            len(self.seg1) / 2, len(speedup_seg), percentage=0.02)

    def test_speedup_matches_chained_append(self):
        # speedup() used to build its output with repeated append() calls,
        # which is quadratic (#736). Make sure the faster implementation
        # produces exactly the same audio.
        def speedup_with_append(seg, playback_speed, chunk_size, crossfade):
            atk = 1.0 / playback_speed
            if playback_speed < 2.0:
                ms_to_remove_per_chunk = int(chunk_size * (1 - atk) / atk)
            else:
                ms_to_remove_per_chunk = int(chunk_size)
                chunk_size = int(atk * chunk_size / (1 - atk))
            crossfade = min(crossfade, ms_to_remove_per_chunk - 1)
            chunks = make_chunks(seg, chunk_size + ms_to_remove_per_chunk)
            ms_to_remove_per_chunk -= crossfade
            last_chunk = chunks[-1]
            chunks = [chunk[:-ms_to_remove_per_chunk] for chunk in chunks[:-1]]
            out = chunks[0]
            for chunk in chunks[1:]:
                out = out.append(chunk, crossfade=crossfade)
            return out + last_chunk

        segs = [
            self.seg1[:3000],
            self.seg1[:3000].set_channels(1).set_sample_width(1),
            WhiteNoise().to_audio_segment(duration=2017).set_frame_rate(11025),
        ]
        settings = [
            (1.5, 150, 25), (1.1, 150, 25), (2.0, 150, 25), (3.0, 150, 25),
            (1.5, 150, 0), (1.25, 50, 10), (1.5, 37, 13), (1.5, 150, 100),
        ]
        for seg in segs:
            for playback_speed, chunk_size, crossfade in settings:
                expected = speedup_with_append(seg, playback_speed, chunk_size, crossfade)
                actual = seg.speedup(playback_speed, chunk_size, crossfade)
                self.assertEqual(expected.raw_data, actual.raw_data)
                self.assertEqual(expected.frame_rate, actual.frame_rate)
                self.assertEqual(expected.sample_width, actual.sample_width)
                self.assertEqual(expected.channels, actual.channels)

    def test_dBFS(self):
        seg_8bit = self.seg1.set_sample_width(1)
        self.assertWithinTolerance(seg_8bit.dBFS, -18.06, tolerance=1.5)
        self.assertWithinTolerance(self.seg1.dBFS, -17.76, tolerance=1.5)
        self.assertWithinTolerance(self.seg2.dBFS, -20.78, tolerance=1.5)
        self.assertWithinTolerance(self.seg3.dBFS, -12.94, tolerance=1.5)

    def test_compress(self):
        compressed = self.seg1.compress_dynamic_range()
        self.assertWithinTolerance(self.seg1.dBFS - compressed.dBFS,
                                   10.0,
                                   tolerance=10.0)

        # Highest peak should be lower
        self.assertTrue(compressed.max < self.seg1.max)

        # average volume should be reduced
        self.assertTrue(compressed.rms < self.seg1.rms)

    @unittest.skipUnless('aac' in get_supported_decoders() and HAS_LIBVORBIS,
                         "Unsupported codecs")
    def test_exporting_to_ogg_uses_default_codec_when_codec_param_is_none(self):
        delete = sys.platform != 'win32'

        with NamedTemporaryFile('w+b', suffix='.ogg', delete=delete) as tmp_ogg_file:
            AudioSegment.from_file(self.mp4_file_path).export(tmp_ogg_file, format="ogg")

            if sys.platform == 'win32':
                tmp_ogg_file.close()

            info = mediainfo(filepath=tmp_ogg_file.name)

            if sys.platform == 'win32':
                os.remove(tmp_ogg_file.name)

        self.assertEqual(info["codec_name"], "vorbis")
        self.assertEqual(info["format_name"], "ogg")

    def test_zero_length_segment(self):
        self.assertEqual(0, len(self.seg1[0:0]))

    def test_invert(self):
        s_mono = Sine(100).to_audio_segment()
        s = s_mono.set_channels(2)

        try:
            s_mono.invert_phase(channels=(1, 0))
        except Exception:
            pass
        else:
            raise Exception("AudioSegment.invert_phase() didn't catch a bad input (mono)")

        s_inv = s.invert_phase()
        self.assertFalse(s == s_inv)
        self.assertTrue(s.rms == s_inv.rms)
        self.assertTrue(s == s_inv.invert_phase())

        s_inv_right = s.invert_phase(channels=(0, 1))
        left, right = s_inv_right.split_to_mono()

        self.assertFalse(s_mono == s_inv_right)
        self.assertFalse(s_inv == s_inv_right)
        self.assertTrue(left == s_mono)
        self.assertFalse(right == s_mono)

        s_inv_left = s.invert_phase(channels=(1, 0))
        left, right = s_inv_left.split_to_mono()

        self.assertFalse(s_mono == s_inv_left)
        self.assertFalse(s_inv == s_inv_left)
        self.assertFalse(left == s_mono)
        self.assertTrue(right == s_mono)

    def test_max_dBFS(self):
        sine_0_dbfs = Sine(1000).to_audio_segment()
        sine_minus_3_dbfs = Sine(1000).to_audio_segment(volume=-3.0)
        self.assertAlmostEqual(-0.0, sine_0_dbfs.max_dBFS, 2)
        self.assertAlmostEqual(-3.0, sine_minus_3_dbfs.max_dBFS, 2)

    def test_array_type(self):
        self.assertEqual(self.seg1.array_type, "h")
        self.assertEqual(self.seg2.array_type, "h")
        self.assertEqual(self.seg3.array_type, "h")
        self.assertEqual(self.mp3_seg_party.array_type, "h")

        silence = AudioSegment.silent(50)
        self.assertEqual(silence.array_type, "h")
        self.assertEqual(silence.set_sample_width(1).array_type, "b")
        self.assertEqual(silence.set_sample_width(4).array_type, "i")

    def test_sample_array(self):
        samples = Sine(450).to_audio_segment().get_array_of_samples()
        self.assertEqual(
            list(samples[:8]),
            [0, 2099, 4190, 6263, 8311, 10325, 12296, 14217]
        )

    def test_get_dc_offset(self):
        seg = self.seg_dc_offset
        self.assertWithinTolerance(seg.get_dc_offset(), -0.16, tolerance=0.01)
        self.assertWithinTolerance(seg.get_dc_offset(1), -0.16, tolerance=0.01)
        self.assertWithinTolerance(seg.get_dc_offset(2), 0.1, tolerance=0.01)

    def test_remove_dc_offset(self):
        seg = self.seg_dc_offset

        seg1 = seg.remove_dc_offset()
        self.assertWithinTolerance(seg1.get_dc_offset(1), 0.0, tolerance=0.0001)
        self.assertWithinTolerance(seg1.get_dc_offset(2), 0.0, tolerance=0.0001)

        seg1 = seg.remove_dc_offset(1)
        self.assertWithinTolerance(seg1.get_dc_offset(1), 0.0, tolerance=0.0001)
        self.assertWithinTolerance(seg1.get_dc_offset(2), 0.1, tolerance=0.01)

        seg1 = seg.remove_dc_offset(2)
        self.assertWithinTolerance(seg1.get_dc_offset(1), -0.16, tolerance=0.01)
        self.assertWithinTolerance(seg1.get_dc_offset(2), 0.0, tolerance=0.0001)

        seg1 = seg.remove_dc_offset(channel=1, offset=(-0.06))
        self.assertWithinTolerance(seg1.get_dc_offset(1), -0.1, tolerance=0.01)

    def test_from_file_clean_fail(self):
        delete = sys.platform != 'win32'
        orig_tmpdir = gettempdir()
        new_tmpdir = mkdtemp()
        tempfile.tempdir = new_tmpdir

        with NamedTemporaryFile('w+b', suffix='.wav', delete=delete) as tmp_wav_file:
            tmp_wav_file.write("not really a wav".encode('utf-8'))
            tmp_wav_file.flush()
            self.assertRaises(CouldntDecodeError, AudioSegment.from_file, tmp_wav_file.name)
            files = os.listdir(tempfile.tempdir)
            self.assertEqual(files, [os.path.basename(tmp_wav_file.name)])

        if sys.platform == 'win32':
            os.remove(tmp_wav_file.name)

        tempfile.tempdir = orig_tmpdir
        os.rmdir(new_tmpdir)

    def test_from_file_image_clean_fail(self):
        # an image has a video stream but no audio stream
        with self.assertRaises(CouldntDecodeError):
            AudioSegment.from_file(os.path.join(data_dir, 'cover.png'))


class SilenceTests(unittest.TestCase):

    def setUp(self):
        global test1wav, test4wav
        if not test1wav:
            test1wav = AudioSegment.from_wav(os.path.join(data_dir, 'test1.wav'))
        if not test4wav:
            test4wav = AudioSegment.from_wav(os.path.join(data_dir, 'test4.wav'))

        self.seg1 = test1wav
        self.seg4 = test4wav

    def test_split_on_silence_complete_silence(self):
        seg = AudioSegment.silent(5000)
        self.assertEqual( split_on_silence(seg), [] )

    def test_split_on_silence_test1(self):
        self.assertEqual(
            len(split_on_silence(self.seg1, min_silence_len=500, silence_thresh=-20)),
            3
        )
    def test_split_on_silence_no_silence(self):
        splits = split_on_silence(self.seg1, min_silence_len=5000, silence_thresh=-200, keep_silence=True)
        lens = [len(split) for split in splits]
        self.assertEqual( lens, [len(self.seg1)] )

    def test_detect_completely_silent_segment(self):
        seg = AudioSegment.silent(5000)
        silent_ranges = detect_silence(seg, min_silence_len=1000, silence_thresh=-20)
        self.assertEqual(silent_ranges, [[0, 5000]])

    def test_detect_tight_silent_segment(self):
        seg = AudioSegment.silent(1000)
        silent_ranges = detect_silence(seg, min_silence_len=1000, silence_thresh=-20)
        self.assertEqual(silent_ranges, [[0, 1000]])

    def test_detect_too_long_silence(self):
        seg = AudioSegment.silent(3000)
        silent_ranges = detect_silence(seg, min_silence_len=5000, silence_thresh=-20)
        self.assertEqual(silent_ranges, [])

    def test_detect_silence_seg1(self):
        silent_ranges = detect_silence(self.seg1, min_silence_len=500, silence_thresh=-20)
        self.assertEqual(silent_ranges, [[0, 775], [3141, 4033], [5516, 6051]])

    def test_detect_silence_seg1_with_seek_split(self):
        silent_ranges = detect_silence(self.seg1, min_silence_len=500, silence_thresh=-20,
                                       seek_step=10)
        self.assertEqual(silent_ranges, [[0, 770], [3150, 4030], [5520, 6050]])

    def test_realistic_audio(self):
        silent_ranges = detect_silence(self.seg4, min_silence_len=1000, silence_thresh=self.seg4.dBFS)

        prev_end = -1
        for start, end in silent_ranges:
            self.assertTrue(start > prev_end)
            prev_end = end


class GeneratorTests(unittest.TestCase):

    def test_with_smoke(self):
        Sine(440).to_audio_segment()
        Square(440).to_audio_segment()
        Triangle(440).to_audio_segment()

        Pulse(440, duty_cycle=0.75).to_audio_segment()
        Sawtooth(440, duty_cycle=0.75).to_audio_segment()

        WhiteNoise().to_audio_segment()

    def test_loudness(self):
        sine_dbfs = Sine(440).to_audio_segment().dBFS
        square_dbfs = Square(440).to_audio_segment().dBFS
        white_noise_dbfs = WhiteNoise().to_audio_segment().dBFS

        self.assertAlmostEqual(sine_dbfs, -3.0, places=1)
        self.assertAlmostEqual(square_dbfs, 0.0, places=1)
        self.assertAlmostEqual(white_noise_dbfs, -5, places=0)

    def test_duration(self):
        one_sec = Sine(440).to_audio_segment(duration=1000)
        five_sec = Sine(440).to_audio_segment(duration=5000)
        half_sec = Sine(440).to_audio_segment(duration=500)

        self.assertAlmostEqual(len(one_sec), 1000)
        self.assertAlmostEqual(len(five_sec), 5000)
        self.assertAlmostEqual(len(half_sec), 500)


class FromFileClosesFilesTests(unittest.TestCase):
    """from_file() must close files it opened itself, on every code path"""

    def setUp(self):
        import pydub.audio_segment as audio_segment_module
        self.module = audio_segment_module
        self.orig_fd_or_path = audio_segment_module._fd_or_path_or_tempfile
        self.opened = []

        def recording_fd_or_path(*args, **kwargs):
            fd, close_fd = self.orig_fd_or_path(*args, **kwargs)
            if close_fd:
                self.opened.append(fd)
            return fd, close_fd

        audio_segment_module._fd_or_path_or_tempfile = recording_fd_or_path

    def tearDown(self):
        self.module._fd_or_path_or_tempfile = self.orig_fd_or_path

    def assertAllClosed(self):
        self.assertTrue(len(self.opened) > 0)
        for f in self.opened:
            self.assertTrue(f.closed, "{0!r} was left open".format(f))

    def test_raw_file_is_closed(self):
        AudioSegment.from_file(os.path.join(data_dir, 'test1.raw'), format='raw',
                               sample_width=2, frame_rate=32000, channels=1)
        self.assertAllClosed()

    def test_raw_file_is_closed_with_start_second_and_duration(self):
        AudioSegment.from_file(os.path.join(data_dir, 'test1.raw'), format='raw',
                               sample_width=2, frame_rate=32000, channels=1,
                               start_second=0.5, duration=0.5)
        self.assertAllClosed()

    def test_file_is_closed_when_raw_parameters_are_missing(self):
        with self.assertRaises(KeyError):
            AudioSegment.from_file(os.path.join(data_dir, 'test1.raw'), format='raw')
        self.assertAllClosed()

    def test_wav_file_is_closed(self):
        AudioSegment.from_file(os.path.join(data_dir, 'test1.wav'))
        self.assertAllClosed()

    def test_mp3_file_is_closed(self):
        AudioSegment.from_file(os.path.join(data_dir, 'test1.mp3'), duration=0.5)
        self.assertAllClosed()

    def test_file_is_closed_when_decoding_fails(self):
        with NamedTemporaryFile('w+b', suffix='.mp3', delete=False) as tmp_file:
            tmp_file.write("not really an mp3".encode('utf-8'))
        try:
            with self.assertRaises(CouldntDecodeError):
                AudioSegment.from_file(tmp_file.name)
            self.assertAllClosed()
        finally:
            os.unlink(tmp_file.name)



class CompressedWavTests(unittest.TestCase):
    """
    The stored bits_per_sample of compressed codecs in WAV (mu-law, A-law,
    ADPCM) is not the decoded width, so from_file() must not use it (#417).
    """

    def setUp(self):
        self.tmp_dir = mkdtemp()
        self.seg = Sine(440).to_audio_segment(duration=1000).apply_gain(-6)
        self.seg = self.seg.set_frame_rate(8000)

    def tearDown(self):
        for name in os.listdir(self.tmp_dir):
            os.remove(os.path.join(self.tmp_dir, name))
        os.rmdir(self.tmp_dir)

    def encode(self, codec):
        path = os.path.join(self.tmp_dir, codec + '.wav')
        self.seg.export(path, format='wav', codec=codec).close()
        return path

    def assertDecodes(self, codec, sample_width):
        path = self.encode(codec)
        self.assertEqual(mediainfo_json(path)['streams'][0]['codec_name'], codec)
        seg = AudioSegment.from_file(path)
        self.assertEqual(seg.sample_width, sample_width)
        self.assertEqual(seg.frame_rate, 8000)
        self.assertEqual(seg.channels, 1)
        # ADPCM pads the last block, so allow a few tens of ms
        self.assertTrue(1000 <= len(seg) <= 1050, len(seg))
        self.assertAlmostEqual(seg.dBFS, self.seg.dBFS, places=0)

    def test_mulaw_decodes_to_16_bit(self):
        self.assertDecodes('pcm_mulaw', 2)

    def test_alaw_decodes_to_16_bit(self):
        self.assertDecodes('pcm_alaw', 2)

    def test_adpcm_ima_wav(self):
        self.assertDecodes('adpcm_ima_wav', 2)

    def test_adpcm_ms(self):
        self.assertDecodes('adpcm_ms', 2)

    def test_plain_pcm_widths_unchanged(self):
        self.assertDecodes('pcm_u8', 1)
        self.assertDecodes('pcm_s16le', 2)
        # 24-bit WAV is decoded as 32-bit, as before
        self.assertDecodes('pcm_s24le', 4)
        self.assertDecodes('pcm_s32le', 4)

class LargeExportTests(unittest.TestCase):
    """
    More than 4 GiB of pcm can't be put in a wav header, so export() must
    hand ffmpeg raw pcm instead of a temporary wav (#602). The limit is
    lowered here so the tests don't need gigabytes of audio.
    """

    def setUp(self):
        self.orig_limit = audio_segment.WAV_MAX_DATA_BYTES
        audio_segment.WAV_MAX_DATA_BYTES = 4096
        self.tmp_dir = mkdtemp()
        self.seg = Sine(440).to_audio_segment(duration=2000).apply_gain(-6)
        self.seg = AudioSegment.from_mono_audiosegments(self.seg, self.seg.invert_phase())

    def tearDown(self):
        audio_segment.WAV_MAX_DATA_BYTES = self.orig_limit
        for name in os.listdir(self.tmp_dir):
            os.remove(os.path.join(self.tmp_dir, name))
        os.rmdir(self.tmp_dir)

    def test_export_mp3_above_wav_limit(self):
        self.assertTrue(len(self.seg.raw_data) > audio_segment.WAV_MAX_DATA_BYTES)
        exported = AudioSegment.from_mp3(self.seg.export(format='mp3'))
        self.assertEqual(exported.channels, 2)
        self.assertEqual(exported.frame_rate, self.seg.frame_rate)
        self.assertTrue(abs(len(exported) - len(self.seg)) < 100,
                        (len(exported), len(self.seg)))
        self.assertAlmostEqual(exported.dBFS, self.seg.dBFS, places=0)

    def test_raw_input_is_sample_identical(self):
        for sample_width, codec in [(1, 'pcm_u8'), (2, 'pcm_s16le'), (4, 'pcm_s32le')]:
            seg = self.seg.set_sample_width(sample_width)
            exported = AudioSegment.from_wav(seg.export(format='wav', codec=codec))
            self.assertEqual(exported.sample_width, sample_width)
            self.assertEqual(exported.channels, 2)
            self.assertEqual(exported.raw_data, seg.raw_data)

    def test_easy_wav_above_limit_raises(self):
        path = os.path.join(self.tmp_dir, 'big.wav')
        with self.assertRaises(CouldntEncodeError) as cm:
            self.seg.export(path, format='wav')
        self.assertIn('-rf64', str(cm.exception))
        self.assertFalse(os.path.exists(path))

    def test_rf64_wav_above_limit(self):
        path = os.path.join(self.tmp_dir, 'big.wav')
        self.seg.export(path, format='wav', parameters=['-rf64', 'auto']).close()
        exported = AudioSegment.from_file(path)
        self.assertEqual(exported.raw_data, self.seg.raw_data)

    def test_input_format_switches_at_limit(self):
        commands = []
        orig_log = audio_segment.log_conversion
        audio_segment.log_conversion = commands.append
        try:
            seg = self.seg[:10]
            audio_segment.WAV_MAX_DATA_BYTES = len(seg.raw_data)
            seg.export(format='mp3').close()
            audio_segment.WAV_MAX_DATA_BYTES = len(seg.raw_data) - 1
            seg.export(format='mp3').close()
        finally:
            audio_segment.log_conversion = orig_log

        # at the limit the command is unchanged
        self.assertEqual(commands[0][2:5], ['-f', 'wav', '-i'])
        self.assertEqual(commands[1][2:9],
                         ['-f', 's16le', '-ar', '44100', '-ac', '2', '-i'])


class NoConverterTests(unittest.TestCase):

    def setUp(self):
        self.wave_file = os.path.join(data_dir, 'test1.wav')
        self.wave24_file = os.path.join(data_dir, 'test1-24bit.wav')
        self.wave_empty = os.path.join(data_dir, 'test1_empty.wav')
        self.mp3_file = os.path.join(data_dir, 'test1.mp3')
        self.raw_file = os.path.join(data_dir, 'test1.raw')
        AudioSegment.converter = "definitely-not-a-path-to-anything-asdjklqwop"

    def tearDown(self):
        AudioSegment.converter = get_encoder_name()

    def test_opening_wav_file(self):
        seg = AudioSegment.from_wav(self.wave_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave_file, "wav")
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave_file, format="wav")
        self.assertTrue(len(seg) > 1000)

    def test_opening_wav24_file(self):
        seg = AudioSegment.from_wav(self.wave24_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave24_file)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave24_file, "wav")
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.wave24_file, format="wav")
        self.assertTrue(len(seg) > 1000)

    def test_opening_raw_file(self):
        seg = AudioSegment.from_raw(self.raw_file, sample_width=2, frame_rate=32000, channels=2)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.raw_file, "raw", sample_width=2, frame_rate=32000, channels=2)
        self.assertTrue(len(seg) > 1000)

        seg = AudioSegment.from_file(self.raw_file, format="raw", sample_width=2, frame_rate=32000, channels=2)
        self.assertTrue(len(seg) > 1000)

    def test_opening_raw_file_with_missing_args_fails(self):
        func = partial(AudioSegment.from_raw, self.raw_file)
        self.assertRaises(KeyError, func)

    def test_opening_mp3_file_fails(self):
        func = partial(AudioSegment.from_mp3, self.mp3_file)
        self.assertRaises(OSError, func)

        func = partial(AudioSegment.from_file, self.mp3_file)
        self.assertRaises(OSError, func)

        func = partial(AudioSegment.from_file, self.mp3_file, "mp3")
        self.assertRaises(OSError, func)

        func = partial(AudioSegment.from_file, self.mp3_file, format="mp3")
        self.assertRaises(OSError, func)

    def test_init_AudioSegment_data_buffer(self):
        seg = AudioSegment(data="\0" * 34, sample_width=2, frame_rate=4, channels=1)

        self.assertEqual(seg.duration_seconds, 4.25)

        self.assertEqual(seg.sample_width, 2)

        self.assertEqual(seg.frame_rate, 4)

    def test_init_AudioSegment_data_buffer_with_missing_args_fails(self):
        func = partial(AudioSegment, data="\0" * 16, sample_width=2, frame_rate=2)
        self.assertRaises(MissingAudioParameter, func)

        func = partial(AudioSegment, data="\0" * 16, sample_width=2, channels=1)
        self.assertRaises(MissingAudioParameter, func)

        func = partial(AudioSegment, data="\0" * 16, frame_rate=2, channels=1)
        self.assertRaises(MissingAudioParameter, func)

    def test_init_AudioSegment_data_buffer_with_bad_values_fails(self):
        func = partial(AudioSegment, data="\0" * 14, sample_width=4, frame_rate=2, channels=1)
        self.assertRaises(ValueError, func)

    def test_exporting(self):
        seg = AudioSegment.from_wav(self.wave_file)
        out_f = seg.export(format="wav")
        exported = AudioSegment.from_wav(out_f)
        out_f.close()

        self.assertEqual(len(exported), len(seg))

    def test_opening_empty_wav_file(self):
        seg = AudioSegment.from_wav(self.wave_empty)
        self.assertTrue(len(seg) == 0)

        seg = AudioSegment.from_file(self.wave_empty)
        self.assertTrue(len(seg) == 0)

        seg = AudioSegment.from_file(self.wave_empty, "wav")
        self.assertTrue(len(seg) == 0)

        seg = AudioSegment.from_file(self.wave_empty, format="wav")
        self.assertTrue(len(seg) == 0)


class MissingProberTests(unittest.TestCase):

    def setUp(self):
        self.mp3_file = os.path.join(data_dir, 'test1.mp3')
        self.orig_path = os.environ['PATH']
        self.orig_converter = AudioSegment.converter
        self.ffmpeg_path = which('ffmpeg')
        self.ffprobe_path = which('ffprobe')
        os.environ['PATH'] = os.path.join(data_dir, 'definitely-not-a-dir-asdjklqwop')

    def tearDown(self):
        os.environ['PATH'] = self.orig_path
        AudioSegment.converter = self.orig_converter

    def test_missing_prober_raises_helpful_error(self):
        AudioSegment.converter = "definitely-not-a-path-to-anything-asdjklqwop"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for func in (mediainfo_json, mediainfo):
                with self.assertRaises(OSError) as cm:
                    func(self.mp3_file)
                self.assertIn("ffprobe", str(cm.exception))
                self.assertIn("AudioSegment.converter", str(cm.exception))

            with open(self.mp3_file, 'rb') as f:
                with self.assertRaises(OSError) as cm:
                    mediainfo_json(f)
                self.assertIn("AudioSegment.converter", str(cm.exception))

    @unittest.skipUnless(which('ffmpeg') and which('ffprobe'), "requires ffmpeg and ffprobe")
    def test_prober_found_next_to_converter(self):
        tmpdir = tempfile.mkdtemp()
        try:
            for name, target in (('ffmpeg', self.ffmpeg_path), ('ffprobe', self.ffprobe_path)):
                os.symlink(os.path.abspath(target), os.path.join(tmpdir, name))
            AudioSegment.converter = os.path.join(tmpdir, 'ffmpeg')

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                info = mediainfo_json(self.mp3_file)
                self.assertEqual(info['streams'][0]['codec_name'], 'mp3')
                with open(self.mp3_file, 'rb') as f:
                    info = mediainfo_json(f)
                self.assertEqual(info['streams'][0]['codec_name'], 'mp3')
                seg = AudioSegment.from_file(self.mp3_file)
            self.assertTrue(len(seg) > 1000)
        finally:
            for name in ('ffmpeg', 'ffprobe'):
                if os.path.lexists(os.path.join(tmpdir, name)):
                    os.remove(os.path.join(tmpdir, name))
            os.rmdir(tmpdir)


class MissingConverterTests(unittest.TestCase):

    def setUp(self):
        self.mp3_file = os.path.join(data_dir, 'test1.mp3')
        self.orig_converter = AudioSegment.converter
        self.missing = os.path.join(data_dir, 'definitely-not-a-dir-asdjklqwop', 'ffmpeg')
        AudioSegment.converter = self.missing

    def tearDown(self):
        AudioSegment.converter = self.orig_converter

    def assertHelpfulError(self, cm):
        # same type as before, so existing except clauses keep working
        self.assertIsInstance(cm.exception, OSError)
        self.assertEqual(cm.exception.errno, errno.ENOENT)
        self.assertEqual(cm.exception.filename, self.missing)
        self.assertIsInstance(cm.exception.__cause__, OSError)
        message = str(cm.exception)
        self.assertIn(self.missing, message)
        self.assertIn("ffmpeg", message)
        self.assertIn("AudioSegment.converter", message)

    def test_from_file_missing_converter(self):
        with self.assertRaises(OSError) as cm:
            AudioSegment.from_file(self.mp3_file, format="mp3")
        self.assertHelpfulError(cm)

    def test_from_file_using_temporary_files_missing_converter(self):
        with self.assertRaises(OSError) as cm:
            AudioSegment.from_file_using_temporary_files(self.mp3_file, format="mp3")
        self.assertHelpfulError(cm)

    def test_export_missing_converter(self):
        seg = AudioSegment.silent(duration=100)
        with self.assertRaises(OSError) as cm:
            seg.export(BytesIO(), format="mp3")
        self.assertHelpfulError(cm)

    def test_wav_does_not_need_converter(self):
        seg = AudioSegment.silent(duration=100)
        out = seg.export(BytesIO(), format="wav")
        self.assertEqual(len(AudioSegment.from_file(out, format="wav")), 100)


class PlaybackTests(unittest.TestCase):

    def test_ffplay_is_quiet(self):
        from pydub import playback
        calls = []
        orig_call = playback.subprocess.call
        playback.subprocess.call = lambda cmd, *a, **kw: calls.append(cmd)
        try:
            playback._play_with_ffplay(AudioSegment.silent(duration=10))
        finally:
            playback.subprocess.call = orig_call

        self.assertEqual(len(calls), 1)
        cmd = calls[0]
        self.assertIn("-loglevel", cmd)
        self.assertEqual(cmd[cmd.index("-loglevel") + 1], "error")
        # the file to play stays the last argument
        self.assertTrue(cmd[-1].endswith(".wav"))

    def _run_ffplay_tracking_temp_files(self, fake_call):
        # Windows can't reopen a NamedTemporaryFile by name while it is still
        # open, so record every temp file and check it's closed before export()
        from pydub import playback
        opened = []
        orig_ntf = playback.NamedTemporaryFile
        orig_call = playback.subprocess.call

        def tracking_ntf(*args, **kwargs):
            f = orig_ntf(*args, **kwargs)
            opened.append(f)
            return f

        seg = AudioSegment.silent(duration=10)
        orig_export = seg.export

        def checking_export(out_f, *args, **kwargs):
            self.assertTrue(all(f.closed for f in opened))
            return orig_export(out_f, *args, **kwargs)

        seg.export = checking_export
        playback.NamedTemporaryFile = tracking_ntf
        playback.subprocess.call = fake_call
        try:
            playback._play_with_ffplay(seg)
        finally:
            playback.NamedTemporaryFile = orig_ntf
            playback.subprocess.call = orig_call
        return opened

    def test_ffplay_temp_file_closed_before_export_and_removed(self):
        seen = []

        def fake_call(cmd, *a, **kw):
            path = cmd[-1]
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as f:
                self.assertEqual(f.read(4), b"RIFF")
            seen.append(path)

        opened = self._run_ffplay_tracking_temp_files(fake_call)
        self.assertEqual(len(opened), 1)
        self.assertEqual(seen, [opened[0].name])
        self.assertFalse(os.path.exists(seen[0]))

    def test_ffplay_temp_file_removed_when_player_fails(self):
        seen = []

        def fake_call(cmd, *a, **kw):
            seen.append(cmd[-1])
            raise KeyboardInterrupt()

        self.assertRaises(KeyboardInterrupt,
                          self._run_ffplay_tracking_temp_files, fake_call)
        self.assertEqual(len(seen), 1)
        self.assertFalse(os.path.exists(seen[0]))


class FilterTests(unittest.TestCase):

    def setUp(self):
        global test1wav
        if not test1wav:
            test1wav = AudioSegment.from_wav(os.path.join(data_dir, 'test1.wav'))

        self.seg1 = test1wav

    def test_highpass_works_on_multichannel_segments(self):
        self.assertEqual(self.seg1.channels, 2)
        less_bass = self.seg1.high_pass_filter(800)
        self.assertTrue(less_bass.dBFS < self.seg1.dBFS)

    def test_highpass_filter_reduces_loudness(self):
        s = Square(200).to_audio_segment()
        less_bass = s.high_pass_filter(400)
        self.assertTrue(less_bass.dBFS < s.dBFS)

    def test_highpass_filter_cutoff_frequency(self):
        # A Sine wave should not be affected by a HPF 3 octaves lower
        s = Sine(800).to_audio_segment()
        less_bass = s.high_pass_filter(100)
        self.assertAlmostEqual(less_bass.dBFS, s.dBFS, places=0)

    def test_lowpass_filter_reduces_loudness(self):
        s = Square(200).to_audio_segment()
        less_treble = s.low_pass_filter(400)
        self.assertTrue(less_treble.dBFS < s.dBFS)

    def test_lowpass_filter_cutoff_frequency(self):
        # A Sine wave should not be affected by a LPF 3 octaves Higher
        s = Sine(100).to_audio_segment()
        less_treble = s.low_pass_filter(800)
        self.assertAlmostEqual(less_treble.dBFS, s.dBFS, places=0)


class PartialAudioSegmentLoadTests(unittest.TestCase):

    def setUp(self):
        self.wave_path_str = os.path.join(data_dir, 'test1.wav')
        self.mp3_path_str = os.path.join(data_dir, 'test1.mp3')
        self.raw_path_str = os.path.join(data_dir, 'test1.raw')

    def tearDown(self):
        AudioSegment.converter = get_encoder_name()

    def test_partial_load_duration_equals_cropped_mp3_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.mp3_path_str)[:1000]
        partial_seg2 = AudioSegment.from_file(self.mp3_path_str, duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_equals_cropped_mp3_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.mp3_path_str)[1000:]
        partial_seg2 = AudioSegment.from_file(self.mp3_path_str, start_second=1.)[0:]
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_and_duration_equals_cropped_mp3_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.mp3_path_str)[1000:2000]
        partial_seg2 = AudioSegment.from_file(self.mp3_path_str, start_second=1., duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_small_start_second_and_duration_equals_cropped_mp3_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.mp3_path_str)[10:1010]
        partial_seg2 = AudioSegment.from_file(self.mp3_path_str, start_second=0.01, duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_duration_equals_cropped_wav_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.wave_path_str)[:1000]
        partial_seg2 = AudioSegment.from_file(self.wave_path_str, duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_equals_cropped_wav_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.wave_path_str)[1000:]
        partial_seg2 = AudioSegment.from_file(self.wave_path_str, start_second=1.)[0:]
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_and_duration_equals_cropped_wav_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.wave_path_str)[1000:2000]
        partial_seg2 = AudioSegment.from_file(self.wave_path_str, start_second=1., duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_duration_equals_cropped_raw_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2)[:1000]
        partial_seg2 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2, duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_equals_cropped_raw_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2)[1000:]
        partial_seg2 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2, start_second=1.)[0:]
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def test_partial_load_start_second_and_duration_equals_cropped_raw_audio_segment(self):
        partial_seg1 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2)[1000:2000]
        partial_seg2 = AudioSegment.from_file(self.raw_path_str, format="raw", sample_width=2, frame_rate=32000, channels=2, start_second=1., duration=1.)
        self.assertEqual(len(partial_seg1), len(partial_seg2))
        self.assertEqual(partial_seg1._data, partial_seg2._data)

    def _from_file_logging_command(self, *args, **kwargs):
        # returns the segment and the ffmpeg command from_file() ran
        import logging
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger = logging.getLogger("pydub.converter")
        old_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            seg = AudioSegment.from_file(*args, **kwargs)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
        commands = [r.args[0] for r in records if r.msg.startswith("subprocess.call")]
        return seg, eval(commands[-1])

    def _from_file_output_seek_only(self, *args, **kwargs):
        import pydub.audio_segment
        orig = pydub.audio_segment._input_seek_is_exact
        pydub.audio_segment._input_seek_is_exact = lambda info: False
        try:
            return AudioSegment.from_file(*args, **kwargs)
        finally:
            pydub.audio_segment._input_seek_is_exact = orig

    def test_partial_load_seeks_input_and_matches_output_seeking(self):
        # #729: seek on the input side too so ffmpeg does not decode everything
        # before start_second, with identical audio. (A wav given by path is
        # read without ffmpeg, so wav is checked through a file object.)
        sources = [(self.mp3_path_str, False), (self.mp3_path_str, True),
                   (self.wave_path_str, True)]
        for path, as_file in sources:
            for start_second, duration in [(5.25, None), (6, 1.5), (4.14401, 1.), (3.0001, None)]:
                with open(path, "rb") as f:
                    expected = self._from_file_output_seek_only(
                        f if as_file else path, start_second=start_second, duration=duration)
                with open(path, "rb") as f:
                    seg, command = self._from_file_logging_command(
                        f if as_file else path, start_second=start_second, duration=duration)
                self.assertTrue(len(seg) > 0)
                self.assertEqual(expected._data, seg._data)
                j = command.index("-ss")
                self.assertTrue(j < command.index("-i"))
                self.assertEqual(command[j + 1], str(int(start_second - 2)))
                self.assertNotIn("e-", " ".join(command))

    def test_partial_load_near_start_seeks_output_only(self):
        seg, command = self._from_file_logging_command(self.mp3_path_str, start_second=1.5)
        self.assertEqual(command.count("-ss"), 1)
        self.assertTrue(command.index("-ss") > command.index("-i"))
        self.assertEqual(AudioSegment.from_file(self.mp3_path_str)[1500:]._data, seg._data)

    def test_partial_load_unchecked_format_seeks_output_only(self):
        # aac (and opus, ac3, mp2, flac in ogg) decode slightly differently
        # after an input seek, so they keep output-only seeking
        path = os.path.join(data_dir, 'format_test.m4a')
        seg, command = self._from_file_logging_command(path, start_second=5.5)
        self.assertEqual(command.count("-ss"), 1)
        self.assertTrue(command.index("-ss") > command.index("-i"))
        self.assertEqual(self._from_file_output_seek_only(path, start_second=5.5)._data, seg._data)


class PyAudioopTests(unittest.TestCase):
    """
    pydub.pyaudioop is the fallback used when the stdlib audioop module is
    missing (it was removed in Python 3.13), so it must work on Python 3 and
    give the same results as audioop.
    """

    def setUp(self):
        from pydub import pyaudioop
        self.pyaudioop = pyaudioop
        self.samples = [1000, -2000, 30000, -32768, 7, -7]
        self.data = struct.pack("6h", *self.samples)

    def unpack(self, data, size=2):
        fmt = {1: "b", 2: "h", 4: "i"}[size]
        return list(struct.unpack("%d%s" % (len(data) // size, fmt), data))

    def test_known_values(self):
        op = self.pyaudioop
        d = self.data
        self.assertEqual(op.rms(d, 2), 18160)
        self.assertEqual(op.max(d, 2), 32768)
        self.assertEqual(op.avg(d, 2), -628)
        self.assertEqual(self.unpack(op.mul(d, 2, 1.5)),
                         [1500, -3000, 32767, -32768, 10, -11])
        self.assertEqual(self.unpack(op.bias(d, 2, 1000)),
                         [2000, -1000, 31000, -31768, 1007, 993])
        self.assertEqual(self.unpack(op.add(d, d, 2)),
                         [2000, -4000, 32767, -32768, 14, -14])
        self.assertEqual(self.unpack(op.reverse(d, 2)), self.samples[::-1])
        self.assertEqual(self.unpack(op.tomono(d, 2, 0.5, 0.5)),
                         [-500, -1384, 0])
        self.assertEqual(self.unpack(op.tostereo(d, 2, 1, 0.5))[:4],
                         [1000, 500, -2000, -1000])
        self.assertEqual(self.unpack(op.lin2lin(d, 2, 1), 1),
                         [3, -8, 117, -128, 0, -1])
        self.assertEqual(self.unpack(op.lin2lin(d, 2, 4), 4),
                         [s << 16 for s in self.samples])
        converted, state = op.ratecv(d, 2, 2, 3, 2, None)
        self.assertEqual(self.unpack(converted), [1000, -2000, 15003, -16388])
        self.assertEqual(state, (-2, ((1966080000, 458752),
                                      (-2147483648, -458752))))

    def test_ulaw_alaw_known_values(self):
        op = self.pyaudioop
        d = self.data
        self.assertEqual(list(bytearray(op.lin2ulaw(d, 2))),
                         [206, 63, 130, 0, 254, 126])
        self.assertEqual(list(bytearray(op.lin2alaw(d, 2))),
                         [250, 106, 168, 42, 213, 85])
        codes = b"\x00\x7f\x80\xff\x55\xd5"
        self.assertEqual(self.unpack(op.ulaw2lin(codes, 2)),
                         [-32124, 0, 32124, 0, -716, 716])
        self.assertEqual(self.unpack(op.alaw2lin(codes, 2)),
                         [-5504, -848, 5504, 848, -8, 8])
        self.assertEqual(self.unpack(op.ulaw2lin(codes, 1), 1),
                         [-126, 0, 125, 0, -3, 2])
        self.assertEqual(self.unpack(op.alaw2lin(codes, 4), 4),
                         [-5504 << 16, -848 << 16, 5504 << 16, 848 << 16,
                          -8 << 16, 8 << 16])
        # one byte per sample, whatever the input width
        self.assertEqual(len(op.lin2ulaw(op.lin2lin(d, 2, 4), 4)), 6)
        self.assertEqual(len(op.lin2alaw(op.lin2lin(d, 2, 1), 1)), 6)
        self.assertRaises(op.error, op.lin2ulaw, b"\x00", 2)
        self.assertRaises(op.error, op.alaw2lin, b"\x00", 3)

    def test_audio_segment_operations_without_audioop(self):
        from pydub import audio_segment, effects, utils
        modules = (audio_segment, effects, utils)
        originals = [m.audioop for m in modules]
        try:
            for m in modules:
                m.audioop = self.pyaudioop
            seg = Sine(440).to_audio_segment(duration=50).set_channels(2)
            seg = seg.set_frame_rate(22050).set_sample_width(4)
            seg = seg.overlay(seg - 6).fade_in(10).fade_out(10)
            seg = seg.set_channels(1).reverse()
            self.assertEqual(seg.frame_rate, 22050)
            self.assertEqual(seg.sample_width, 4)
            self.assertTrue(seg.rms > 0)
            processed = seg._data
        finally:
            for m, original in zip(modules, originals):
                m.audioop = original

        try:
            import audioop  # noqa: F401
        except ImportError:
            return
        seg = Sine(440).to_audio_segment(duration=50).set_channels(2)
        seg = seg.set_frame_rate(22050).set_sample_width(4)
        seg = seg.overlay(seg - 6).fade_in(10).fade_out(10)
        seg = seg.set_channels(1).reverse()
        self.assertEqual(processed, seg._data)

    def test_matches_stdlib_audioop(self):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import audioop
        except ImportError:
            raise unittest.SkipTest("stdlib audioop is not available")
        import random
        rnd = random.Random(0)
        op = self.pyaudioop
        for size in (1, 2, 4):
            for n in (0, 2, 10, 101):
                d = bytes(bytearray(rnd.getrandbits(8) for _ in range(2 * n * size)))
                d2 = bytes(bytearray(rnd.getrandbits(8) for _ in range(2 * n * size)))
                calls = [
                    ("max", d, size), ("rms", d, size), ("avg", d, size),
                    ("minmax", d, size), ("cross", d, size),
                    ("reverse", d, size), ("add", d, d2, size),
                    ("bias", d, size, 1000), ("bias", d, size, -1),
                ]
                for factor in (0, 0.5, 2, 3.7, -0.3):
                    calls.append(("mul", d, size, factor))
                for fac1, fac2 in ((1, 1), (0.5, 0.5), (0.7, 0.3), (-1, 1)):
                    calls.append(("tomono", d, size, fac1, fac2))
                    calls.append(("tostereo", d, size, fac1, fac2))
                for size2 in (1, 2, 4):
                    calls.append(("lin2lin", d, size, size2))
                for name in ("lin2ulaw", "lin2alaw"):
                    calls.append((name, d, size))
                for name in ("ulaw2lin", "alaw2lin"):
                    calls.append((name, d, size))
                for channels in (1, 2):
                    for rates in ((44100, 22050), (8000, 44100), (44100, 48000)):
                        calls.append(("ratecv", d, size, channels) + rates + (None,))
                        calls.append(("ratecv", d, size, channels) + rates + (None, 2, 1))
                for call in calls:
                    expected = getattr(audioop, call[0])(*call[1:])
                    actual = getattr(op, call[0])(*call[1:])
                    if call[0] == "tomono" and expected != actual:
                        # C compilers may fuse l*fac1 + r*fac2 into one FMA (e.g. clang on arm64),
                        # which can land one step lower after floor(); allow 1 LSB there.
                        fmt = "<%d%s" % (len(expected) // size, {1: "b", 2: "h", 4: "i"}[size])
                        diffs = [abs(a - b) for a, b in zip(struct.unpack(fmt, expected), struct.unpack(fmt, actual))]
                        self.assertLessEqual(max(diffs), 1, (call[0], size, n) + call[3:])
                        continue
                    self.assertEqual(expected, actual, (call[0], size, n) + call[3:])

        # every u-law/A-law code, and every 8 and 16 bit sample
        codes = bytes(bytearray(range(256)))
        for size in (1, 2, 4):
            for name in ("ulaw2lin", "alaw2lin"):
                self.assertEqual(getattr(audioop, name)(codes, size),
                                 getattr(op, name)(codes, size), (name, size))
        for size, fmt in ((1, "b"), (2, "h")):
            bits = 8 * size
            d = struct.pack("%d%s" % (2 ** bits, fmt), *range(-2 ** (bits - 1), 2 ** (bits - 1)))
            for name in ("lin2ulaw", "lin2alaw"):
                self.assertEqual(getattr(audioop, name)(d, size),
                                 getattr(op, name)(d, size), (name, size))

        # resampling in chunks, passing the state along
        d = bytes(bytearray(rnd.getrandbits(8) for _ in range(800)))
        state_a = state_p = None
        out_a = out_p = b""
        for i in range(0, len(d), 80):
            chunk_a, state_a = audioop.ratecv(d[i:i + 80], 2, 2, 44100, 32000, state_a)
            chunk_p, state_p = op.ratecv(d[i:i + 80], 2, 2, 44100, 32000, state_p)
            out_a += chunk_a
            out_p += chunk_p
        self.assertEqual((out_a, state_a), (out_p, state_p))

    def test_utils_imports_fallback_without_audioop(self):
        import subprocess
        code = (
            "import sys; sys.modules['audioop'] = None; "
            "from pydub.utils import audioop; "
            "assert audioop.__name__ == 'pydub.pyaudioop', audioop.__name__"
        )
        here = os.path.dirname(os.path.abspath(__file__))
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=here,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = proc.communicate()
        self.assertEqual(proc.returncode, 0, err.decode("utf-8", "replace"))


if __name__ == "__main__":
    import sys

    if sys.version_info >= (3, 1):
        unittest.main(warnings="ignore")
    else:
        unittest.main()
