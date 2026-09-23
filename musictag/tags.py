"""Format-agnostic tag reading and writing on top of mutagen.

Every supported container gets an adapter that maps our :class:`TrackTags`
onto the field names that container actually uses.  The mappings deliberately
follow MusicBrainz Picard's conventions, because that is what Plex (and
basically every other player) expects to find.

The fields that matter most for Plex:

* ``album_artist`` - Plex groups albums by this. Without it, a compilation
  explodes into one "album" per track artist.
* ``album`` + ``track_no`` + ``disc_no`` - ordering within the album.
* ``date``/``year`` - disambiguates re-releases.
* embedded cover art - Plex prefers it over folder images.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Collection, Optional

import mutagen
from mutagen.aiff import AIFF
from mutagen.asf import ASF, ASFUnicodeAttribute
from mutagen.flac import FLAC, Picture
from mutagen.id3 import (
    APIC, ID3, ID3NoHeaderError, TALB, TCMP, TCOM, TCON, TDRC, TIT2, TPE1,
    TPE2, TPOS, TRCK, TSRC, TXXX, UFID,
)
from mutagen.mp3 import MP3, BitrateMode
from mutagen.mp4 import MP4, MP4Cover
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

from .models import AudioProps, TrackTags
from .util import safe_int, split_pair, year_from_date

SUPPORTED_EXTENSIONS = {
    ".mp3", ".flac", ".m4a", ".mp4", ".m4b", ".ogg", ".oga", ".opus",
    ".wma", ".wav", ".aiff", ".aif",
}

LOSSLESS_EXTENSIONS = {".flac", ".wav", ".aiff", ".aif"}

MB_UFID_OWNER = "http://musicbrainz.org"


class UnsupportedFormat(Exception):
    pass


# ===========================================================================
# Adapters
# ===========================================================================

class BaseAdapter:
    """Interface every format adapter implements."""

    supports_art = True

    def __init__(self, path: Path):
        self.path = path
        self.file = None

    # -- reading -------------------------------------------------------
    def read(self) -> TrackTags:
        raise NotImplementedError

    def props(self) -> AudioProps:
        raise NotImplementedError

    # -- writing -------------------------------------------------------
    def write(self, tags: TrackTags, *, art: Optional[bytes] = None,
              art_mime: str = "image/jpeg", id3v2_version: int = 4,
              write_mb_ids: bool = True, clear: Collection[str] = ()) -> None:
        """Write ``tags`` to the file.

        An empty field means "no opinion": whatever the file has is left
        alone, so applying never erases a tag the proposal does not cover.

        ``clear`` names the :class:`TrackTags` fields for which an empty value
        should instead *remove* the tag (``"has_art"`` covers embedded art).
        Undo passes the fields Apply wrote, so a tag Apply added to a blank
        field is taken away again rather than surviving the undo. Only those
        fields: a tag this app could not parse reads as empty too, and must
        not be deleted just because it looked blank.
        """
        raise NotImplementedError


class ID3Adapter(BaseAdapter):
    """MP3, WAV and AIFF - all of which carry ID3v2."""

    def __init__(self, path: Path):
        super().__init__(path)
        ext = path.suffix.lower()
        if ext == ".mp3":
            self.file = MP3(path)
        elif ext == ".wav":
            self.file = WAVE(path)
        else:
            self.file = AIFF(path)
        if self.file.tags is None:
            try:
                self.file.add_tags()
            except Exception:
                pass
        self.id3: ID3 = self.file.tags

    @staticmethod
    def _text(frame) -> Optional[str]:
        if frame is None:
            return None
        try:
            value = frame.text[0]
        except (AttributeError, IndexError):
            return None
        return str(value).strip() or None

    def _txxx(self, desc: str) -> Optional[str]:
        frame = self.id3.get(f"TXXX:{desc}") if self.id3 else None
        return self._text(frame)

    def read(self) -> TrackTags:
        t = TrackTags()
        if not self.id3:
            return t
        g = self.id3.get
        t.title = self._text(g("TIT2"))
        t.artist = self._text(g("TPE1"))
        t.album = self._text(g("TALB"))
        t.album_artist = self._text(g("TPE2"))
        t.track_no, t.track_total = split_pair(self._text(g("TRCK")))
        t.disc_no, t.disc_total = split_pair(self._text(g("TPOS")))
        t.date = self._text(g("TDRC")) or self._text(g("TYER"))
        t.year = year_from_date(t.date)
        t.genre = self._text(g("TCON"))
        t.composer = self._text(g("TCOM"))
        t.isrc = self._text(g("TSRC"))
        t.compilation = (self._text(g("TCMP")) or "0") not in ("0", "", None)

        t.mb_release_id = self._txxx("MusicBrainz Album Id")
        t.mb_release_group_id = self._txxx("MusicBrainz Release Group Id")
        t.mb_artist_id = self._txxx("MusicBrainz Artist Id")
        t.mb_album_artist_id = self._txxx("MusicBrainz Album Artist Id")
        ufid = self.id3.get(f"UFID:{MB_UFID_OWNER}")
        if ufid is not None and getattr(ufid, "data", None):
            try:
                t.mb_recording_id = ufid.data.decode("ascii", "ignore") or None
            except Exception:
                t.mb_recording_id = None
        t.has_art = any(k.startswith("APIC") for k in self.id3.keys())
        return t

    def props(self) -> AudioProps:
        info = self.file.info
        ext = self.path.suffix.lower()
        mode = ""
        raw_mode = getattr(info, "bitrate_mode", None)
        if raw_mode is not None and raw_mode != BitrateMode.UNKNOWN:
            mode = {BitrateMode.CBR: "cbr", BitrateMode.VBR: "vbr", BitrateMode.ABR: "abr"}.get(raw_mode, "")
        return AudioProps(
            container=ext.lstrip("."),
            codec="mp3" if ext == ".mp3" else "pcm",
            duration_s=float(getattr(info, "length", 0.0) or 0.0),
            bitrate_kbps=int(round((getattr(info, "bitrate", 0) or 0) / 1000)),
            bitrate_mode=mode,
            sample_rate=int(getattr(info, "sample_rate", 0) or 0),
            bit_depth=int(getattr(info, "bits_per_sample", 0) or 0),
            channels=int(getattr(info, "channels", 0) or 0),
            filesize=self.path.stat().st_size,
            lossless=ext in LOSSLESS_EXTENSIONS,
        )

    def write(self, tags: TrackTags, *, art=None, art_mime="image/jpeg",
              id3v2_version: int = 4, write_mb_ids: bool = True,
              clear: Collection[str] = ()) -> None:
        id3 = self.id3

        def put(frame_cls, key: str, field: str):
            value = getattr(tags, field)
            if value in (None, ""):
                if field in clear:
                    id3.delall(key)
                return
            id3.setall(key, [frame_cls(encoding=3, text=[str(value)])])

        put(TIT2, "TIT2", "title")
        put(TPE1, "TPE1", "artist")
        put(TALB, "TALB", "album")
        put(TPE2, "TPE2", "album_artist")
        put(TCON, "TCON", "genre")
        put(TCOM, "TCOM", "composer")
        put(TSRC, "TSRC", "isrc")
        put(TDRC, "TDRC", "date")
        if tags.track_no:
            value = f"{tags.track_no}/{tags.track_total}" if tags.track_total else str(tags.track_no)
            id3.setall("TRCK", [TRCK(encoding=3, text=[value])])
        elif "track_no" in clear:
            id3.delall("TRCK")
        if tags.disc_no:
            value = f"{tags.disc_no}/{tags.disc_total}" if tags.disc_total else str(tags.disc_no)
            id3.setall("TPOS", [TPOS(encoding=3, text=[value])])
        elif "disc_no" in clear:
            id3.delall("TPOS")
        # Only a real compilation gets the flag. A file that never had one is
        # left without it, rather than gaining a "0" nobody asked for; one
        # that wrongly says "1" is corrected.
        if tags.compilation:
            id3.setall("TCMP", [TCMP(encoding=3, text=["1"])])
        elif "compilation" in clear:
            id3.delall("TCMP")
        elif id3.getall("TCMP"):
            id3.setall("TCMP", [TCMP(encoding=3, text=["0"])])

        if write_mb_ids:
            def txxx(desc: str, field: str):
                value = getattr(tags, field)
                if not value:
                    if field in clear:
                        id3.delall(f"TXXX:{desc}")
                    return
                id3.delall(f"TXXX:{desc}")
                id3.add(TXXX(encoding=3, desc=desc, text=[value]))

            txxx("MusicBrainz Album Id", "mb_release_id")
            txxx("MusicBrainz Release Group Id", "mb_release_group_id")
            txxx("MusicBrainz Artist Id", "mb_artist_id")
            txxx("MusicBrainz Album Artist Id", "mb_album_artist_id")
            if tags.mb_recording_id:
                id3.delall(f"UFID:{MB_UFID_OWNER}")
                id3.add(UFID(owner=MB_UFID_OWNER, data=tags.mb_recording_id.encode("ascii")))
            elif "mb_recording_id" in clear:
                id3.delall(f"UFID:{MB_UFID_OWNER}")

        if art:
            id3.delall("APIC")
            id3.add(APIC(encoding=3, mime=art_mime, type=3, desc="Front cover", data=art))
        elif "has_art" in clear and not tags.has_art:
            id3.delall("APIC")

        if id3v2_version == 3:
            id3.update_to_v23()
            self.file.save(v2_version=3)
        else:
            self.file.save(v2_version=4)


class VorbisAdapter(BaseAdapter):
    """FLAC, Ogg Vorbis and Opus - all use Vorbis comments."""

    def __init__(self, path: Path):
        super().__init__(path)
        ext = path.suffix.lower()
        if ext == ".flac":
            self.file = FLAC(path)
        elif ext == ".opus":
            self.file = OggOpus(path)
        else:
            self.file = OggVorbis(path)
        if self.file.tags is None:
            self.file.add_tags()

    def _get(self, *keys: str) -> Optional[str]:
        for key in keys:
            values = self.file.tags.get(key) or self.file.tags.get(key.lower())
            if values:
                value = str(values[0]).strip()
                if value:
                    return value
        return None

    def read(self) -> TrackTags:
        t = TrackTags()
        t.title = self._get("TITLE")
        t.artist = self._get("ARTIST")
        t.album = self._get("ALBUM")
        t.album_artist = self._get("ALBUMARTIST", "ALBUM ARTIST", "ENSEMBLE")
        t.track_no = safe_int(self._get("TRACKNUMBER"))
        t.track_total = safe_int(self._get("TRACKTOTAL", "TOTALTRACKS"))
        if t.track_total is None:
            _, total = split_pair(self._get("TRACKNUMBER"))
            t.track_total = total
        t.disc_no = safe_int(self._get("DISCNUMBER"))
        t.disc_total = safe_int(self._get("DISCTOTAL", "TOTALDISCS"))
        t.date = self._get("DATE", "YEAR", "ORIGINALDATE")
        t.year = year_from_date(t.date)
        t.genre = self._get("GENRE")
        t.composer = self._get("COMPOSER")
        t.isrc = self._get("ISRC")
        t.compilation = (self._get("COMPILATION") or "0") not in ("0", "", None)
        t.mb_recording_id = self._get("MUSICBRAINZ_TRACKID")
        t.mb_release_id = self._get("MUSICBRAINZ_ALBUMID")
        t.mb_release_group_id = self._get("MUSICBRAINZ_RELEASEGROUPID")
        t.mb_artist_id = self._get("MUSICBRAINZ_ARTISTID")
        t.mb_album_artist_id = self._get("MUSICBRAINZ_ALBUMARTISTID")

        if isinstance(self.file, FLAC):
            t.has_art = bool(self.file.pictures)
        else:
            t.has_art = bool(self.file.tags.get("metadata_block_picture"))
        return t

    def props(self) -> AudioProps:
        info = self.file.info
        ext = self.path.suffix.lower()
        codec = {"flac": "flac", "opus": "opus", "ogg": "vorbis", "oga": "vorbis"}.get(ext.lstrip("."), "")
        return AudioProps(
            container=ext.lstrip("."),
            codec=codec,
            duration_s=float(getattr(info, "length", 0.0) or 0.0),
            bitrate_kbps=int(round((getattr(info, "bitrate", 0) or 0) / 1000)),
            bitrate_mode="vbr" if ext != ".flac" else "",
            sample_rate=int(getattr(info, "sample_rate", 0) or 48000),
            bit_depth=int(getattr(info, "bits_per_sample", 0) or 0),
            channels=int(getattr(info, "channels", 0) or 0),
            filesize=self.path.stat().st_size,
            lossless=ext == ".flac",
        )

    def write(self, tags: TrackTags, *, art=None, art_mime="image/jpeg",
              id3v2_version: int = 4, write_mb_ids: bool = True,
              clear: Collection[str] = ()) -> None:
        tag = self.file.tags

        def drop(key: str) -> None:
            if key in tag:
                del tag[key]

        def put(key: str, field: str):
            value = getattr(tags, field)
            if value in (None, ""):
                if field in clear:
                    drop(key)
                return
            tag[key] = [str(value)]

        put("TITLE", "title")
        put("ARTIST", "artist")
        put("ALBUM", "album")
        put("ALBUMARTIST", "album_artist")
        put("TRACKNUMBER", "track_no")
        put("TRACKTOTAL", "track_total")
        put("TOTALTRACKS", "track_total")
        put("DISCNUMBER", "disc_no")
        put("DISCTOTAL", "disc_total")
        put("TOTALDISCS", "disc_total")
        put("DATE", "date")
        put("GENRE", "genre")
        put("COMPOSER", "composer")
        put("ISRC", "isrc")
        # Same rule as ID3: flag real compilations, correct a stale "1",
        # otherwise leave the tag absent.
        if tags.compilation:
            tag["COMPILATION"] = ["1"]
        elif "compilation" in clear:
            drop("COMPILATION")
        elif "COMPILATION" in tag:
            tag["COMPILATION"] = ["0"]

        if write_mb_ids:
            put("MUSICBRAINZ_TRACKID", "mb_recording_id")
            put("MUSICBRAINZ_ALBUMID", "mb_release_id")
            put("MUSICBRAINZ_RELEASEGROUPID", "mb_release_group_id")
            put("MUSICBRAINZ_ARTISTID", "mb_artist_id")
            put("MUSICBRAINZ_ALBUMARTISTID", "mb_album_artist_id")

        if art:
            pic = Picture()
            pic.data = art
            pic.type = 3
            pic.mime = art_mime
            pic.desc = "Front cover"
            if isinstance(self.file, FLAC):
                self.file.clear_pictures()
                self.file.add_picture(pic)
            else:
                tag["metadata_block_picture"] = [
                    base64.b64encode(pic.write()).decode("ascii")
                ]
        elif "has_art" in clear and not tags.has_art:
            if isinstance(self.file, FLAC):
                self.file.clear_pictures()
            drop("metadata_block_picture")

        self.file.save()


class MP4Adapter(BaseAdapter):
    """M4A / MP4 (AAC and ALAC)."""

    KEYS = {
        "title": "\xa9nam",
        "artist": "\xa9ART",
        "album": "\xa9alb",
        "album_artist": "aART",
        "date": "\xa9day",
        "genre": "\xa9gen",
        "composer": "\xa9wrt",
    }
    FREEFORM = {
        "isrc": "----:com.apple.iTunes:ISRC",
        "mb_recording_id": "----:com.apple.iTunes:MusicBrainz Track Id",
        "mb_release_id": "----:com.apple.iTunes:MusicBrainz Album Id",
        "mb_release_group_id": "----:com.apple.iTunes:MusicBrainz Release Group Id",
        "mb_artist_id": "----:com.apple.iTunes:MusicBrainz Artist Id",
        "mb_album_artist_id": "----:com.apple.iTunes:MusicBrainz Album Artist Id",
    }

    def __init__(self, path: Path):
        super().__init__(path)
        self.file = MP4(path)
        if self.file.tags is None:
            self.file.add_tags()

    def _get(self, key: str) -> Optional[str]:
        values = self.file.tags.get(key)
        if not values:
            return None
        value = values[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", "ignore")
        return str(value).strip() or None

    def _pair(self, key: str) -> tuple[Optional[int], Optional[int]]:
        """``trkn``/``disk`` hold a list of (number, total) tuples."""
        values = self.file.tags.get(key)
        if not values:
            return None, None
        first = values[0]
        if isinstance(first, (tuple, list)):
            number = safe_int(first[0]) if len(first) > 0 else None
            total = safe_int(first[1]) if len(first) > 1 else None
            # MP4 uses 0 to mean "not set", which is not the same as track 0.
            return (number or None), (total or None)
        return safe_int(first), None

    def read(self) -> TrackTags:
        t = TrackTags()
        for attr, key in self.KEYS.items():
            setattr(t, attr, self._get(key))
        for attr, key in self.FREEFORM.items():
            setattr(t, attr, self._get(key))
        t.track_no, t.track_total = self._pair("trkn")
        t.disc_no, t.disc_total = self._pair("disk")
        t.year = year_from_date(t.date)
        # mutagen hands back a bare bool for cpil, not a list.
        cpil = self.file.tags.get("cpil")
        t.compilation = bool(cpil[0]) if isinstance(cpil, (list, tuple)) and cpil else bool(cpil)
        t.has_art = bool(self.file.tags.get("covr"))
        return t

    def props(self) -> AudioProps:
        info = self.file.info
        codec = getattr(info, "codec", "") or ""
        lossless = "alac" in codec.lower()
        return AudioProps(
            container=self.path.suffix.lower().lstrip("."),
            codec="alac" if lossless else "aac",
            duration_s=float(getattr(info, "length", 0.0) or 0.0),
            bitrate_kbps=int(round((getattr(info, "bitrate", 0) or 0) / 1000)),
            bitrate_mode="",
            sample_rate=int(getattr(info, "sample_rate", 0) or 0),
            bit_depth=int(getattr(info, "bits_per_sample", 0) or 0),
            channels=int(getattr(info, "channels", 0) or 0),
            filesize=self.path.stat().st_size,
            lossless=lossless,
        )

    def write(self, tags: TrackTags, *, art=None, art_mime="image/jpeg",
              id3v2_version: int = 4, write_mb_ids: bool = True,
              clear: Collection[str] = ()) -> None:
        tag = self.file.tags

        def drop(key: str, field: str) -> None:
            if field in clear and key in tag:
                del tag[key]

        for attr, key in self.KEYS.items():
            value = getattr(tags, attr)
            if value not in (None, ""):
                tag[key] = [str(value)]
            else:
                drop(key, attr)
        if tags.track_no:
            tag["trkn"] = [(int(tags.track_no), int(tags.track_total or 0))]
        else:
            drop("trkn", "track_no")
        if tags.disc_no:
            tag["disk"] = [(int(tags.disc_no), int(tags.disc_total or 0))]
        else:
            drop("disk", "disc_no")
        # Same rule as ID3: flag real compilations, correct a stale "1",
        # otherwise leave the atom absent.
        if tags.compilation:
            tag["cpil"] = True
        elif "compilation" in clear:
            drop("cpil", "compilation")
        elif "cpil" in tag:
            tag["cpil"] = False

        for attr, key in self.FREEFORM.items():
            if attr.startswith("mb_") and not write_mb_ids:
                continue
            value = getattr(tags, attr)
            if value not in (None, ""):
                tag[key] = [str(value).encode("utf-8")]
            else:
                drop(key, attr)

        if art:
            fmt = MP4Cover.FORMAT_PNG if art_mime == "image/png" else MP4Cover.FORMAT_JPEG
            tag["covr"] = [MP4Cover(art, imageformat=fmt)]
        elif not tags.has_art:
            drop("covr", "has_art")

        self.file.save()


class ASFAdapter(BaseAdapter):
    """Windows Media Audio. Read/write text; cover art is not written."""

    supports_art = False

    KEYS = {
        "title": "Title",
        "artist": "Author",
        "album": "WM/AlbumTitle",
        "album_artist": "WM/AlbumArtist",
        "genre": "WM/Genre",
        "composer": "WM/Composer",
        "isrc": "WM/ISRC",
        "date": "WM/Year",
        "mb_recording_id": "MusicBrainz/Track Id",
        "mb_release_id": "MusicBrainz/Album Id",
        "mb_release_group_id": "MusicBrainz/Release Group Id",
        "mb_artist_id": "MusicBrainz/Artist Id",
        "mb_album_artist_id": "MusicBrainz/Album Artist Id",
    }

    def __init__(self, path: Path):
        super().__init__(path)
        self.file = ASF(path)

    def _get(self, key: str) -> Optional[str]:
        values = self.file.tags.get(key)
        if not values:
            return None
        return str(values[0]).strip() or None

    def read(self) -> TrackTags:
        t = TrackTags()
        for attr, key in self.KEYS.items():
            setattr(t, attr, self._get(key))
        t.track_no = safe_int(self._get("WM/TrackNumber"))
        t.disc_no = safe_int(self._get("WM/PartOfSet"))
        t.year = year_from_date(t.date)
        t.has_art = bool(self.file.tags.get("WM/Picture"))
        return t

    def props(self) -> AudioProps:
        info = self.file.info
        return AudioProps(
            container="wma",
            codec="wma",
            duration_s=float(getattr(info, "length", 0.0) or 0.0),
            bitrate_kbps=int(round((getattr(info, "bitrate", 0) or 0) / 1000)),
            sample_rate=int(getattr(info, "sample_rate", 0) or 0),
            channels=int(getattr(info, "channels", 0) or 0),
            filesize=self.path.stat().st_size,
            lossless=False,
        )

    def write(self, tags: TrackTags, *, art=None, art_mime="image/jpeg",
              id3v2_version: int = 4, write_mb_ids: bool = True,
              clear: Collection[str] = ()) -> None:
        for attr, key in self.KEYS.items():
            if attr.startswith("mb_") and not write_mb_ids:
                continue
            value = getattr(tags, attr)
            if value not in (None, ""):
                self.file.tags[key] = [ASFUnicodeAttribute(str(value))]
            elif attr in clear and key in self.file.tags:
                del self.file.tags[key]
        for key, attr in (("WM/TrackNumber", "track_no"), ("WM/PartOfSet", "disc_no")):
            value = getattr(tags, attr)
            if value:
                self.file.tags[key] = [ASFUnicodeAttribute(str(value))]
            elif attr in clear and key in self.file.tags:
                del self.file.tags[key]
        self.file.save()


# ===========================================================================
# Public API
# ===========================================================================

def get_adapter(path: Path) -> BaseAdapter:
    ext = path.suffix.lower()
    if ext in (".mp3", ".wav", ".aiff", ".aif"):
        return ID3Adapter(path)
    if ext in (".flac", ".ogg", ".oga", ".opus"):
        return VorbisAdapter(path)
    if ext in (".m4a", ".mp4", ".m4b"):
        return MP4Adapter(path)
    if ext == ".wma":
        return ASFAdapter(path)
    raise UnsupportedFormat(f"No tag adapter for {ext or path.name}")


def read_file(path: Path) -> tuple[TrackTags, AudioProps]:
    """Read tags + technical properties in one open."""
    adapter = get_adapter(path)
    return adapter.read(), adapter.props()


def write_file(path: Path, tags: TrackTags, *, art: Optional[bytes] = None,
               art_mime: str = "image/jpeg", id3v2_version: int = 4,
               write_mb_ids: bool = True, clear: Collection[str] = ()) -> None:
    """Write tags; see :meth:`BaseAdapter.write` for what ``clear`` does."""
    adapter = get_adapter(path)
    if art and not adapter.supports_art:
        art = None
    adapter.write(tags, art=art, art_mime=art_mime,
                  id3v2_version=id3v2_version, write_mb_ids=write_mb_ids,
                  clear=clear)


def read_embedded_art(path: Path) -> Optional[tuple[bytes, str]]:
    """Return ``(data, mime)`` of the front cover, if the file has one."""
    try:
        audio = mutagen.File(path)
    except Exception:
        return None
    if audio is None:
        return None
    if isinstance(audio, FLAC) and audio.pictures:
        pic = audio.pictures[0]
        return pic.data, pic.mime
    tags = audio.tags
    if tags is None:
        return None
    try:
        if isinstance(tags, ID3):
            frames = tags.getall("APIC")
            if frames:
                return frames[0].data, frames[0].mime
        if isinstance(audio, MP4):
            covers = tags.get("covr")
            if covers:
                mime = "image/png" if covers[0].imageformat == MP4Cover.FORMAT_PNG else "image/jpeg"
                return bytes(covers[0]), mime
        blocks = tags.get("metadata_block_picture") if hasattr(tags, "get") else None
        if blocks:
            pic = Picture(base64.b64decode(blocks[0]))
            return pic.data, pic.mime
    except (ID3NoHeaderError, Exception):
        return None
    return None
