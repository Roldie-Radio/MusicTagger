from pathlib import Path

import pytest

from musictag.util import (
    guess_from_filename, normalize, safe_int, sanitize_component, similarity,
    split_pair, year_from_date,
)


class TestNormalize:
    def test_strips_accents_and_case(self):
        assert normalize("Björk") == "bjork"

    def test_expands_ampersand(self):
        assert normalize("Simon & Garfunkel") == "simon and garfunkel"

    def test_drops_marketing_noise(self):
        assert normalize("Roads (Official Video)") == "roads"
        assert normalize("Wonderwall (Remastered 2014)") == "wonderwall"

    def test_optional_feat_removal(self):
        assert normalize("Numb (feat. Jay-Z)", drop_feat=True) == "numb"
        assert "jay" in normalize("Numb (feat. Jay-Z)")

    def test_optional_article_removal(self):
        assert normalize("The Beatles", drop_articles=True) == "beatles"


class TestSimilarity:
    def test_identical(self):
        assert similarity("Glory Box", "Glory Box") == 1.0

    def test_case_and_punctuation_insensitive(self):
        assert similarity("Don't Look Back", "dont look back") > 0.95

    def test_unrelated_strings_score_low(self):
        assert similarity("Glory Box", "Enter Sandman") < 0.5

    def test_empty_is_zero(self):
        assert similarity("", "anything") == 0.0
        assert similarity(None, "anything") == 0.0


class TestSanitize:
    @pytest.mark.parametrize("raw,expected", [
        ("AC/DC", "AC-DC"),
        ("Where Are You?", "Where Are You"),
        ('He said "hi"', "He said 'hi'"),
        ("trailing dots...", "trailing dots"),
        ("  spaced  out  ", "spaced out"),
    ])
    def test_replaces_illegal_characters(self, raw, expected):
        assert sanitize_component(raw) == expected

    def test_windows_reserved_names_are_escaped(self):
        assert sanitize_component("CON") == "_CON"
        assert sanitize_component("aux") == "_aux"

    def test_empty_becomes_placeholder(self):
        assert sanitize_component("") == "Unknown"
        assert sanitize_component("???") == "Unknown"

    def test_length_is_capped(self):
        assert len(sanitize_component("x" * 500)) <= 120


class TestSafeInt:
    @pytest.mark.parametrize("raw,expected", [
        ("3/12", 3), ("03", 3), (7, 7), ((5, 12), 5), ("no digits", None), (None, None),
    ])
    def test_parses(self, raw, expected):
        assert safe_int(raw) == expected


class TestSplitPair:
    def test_slash_form(self):
        assert split_pair("3/12") == (3, 12)

    def test_tuple_form(self):
        assert split_pair((3, 12)) == (3, 12)

    def test_bare_number(self):
        assert split_pair("3") == (3, None)


class TestYearFromDate:
    @pytest.mark.parametrize("raw,expected", [
        ("1997", 1997), ("1997-06-16", 1997), ("16/06/1997", 1997), ("", None), (None, None),
    ])
    def test_extracts_year(self, raw, expected):
        assert year_from_date(raw) == expected


class TestGuessFromFilename:
    def test_track_artist_title(self):
        guess = guess_from_filename(Path("/m/Portishead/Dummy/01 - Portishead - Mysterons.mp3"))
        assert guess["track_no"] == 1
        assert guess["artist"] == "Portishead"
        assert guess["title"] == "Mysterons"

    def test_track_and_title(self):
        guess = guess_from_filename(Path("/m/Portishead/Dummy (1994)/10 - Glory Box.flac"))
        assert guess["track_no"] == 10
        assert guess["title"] == "Glory Box"
        assert guess["album"] == "Dummy"
        assert guess["year"] == 1994

    def test_disc_and_track_prefix(self):
        guess = guess_from_filename(Path("/m/Artist/Album/2-05 - Song.mp3"))
        assert guess["disc_no"] == 2
        assert guess["track_no"] == 5
        assert guess["title"] == "Song"

    def test_year_prefixed_album_folder(self):
        guess = guess_from_filename(Path("/m/Radiohead/1997 - OK Computer/03 - Subterranean.mp3"))
        assert guess["album"] == "OK Computer"
        assert guess["year"] == 1997

    def test_cd_subfolder_steps_up_for_album(self):
        guess = guess_from_filename(Path("/m/Artist/The Album/CD2/03 - Song.mp3"))
        assert guess["album"] == "The Album"

    def test_bare_filename_is_the_title(self):
        guess = guess_from_filename(Path("/m/whatever/mystery.mp3"))
        assert guess["title"] == "mystery"

    def test_parent_folder_becomes_album_artist(self):
        guess = guess_from_filename(Path("/m/Massive Attack/Mezzanine/01 - Angel.mp3"))
        assert guess["album_artist"] == "Massive Attack"
