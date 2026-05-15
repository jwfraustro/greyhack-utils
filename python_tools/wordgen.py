#!/usr/bin/env python3
"""
Faithful Python port of MarkovNameGenerator + WordGenerator from Grey Hack.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .appleseed import DotNetRandom


class Word(enum.IntEnum):
    """Mirrors WordGenerator.word — order matters, values are used as indices."""
    apellido = 0
    usernames = 1
    corp_names = 2
    passwords = 3
    libVar = 4
    exploitNames = 5


# Filename mapping matches the C# nameFiles array, by enum index.
WORDLIST_FILES = {
    Word.apellido: "Surnamesmod",
    Word.usernames: "usernames",
    Word.corp_names: "CorpNames",
    Word.passwords: "Passwords",
    Word.libVar: "lib_variables",
    Word.exploitNames: "exploitNames",
}


class MarkovNameGenerator:
    """Faithful port of MarkovNameGenerator.cs."""

    def __init__(self, sample_lines: Sequence[str], order: int, min_length: int):
        if order < 1:
            order = 1
        if min_length < 1:
            min_length = 1
        self._order = order
        self._min_length = min_length

        # _samples: List<string>, iteration order preserved, duplicates allowed.
        self._samples: List[str] = []
        for line in sample_lines:
            # C# splits on ',', trims, uppercases, keeps tokens with len >= order+1
            for token in line.split(","):
                tok = token.strip().upper()
                if len(tok) >= order + 1:
                    self._samples.append(tok)

        # _chains: Dictionary<string, List<char>>. Key insertion order matters
        # only because Python preserves dict insertion order — but since we
        # only do lookups by key, not iteration, it actually doesn't matter
        # here. Values (chain lists) must match C# exactly.
        self._chains: Dict[str, List[str]] = {}
        for sample in self._samples:
            for j in range(len(sample) - order):
                key = sample[j : j + order]
                nxt = sample[j + order]
                if key in self._chains:
                    self._chains[key].append(nxt)
                else:
                    self._chains[key] = [nxt]

    def _is_vocal(self, c: str) -> bool:
        return c in "aeiouAEIOU"

    def _get_letter(self, token: str, rng: DotNetRandom) -> str:
        """Returns '?' without consuming RNG if key is missing; else one draw."""
        if token not in self._chains:
            return "?"
        chain = self._chains[token]
        idx = rng.next(max_value=len(chain))
        return chain[idx]

    def next_name(self, rng: DotNetRandom) -> str:
        """Generate a name using the supplied DotNetRandom; mutates RNG state."""
        while True:  # outer `do { } while (text.Length < this._minLength)`
            n = len(self._samples)
            num = rng.next(max_value=n)
            length = len(self._samples[num])

            # Substring(Next(0, sample.Length - _order), _order)
            # Note: in C# Random.Next(min, max), max is exclusive.
            sample = self._samples[num]
            start = rng.next(min_value=0, max_value=len(sample) - self._order)
            text = sample[start : start + self._order]

            while len(text) < length:
                token = text[len(text) - self._order :]
                # !!! Game code calls GetLetter TWICE per iteration. Both
                # calls consume RNG if the key exists. We reproduce this.
                first = self._get_letter(token, rng)
                if first == "?":
                    break
                second = self._get_letter(token, rng)
                if second == "?":
                    # Edge case: first call succeeded (drew a value), second
                    # call found the key missing somehow. Can only happen if
                    # the chains dict were mutated mid-call, which it isn't,
                    # so this branch is unreachable in practice. Still, mirror
                    # the C# control flow: append the second result, then
                    # the next while-check will break.
                    text += second
                    break
                text += second

            # Post-processing: capitalization rules
            if " " in text:
                parts = text.split(" ")
                rebuilt = ""
                for i, part in enumerate(parts):
                    if part == "":
                        continue
                    if len(part) == 1:
                        part = part.upper()
                    else:
                        part = part[0] + part[1:].lower()
                    if rebuilt != "":
                        rebuilt += " "
                    rebuilt += part
                text = rebuilt
            else:
                text = text[0] + text[1:].lower()

            if len(text) >= self._min_length:
                break

        # Final initial-letter fixup
        c0 = text[0]
        c1 = text[1] if len(text) > 1 else ""
        c01 = c0 + c1
        if c0.lower() == c1 or (
            c1 != "h"
            and c1 != "r"
            and c1 != "l"
            and c1 != "'"
            and c01 != "Ch"
            and not self._is_vocal(c0)
            and c01 != "Mc"
            and not self._is_vocal(c1)
            and c0 != "S"
        ):
            text = c1.upper() + text[2:]

        return text


class WordGenerator:
    """Mirrors the C# WordGenerator static class."""

    def __init__(self, wordlist_dir: str, ext: str = ".txt"):
        """Construct from a directory containing the wordlist files.

        File names must match WORDLIST_FILES values (with the supplied extension).
        Pass ext='' if the game's resources have no extension.
        """
        base = Path(wordlist_dir)
        self.markov: Dict[Word, MarkovNameGenerator] = {}
        for word_type, filename in WORDLIST_FILES.items():
            path = base / (filename + ext)
            if not path.exists():
                raise FileNotFoundError(f"Missing wordlist {path}")
            # C# splits on '\n' (NOT '\r\n'), so we read raw and split on \n.
            # Lines may have trailing '\r' which C# would keep — match that.
            raw = path.read_text(encoding="utf-8")
            lines = raw.split("\n")
            self.markov[word_type] = MarkovNameGenerator(lines, order=3, min_length=3)

    def get_next_word(self, word_type: Word, rng: DotNetRandom) -> str:
        """Mirrors WordGenerator.GetNextWord exactly."""
        text = self.markov[word_type].next_name(rng).strip("'-")
        if word_type == Word.passwords:
            # Extra draw to decide lowercase
            if rng.next(min_value=0, max_value=2) == 1:
                text = text.lower()
        return text


# ---------------------------------------------------------------------------
# Convenience: load once, query many.
# ---------------------------------------------------------------------------

_default_generator: Optional[WordGenerator] = None


def configure(wordlist_dir: str, ext: str = ".txt") -> WordGenerator:
    """Initialize a module-level WordGenerator. Returns it for direct use too."""
    global _default_generator
    _default_generator = WordGenerator(wordlist_dir, ext=ext)
    return _default_generator


def get_next_word(word_type: Word, rng: DotNetRandom) -> str:
    """Module-level convenience; requires configure() to have been called."""
    if _default_generator is None:
        raise RuntimeError(
            "WordGenerator not configured. Call wordgen.configure(wordlist_dir) first."
        )
    return _default_generator.get_next_word(word_type, rng)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sanity check: feed a known RNG state through the Markov chain"
    )
    parser.add_argument("--wordlist-dir", required=True, help="Path to wordlist files")
    parser.add_argument("--ext", default=".txt", help="Wordlist file extension")
    parser.add_argument("--seed", type=int, required=True, help="DotNetRandom seed")
    parser.add_argument(
        "--word-type",
        choices=[w.name for w in Word],
        default="passwords",
    )
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    gen = WordGenerator(args.wordlist_dir, ext=args.ext)
    rng = DotNetRandom(args.seed)
    wt = Word[args.word_type]
    for _ in range(args.count):
        print(gen.get_next_word(wt, rng))
