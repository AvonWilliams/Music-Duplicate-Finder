# Music-Duplicate-Finder
MusicBrainzPicard plugin designed to leverage DGPU to rapidly identify and manage duplicate music files in large music collections

Two independent detection engines for your Picard library:

1. Find Duplicates (AcoustID)
   Uses chromaprint audio fingerprints (the same algorithm MusicBrainz
   itself uses) to find files that are the same master recording —
   regardless of codec, bitrate, or container. Tolerant of minor
   encoder/padding differences via user-configurable alignment window
   (off / narrow / standard / wide / exhaustive). GPU-accelerated via
   torch+CUDA with CPU fallback.

2. Find Similar Songs (CLAP)
   Uses CLAP neural audio embeddings to cluster songs by genre, mood,
   and production. Finds "songs that sound alike" — NOT duplicates.
   Runs on a remote FastAPI GPU server or on the local machine's GPU.

Both engines share:
- Three configurable confidence tiers (Certain / Likely / Unsure)
- Complete-linkage (clique) clustering — every pair in a group is
  guaranteed above the Unsure threshold
- Full results browser with per-file quality scoring, file path display,
  and per-file actions (play, view tags, move, delete)
- Checkbox selection for batch move and batch delete across multiple files
- Save results to a .mdupe file and reload them later without rescanning
  (Tools → Load Duplicate Results…)
- Full diagnostic logging for reproducibility and threshold tuning

Requirements:
- AcoustID engine: pyacoustid, numpy, torch (for GPU); fingerprints must
  be calculated by Picard first via Tools → Scan
- CLAP engine (local):  torch, transformers, soundfile, scipy; ffmpeg on PATH
- CLAP engine (remote): FastAPI inference server reachable on the network
"""
