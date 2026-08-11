# artifacts/ — local working directory (not tracked)

Everything under this directory is produced by the pipeline and is excluded from
version control, because all of it derives from the CEJC corpus, which cannot be
redistributed. See [../docs/data-availability.md](../docs/data-availability.md).

The pipeline expects this layout. Create it by running the pipeline in order;
nothing here needs to be produced by hand.

```
artifacts/
├── tmp_meta/
│   ├── cejc_convlist.parquet            # corpus metadata (conversation list)
│   ├── cejc_speaker.csv                 # speaker attributes
│   └── cejc_speaker_conversation.csv    # speaker x conversation mapping
├── _tmp_utt/
│   └── cejc_utterances/part-00000.parquet   # utterance table
├── analysis/
│   ├── target_pairs/                    # step 1  HQ1 sample
│   ├── features_min/                    # step 5  19 interaction features
│   ├── datasets/                        # step 6  joined X + Y tables
│   ├── cejc_speaker_metadata.tsv        # step 6  sex / age per record
│   └── results/                         # step 7  all statistical output
├── cejc/
│   ├── monologues_cejc_home2_hq1_v1.parquet      # step 2  pinned input to scoring
│   ├── monologues_cejc_home2_hq1_v1.sha256.txt   # step 2  identity of the above
│   └── shards_home2_hq1/                         # step 3  resumable scoring units
├── big5/
│   ├── items_ipipneo120_ja.csv          # IPIP-NEO-120 items (Japanese)
│   ├── models.txt                       # '<label> <bedrock model id>' per line
│   └── llm_scores/                      # step 4  model responses and trait scores
├── baseline/                            # baseline conditions 2 and 3
└── dose_response/                       # feature manipulation experiment
```

## Why sha256 files sit next to the parquet files

The monologue table is the exact text the models were shown. Rebuilding it from
a slightly different corpus export, or from different filter thresholds, would
change what "the same analysis" means without changing any filename. The pinned
copy plus its recorded digest makes that detectable:

```bash
shasum -a 256 artifacts/cejc/monologues_cejc_home2_hq1_v1.parquet
cat artifacts/cejc/monologues_cejc_home2_hq1_v1.sha256.txt
```

The digest for the version used in the reported results is listed in
[../docs/data-availability.md](../docs/data-availability.md).

`make verify` is a different check: it compares the values reported in the
manuscript against the result files in `analysis/results/`. It does not verify
input digests, so run the comparison above as well after rebuilding inputs.
