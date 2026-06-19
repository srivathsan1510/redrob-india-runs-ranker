# Running the Redrob Ranker on the Real 100K Candidate Pool

This is the exact sequence to run locally, where you have normal internet
access (needed once, to download the sentence-transformers model).

## 1. Set up the environment

```bash
# from inside the redrob/ project folder
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

This will take a couple of minutes — sentence-transformers pulls in torch,
which is the largest dependency.

## 2. Get the candidate data

Download `candidates.jsonl.gz` from the Drive link the hackathon gave you,
and unzip it:

```bash
gunzip -k candidates.jsonl.gz   # -k keeps the .gz, you get both files
wc -l candidates.jsonl          # sanity check: should print 100000
```

Place `candidates.jsonl` (or keep it gzipped, both work) inside `data/`.

## 3. Run precompute (Phase A — no time limit)

This is the slow step: it downloads `all-MiniLM-L6-v2` (~80MB, one-time,
needs network) and then embeds all 100K candidate text blocks. On a normal
laptop CPU this should take somewhere in the 2-10 minute range depending on
your hardware — there's no hard time limit on this phase, so don't worry if
it takes longer.

```bash
python3 precompute.py \
  --candidates data/candidates.jsonl \
  --jd-text-file config/jd_text.txt \
  --out-dir artifacts \
  --embedder sentence-transformers
```

Watch for the progress bar during embedding. When it finishes you should see:

```
Saved embeddings -> artifacts/embeddings.npy shape=(100000, 384)
Saved JD embedding -> artifacts/jd_embedding.npy shape=(384,)
Precompute complete.
```

If you used the gzipped file directly instead of unzipping, just point
`--candidates` at `data/candidates.jsonl.gz` — the script handles both.

## 4. Run rank (Phase B — must finish in 5 min / 16GB / CPU / no network)

This step does NOT touch the network or recompute embeddings — it's pure
numpy/pandas arithmetic over the artifacts from step 3, so it should run in
seconds even on 100K rows. This is also the command you'll put in
`submission_metadata.yaml` as `reproduce_command` (see note below).

```bash
python3 rank.py \
  --candidates data/candidates.jsonl \
  --out outputs/submission.csv \
  --top-n 100
```

You should see:

```
Wrote 100 ranked rows -> outputs/submission.csv
```

## 5. Validate

```bash
python3 validate_submission.py outputs/submission.csv
```

Should print `Submission is valid.` with zero errors. If you see tie-break
or row-count errors, something went wrong upstream — don't submit until
this passes cleanly.

## 6. Sanity-check the actual rankings

Before trusting the output, spot check it the way I did with the 50-sample:

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('outputs/submission.csv')
print(df.head(15)[['rank','candidate_id','score']].to_string())
print()
print(df.tail(10)[['rank','candidate_id','score']].to_string())
"
```

Look for: does the top 15 look like genuinely relevant titles when you
cross-reference candidate_id against the candidate data? Is there a
believable score gap between rank 1 and rank 50, or does everything look
flat (a flat distribution at 100K scale would suggest the scoring isn't
discriminating well and needs another look)?

## Send the result back

Once `outputs/submission.csv` validates cleanly, upload just that file
(it'll be small, well under a few hundred KB) back to our chat along with
a quick description of what the top 15 and bottom 10 titles look like — I'll
review the distribution and we'll decide if any further calibration is
needed before you do the actual portal submission.

## Note on `submission_metadata.yaml`'s `reproduce_command`

The metadata template asks for the single command that reproduces your
submission. Since our pipeline is two phases (precompute is slow/one-time,
rank is the fast timed step), use:

```
reproduce_command: "python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv --top-n 100"
```

...and document in your README that `precompute.py` must be run once first
to generate `artifacts/`. This is honest and matches how the spec describes
"pre-computation" as a separate, allowed step (Section 3 / the
`pre_computation_required` field in the metadata template already
anticipates this).
