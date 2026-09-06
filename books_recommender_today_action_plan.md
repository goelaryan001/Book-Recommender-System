# Books Recommender System — Today's Action Plan (5 Hours to Interview)

Time budget suggestion, adjust as needed: **~45-60 min getting this running and fixed, ~20-30 min understanding the fixes, the rest of your time on rehearsal** (both this project's Q&A and your "tell me about yourself"). Understanding beats extra features today — don't build anything new unless the fixes above are done with real time to spare.

---

## Step 1: Run it as-is first (10 min)

```bash
cd Books-Recommender-System
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then try running the training pipeline:
```bash
python main.py
```

**It will very likely crash** — see Bug #1 below. That's expected and fine; seeing the real error first means you're not just trusting a summary, you're watching the actual failure.

## Step 2: The real bugs, verified against the actual code

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `stage_01_data_validation.py` | `error_bad_lines=False` — removed in pandas 2.0+, will crash outright | Replace with `on_bad_lines='skip'` |
| 2 | `stage_02_data_transformation.py` | `book_pivot.fillna(0, inplace=True)` — treats "never rated" as "rated zero," distorting similarity | Acceptable to leave as-is for time, but know this is a real limitation — mention it if asked |
| 3 | `stage_03_model_trainer.py` | `NearestNeighbors(algorithm='brute')` — no metric specified, defaults to Euclidean instead of cosine | Add `metric='cosine'` |
| 4 | `app.py` | `model.kneighbors(..., n_neighbors=6)` likely returns the queried book as its own neighbor (distance 0) | Request `n_neighbors=7` and drop the first result, or filter the query book out of the output |

**Given the time crunch: fix #1 (required, nothing runs without it) and #3 (one line, real improvement, easy to explain). Bugs #2 and #4 are fine to just be able to *talk about* as known limitations rather than fix, if time is short.**

## Step 3: The Claude Code prompt — paste this in now

```
I have an existing Books Recommender System project (item-based collaborative
filtering with k-Nearest Neighbors on the Book-Crossing dataset). I have an
interview in a few hours and need it running correctly, with two specific
fixes applied. Please:

1. Fix stage_01_data_validation.py: `error_bad_lines=False` is removed in
   pandas 2.0+ and will crash. Replace with `on_bad_lines='skip'`.

2. Fix stage_03_model_trainer.py: NearestNeighbors(algorithm='brute') has no
   metric specified, defaulting to Euclidean distance. Change to
   NearestNeighbors(algorithm='brute', metric='cosine') -- cosine similarity
   is the correct choice for comparing rating patterns since it's invariant
   to rating scale, unlike Euclidean distance.

3. Run the full training pipeline (main.py) end to end and confirm it
   completes without errors, producing the pickled model and pivot table.

4. Run the Streamlit app (streamlit run app.py) and confirm it actually
   loads and returns book recommendations for at least one real book title
   from the dataset.

5. Tell me clearly: did the recommendations look sensible after the cosine
   fix? Show me one example query book and its recommended neighbors.

Don't add any new features or restructure anything beyond these two fixes --
I need this working and verified, not expanded, given my timeline today.
```

## Step 4: Optional, only if you finish Steps 1-3 with real time to spare

**Do not attempt this unless you have at least 90 minutes left after Step 3 is done and verified.** A half-finished enhancement is worse than no enhancement.

If you do have time, the single highest-value addition is fixing Bug #4 (the self-recommendation artifact) — it's small, and "I noticed the model was recommending the book itself, so I fixed it" is a genuinely good, easy-to-explain interview moment:

```
One more small fix, only if the above is done and verified: in app.py's
recommend_book function, the model likely returns the queried book itself
as one of its own "recommendations" since it's searching the same pivot
table it's a row of. Request n_neighbors=7 instead of 6, and filter the
query book out of the final results list before returning them.
```

## Step 5: Rehearsal, not more building

Once Step 3 is confirmed working:
1. Actually query 2-3 different books in the running app, look at real output
2. Reread `books_recommender_interview_prep.md` (the Q&A doc from earlier) once, out loud
3. Run your "tell me about yourself" once, out loud, timed
4. Stop building. Rest matters more than a sixth bug fix at this point.
