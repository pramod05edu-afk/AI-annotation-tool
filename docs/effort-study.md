# Effort study: manual vs. AI-assisted annotation

**Question:** Does AI assistance (pre-labelling with the fine-tuned detector + magic click)
reduce the effort of labelling images, without lowering label quality?

**Measured automatically** (interaction log, `POST /api/logs`):

| Metric | Definition |
|---|---|
| Time per image | from `image_opened` to `image_done` (only while the image is open) |
| Interactions per image | boxes / polygons created, edited, deleted, class changes, undos, magic clicks |
| AI share | of the final labels: how many are AI-unchanged, AI-edited, manual |
| Label quality | final labels vs. the dataset's ground truth: precision, recall, F1 at IoU ≥ 0.5 |

## Design

- **Images:** 60 images from the **test split** (never seen in training), 3–15 objects each,
  split into two sets with the same difficulty (184 vs. 183 objects; `eval/effort_study_sets.json`).
- **Set A → manual mode**, project *Effort study A (manual)*.
  **Set B → assisted mode**, project *Effort study B (assisted)*.
- **Different images per mode on purpose:** labelling the same image twice would be faster the
  second time (memory), which would bias the comparison.
- **Blocks of 10, alternating** (A 1–10, B 1–10, A 11–20, B 11–20, A 21–30, B 21–30), so that
  getting tired or getting better with practice affects both modes equally.
- Statistics: per-image times per mode, mean ± standard deviation and median;
  **Mann-Whitney U test** (two independent groups, no normality assumption) and effect size.

## Protocol (please follow exactly)

**Before you start**
1. `docker compose up -d`; wait until the header shows "Backend ok" and the AI is ready.
2. **Warm-up (not measured):** in the project *Helmet test*, label 3–4 images in each mode until
   the shortcuts feel natural (B box, 1/2 class, Del, Enter, S suggest, C smart click).
3. Close other programs and notifications. Use the same screen and mouse for the whole study.

**Labelling rules (both modes)**
- Label **every** visible `helmet` (a head wearing a hard hat) and `head` (a head without one).
- Tight boxes around the **head / helmet** only, like the existing dataset labels.
- Use the **box tool** (B). In assisted mode also: Suggest (automatic), delete wrong boxes (Del),
  fix boxes (select V, drag), change classes (1 / 2), add missing ones with **smart click (C)**
  (click the object, the box appears with its class) or, if that finds nothing, the box tool (B).
- Work at your **normal pace**: accurate, not rushed.
- When an image is complete, press **Enter** (Done). Do not go back to finished images.
- **Breaks only between images.** Never pause while an image is open: go back to the project
  page instead (the clock runs while an image is open).

**Session order**
| Block | Project | Images | Mode button in the toolbar |
|---|---|---|---|
| 1 | Effort study A (manual) | 1–10 | **Manual** |
| 2 | Effort study B (assisted) | 1–10 | **Assisted** |
| 3 | Effort study A (manual) | 11–20 | **Manual** |
| 4 | Effort study B (assisted) | 11–20 | **Assisted** |
| 5 | Effort study A (manual) | 21–30 | **Manual** |
| 6 | Effort study B (assisted) | 21–30 | **Assisted** |

Open the first *To do* image of the project ("Start annotating") and work through it with Enter;
after 10 images go back to the project page. **Check the mode button every time you switch.**

Expected duration: about 1–1.5 hours in total, plus breaks.

**Afterwards**
```bash
cd backend
uv run python scripts/analyze_effort.py     # writes eval/results/effort_study.{json,md} + figures
```

## Threats to validity (for the report)

- **One annotator** (the developer), who knows the tool well; results may differ for new users.
- Not blind: the annotator knows which mode is measured.
- Ground truth of the public dataset contains some missed heads (seen while preparing the data),
  so quality is measured against imperfect labels, equally for both modes.
- Assisted labels may inherit model errors that a human does not notice (automation bias):
  checked by comparing label quality of both modes against the ground truth.
