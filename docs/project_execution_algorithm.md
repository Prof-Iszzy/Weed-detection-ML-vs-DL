# Project Execution Algorithm
## Comparative Analysis of Machine Learning, CNNs, and Vision Transformers for Weed Detection in Smart Agriculture

This is the full pipeline, from raw data to submitted manuscript, structured so each step maps to a
specific requirement in the handbook (Section 3 rubric, Section 4 technical requirements, Section 5
timeline).

---

## ALGORITHM: End-to-End Project Pipeline

```
BEGIN Project

────────────────────────────────────────────
PHASE 0 — SETUP (Week 1-2, 5% of grade)
────────────────────────────────────────────
1. [DECIDED] Task type = CLASSIFICATION
   (cleaner 3-way comparison; classical ML stays a fair, natural baseline)
2. [DECIDED] Datasets:
   - Primary: DeepWeeds (17,509 images, 8 weed species, imaged June 2017-Mar 2018)
   - Secondary: MFWD / Moving Fields Weed Dataset (94,321 images, 28 weed species,
     pre-cropped per-species) — used ONLY in Phase 7 for cross-dataset generalization
     testing, NOT merged into the main train/val/test split
   - Scope note: project renamed from "weed and crop detection" to "weed detection"
     since neither dataset has a labeled crop class (only weeds + negative/background)
3. Create GitHub repo:
   - README.md (problem statement, setup instructions)
   - requirements.txt / environment.yml
   - /src, /configs, /notebooks, /data (gitignored), /results
4. Write and submit one-page proposal (research question + method)
5. Assign roles (Team Lead / Implementation Lead / Custom-Model Lead)
   IF team_size == 3:
       collapse 5 handbook roles into 3 (see literature_assignment_plan.md)

────────────────────────────────────────────
PHASE 1 — LITERATURE REVIEW (Week 3, 10%)
────────────────────────────────────────────
6. FOR each cluster IN [datasets, classical_ML, CNN, ViT, hybrid,
                         detection_archs, robotics, reviews]:
       assigned_member.deep_read(cluster)
       team.skim(cluster)
7. Build annotated bibliography (15+ papers minimum, target 30-50 by final)
8. Identify literature gap → this becomes your Introduction's
   "no prior work directly compares ML vs CNN vs ViT on X crop/region" claim

────────────────────────────────────────────
PHASE 2 — DATA ACQUISITION & EDA (Week 4, 10%)
────────────────────────────────────────────
9. Download BOTH datasets (DeepWeeds primary, MFWD secondary);
   verify class balance, image count, resolution for each independently
10. ASSERT DeepWeeds.sample_count >= 1000   # classical-ML minimum
    ASSERT DeepWeeds.sample_count >= 5000   # DL minimum (else justify explicitly)
    NOTE: MFWD is eval-only (Phase 7) — no train/val split needed for it
11. Run EDA:
    - class distribution plot
    - sample images per class (grid)
    - missing/corrupt file check
    - if tabular features involved: correlation heatmap
12. Document data source, licensing, and collection ethics
13. Split data: train / val / test (e.g. 70/15/15), STRATIFIED by class
    → freeze test set now, do not touch again until Phase 7

────────────────────────────────────────────
PHASE 3 — PREPROCESSING PIPELINE (Week 4-5)
────────────────────────────────────────────
14. FOR classical ML:
       extract handcrafted features:
           color indices (ExG, ExGR), GLCM texture, HOG, shape descriptors
       normalize/scale features
       apply PCA or feature selection (document method)
15. FOR CNN and ViT:
       resize images to backbone's expected input size
       normalize using backbone's expected mean/std (e.g. ImageNet stats)
       define augmentation pipeline:
           rotation, flip, color jitter, random crop
           (apply ONLY to train split)
16. IF class imbalance detected:
       apply class weighting OR oversampling
       (test both — Cluster 7 papers show oversampling can hurt ViT performance)

────────────────────────────────────────────
PHASE 4 — BASELINE 1: CLASSICAL ML (Week 5-6, part of 15%)
────────────────────────────────────────────
17. FOR model IN [Random Forest, SVM]:      # at least one required
        define hyperparameter search space (documented)
        run grid_search OR random_search OR bayesian_opt
        train on train split with 5-fold CV on train+val
        record mean ± std accuracy, F1, precision, recall
18. Select best classical model → this is Baseline 1

────────────────────────────────────────────
PHASE 5 — BASELINE 2: CNN (Week 5-6, part of 15%)
────────────────────────────────────────────
19. Select backbone (ResNet-50 or EfficientNet, pretrained on ImageNet)
20. Replace/fine-tune classification head for your class count
21. Define hyperparameter search space: learning rate, batch size,
    freeze/unfreeze schedule, optimizer
22. Train with 5-fold CV, early stopping on val loss
23. Record mean ± std accuracy, F1, precision, recall, inference time
24. THIS is Baseline 2 (satisfies "at least one DL baseline" requirement)

────────────────────────────────────────────
PHASE 6 — CUSTOM MODEL: ViT / HYBRID (Week 7-8, 20%)
────────────────────────────────────────────
25. Build custom contribution — choose ONE:
    (a) ViT-B/16 with a customized classification head + tailored augmentation
    (b) Lightweight hybrid CNN-ViT (CNN stem for local features +
        transformer blocks for global attention) — see Cluster 8 papers
26. Justify architectural choices against Cluster 6/7/8 literature
27. Define hyperparameter search space (patch size, depth, heads, LR schedule)
28. Train with 5-fold CV, same protocol as Phases 4-5 for fair comparison
29. Record mean ± std accuracy, F1, precision, recall, inference time,
    parameter count, FLOPs (for the deployability discussion)

────────────────────────────────────────────
PHASE 7 — RIGOROUS COMPARISON (Week 7-8, part of Results 20%)
────────────────────────────────────────────
30. Evaluate all three finalized models ONCE on the frozen test set
31. Run statistical significance test across folds
    (paired t-test or Wilcoxon signed-rank between model pairs)
32. Build comparison table:
       columns = [Model, Accuracy, F1, Precision, Recall, Params, Inference_ms]
       rows = [Classical ML, CNN, ViT/Hybrid]
33. Generate confusion matrices for each model
34. Error analysis: pull misclassified examples, look for patterns
    (e.g. does ViT fail on the same weed species as CNN, or different ones?)
34b. CROSS-DATASET GENERALIZATION TEST:
     FOR each of the 3 finalized models (trained on DeepWeeds):
         evaluate zero-shot on MFWD (no retraining/fine-tuning)
         record accuracy drop relative to in-distribution DeepWeeds test accuracy
         NOTE: MFWD's 28 species only partially overlap DeepWeeds' 8 —
         restrict evaluation to the overlapping species, or reframe as an
         open-set/novel-species detection question (worth deciding explicitly
         and stating clearly in Methodology)
     → this is your novelty hook: report which paradigm (ML/CNN/ViT)
       degrades LEAST when moved to a different, more species-diverse weed set

────────────────────────────────────────────
PHASE 8 — ABLATION STUDIES (Week 7-8, part of Results 20%)
────────────────────────────────────────────
35. FOR your custom model specifically:
        ablate ONE component at a time:
            - custom head vs default head
            - with/without your augmentation strategy
            - with/without class weighting
        measure delta in accuracy/F1
36. Table: [Configuration, Accuracy, Δ from full model]

────────────────────────────────────────────
PHASE 9 — VISUALIZATION (ongoing, Visualisation lead)
────────────────────────────────────────────
37. Produce, with colour-blind-safe palettes and captions:
    - class distribution bar chart (EDA)
    - training/validation loss & accuracy curves, all 3 models
    - final comparison bar chart (Phase 7 table, visualized)
    - confusion matrices (heatmaps)
    - attention-map or Grad-CAM visualizations (interpretability bonus)

────────────────────────────────────────────
PHASE 10 — WRITE-UP (Week 9, 10% draft / Week 10, 30% final)
────────────────────────────────────────────
38. Draft IMRaD manuscript, each member writes their assigned section:
        Title, Abstract, Keywords        → Team Lead + Writing Coordinator
        Introduction, Related Work       → Literature Reviewer
        Methodology, Results             → Implementation Lead
        Discussion, Conclusion           → Writing Coordinator + all
        Figures/Tables + captions        → Visualisation Specialist
39. Cross-check every section against Section 3.1 rubric criteria
40. Peer-review exchange with another team (Week 9 requirement)
41. Revise based on peer feedback

────────────────────────────────────────────
PHASE 11 — REPRODUCIBILITY PACKAGE (Week 9-10, part of Code req.)
────────────────────────────────────────────
42. Freeze final code; ensure `python -m src.train --config configs/main.yaml`
    reproduces all three models from scratch
43. Finalize README: setup, data download instructions, how to reproduce
    each table/figure in the paper
44. Add Team Contribution Statement (individual contributions documented)
45. Disclose any AI-tool usage per Section 8, Rule 6

────────────────────────────────────────────
PHASE 12 — FINAL SUBMISSION (Week 10, 30%)
────────────────────────────────────────────
46. Submit: final PDF paper + GitHub repo link + supplementary materials
47. Prepare and deliver 10-minute team presentation
48. VERIFY before submitting:
    ASSERT references.count >= 20 AND references.count <= 30  # final paper min
    ASSERT both_ML_and_DL_baselines_present == True
    ASSERT custom_model_is_not_default_library_call == True
    ASSERT five_fold_CV_reported_with_mean_and_std == True
    ASSERT code_reproducible_from_single_command == True

END Project
```

---

## Critical path / risk notes

- **Biggest failure mode (per the handbook):** starting experiments late and compressing Weeks 9–10.
  Phases 4–8 above should have *preliminary* results by end of Week 7, even if not fully tuned.
- **Fairness of comparison matters more than any single model's accuracy.** Phase 7's protocol
  (same splits, same CV folds, same metrics, statistical test) is what the rubric's "with adequate
  statistical rigour" language is actually grading.
- **Don't let Phase 6 (custom model) become "just call `timm.create_model()`."** The rubric explicitly
  penalizes that — Phase 6 step 25-27 is where your novelty has to live.
