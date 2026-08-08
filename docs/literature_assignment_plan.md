# Literature Assignment Plan
## Comparative Analysis of Machine Learning, CNNs, and Vision Transformers for Weed Detection in Smart Agriculture

**Decisions locked so far:** classification task · DeepWeeds (primary) + MFWD (secondary, 28 weed
species, pre-cropped) for cross-dataset generalization testing · scope narrowed from "weed and crop"
to "weed detection" since neither dataset has a labeled crop class.

**Team roles (from project plan):**
- **Person A** — Team Lead + Literature Reviewer → Intro / Related Work / Discussion context
- **Person B** — Implementation Lead → Classical ML baseline + CNN baseline, Methodology/Results
- **Person C** — Custom Model + Visualisation → ViT / hybrid model, ablations, figures

Each cluster below = 5 papers. Read *all 50 abstracts* as a team by Week 3 (handbook requires a 15+ paper annotated bibliography); each owner does a **deep read + 1-paragraph annotation** of their assigned clusters, and a **skim + 2-sentence note** of everyone else's, so the whole team can speak to any citation in the final paper.

---

## PERSON A — Context & Framing (15 papers)

### Cluster 1: Foundational Datasets & Benchmarks
1. Olsen, A. et al. (2019). *DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning.* Scientific Reports, 9, 2058. — **Primary dataset.** Images collected June 2017–March 2018 across 8 Queensland rangeland sites; 17,509 images, 8 weed species + negative class. Note the ~7-year age of the imagery in your Methodology's limitations.
2. Genze, N. et al. *Moving Fields Weed Dataset (MFWD).* — **Second dataset**, used for the cross-dataset generalization test. 94,321 images, 28 weed species, pre-cropped per-species (originally built for segmentation, but the per-species crops work directly for classification). Single-region collection, so this tests species-diversity generalization more than geographic generalization.
3. Steininger, D., Trondl, A., Croonen, G., Simon, J., & Widhalm, V. (2023). *The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation.* WACV 2023, 3718-3727. (Background reading — 74 species, most diverse option considered, not used directly since it needs bounding-box crop-out work we're skipping.)
4. Dyrmann, M. et al. — weed classification across 22 species, 10,413 images, 86.2% accuracy (cited in WeedSwin, Sci. Reports 2025).
5. Mesías-Ruiz, G.A., Borra-Serrano, I., Peña, J.M., De Castro, A.I., Fernández-Quintanilla, C., & Dorado, J. (2024). *Weed species classification with UAV imagery and standard CNN models: assessing the frontiers of training and inference phases.* Crop Protection, 182, 106721.

### Cluster 9: Precision-Agriculture Robotics & Deployment
6. Bawden, O. et al. — AgBotII modular weeding robot (LBP + covariance features for crop/weed ID).
7. Lottes, P., Khanna, R., Pfeifer, J., Siegwart, R., & Stachniss, C. (2017). *UAV-based crop and weed classification for smart farming.* ICRA 2017, 3024–3031.
8. Hasan, A.S.M.M., Sohel, F., Diepeveen, D., Laga, H., & Jones, M.G.K. (2021). *A survey of deep learning techniques for weed detection from images.* Computers and Electronics in Agriculture, 184, 106067.
9. *Advances in ground robotic technologies for site-specific weed management in precision agriculture: A review.* Computers and Electronics in Agriculture, 225 (2024), 109363.
10. *SPARROW: Smart Precision Agriculture Robot for Ridding of Weeds.* arXiv:2405.20896 (2024) — YOLO-based detection + spot-spraying.

### Cluster 10: Reviews, Evaluation Methodology & Explainability
11. *Comparative Evaluation of Convolutional and Transformer-Based Detectors for Automated Weed Detection in Precision Agriculture.* arXiv:2605.00908 (2026) — directly analogous to your comparison design.
12. Qu, H., & Su, W. (2024). *Deep learning-based weed–crop recognition for smart agricultural equipment: a review.* Agronomy, 14(2), 363.
13. *Deep learning–based approaches for weed detection in crops.* Frontiers in Plant Science, 16 (2025), 1746406.
14. Murad, N.Y., Mahmood, T., Forkan, A.R.M., Morshed, A., Jayaraman, P.P., & Siddiqui, M.S. (2023). *Weed detection using deep learning: a systematic literature review.* Sensors, 23(7), 3670.
15. *WeedSwin: hierarchical vision transformer with SAM-2 for multi-stage weed detection and classification.* Scientific Reports (2025).

**A's job:** use these to write the Introduction's problem framing + 8–12-work literature gap, the "why compare 3 paradigms" justification, and the Discussion's real-world deployment implications.

---

## PERSON B — Classical ML & CNN Baselines (20 papers)

### Cluster 2: Classical ML for Weed/Crop Classification
16. Islam, N. et al. (2021). *Early Weed Detection Using Image Processing and Machine Learning Techniques in an Australian Chilli Farm.* Agriculture, 11(5), 387. (RF 96%, SVM 94%, KNN 63%)
17. Tufail, M. et al. (2021). SVM-based weed detection in tobacco fields using texture, shape, and colour features (96% accuracy, 6 FPS).
18. Espejo-Garcia, B. et al. (2020). *Towards weeds identification assistance through transfer learning.* Computers and Electronics in Agriculture, 171, 105306.
19. Karimi, Y. et al. (2006). SVM for weed detection in corn from hyperspectral images.
20. *Machine learning and handcrafted image processing methods for classifying common weeds in corn field.* Smart Agricultural Technology (2023) — RF > SVM > kNN with 21 shape features + PCA.

### Cluster 3: CNN Backbone Architectures (general, for transfer learning)
21. Krizhevsky, A., Sutskever, I., & Hinton, G.E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet). NeurIPS.
22. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG). arXiv:1409.1556.
23. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition* (ResNet). CVPR.
24. Howard, A.G. et al. (2017). *MobileNets: Efficient CNNs for Mobile Vision Applications.* arXiv:1704.04861.
25. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML.

### Cluster 4: CNN Applied to Weed/Crop Tasks
26. Sunil, G.C., Zhang, Y., Howatt, K., Schumacher, L.G., & Sun, X. (2024). *Multi-Species Weed and Crop Classification Comparison Using Five Different Deep Learning Network Architectures.* J. ASABE, 67, 43–55.
27. Razfar, N., True, J., Bassiouny, R., Venkatesh, V., & Kashef, R. (2021). *Weed detection in soybean crops using custom lightweight deep learning models.* Computers and Electronics in Agriculture, 185, 106016.
28. Beeharry, Y., & Bassoo, V. — AlexNet CNN, 99.8% accuracy distinguishing soil/soybean/weed (15,336 images).
29. Ramcharan, A., McCloskey, P., Baranowski, K., et al. (2017). *Deep Learning for Image-Based Cassava Disease Detection.* Frontiers in Plant Science, 8, 1852.
30. Shackleton et al. (2024) — 7 pretrained CNNs for rangeland weed detection; EfficientNetV2B1 best at 94.2%.

### Cluster 5: Object Detection & Segmentation Architectures
31. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection* (YOLO). CVPR.
32. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8.*
33. Kirillov, A., Mintun, E., Ravi, N., et al. (2023). *Segment Anything* (SAM). ICCV.
34. Wang et al. (2024). *CSCW-YOLOv7* for weed detection in wheat fields (P 97.7%, R 98%, mAP 94.4%).
35. Khan, A. et al. (2020). *CED-Net* — outperforms U-Net and SegNet for crop/weed segmentation.

**B's job:** implement and document the classical-ML pipeline (feature engineering + RF/SVM) and the CNN baseline (transfer learning from a backbone in Cluster 3). Cluster 5 (YOLO/U-Net/SAM) is now background context only — kept for the Related Work section's discussion of detection/segmentation alternatives, since we locked in classification over detection. Writes Methodology + Results tables.

---

## PERSON C — Vision Transformers & Hybrid Models (15 papers)

### Cluster 6: ViT Foundations
36. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
37. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT). ICLR.
38. Touvron, H. et al. (2021). *Training data-efficient image transformers & distillation through attention* (DeiT). ICML.
39. Liu, Z. et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV.
40. Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision.* TMLR.

### Cluster 7: ViT Applied to Agriculture
41. Hasasneh, A. et al. (2025) — ViT on DeepWeeds dataset, 96.9% accuracy across 9 weed species.
42. Salamai, A. et al. (2023) — lesion-aware ViT for paddy disease, 98.74% accuracy.
43. Parez, S. et al. (2023) — *GreenViT*, ViT-based plant disease detection outperforming CNN baselines.
44. Barman, U. et al. (2024) — *ViT-SmartAgri*, smartphone ViT app for tomato disease (10 classes, 90.99% acc.).
45. Thakur, P. et al. (2021/2023) — *PlantViT*, hybrid CNN+ViT for leaf disease diagnosis.

### Cluster 8: Hybrid CNN-Transformer Models (your "custom model" territory)
46. Wei, Y., Feng, Y., Zu, D., & Zhang, X. (2025). *A hybrid CNN-transformer network: accurate and efficient semantic segmentation of crops and weeds on resource-constrained embedded devices.* Crop Protection, 188, 107018.
47. Zhang, J. et al. (2023). *Weed Recognition Method based on Hybrid CNN-Transformer Model.* Frontiers in Computing and Intelligent Systems, 4, 72–77. (89.43% vs. plain ViT on DeepWeeds)
48. Borhani, Y. et al. (2022) — vision-transformer-based plant disease classification, lightweight real-time variant.
49. Li, G. et al. (2023) — MobileViT + CBAM attention for real-time mobile plant disease detection.
50. Xie, W. et al. (2024). *Recent advances in Transformer technology for agriculture: A comprehensive survey.* Engineering Applications of Artificial Intelligence, 138, 109412.

**C's job:** this cluster *is* your "custom model, not just a library call with defaults" requirement — use paper 46 or 47 as your architectural template for a lightweight hybrid CNN-ViT, and papers 41–45 as your accuracy benchmarks to beat/match. Writes the custom-model section of Methodology, all figures/ablation tables, and the ViT vs. CNN efficiency-accuracy tradeoff discussion.

---

## How this maps onto the Week-by-Week Deliverables

| Week | Task | Who |
|---|---|---|
| 1–2 | Skim all 50 abstracts as a team; task = classification, datasets = DeepWeeds + CottonWeedID15 (locked) | All |
| 3 | Annotated bibliography (15+ papers, deep annotations from your own cluster) | Each person annotates their clusters |
| 4 | EDA — use Cluster 1 dataset papers to justify preprocessing choices | A leads, all contribute |
| 5–6 | Baseline models — Clusters 2–5 inform RF/SVM + CNN implementation | B leads |
| 7–8 | Custom ViT/hybrid model + ablations — Clusters 6–8 inform architecture | C leads |
| 9 | Draft paper — each person drafts the section tied to their clusters | All |
| 10 | Final paper — extend citation list toward 30–50 total (you're already there) | All |

By final submission you'll already be sitting at the handbook's target of 30–50 references (Section 2 intro) without needing to scramble for more in Week 9.
