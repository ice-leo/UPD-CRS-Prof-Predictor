# LingapKalusugan: Sagot Ka Saan Mang Dako ng Bansa

[![Project Status](https://img.shields.io/badge/status-Prototype-yellow)](https://github.com) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) ![Made by Team Mathemarizz](https://img.shields.io/badge/team-Mathemarizz-lightgrey)

> **Philippine Junior Data Science Challenge 2 (Hackathon)**  
> Segmentation of BPI customers in far-flung areas using RFM Analysis + K‑Means (with PCA).

---

## 🚀 Project Summary

**LingapKalusugan** is a data-driven project that segments BPI customers living in remote/far-flung areas and proposes an affordable health insurance product specially designed for the most promising segment. The clustering uses **RFM features**, K‑Means clustering (with PCA), and cluster validation metrics to select a target group for a product prototype.

**Key takeaway:** Cluster 5 — *middle-aged, mid-high spenders, salaried, mid-educated, frequent transactors* — is the primary target for the proposed health insurance offering.

---

## 📌 Project Info

<table>
  <tbody>
    <tr>
      <th scope="row">Project Start Date</th>
      <td>September 12, 2023</td>
    </tr>
    <tr>
      <th scope="row">Project End Date</th>
      <td>October 13, 2023</td>
    </tr>
    <tr>
      <th scope="row">Project Creators:</th>
      <td>Isaiah John Mariano, Jeremiah Daniel Regalario, Meluisa Montealto</td>
    </tr>
    <tr>
      <th scope="row">University</th>
      <td>University of the Philippines - Diliman</td>
    </tr>
    <tr>
      <th scope="row">Project Tags</th>
      <td><em>health-insurance</em>, <em>debit-card</em>, <em>RFM</em>, <em>pca</em>, <em>kmeans</em>, <em>clustering</em></td>
    </tr>
  </tbody>
</table>

---

## 🧾 Data & Methods (brief)

- **Data sources:** 6 datasets from BPI; debit card transactions selected for clustering due to coverage and correlation.  
- **RFM features used:** `Recency`, `Frequency`, `Monetary` (3‑month total) plus demographic features (`Age`, `Education`, `Income Source`).  
- **Preprocessing highlights:** duplicates removed, March entries excluded when needed, missing values handled (RF imputation for Education ~31% missing), log transform & `StandardScaler`.  
- **Clustering:** K‑Means (w/o PCA), K‑Means with PCA (chosen), DBSCAN.  
- **Validation metrics:** Silhouette (~0.43 for KMeans+PCA), Calinski‑Harabasz (~4800), Davies‑Bouldin (~1.25).

---

## 📊 Resulting Clusters (summary)

- **Cluster 5 (Target):** Middle‑aged, mid‑high spender, salaried, mid‑educated, high frequency — *primary focus*.  
- Cluster 0: Middle‑aged, low spender, salaried  
- Cluster 4: Young, low‑educated, allowance recipients  
- Cluster 3: Older, low spender, pension recipients  
- Cluster 2: Older, high spender, remittance-related  
- Cluster 6: Scattered; business people  
- Cluster 1: Scattered; mixed profile

---

## 💡 Product Prototype — *LingapKalusugan*

We designed three tiered packages with accessibility and simplicity in mind. The pricing strategy is intentionally affordable for the target cluster and illustrated with expected coverage & benefits below.

> **Affordability note:** average monthly debit transactions ≈ **₱20,000** (observed). A ₱500 monthly plan equals **2.5%** of that monthly transaction level — designed to be reachable for salaried customers in Cluster 5.

### Pricing & Benefits Comparison

| Package | Price (monthly) | Price (quarterly) | Price (semi-annually) | Price (annual) | Hospital Cover | Daily Allowance (max 100 days) | Free Transport | Billing Freeze |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| **LK Silver** | **₱500** | ₱1,900 | ₱2,800 | ₱5,500 (10 yrs option) | ₱80,000 | ₱100/day | 1 one‑way (to & fro) | 1 month (with proof) |
| **LK Gold** | **₱800** | ₱3,100 | ₱4,600 | ₱9,000 (10 yrs option) | ₱120,000 | ₱170/day | 1 one‑way (to & fro) | 2 months |
| **LK Platinum** | **₱1,000** | ₱3,100* | ₱4,600* | ₱9,000* | ₱150,000 | ₱200/day | 2 one‑ways | 2 months |
\* (Note: semi/annual/annual pricing for Platinum adjusted for demonstration — confirm with actuarial pricing.)

> Tip: Use the **monthly** price as the quick affordability metric for customer outreach communications.

---

## 🧾 Package Details (expanded)

<details>
<summary><strong>LK Silver — ₱500 / month</strong></summary>

**Inclusions & Notes:**

- Hospital bill coverage up to **₱80,000**.  
- Daily support allowance **₱100/day** for up to **100 days** during hospitalization.  
- **One free round-trip** transport to a metropolitan city (one-time).  
- **Temporary freeze** of billing accounts for **1 month** with medical proof.  
- A compact mini‑diary in the customer's dialect explaining coverage, contribution schedule, and benefits.  
</details>

<details>
<summary><strong>LK Gold — ₱800 / month</strong></summary>

**Inclusions & Notes:**

- Hospital bill coverage up to **₱120,000**.  
- Daily support allowance **₱170/day** for up to **100 days**.  
- **One free round-trip** transport to a metropolitan city (one-time).  
- **Temporary freeze** of billing accounts for **2 months** with medical proof.  
- Quarterly check-in / survey for the first 3 months.  
</details>

<details>
<summary><strong>LK Platinum — ₱1,000 / month</strong></summary>

**Inclusions & Notes:**

- Hospital bill coverage up to **₱150,000**.  
- Daily support allowance **₱200/day** for up to **100 days**.  
- **Two** one-time free round-trips to a metropolitan city.  
- **Temporary freeze** of billing accounts for **2 months** with medical proof.  
- Premium onboarding with the mini-diary plus assisted registration.  
</details>

---

## 🔢 Example: Why ₱500/month is feasible

Based on observed averages (project data):

- **Debit transactions:** ₱20,000 / month  
- **Auto-pay for credit:** ₱15,000  
- **Remaining balance:** ₱5,000  

A ₱500 monthly premium is **2.5%** of ₱20,000 — framed as a small, recurring contribution for health protection and peace of mind. This is highlighted in customer outreach materials as the "small daily equivalent":

- ₱500 / 30 ≈ **₱16.67 / day** — less than the price of a single inexpensive meal in many provinces.

---

## 🧭 Customer Journey & Experience

1. **Acquisition & Education:** Digital campaigns, local dialect mini‑diary, partner outreach via barangay centers.  
2. **Onboarding:** Simple form (digital/assisted), optional auto‑pay via debit account.  
3. **Usage:** Claims initiated with proof; partner hospitals/transport providers assist logistics.  
4. **Support & Retention:** Monthly feedback for first 3 months, quarterly thereafter; incentives (raffles, vouchers) to boost participation.
5. **Scaling:** Use positive feedback to expand via media endorsements & partner channels.

---

## ⚖️ Risks & Considerations

- Pricing & actuarial validation required (consult actuaries).  
- Communication barriers in remote areas; natural disasters may disrupt service.  
- Financial literacy gaps — invest in education materials and local engagement.  

---

## 📈 Visualizations (placeholders)

Exported figures from `eda.ipynb` should be saved in `images/` and referenced below:

```
![Missing values heatmap](images/fig_01_missing_values.png)
![Transactions over time](images/fig_02_transactions_over_time.png)
![Cluster distributions](images/fig_03_clusters.png)
![Target segment profile](images/fig_04_target_cluster.png)
```

---

## 🛠️ How to run the notebook

```bash
python -m venv venv
source venv/bin/activate  # mac/linux; use venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter lab eda.ipynb
# or
jupyter notebook eda.ipynb
```

---

## 📁 Repository structure

```
├─ README.md
├─ eda.ipynb
├─ docs/                       # project documentation (this content)
├─ requirements.txt
├─ data/
│  ├─ debit_transactions.csv
│  └─ other_datasets...
└─ images/
   ├─ fig_01_missing_values.png
   └─ ...
```

## Authors

- [Isaiah John Mariano](https://github.com/ice-leo)
- Meluisa Montealto
- [Jeremiah Daniel Regalario](https://github.com/jeremiahdanielregalario)


---

## 📚 References

A full list of references and research resources used by the team is included in the project documentation (Kaggle notebooks, Analytics Vidhya posts, DataReportal Philippines, Google ML clustering references, etc.).

---

## 📝 License & Contact

This project is released under the **MIT License**.  
For questions, collaboration inquiries, or to request the full documentation, contact **Team Mathemarizz** — University of the Philippines Diliman.

