# Data Description

This folder contains the raw and processed datasets used for predicting fatal road traffic accidents in Thailand at emergency call intake (1669 dispatch triage).

## Dataset Source
- **Dataset Name:** Thailand Road Accident 2019–2022
- **Kaggle Source:** [`thaweewatboy/thailand-road-accident-2019-2022`](https://www.kaggle.com/datasets/thaweewatboy/thailand-road-accident-2019-2022)
- **Primary Data Provider:** Department of Highways (DOH) & Open Government Data of Thailand
- **Coverage:** Road accident records nationwide from 2019 to 2022 (~81,735 incidents).

## Folder Structure
```
data/
├── raw/                # Unmodified original CSV dataset from Kaggle / Open Data
├── processed/          # Cleaned, feature-engineered, and split dataset
└── README.md           # Dataset documentation, schema, and leakage rules
```

> **Note on Data Files:** Due to GitHub file size limits and best practices, large raw `.csv` files are ignored by git (`.gitignore`). You can download the dataset automatically via `kagglehub` or place the extracted CSV into `data/raw/`.

---

## Data Schema & Variables

### 1. Target Variable
- **`fatal`**: Binary indicator (1 if `number_of_fatalities > 0`, else 0).
  - Train (2019–2021) baseline fatality rate: **~12.9%**
  - Test (2022) baseline fatality rate: **~10.8%**

### 2. Available Intake Features (Usable at Call Time)
Only information known to the emergency caller at the time of calling 1669 is retained:
- **`agency`**: Responding authority or jurisdiction (e.g., Department of Highways).
- **`province_en`**: Province where the accident occurred (77 provinces in Thailand).
- **`vehicle_type`**: Type of vehicle involved (e.g., motorcycle, pickup truck, passenger car, large truck).
- **`weather_condition`**: Weather at the time of the event (e.g., clear, rain, fog).
- **`road_description`**: Physical road layout (e.g., straight, curve, intersection, slope).
- **`slope_description`**: Incline/grade description (e.g., flat, steep slope).
- **`number_of_vehicles_involved`**: Number of vehicles in the collision.
- **Engineered Temporal Features (from `incident_datetime`):**
  - `hour`: Hour of the day (0–23)
  - `dayofweek`: Day of the week (0 = Monday, 6 = Sunday)
  - `month`: Month (1–12)
  - `is_weekend`: 1 if Saturday or Sunday, else 0
  - `is_night`: 1 if nighttime (18:00–05:59), else 0
  - `is_songkran`: 1 during 7 dangerous days of Songkran (April 11–17)
  - `is_newyear`: 1 during 7 dangerous days of New Year (Dec 29 – Jan 4)

---

## Data Leakage Prevention (Crucial Methodology)

In an operational 1669 dispatch environment, the decision to send an **Advanced Life Support (ALS)** vs. **Basic Life Support (BLS)** team must be made **immediately upon receiving the phone call**. 

The following columns are strictly removed to prevent data leakage:
| Column | Reason for Removal |
|---|---|
| `number_of_fatalities` | Direct target variable |
| `number_of_injuries` | Only known after the EMS team arrives on scene |
| `presumed_cause` | Determined after police and investigator post-crash review |
| `report_datetime` | System entry timestamp; often logged days or months after the call |
| `accident_type` | Precise collision classification determined after physical investigation |
| `acc_code`, `route`, `latitude`, `longitude` | Unique reference IDs / high-cardinality values avoided in first-stage triage |

---

## Train / Test Splitting Strategy
- **Temporal Splitting Rule:** Random splitting (e.g., standard `train_test_split`) must NOT be used because future accident patterns would leak into historical training.
- **Training Set:** Accidents occurring between **2019 and 2021** (60,703 rows).
- **Validation Set:** Year **2021** used to tune classification decision thresholds.
- **Test Set:** Year **2022** (21,032 rows) held out to simulate real-world prospective deployment.
- **Rare Category Handling:** Categories occurring < 100 times in the training set are grouped into `'other'`. The frequency table is computed strictly from training data and applied downstream.
