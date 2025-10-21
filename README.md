# Surgical Intelligence Engine

## Overview
An AI-powered analytics platform that uses dual-model SHAP analysis to predict surgical duration and overrun risk, automatically discovering hidden risk factors and temporal patterns in surgical operations data.

## Description
The Surgical Intelligence Engine transforms traditional surgical analytics into an intelligent system that not only predicts outcomes but actively discovers insights. Built on a foundation of dual machine learning models (duration regression and overrun classification), the system employs advanced SHAP (SHapley Additive exPlanations) analysis to understand *why* predictions are made.

**Key Capabilities:**
- **Dual-Model SHAP Analysis**: Simultaneous analysis of duration predictions and overrun risk
- **Smoking Gun Detection**: Automatically identifies features with low duration impact but high overrun risk - the hidden factors that cause unexpected delays
- **Temporal Drift Monitoring**: Tracks how feature importance changes over time, detecting shifts in recording practices or operational patterns
- **Cross-SHAP Analysis**: Discovers where duration and overrun models diverge, revealing complex operational dynamics
- **Anomaly Detection**: Flags unusual prediction patterns for investigation
- **Interactive Dashboard**: Streamlit-based interface with AI-powered data exploration

**What Makes This Different:**
Unlike traditional analytics that simply report what happened, this system actively searches for patterns, identifies risks you didn't know to look for, and explains its findings in actionable terms.

## Project Structure

```
surgery-analytics/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
│
├── Data Files (not in repo)
│   ├── df_cleaned.csv                 # Original dataset (2010-2025)
│   ├── df_cleaned_filtered.csv        # Filtered dataset (2012+, quality controlled)
│   └── df_cleaned_original_backup.csv # Backup of original
│
├── Core Analytics Modules
│   ├── shap_enhanced.py               # Enhanced SHAP data structure
│   │                                  # - EnhancedSHAPData class
│   │                                  # - Combines dual models with metadata
│   │                                  # - Temporal and team composition data
│   │                                  # - Save/load functionality
│   │
│   └── insight_engine.py              # Automated insight discovery
│                                      # - CrossSHAPAnalyzer: model divergence
│                                      # - AnomalyDetector: unusual patterns
│                                      # - DriftDetector: temporal changes
│                                      # - InsightEngine: unified insight generation
│
├── Dashboard Application
│   ├── surgical_dashboard.py          # Main Streamlit dashboard
│   │                                  # - Tab 1: Data Exploration (AI-powered)
│   │                                  # - Tab 2: SHAP Analysis (visualizations)
│   │                                  # - Tab 3: Insight Feed (automated discoveries)
│   │
│   └── data_explorer.py               # Basic data exploration tool
│
├── SHAP Data (not in repo)
│   ├── shap_data/                     # Original SHAP analysis (2010-2025)
│   ├── shap_data_enhanced/            # Enhanced format (2012+) ← Used by dashboard
│   │   ├── duration_shap.npy          # Duration model SHAP values
│   │   ├── overrun_shap.npy           # Overrun model SHAP values
│   │   ├── X_test.csv                 # Test features
│   │   ├── X_test_encoded.csv         # Encoded test features
│   │   ├── predicted_duration.npy     # Duration predictions
│   │   ├── predicted_overrun_prob.npy # Overrun probabilities
│   │   ├── actual_duration.npy        # Actual durations
│   │   ├── actual_overrun.npy         # Actual overrun flags
│   │   ├── years.npy                  # Temporal metadata
│   │   ├── seasons.npy
│   │   ├── weekdays.npy
│   │   ├── surgeons.npy               # Team composition
│   │   ├── consultants.npy
│   │   ├── anaesthetists.npy
│   │   ├── procedure_codes.npy        # Procedure information
│   │   ├── expected_lengths.npy
│   │   ├── ages.npy                   # Patient demographics
│   │   └── metadata.json              # Metadata and config
│   │
│   └── shap_data_enhanced_full/       # Full dataset backup (includes 2010-2011)
│
└── venv/                              # Python virtual environment (not in repo)
```

## Key Features by Module

### `shap_enhanced.py` - Enhanced SHAP Foundation
- **EnhancedSHAPData**: Unified data structure combining:
  - Dual SHAP values (duration + overrun models)
  - Temporal metadata (years, seasons, weekdays)
  - Team composition (surgeons, consultants, anaesthetists)
  - Procedure and patient information
  - Predictions and actuals
- **Migration utilities**: Convert legacy SHAP data to enhanced format
- **Data quality filtering**: Excludes 2010-2011 data (62.4% → 27.4% overrun rate)

### `insight_engine.py` - Insight Discovery Layer

**1. CrossSHAPAnalyzer**
- Finds features where duration vs overrun models diverge
- **Smoking Gun Finder**: Identifies hidden overrun risks
  - Low duration importance + high overrun importance = hidden risk
- Feature direction comparison (aligned vs opposed effects)

**2. AnomalyDetector**
- SHAP pattern anomalies using Isolation Forest
- Prediction outliers (cases where models were very wrong)
- Identifies data quality issues or special circumstances

**3. DriftDetector**
- Temporal changes in feature importance
- Rolling window analysis for trend detection
- Multi-period comparison for major shifts
- Example findings: 300-600% drift in comorbidity features

**4. InsightEngine**
- Coordinates all analyzers
- Generates prioritized, actionable insights
- Categories: smoking_gun, divergence, drift, anomaly
- Importance scoring and recommendations

### `surgical_dashboard.py` - Interactive Dashboard

**Tab 1: Data Exploration**
- AI-powered visualization generation (Google Gemini)
- Natural language queries → executable code
- Automatic chart generation

**Tab 2: SHAP Analysis**
- Global and procedure-specific analysis
- Feature importance visualizations
- Model performance metrics
- Dependence plots and interaction analysis
- Download SHAP values and predictions

**Tab 3: Insight Feed** ⭐ NEW
- Real-time automated insight discovery
- Smoking gun findings
- Temporal drift alerts
- Model divergence insights
- Anomaly detection results
- Filterable by category
- Expandable evidence details

## Getting Started

### Prerequisites
```bash
Python 3.10+
Virtual environment (venv)
```

### Installation
```bash
# Navigate to project directory
cd "surgery-analytics"

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install streamlit pandas numpy scipy scikit-learn matplotlib seaborn google-generativeai
```

### Running the Dashboard
```bash
streamlit run surgical_dashboard.py
```

The dashboard will open at `http://localhost:8501`

### Environment Variables
For AI-powered data exploration (optional):
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Data Quality Notes

**Filtered Years (2010-2011)**
- Original data: 94,446 rows (2010-2025)
- Filtered data: 79,621 rows (2012-2025)
- Reason: Recording practices and scheduling differed significantly
- Impact: Overrun rate dropped from 62.4% → 27.4%
- Files: `df_cleaned_filtered.csv` used by dashboard

**Enhanced SHAP Data**
- Location: `shap_data_enhanced/`
- Samples: 11,964 (test set from 2012-2025 data)
- Features: 25 features tracked across both models
- Format: Enhanced format with rich temporal/team metadata

## Example Insights Discovered

The system has automatically discovered:

1. **Smoking Gun: "Year"**
   - Low impact on duration prediction
   - High impact on overrun risk
   - Suggests systematic temporal changes in scheduling

2. **Massive Temporal Drift**
   - Cancer importance: +572% (duration), +586% (overrun)
   - Chronic Kidney Disease: +536% (duration), +570% (overrun)
   - Indicates major changes in recording practices

3. **Prediction Outliers**
   - 240 cases with unusual prediction errors
   - Maximum error: 292 minutes
   - Flagged for investigation

4. **Model Divergence**
   - "Intended_Management" shows strong alignment
   - "Year" shows moderate opposition (opposite effects)
   - "Theatre_Code" shows moderate alignment

## Technical Architecture

**Data Flow:**
```
Raw Data (CSV)
    ↓
SHAP Analysis (external)
    ↓
Enhanced SHAP Data (shap_enhanced.py)
    ↓
Insight Engine (insight_engine.py)
    ↓
Dashboard Visualization (surgical_dashboard.py)
```

**Key Design Principles:**
- **Modularity**: Each component is independent and reusable
- **Caching**: Streamlit caching for performance
- **Extensibility**: Easy to add new analyzers or insights
- **Transparency**: All insights include evidence and recommendations

## Future Enhancements

Planned features:
- **Performance Analytics**: Surgeon profiling and team dynamics analysis
- **Natural Language Interface**: AI-generated narrative reports
- **Temporal Trends Dashboard**: Interactive time-series analysis
- **Team Chemistry Scoring**: Quantify how well staff work together
- **Automated Alert System**: Real-time notifications for significant drift

## License

[Add your license here]

## Contact

[Add your contact information here]
