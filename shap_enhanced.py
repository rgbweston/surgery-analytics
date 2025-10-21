"""
Enhanced SHAP Data Structure for Surgical Intelligence Engine
Combines dual-model SHAP analysis with temporal and team metadata
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class EnhancedSHAPData:
    """
    Unified data structure for comprehensive SHAP analysis
    Combines both models with rich metadata
    """
    # SHAP values
    duration_shap: np.ndarray  # Regression SHAP values
    overrun_shap: np.ndarray   # Classification SHAP values

    # Feature information
    feature_names: List[str]
    X_test: pd.DataFrame       # Original features
    X_test_encoded: pd.DataFrame  # Encoded features

    # Predictions
    predicted_duration: np.ndarray
    predicted_overrun_prob: np.ndarray
    actual_duration: np.ndarray
    actual_overrun: np.ndarray

    # Expected values (baselines)
    expected_duration: float
    expected_overrun_prob: float

    # Temporal metadata
    years: np.ndarray
    seasons: np.ndarray
    weekdays: np.ndarray

    # Team composition
    surgeons: np.ndarray
    consultants: np.ndarray
    anaesthetists: np.ndarray

    # Procedure information
    procedure_codes: np.ndarray
    expected_lengths: np.ndarray

    # Patient demographics
    ages: np.ndarray

    # Optional fields
    timestamps: Optional[np.ndarray] = None  # If available

    # Metadata
    scope: str = "global"  # "global" or procedure code
    n_samples: int = 0
    generated_at: str = ""

    def __post_init__(self):
        if self.n_samples == 0:
            self.n_samples = len(self.duration_shap)
        if self.generated_at == "":
            self.generated_at = datetime.now().isoformat()

    def save(self, directory: Path):
        """Save enhanced SHAP data to directory"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save numpy arrays
        np.save(directory / "duration_shap.npy", self.duration_shap)
        np.save(directory / "overrun_shap.npy", self.overrun_shap)
        np.save(directory / "predicted_duration.npy", self.predicted_duration)
        np.save(directory / "predicted_overrun_prob.npy", self.predicted_overrun_prob)
        np.save(directory / "actual_duration.npy", self.actual_duration)
        np.save(directory / "actual_overrun.npy", self.actual_overrun)

        np.save(directory / "years.npy", self.years)
        np.save(directory / "seasons.npy", self.seasons)
        np.save(directory / "weekdays.npy", self.weekdays)
        if self.timestamps is not None:
            np.save(directory / "timestamps.npy", self.timestamps)

        np.save(directory / "surgeons.npy", self.surgeons)
        np.save(directory / "consultants.npy", self.consultants)
        np.save(directory / "anaesthetists.npy", self.anaesthetists)

        np.save(directory / "procedure_codes.npy", self.procedure_codes)
        np.save(directory / "expected_lengths.npy", self.expected_lengths)
        np.save(directory / "ages.npy", self.ages)

        # Save DataFrames
        self.X_test.to_csv(directory / "X_test.csv")
        self.X_test_encoded.to_csv(directory / "X_test_encoded.csv")

        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "expected_duration": float(self.expected_duration),
            "expected_overrun_prob": float(self.expected_overrun_prob),
            "scope": self.scope,
            "n_samples": int(self.n_samples),
            "generated_at": self.generated_at
        }

        with open(directory / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, directory: Path) -> 'EnhancedSHAPData':
        """Load enhanced SHAP data from directory"""
        directory = Path(directory)

        # Load metadata
        with open(directory / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Load numpy arrays
        duration_shap = np.load(directory / "duration_shap.npy")
        overrun_shap = np.load(directory / "overrun_shap.npy")
        predicted_duration = np.load(directory / "predicted_duration.npy")
        predicted_overrun_prob = np.load(directory / "predicted_overrun_prob.npy")
        actual_duration = np.load(directory / "actual_duration.npy")
        actual_overrun = np.load(directory / "actual_overrun.npy")

        years = np.load(directory / "years.npy")
        seasons = np.load(directory / "seasons.npy", allow_pickle=True)
        weekdays = np.load(directory / "weekdays.npy", allow_pickle=True)
        timestamps = None
        if (directory / "timestamps.npy").exists():
            timestamps = np.load(directory / "timestamps.npy")

        surgeons = np.load(directory / "surgeons.npy", allow_pickle=True)
        consultants = np.load(directory / "consultants.npy", allow_pickle=True)
        anaesthetists = np.load(directory / "anaesthetists.npy", allow_pickle=True)

        procedure_codes = np.load(directory / "procedure_codes.npy", allow_pickle=True)
        expected_lengths = np.load(directory / "expected_lengths.npy")
        ages = np.load(directory / "ages.npy")

        # Load DataFrames
        X_test = pd.read_csv(directory / "X_test.csv", index_col=0)
        X_test_encoded = pd.read_csv(directory / "X_test_encoded.csv", index_col=0)

        return cls(
            duration_shap=duration_shap,
            overrun_shap=overrun_shap,
            feature_names=metadata["feature_names"],
            X_test=X_test,
            X_test_encoded=X_test_encoded,
            predicted_duration=predicted_duration,
            predicted_overrun_prob=predicted_overrun_prob,
            actual_duration=actual_duration,
            actual_overrun=actual_overrun,
            expected_duration=metadata["expected_duration"],
            expected_overrun_prob=metadata["expected_overrun_prob"],
            years=years,
            seasons=seasons,
            weekdays=weekdays,
            timestamps=timestamps,
            surgeons=surgeons,
            consultants=consultants,
            anaesthetists=anaesthetists,
            procedure_codes=procedure_codes,
            expected_lengths=expected_lengths,
            ages=ages,
            scope=metadata["scope"],
            n_samples=metadata["n_samples"],
            generated_at=metadata["generated_at"]
        )


def migrate_existing_shap_data(shap_dir: str, df: pd.DataFrame) -> EnhancedSHAPData:
    """
    Migrate existing SHAP data to enhanced format

    Args:
        shap_dir: Directory containing existing SHAP data
        df: Original dataframe with all columns

    Returns:
        EnhancedSHAPData object
    """
    shap_dir = Path(shap_dir)

    # Load existing SHAP data
    duration_shap = np.load(shap_dir / "shap_values_regression.npy")
    overrun_shap = np.load(shap_dir / "shap_values_classification.npy")
    X_test = pd.read_csv(shap_dir / "X_test.csv", index_col=0)
    X_test_encoded = pd.read_csv(shap_dir / "X_test_encoded.csv", index_col=0)
    predictions = pd.read_csv(shap_dir / "predictions.csv", index_col=0)

    with open(shap_dir / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    with open(shap_dir / "expected_values.pkl", "rb") as f:
        expected_values = pickle.load(f)

    # Extract temporal and team metadata from X_test
    # Match X_test indices to original dataframe
    df_matched = df.loc[X_test.index]

    # Extract metadata
    years = df_matched['Year'].values if 'Year' in df_matched.columns else np.zeros(len(X_test))
    seasons = df_matched['Season'].values if 'Season' in df_matched.columns else np.array(['Unknown'] * len(X_test))
    weekdays = df_matched['weekday_name'].values if 'weekday_name' in df_matched.columns else np.array(['Unknown'] * len(X_test))

    surgeons = df_matched['Pseudo_Surgeon'].values if 'Pseudo_Surgeon' in df_matched.columns else np.array(['Unknown'] * len(X_test))
    consultants = df_matched['Pseudo_Consultant'].values if 'Pseudo_Consultant' in df_matched.columns else np.array(['Unknown'] * len(X_test))
    anaesthetists = df_matched['Pseudo_Anaesthetist'].values if 'Pseudo_Anaesthetist' in df_matched.columns else np.array(['Unknown'] * len(X_test))

    procedure_codes = df_matched['Proc_Code_1_Read'].values if 'Proc_Code_1_Read' in df_matched.columns else np.array(['Unknown'] * len(X_test))
    expected_lengths = df_matched['Expected_Length'].values if 'Expected_Length' in df_matched.columns else np.zeros(len(X_test))
    ages = df_matched['Operation_Age'].values if 'Operation_Age' in df_matched.columns else np.zeros(len(X_test))

    # Extract predictions
    predicted_duration = predictions['Pred_Duration'].values
    predicted_overrun_prob = predictions['Overrun_Prob'].values
    actual_duration = predictions['Actual_Length'].values
    actual_overrun = predictions['Overrun_Actual'].values

    return EnhancedSHAPData(
        duration_shap=duration_shap,
        overrun_shap=overrun_shap,
        feature_names=feature_names,
        X_test=X_test,
        X_test_encoded=X_test_encoded,
        predicted_duration=predicted_duration,
        predicted_overrun_prob=predicted_overrun_prob,
        actual_duration=actual_duration,
        actual_overrun=actual_overrun,
        expected_duration=expected_values['regression'],
        expected_overrun_prob=expected_values['classification'],
        years=years,
        seasons=seasons,
        weekdays=weekdays,
        timestamps=None,
        surgeons=surgeons,
        consultants=consultants,
        anaesthetists=anaesthetists,
        procedure_codes=procedure_codes,
        expected_lengths=expected_lengths,
        ages=ages,
        scope="global"
    )


if __name__ == "__main__":
    # Example: migrate existing SHAP data
    # NOTE: The existing SHAP data was generated from df_cleaned.csv (includes 2010-2011)
    # We use the original for migration, but will regenerate SHAP with filtered data later
    print("Enhanced SHAP Data Structure Module")
    print("Migrating existing SHAP data...")

    df = pd.read_csv("df_cleaned.csv")  # Use original for migration
    enhanced_data = migrate_existing_shap_data("shap_data", df)

    print(f"Migrated {enhanced_data.n_samples:,} samples")
    print(f"Features: {len(enhanced_data.feature_names)}")
    print(f"Date range: {enhanced_data.years.min():.0f} - {enhanced_data.years.max():.0f}")

    # Filter out 2010-2011 from enhanced data
    mask_2012_plus = enhanced_data.years >= 2012
    n_filtered = mask_2012_plus.sum()
    print(f"\nFiltering to 2012+: {n_filtered:,} samples (removed {(~mask_2012_plus).sum():,})")

    # Create filtered version
    filtered_data = EnhancedSHAPData(
        duration_shap=enhanced_data.duration_shap[mask_2012_plus],
        overrun_shap=enhanced_data.overrun_shap[mask_2012_plus],
        feature_names=enhanced_data.feature_names,
        X_test=enhanced_data.X_test[mask_2012_plus],
        X_test_encoded=enhanced_data.X_test_encoded[mask_2012_plus],
        predicted_duration=enhanced_data.predicted_duration[mask_2012_plus],
        predicted_overrun_prob=enhanced_data.predicted_overrun_prob[mask_2012_plus],
        actual_duration=enhanced_data.actual_duration[mask_2012_plus],
        actual_overrun=enhanced_data.actual_overrun[mask_2012_plus],
        expected_duration=enhanced_data.expected_duration,
        expected_overrun_prob=enhanced_data.expected_overrun_prob,
        years=enhanced_data.years[mask_2012_plus],
        seasons=enhanced_data.seasons[mask_2012_plus],
        weekdays=enhanced_data.weekdays[mask_2012_plus],
        surgeons=enhanced_data.surgeons[mask_2012_plus],
        consultants=enhanced_data.consultants[mask_2012_plus],
        anaesthetists=enhanced_data.anaesthetists[mask_2012_plus],
        procedure_codes=enhanced_data.procedure_codes[mask_2012_plus],
        expected_lengths=enhanced_data.expected_lengths[mask_2012_plus],
        ages=enhanced_data.ages[mask_2012_plus],
        scope="global_filtered_2012plus"
    )

    # Save both versions
    enhanced_data.save("shap_data_enhanced_full")
    print("Saved full data to: shap_data_enhanced_full/")

    filtered_data.save("shap_data_enhanced")
    print(f"Saved filtered (2012+) data to: shap_data_enhanced/")
    print(f"Date range: {filtered_data.years.min():.0f} - {filtered_data.years.max():.0f}")
