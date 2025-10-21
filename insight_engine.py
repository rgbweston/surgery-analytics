"""
Insight Discovery Layer for Surgical Intelligence Engine
Automatically discovers patterns, anomalies, and temporal drift in SHAP data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from scipy import stats
from sklearn.ensemble import IsolationForest
from shap_enhanced import EnhancedSHAPData


@dataclass
class Insight:
    """Represents a discovered insight"""
    title: str
    description: str
    importance: float  # 0-1 score
    category: str  # "divergence", "anomaly", "drift", "smoking_gun"
    evidence: Dict  # Supporting data
    recommendation: Optional[str] = None


class CrossSHAPAnalyzer:
    """
    Analyzes where duration and overrun models diverge
    These are the most interesting cases!
    """

    def __init__(self, enhanced_data: EnhancedSHAPData):
        self.data = enhanced_data

    def find_divergent_features(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find features where duration and overrun models have different importance

        Returns:
            List of (feature_name, divergence_score) tuples
        """
        # Mean absolute SHAP for each model
        duration_importance = np.abs(self.data.duration_shap).mean(axis=0)
        overrun_importance = np.abs(self.data.overrun_shap).mean(axis=0)

        # Normalize to 0-1
        duration_norm = duration_importance / (duration_importance.max() + 1e-10)
        overrun_norm = overrun_importance / (overrun_importance.max() + 1e-10)

        # Calculate divergence (absolute difference)
        divergence = np.abs(duration_norm - overrun_norm)

        # Get top divergent features
        top_indices = np.argsort(divergence)[-top_n:][::-1]

        results = []
        for idx in top_indices:
            feature_name = self.data.feature_names[idx]
            div_score = divergence[idx]
            results.append((feature_name, div_score))

        return results

    def find_smoking_guns(self, threshold: float = 0.2) -> List[Dict]:
        """
        Find "smoking gun" features:
        - Low importance for duration (not expected to matter much)
        - High importance for overrun (actually causes delays)

        These are hidden risk factors!

        Args:
            threshold: Minimum importance difference to consider

        Returns:
            List of smoking gun insights
        """
        duration_importance = np.abs(self.data.duration_shap).mean(axis=0)
        overrun_importance = np.abs(self.data.overrun_shap).mean(axis=0)

        # Normalize
        duration_norm = duration_importance / (duration_importance.max() + 1e-10)
        overrun_norm = overrun_importance / (overrun_importance.max() + 1e-10)

        # Find features where overrun >> duration
        smoking_guns = []
        for idx, feature_name in enumerate(self.data.feature_names):
            dur_imp = duration_norm[idx]
            ovr_imp = overrun_norm[idx]

            # Smoking gun criteria: high overrun importance, low duration importance
            if ovr_imp > 0.3 and (ovr_imp - dur_imp) > threshold:
                smoking_guns.append({
                    'feature': feature_name,
                    'duration_importance': float(dur_imp),
                    'overrun_importance': float(ovr_imp),
                    'ratio': float(ovr_imp / (dur_imp + 0.01)),
                    'interpretation': f"{feature_name} has low impact on total duration but high impact on overrun risk"
                })

        # Sort by ratio
        smoking_guns.sort(key=lambda x: x['ratio'], reverse=True)

        return smoking_guns

    def compare_feature_directions(self, feature_name: str) -> Dict:
        """
        Compare how a feature affects duration vs overrun

        Args:
            feature_name: Name of the feature to analyze

        Returns:
            Comparison statistics
        """
        if feature_name not in self.data.feature_names:
            raise ValueError(f"Feature {feature_name} not found")

        feature_idx = self.data.feature_names.index(feature_name)

        duration_shap = self.data.duration_shap[:, feature_idx]
        overrun_shap = self.data.overrun_shap[:, feature_idx]

        # Calculate correlation between SHAP values
        correlation = np.corrcoef(duration_shap, overrun_shap)[0, 1]

        return {
            'feature': feature_name,
            'duration_mean_shap': float(duration_shap.mean()),
            'overrun_mean_shap': float(overrun_shap.mean()),
            'correlation': float(correlation),
            'interpretation': self._interpret_correlation(correlation)
        }

    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation between duration and overrun SHAP"""
        if corr > 0.7:
            return "Feature affects duration and overrun in the same direction (strongly aligned)"
        elif corr > 0.3:
            return "Feature affects duration and overrun in similar direction (moderately aligned)"
        elif corr > -0.3:
            return "Feature affects duration and overrun independently (no clear pattern)"
        elif corr > -0.7:
            return "Feature affects duration and overrun in opposite directions (moderately opposed)"
        else:
            return "Feature affects duration and overrun in opposite directions (strongly opposed)"


class AnomalyDetector:
    """
    Detects unusual SHAP patterns that might indicate data quality issues
    or special cases
    """

    def __init__(self, enhanced_data: EnhancedSHAPData):
        self.data = enhanced_data

    def detect_shap_anomalies(self, contamination: float = 0.05, model_type: str = 'duration') -> np.ndarray:
        """
        Detect cases with unusual SHAP patterns

        Args:
            contamination: Expected proportion of anomalies
            model_type: 'duration' or 'overrun'

        Returns:
            Boolean array indicating anomalies
        """
        shap_values = self.data.duration_shap if model_type == 'duration' else self.data.overrun_shap

        # Use Isolation Forest
        detector = IsolationForest(contamination=contamination, random_state=42)
        predictions = detector.fit_predict(shap_values)

        # -1 = anomaly, 1 = normal
        is_anomaly = predictions == -1

        return is_anomaly

    def find_prediction_outliers(self, threshold: float = 3.0) -> Dict:
        """
        Find cases where predictions are unusually far from actual

        Args:
            threshold: Number of standard deviations to consider outlier

        Returns:
            Dictionary with outlier information
        """
        # Duration outliers
        duration_errors = self.data.predicted_duration - self.data.actual_duration
        duration_std = duration_errors.std()
        duration_mean = duration_errors.mean()
        duration_outliers = np.abs(duration_errors - duration_mean) > (threshold * duration_std)

        # Overrun probability outliers (cases where we were very wrong)
        overrun_errors = self.data.predicted_overrun_prob - self.data.actual_overrun
        overrun_outliers = np.abs(overrun_errors) > 0.8  # Very confident but wrong

        return {
            'duration_outliers': {
                'count': int(duration_outliers.sum()),
                'indices': np.where(duration_outliers)[0],
                'mean_error': float(duration_errors[duration_outliers].mean()),
                'max_error': float(np.abs(duration_errors[duration_outliers]).max())
            },
            'overrun_outliers': {
                'count': int(overrun_outliers.sum()),
                'indices': np.where(overrun_outliers)[0],
                'mean_confidence': float(np.abs(overrun_errors[overrun_outliers]).mean())
            }
        }


class DriftDetector:
    """
    Detects temporal changes in feature importance and patterns
    """

    def __init__(self, enhanced_data: EnhancedSHAPData):
        self.data = enhanced_data

    def detect_feature_drift(self, feature_name: str, window_size: int = 1000) -> Dict:
        """
        Detect if a feature's importance is changing over time

        Args:
            feature_name: Name of the feature
            window_size: Size of rolling window for analysis

        Returns:
            Drift statistics
        """
        if feature_name not in self.data.feature_names:
            raise ValueError(f"Feature {feature_name} not found")

        feature_idx = self.data.feature_names.index(feature_name)

        # Sort by year
        sort_idx = np.argsort(self.data.years)
        sorted_years = self.data.years[sort_idx]
        sorted_duration_shap = self.data.duration_shap[sort_idx, feature_idx]
        sorted_overrun_shap = self.data.overrun_shap[sort_idx, feature_idx]

        # Calculate rolling mean absolute SHAP
        duration_rolling = pd.Series(np.abs(sorted_duration_shap)).rolling(window=window_size, min_periods=100).mean()
        overrun_rolling = pd.Series(np.abs(sorted_overrun_shap)).rolling(window=window_size, min_periods=100).mean()

        # Calculate trend (simple linear regression on rolling means)
        valid_idx = ~duration_rolling.isna()
        if valid_idx.sum() > 10:
            x = np.arange(len(duration_rolling))[valid_idx]
            y_dur = duration_rolling[valid_idx].values
            y_ovr = overrun_rolling[valid_idx].values

            # Linear trend
            duration_trend = np.polyfit(x, y_dur, 1)[0] if len(x) > 0 else 0
            overrun_trend = np.polyfit(x, y_ovr, 1)[0] if len(x) > 0 else 0
        else:
            duration_trend = 0
            overrun_trend = 0

        return {
            'feature': feature_name,
            'duration_trend': float(duration_trend),
            'overrun_trend': float(overrun_trend),
            'duration_trend_interpretation': self._interpret_trend(duration_trend),
            'overrun_trend_interpretation': self._interpret_trend(overrun_trend),
            'has_significant_drift': abs(duration_trend) > 1e-5 or abs(overrun_trend) > 1e-5
        }

    def detect_temporal_shifts(self, n_periods: int = 4) -> List[Dict]:
        """
        Detect major shifts in SHAP patterns across time periods

        Args:
            n_periods: Number of time periods to divide data into

        Returns:
            List of detected shifts
        """
        unique_years = np.unique(self.data.years)
        period_size = len(unique_years) // n_periods

        shifts = []

        for feat_idx, feature_name in enumerate(self.data.feature_names):
            period_importances_dur = []
            period_importances_ovr = []

            for period in range(n_periods):
                start_year = unique_years[period * period_size]
                end_year = unique_years[min((period + 1) * period_size, len(unique_years) - 1)]

                mask = (self.data.years >= start_year) & (self.data.years <= end_year)

                if mask.sum() > 0:
                    dur_imp = np.abs(self.data.duration_shap[mask, feat_idx]).mean()
                    ovr_imp = np.abs(self.data.overrun_shap[mask, feat_idx]).mean()

                    period_importances_dur.append(dur_imp)
                    period_importances_ovr.append(ovr_imp)

            # Check for significant change
            if len(period_importances_dur) >= 2:
                dur_change = (period_importances_dur[-1] - period_importances_dur[0]) / (period_importances_dur[0] + 1e-10)
                ovr_change = (period_importances_ovr[-1] - period_importances_ovr[0]) / (period_importances_ovr[0] + 1e-10)

                if abs(dur_change) > 0.5 or abs(ovr_change) > 0.5:  # 50% change threshold
                    shifts.append({
                        'feature': feature_name,
                        'duration_change_pct': float(dur_change * 100),
                        'overrun_change_pct': float(ovr_change * 100),
                        'significance': 'high' if abs(dur_change) > 1.0 or abs(ovr_change) > 1.0 else 'moderate'
                    })

        # Sort by absolute change
        shifts.sort(key=lambda x: abs(x['duration_change_pct']) + abs(x['overrun_change_pct']), reverse=True)

        return shifts

    def _interpret_trend(self, trend: float) -> str:
        """Interpret trend direction"""
        if trend > 1e-5:
            return "Increasing importance over time"
        elif trend < -1e-5:
            return "Decreasing importance over time"
        else:
            return "Stable importance over time"


class InsightEngine:
    """
    Main insight generation engine that coordinates all analyzers
    """

    def __init__(self, enhanced_data: EnhancedSHAPData):
        self.data = enhanced_data
        self.cross_shap = CrossSHAPAnalyzer(enhanced_data)
        self.anomaly = AnomalyDetector(enhanced_data)
        self.drift = DriftDetector(enhanced_data)

    def generate_all_insights(self, max_insights: int = 20) -> List[Insight]:
        """
        Generate comprehensive insights from all analyzers

        Args:
            max_insights: Maximum number of insights to return

        Returns:
            List of insights sorted by importance
        """
        insights = []

        # 1. Smoking guns (highest priority!)
        smoking_guns = self.cross_shap.find_smoking_guns(threshold=0.2)
        for sg in smoking_guns[:5]:  # Top 5 smoking guns
            insights.append(Insight(
                title=f"Hidden Overrun Risk: {sg['feature']}",
                description=sg['interpretation'],
                importance=min(sg['ratio'] / 10, 1.0),  # Normalize
                category="smoking_gun",
                evidence=sg,
                recommendation=f"Monitor {sg['feature']} closely for overrun risk, even if duration predictions seem normal"
            ))

        # 2. Cross-SHAP divergence
        divergent = self.cross_shap.find_divergent_features(top_n=5)
        for feature, div_score in divergent:
            comparison = self.cross_shap.compare_feature_directions(feature)
            insights.append(Insight(
                title=f"Model Divergence: {feature}",
                description=f"{feature} affects duration and overrun differently. {comparison['interpretation']}",
                importance=float(div_score),
                category="divergence",
                evidence=comparison
            ))

        # 3. Temporal drift
        shifts = self.drift.detect_temporal_shifts(n_periods=4)
        for shift in shifts[:5]:
            insights.append(Insight(
                title=f"Temporal Shift: {shift['feature']}",
                description=f"{shift['feature']} importance changed by {shift['duration_change_pct']:.1f}% (duration) and {shift['overrun_change_pct']:.1f}% (overrun)",
                importance=min((abs(shift['duration_change_pct']) + abs(shift['overrun_change_pct'])) / 200, 1.0),
                category="drift",
                evidence=shift,
                recommendation=f"Investigate why {shift['feature']} has changed in importance over time"
            ))

        # 4. Prediction anomalies
        outliers = self.anomaly.find_prediction_outliers(threshold=3.0)
        if outliers['duration_outliers']['count'] > 0:
            insights.append(Insight(
                title="Duration Prediction Outliers Detected",
                description=f"Found {outliers['duration_outliers']['count']} cases with unusual prediction errors (max error: {outliers['duration_outliers']['max_error']:.0f} min)",
                importance=min(outliers['duration_outliers']['count'] / 100, 1.0),
                category="anomaly",
                evidence=outliers['duration_outliers'],
                recommendation="Review these cases for data quality issues or special circumstances"
            ))

        # Sort by importance
        insights.sort(key=lambda x: x.importance, reverse=True)

        return insights[:max_insights]

    def get_insight_summary(self) -> Dict:
        """Get high-level summary of insights"""
        insights = self.generate_all_insights()

        return {
            'total_insights': len(insights),
            'smoking_guns': len([i for i in insights if i.category == 'smoking_gun']),
            'divergences': len([i for i in insights if i.category == 'divergence']),
            'drift_detected': len([i for i in insights if i.category == 'drift']),
            'anomalies': len([i for i in insights if i.category == 'anomaly']),
            'top_insight': insights[0].title if insights else None
        }


if __name__ == "__main__":
    print("Insight Discovery Engine")
    print("=" * 60)

    # Load enhanced SHAP data
    enhanced_data = EnhancedSHAPData.load("shap_data_enhanced")
    print(f"Loaded {enhanced_data.n_samples:,} samples from {enhanced_data.years.min():.0f}-{enhanced_data.years.max():.0f}")

    # Create insight engine
    engine = InsightEngine(enhanced_data)

    # Generate insights
    print("\nGenerating insights...")
    insights = engine.generate_all_insights(max_insights=10)

    print(f"\nFound {len(insights)} insights:\n")

    for i, insight in enumerate(insights, 1):
        print(f"{i}. [{insight.category.upper()}] {insight.title}")
        print(f"   Importance: {insight.importance:.2f}")
        print(f"   {insight.description}")
        if insight.recommendation:
            print(f"   → Recommendation: {insight.recommendation}")
        print()

    # Summary
    summary = engine.get_insight_summary()
    print("=" * 60)
    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
