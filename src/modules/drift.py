import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import logging

logger = logging.getLogger(__name__)


def detect_drift(reference_data: pd.DataFrame, 
                 current_data: pd.DataFrame,
                 numerical_column: str,
                 categorical_column: str) -> dict:
    
    """

    """
    
    columns = [numerical_column, categorical_column]
    
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    report.run(
        reference_data=reference_data[columns],
        current_data=current_data[columns]
    )
    
    results = report.dict()
    
    drift_summary = {
        "numerical_column": numerical_column,
        "categorical_column": categorical_column,
        "dataset_drift_detected": results["metrics"][0]["result"]["dataset_drift"],
        "number_of_drifted_columns": results["metrics"][0]["result"]["number_of_drifted_columns"],
        "share_of_drifted_columns": results["metrics"][0]["result"]["share_of_drifted_columns"],
    }
    
    return drift_summary


def save_drift_report(reference_data: pd.DataFrame, 
                      current_data: pd.DataFrame,
                      output_path: str,
                      numerical_column: str,
                      categorical_column: str):
    """

    """
    
    columns = [numerical_column, categorical_column]
    
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    report.run(
        reference_data=reference_data[columns],
        current_data=current_data[columns]
    )
    
    report.save_html(output_path)
    logger.info(f"HTML report saved to {output_path}")
