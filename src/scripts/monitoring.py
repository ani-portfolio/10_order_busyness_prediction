import pandas as pd
import json
from datetime import datetime
from src.modules import drift
from src import config
import logging

logger = logging.getLogger(__name__)

def main():
    
    logger.info("Loading data...")
    reference_data = pd.read_csv(config.training_input_data )
    current_data = pd.read_csv(config.inference_input_data)
    
    logger.info(f"Reference data shape: {reference_data.shape}")
    logger.info(f"Current data shape: {current_data.shape}")
    
    # Detect drift
    logger.info("\nDetecting drift...")
    drift_results = drift.detect_drift(
        reference_data=reference_data,
        current_data=current_data,
        numerical_column="dist_to_restaurant",
        categorical_column="Five_Clusters_embedding"
    )
    
    # Add timestamp
    drift_results["timestamp"] = datetime.utcnow().isoformat()
    
    # logger.info results
    logger.info("\n" + "="*50)
    logger.info("DRIFT DETECTION RESULTS")
    logger.info("="*50)
    logger.info(f"Monitoring columns: {drift_results['numerical_column']}, {drift_results['categorical_column']}")
    logger.info(f"Dataset drift detected: {drift_results['dataset_drift_detected']}")
    logger.info(f"Number of drifted columns: {drift_results['number_of_drifted_columns']}")
    logger.info(f"Share of drifted columns: {drift_results['share_of_drifted_columns']:.2%}")
    logger.info("="*50)
    
    # Save metrics
    with open(config.metrics_output_path, 'w') as f:
        json.dump(drift_results, f, indent=2)
    logger.info(f"\nMetrics saved to {config.metrics_output_path}")
    
    # Generate HTML report
    logger.info("\nGenerating HTML report...")
    drift.save_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        output_path=config.report_output_path,
        numerical_column="dist_to_restaurant",
        categorical_column="Five_Clusters_embedding"
    )
    
    # Alert if drift detected
    if drift_results['dataset_drift_detected']:
        logger.info("\n⚠️  ALERT: Data drift detected!")
        # In production: send alert to Slack/email/PagerDuty
    else:
        logger.info("\n✅ No drift detected")


if __name__ == "__main__":
    main()