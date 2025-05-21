"""
Evaluation module for Named Entity Recognition model.
"""

import logging
import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.metrics import compute_ner_metrics, visualize_prediction

logger = logging.getLogger(__name__)


def evaluate_model(model, test_dataset, tag_vocab, output_dir=None):
    """
    Evaluate the NER model and save results.
    
    Args:
        model: Trained model to evaluate
        test_dataset: Test dataset for evaluation
        tag_vocab: Tag vocabulary
        output_dir: Directory to save evaluation results
        
    Returns:
        dict: Metrics dictionary
    """
    logger.info("Starting model evaluation")
    
    # Compute metrics
    metrics = compute_ner_metrics(model, test_dataset, tag_vocab)
    
    # Save metrics if output directory is provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
          # Save metrics as JSON
        metrics_file = output_path / "evaluation_metrics.json"
        
        # Helper function to convert numpy values to Python types
        def convert_numpy_types(obj):
            import numpy as np
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            else:
                return obj
        
        # Convert metrics to JSON-serializable format
        metrics_json = convert_numpy_types(metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Evaluation metrics saved to {metrics_file}")
          # Generate a report
        report_file = output_path / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write("# Named Entity Recognition Model Evaluation\n\n")
            f.write(f"## Overall Metrics\n\n")
            f.write(f"- F1 Score: {metrics.get('entity_f1', metrics.get('f1', 0.0)):.4f}\n")
            f.write(f"- Precision: {metrics.get('entity_precision', metrics.get('precision', 0.0)):.4f}\n")
            f.write(f"- Recall: {metrics.get('entity_recall', metrics.get('recall', 0.0)):.4f}\n\n")
            
            f.write(f"## Entity-Level Metrics\n\n")
            f.write("| Entity Type | Precision | Recall | F1-Score | Support |\n")
            f.write("|------------|-----------|--------|----------|--------|\n")
            
            entity_metrics_dict = metrics.get('entity_metrics', {})
            for entity, entity_metrics in entity_metrics_dict.items():
                if isinstance(entity_metrics, dict):
                    precision = entity_metrics.get('precision', 0.0)
                    recall = entity_metrics.get('recall', 0.0)
                    f1_score = entity_metrics.get('f1-score', 0.0)
                    support = entity_metrics.get('support', 0)
                    f.write(f"| {entity} | {precision:.4f} | {recall:.4f} | {f1_score:.4f} | {support} |\n")
        
        logger.info(f"Evaluation report saved to {report_file}")
    
    
    return metrics


def evaluate_on_examples(model, predictor, examples, tag_vocab, output_dir=None):
    """
    Evaluate model on specific examples and visualize results.
    
    Args:
        model: Trained model
        predictor: NERPredictor instance
        examples: List of example sentences or texts
        tag_vocab: Tag vocabulary
        output_dir: Directory to save visualization results
        
    Returns:
        list: List of visualization DataFrames
    """
    logger.info(f"Evaluating on {len(examples)} example texts")
    
    visualizations = []
    
    for i, text in enumerate(examples):
        # Make prediction
        tokens, predicted_tags, entities = predictor.predict(text)
        
        # Create visualization
        visualization = predictor.visualize_entities(tokens, predicted_tags, entities)
        visualizations.append(visualization)
        
        # Save visualization if output directory is provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV
            visualization.to_csv(output_path / f"example_{i+1}.csv", index=False)
            
            # Save entities as JSON
            with open(output_path / f"entities_{i+1}.json", 'w') as f:
                json.dump(entities, f, indent=2, default=str)
            
            logger.info(f"Example {i+1} visualization saved to {output_path}")
    
    return visualizations
