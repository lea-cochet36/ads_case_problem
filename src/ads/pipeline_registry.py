"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline


from ads.pipelines.ads_data_science.pipeline import create_preprocessing_pipeline, create_training_pipeline, create_inference_pipeline, create_postprocessing_pipeline, create_evaluation_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    preprocessing_pipeline = create_preprocessing_pipeline()
    training_pipeline = create_training_pipeline()
    inference_pipeline = create_inference_pipeline()
    postprocessing_pipeline = create_postprocessing_pipeline()
    evaluation_pipeline = create_evaluation_pipeline()

    return {
        "preprocessing_pipeline" : preprocessing_pipeline,
        "training_pipeline" : training_pipeline,
        "inference_pipeline"  : inference_pipeline,
        "evaluation_pipeline" : evaluation_pipeline,
        "user_app" : inference_pipeline + postprocessing_pipeline
    }
