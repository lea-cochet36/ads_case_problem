from kedro.pipeline import node, pipeline

from .nodes import preprocess_data, train_model_arima, predict_model, postprocess_data

def create_preprocessing_pipeline():
    return pipeline(
        [
            node(
                preprocess_data,
                inputs=dict(data="train"),
                outputs="train_preprocessed",
            )
        ]
    )

def create_training_pipeline():
    return pipeline(
        [
            node(
                train_model_arima,
                inputs=dict(data= "train_preprocessed", stationary="params:stationary", test="params:test", sp="params:sp"),
                outputs="model_trained",
            )
        ]
    )


def create_inference_pipeline():
    return pipeline(
        [
            node(
                predict_model,
                dict(model="model_trained",horizon="params:horizon"),
                outputs = "y_pred"
            ),
        ]
    )

def create_postprocessing_pipeline():
    return pipeline(
        [
            node(
                postprocess_data,
                dict(y_pred="y_pred"),
                outputs = "test"
            )
        ]
    )
