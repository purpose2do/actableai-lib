def transform_features(predictor, model_name, data):
    """
    TODO write documentation
    """
    autogluon_model = predictor._trainer.load_model(model_name)

    df_transformed_data = predictor.transform_features(data, model=model_name)
    transformed_data = autogluon_model.preprocess(df_transformed_data)

    return df_transformed_data, transformed_data


def _get_features_to_drop(autogluon_model):
    """
    TODO write documentation
    """
    if autogluon_model._is_features_in_same_as_ex:
        return {}

    return set(autogluon_model.features).difference(set(autogluon_model._features))


def _get_xgboost_feature_links(xgboost_model):
    """
    TODO write documentation
    """
    ohe_generator = xgboost_model._ohe_generator

    feature_links = {}

    # Add cat cols
    for feature, categories in ohe_generator.labels.items():
        feature_links[feature] = []
        for category in categories:
            feature_links[feature].append(f"{feature}_{category}")

    # Add other cols
    for feature in ohe_generator.other_cols:
        feature_links[feature] = [feature]

    return feature_links


def get_feature_links(predictor, model_name):
    """
    TODO write documentation
    """
    autogluon_model = predictor._trainer.load_model(model_name)
    feature_generator = predictor._learner.feature_generator

    # First preprocessing level links
    first_feature_links = feature_generator.get_feature_links()

    # Second preprocessing level links
    second_feature_links = None
    features_to_drop = []
    if model_name == "XGBoost":
        second_feature_links = _get_xgboost_feature_links(autogluon_model)
    else:
        features_to_drop = _get_features_to_drop(autogluon_model)

    # Merge layers
    feature_links = {}
    if second_feature_links is None:
        feature_links = first_feature_links
    else:
        for feature in first_feature_links.keys():
            feature_links[feature] = []
            for first_link in first_feature_links[feature]:
                if first_link in second_feature_links:
                    second_links = second_feature_links[first_link]
                    feature_links[feature] += second_links

    for feature in feature_links.keys():
        feature_links[feature] = [
            link for link in feature_links[feature] if link not in features_to_drop
        ]

    return feature_links


def _get_xgboost_final_features(xgboost_model):
    """
    TODO write documentation
    """
    return list(xgboost_model._ohe_generator._feature_map.keys())


def get_final_features(predictor, model_name):
    """
    TODO write documentation
    """
    autogluon_model = predictor._trainer.load_model(model_name)
    feature_generator = predictor._learner.feature_generator

    final_features = None
    if model_name == "XGBoost":
        final_features = _get_xgboost_final_features(autogluon_model)

    if final_features is None:
        final_features = autogluon_model._features

    return final_features
