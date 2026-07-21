# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest classifier built with scikit-learn using the default hyperparameters and a fixed random state of 42 for reproducibility. It was developed as part of a project to deploy a machine learning pipeline with FastAPI. The model predicts whether a person's annual income is above or below $50,000 based on U.S. Census demographic data. Categorical features are encoded with a one-hot encoder and the target label is processed with a label binarizer.

## Intended Use

The model is intended to predict whether an individual earns more than $50,000 per year from census-style demographic attributes such as age, education, occupation, and hours worked per week. It is meant for educational and demonstration purposes as part of a deployed inference API. It is not intended for real-world decisions about individuals, such as lending, hiring, or benefits eligibility.

## Training Data

The training data comes from the publicly available UCI Census Income dataset (also known as the Adult dataset), extracted from the 1994 U.S. Census database. The full dataset contains 32,561 rows and 15 columns, including 8 categorical features and several continuous features, with salary as the target label. The data was split into 80 percent for training and 20 percent for testing. The training portion was processed with a one-hot encoder for the categorical features and a label binarizer for the target.

## Evaluation Data

The evaluation data is the 20 percent test split held out from the same census dataset. It was processed using the encoder and label binarizer that were fit on the training data, so the test data is transformed in the same way without leaking information from the test set into training.

## Metrics

The model was evaluated using precision, recall, and F1 score. On the held-out test set the model achieved the following:

- Precision: 0.7419
- Recall: 0.6384
- F1: 0.6863

Performance was also computed on slices of the data, holding one categorical feature fixed at a single value at a time. These slice metrics are saved in slice_output.txt. Performance varies across slices, with some groups scoring higher or lower than the overall metrics, which reflects differences in how well the model performs across different segments of the population.

## Ethical Considerations

The dataset includes sensitive attributes such as race, sex, and native country. A model trained on this data can reflect and reproduce historical biases present in the 1994 census, and performance differs across demographic slices. Because of this, the model should not be used to make decisions that affect real people. Predictions could disadvantage groups that are underrepresented or that the model performs poorly on.

## Caveats and Recommendations

The data is from 1994 and does not reflect current economic conditions, income levels, or population demographics, so predictions should not be treated as accurate for the present day. The model uses default Random Forest settings with no hyperparameter tuning, so performance could likely be improved with tuning or cross-validation. The slice metrics should be reviewed before trusting the model on any particular subgroup. This model is best used as a learning example of an end-to-end machine learning pipeline rather than a production tool.