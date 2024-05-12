# Set-Up

1. In the paths.yaml file you have to add the path to the repository and the path to the data that you want to use.




# data_collection

## This folder is provided for the purpose of cross-checking the methods for data collection. The Data need not be collected again and can instead be downloaded from the following sources
1. Create a data repository and copy the repository path into the paths.yaml file
2. *Geoguessr:* download the 'compressed_dataset' folder from https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k/data and rename it to 'geoguessr'
3. *Tourist:* download the 'tourist' folder from https://osf.io/pe453/?view_only=d4ebd0f1fcb54dd8b24312fed3e5b722
4. *Aerial:* download the 'aerial' folder from https://osf.io/pe453/?view_only=d4ebd0f1fcb54dd8b24312fed3e5b722




# CLIP_Experiment

## Run experiments
1. Run '/CLIP_Experiment/run_datasets_and_prompts.py'
2. The results will be saved as .csv files within the folder '/CLIP_Experiment/clip_results'

## Evaluate Results with Metrics (Requires run_datasets_and_prompts.py to be succesfully completed)
1. Run '/CLIP_Experiment/evaluate_results_with_metrics.py'
2. The results will be saved as .csv files within the folder '/CLIP_Experiment/result_accuracy'

## Run statistical Tests (Requires evaluate_results_with_metrics.py to be succesfully completed)
1. Run '/CLIP_Experiment/run_statistical_tests.py'
2. Statistical Results (Mean, Median, Standard Deviation and t-tests) will be saved in .csv files within the folder '/CLIP_Experiment/statistical_tests'
3. Box Plot and Violin Plot graphs will be saved as .png files within the folder '/CLIP_Experiment/statistical_tests'

## Generate Confusion Matrices (Requires run_datasets_and_prompts.py to be succesfully completed)
1. Run '/CLIP_Experiment/generate_confusion_matrices.py'
2. The plots will be saved as .png files within the folder '/CLIP_Experiment/confusion_matrices'





# CLIP_Embeddings

## Generate Embeddings
1. Run generate_image_embeddings.py file
2. Prompt embeddings will be saved in the folder '/CLIP_Embeddings/Prompt'
3. Image embeddings in association with the *extended prompt* will be saved in the folder '/CLIP_Embeddings/Image'

## t-SNE
1. Run '/CLIP_Embeddings/t-SNE/tsne.py'
2. The resulting plots will be saved as .png files within the folder '/CLIP_Embeddings/t-SNE' in sub-folders according to the dataset and region




# finetuning

