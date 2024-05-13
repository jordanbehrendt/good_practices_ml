# Set-Up

1. In the paths.yaml file you have to add the path to the repository and the path to the data that you want to use.




# data_collection

## This folder is provided for the purpose of cross-checking the methods for data collection. The Data need not be collected again and can instead be downloaded from the following sources
1. Create a data repository and copy the repository path into the paths.yaml file
2. *Geoguessr:* download the 'compressed_dataset' folder from https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k/data and rename it to 'geoguessr'
3. *Tourist:* download the 'tourist' folder from https://osf.io/pe453/?view_only=d4ebd0f1fcb54dd8b24312fed3e5b722
4. *Aerial:* download the 'aerial' folder from https://osf.io/pe453/?view_only=d4ebd0f1fcb54dd8b24312fed3e5b722

## data_exploration
The data_profile script, located in the data_collection/data_exploration directory, is designed for analyzing and visualizing image distribution within datasets. It generates comprehensive reports, CSV files for image distribution, and several visualizations, including heat maps and graphs, to better understand data characteristics.
### Features
1. Profile Reports: Generates detailed statistical profiles of datasets, which can be used for initial data analysis.
2. Image Distribution CSV: Creates a CSV file listing the distribution of images across different labels/categories.
3. Visualizations:
    3.1 Line Graphs: Shows the distribution of images across categories.
    3.2 Bar Graphs: Provides a horizontal view of the image distribution.
    3.3 World Heat Maps: Displays a geographical distribution of images across the world, both in logarithmic and linear scales.
### Command line interface
To run the script from the command line, navigate to the script's directory and execute the following command:
python data_profile.py --user [user] --yaml_path [path_to_yaml] --dataset_dir [path_to_dataset] --dataset_name [name_of_dataset]

Arguments
    --user: Specifies the user of the GPML group.
    --yaml_path: Path to the YAML configuration file containing paths needed for the script.
    --dataset_dir: Directory path where the dataset is located.
    --dataset_name: Name of the dataset for which the profile and visualizations are generated.
#### Example
python data_profile.py --user bjordan --yaml_path /path/to/paths.yaml --dataset_dir /path/to/dataset --dataset_name "Tourist"
### Output
The script will generate the following outputs:
1. A profile report as an HTML file.
2. A CSV file detailing image distribution.
3. Various graphs and maps saved as JPEG files.
4. A combined PDF document containing all the generated visualizations.


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

