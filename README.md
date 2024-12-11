# Set-Up

1. In the paths.yaml file you have to add the absolute path to the code directory (where this repository lives) and the path to the data directory (where you will place the folders containing geoguessr, tourist, and aerial datasets).
2. Create a python environment with the same python version as stated in `.python-version`.
3. Install the dependencies listed in `requirements.txt`. :warning: In its current version, `requirements.txt` lists all dependencies listed in the environment where this code was produced (Ubunto 20.04.6), if you encounter installation errors, consider whether they might be caused by platform specific dependencies, and CUDA version/capabilities compatibility. In these cases, remove those dependencies from your local version of `requirements.txt` and let pip dependency resolver figure out which packages you need. In a future revision of this project we will fix this.

# data_collection

## This folder is provided for the purpose of cross-checking the methods for data collection. The Data need not be collected again and can instead be downloaded from the following sources

1. Create a data directory and copy its absolute path into the paths.yaml file, replacing the `data_path: default:` field.
2. *Geoguessr:* download the 'compressed_dataset' folder from <https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k/data>, rename it to 'geoguessr' and run '/data_collection/geoguessr/geoguessr_data_preparation.py'
3. *Tourist:* download the 'tourist' folder from <https://osf.io/pe453/?view_only=d4ebd0f1fcb54dd8b24312fed3e5b722>
4. *Aerial:* download the 'aerial' folder from <https://osf.io/wrmzx/?view_only=bbd7cf7d0f6243e7ac6b87fb45fac04a>

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
python data_profile.py --yaml_path [path_to_yaml] --dataset_name [name_of_dataset]

Arguments
    --yaml_path: Path to the YAML configuration file containing paths needed for the script.
    --dataset_name: Name of the dataset for which the profile and visualizations are generated.

#### Example

python data_profile.py --yaml_path /path/to/paths.yaml --dataset_name "Tourist"

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

# Finetuning

Except for the embedding generations all necessary files for the fine tuning are in the fine-tuning folder.

## Generate Training and Test Data

To generate the training and test data, clip embeddings are used.
If they were not generated yet:

1. Run generate_image_embeddings.py file
2. Prompt embeddings will be saved in the folder '/CLIP_Embeddings/Prompt'
3. Image embeddings in association with the *extended prompt* will be saved in the folder '/CLIP_Embeddings/Image'
Then generate the CSV files for the training, test and zero-shot data using create_datasets_from_embeddings.py.

To recreate the papers experiment just specify your repository path, for example:

```shell
python create_datasets_from_embddings.py --repo_path "path/to/the/repo"
```

The CSV files are saved in the repository in the CLIP_Embeddings/ directory.

## Training the Model

To train the model adjust the REPO_PATH in the run_experiments.py and then start the script.
If you want to run the different datasets separately use the corresponding python script.

To monitor the training process you can connect tensorboard to the runs folder. 

## Evaluating finetuning Results

To recreate the evaluation use the analyze_csv_files.ipynb notebook.
Before using the notebook, the output of the model needs to restructured using the merge_seed.sh script.
Adjust the paths in the script and run it to create the wanted structure.
This will simply copy the files in a way that all seeds for one experiment are in the same folder.
Then adjust the paths in the notebook and run the different cells.

