
Dataset exploration
- Analysis
- World map
- Line graph (Kieran does this later)
- Gleichverteilung von dataset


Model Training



Model Testing
 - Inputs: Dataset, Model, Prompt, batch_size, country_list
 - Output: csv for each batch with 
	 - Ground Truth Label
	 - All Probs
 - Output Folder Names: Experiments/{model_name}/{prompt_name}/{dateset_name}-{custom_tag}/{date}-{batch_number}.csv


Evaluation
 - Input: 1 Experiment
 - Metric
 - Accuracy
 -  Fairness


Comparison
 - Input: More than 1 Experiments
 - Box Plot + Violin Plot
 - Report


Fetching new Data
 - Mars Data
 - Aerial Shots
 - Travel Photos