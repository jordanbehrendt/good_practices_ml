import create_datasets_from_embddings
import sys
sys.path.append("./../")
import finetuning.model.model_trainer as trainer

REPO_PATH = "/home/lbrenig/Documents/Uni/GPML/good_practices_ml"

seeds = [4808,4947,5723,3838,5836,3947,8956,5402,1215,8980]
for seed in seeds:
    create_datasets_from_embddings.create_datasets_from_embddings(REPO_PATH, seed=seed)
    trainer.create_and_train_model(REPO_PATH, seed)