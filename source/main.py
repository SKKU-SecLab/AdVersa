import sys
import h2o
from h2o.automl import H2OAutoML
import os
import joblib
import argparse
import yaml
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier

def main(program, argv):
    main_parser=argparse.ArgumentParser(description="Reproduce AdVersa's Evaluation")
    main_parser.add_argument("-p", type=str, required=True, help="Evaluation process you want to run", choices=[
        'model_selection',
        'sota_comparison',
        'unseen_domain',
        'temporal_shift',
        'robustness',
        'ablation'
    ])

    main_parser.add_argument("-t", type=int,default=10, required=False, help="Num. threads to use")

    main_args=main_parser.parse_args()
    h2o.init(enable_assertions = False, log_level="ERRR")
    h2o.no_progress()
    
    print("================Start Process================")
    if main_args.p=="model_selection":
        model_selection()
    elif main_args.p=="sota_comparison":
        sota_comparison()
    elif main_args.p=="unseen_domain":
        unseen_domain()
    elif main_args.p=="temporal_shift":
        temporal_shift()
    elif main_args.p=="robustness":
        robustness()
    elif main_args.p=="ablation":
        ablation()
    print("================End Process================")
    return

def model_selection():
    print("---------------Model Selection---------------")
    dataframe=[]
    data_parent_dir=os.path.join("..","dataset","testing")
    model_parent_dir=os.path.join("..","models","model_selection")
    datasets=os.listdir(data_parent_dir)
    models=os.listdir(model_parent_dir)

    with open(os.path.join("..","models","features.yaml"),"r") as yamlfile:
        features=yaml.load(yamlfile, yaml.Loader)
    print("Loading Datasets...")
    for dataset in datasets:
        pq=pd.read_parquet(os.path.join(data_parent_dir,dataset))
        dataframe.append(pq)
    dataframe=pd.concat(dataframe)
    print("Converting to H2O frame...")
    h2o_dataframe=h2o.H2OFrame(dataframe)
    label=dataframe["label"]

    for model in models:
        h2o_model=h2o.import_mojo(os.path.join(model_parent_dir,model))
        model_input=h2o_dataframe[features[f"model_{model}"]]
        pred=h2o_model.predict(model_input)
        print("Model:",model)
        metrics(label, pred.as_data_frame().predict.tolist(), False, False,False,None)
        print("")
    return

def sota_comparison():
    print("---------------Sota Comparison---------------")
    dataframe=[]
    data_parent_dir=os.path.join("..","dataset","testing")
    model_parent_dir=os.path.join("..","models","sota_comparison")
    datasets=os.listdir(data_parent_dir)
    models=os.listdir(model_parent_dir)

    with open(os.path.join("..","models","features.yaml"),"r") as yamlfile:
        features=yaml.load(yamlfile, yaml.Loader)
    print("Loading Datasets...")
    for dataset in datasets:
        pq=pd.read_parquet(os.path.join(data_parent_dir,dataset))
        dataframe.append(pq)
    dataframe=pd.concat(dataframe)
    print("Converting to H2O frame...")
    h2o_dataframe=h2o.H2OFrame(dataframe)
    label=dataframe["label"]

    for model in models:
        if model.endswith("joblib"):
            model_object=joblib.load(os.path.join(model_parent_dir,model))
            model=model.replace(".joblib","")
            model_input=dataframe[features[f"model_{model}"]]
            pred=model_object.predict(model_input)
        else:
            model_object=h2o.import_mojo(os.path.join(model_parent_dir,model))
            model_input=h2o_dataframe[features[f"model_{model}"]]
            pred=model_object.predict(model_input)
            pred=pred.as_data_frame().predict
        print("Model:",model)
        metrics(label, pred.tolist(), False, False,False,None)
        print("")
    return

def unseen_domain():
    print("---------------Unseen Domain---------------")
    dataframe=[]
    data_parent_dir=os.path.join("..","dataset","unseen_domain")
    model_parent_dir=os.path.join("..","models","sota_comparison")
    datasets=os.listdir(data_parent_dir)
    models=["AdGraph","WebGraph","AdFlush","AdVersa"]

    with open(os.path.join("..","models","features.yaml"),"r") as yamlfile:
        features=yaml.load(yamlfile, yaml.Loader)
    print("Loading Datasets...")
    for dataset in datasets:
        pq=pd.read_parquet(os.path.join(data_parent_dir,dataset))
        dataframe.append(pq)
    dataframe=pd.concat(dataframe)
    print("Converting to H2O frame...")
    h2o_dataframe=h2o.H2OFrame(dataframe)
    label=dataframe["label"]

    for model in models:
        if model.endswith("Graph"):
            model_object=joblib.load(os.path.join(model_parent_dir,model+".joblib"))
            model_input=dataframe[features[f"model_{model}"]]
            pred=model_object.predict(model_input)
        else:
            model_object=h2o.import_mojo(os.path.join(model_parent_dir,model))
            model_input=h2o_dataframe[features[f"model_{model}"]]
            pred=model_object.predict(model_input)
            pred=pred.as_data_frame().predict
        print("Model:",model)
        metrics(label, pred.tolist(), False, False,False,None)
        print("")
    return

def robustness():
    return

def ablation():
    return

def metrics(true, pred, _is_mutated, _return,fprint,outfile):
    asr=None
    if _is_mutated:
        total_attacks = len(true)
        successful_attacks = sum(true != pred)
        asr=successful_attacks/total_attacks

    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

    # Calculate FNR
    fnr = fn / (tp + fn)

    # Calculate FPR
    fpr = fp / (fp + tn)

    acc=accuracy_score(true, pred)
    pre=precision_score(true, pred)
    rec=recall_score(true, pred)
    f1=f1_score(true, pred)


    auroc=roc_auc_score(true, pred)
    fprlist, tprlist, thresholds=roc_curve(true, pred)
    cutoff=int(thresholds[np.argmax(tprlist-fprlist)])
    opttpr=tprlist[cutoff]
    optfpr=fprlist[cutoff]

    if _return:
        return acc,pre,rec,f1,fnr,fpr
        
    else:   
        output_lines = [
            f"Accuracy : {acc}",
            f"Precision : {pre}",
            f"Recall : {rec}",
            f"F1 : {f1}",
            f"False Negative Rate: {fnr}",
            f"False Positive Rate: {fpr}",
            f"AUROC: {auroc}",
            f"TPR {opttpr} at FPR {optfpr}"
        ]
        if asr is not None:
            output_lines.append(f"Attack Success Rate: {asr}")   
          
        # Logic for file printing
        if fprint:
            if not outfile:
                raise ValueError("An 'outfile' name must be provided when 'fprint' is True.")
            with open(outfile, 'w') as f:
                for line in output_lines:
                    f.write(line + "\n")
            print(f"Metrics successfully written to {outfile}")
        
        # Original logic for console printing
        else:
            for line in output_lines:
                print(line)




if __name__=="__main__":    
    if not os.getcwd().endswith("source"):
        print("Please run this file from the 'source' directory")
    else:
        main(sys.argv[0], sys.argv[1:])