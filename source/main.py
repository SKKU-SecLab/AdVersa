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
import statistics
os.environ['JOBLIB_VERBOSITY'] = '0'

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
    main_parser.add_argument("-m", type=str,default="AdVersa",choices=["AdVersa","AdGraph","WebGraph","AdFlush"], required=False,help="Target model for YOPO attack")
    main_parser.add_argument("-e", type=int,default=10, choices=[5,10,20,40],required=False, help="Epsilon for YOPO attack")
    main_parser.add_argument("-c", type=str,default="DC", choices=["DC","HSC","HCC","HJC"], required=False, help="Cost model for YOPO attack")    

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
        nostat=False
        if main_args.m=="AdVersa":
            if main_args.c not in ["DC","HJC","HCC"]:
                nostat=True
        elif main_args.m=="AdGraph":
            if main_args.c not in ["DC","HSC","HCC"]:
                nostat=True
        elif main_args.m=="WebGraph":
            if main_args.c not in ["DC","HSC"]:
                nostat=True
        elif main_args.m=="AdFlush":
            if main_args.c not in ["DC","HJC","HCC"]:
                nostat=True
        
        if nostat:
            print("Unavailable strategy. Exiting program.")
            exit(1)
        else:
            robustness(main_args.m,main_args.e,main_args.s)
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
        if dataset.endswith(".parquet"):
            pq=pd.read_parquet(os.path.join(data_parent_dir,dataset))
            dataframe.append(pq)
    dataframe=pd.concat(dataframe)
    print("Converting to H2O frame...")
    h2o_dataframe=h2o.H2OFrame(dataframe)
    label=dataframe["label"].tolist()

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
        if dataset.endswith(".parquet"):
            pq=pd.read_parquet(os.path.join(data_parent_dir,dataset))
            dataframe.append(pq)
    dataframe=pd.concat(dataframe)
    print("Converting to H2O frame...")
    h2o_dataframe=h2o.H2OFrame(dataframe)
    label=dataframe["label"].tolist()

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
        if dataset.endswith(".parquet"):
            pq=pd.read_parquet(os.path.join(data_parent_dir,dataset))
            dataframe.append(pq)
    dataframe=pd.concat(dataframe)
    print("Converting to H2O frame...")
    h2o_dataframe=h2o.H2OFrame(dataframe)
    label=dataframe["label"].tolist()

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

def temporal_shift():
    print("---------------Temporal Shift---------------")
    data_parent_dir=os.path.join("..","dataset","temporal_shift")
    model_parent_dir_outdate=os.path.join("..","models","sota_comparison")
    model_parent_dir_retrain=os.path.join("..","models","temporal_shift")
    models=["AdGraph","WebGraph","AdFlush","AdVersa"]
    methods=["none","retrain_filterlist","retrain_pseudo10","retrain_pseudo20","retrain_pseudo30"]

    with open(os.path.join("..","models","features.yaml"),"r") as yamlfile:
        features=yaml.load(yamlfile, yaml.Loader)
    print("Loading Dataset...")
    dataframe=pd.read_parquet(os.path.join(data_parent_dir,"eval_8.parquet"))

    print("Converting to H2O frame...")
    h2o_dataframe=h2o.H2OFrame(dataframe)
    label=dataframe["label"].tolist()
    for method in methods:
        print("Retraining method:",method)
        if method=="none":
            model_parent_dir=model_parent_dir_outdate
        else:
            model_parent_dir=os.path.join(model_parent_dir_retrain,method)
        for model in models:
            try:
                if model.endswith("Graph"):
                    model_object=joblib.load(os.path.join(model_parent_dir,model+".joblib"))
                    model_input=dataframe[features[f"model_{model}"]]
                    pred=model_object.predict(model_input)
                else:
                    model_object=h2o.import_mojo(os.path.join(model_parent_dir,model))
                    model_input=h2o_dataframe[features[f"model_{model}"]]
                    pred=model_object.predict(model_input)
                    pred=pred.as_data_frame().predict
                f1=f1_score(label, pred.tolist())
                print(f"\tModel: {model}, F1 Score: {f1}")
            except Exception as e:
                print(e)
        print("")
    return

def robustness(model, eps,cost_model):
    print("-----------------Robustness-----------------")
    dataframe=[]
    data_parent_dir=os.path.join("..","dataset","testing")
    model_parent_dir=os.path.join("..","models","sota_comparison")
    datasets=os.listdir(data_parent_dir)
    models=["AdGraph","WebGraph","AdFlush","AdVersa"]

    with open(os.path.join("..","models","features.yaml"),"r") as yamlfile:
        features=yaml.load(yamlfile, yaml.Loader)
    print("Loading Datasets...")
    for dataset in datasets:
        if dataset.endswith(".parquet"):
            pq=pd.read_parquet(os.path.join(data_parent_dir,dataset))
            dataframe.append(pq)
    dataframe=pd.concat(dataframe)
    label=dataframe["label"].tolist()
    dataframe=dataframe[features[f"model_{model}"]]

    if not model.endswith("Graph"):
        print("Converting to H2O frame...")
        h2o_dataframe=h2o.H2OFrame(dataframe)
    
    perturb_parent_dir=os.path.join("yopo","perturbations",model)
    asrs=[]
    rec_drops=[]
    f1_drops=[]
    for i in range(1,11):
        perturbation=np.load(os.path.join(perturb_parent_dir,f"{cost_model}_eps{eps}_iter{i}.npy"))

        # Clean results
        if model.endswith("Graph"):
            model_object=joblib.load(os.path.join(model_parent_dir,model+".joblib"))
            pred=model_object.predict(dataframe)
        else:
            model_object=h2o.import_mojo(os.path.join(model_parent_dir,model))
            pred=model_object.predict(h2o_dataframe)
            pred=pred.as_data_frame().predict
        _,_,clean_rec,clean_f1,_,_=metrics(label, pred.tolist(), False, True,False,None)
        # Adversarial results
        if model.endswith("Graph"):
            model_object=joblib.load(os.path.join(model_parent_dir,model+".joblib"))
            model_input=dataframe.add(perturbation,axis="columns")
            pred=model_object.predict(model_input)
        else:
            model_object=h2o.import_mojo(os.path.join(model_parent_dir,model))
            perturbation_frame = h2o.H2OFrame([perturbation]) 
            perturbation_frame.columns = h2o_dataframe.columns
            model_input = h2o_dataframe + perturbation_frame
            pred=model_object.predict(model_input)
            pred=pred.as_data_frame().predict
        asr,_,_,rec,f1,_,_=metrics(label, pred.tolist(), True, True,False,None)
        asrs.append(asr)
        rec_drops.append(clean_rec-rec)
        f1_drops.append(clean_f1-f1)
    print(f"Model: {model}, epsilon: {eps}, cost model: {cost_model}, ASR: {statistics.median(asrs)}, REC drop: {statistics.median(rec_drops)}, F1 drop: {statistics.median(f1_drops)}")
    return

def ablation():
    print("---------------Ablation---------------")
    model_parent_dir=os.path.join("..","models","ablation")
    models=["ablation_url","ablation_targeturl","ablation_sourcefqdn","ablation_codeemb","ablation_handcrafted"]
    with open(os.path.join("..","models","features.yaml"),"r") as yamlfile:
        features=yaml.load(yamlfile, yaml.Loader)
    features=features["model_AdVersa"]
    featuresets=[]
    rem_url=[f for f in features if "nomic" not in f]
    rem_target=[f for f in features if "url_nomic" not in f]
    rem_source=[f for f in features if "fqdn_nomic" not in f]
    rem_code=[f for f in features if "codemalt" not in f]
    rem_hand=[f for f in features if "nomic" not in f and "codemalt" not in f]
    featuresets.append(rem_url)
    featuresets.append(rem_target)
    featuresets.append(rem_source)
    featuresets.append(rem_code)
    featuresets.append(rem_hand)

    print("Loading all evaluation data")
    print("\tLoading Test Datasets...")
    evaluation_datasets=[]
    evaluation_labels=[]
    test_data_parent_dir=os.path.join("..","dataset","testing")
    test_datasets=os.listdir(test_data_parent_dir)
    
    test_dataframe=[]
    for dataset in test_datasets:
        if dataset.endswith(".parquet"):
            pq=pd.read_parquet(os.path.join(test_data_parent_dir,dataset))
            test_dataframe.append(pq)
    test_dataframe=pd.concat(test_dataframe)
    print("\tConverting to H2O frame...")
    test_h2o_dataframe=h2o.H2OFrame(test_dataframe)
    test_label=test_dataframe["label"].tolist()
    evaluation_datasets.append(test_h2o_dataframe)
    evaluation_labels.append(test_label)

    print("\tLoading Unseen Domain Datasets...")
    evaluation_datasets=[]
    evaluation_labels=[]
    unseen_data_parent_dir=os.path.join("..","dataset","unseen_domain")
    unseen_datasets=os.listdir(unseen_data_parent_dir)
    
    unseen_dataframe=[]
    for dataset in unseen_datasets:
        if dataset.endswith(".parquet"):
            pq=pd.read_parquet(os.path.join(unseen_data_parent_dir,dataset))
            unseen_dataframe.append(pq)
    unseen_dataframe=pd.concat(unseen_dataframe)
    print("\tConverting to H2O frame...")
    unseen_h2o_dataframe=h2o.H2OFrame(unseen_dataframe)
    unseen_label=unseen_dataframe["label"].tolist()
    evaluation_datasets.append(unseen_h2o_dataframe)
    evaluation_labels.append(unseen_label)

    print("\tLoading Temporal Shift Datasets...")
    evaluation_datasets=[]
    evaluation_labels=[]
    temporal_data_parent_dir=os.path.join("..","dataset","temporal_shift")
    temporal_dataframe=pd.read_parquet(os.path.join(temporal_data_parent_dir,"eval_8.parquet"))
    print("\tConverting to H2O frame...")
    temporal_h2o_dataframe=h2o.H2OFrame(temporal_dataframe)
    temporal_label=temporal_dataframe["label"].tolist()
    evaluation_datasets.append(temporal_h2o_dataframe)
    evaluation_labels.append(temporal_label)

    evaluation_names=["Testing","Unseen Domain","Temporal Shift"]
    model_features=zip(models, featuresets)
    datasets_labels=zip(evaluation_names,evaluation_datasets, evaluation_labels)
    for eval_name, eval_dataset, eval_label in datasets_labels:
        print("Dataset:",eval_name)
        for model,featureset in model_features:
            h2o_model=h2o.import_mojo(os.path.join(model_parent_dir,model))
            model_input=eval_dataset[featureset]
            pred=h2o_model.predict(model_input)
            print("Model:",model)
            metrics(eval_label, pred.as_data_frame().predict.tolist(), False, False,False,None)
            print("")
    return

def metrics(true, pred, _is_mutated, _return,fprint,outfile):
    asr=None
    if _is_mutated:
        total_attacks = sum(1 for e in true if e==True)
        successful_attacks = sum(1 for i in range(len(true)) if true[i]==True and pred[i]==False)
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
        if _is_mutated:
            return asr,acc,pre,rec,f1,fnr,fpr
        else:
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