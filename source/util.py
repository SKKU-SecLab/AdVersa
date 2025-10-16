import h2o
from h2o.automl import H2OAutoML
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def model_checker():
    folders=list(os.listdir("../models"))
    folders.remove("features.yaml")
    for f in folders:
        print(f)
        if not f=="temporal_shift":
            for model in os.listdir("../models/"+f):
                print("\t",model)
                try:
                    if model.endswith(".joblib"):
                        m=joblib.load(f"../models/{f}/{model}")
                        
                    print("\tOkay")
                except Exception as e:
                    pass
                    print(e)
        else:
            for t in os.listdir("../models/"+f):
                print("\t",t)
                for model in os.listdir("../models/"+f+"/"+t):
                    print("\t\t",model)
                    try:
                        if model.endswith(".joblib"):
                            m=joblib.load(f"../models/{f}/{t}/{model}")
                        print("\t\tOkay")
                    except Exception as e:
                        print("\t\tNo")

model_checker()