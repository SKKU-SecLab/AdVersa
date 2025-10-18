import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings('identifier')
import onnxruntime as ort
import sys
import json
import subprocess
import logging
import re
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"]="false"
from urllib.parse import urlparse
from mitmproxy import http
from model2vec import StaticModel
import numpy as np
import time
import tldextract
import torch
import requests
# import torch.nn.functional as F
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import base64

# print("Adjusting Recursion Limit",sys.getrecursionlimit(),"-> 5000")
# sys.setrecursionlimit(5000)
print("Loading models..")

codemaltmodel = StaticModel.from_pretrained("sarthak1/codemalt")
matryoshka_dim = 128
tokenizer = AutoTokenizer.from_pretrained('nomic',local_files_only=True)
nomicmodel=ORTModelForFeatureExtraction.from_pretrained("nomic",local_files_only=True,trust_remote_code=True)

usefeatures=['ad_size_in_qs_present', 'ast_depth', 'avg_charperline','brackettodot', 'codemalt_111', 'codemalt_139', 'codemalt_152','codemalt_193', 'codemalt_202', 'codemalt_38', 'codemalt_74','codemalt_95', 'content_policy_type', 'fqdn_nomic_0','fqdn_nomic_1', 'fqdn_nomic_10', 'fqdn_nomic_101','fqdn_nomic_103', 'fqdn_nomic_105', 'fqdn_nomic_106','fqdn_nomic_108', 'fqdn_nomic_109', 'fqdn_nomic_11','fqdn_nomic_110', 'fqdn_nomic_111', 'fqdn_nomic_112','fqdn_nomic_113', 'fqdn_nomic_115', 'fqdn_nomic_116','fqdn_nomic_117', 'fqdn_nomic_118', 'fqdn_nomic_121','fqdn_nomic_122', 'fqdn_nomic_124', 'fqdn_nomic_126','fqdn_nomic_127', 'fqdn_nomic_14', 'fqdn_nomic_15','fqdn_nomic_16', 'fqdn_nomic_17', 'fqdn_nomic_18', 'fqdn_nomic_2','fqdn_nomic_20', 'fqdn_nomic_22', 'fqdn_nomic_24', 'fqdn_nomic_25','fqdn_nomic_26', 'fqdn_nomic_27', 'fqdn_nomic_29', 'fqdn_nomic_3','fqdn_nomic_33', 'fqdn_nomic_34', 'fqdn_nomic_36', 'fqdn_nomic_37','fqdn_nomic_38', 'fqdn_nomic_39', 'fqdn_nomic_4', 'fqdn_nomic_42','fqdn_nomic_45', 'fqdn_nomic_46', 'fqdn_nomic_47', 'fqdn_nomic_48','fqdn_nomic_5', 'fqdn_nomic_51', 'fqdn_nomic_53', 'fqdn_nomic_55','fqdn_nomic_59', 'fqdn_nomic_60', 'fqdn_nomic_64', 'fqdn_nomic_65','fqdn_nomic_66', 'fqdn_nomic_68', 'fqdn_nomic_71', 'fqdn_nomic_72','fqdn_nomic_73', 'fqdn_nomic_74', 'fqdn_nomic_75', 'fqdn_nomic_8','fqdn_nomic_81', 'fqdn_nomic_86', 'fqdn_nomic_87', 'fqdn_nomic_90','fqdn_nomic_91', 'fqdn_nomic_92', 'fqdn_nomic_96', 'fqdn_nomic_97','fqdn_nomic_98', 'is_third_party', 'keyword_char_present','num_requests_sent', 'num_set_cookie', 'num_set_storage','url_nomic_0', 'url_nomic_1', 'url_nomic_10', 'url_nomic_100','url_nomic_101', 'url_nomic_102', 'url_nomic_103', 'url_nomic_104','url_nomic_105', 'url_nomic_106', 'url_nomic_107', 'url_nomic_108','url_nomic_109', 'url_nomic_11', 'url_nomic_110', 'url_nomic_111','url_nomic_112', 'url_nomic_113', 'url_nomic_114', 'url_nomic_116','url_nomic_118', 'url_nomic_12', 'url_nomic_120', 'url_nomic_121','url_nomic_122', 'url_nomic_123', 'url_nomic_125', 'url_nomic_126','url_nomic_127', 'url_nomic_13', 'url_nomic_14', 'url_nomic_15','url_nomic_16', 'url_nomic_17', 'url_nomic_18', 'url_nomic_19','url_nomic_2', 'url_nomic_20', 'url_nomic_21', 'url_nomic_22','url_nomic_23', 'url_nomic_25', 'url_nomic_26', 'url_nomic_27','url_nomic_28', 'url_nomic_29', 'url_nomic_3', 'url_nomic_30','url_nomic_31', 'url_nomic_32', 'url_nomic_33', 'url_nomic_34','url_nomic_35', 'url_nomic_36', 'url_nomic_37', 'url_nomic_39','url_nomic_4', 'url_nomic_41', 'url_nomic_42', 'url_nomic_43','url_nomic_45', 'url_nomic_46', 'url_nomic_47', 'url_nomic_48','url_nomic_5', 'url_nomic_50', 'url_nomic_51', 'url_nomic_52','url_nomic_53', 'url_nomic_54', 'url_nomic_55', 'url_nomic_57','url_nomic_58', 'url_nomic_59', 'url_nomic_6', 'url_nomic_60','url_nomic_61', 'url_nomic_62', 'url_nomic_63', 'url_nomic_64','url_nomic_65', 'url_nomic_66', 'url_nomic_69', 'url_nomic_7','url_nomic_70', 'url_nomic_71', 'url_nomic_72', 'url_nomic_73','url_nomic_74', 'url_nomic_75', 'url_nomic_76', 'url_nomic_77','url_nomic_78', 'url_nomic_79', 'url_nomic_80', 'url_nomic_81','url_nomic_82', 'url_nomic_84', 'url_nomic_85', 'url_nomic_86','url_nomic_87', 'url_nomic_88', 'url_nomic_89', 'url_nomic_9','url_nomic_90', 'url_nomic_91', 'url_nomic_92', 'url_nomic_93','url_nomic_94', 'url_nomic_95', 'url_nomic_96', 'url_nomic_98','url_nomic_99']

req_reg = re.compile(r'https?://')
set_stor_reg = re.compile(r'([sS]torage\.setItem)|([sS]torage\[[^\]]+\][^;\n]*=)|([sS]torage\.[^=;\n]+=)|([sS]torage[^;\n]*=)')
# set_cook_reg = re.compile(r'([cC]ookies?\.set)|([cC]ookies?\[[^\]]+\][^;\n=]*=)|([cC]ookies?\.[^=;\n]+=)|([cC]ookies?[^;\n]*=)')
get_bracket_reg = re.compile(r'\[|\]|\(|\)')
get_dot_reg = re.compile(r'\.')
adsize_reg=re.compile(r'\d{2,4}[xX]\d{2,4}')
clean_js_reg = re.compile(r'(?<!/)"[^"]*\.?[^"]*"|\'[^\']*\.?[^\']*\'')
epsilon = 1e-8
conf=0.98
cssconf=0.99
codemalt_index=[111, 139, 152, 193, 202, 38, 74, 95]
fqdn_nomic_index=[0,1,10,101,103,105,106,108,109,11,110,111,112,113,115,116,117,118,121,122,124,126,127,14,15,16,17,18,2,20,22,24,25,26,27,29,3,33,34,36,37,38,39,4,42,45,46,47,48,5,51,53,55,59,60,64,65,66,68,71,72,73,74,75,8,81,86,87,90,91,92,96,97,98]
url_nomic_index=[0,1,10,100,101,102,103,104,105,106,107,108,109,11,110,111,112,113,114,116,118,12,120,121,122,123,125,126,127,13,14,15,16,17,18,19,2,20,21,22,23,25,26,27,28,29,3,30,31,32,33,34,35,36,37,39,4,41,42,43,45,46,47,48,5,50,51,52,53,54,55,57,58,59,6,60,61,62,63,64,65,66,69,7,70,71,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,9,90,91,92,93,94,95,96,98,99]
keyword_raw = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser", "advertise", "redirect",
                 "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid", "pb.min", "affiliate", "ban", "delivery",
                 "promo","tag", "zoneid", "siteid", "pageid", "size", "viewid", "zone_id", "google_afc" , "google_afs"]
keyword_char = [".", "/", "&", "=", ";", "-", "_", "/", "*", "^", "?", ";", "|", ","]

with open("targetencode.json","r") as f:
    contentencoder=json.load(f)

def count_matches(reg, code):
    return len(reg.findall(code))

def tree_walk_feat_ext(node):
    def walk(n):
        if not isinstance(n, (dict, list)):
            return 1

        if isinstance(n, list):
            max_child_depth = 0
            for child in n:
                res = walk(child)
                if res> max_child_depth:
                    max_child_depth = res
            return 1

        # Case for dict-like AST nodes
        if isinstance(n, dict):
            max_child_depth = 0
            found_complex_child = False
            
            for key in n:
                child = n[key]
                if isinstance(child, (dict, list)) and child:
                    found_complex_child = True
                    res = walk(child)
                    if res> max_child_depth:
                        max_child_depth = res
            
            if not found_complex_child:
                return 1
            else: 
                return 1 + max_child_depth
    
    result = walk(node)
    return result

def parseAst(js_code):
    try:
        result = subprocess.run(
                ["node","jsparser.js"],
                input=js_code,          # Pass the JS code to the process's stdin
                capture_output=True,    # Capture stdout and stderr
                text=True,              # Work with text strings instead of bytes
                check=True,              # Raise CalledProcessError if Node.js exits with an error
                encoding="utf-8"
        )
        ast_json = result.stdout
        ast_dict = json.loads(ast_json)
        # logging.debug(f"SUCCESS :ASTPARSE: {jsidx}")
        return ast_dict

    except subprocess.CalledProcessError as e:
        return None
    except Exception as e:
        return None

def extJSFeature(js_code):
    codelen=len(js_code)
    linenum=js_code.count("\n")+1
    charperline=codelen/linenum
    ast=parseAst(js_code)
    if ast is not None:
        depth=tree_walk_feat_ext(ast)
        clean_js = re.sub(clean_js_reg, '""', js_code)

        num_req = count_matches(req_reg, clean_js)
        num_set_storage = count_matches(set_stor_reg, clean_js)
        # num_set_cookie = count_matches(set_cook_reg, clean_js)
        num_brackets = count_matches(get_bracket_reg, clean_js)
        num_dots = count_matches(get_dot_reg, clean_js)

        bracket_to_dot = num_brackets / num_dots if num_dots else 0

        features = {
        "num_requests_sent": num_req,
        "num_set_storage": num_set_storage,
        # "num_set_cookie": num_set_cookie,
        "ast_depth": depth,
        "avg_charperline":charperline,
        "brackettodot": bracket_to_dot
        }
        return features

    else:
        features = {
        "num_requests_sent": 0,
        "num_set_storage": 0,
        # "num_set_cookie": 0,
        "ast_depth": 0,
        "avg_charperline":charperline,
        "brackettodot": 0
        }
        return features

def mean_pooling(model_output, attention_mask):
    model_output=model_output[:,:,:128]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_nomic(seq):
    seq=["classification: "+s for s in seq]
    inputs=tokenizer(seq, return_tensors="pt", padding=True)
    outputs=nomicmodel(**inputs)[0]
    outputs=mean_pooling(outputs,inputs["attention_mask"]).numpy()
    return outputs

def embed_codemalt(js_code):
    embeddings = codemaltmodel.encode([js_code])
    norm=embeddings/(np.linalg.norm(embeddings,axis=1,keepdims=True)+epsilon)
    return norm.flatten()

def extAllFeatures(url, content_type, tlu, _isjs, js_code, num_set_cookie):
    start=time.time()

    #ad_size_in_qs_present
    url_parsed=urlparse(url)
    query=url_parsed.query
    if adsize_reg.search(query):
        ad_size_in_qs_present=1
    else:
        ad_size_in_qs_present=0

    #is_third_party
    tlu_parsed=tldextract.extract(tlu)
    url_parsed2=tldextract.extract(url)
    tld=tlu_parsed.domain+"."+tlu_parsed.suffix
    urld=url_parsed2.domain+"."+url_parsed2.suffix
    is_third_party=0
    if tld and urld:
        if tld!=urld:
            is_third_party=1
    #keyword_char_present
    keyword_char_present = 0
    for key in keyword_raw:
        key_matches = [m.start() for m in re.finditer(key, url, re.I)]

        for key_match in key_matches:
            if url[key_match - 1] in keyword_char:
                keyword_char_present = 1
                break
        if keyword_char_present == 1:
            break

    #nomic embeddings
    nomic=embed_nomic([url,tld])
    url_nomic=nomic[0]
    fqdn_nomic=nomic[1]

    if _isjs:
        #content_policy_type
        te_content_type=0.411584860436903
        
        #js features
        jsfeatures=extJSFeature(js_code)
        codemalt_full=embed_codemalt(js_code)
        # print(codemalt_full.shape)
        codemalt=[]
        for i in codemalt_index:
            codemalt.append(codemalt_full[i])
        
        ast_depth=jsfeatures["ast_depth"]
        avg_charperline=jsfeatures["avg_charperline"]
        brackettodot=jsfeatures["brackettodot"]
        # num_set_cookie_=jsfeatures["num_set_cookie"]
        num_requests_sent=jsfeatures["num_requests_sent"]
        num_set_storage=jsfeatures["num_set_storage"]

    else:
        #content_policy_type
        te_content_type=contentencoder[content_type]

        #js features
        ast_depth=0
        avg_charperline=0
        brackettodot=0
        num_requests_sent=0
        # num_set_cookie_=num_set_cookie
        num_set_storage=0
        codemalt=np.zeros(len(codemalt_index))        
        
    #make envelope
    features={}
    features["ad_size_in_qs_present"]=ad_size_in_qs_present
    features["ast_depth"]=ast_depth
    features["avg_charperline"]=avg_charperline
    features["brackettodot"]=brackettodot
    for i in range(len(codemalt_index)):
        features[f"codemalt_{codemalt_index[i]}"]=codemalt[i]
    features["te_content_type"]=te_content_type
    for i in fqdn_nomic_index:
        features[f"fqdn_nomic_{i}"]=fqdn_nomic[i]
    features["is_third_party"]=is_third_party
    features["keyword_char_present"]=keyword_char_present
    features["num_requests_sent"]=num_requests_sent
    features["num_set_cookie"]=num_set_cookie
    features["num_set_storage"]=num_set_storage
    for i in url_nomic_index:
        features[f"url_nomic_{i}"]=url_nomic[i]
    features_str = {key: str(value) for key, value in features.items()}

    return features_str,time.time()-start

class Blocker:
    def request(self, flow: http.HTTPFlow) -> None:
        """
        This function is called for every single request.
        """
        if flow.request.host == "127.0.0.1":
            # print(flow.request)
            if flow.request.path=="/health-check":
                flow.response = http.Response.make(
                    200,  # Status code OK
                    b'{"status": "ok"}',
                    {"Content-Type": "application/json"}
                )
                print("[SYS] Connected to extension.")
                return

    def response(self,flow: http.HTTPFlow) -> None:
        if not flow.response or not flow.response.content:
            return
        keys=flow.request.headers.keys()
        if "afp-resource-type" not in keys:
            # print(f"[SYS] Passing system call: {flow.request.pretty_host}")
            return
        if "afp-top-level-url" not in keys:
            # print(f"[SYS] Blank request: {flow.request.pretty_host}")
            return
        if flow.request.pretty_host=="127.0.0.1":
            return
        headers=flow.request.headers
        res_headers=flow.response.headers
        num_set_cookie=0
        for key in res_headers.keys(multi=True):
            if key.lower()=="set-cookie":
                num_set_cookie+=1
        # print(num_set_cookie)

        content_type=headers["afp-resource-type"]
        if content_type=="main_frame":
            return

        url=flow.request.pretty_url
        tlu=headers["afp-top-level-url"]

        if content_type=="script":
            js_code=flow.response.text
            features_str,featexttime=extAllFeatures(url, content_type, tlu, True, js_code,num_set_cookie)
        else:
            features_str,featexttime=extAllFeatures(url, content_type, tlu, False, None,num_set_cookie)
            
        start=time.time()

        result=""
        proba=0
        try:
            response = requests.post("http://127.0.0.1:9090/predict", headers={"Content-Type": "application/json"}, data=json.dumps(features_str))
            if response.status_code == 200:
                res=response.json()
                result=res["predictedLabel"]
                proba=res["classProbabilities"][1]
                # print(result,proba)
            else:
                # Print an error message if something went wrong
                print(f"Response: {response.text}, status code: {response.status_code}")
                return

        except requests.exceptions.ConnectionError as e:
            print("Connection error: Is your Java server running?")
            return
        
        if result.lower()=="true" and ((proba>conf and content_type!="stylesheet") or (proba>cssconf and content_type=="stylesheet")) :
            print(f"🚫 Blocking request to: {str(flow.request.pretty_url)[:20]}... in {featexttime:.2f} + {time.time()-start:.2f} seconds")
            flow.kill()            
            return
        else:
            return


print("Running proxy...")
addons = [
    Blocker()
]