import google.generativeai as genai
import os
import hydra
import json
from  tqdm import tqdm
key = os.environ['GEMINI_KEY']
genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-1.0-pro-latest')

#result_descs = [x[4:-1] for x in result.text.split("\n")]
#print(result_descs)



def combine_prompt(problem_setup, cond_str, conditions, examples, current_utterance):
    prompt = f"{problem_setup}\n{cond_str}"
    for i, condition in enumerate(conditions):
        prompt = f"{prompt}\n{i+1}) {condition}"
    for i, example in enumerate(examples):
        prompt = f"{prompt}\n{i+1}) {example}"
    return f"{prompt}\n\n {current_utterance}"

def get_prompt(query):
    problem_setup = 'There is a scene where a "HUMAN" is pointing at a "TARGET OBJECT". \
There is an external "DESCRIPTION" of the target object which was given to POINT AT the TARGET OBJECT in case the human was not present. \
Assume that you are that human. What would you say to point at the object while doing the pointing gesture?"'

    cond_str = '\n\nGive special attention the following while answering:\n'

    conditions = [
        "There may be multiple objects of same class as the TARGET OBJECT.",
        "The human is POINTING at the TARGET OBJECT.",
        "Be super specific only in case of utmost necessity",
        "Do not add information that can not be directly inferred from the query",
        "Give any 3 possible distinct expressions and no other text.",
        "Do not use any special characters or punctuation marks other than period and comma.",
        #"T",
        "Output format is as follows:\n\
                1) Output expression 1\n2) Output expression 2\n3) Output expression 3"
    ]

    examples = [
    #    'NON-HUMAN DESCRIPTION: The large cupboard. The cupboard is along the left wall.\
    #    \nEXAMPLE OUTPUT 1: "That large cupboard."\
    #    \nEXAMPLE OUTPUT 2: "The cupboard over there."\
    #    \nEXAMPLE OUTPUT 3: "That cupboard by the wall."'
    ]
    current_utterance = f"CURRENT QUERY:\n'{query}'"
    return combine_prompt(problem_setup, cond_str, conditions, examples, current_utterance)

def process_split(base_file_path, result_file_path):
    results = []
    with open(base_file_path) as f:
        base_data = json.load(f)
    if os.path.isfile(result_file_path):
        with  open(result_file_path) as f:
            results = json.load(f)
        base_data = base_data[len(results):]
    bads = []
    for base_example in tqdm(base_data):
        result_example = {
                "scene_id": base_example["scene_id"],
                "object_id": base_example["object_id"],
                "object_name": base_example["object_name"],
                "ann_id": base_example["ann_id"],
                "base_description": base_example["description"]
                }
        prompt = get_prompt(base_example["description"])
        result = model.generate_content([prompt])
       # print(result.candidates)
        if not result.candidates==[] and result.candidates[0].finish_reason==1:
            result_descs = result.text.split("\n")
            result_example["potential_descriptions"] = result_descs
            results.append(result_example)
        else:
            result_example["potential_descriptions"] = []
            bads.append(result_example)
        with open(result_file_path, 'w') as f:
            json.dump(results, f, indent=4)
        with open(f"bads_.json", 'a') as f:
            json.dump(bads, f, indent=4)
    #Desc+Token

@hydra.main(version_base=None, config_path="../../config", config_name="global_imputed_config")
def main(cfg):
    print("\nDefault: using all CPU cores.")
    for split in ["test"]:
        base_file_path = getattr(cfg.data.base_lang_metadata, f"{split}_language_data")
        result_file_path = getattr(cfg.data.lang_metadata, f"{split}_language_data")
        process_split(base_file_path, result_file_path)
        #result = model.generate_content([prompt])
        #result_descs = [x[4:-1] for x in result.text.split("\n")]
        #print(result_descs)
    print(f"==> Complete.")
if __name__ == "__main__":
    main()

