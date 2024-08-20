import numpy as np
import sys
from time import sleep
import joblib

SEED = 1
SEED1 = 1

API_KEY = ""

def get_client():
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    return client

def exponential_backoff(client, messages, model='gpt-4-turbo-2024-04-09', max_retries=5, base_delay=1, id=0, json=False):
    # if "gpt4" in sys.argv:
    #     model = 'gpt-4-turbo-2024-04-09'
    # elif "gpt4o" in sys.argv:
    #     model = 'gpt-4o-2024-05-13'
    # else:
    #     model = 'gpt-3.5-turbo-0125'

    retries = 0
    while retries < max_retries:
        try:
            if not json:
                if "claude" not in model:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=2500
                    )
                else:
                    completion = client.messages.create(
                        model=model,
                        max_tokens=2500,
                        temperature=1.0,
                        system=messages[0]["content"],
                        messages=messages[1:]
                    )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=1024
                )
            return completion
        except Exception as e:
            print(f"Request failed: {str(e)}")
            delay = base_delay * (2 ** retries)
            print(f"Retrying in {delay} seconds...")
            sleep(delay)
            retries += 1
    raise Exception(f"Request failed even after retries. id: {id}")

def get_world_model(seed1=1):
    from lever_world import LeverBalanceWorld
    model = LeverBalanceWorld(normal="dirichlet" not in sys.argv)
    model.generate_world(seed=seed1, scale=1.)
    return model

def get_samples(model, n_samples=10, test=False, return_probs=False, seed=1):
    if test:
        data = model.generate_samples(num_samples=n_samples, seed=seed+42)
    else:
        data = model.generate_samples(num_samples=n_samples, seed=seed)
    samples = model.get_text_samples(data)
    if return_probs:
        _samples = model.get_samples(data)
        probs = [model.predict(input)[0] for input in _samples]
        return samples, probs
    return samples

def _get_client():
    from openai import OpenAI
    # from openai_client import get_client
    client = get_client()
    return client

def main(icl=False):
    global SEED
    global SEED1
    if "-seed" in sys.argv:
        SEED = int(sys.argv[sys.argv.index("-seed") + 1])
        SEED1 = SEED
    if "-seed1" in sys.argv:  # for structure
        SEED1 = int(sys.argv[sys.argv.index("-seed1") + 1])
    hint0 = "hint0" in sys.argv
    hint1 = "hint1" in sys.argv
    hint2 = "hint2" in sys.argv
    hint3 = "hint3" in sys.argv
    h = (int(hint0), int(hint1), int(hint2), int(hint3))

    if "gpt4" in sys.argv:
        m = 'gpt-4-turbo-2024-04-09'
    elif "gpt4o" in sys.argv:
        m = 'gpt-4o-2024-05-13'
    else:
        m = 'gpt-3.5-turbo-0125'

    client = _get_client()
    model = get_world_model()
    hints = ["We have a lever on a fulcrum with objects on the lever.\n",
             "Notice that some variables might be latent.\n",
             "Notice that the distance of the last object is latent.\n",
             "Notice that it is possible to reduce the number of variables to allow better learning with a linear model."]

    completions = {}
    for s1 in ([2,3] if icl else [2, 3, 4]):
        for s in ([1,2] if icl else [1, 2, 3]):
            ns = [10, 100, 1000] if icl and "gpt35" not in sys.argv else [10, 100] if icl else [10]
            for n in ns:
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                if icl:
                    samples = get_samples(model, seed=s, n_samples=n)
                    test_samples, probs = get_samples(model, n_samples=10, test=True, return_probs=True, seed=s)
                else:
                    samples = get_samples(model, seed=s, n_samples=10)
                content = f"""
                Assume we have a model representing physical setting. 
                {hints[0] if hint0 else ""}
                Here's a list of partial observations of the states of the model. 
                {hints[1] if hint1 else ""}{hints[2] if hint2 else ""}
                Samples:
                {samples}
                
                I want to learn the distribution using Statistics or Machine Learning. 
                Specifically, I want to use Logistic Regression to predict the balance probabilities of new samples. Here is an example of the code:
                ```python
                def fit_lr(X, y):
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(max_iter=10000, solver='saga')
                    model.fit(X, y)
                    return model
                    
                def predict_lr(model, X):
                    return model.predict_proba(X)
                ```
                
                Write me python function parse_samples(), that parses each sample and creates a feature function that can be used in the snippet above.
                Make sure the function is appropriate for both training and inference.
                Give me a code only."""

                if icl:
                    content = f"""
                    Assume we have a model representing a lever on a fulcrum, with two objects on it. The first object is on the right and the second is on the left.
         
                    I'll give you a list of partial observations of the states of the model. Notice that some values might be latent. Then I'll ask you to give me the probability for the continuation of some prompt, based on the distribution you can derive from the samples. Be prompt in your answer.
                    
                    Samples:
                    {samples}
                    
                    Question:
                    I'll give you a list of prompts. Give me a python list with the probabilities of "L", one probability for each input.
                    Samples:
                    {test_samples}
                    Give me a list only with no additional explanations."""

                messages.append({"role": "user", "content": content})
                completion = exponential_backoff(model=m, client=client, messages=messages)
                messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
                print(completion.choices[0].message.content)
                if icl:
                    completions[((s1, s), h, m, n)] = [completion.choices[0].message.content, probs]
                else:
                    completions[((s1, s), h, m)] = completion.choices[0].message.content
    joblib.dump(completions, f'/cs/snapless/oabend/eitan.wagner/calibration/BN scores/gpt_{m}_h{h}{"_icl" if icl else ""}.pkl')
    return

def evaluate_icl():
    h = (0, 0, 0, 0)
    completions = {}
    for m in ['gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13', 'gpt-3.5-turbo-0125']:
    # for m in ['gpt4', 'gpt4o', 'gpt35']:
        completions[m] = joblib.load(f'/cs/snapless/oabend/eitan.wagner/calibration/BN scores/gpt_{m}_h{h}_icl.pkl')

    probs = {}
    for k, kv in completions.items():
        print(k)
        probs[k] = []
        for s, sv in kv.items():
            sv0 = sv[0][sv[0].find("[")+1: sv[0].find("]")]
            sv0 = [float(s) for s in sv0.split(',')]
            probs[k].append(np.mean(abs(np.array(sv0) - sv[1])))
            # probs[k].append([sv0, sv[1]])

    for k, v in probs.items():
        print(k)
        print(np.mean(v))

    print("done")



#******************************8
def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=10000, solver='saga')
    model.fit(X, y)
    return model

def predict_lr(model, X):
    return model.predict_proba(X)

def run_lr(train_samples, test_samples):
    X, y = parse_samples(train_samples)
    testX = parse_samples(test_samples)
    model = fit_lr(X, fit_lr)
    probs = predict_lr(model, testX)
    return probs

# by GPT4
def parse_samples(samples):
    """
    Parse each sample into a feature list and balance outcome.
    """
    # Initialize lists to hold the feature vectors and balance outcomes
    X = []
    y = []
    for sample in samples:
        # Split each sample string by comma
        parts = sample.split(', ')
        # Initialize a dictionary to hold the features for this sample
        features = {}
        # Parse each part
        for part in parts:
            key, val = part.split(': ')
            # Check if the part describes a mass, distance, side, or balance
            if key.startswith('mass') or key.startswith('distance'):
                # Convert mass and distance values to floats
                features[key] = float(val)
            elif key.startswith('side'):
                # Convert side to a feature with 1 for R and -1 for L
                features[key] = 1 if val == 'R' else -1
            elif key == 'balance':
                # Balance is treated as the outcome, not a feature
                y.append(val)
        # Ensure features are in the consistent order and convert to feature vector
        feature_vector = [
            features.get('mass1', 0),
            features.get('distance1', 0),
            features.get('side1', 0),
            features.get('mass2', 0),
            # Side2 is implicitly provided by the balance and existing side; skipping it
            # Assuming distance2 is relevant but not provided; not adding it as a feature here
        ]
        X.append(feature_vector)
    return X, y

def summarize_scores(d):
    import pandas as pd
    # scores = {}
    rows = []
    for k, v in d.items():
        (seed1, seed), name = k
        model = "gpt35" if name[-1] == "5" else "gpt4" if name[-1] == "4" else "gpt4o"
        rows.append(pd.Series({"model": model, "seed1": seed1, "seed": seed, "score": v}))
    df = pd.concat(rows, axis=1).T
    df["score2"] = df["score"] < 0.1
    df2 = df.pivot_table(values=["score"], columns=["model"], aggfunc='mean')
    df3 = df.pivot_table(values=["score2"], columns=["model"], aggfunc='mean')
    # df2 = df.groupby("model", axis=0)
    return


def main2(load_functions=False):
    completions ={}
    for m in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13']:
    # for m in ['gpt-3.5-turbo-0125']:
        for h in [(1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0)]:
            completions.update(joblib.load(f'/cs/snapless/oabend/eitan.wagner/calibration/BN scores/gpt_{m}_h{h}.pkl'))

    name_conversion = {'gpt-3.5-turbo-0125': "gpt35",  'gpt-4-turbo-2024-04-09': "gpt4",  'gpt-4o-2024-05-13': "gpt4o"}
    # make functions
    if not load_functions:
        contents = []
        for (s, h, m), c in completions.items():
            content = c.split("python")[1].split("```")[0]
            content = content.replace("parse_samples(samples):", f"parse_samples_s{s[0]}{s[1]}_h{h[0]}{h[1]}{h[2]}{h[3]}_{name_conversion[m]}(samples):")
            contents.append(content)

        with open(f'/cs/snapless/oabend/eitan.wagner/calibration/BN scores/gpt_functions.py', "w") as file1:
            file1.write("\n".join(contents))

    else:
        import gpt_functions
        funcs = {}
        names = [f"parse_samples_s{s[0]}{s[1]}_h{h[0]}{h[1]}{h[2]}{h[3]}_{name_conversion[m]}" for (s, h, m), c in
                 completions.items()]
        for n in names:
            funcs[n] = getattr(gpt_functions, n)

        tvs = {}
        for s1 in [2, 3, 4]:
            for s in [1, 2, 3]:
                print((s1, s))
                model = get_world_model(seed1=s1)
                train_samples = get_samples(model, n_samples=100, seed=s)
                test_samples, probs = get_samples(model, n_samples=100, test=True, return_probs=True, seed=s)

                for n, func in funcs.items():
                    s1s = n.split("samples_s")[1].split("_h")[0]
                    if int(s1s[0]) != s1 or int(s1s[1]) != s:
                        continue
                    print(n)
                    try:
                        X_train, y_train = func(train_samples)
                        X_test, _ = func(test_samples)
                        lr_model = fit_lr(X_train, y_train)
                        preds = predict_lr(lr_model, X_test)
                        tvs[((s1, s), n)] = np.mean(np.abs(preds[:, 0] - probs))
                    except:
                        print("error")
                        tvs[((s1, s), n)] = 1.
        print(tvs)
        summarize_scores(tvs)


        return


if __name__ == "__main__":
    print(sys.argv)
    # evaluate_icl()
    # main(icl="icl" in sys.argv)
    # main2()
    main2(load_functions=True)
