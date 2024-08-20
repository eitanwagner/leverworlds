

# from pomegranate import *
import numpy as np
import torch
dev = "cuda" if torch.cuda.is_available() else torch.device("cpu")
import random
import sys
import pandas as pd
from transformers import OPTForCausalLM, BigBirdForCausalLM, BigBirdTokenizer
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, \
    Trainer, AutoConfig, RobertaForMaskedLM
from torch.utils.data import Dataset
from scipy.spatial import distance
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    pipeline,
    logging,
)
from accelerate.utils import load_and_quantize_model, BnbQuantizationConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from structure_learning import learn_lever_bn
import pgmpy

import logging
logging.basicConfig(level=logging.WARNING)
import datasets
datasets.disable_progress_bar()

results = {}
SEED = 1
SEED1 = 1
# # Dataset class
# class LDataset(Dataset):
#     def __init__(self, texts, prompt=""):
#         self.texts = [prompt + " " + t for t in texts]
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         return self.texts[idx]

import functools
def after_epoch(function):
    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        function(self, *args, **kwargs)
        print("Evaluating...")
        d, p_list, r_list, h_list = evaluate(model=self.bn_model, bn_marginal=self.bn_marginal, t_model=self.t_model,
                                             tokenizer=self.t_tokenizer,
                                             hidden="hidden" in sys.argv)
        r = evaluate_independencies(d)
        print("Evaluation on independent pairs")
        r.update(pair_evaluate(p_list))
        print("Evaluation on random pairs")
        r.update(pair_evaluate(r_list, random_list=True))
        print("Evaluation on heuristic cases")
        r.update(heuristic_evaluate(h_list))
        results[f"n_sample: {self.num_samples}, epoch: {self.epoch}"] = r
        self.epoch += 1
        print("\n")
    return wrapper

class CustomTrainer(Trainer):
    def __init__(self, t_model, t_tokenizer, bn_model, bn_marginal, num_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bn_model = bn_model
        self.bn_marginal = bn_marginal
        self.t_model = t_model
        self.t_tokenizer = t_tokenizer
        self.num_samples = num_samples
        self.epoch = 0

    # for adding custom evaluation after each epoch
    @after_epoch
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        # if self.control.should_log:
        #     if is_torch_tpu_available():
        #         xm.mark_step()
        #
        #     logs: Dict[str, float] = {}
        #
        #     # all_gather + mean() to get average loss over all processes
        #     tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
        #
        #     # reset tr_loss to zero
        #     tr_loss -= tr_loss
        #
        #     logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
        #     logs["learning_rate"] = self._get_learning_rate()
        #
        #     self._total_loss_scalar += tr_loss_scalar
        #     self._globalstep_last_logged = self.state.global_step
        #     self.store_flos()
        #
        #     self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

# Dataset class
class OPTDataset(Dataset):
    def __init__(self, texts, tokenizer, prompt="", only_last=True):
        # define variables
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        max_length = len(tokenizer(prompt + (" " if len(prompt) >= 1 else "") + texts[0], return_tensors="pt")["input_ids"][0]) + 30
        if "bbird" in sys.argv:
            max_length = 512
        # iterate through the dataset
        for txt in texts:
            # prepare the text
            p_encodings = tokenizer(prompt + (" " if len(prompt) >= 1 else "") + txt, padding="max_length",
                                    max_length=max_length, return_tensors="pt")
            # p_encodings = tokenizer(prompt + " " + txt, truncation=True, return_tensors="pt")
            # l_encodings = tokenizer(txt, truncation=True)
            l_encodings = torch.clone(p_encodings["input_ids"])
            if only_last:
                l_encodings[:, :-1] = -100  # prediction only for the last token
            l_encodings[:-len(tokenizer(txt, truncation=True)["input_ids"])+1] = -100
            # append to list
            self.input_ids.append(torch.tensor(p_encodings['input_ids']))
            self.attn_masks.append(torch.tensor(p_encodings['attention_mask']))
            self.labels.append(l_encodings)
            # self.labels.append(torch.tensor(l_encodings['input_ids']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]
        return {"input_ids": self.input_ids[idx].squeeze(),
                "attention_mask": self.attn_masks[idx].squeeze(),
                "labels": self.labels[idx].squeeze()}


class MLMDataset(Dataset):
    """
    Dataset object
    """
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is None:
            item['labels'] = item["input_ids"]
        else:
            item['labels'] = torch.tensor(self.labels[idx])
        # item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def create_sample_model():
    # example from https://www.cs.ubc.ca/~murphyk/Bayes/Charniak_91.pdf

    # family_out
    fo = DiscreteDistribution({'T': .15, 'F': .85})
    # fo = BernoulliDistribution(.15)
    # bowel_problem
    bp = DiscreteDistribution({'T': .01, 'F': .99})
    # dog_out
    do = ConditionalProbabilityTable(
        [['T', 'T', 'T', 0.99],
         ['T', 'T', 'F', 0.03],
         ['T', 'F', 'T', 0.9],
         ['T', 'F', 'F', 0.1],
         ['F', 'T', 'T', 0.97],
         ['F', 'T', 'F', 0.03],
         ['F', 'F', 'T', 0.3],
         ['F', 'F', 'F', 0.7]], [fo, bp])
    # light_on
    lo = ConditionalProbabilityTable(
        [['T', 'T', 0.6],
         ['T', 'F', 0.4],
         ['F', 'T', 0.05],
         ['F', 'F', 0.95]], [fo])
    # hear_bark
    hb = ConditionalProbabilityTable(
        [['T', 'T', 0.7],
         ['T', 'F', 0.3],
         ['F', 'T', 0.01],
         ['F', 'F', 0.99]], [do])


    s1 = Node(fo, name="family out")
    s2 = Node(bp, name="bowel problem")
    s3 = Node(do, name="dog out")
    s4 = Node(lo, name="light on")
    s5 = Node(hb, name="hear bark")

    model = BayesianNetwork("Barking Dog")
    model.add_states(s1, s2, s3, s4, s5)
    model.add_edge(s1, s3)
    model.add_edge(s2, s3)
    model.add_edge(s1, s4)
    model.add_edge(s3, s5)
    model.bake()
    return model

def create_MontyHall():
    # car location
    car = DiscreteDistribution({'A': 1/3, 'B': 1/3, 'C': 1/3})
    # guest's choice
    guest = DiscreteDistribution({'A': 1/3, 'B': 1/3, 'C': 1/3})
    # door Monty opens
    probs = []
    for i in ["A", "B", "C"]:
        for j in ["A", "B", "C"]:
            for k in ["A", "B", "C"]:
                probs.append([i, j, k, 0. if k in [i, j] else (1. if i != j else 0.5)])
    monty = ConditionalProbabilityTable(probs, [car, guest])

    s1 = Node(car, name="car location")
    s2 = Node(guest, name="guest's choice")
    s3 = Node(monty, name="door Monty opens")

    model = BayesianNetwork("Monty Hall")
    model.add_states(s1, s2, s3)
    model.add_edge(s1, s3)
    model.add_edge(s2, s3)
    model.bake()
    return model

def create_Cloudy():
    # part of the classic BN
    # is cloudy
    cloudy = DiscreteDistribution({'T': .5, 'F': .5})
    # sprinklers
    sprinklers = ConditionalProbabilityTable(
        [['T', 'T', 0.1],
         ['T', 'F', 0.9],
         ['F', 'T', 0.5],
         ['F', 'F', 0.5]], [cloudy])
    # rain
    rain = ConditionalProbabilityTable(
        [['T', 'T', 0.8],
         ['T', 'F', 0.2],
         ['F', 'T', 0.2],
         ['F', 'F', 0.8]], [cloudy])
    wet_grass = ConditionalProbabilityTable(
        [['T', 'T', 'T', 0.99],
         ['T', 'T', 'F', 0.01],
         ['T', 'F', 'T', 0.9],
         ['T', 'F', 'F', 0.1],
         ['F', 'T', 'T', 0.9],
         ['F', 'T', 'F', 0.1],
         ['F', 'F', 'T', 0.0],
         ['F', 'F', 'F', 1.]], [sprinklers, rain])
    # wet grass

    s1 = Node(cloudy, name="is cloudy")
    s2 = Node(sprinklers, name="turn on sprinklers")
    s3 = Node(rain, name="will it rain")
    s4 = Node(wet_grass, name="is the grass wet")

    model = BayesianNetwork("Cloudy")
    model.add_states(s1, s2, s3, s4)
    model.add_edge(s1, s2)
    model.add_edge(s1, s3)
    model.add_edge(s2, s4)
    model.add_edge(s3, s4)
    model.bake()
    return model

class Cloudy:
    def __init__(self):
        self.name = "Cloudy"
        self.states = ["is cloudy", "turn on sprinklers", "will it rain", "is the grass wet"]

    def sample(self, num_samples, evidences=None, random_state=SEED):
        def grass_cutoff_func(s, r):
            if s and r:
                return 0.99
            elif s or r:
                return 0.9
            else:  # not s and not r
                return 0.

        np.random.seed(random_state)
        # cloudy = np.random.randint(2, size=num_samples)
        cloudy = np.random.uniform(0, 1, size=num_samples) < 0.5
        sprinkers_cutoff = np.array([0.1 if c else 0.5 for c in cloudy])
        sprinklers = np.random.uniform(0, 1, size=num_samples) < sprinkers_cutoff
        rain_cutoff = np.array([0.8 if c else 0.2 for c in cloudy])
        rain = np.random.uniform(0, 1, size=num_samples) < rain_cutoff
        grass_cutoff = np.array([grass_cutoff_func(s, r) for s, r in zip(sprinklers, rain)])
        wet_grass = np.random.uniform(0, 1, size=num_samples) < grass_cutoff
        to_TF = lambda arr: np.array(["T" if b else "F" for b in arr])
        return np.vstack([to_TF(cloudy), to_TF(sprinklers), to_TF(rain), to_TF(wet_grass)]).T

class LeverBalance:
    def __init__(self, max_value, dist, normal=False, extras=None):
        self.max_value = max_value
        self.normal = normal
        self.dist = dist  # the multinomial distribution of the mean of the Gaussian
        self.name = "LeverBalance"
        self.extras = extras
        if extras is not None:
            self.states = []
            if "density1" in extras:
                self.states = self.states + ["object1 density", "object1 volume", "object1 location"]
            else:
                self.states = self.states + ["object1 mass", "object1 location"]
            if "direction" in extras:
                self.states = self.states + ["object1 side"]

            if "density2" in extras:
                self.states = self.states + ["object2 density", "object2 volume", "object2 location"] \
                              + (["object2 side"] if "direction" in extras else [])
            else:
                self.states = self.states + ["object2 mass", "object2 location"]
            if "direction" in extras:
                self.states = self.states + ["object2 side"]
            self.states = self.states + ["balance"]
        else:
            self.states = ["object1 location", "object2 location", "location ratio", "object1 mass", "object2 mass",
                           "mass ratio", "balance"]

    def sample(self, *args, **kwargs):
        return self._sample2(*args, **kwargs) if self.normal else self._sample(*args, **kwargs)

    def _sample(self, num_samples, evidences=None, random_state=SEED):
        samples = np.random.randint(1, 11, size=(num_samples, 7)).astype(str)
        np.random.seed(random_state)
        print("using dirichlet prior for probabilities")
        if self.dist is not None:
            samples[:, 4] = np.random.choice(np.arange(1, 11), size=num_samples, p=self.dist).astype(str)
        samples[:, 2] = np.around(samples[:, 0].astype(float) / samples[:, 1].astype(float), 3)
        samples[:, 5] = np.around(samples[:, 3].astype(float) / samples[:, 4].astype(float), 3)
        samples[:, 6] = ["L" if (int(s[0]) * int(s[3]) > int(s[1]) * int(s[4])) else "R" for s in samples]
        return samples

    def _sample2(self, num_samples, evidences=None, random_state=SEED):
        samples = np.random.randint(1, 6, size=(num_samples, 7)).astype(str)
        samples[:, 4] = np.random.normal(loc=self.dist, scale=2., size=num_samples)
        samples[:, 2] = np.around(samples[:, 0].astype(float) / samples[:, 1].astype(float), 3)
        samples[:, 5] = np.around(samples[:, 3].astype(float) / samples[:, 4].astype(float), 3)
        samples[:, 6] = ["L" if (int(s[0]) * int(s[3]) > int(s[1]) * float(s[4])) else "R" for s in samples]
        return samples

    def _sample3(self, num_samples, evidences=None, random_state=SEED):
        """
        for extras
        :param num_samples:
        :param evidences:
        :param random_state:
        :return:
        """
        # samples = np.zeros(size=(num_samples, 5 + len(self.extras))).astype(str)
        vars = []
        location1 = np.random.randint(1, 6, size=num_samples)
        if "density1" in self.extras:
            # and "density2" in self.extras:
            density1 = np.random.uniform(0.5, 1.5, size=num_samples)
            volume1 = np.random.randint(1, 6, size=num_samples)
            mass1 = density1 * volume1
            vars = vars + [density1, volume1]
        else:
            mass1 = np.random.randint(1, 6, size=num_samples)
            vars = vars + [mass1]
        if "direction" in self.extras:
            direction1 = np.random.randint(2, size=num_samples) - 1
            vars.append(direction1)
        else:
            direction1 = np.ones(num_samples)
            direction2 = - np.ones(num_samples)
        torque1 = mass1 * location1 * direction1
        vars.append(location1)

        location2 = np.random.normal(loc=self.dist, scale=2., size=num_samples)
        if "density2" in self.extras:
            density2 = np.random.uniform(0.5, 1.5, size=num_samples)
            volume2 = np.random.randint(1, 6, size=num_samples)
            mass2 = density2 * volume2
            vars = vars + [density2, volume2]
        else:
            mass2 = np.random.randint(1, 6, size=num_samples)
            vars = vars + [mass2]
        if "direction" in self.extras:
            direction2 = (np.random.randint(2, size=num_samples) - 1)
            vars.append(direction2)
        vars.append(location2)
        torque2 = mass2 * location2 * direction2
        balances = torque1 >= torque2
        vars.append(balances)

        samples = np.c_[vars]
        return samples

def predict_conditional(model, masked_state=None, values=None, p_values=None, hidden=False):
    """
    Should receive the number of the masked state and the values of the rest ("T" or "F")
    :param model:
    :param masked_state:
    :param values:
    :return:
    """
    if model.name == "LeverBalance":
        if not model.normal:
            _l_count = int(values[0] * values[3] / values[1])
            l_count = _l_count
            half = 0.
            if (values[0] * values[3] / values[1]) % 1 == 0 and values[0] * values[3] / values[1] <= model.max_value:
                l_count -= 1/2
                half = 1.
            if model.dist is None:
                return [min(l_count / model.max_value, 1.), 1 - min(l_count / model.max_value, 1.)]
            else:
                p = sum(model.dist[:int(l_count)]) + (0.5 * model.dist[int(_l_count) - 1] if half else 0.)
                # p = sum(model.dist[:int(l_count)]) + half * 0.5 * model.dist[int(_l_count) - 1]
                return [p, 1-p]
        else:
            from scipy.stats import norm
            e = values[0] * values[3] / values[1]
            p = norm.cdf(e, loc=model.dist, scale=2)
            return [p, 1 - p]
    if model.name == "Cloudy":
        if values[1] == "T" and values[2] == "T":
            return [0.99, 0.01]
        elif values[1] == "T" or values[2] == "T":
            return [0.9, 0.1]
        else:
            return [0., 1.]

def create_LeverBalance(balance=True, dists=None):
    # object1 location (m) to the left of the fulcrum
    if "degenerate" in sys.argv:
        min_value = 1
        min_value2 = 1
        max_value = 3
        max_value2 = 3
    else:
        min_value = 1
        min_value2 = 1
        max_value = 10
        max_value2 = 10
    probs = {i: 1 / (max_value - min_value + 1) for i in range(min_value, max_value + 1)}
    probs2 = probs

    if "degenerate" in sys.argv:
        probs[1] = 1.
        probs[2] = 0.
        probs[3] = 0.
    l1 = DiscreteDistribution(probs)
    # object2 location to the right
    l2 = DiscreteDistribution(probs2)
    # ratio
    ks1 = list(set([i / j for i in range(min_value, max_value + 1) for j in range(min_value, max_value + 1)]))
    ks1.sort()
    l_ratios = []
    for i in range(min_value, max_value + 1):
        for j in range(min_value, max_value + 1):
            for k in ks1:
                l_ratios.append([i, j, float("{:.3f}".format(k)), 1. if k == i/j else 0.])
    r_l1l2 = ConditionalProbabilityTable(l_ratios, [l1, l2])

    # object1 mass (kg)
    probs3 = {i: 1 / (max_value2 - min_value2 + 1) for i in range(min_value2, max_value2 + 1)}
    if "degenerate" in sys.argv:
        probs[3] = 1.
        probs[2] = 0.
        probs[1] = 0.
    m1 = DiscreteDistribution(probs3)
    # object2 mass
    if dists is None:
        probs4 = {i: 1 / (max_value2 - min_value + 1) for i in range(min_value, max_value2 + 1)}
    else:
        probs4 = {i + 1: dists[i - 1] for i in range(len(dists))}
    m2 = DiscreteDistribution(probs4)
    # ratio
    ks2 = list(set([i / j for i in range(min_value2, max_value2 + 1) for j in range(min_value, max_value2 + 1)]))
    ks2.sort()
    m_ratios = []
    for i in range(min_value2, max_value2 + 1):
        for j in range(min_value, max_value2 + 1):
            for k in ks2:
                # here we take the reverse ratio (m2/m1)
                m_ratios.append([i, j, float("{:.3f}".format(k)), 1. if k == j/i else 0.])
    r_m1m2 = ConditionalProbabilityTable(m_ratios, [m1, m2])

    # balance
    balances = []
    for lr in ks1:
        for mr in ks2:
            if lr > mr:
                correct = "L"
            elif lr < mr:
                correct = "R"
            elif lr == mr:
                correct = "B"
            for b in ["R", "L", "B"]:
                if balance or (correct != "B" and b != "B"):
                    balances.append([float("{:.3f}".format(lr)), float("{:.3f}".format(mr)), b, 1. if b == correct else 0.])
                elif correct == "B" and b != "B":
                    balances.append([float("{:.3f}".format(lr)), float("{:.3f}".format(mr)), b, 0.5])
    b = ConditionalProbabilityTable(balances, [r_l1l2, r_m1m2])

    s1 = Node(l1, name="object1 location")
    s2 = Node(l2, name="object2 location")
    s3 = Node(r_l1l2, name="location ratio")
    s4 = Node(m1, name="object1 mass")
    s5 = Node(m2, name="object2 mass")
    s6 = Node(r_m1m2, name="mass ratio")
    s7 = Node(b, name="balance")

    model = BayesianNetwork("LeverBalance")
    model.add_states(s1, s2, s3, s4, s5, s6, s7)
    model.add_edge(s1, s3)
    model.add_edge(s2, s3)
    model.add_edge(s4, s6)
    model.add_edge(s5, s6)
    model.add_edge(s6, s7)
    model.add_edge(s3, s7)
    model.bake()
    return model

def create_samples(model, num_samples=1000, evidences=None, return_raw=False, type='sample', hidden=False, shuffle=True, drop_values=False, marginal=False, full_e=False):
    print(f"Num generated samples: {num_samples}")
    print(f"SEED: {SEED}")
    if evidences is None:
        if marginal:
            np.random.seed(SEED)
            if type == "lever" and "full_e" not in sys.argv:
                if "dirichlet" in sys.argv:
                    samples = np.random.randint(1, 11, size=(num_samples, 7)).astype(str)
                    np.random.seed(SEED)
                    print("using dirichlet prior for probabilities")
                    alpha = np.ones(10)
                    # dists = np.random.dirichlet(alpha, size=4)
                    dists = np.random.dirichlet(alpha)
                    # samples = np.zeros((num_samples, 7))
                    # for i, idx in enumerate([0, 1, 3, 4]):
                    #     samples[:, idx] = np.random.choice(np.arange(1, 11), size=num_samples, p=dists[i])
                    samples[:, 4] = np.random.choice(np.arange(1, 11), size=num_samples, p=dists)
                    # samples = samples.astype(int).astype(str)
                elif "normal" in sys.argv:
                    samples = np.random.randint(1, 6, size=(num_samples, 7)).astype(str)
                    print("Using normal prior for latent variable")
                    np.random.seed(SEED)
                    # dists = np.random.dirichlet(alpha, size=4)
                    dists = np.random.uniform(1., 5.)
                    samples[:, 4] = np.random.normal(loc=dists, scale=2., size=num_samples)
                else:
                    samples = np.random.randint(1, 11, size=(num_samples, 7)).astype(str)
                if "degenerate" in sys.argv:
                    samples[:, [0, 1]] = "1"
                    samples[:, 3] = "3"
            elif type == "lever" and "full_e" in sys.argv:
                if "normal" not in sys.argv:
                    samples = np.zeros((1000, 7)).astype(str)
                    for i in range(1000):
                        samples[i, :] = [(i // 100) % 10 + 1, (i // 10) % 10 + 1, 1., i % 10 + 1, 1., 1., "M"]
                else:
                    samples = np.zeros((5**3, 7)).astype(str)
                    for i in range(5**3):
                        samples[i, :] = [(i // 25) % 5 + 1, (i // 5) % 5 + 1, 1., i % 5 + 1, 1., 1., "M"]
            else:
                samples = np.full((16, 4), "T")
                for i in range(16):
                    samples[i, :] = ["T" if b == "1" else "F" for b in ("000" + bin(i)[2:])[-4:]]
        else:
            samples = model.sample(num_samples, random_state=SEED)
        # samples = model.sample(num_samples)
    else:
        samples = model.sample(num_samples, evidences=evidences, random_state=SEED)

    # names = [s.name for s in model.states]
    names = model.states
    if "shuffle_names" in sys.argv:
        print("fixed shuffle_names")
        names = names[:-1][::-1] + names[-1:]
    elif "neutral_names" in sys.argv:
        print("fixed neutral_names")
        names = [f"Variable_{i+1}" for i in range(len(names)-1)] + names[-1:]
    print("names in training")
    print(names)

    # n_names = ["family not out", "no bowel problem", "dog not out", "light off", "didn't hear bark"]
    if type in ["sample", "cloudy"]:
        d_samples = [", ".join([n + ": yes" if _s == "T" else n + ": no" for _s, n in zip(s, names)]) for s in samples]
    elif type == "lever":
        ignore = []
        if hidden:
            ignore = ['location ratio', 'mass ratio']
            if "only_partial" in sys.argv:
                ignore = ignore + ['object2 mass']
        if shuffle:
            if "add_tokens" in sys.argv:
                d_samples = ["".join(random.sample([n + ": " + str(_s) for _s, n in zip(s, names) if n not in ignore], len(s)-len(ignore))) for s in samples]
            else:
                d_samples = [", ".join(random.sample([n + ": " + str(_s) for _s, n in zip(s, names) if n not in ignore], len(s)-len(ignore))) for s in samples]
        else:
            if "add_tokens" in sys.argv:
                d_samples = ["".join([n + ": " + str(_s) for _s, n in zip(s, names) if n not in ignore]) for s in samples]
            else:
                d_samples = [", ".join([n + ": " + str(_s) for _s, n in zip(s, names) if n not in ignore]) for s in samples]

    if return_raw:
        return d_samples, samples
    return d_samples

def _predict_conditional(model, masked_state, values=None, p_values=None, hidden=False):
    """
    Should receive the number of the masked state and the values of the rest ("T" or "F")
    :param model:
    :param masked_state:
    :param values:
    :return:
    """
    # values.insert(masked_state, "M")
    evidences = {s.name: v for s, v in zip(model.states, values) if (v != "M" and not (hidden and s.name.find("ratio") > -1))}
    if p_values is None:
        return model.predict_proba(evidences)[masked_state].parameters[0].get("T", 0.)
    else:
        return [model.predict_proba(evidences)[masked_state].parameters[0].get(pv, 0.) for pv in p_values]

# *************** MLE model **************
def MLE(samples=None):
    model = JointProbabilityTable.from_samples(samples)
    return model


class MLE2:
    def __init__(self, label_dict, value_range):
        from collections import defaultdict
        self.dict = defaultdict(lambda: np.full(len(label_dict), 1/len(label_dict)))
        self.label_dict = label_dict
        self.value_range = value_range

    def fit(self, samples, labels):
        data_dict = {" ".join(s): self.label_dict[l] for s, l in zip(samples, labels)}
        count_dict = {" ".join(s): np.zeros(len(self.label_dict)) for s in samples}
        if self.value_range is None:
            for s, l in zip(samples, labels):
                count_dict[" ".join(s)][self.label_dict[l]] += 1
        else:  # probably doesn't make a difference
            for k, v in data_dict.items():
                count_dict[k][v] += 1
        for k, v in count_dict.items():
            self.dict[k] = v / sum(v)
        return self

    def predict(self, sample, mask=None):
        # sample does not include the mask. in the form of a list
        # assuming now that mask is last
        if self.value_range is not None:
            marginal = np.zeros(len(self.label_dict))
            for v in self.value_range:
                marginal = marginal + self.dict[" ".join(sample + [str(v)])]
        else:
            marginal = self.dict[" ".join(sample)]
        return marginal / sum(marginal)

class MLE3:
    """ uses the multiplication of the first two features"""
    def __init__(self, label_dict, multiplication=True):
        from collections import defaultdict
        self.dict = defaultdict(lambda: np.zeros(len(label_dict)))
        self.label_dict = label_dict
        self.multiplication = multiplication
        self.dim = None

    def fit(self, samples, labels):
        self.dim = len(samples[0])
        for s, l in zip(samples, labels):
            if self.multiplication:
                n = (int(s[0]) * int(s[2]), int(s[1]))
            else:
                n = tuple(s)
                # n = (int(s[0]), int(s[2]), int(s[1]))
            self.dict[n][self.label_dict[l]] = self.dict[n][self.label_dict[l]] + 1

        # for k, v in self.dict.items():
        #     if sum(v) == 0.:
        #         self.dict[k] = np.ones(2)
        #     self.dict[k] = v / sum(v)
        return self

    def predict(self, sample, real=None):
        if self.multiplication:
            n = (int(sample[0]) * int(sample[2]), int(sample[1]))
        else:
            n = tuple(sample)
            # n = (int(sample[0]), int(sample[2]), int(sample[1]))
        if sum(self.dict[n]) == 0.:
            return np.ones(len(self.label_dict)) / len(self.label_dict)
        return self.dict[n] / sum(self.dict[n])


def MLE_conditional(mle_model, sample, masked_state=6, p_values=None):
    if p_values is None:
        if mle_model.keymap.get(tuple(['T'] + sample), None) is None or mle_model.log_probability(['T'] + sample) == -np.inf:
            return 0.
        if mle_model.keymap.get(tuple(['F'] + sample), None) is None or mle_model.log_probability(['F'] + sample) == -np.inf:
            return 1.
        return np.exp(mle_model.log_probability(['T'] + sample)) / \
               (np.exp(mle_model.log_probability(['T'] + sample)) + np.exp(mle_model.log_probability(['F'] + sample)))
    else:
        mle_model.marginal()
        p = np.zeros(len(p_values))
        for i, pv in enumerate(p_values):
            _sample = sample[:]
            _sample[masked_state] = pv
            p[i] = mle_model.keymap.get(tuple(_sample), 0.)
        if sum(p) == 0.:
            p = np.full(len(p_values), 1/len(p_values))
        return p

def MLE_conditional2(mle_model, sample, masked_states=None, marginal_state=4, p_state = 6, p_values=None):
    # if masked_states is None:
    masked_states = [2, 5]
    states = [i for i in range(len(mle_model.parameters[0][0])) if i not in masked_states]
    table = pd.DataFrame(mle_model.parameters[0])[states]
    table = table.drop_duplicates()
    table[7] = (table[7].astype(float) > 0) * 1
    # do not fix the marginal state
    c_table = table.loc[table[0] == sample[0]].loc[table[1] == sample[1]].loc[table[3] == sample[3]]
    c_table = c_table.loc[c_table[7] == 1]
    if c_table.shape[0] == 0:
        return np.full(len(p_values), 1/len(p_values))
    d = c_table[p_state].value_counts().to_dict()
    ps = np.array([d.get(pv, 0.) for pv in p_values])
    return ps / ps.sum()

# *************** deberta model *************

def mask_result(text, mask_token, result_prompt="balance: "):
    return text.split(result_prompt)[0] + result_prompt + mask_token + text.split(result_prompt)[1][1:]

def train_mlm(data, out_path, type="lever", bn_model=None, bn_marginal=None, num_samples=None):
    pretrained = "pretrained" in sys.argv
    size = "large" if "large" in sys.argv else "base"
    e_eval = "epoch_eval" in sys.argv
    epochs = 3
    if "-e" in sys.argv:
        epochs = int(sys.argv[sys.argv.index("-e") + 1])
    lr = 5e-5
    if "-lr" in sys.argv:
        lr = float(sys.argv[sys.argv.index("-lr") + 1])
    tokenizer = AutoTokenizer.from_pretrained(f"roberta-{size}", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    if pretrained:
        model = AutoModelForMaskedLM.from_pretrained(f"roberta-{size}", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    else:
        config = AutoConfig.from_pretrained(f"roberta-{size}")
        model = RobertaForMaskedLM(config=config)

    model.to(dev)
    if pretrained:
        prompt = ""
        if type == "lever":
            prompt = "We put two weights on a lever and check if it's balanced (\"B\"), leans to the right (\"R\") or leans to the left (\"L\")). " \
                     "Weight1 is towards the left and weight 2 is towards the right. " \
                     "We measure the locations of the weights, their mass, and the resulting balance. We got:\n"
        data = [prompt + d for d in data]

    print("not using data collator. masking labels instead")
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    train_texts, val_texts = data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):]
    if "testing" in sys.argv:
        train_texts, val_texts = train_texts[:10], val_texts[:10]
    l_train_encodings = tokenizer(train_texts, truncation=True, padding=True).input_ids
    train_encodings = tokenizer([mask_result(text, tokenizer.mask_token) for text in train_texts], truncation=True, padding=True)
    l_val_encodings = tokenizer(val_texts, truncation=True, padding=True).input_ids
    val_encodings = tokenizer([mask_result(text, tokenizer.mask_token) for text in val_texts], truncation=True, padding=True)

    train_dataset = MLMDataset(train_encodings, l_train_encodings)
    val_dataset = MLMDataset(val_encodings, l_val_encodings)
    # train_dataset = MLMDataset(train_encodings)
    # val_dataset = MLMDataset(val_encodings)
    # print("Train-set log perplexity before training:")
    # print(model_perplexity(model, tokenizer, train_texts))
    # print("Val-set log perplexity before training:")
    # print(model_perplexity(model, tokenizer, val_texts))
    print("Wikitext log perplexity before training:")
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="test")
    texts = [t for t in dataset['text'] if len(t) > 0]
    if "testing" not in sys.argv:
        print(model_perplexity(model, tokenizer, texts))

    training_args = TrainingArguments(
        output_dir='/cs/labs/oabend/eitan.wagner/checkpoints/results',          # output directory
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        gradient_accumulation_steps=1,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        logging_dir='./logs',            # directory for storing logs
        logging_steps=len(train_dataset)//100 + 1,
        save_strategy="no",
        learning_rate=lr,
        report_to="none",
        fp16=dev == "cuda",
        evaluation_strategy="epoch",
        disable_tqdm=True
    )

    if e_eval:
        trainer = CustomTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
                      t_model=model, t_tokenizer=tokenizer, bn_model=bn_model, bn_marginal=bn_marginal,
                      num_samples=num_samples)
    else:
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            # data_collator=data_collator,
        )
    trainer.train()

    # print("Train-set log perplexity after training:")
    # print(model_perplexity(model, tokenizer, train_texts))
    # print("Val-set log perplexity after training:")
    # print(model_perplexity(model, tokenizer, val_texts))
    print("Wikitext log perplexity after training:")
    if "testing" not in sys.argv:
        print(model_perplexity(model, tokenizer, texts))

    model.save_pretrained(out_path)
    print("done")

def create_opt_data(data, type="lever"):
    pretrained = "pretrained" in sys.argv
    if pretrained:
        prompt = ""
        if type == "lever":
            prompt = "We put two weights on a lever and check if it's balanced (\"B\"), leans to the right (\"R\") or leans to the left (\"L\")). " \
                     "Weight1 is towards the left and weight 2 is towards the right. " \
                     "We measure the locations of the weights, their mass, and the resulting balance. We got:\n"
        data = [prompt + d for d in data]
    return

def perplexity(model, dataset):
    loss = 0.
    for text in dataset:
        with torch.no_grad():
            loss += model(input_ids=text["input_ids"].unsqueeze(0).to(dev),
                          attention_mask=text["attention_mask"].unsqueeze(0).to(dev),
                          labels=text["labels"].unsqueeze(0).to(dev)).loss.item()
    return loss/len(dataset)

def model_perplexity(model, tokenizer, texts):
    ppl = 0.
    for text in texts:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            ppl += model(input_ids=inputs.input_ids.to(dev),
                          attention_mask=inputs.attention_mask.to(dev),
                          labels=inputs.input_ids.to(dev)).loss.item()
    return ppl/len(texts)

def train_llama(data, out_path, type="lever", perplexity='none'):
    """

    :param data:
    :param out_path:
    :param type:
    :param
    :return:
    """
    # https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing#scrollTo=OJXpOgBFuSrc
    from datasets import Dataset
    lr = 2e-4
    if "-lr" in sys.argv:
        lr = float(sys.argv[sys.argv.index("-lr") + 1])
    print("LORA training")
    pretrained = "pretrained" in sys.argv
    shuffled = "shuffle" in sys.argv
    prompt = ""
    if pretrained:
        if type == "lever" and "no_prompt" not in sys.argv:
            prompt = "We put two weights on a lever and check if it's balanced (\"B\"), leans to the right (\"R\") or leans to the left (\"L\")). " \
                     "Weight1 is towards the left and weight 2 is towards the right. " \
                     "We measure the locations of the weights, their mass, and the resulting balance. We got:\n"

    if shuffled:
        data = [d.split("balance: ")[0] + "balance: " + d[d.find("balance: ") + len("balance: ")] for d in data]
        if "only34" in sys.argv:
            data = [d for d in data if d.count(": ") > 3]
            print("Number of elements with 3 or 4 inputs:")
            print(len(data))
        print("Average number of elements:")
        print(sum([d.count(": ") for d in data]) / len(data))

    # The model that you want to train from the Hugging Face hub
    if "opt" not in sys.argv and "bbird" not in sys.argv:
        sizes = {"7b", "13b"}
        size = sizes.intersection(sys.argv)
        if len(size) >= 1:
            size = size.pop()
        else:
            size = "7b"
        model_name = f"meta-llama/Llama-2-{size}-hf"
        new_model = f"/llama-{size}"
    elif "opt" in sys.argv:
        sizes = {"125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b"}
        size = sizes.intersection(sys.argv).pop()
        # "125m" if "125m" in sys.argv else (
        # "2.7b" if "2.7b" in sys.argv else "1.3b" if "1.3b" in sys.argv else "350m")
        model_name = f"facebook/opt-{size}"
        new_model = f"/opt-{size}"
    else:
        size = "large" if "large" in sys.argv else "base" if "base" in sys.argv else ""
        # "125m" if "125m" in sys.argv else (
        # "2.7b" if "2.7b" in sys.argv else "1.3b" if "1.3b" in sys.argv else "350m")
        model_name = f"google/bigbird-roberta-{size}"
        new_model = f"/bbird-{size}"
    print(model_name)
    # Fine-tuned model name
    out_path = out_path + new_model

    ################################################################################
    # QLoRA parameters
    ################################################################################
    # LoRA attention dimension
    lora_r = 64
    if "-r" in sys.argv:
        lora_r = int(sys.argv[sys.argv.index("-r") + 1])
    # Alpha parameter for LoRA scaling
    lora_alpha = 16
    if "-alpha" in sys.argv:
        lora_alpha = int(sys.argv[sys.argv.index("-alpha") + 1])
    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################
    # Activate 4-bit precision base model loading
    use_4bit = torch.cuda.is_available()
    # use_4bit = False
    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16" if torch.cuda.is_available() else "float32"
    # bnb_4bit_compute_dtype = "float32"
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"
    # bnb_4bit_quant_type = "fp4"
    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    ################################################################################
    # TrainingArguments parameters
    ################################################################################
    # Output directory where the model predictions and checkpoints will be stored
    output_dir = "./results"
    # Number of training epochs
    num_train_epochs = 1
    if "-e" in sys.argv:
        num_train_epochs = int(sys.argv[sys.argv.index("-e") + 1])
    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False
    # Batch size per GPU for training
    per_device_train_batch_size = 1 if size in ["13b", "30b"] else 2 if size in ["1.3b", "2.7b"] else 4
    # Batch size per GPU for evaluation
    per_device_eval_batch_size = 4
    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 4 if size in ["13b", "30b"] else 2 if size in ["1.3b", "2.7b"] else 1
    if "-b" in sys.argv:
        per_device_train_batch_size = int(sys.argv[sys.argv.index("-b") + 1])
    if "-ga" in sys.argv:
        gradient_accumulation_steps = int(sys.argv[sys.argv.index("-ga") + 1])
    print(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    sys.stdout.flush()
    # Enable gradient checkpointing
    # gradient_checkpointing = True
    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3
    # Initial learning rate (AdamW optimizer)
    learning_rate = lr
    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001
    # Optimizer to use
    optim = "paged_adamw_32bit" if torch.cuda.is_available() else "adamw_torch"
    # Learning rate schedule
    lr_scheduler_type = "cosine"
    # Number of training steps (overrides num_train_epochs)
    max_steps = -1
    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03
    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True
    # Save checkpoint every X updates steps
    save_steps = 0
    # Log every X updates steps
    logging_steps = 250
    ################################################################################
    # SFT parameters
    ################################################################################
    # Maximum sequence length to use
    max_seq_length = None
    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False
    # Load the entire model on the GPU 0
    device_map = "auto" if "auto" in sys.argv else {"": 0} if torch.cuda.is_available() else None

    # Load LLaMA tokenizer
    if "bbird" in sys.argv:
        tokenizer = BigBirdTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    # train_texts, val_texts, test_texts = data[:int(len(data) * 0.8)], \
    #                                      data[-int(len(data) * 0.2):-int(len(data) * 0.1)], \
    #                                      data[-int(len(data) * 0.1):]
    print("Using all samples for training")
    train_texts, val_texts, test_texts = data[:], [], []

    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="test")
    texts = [t for t in dataset['text'] if len(t) > 0]
    if type == "lever":
        if "add_tokens" in sys.argv:
            response_template = "balance: "
            train_texts = [t.replace(" L", " pL").replace(" R", " pR").replace(" B", " pB") for t in train_texts]
        else:
            response_template = " balance:"
            if "natural" in sys.argv:
                response_template = " balance is"
            elif "=" in sys.argv:
                response_template = " balance="
    else:
        response_template = " grass wet:"

    if "lm_training" in sys.argv:
        # train_texts = [prompt + " " + t.replace("balance:", "<balance> balance:") for t in train_texts]
        train_texts = [prompt + " " + t for t in train_texts]
        train_texts = train_texts + [response_template + t for t in texts[:len(train_texts)]]
        train_dataset = Dataset.from_dict({"text": train_texts})
        # val_texts = [prompt + " " + t.replace("balance:", "<balance> balance:") for t in val_texts]
        val_texts = [prompt + " " + t for t in val_texts]
        val_texts = val_texts + [response_template + t for t in texts[-len(val_texts):]]
        val_dataset = Dataset.from_dict({"text": val_texts})
    else:
        train_dataset = Dataset.from_dict({"text": [prompt + " " + t for t in train_texts]})
        val_dataset = Dataset.from_dict({"text": [prompt + " " + t for t in val_texts]})
        # if len(train_texts) < 500:
        #     print(train_texts)
    if "print" in sys.argv:
        print("train texts")
        print(train_texts[:10])
        print(response_template)

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    # bnb_config=None
    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
            sys.stdout.flush()
    # Load base model
    if ("pretrained" in sys.argv or "llama" in sys.argv) and "add_tokens" not in sys.argv:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map
        )
        print("device:")
        print(model.device)
    else:
        if "pretrained" in sys.argv:
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map
            )
        else:
            config = AutoConfig.from_pretrained(model_name)
            _model = OPTForCausalLM(config)

        if "add_tokens" in sys.argv:
            print("adding tokens - phrase and value as one new token")
            from structure_learning import add_tokens
            override = "override" in sys.argv
            _model, tokenizer = add_tokens(_model, tokenizer, path=out_path, override=override)

        if torch.cuda.is_available():
            print("Loading on CUDA")
            sys.stdout.flush()
            _model.to("cuda")
            bnb_config = BnbQuantizationConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
            )
            model = load_and_quantize_model(_model, bnb_quantization_config=bnb_config, device_map=device_map)
            print("Loaded on CUDA")
            sys.stdout.flush()
        else:
            model = _model
        print("device:")
        print(model.device)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    # if hasattr(model, "enable_input_require_grads"):
    #     model.enable_input_require_grads()
    # else:
    #     def make_inputs_require_grad(module, input, output):
    #         output.requires_grad_(True)
    #
    #     model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if "add_tokens" in sys.argv and "used_tokens" not in sys.argv and "opt" not in sys.argv:
        print("adding tokens - phrase and value as one new token")
        from structure_learning import add_tokens
        override = "override" in sys.argv
        model, tokenizer = add_tokens(model, tokenizer, path=out_path, override=override)
        # model.enable_input_require_grads()
        # model, tokenizer = add_tokens(model, tokenizer, path=out_path)
    elif "used_tokens" in sys.argv:
        from structure_learning import SpecialTokenizer
        tokenizer = SpecialTokenizer(tokenizer)
    # elif "used_tokens" in sys.argv:
    #     from structure_learning import get_to_non_ascii_dict
    #     na_dict = get_to_non_ascii_dict(tokenizer)

    perplexities = []
    if perplexity in ['both']:
    # if "testing" not in sys.argv and "used_tokens" not in sys.argv:
        print("Wikitext log perplexity before training:")
        b_perplexity = model_perplexity(model, tokenizer, texts[:500])
        print(b_perplexity)
        perplexities.append(b_perplexity)

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        # save_steps=save_steps,
        save_strategy="no",
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="none",
        disable_tqdm=True
    )
    if "opt" in sys.argv or "add_tokens" in sys.argv and "used_tokens" not in sys.argv:
        print("Response template:")
        print(response_template)
        # print(tokenizer(response_template).input_ids)
        # print(train_texts[:10])
        # print(tokenizer(train_texts[:10]).input_ids)
        if "add_tokens" in sys.argv:
            response_template_ids = [tokenizer.convert_tokens_to_ids(response_template)]
            collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
        else:
            collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    elif "only_last" in sys.argv and "used_tokens" not in sys.argv:
        response_template_with_context = ", balance:"  # We added context here: "\n"
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:]
        collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    elif "used_tokens" in sys.argv:
        collator = DataCollatorForCompletionOnlyLM([tokenizer.token_dict["balance: "]], tokenizer=tokenizer)

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config if torch.cuda.is_available() else None,
        # dataset_text_field="text" if "opt" not in sys.argv else None,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        # tokenizer=tokenizer if "opt" not in sys.argv else None,
        data_collator=collator if ("opt" in sys.argv or "only_last" in sys.argv) else None,
        args=training_arguments,
        packing=packing if ("opt" not in sys.argv and "only_last" not in sys.argv) else False,
        # packing=packing,
    )
    # Train model

    # print([(n, p.requires_grad, p.dtype) for n, p in model.named_parameters()])
    # for p in model.parameters():
    #     p.requires_grad = True
    # print([p.requires_grad for n, p in model.named_parameters()])
    sys.stdout.flush()
    trainer.train()
    sys.stdout.flush()
    if perplexity in ['both', 'after']:
        print("Wikitext log perplexity after training:")
        # print(model_perplexity(trainer.model, tokenizer, texts[:500]))
        a_perplexity = model_perplexity(trainer.model, tokenizer, texts[:500])
        print(a_perplexity)
        perplexities.append(a_perplexity)
    # Save trained model
    # if "add_tokens" in sys.argv:
    #     model.disable_input_require_grads()
    # print(model.model.decoder.embed_tokens.weight)
    # print(model.model.decoder.embed_positions.weight)
    trainer.model.save_pretrained(out_path)
    trainer.tokenizer.save_pretrained(out_path, legacy_format=False)
    # tokenizer.save_pretrained(out_path)
    del model
    del trainer
    import gc
    gc.collect()
    return perplexities

def train_opt(data, out_path, type="lever", bn_model=None, bn_marginal=None, num_samples=None):
    print("fixed only_last")
    pretrained = "pretrained" in sys.argv
    shuffled = "shuffle" in sys.argv
    e_eval = "epoch_eval" in sys.argv
    epochs = 2
    if "-e" in sys.argv:
        epochs = int(sys.argv[sys.argv.index("-e") + 1])
    lr = 5e-5
    if "-lr" in sys.argv:
        lr = float(sys.argv[sys.argv.index("-lr") + 1])
    if "bbird" in sys.argv:
        size = "large" if "large" in sys.argv else "base" if "base" in sys.argv else ""
        # "125m" if "125m" in sys.argv else (
        # "2.7b" if "2.7b" in sys.argv else "1.3b" if "1.3b" in sys.argv else "350m")
        model_name = f"google/bigbird-roberta-{size}"
        from transformers import BigBirdConfig
        config = BigBirdConfig.from_pretrained(model_name, cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        config.block_size = 4
        config.num_random_blocks = 1
        config.is_decoder=True
        model = BigBirdForCausalLM.from_pretrained(model_name, config=config, cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        tokenizer = BigBirdTokenizer.from_pretrained(model_name)
        model.to(dev)
    else:
        size = "125m" if "125m" in sys.argv else ("2.7b" if "2.7b" in sys.argv else "1.3b" if "1.3b" in sys.argv else "350m")
        # load tokenizer and model
        if pretrained:
            if size in ["1.3b", "2.7b"]:
                model = OPTForCausalLM.from_pretrained(f"facebook/opt-{size}", device_map="auto")
            else:
                model = OPTForCausalLM.from_pretrained(f"facebook/opt-{size}")
                model.to(dev)
        else:
            config = AutoConfig.from_pretrained(f"facebook/opt-{size}")
            if size == "1.3b":
                model = OPTForCausalLM(config=config, device_map="auto")
            else:
                model = OPTForCausalLM(config=config)
                model.to(dev)
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-{size}")
        tokenizer.pad_token = tokenizer.eos_token

    if "add_tokens" in sys.argv:
        print("adding tokens - phrase and value as one new token")
        from structure_learning import add_tokens
        override = "override" in sys.argv
        model, tokenizer = add_tokens(model, tokenizer, override=override)

    prompt = ""
    if pretrained:
        if type == "lever":
            prompt = "We put two weights on a lever and check if it's balanced (\"B\"), leans to the right (\"R\") or leans to the left (\"L\")). " \
                     "Weight1 is towards the left and weight 2 is towards the right. " \
                     "We measure the locations of the weights, their mass, and the resulting balance. We got:\n"

    if shuffled:
        data = [d.split("balance: ")[0] + "balance: " + d[d.find("balance: ") + len("balance: ")] for d in data]
        print("Average number of elements:")
        print(sum([d.count(": ") for d in data]) / len(data))
    # prepare and load dataset
    train_texts, val_texts, test_texts = data[:int(len(data) * 0.8)], \
                                         data[-int(len(data) * 0.2):-int(len(data) * 0.1)], \
                                         data[-int(len(data) * 0.1):]
    # train_encodings = tokenizer(train_texts[:], truncation=True, padding=True)
    # val_encodings = tokenizer(val_texts[:], truncation=True, padding=True)
    # test_encodings = tokenizer(test_texts[:], truncation=True, padding=True)

    # train_dataset = MLMDataset(train_encodings)
    # val_dataset = MLMDataset(val_encodings)
    # test_dataset = MLMDataset(test_encodings)
    train_dataset = OPTDataset(train_texts, tokenizer, prompt=prompt, only_last=True)
    val_dataset = OPTDataset(val_texts, tokenizer, prompt=prompt, only_last=True)
    test_dataset = OPTDataset(test_texts, tokenizer, prompt=prompt, only_last=True)
    # train_dataset, val_dataset, test_dataset = create_opt_data(data=data)

    if "testing" not in sys.argv:
        print("Train-set log perplexity before training:")
        print(perplexity(model, train_dataset))
        print("Test-set log perplexity before training:")
        print(perplexity(model, test_dataset))
        print("Wikitext log perplexity before training:")
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="test")
    texts = [t for t in dataset['text'] if len(t) > 0]
    wt_dataset = OPTDataset(texts, tokenizer, prompt="", only_last=False)
    if "testing" not in sys.argv:
        print(perplexity(model, wt_dataset))
    ## Train
    #--------
    # creating training arguments
    batch_size = 8 if "1.3b" not in sys.argv else 1
    gradient_accumulation = 1
    if "bbird" in sys.argv:
        batch_size = 2
        gradient_accumulation = 4
    print("Batch size:")
    print(batch_size)
    print("Gradient accumulation:")
    print(gradient_accumulation)
    training_args = TrainingArguments(output_dir='results',
                                      num_train_epochs=epochs,
                                      logging_steps=100,
                                      # load_best_model_at_end=True,
                                      # load_best_model_at_end=False,
                                      save_strategy="no",
                                      evaluation_strategy="epoch",
                                      # evaluation_strategy="no",
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      gradient_accumulation_steps=gradient_accumulation,
                                      warmup_steps=500,
                                      # weight_decay=0.01,
                                      logging_dir='logs',
                                      fp16=dev == "cuda",
                                      learning_rate=lr,
                                      report_to="none",
                                      disable_tqdm=True)

    # start training
    if e_eval:
        CustomTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
                      t_model=model, t_tokenizer=tokenizer, bn_model=bn_model, bn_marginal=bn_marginal,
                      num_samples=num_samples).train()
    else:
        Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset).train()
                # data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                #                             'attention_mask': torch.stack([f[1] for f in data]),
                #                             'labels': torch.stack([f[0] for f in data])}).train()
    model.save_pretrained(out_path)
    print("Train-set log perplexity after training:")
    print(perplexity(model, train_dataset))
    print("Test-set log perplexity after training:")
    print(perplexity(model, test_dataset))
    print("Wikitext log perplexity after training:")
    if "testing" not in sys.argv:
        print(perplexity(model, wt_dataset))


def load_model(path, type="mlm"):
    size = "large" if "large" in sys.argv else "base"

    if type == "mlm":
        print("\nLoading MLM")
        tokenizer = AutoTokenizer.from_pretrained(f"roberta-{size}", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        model = AutoModelForMaskedLM.from_pretrained(path, cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        model.to(dev)
        model.eval()
    elif type == "llama" or "lora" in sys.argv:
        # dev = "cuda"
        if type == "llama":
            print("\nLoading Llama")
            sizes = {"7b", "13b"}
            size = sizes.intersection(sys.argv)
            if len(size) >= 1:
                size = size.pop()
            else:
                size = "7b"
            model_name = f"meta-llama/Llama-2-{size}-hf"
            new_model = f"/llama-{size}"
        else:
            if "bbird" in sys.argv:
                print("\nLoading BigBird")
                model_name = f"google/bigbird-roberta-{size}"
                new_model = f"/bbird-{size}"
            else:
                print("\nLoading LORA OPT")
                sizes = {"125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b"}
                size = sizes.intersection(sys.argv).pop()
            # "125m" if "125m" in sys.argv else (
            # "2.7b" if "2.7b" in sys.argv else "1.3b" if "1.3b" in sys.argv else "350m")
                model_name = f"facebook/opt-{size}"
                new_model = f"/opt-{size}"
        # use_4bit = True
        # bnb_4bit_compute_dtype = "float16"
        # bnb_4bit_quant_type = "nf4"
        # use_nested_quant = False
        # compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=use_4bit,
        #     bnb_4bit_quant_type=bnb_4bit_quant_type,
        #     bnb_4bit_compute_dtype=compute_dtype,
        #     bnb_4bit_use_double_quant=use_nested_quant,
        # )
        device_map = "auto"
        # Reload model in FP16 and merge it with LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            # quantization_config=bnb_config,
            device_map=device_map,
        )

        # Reload tokenizer to save it
        if "add_tokens" in sys.argv:
            print("loading new tokenizer")
            # if "opt" in sys.argv:
                # from transformers import GPT2TokenizerFast
                # tokenizer = GPT2TokenizerFast.from_pretrained(path + new_model, trust_remote_code=True)
            # else:
            tokenizer = AutoTokenizer.from_pretrained(path + new_model, trust_remote_code=True)
            print("Tokenizer vocab:", len(tokenizer.vocab))
            # print(base_model.model.decoder.embed_tokens.weight)
            # print(base_model.model.decoder.embed_positions.weight)
            base_model.resize_token_embeddings(len(tokenizer))
            # print("weights before loading")
            base_model.model.decoder.embed_tokens = torch.load(path + new_model + "/token_embeddings.pt")
            # base_model.model.decoder.embed_tokens.weight.requires_grad = True
            base_model.model.decoder.embed_positions = torch.load(path + new_model + "/pos_embeddings.pt")
            # base_model.model.decoder.embed_positions.weight.requires_grad = True
            base_model.model.decoder.embed_tokens.weight.to(dtype=base_model.model.decoder.final_layer_norm.weight.dtype)
            base_model.tie_weights()
            # print("weights after loading")
            # print(base_model.model.decoder.embed_tokens.weight)
            # print(base_model.model.decoder.embed_positions.weight)
            base_model.to(dev)
        else:
            if "bbird" in sys.argv:
                tokenizer = BigBirdTokenizer.from_pretrained(model_name, trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        model = PeftModel.from_pretrained(base_model, path + new_model)
        model = model.merge_and_unload()
        # print("weights after merging")
        # print(base_model.model.decoder.embed_tokens.weight)
        # print(base_model.model.decoder.embed_positions.weight)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.eval()
    else:
        if "bbird" in sys.argv:
            print("\nLoading BigBird")
            model_name = f"google/bigbird-roberta-{size}"
            tokenizer = BigBirdTokenizer.from_pretrained(model_name, cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        else:
            print("\nLoading OPT")
            sizes = {"125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b"}
            size = sizes.intersection(sys.argv).pop()
            # "125m" if "125m" in sys.argv else (
            # "2.7b" if "2.7b" in sys.argv else "1.3b" if "1.3b" in sys.argv else "350m")
            model_name = f"facebook/opt-{size}"
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        if "mlp" in sys.argv:
            model = torch.load(path + "/full_model.pt")
        elif "bbird" in sys.argv:
            model = BigBirdForCausalLM.from_pretrained(path, cache_dir="/cs/snapless/oabend/eitan.wagner/cache/", is_decoder=True)
            model.config.block_size = 4
            model.config.num_random_blocks = 1
        else:
            model = AutoModelForCausalLM.from_pretrained(path, cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        model.to(dev)
        model.eval()
    return model, tokenizer


def lm_probability(model, tokenizer, names, values, masked_state=0, label=None, p_values=None, type=None, i=-1, masked_value="", text_sample=None):
    pretrained = "pretrained" in sys.argv
    opt = "opt" in sys.argv
    llama = "llama" in sys.argv
    def make_input(values, names, ignore, use_full=False):
        mv = masked_value if use_full else ""
        if "add_tokens" in sys.argv:
            text = "".join([n + ": " + (str(_s) if _s != "M" else (mv if opt or llama else tokenizer.mask_token)) for _s, n in zip(values, names) if n not in ignore])
            # print(text)
        elif type in ["sample", "cloudy"]:
            text = ", ".join([n + ": yes" if _s == "T" else (n + ": no" if _s != "M" else n + ": ") for _s, n in zip(values, names)])
        elif type == "lever":
            text = ", ".join([n + ": " + (str(_s) if _s != "M" else (mv if opt or llama else tokenizer.mask_token)) for _s, n in zip(values, names) if n not in ignore])
        else:
            text = ""
        if use_full:
            if text[-1] != " ":
                return text[:-1]
        return text
    with torch.no_grad():
        # values1 = values[:]
        # values.insert(masked_state, "M")
        # values1.insert(masked_state, label)
        ignore = []
        if opt or llama:
            # ignore first mask
            ignore.append(names[3])

        prompt = ""
        if pretrained and type == "lever" and "no_prompt" not in sys.argv and "lever_world" not in sys.argv:
            prompt = "We put two weights on a lever and check if it's balanced (\"B\"), leans to the right (\"R\") or leans to the left (\"L\")). " \
                     "Weight1 is towards the left and weight 2 is towards the right. " \
                     "We measure the locations of the weights, their mass, and the resulting balance. We got:\n"

        if text_sample is None:
            text = make_input(values, names, ignore)
        else:
            text = text_sample + ", balance: "

        if not opt and not llama:
            inputs = tokenizer(prompt + text, return_tensors="pt").to(dev)
            # inputs = tokenizer(", ".join([n + ": Yes" if _s == "T" else n + ": No" if _s == "F" else n+ ": [MASK]" for _s, n in zip(values, names)]), return_tensors="pt")
            # label_ids = tokenizer([", ".join([n + ": Yes" if _s == "T" else n + ": No" for _s, n in zip(values1, names)])]).input_ids
            logits = model(**inputs).logits
            mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero()[-1]

            if p_values is None:
                yes_token = tokenizer("Yes")['input_ids'][1]
                no_token = tokenizer("No")['input_ids'][1]
                yes_token = 3216
                no_token = 440

                probs = logits[0, mask_token_index].softmax(axis=-1)
                yes_prob = probs[0, yes_token]
                return yes_prob.item()
            else:
                probs = logits[0, mask_token_index].softmax(axis=-1)
                p_ids = [l[1] for l in tokenizer([" " + pv for pv in p_values]).input_ids]
                # p = np.array([probs[0, tokenizer.convert_tokens_to_ids(pv)].item() for pv in p_values])
                p = np.array([probs[0, pid].item() for pid in p_ids])
                return p / sum(p)
        elif p_values is not None:
            inputs = tokenizer(prompt + text, return_tensors="pt")
            # print("text:", text)
            # print("Inputs:", inputs.input_ids)
            if not llama:
                inputs = inputs.to(dev)
            else:
                inputs = inputs.to("cuda")

            if "add_tokens" in sys.argv:
                p_values_ids = tokenizer.convert_tokens_to_ids(p_values)
            else:
                p_values_ids = tokenizer([" " + pv for pv in p_values]).input_ids
                p_values_ids = [pv[-1] for pv in p_values_ids]
            # if 0 <= i < 30:
            #     print(text)
            #     print(inputs)
            #     print(p_values_ids)
            if torch.cuda.is_available():
                from torch.cuda.amp import autocast
                with autocast(dtype=torch.float16):
                    generate_ids = model.generate(inputs.input_ids if "add_tokens" in sys.argv else inputs.input_ids[:, :-1], num_return_sequences=1, do_sample=False,
                                                  max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
            else:
                generate_ids = model.generate(
                    inputs.input_ids if "add_tokens" in sys.argv else inputs.input_ids[:, :-1], num_return_sequences=1,
                    do_sample=False,
                    max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
            # if llama:
            #     print(generate_ids)
            #     sys.stdout.flush()

            p = [generate_ids.scores[0].softmax(dim=-1)[0, pv].item() for pv in p_values_ids]
            if 0 <= i < 20 and "lora" in sys.argv and "print" in sys.argv:
                print("\n Tests:")
                print(text)
                print(p_values)
                # for j in range(5):
                #     print(f"\n {j} inputs:")
                #     _text = make_input(values[:j] + values[-1:], names[:j] + names[-1:], ignore=[], use_full=(j == 4))
                #     print(_text)
                #     inputs = tokenizer(prompt + _text, return_tensors="pt")
                #     # print("text:", text)
                #     # print("Inputs:", inputs.input_ids)
                #     if not llama:
                #         inputs = inputs.to(dev)
                #     else:
                #         inputs = inputs.to("cuda")
                #     if torch.cuda.is_available():
                #         from torch.cuda.amp import autocast
                #         with autocast(dtype=torch.float16):
                #             generate_ids = model.generate(
                #                 inputs.input_ids if "add_tokens" in sys.argv else inputs.input_ids[:, :-1],
                #                 num_return_sequences=1, do_sample=False,
                #                 max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
                #     else:
                #         generate_ids = model.generate(
                #             inputs.input_ids if "add_tokens" in sys.argv else inputs.input_ids[:, :-1],
                #             num_return_sequences=1,
                #             do_sample=False,
                #             max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
                #     #     print(inputs)
                #     print(tokenizer.convert_ids_to_tokens(generate_ids.scores[0][0].sort()[1][-5:]))
                #     print(generate_ids.scores[0].softmax(dim=-1)[0, generate_ids.scores[0][0].sort()[1][-5:]])
                #     # print(p)

            return np.array(p) / sum(p)



def marginal(model, idxs=None):
    # calculate joint
    p = np.zeros([2,  2, 2, 2, 2])
    for i0 in [0, 1]:
        for i1 in [0, 1]:
            for i2 in [0, 1]:
                for i3 in [0, 1]:
                    for i4 in [0, 1]:
                        v = ["T" if i0 else "F", "T" if i1 else "F", "T" if i2 else "F", "T" if i3 else "F",
                             "T" if i4 else "F"]
                        p[i0, i1, i2, i3, i4] = np.exp(model.log_probability(v))

    for idx in idxs:
        p = p.sum(axis=idx, keepdims=True)
    p = p.squeeze()
    return p

def mutual_information(preds):
    """
    between the first and the last, given the others
    :return:
    """
    from scipy.stats import entropy
    weights = np.array([0.5*0.5*0.8, 0.5*0.5*0.2, 0.5*0.5*0.8, 0.5*0.5*0.2, 0.5*0.9*0.2, 0.5*0.9*0.8, 0.5*0.1*0.2, 0.5*0.1*0.8])
    w_preds = weights * preds  # joint
    c_weights = weights[:4] + weights[4:]
    c_preds = (w_preds[:4] + w_preds[4:]) / c_weights  # probabilities for FF, FT.. in the middle

    H_YgivenX = np.array([entropy([p, 1 - p]) for p in preds]) @ weights
    H_Y = np.array([entropy([p, 1 - p]) for p in c_preds]) @ c_weights
    return H_Y -  H_YgivenX

def evaluate(model, mle_model=None, bn_full=None, bn_marginal=None, t_model=None, tokenizer=None, hidden=False):
    all_d = {}
    lb = model.name == "LeverBalance"
    cloudy = model.name == "Cloudy"
    generate_pairs = True
    with_b = "no_balance" not in sys.argv
    blankets = {3: []}
    sample_pairs = []
    if "cloudy" in sys.argv:
        weights = []
        _, samples = create_samples(model, num_samples=16, return_raw=True, type="cloudy", marginal=True, shuffle="shuffle" in sys.argv)  # or an untrained marginal one??
    elif "lever" in sys.argv:
        # _, samples = create_samples(bn_marginal, num_samples=500, return_raw=True, marginal=True, type="lever")  # or an untrained marginal one??
        _, samples = create_samples(model, num_samples=500, return_raw=True, marginal=True, type="lever", shuffle="shuffle" in sys.argv)  # or an untrained marginal one??
    elif "lever_world" in sys.argv:
        data = model.generate_samples(num_samples=500, seed=SEED, eval=True)
        samples = model.get_samples(data, with_output=False)
        samples2 = model.get_samples(model.eval_data2, with_output=False)
        format = "natural" if "natural" in sys.argv else "=" if "=" in sys.argv else ":"
        text_samples = model.get_text_samples(data, with_output=False, format=format)
        text_samples2 = model.get_text_samples(model.eval_data2, with_output=False, format=format)
        blankets = {1: 1}
        model.eval_data["real"] = 0.
        model.eval_data["MLM"] = 0.
        model.eval_data["MLE"] = 0.
    if "testing" in sys.argv:
        samples = samples[:50]
    if "degenerate" in sys.argv:
        generate_pairs = False
    print(f"Generate pairs: {generate_pairs}")

    h_samples = [[]]
    if lb:
        for s in samples:
            s[2] = str(float(s[0]) / float(s[1]))
            s[5] = str(float(s[3]) / float(s[4]))
        if generate_pairs:
            # generate Heuristic samples
            print("new method for heuristic samples - set_2 has lengths equal")
            set_3 = [np.array([i, i, 1., i, "M", -1., "M"]) for i in range(1, 11)]
            set_2 = []
            set_1 = []
            for s in set_3:
                set_2.append(s.copy())
                # j = np.random.choice([0, 1, 3])
                j, k, l = 3, 1, 0
                vals = [v for v in range(1, 11) if v != set_2[-1][j]]
                set_2[-1][j] = np.random.choice(vals)
                set_1.append(set_2[-1].copy())
                # k = np.random.choice([_k for _k in [0, 1, 3] if _k != j])
                # l = [_k for _k in [0, 1, 3] if _k not in [j, k]][0]
                vals = [v for v in range(1, 11) if v not in [set_2[-1][j], set_2[-1][k]]]
                set_1[-1][l] = np.random.choice(vals)
            h_samples = [set_3, set_2, set_1]

            for s in samples:
                multiplications = []
                if int(s[0]) <= 5 and int(s[1]) <= 5:
                    multiplications.append(2)
                if int(s[0]) <= 3 and int(s[1]) <= 3:
                    multiplications.append(3)
                if int(s[0]) <= 2 and int(s[1]) <= 2:
                    multiplications = multiplications + [4, 5]
                    multiplications.append(5)
                if int(s[0]) <= 1 and int(s[1]) <= 1:
                    multiplications = multiplications + [6, 7, 8, 9, 10]
                for m in multiplications:
                    n_s = np.copy(s)
                    n_s[0] = str(int(s[0]) * m)
                    n_s[1] = str(int(s[1]) * m)
                    sample_pairs.append([s, n_s])

                divisions = [d for d in range(2, 11) if int(s[0]) % d == 0 and int(s[1]) % d == 0]
                for d in divisions:
                    n_s = np.copy(s)
                    n_s[0] = str(int(s[0]) // d)
                    n_s[1] = str(int(s[1]) // d)
                    sample_pairs.append([s.copy(), n_s])

            for s_p in sample_pairs:
                s_p[1][2] = str(float(s_p[1][0]) / float(s_p[1][1]))
                s_p[1][5] = -1.
                s_p[1][6] = "M"
                s_p[0][6] = "M"
                s_p[1][4] = "M"  # mask one mass without predicting
                s_p[0][4] = "M"  # mask one mass without predicting

        masked_values = [s[4] for s in samples]
        for s in samples:
            s[4] = "M"
            s[6] = "M"
        blankets = {6: [2, 5]}
    for b, vs in blankets.items():
        print(f"B: {b}")
        print(f"Blanket: {vs}")
        if "lever_world" in sys.argv:
            sample_values = [tuple(_values) for _values in samples]
        else:
            sample_values = [" ".join(_values[:6].tolist() + ["M"]) for _values in samples]
        all_values = samples.tolist() + [s_p[1] for s_p in sample_pairs] + [s for h_s in h_samples for s in h_s]
        if "lever_world" in sys.argv:
            all_values = all_values + samples2.tolist()
            text_samples = text_samples + text_samples2
        for i, values in enumerate(all_values):
            # names = [s.name for s in model.states]
            names = model.states
            if "shuffle_names" in sys.argv:
                names = names[:-1][::-1] + names[-1:]
            elif "neutral_names" in sys.argv:
                names = [f"Variable_{i + 1}" for i in range(len(names) - 1)] + names[-1:]
            # print("names in evaluation")
            # print(names)
            if "lever_world" in sys.argv:
                _sample_name = tuple(values)
            else:
                _values = list(values)
                masked_value = "M"
                if i < len(samples) and len(sample_pairs) > 0:
                    masked_value = masked_values[i]
                _values[b] = "M"
                _sample_name = " ".join(_values)

            if _sample_name not in all_d:
                # if "testing" not in sys.argv:
                if "lever_world" in sys.argv:
                    bayes_pred = model.predict(values)
                    mle_pred = mle_model.predict(values, real=False)
                    if "logistic" in sys.argv:
                        from bayesian_mlp import predict_marginal
                        mlm_pred = predict_marginal(t_model, values, only_partial=True, num_classes=2)
                    else:
                        p_values = ["L", "R"]
                        with torch.no_grad():
                            mlm_pred = lm_probability(t_model, tokenizer, names, values=None,
                                                      p_values=p_values, type="lever", i=i, text_sample=text_samples[i])
                    mle_pred3 = np.array([1/2, 1/2])
                    mle_pred2 = np.array([1/2, 1/2])
                else:
                    f_values = [(float(v) if v.find(".") > -1 else int(v)) if v[0].isnumeric() else v for v in _values]
                    if lb:
                        p_v = ["L", "R", "B"] if with_b else ["L", "R"]
                        bayes_pred = predict_conditional(model, masked_state=b, values=f_values, p_values=p_v, hidden=hidden)
                    else:
                        bayes_pred = predict_conditional(model, masked_state=b, values=f_values, p_values=["T", "F"], hidden=hidden)
                    if mle_model is None:
                        mle_pred = np.array([1/3, 1/3, 1/3]) if (lb and with_b) else np.array([1/2, 1/2])
                    elif "s_learning" in sys.argv:
                        if "lever" in sys.argv:
                            df = pd.DataFrame(np.array(_values[:2] + _values[3:4]).reshape(1, -1), columns=names[:2] + names[3:4], dtype=float)
                        else:
                            i_values = np.array([["T", "F"].index(v) for v in _values[:3]])
                            df = pd.DataFrame(i_values.reshape(1, -1), columns=names[:3])
                        mle_pred = mle_model.predict_probability(df)
                        if "lever" not in sys.argv:
                            mle_pred = mle_pred[['is the grass wet_0', 'is the grass wet_1']].to_numpy().ravel()
                        else:
                            mle_pred = mle_pred[['balance_0.0', 'balance_1.0', 'balance_2.0']].to_numpy().ravel()
                        if abs(sum(mle_pred) - 1.) > 0.1:
                            print("**********************something's wrong... *****************")
                        # mle_pred = mle_pred.loc[0].array[-3:]
                    elif "hidden" not in sys.argv and "cloudy" not in sys.argv:
                        mle_pred = MLE_conditional2(mle_model=mle_model, sample=_values, p_values=["L", "R", "B"])
                    elif "cloudy" not in sys.argv:
                        mle_pred = mle_model.predict(_values[:2] + _values[3:4])
                    else:
                        mle_pred = mle_model.predict(_values[:3])
                    if bn_full is not None:
                        mle_pred2 = predict_conditional(bn_full, masked_state=b, values=_values, p_values=["L", "R", "B"])
                    else:
                        mle_pred2 = []
                    if bn_marginal is None:
                        mle_pred3 = np.array([1/3, 1/3, 1/3]) if (lb and with_b) else np.array([1/2, 1/2])
                    else:
                        if lb:
                            p_v = ["L", "R", "B"] if with_b else ["L", "R"]
                            mle_pred3 = predict_conditional(bn_marginal, masked_state=b, values=_values, p_values=p_v)
                        else:
                            mle_pred3 = predict_conditional(bn_marginal, masked_state=b, values=_values, p_values=["T", "F"])

                    mlm_values = _values[:]
                    if hidden:
                        mlm_values = _values[:2] + _values[3:5] + _values[6:]
                        _b = len(mlm_values) - 1
                        names = names[:2] + names[3:5] + names[6:]
                    else:
                        _b = b

                    if "logistic" in sys.argv:
                        from bayesian_mlp import predict_marginal
                        if "lever" in sys.argv:
                            mlm_pred = predict_marginal(t_model, [int(v) for v in mlm_values[:3]], only_partial="only_partial" in sys.argv, num_classes=3 if with_b else 2)
                        elif "cloudy" in sys.argv:
                            mlm_pred = predict_marginal(t_model, [["T", "F"].index(t) for t in mlm_values[:3]], only_partial=True, num_classes=2)
                        else:
                            mlm_pred = predict_marginal(t_model, mlm_values[:-1], only_partial=True, num_classes=2)
                    else:
                        with torch.no_grad():
                            # p_values = [f"balance: {b}" for b in ["L", "R", "B"]] if "add_tokens" in sys.argv else ["L", "R", "B"]
                            if "lever" in sys.argv or "lever_world" in sys.argv:
                                if with_b:
                                    p_values = ["pL", "pR", "pB"] if "add_tokens" in sys.argv else ["L", "R", "B"]
                                else:
                                    p_values = ["pL", "pR"] if "add_tokens" in sys.argv else ["L", "R"]
                            else:
                                p_values = ["pyes", "pno"] if "add_tokens" in sys.argv else ["yes", "no"]
                            if "mlp" not in sys.argv:
                                mlm_pred = lm_probability(t_model, tokenizer, names, masked_state=_b, values=mlm_values,
                                                          p_values=p_values, type="lever" if lb else "cloudy", i=i, masked_value=masked_value)
                            else:
                                from bayesian_mlp import mlp_probability
                                fixed_order = "fixed_order" in sys.argv
                                mlm_pred = mlp_probability(full_model=t_model, tokenizer=tokenizer, names=names[:-1],
                                                           values=mlm_values, fixed_order=fixed_order, i=i)
                if "lever_world" in sys.argv:
                    model.eval_data.loc[i, ["real", "MLM", "MLE"]] = [bayes_pred[0], mlm_pred[0], mle_pred[0]]
                all_d[_sample_name] = [bayes_pred, mlm_pred, mle_pred, mle_pred2, mle_pred3]
    d = {k: v for k, v in all_d.items() if k in sample_values} if lb else all_d
    p_list = []
    r_list = []
    vs = []
    if lb:
        for s_p in sample_pairs:
            p_list.append([all_d[" ".join(s_p[0])], all_d[" ".join(s_p[1])]])
        vs = random.choices(list(all_d.values()), k=2 * len(sample_pairs))
    elif cloudy:
        for k in d.keys():
            if k[0] == "F":
                p_list.append([d[k], d["T" + k[1:]]])
        vs = random.choices(list(all_d.values()), k=2 * len(p_list))
    if "lever_world" in sys.argv:
        for s_p in model.pairs:
            # _s_p = [[str(_s) for _s in s_p[0]], [str(_s) for _s in s_p[1]],]
            # p_list.append([all_d[" ".join(_s_p[0])], all_d[" ".join(_s_p[1])]])
            r_list.append([all_d[tuple(s_p[0])], all_d[tuple(s_p[1])]])
    else:
        r_list = [[vs[i], vs[i+1]] for i in range(0, len(vs), 2)]
    h_list = []
    for h_s in h_samples:
        h_list.append([all_d[" ".join(s)] for s in h_s])
    return d, p_list, r_list, h_list

def heuristic_evaluate(h_list):
    to_return = {}
    # Avg p for balanced
    bn_p = []
    mlm_p = []
    mle_p = []
    marginal_p = []
    # for k, v in reversed_dict2.items():
    for l in h_list:
        bn_p.append(np.mean([np.around(v[0][2], 3) for v in l]))
        mlm_p.append(np.mean([np.around(v[1][2], 3) for v in l]))
        mle_p.append(np.mean([np.around(v[2][2], 3) for v in l]))
        marginal_p.append(np.mean([np.around(v[4][2], 3) for v in l]))
    # print(f"Scores for pairwise comparison (mlm, mle, marginal), {len(reversed_dict2)} test pairs:")
    print(f"Average p(B) for heuristic sets (bn, mlm, mle, marginal), 3 values each (same, 2/3, different) \n{len(h_list)} test pairs:")
    print(bn_p)
    print(mlm_p)
    print(mle_p)
    print(marginal_p)
    to_return[f"Average p(B) for heuristic sets (bn, mlm, mle, marginal), 3 values each (same, 2/3, different)"] = [bn_p,
                                                                             mlm_p,
                                                                             mle_p,
                                                                             marginal_p]
    return to_return

def mi_evaluate(d):
    to_return = {}
    print(f"Mutual information (bn, mlm, mle, marginal), {len(d)} test pairs:")
    for k, v in d.items():
        bn_preds = np.array([v[0][0] for v in d.values()])
        mlm_preds = np.array([v[1][0] for v in d.values()])
        mle_preds = np.array([v[2][0] for v in d.values()])
        marginal_preds = np.array([v[4][0] for v in d.values()])
    to_return[f"Mutual information (bn, mlm, mle, marginal)"] = \
        [mutual_information(bn_preds), mutual_information(mlm_preds), mutual_information(mle_preds),
         mutual_information(marginal_preds)]
    print([mutual_information(bn_preds), mutual_information(mlm_preds), mutual_information(mle_preds),
         mutual_information(marginal_preds)])
    return to_return

def pair_evaluate(pair_list, random_list=False):
    to_return = {}

    bn_p_jss = []
    mlm_p_jss = []
    mle_p_jss = []
    marginal_p_jss = []
    # for k, v in reversed_dict2.items():
    for v in pair_list:
        # mlm_p_jss.append(distance.jensenshannon(np.around(d[v[1]][1], 3), np.around(d[v[0]][1], 3)))
        # mle_p_jss.append(distance.jensenshannon(np.around(d[v[1]][2], 3), np.around(d[v[0]][2], 3)))
        # marginal_p_jss.append(distance.jensenshannon(np.around(d[v[1]][4], 3), np.around(d[v[0]][4], 3)))
        bn_p_jss.append(distance.jensenshannon(np.around(v[1][0], 3), np.around(v[0][0], 3)))
        mlm_p_jss.append(distance.jensenshannon(np.around(v[1][1], 3), np.around(v[0][1], 3)))
        mle_p_jss.append(distance.jensenshannon(np.around(v[1][2], 3), np.around(v[0][2], 3)))
        marginal_p_jss.append(distance.jensenshannon(np.around(v[1][4], 3), np.around(v[0][4], 3)))
    # print(f"Scores for pairwise comparison (mlm, mle, marginal), {len(reversed_dict2)} test pairs:")
    print(f"Scores for pairwise comparison (bn, mlm, mle, marginal), {len(pair_list)} test pairs:")
    print(np.mean(bn_p_jss))
    print(np.mean(mlm_p_jss))
    print(np.mean(mle_p_jss))
    print(np.mean(marginal_p_jss))
    to_return[f"Scores for pairwise comparison (bn, mlm, mle, marginal){' - random_list' if random_list else''}"] = \
        [np.mean(bn_p_jss), np.mean(mlm_p_jss), np.mean(mle_p_jss), np.mean(marginal_p_jss)]
    return to_return

def pair_evaluate2(pair_list):
    def test_pair(p1, p2):
        if p2 > p1:
            return 1
        if 1 - p1 < 1e-2 and 1 - p2 < 1e-2:
            return 1
        if p1 < 1e-2 and p2 < 1e-2:
            return 1
        return 0

    to_return = {}
    mlm_r = []
    mle_r = []
    marginal_r = []
    # for k, v in reversed_dict2.items():
    for v in pair_list:
        mlm_r.append(test_pair(v[0][1][0], v[1][1][0]))
        mle_r.append(test_pair(v[0][2][0], v[1][2][0]))
        marginal_r.append(test_pair(v[0][4][0], v[1][4][0]))
        # mlm_r.append(1 if v[1][1][0] - v[0][1][0] > 0 or v[0][1][0] == v[1][1][0] == 1. or v[0][1][0] == v[1][1][0] == 0. else 0)
        # mle_r.append(1 if v[1][2][0] - v[0][2][0] > 0 or v[0][2][0] == v[1][2][0] == 1. or v[0][2][0] == v[1][2][0] == 0. else 0)
        # marginal_r.append(1 if v[1][4][0] - v[0][4][0] > 0 or v[0][4][0] == v[1][4][0] == 1. or v[0][4][0] == v[1][4][0] == 0. else 0)
    # print(f"Scores for pairwise comparison (mlm, mle, marginal), {len(reversed_dict2)} test pairs:")
    print(f"Scores for pair-ratio comparison (mlm, mle, marginal), {len(pair_list)} test pairs:")
    print(np.mean(mlm_r))
    print(np.mean(mle_r))
    print(np.mean(marginal_r))
    to_return[f"Scores for pair-ratio comparison (mlm, mle, marginal)"] = \
        [np.mean(mlm_r), np.mean(mle_r), np.mean(marginal_r)]
    return to_return

def evaluate_independencies(d, model=None):
    to_return = {}
    # d is a dictionary for one sample_size

    if "correlation_eval" in sys.argv:
        def convert_var(var):
            if var == "T":
                return 1
            elif var == "F":
                return 0
            return float(var)

        from mutual_information import evaluate_structure
        if "lever_world" in sys.argv:
            # n_vars = len(list(d.keys())[0])
            df = model.eval_data.drop([c for c in model.eval_data.columns
                                       if c[0] == "_" or c in ["balance", f"distance{model.num_objects}"]], axis=1)
            input_vars = list(model.visible_states)
            input_vars.remove("balance")# exclude the output
            # ["real", "MLM", "MLE"]
        else:
            n_vars = len(list(d.keys())[0].split()) - 1
            if "lever" in sys.argv:
                n_vars -= 3
            input_vars = [f'var{k}' for k in range(n_vars)]
            vars = np.zeros([len(d), n_vars + 3])  # for 3 values
            for i, (k, v) in enumerate(d.items()):
                if "lever_world" in sys.argv:
                    _vars = k
                else:
                    _vars = k.split()[:-1]
                if "lever" in sys.argv:
                    _vars = [_v for l, _v in enumerate(_vars) if l not in [2, 4, 5]]
                for j, var in enumerate(_vars):
                    vars[i, j] = convert_var(var)
                    vars[i, n_vars] = v[0][0]  # real probability
                    vars[i, n_vars+1] = v[1][0]  # mlm pred probability
                    vars[i, n_vars+2] = v[2][0]  # mle pred probability
            df = pd.DataFrame(vars, columns=input_vars + ["real", "MLM", "MLE"])
        if "bin_cmi" in sys.argv:
            for c in df:
                if c not in ["real", "MLM", "MLE"]:
                    df[c] = pd.cut(df[c], 2, labels=False)
            # for var in input_vars:
            #     df[var] = pd.cut(df[var], 2, labels=False)
        if "testing" in sys.argv:
            df = df.iloc[:50, :]
        normalize = "normalize_cmi" in sys.argv
        with_hidden = "with_hidden" in sys.argv
        mlm_cmi_diff, mlm_corr_diff = evaluate_structure(df, real_output="real", predicted_output="MLM", input_vars=input_vars, normalize=normalize, with_hidden=with_hidden)
        mle_cmi_diff, mle_corr_diff = evaluate_structure(df, real_output="real", predicted_output="MLE", input_vars=input_vars, normalize=normalize, with_hidden=with_hidden)

        print(f"Structure scores - MI - (MLM, MLE), {len(d)} test samples:")
        print(mlm_cmi_diff)
        print(mle_cmi_diff)
        to_return[f"Structure scores - MI - (MLM, MLE)"] = [mlm_cmi_diff, mle_cmi_diff]
        print(f"Structure scores - Correlation - (MLM, MLE), {len(d)} test samples:")
        print(mlm_corr_diff)
        print(mle_corr_diff)
        to_return[f"Structure scores - Correlation - (MLM, MLE)"] = [mlm_corr_diff, mle_corr_diff]

    # find prediction ratios
    bn_preds, mlm_preds, mle_preds = np.zeros(3), np.zeros(3), np.zeros(3)
    for k, v in d.items():
        bn_preds[np.argmax(v[0])] += 1
        mlm_preds[np.argmax(v[1])] += 1
        mle_preds[np.argmax(v[2])] += 1
        # if np.argmax(v[1]) == 2:
        #     print("Example of argmax==B:")
        #     print(k)
    print(f"Prediction ratios (BN, MLM, MLE), {len(d)} test samples:")
    print(bn_preds / len(d))
    print(mlm_preds / len(d))
    print(mle_preds / len(d))
    to_return[f"Prediction ratios (BN, MLM, MLE)"] = [bn_preds / len(d), mlm_preds / len(d), mle_preds / len(d)]

    bn_var = np.var([v[0][0] for v in d.values()], axis=0)
    mlm_var = np.var([v[1][0] for v in d.values()], axis=0)
    mle_var = np.var([v[2][0] for v in d.values()], axis=0)
    marginal_var = np.var([v[4][0] for v in d.values()], axis=0)
    print(f"Output variance for L (BN, MLM, MLE, marginal), {len(d)} test samples:")
    print(bn_var, mlm_var, mle_var, marginal_var)
    to_return[f"Output variance for L (BN, MLM, MLE, marginal)"] = [bn_var, mlm_var, mle_var, marginal_var]

    d_keys, d_values = list(zip(*d.items()))
    weights = np.array([0.5*0.5*0.8, 0.5*0.5*0.2, 0.5*0.5*0.8, 0.5*0.5*0.2, 0.5*0.9*0.2, 0.5*0.9*0.8, 0.5*0.1*0.2, 0.5*0.1*0.8])
    use_weighted = "cloudy" in sys.argv
    # TODO: weights for lever when using Dirichlet

    # find marginals
    bn_preds = np.mean([np.array(v[0]) for v in d.values()], axis=0)
    mlm_preds = np.mean([np.array(v[1]) for v in d.values()], axis=0)
    mle_preds = np.mean([np.array(v[2]) for v in d.values()], axis=0)
    marginal_preds = np.mean([np.array(v[4]) for v in d.values()], axis=0)
    print(f"Marginal ratios (BN, MLM, MLE, marginal), {len(d)} test samples:")
    print(bn_preds, mlm_preds, mle_preds, marginal_preds)
    to_return[f"Marginal ratios (BN, MLM, MLE, marginal)"] = [bn_preds, mlm_preds, mle_preds, marginal_preds]

    # jensen-shannon and total-variation distances from the real distribution
    mlm_jss = []
    mle_jss = []
    marginal_jss = []
    mlm_tv = []
    mle_tv = []
    marginal_tv = []
    # spearman rank correlation
    # from scipy import stats
    mlm_sr = []
    mle_sr = []
    marginal_sr = []
    for k, v in d.items():
        mlm_jss.append(distance.jensenshannon(np.around(v[0], 3), np.around(v[1], 3)))
        mle_jss.append(distance.jensenshannon(np.around(v[0], 3), np.around(v[2], 3)))
        marginal_jss.append(distance.jensenshannon(np.around(v[0], 3), np.around(v[4], 3)))
        mlm_tv.append(sum(abs(np.around(v[0], 3) - np.around(v[1], 3))) / 2)
        mle_tv.append(sum(abs(np.around(v[0], 3) - np.around(v[2], 3))) / 2)
        marginal_tv.append(sum(abs(np.around(v[0], 3) - np.around(v[4], 3))) / 2)
        mlm_sr.append(pd.Series(v[0]).corr(pd.Series(v[1]), method="spearman"))
        mle_sr.append(pd.Series(v[0]).corr(pd.Series(v[2]), method="spearman"))
        marginal_sr.append(pd.Series(v[0]).corr(pd.Series(v[4]), method="spearman"))
        # mlm_sr.append(stats.spearmanr(v[0], v[1], alternative="greater").pvalue)
        # mle_sr.append(stats.spearmanr(v[0], v[2], alternative="greater").pvalue)
        # marginal_sr.append(stats.spearmanr(v[0], v[4], alternative="greater").pvalue)
    print(f"Scores compared to gold distribution (mlm, mle, marginal), {len(d)} test samples:")
    print(np.mean(mlm_jss))
    print(np.mean(mle_jss))
    print(np.mean(marginal_jss))
    to_return[f"Scores compared to gold distribution (mlm, mle, marginal)"] = [np.mean(mlm_jss), np.mean(mle_jss), np.mean(marginal_jss)]
    print(f"Scores compared to gold distribution -TV- (mlm, mle, marginal), {len(d)} test samples:")
    print(np.mean(mlm_tv))
    print(np.mean(mle_tv))
    print(np.mean(marginal_tv))
    to_return[f"Scores compared to gold distribution -TV- (mlm, mle, marginal)"] = [np.mean(mlm_tv), np.mean(mle_tv), np.mean(marginal_tv)]
    print(f"Scores compared to gold distribution -Spearman rank correlation- (mlm, mle, marginal), {len(d)} test samples:")
    print(np.tanh(np.mean(np.arctanh(mlm_sr))))
    print(np.tanh(np.mean(np.arctanh(mle_sr))))
    print(np.tanh(np.mean(np.arctanh(marginal_sr))))
    to_return[f"Scores compared to gold distribution -Spearman rank correlation- (mlm, mle, marginal)"] = [np.mean(mlm_tv), np.mean(mle_tv), np.mean(marginal_tv)]

    # weighted scores
    if use_weighted:
        print(f"Weighted scores compared to gold distribution (mlm, mle, marginal), {len(d)} test samples:")
        print(np.array(mlm_jss) @ weights)
        print(np.array(mle_jss) @ weights)
        print(np.array(marginal_jss) @ weights)
        to_return[f"Weighted scores compared to gold distribution (mlm, mle, marginal)"] = [np.array(mlm_jss) @ weights, np.array(mle_jss) @ weights, np.array(marginal_jss) @ weights]
        print(f"Weighted scores compared to gold distribution -TV- (mlm, mle, marginal), {len(d)} test samples:")
        print(np.array(mlm_tv) @ weights)
        print(np.array(mle_tv) @ weights)
        print(np.array(marginal_tv) @ weights)
        to_return[f"Weighted scores compared to gold distribution -TV- (mlm, mle, marginal)"] = [np.array(mlm_tv) @ weights, np.array(mle_tv) @ weights, np.array(marginal_tv) @ weights]

        # test whether cloudy is conditionally independent of wet_grass
        diffs_mlm = np.array([v[1][0] for v in d_values[:4]]) - np.array([v[1][0] for v in d_values[4:]])
        diffs_mle = np.array([v[2][0] for v in d_values[:4]]) - np.array([v[2][0] for v in d_values[4:]])
        diffs_marginal = np.array([v[4][0] for v in d_values[:4]]) - np.array([v[4][0] for v in d_values[4:]])
        print(f"Effect of cloudy (mlm, mle, marginal), {len(d)} test samples:")
        print(abs(np.mean(diffs_mlm)))
        print(abs(np.mean(diffs_mle)))
        print(abs(np.mean(diffs_marginal)))
        to_return[f"Effect of cloudy (mlm, mle, marginal)"] = [abs(np.mean(diffs_mlm)), abs(np.mean(diffs_mle)), abs(np.mean(diffs_marginal))]

    # measures for binary
    if "no_balance" in sys.argv or "cloudy" in sys.argv:
        mlm_d = []
        mle_d = []
        marginal_d = []
        mlm_r = []
        mle_r = []
        marginal_r = []
        for k, v in d.items():
            mlm_d.append(np.around(np.abs(v[0][0] - v[1][0]), 3))
            mle_d.append(np.around(np.abs(v[0][0] - v[2][0]), 3))
            marginal_d.append(np.around(np.abs(v[0][0] - v[4][0]), 3))

            signs = np.sign([v[0][0] - v[0][1], v[1][0] - v[1][1], v[2][0] - v[2][1], v[4][0] - v[4][1]])
            incorrect = abs(signs[1:] - signs[0]) / 2
            # if v[0][0] - v[0][1] and
            mlm_r.append(incorrect[0])
            mle_r.append(incorrect[1])
            marginal_r.append(incorrect[2])
        print(f"Scores compared to gold distribution -distance of first- (mlm, mle, marginal), {len(d)} test samples:")
        print(np.mean(mlm_d))
        print(np.mean(mle_d))
        print(np.mean(marginal_d))
        to_return[f"Scores compared to gold distribution -distance of first- (mlm, mle, marginal)"] = [np.mean(mlm_d),
                                                                                        np.mean(mle_d),
                                                                                        np.mean(marginal_d)]
        print(f"Scores compared to gold distribution -distance in rank- (mlm, mle, marginal), {len(d)} test samples:")
        print(np.mean(mlm_r))
        print(np.mean(mle_r))
        print(np.mean(marginal_r))
        to_return[f"Scores compared to gold distribution -distance in rank- (mlm, mle, marginal)"] = [np.mean(mlm_r),
                                                                                        np.mean(mle_r),
                                                                                        np.mean(marginal_r)]

    if "lever" in sys.argv:
        # find pairs differing in only one values. the first (L) should have a larger probability for that balance
        ordered_pairs = set()
        mlm_count = mle_count = marginal_count = 0
        for k1 in d.keys():
            for k2 in d.keys():
                if sum(np.array(k1.split())[[0, 1, 3]] != np.array(k2.split())[[0, 1, 3]]) == 1:
                    ordered_pairs.add((k1, k2) if (int(k1.split()[0]) > int(k2.split()[0])
                                                   or int(k1.split()[1]) < int(k2.split()[1])
                                                   or int(k1.split()[3]) > int(k2.split()[3])) else (k2, k1))
        for k1, k2 in ordered_pairs:
            mlm_count += d[k1][1][0] >= d[k2][1][0]
            mle_count += d[k1][2][0] >= d[k2][2][0]
            marginal_count += d[k1][4][0] >= d[k2][4][0]  # should be perfect because it's always equal
        if len(ordered_pairs) > 0:
            print(f"Ratio of order preserving (mlm, mle, marginal), {len(ordered_pairs)} test pairs")
            print(mlm_count / len(ordered_pairs))
            print(mle_count / len(ordered_pairs))
            print(marginal_count / len(ordered_pairs))
            to_return[f"Ratio of order preserving (mlm, mle, marginal)"] = [mlm_count / len(ordered_pairs),
                                                                            mle_count / len(ordered_pairs),
                                                                            marginal_count / len(ordered_pairs)]
        else:
            to_return[f"Ratio of order preserving (mlm, mle, marginal)"] = [1.,
                                                                            1.,
                                                                            1.]
    return to_return

def make_df(d):
    import pandas as pd
    sizes = ["125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b"]
    num_samples = [10, 100, 500, 1000, 5000, 10000, 25000, 50000, 75000, 100000, 125000, 150000]
    results = {}
    for n_s in num_samples:
        results[n_s] = {}
    for s, v in enumerate(d.values()):
        for n_s in num_samples:
            results[n_s][s] = v[n_s]['Scores compared to gold distribution (mlm, mle, marginal)'][0]
    return pd.DataFrame(results)

def main_sample():
    model = create_sample_model()
    d = {}
    for n_samples in [10, 1000, 10000, 100000, 1000000]:
        print(f"\n****************")
        print(f"Num samples: {n_samples}")
        samples_ = create_samples(model, num_samples=n_samples, return_raw=True)
        mle_model = MLE(samples_)
        # bn_full = BayesianNetwork.from_structure(samples_, [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)],
        #                                          state_names=[s.name for s in model.states])
        bn_marginal = BayesianNetwork.from_structure(samples_, [(), (), (), (), ()],
                                                     state_names=[s.name for s in model.states])

        samples = create_samples(model, num_samples=n_samples)
        names = [s.name for s in model.states]
        print(names)
        # print(samples)
        # print(predict_conditional(model, masked_state=2, values=["T", "T", "T", "T"]))
        train_mlm(data=samples, out_path="/cs/labs/oabend/eitan.wagner/calibration/models/bayesian")
        t_model, tokenizer = load_model(path="/cs/labs/oabend/eitan.wagner/calibration/models/bayesian")
        # masked_states = range(1, 6)
        # value_lists = [["F", "F", "F", "F"], ["T", "T", "F", "T"], ["F", "F", "F", "F"]]

        d[n_samples] = {}
        blankets = {0: [1, 2, 3], 1: [0, 3], 2: [0, 1, 4], 3: [0], 4: [2]}
        for b, vs in blankets.items():
            print(f"B: {b}")
            print(f"Blanket: {vs}")
            for i in range(16):
                values = ["T" if i % 2 >= 1 else "F", "T" if i % 4 >= 2 else "F", "T" if i % 8 >= 4 else "F",
                          "T" if i % 16 >= 8 else "F"]
                # print(values)
                _values = values[:]
                _values.insert(b, "M")
                bayes_pred = predict_conditional(model, masked_state=b, values=_values)
                # mle_pred = mle_model.marginal()

                mle_pred = MLE_conditional(mle_model=mle_model, sample=values)
                mle_pred2 = []
                # mle_pred2 = predict_conditional(bn_full, masked_state=b, values=_values)
                mle_pred3 = predict_conditional(bn_marginal, masked_state=b, values=_values)
                mlm_pred = lm_probability(t_model, tokenizer, names, masked_state=b, values=values)
                d[n_samples][" ".join(_values)] = [bayes_pred, mlm_pred, mle_pred, mle_pred2, mle_pred3]
        print("Results: ")
        print(d[n_samples])
    print(d)

def main():
    global SEED
    global SEED1
    if "-seed" in sys.argv:
        SEED = int(sys.argv[sys.argv.index("-seed") + 1])
        SEED1 = SEED
    if "-seed1" in sys.argv:  # for structure
        SEED1 = int(sys.argv[sys.argv.index("-seed1") + 1])
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    dc = ["-dc", str(torch.cuda.device_count())]
    _results = {}
    results[' '.join(sys.argv[1:] + dc)] = _results
    s_epochs = "_"
    s_lr = ""
    s_lr2 = ""
    degree = 2
    perplexities = []
    if "-e_" in sys.argv:
        s_epochs = sys.argv[sys.argv.index("-e") + 1]
    if "-lr" in sys.argv:
        s_lr = sys.argv[sys.argv.index("-lr") + 1]
    if "-lr2" in sys.argv:
        s_lr2 = sys.argv[sys.argv.index("-lr2") + 1]
    if "-d" in sys.argv: # degree for logistic regression
        degree = int(sys.argv[sys.argv.index("-d") + 1])
    sizes = {"125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b"}
    size = sizes.intersection(sys.argv)
    if len(size) >= 1:
        size = size.pop()
    else:
        size = "large" if "large" in sys.argv else "base" if "base" in sys.argv else ""
    # model = create_MontyHall()
    type = ""
    with_b = "no_balance" not in sys.argv
    if "lever" in sys.argv:
        if "dirichlet" in sys.argv:
            print("Using dirichlet prior for probabilities")
            print("Fixed the prediction function!!")
            alpha = np.ones(10)
            np.random.seed(SEED)
            # dists = np.random.dirichlet(alpha, size=4)
            dists = np.random.dirichlet(alpha)
        elif "normal" in sys.argv:
            print("Using normal prior for latent variable")
            np.random.seed(SEED)
            # dists = np.random.dirichlet(alpha, size=4)
            dists = np.random.uniform(1., 5.)
        else:
            dists = None
        # model = create_LeverBalance(balance=with_b, dists=dists)
        model = LeverBalance(max_value=10, dist=dists, normal="normal" in sys.argv)
        type = "lever"
    elif "lever_world" in sys.argv:
        from lever_world import LeverBalanceWorld
        model = LeverBalanceWorld(normal="dirichlet" not in sys.argv)
        model.generate_world(seed=SEED1, scale=1.)
        np.random.seed(SEED)
        type = "lever"
    elif "cloudy" in sys.argv:
        # model = create_Cloudy()
        model = Cloudy()
        type = "cloudy"
    # for n_samples in [10, 1000, 10000, 100000]:
    texts = {}
    # sample_quantities = [10, 50, 100, 500, 1000, 5000, 10000, 25000, 50000, 75000, 100000, 125000, 150000]
    sample_quantities = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 75000, 100000]
    if "more_quantities" in sys.argv:
        sample_quantities = [5, 15, 25, 40, 50, 75, 100, 150, 200, 250, 400, 500, 750, 1000, 1500, 2000, 2500, 3500, 5000, 7500, 10000, 15000, 20000, 25000, 35000, 50000]
    if "less_quantities" in sys.argv:
        sample_quantities = [500, 1000, 10000, 50000, 100000, 150000, 200000]
        # sample_quantities = [10, 100, 1000, 10000, 50000, 100000, 150000, 200000]
    if "size" == "30b" or "mlp" in sys.argv or "used_tokens" in sys.argv:
        sample_quantities = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 150000]
    for n_samples in sample_quantities:
        # if "logistic" in sys.argv and n_samples in [] and with_b:
        #     continue
        _results[n_samples] = {}
        d = {}
        print(f"\n****************")
        print(f"Num training samples: {n_samples}")
        sys.stdout.flush()
        if "lever_world" in sys.argv:
            data = model.generate_samples(num_samples=n_samples, seed=SEED)
            samples_ = model.get_samples(data)
            format = "natural" if "natural" in sys.argv else "=" if "=" in sys.argv else ":"
            samples = model.get_text_samples(data, format=format)
        else:
            samples, samples_ = create_samples(model, num_samples=n_samples, return_raw=True, type=type,
                                               hidden="hidden" in sys.argv, shuffle="shuffle" in sys.argv)
        texts[n_samples] = samples
        # if type == "cloudy":
        #     print(samples[:200])
        mle_model = None
        o_samples_ = samples_.copy()
        s_samples_ = samples_.copy()
        if "only_full" in sys.argv and "cloudy" not in sys.argv:
            # names = [s.name for s in model.states if "ratio" not in s.name]
            names = [s for s in model.states if "ratio" not in s]
            # balance_locs = [s.split("balance: ")[0].count(": ") for s in samples]
            is_full = np.array(
                [len([names.index(n) for n in names if n in s and s.index(n) <= s.index(names[-1])]) for s
                 in samples]) == 5
            samples_ = samples_[is_full]

        if "no_mle" not in sys.argv:
            print("fixed MLE for stochastic case")
            if "cloudy" in sys.argv and "s_learning" not in sys.argv:
                label_dict = {"T": 0, "F": 1}
                mle_model = MLE2(label_dict, None).fit(samples_[:, :3], samples_[:, 3],)
            elif "hidden" not in sys.argv and "s_learning" not in sys.argv and "lever_world" not in sys.argv:
                mle_model = MLE(samples_)
            elif "s_learning" in sys.argv:  # assuming hidden
                if "lever" in sys.argv:
                    s_samples_[:, 6] = [["L", "R", "B"].index(b) for b in s_samples_[:, 6]]
                    s_samples_ = s_samples_[:, [0, 1, 3, 4, 6]].astype(int)
                else:
                    s_samples_[:, :3] = np.array([[["T", "F"].index(_t) for _t in t] for t in s_samples_[:, :3]])
                    s_samples_[:, 3] = [["T", "F"].index(t) for t in s_samples_[:, 3]]
                    s_samples_ = s_samples_.astype(int)
                # names = [s.name for s in model.states if "ratio" not in s.name]
                names = model.states
                # balance_locs = [s.split("balance: ")[0].count(": ") for s in samples]
                in_indices = [[names.index(n) for n in names if n in s and s.index(n) <= s.index(names[-1])] for s in samples]
                mask = np.full_like(s_samples_, True, dtype=bool)
                for i in range(len(mask)):
                    mask[i, in_indices[i]] = False
                samples_df = pd.DataFrame(s_samples_, columns=names).mask(mask, np.nan)
                mle_model = learn_lever_bn(samples_df, with_latent=False, type="lever" if "lever" in sys.argv else "cloudy")
            elif len(samples_) > 0:
                label_dict = {"L": 0, "R": 1, "B": 2} if with_b else {"L": 0, "R": 1}
                if "bayesian_dirichlet" in sys.argv:
                    if "lever_world" in sys.argv:
                        mle_model = model.fit(samples_)
                    else:
                        from bayesian_mlp import DirichletEstimator
                        mle_model = DirichletEstimator(label_dict, dim=10).fit(samples_[:, [0, 1, 3]], samples_[:, 6])
                elif "mle3" in sys.argv:
                    print("Using MLE3 with multiplication)")
                    mle_model = MLE3(label_dict, multiplication=True).fit(samples_[:, [0, 1, 3]], samples_[:, 6],)
                else:
                    print("Using MLE3 with no multiplication)")
                    if "lever_world" in sys.argv:
                        label_dict = {1: 0, -1: 1}
                        mle_model = MLE3(label_dict, multiplication=False).fit(samples_[:, :-1], samples_[:, -1],)
                    else:
                        mle_model = MLE3(label_dict, multiplication=False).fit(samples_[:, [0, 1, 3]], samples_[:, 6],)
                    # mle_model = MLE2(label_dict, range(1, 11)).fit(samples_[:, [0, 1, 3, 4]], samples_[:, 6],)
        # bn_full = BayesianNetwork.from_structure(samples_, [tuple(range(i)) for i in range(len(model.states))],
        #                                          state_names=[s.name for s in model.states])
        bn_marginal = None
        # if len(samples_) > 0:
        #     bn_marginal = BayesianNetwork.from_structure(samples_, [()] * len(model.states),
        #                                                  state_names=[s.name for s in model.states])
        # else:
        #     bn_marginal = BayesianNetwork.from_structure(o_samples_, [()] * len(model.states),
        #                                                  state_names=[s.name for s in model.states])

        # samples = create_samples(model, num_samples=n_samples, type='lever_balance', hidden=False)
        suffix = ("_pp" if "pretrained" in sys.argv else "") + ("_l" if "lever" in sys.argv else "") \
                 + ("_h" if "hidden" in sys.argv else "") + ("_bbird" if "bbird" in sys.argv else "") + ("_opt" if "opt" in sys.argv else "") + ("_llama" if "llama" in sys.argv else "") \
                 + s_epochs + size + ("_mlp" if "mlp" in sys.argv else "") + ("_d" if "degenerate" in sys.argv else "") \
                 + ("_at" if "add_tokens" in sys.argv else "") + ("o" if "override" in sys.argv else "") + ("_logistic" if "logistic" in sys.argv else "") \
                 + ("_op" if "only_partial" in sys.argv else "") + ("_s" if "shuffle" in sys.argv else "") + ("34" if "only34" in sys.argv else "")
        print("Suffix: ")
        print(suffix)
        if ("opt" in sys.argv or "bbird" in sys.argv) and "lora" not in sys.argv and "mlp" not in sys.argv:
            train_opt(data=samples, out_path="/cs/labs/oabend/eitan.wagner/calibration/models/bayesian" + suffix,
                      bn_model=model, bn_marginal=bn_marginal, num_samples=n_samples)
        elif "llama" in sys.argv or "lora" in sys.argv:
            ppl = 'both' if n_samples == sample_quantities[0] else 'after'
            perplexities = train_llama(data=samples, out_path="/cs/labs/oabend/eitan.wagner/calibration/models/bayesian" + suffix, type=type, perplexity=ppl)
        elif "mlp" in sys.argv:
            from bayesian_mlp import train_with_mlp
            _s_lr = s_lr
            _s_lr2 = s_lr2
            _s_epochs = s_epochs
            if s_lr == "":
                _s_lr = "5e-5"
            if s_lr2 == "":
                _s_lr2 = "1e-3"
            if s_epochs == "_":
                _s_epochs = "10"
            stransformer = "stransformer" in sys.argv
            pretrained = "pretrained" in sys.argv
            train_stransformer = "train_stransformer" in sys.argv
            fixed_order = "fixed_order" in sys.argv
            sparsemax = "sparsemax" in sys.argv
            sizes = (50, 30, 30, 10, 3)
            batch_size = 1
            accu_grad_steps = 1
            if stransformer:
                sizes = (30, 30, 10, 3)
            if "long_shape" in sys.argv:
                sizes = tuple(([100] if not stransformer else []) + [100, 50, 30, 10, 3])
            if "short_shape" in sys.argv:
                sizes = tuple(([50] if not stransformer else []) + [30, 3])
            if "-b" in sys.argv:
                batch_size = int(sys.argv[sys.argv.index("-b") + 1])
            if "-ga" in sys.argv:
                accu_grad_steps = int(sys.argv[sys.argv.index("-ga") + 1])
            train_with_mlp(data=samples, out_path="/cs/labs/oabend/eitan.wagner/calibration/models/bayesian" + suffix,
                           size=size, epochs=int(_s_epochs), lr1=float(_s_lr), lr2=float(_s_lr2),
                           sentence_transformer=stransformer, sizes=sizes, pretrained=pretrained,
                           train_stransformer=train_stransformer, fixed_order=fixed_order, batch_size=batch_size,
                           accu_grad_steps=accu_grad_steps, sparsemax=sparsemax, only_partial="only_partial" in sys.argv)
        elif "logistic" in sys.argv:
            from bayesian_mlp import train_logistic
            l_type = "lever" if "lever" in sys.argv else ("cloudy" if "cloudy" in sys.argv else "lever_world")
            t_model = train_logistic(data=samples_, out_path="/cs/labs/oabend/eitan.wagner/calibration/models/bayesian" + suffix,
                                     only_partial="only_partial" in sys.argv, type=l_type, degree=degree)
            tokenizer = None
        else:
            train_mlm(data=samples, out_path="/cs/labs/oabend/eitan.wagner/calibration/models/bayesian" + suffix,
                      bn_model=model, bn_marginal=bn_marginal, num_samples=n_samples)

        if "logistic" not in sys.argv:
            t_model, tokenizer = load_model(path="/cs/labs/oabend/eitan.wagner/calibration/models/bayesian" + suffix,
                                            type="llama" if "llama" in sys.argv else "opt" if "opt" in sys.argv else "bbird" if "bbird" in sys.argv else "mlm")

        print("Evaluating..")
        sys.stdout.flush()
        d[n_samples], p_list, r_list, h_list = evaluate(model=model, mle_model=mle_model, bn_full=None, bn_marginal=bn_marginal,
                                                        t_model=t_model, tokenizer=tokenizer, hidden="hidden" in sys.argv)

        print(f"Num training samples: {n_samples}")
        r = evaluate_independencies(d[n_samples], model=model)
        if "lever_world" in sys.argv:
            print("Evaluation on pair-ratio")
            r.update(pair_evaluate2(r_list))
        print("Evaluation on independent pairs")
        r.update(pair_evaluate(p_list))
        print("Evaluation on random pairs")
        r.update(pair_evaluate(r_list, random_list=True))
        if "cloudy" in sys.argv:
            r.update(mi_evaluate(d[n_samples]))
        if "lever" in sys.argv and with_b:
            print("Evaluation on heuristic cases")
            r.update(heuristic_evaluate(h_list))

        # if "lever_world" in sys.argv:
        #     print("Reconstruction errors - MLM, MLE, real")
        #     r_mlm = model.reconstruction_error(p_model="MLM")
        #     r_mle = model.reconstruction_error(p_model="MLE")
        #     r_real = model.reconstruction_error(p_model="real")
        #     print([r_mlm, r_mle, r_real])
        #     r["Reconstruction errors - MLM, MLE, real"] = [r_mlm, r_mlm, r_real]
        _results[n_samples] = r

        if len(perplexities) > 0:
            _results[n_samples]['Perplexity after training'] = perplexities[-1]
        if len(perplexities) == 2:
            _results['Perplexity before training'] = perplexities[0]

    # for k, v in d.items():
    #     print(f"Num training samples: {k}")
    #     evaluate_independencies(v)
    print("\n\n\n######################### All results:")
    # print(d)
    print(results)
    print("#########################\n\n")
    import joblib
    joblib.dump(results, f'/cs/snapless/oabend/eitan.wagner/calibration/BN scores - partial/{" ".join(sys.argv[1:] + dc)}.pkl')

    if "print_texts" in sys.argv:
        print(texts)


if __name__ == "__main__":
    print(sys.argv)
    dc = ["-dc", str(torch.cuda.device_count())]
    print(dc)
    sys.stdout.flush()
    main()

# from scipy.spatial import distance
# # r = {}
# for k, v in r.items():
#     print(k)
#     js_mlm = [distance.jensenshannon([_r[0], 1-_r[0]], [_r[1], 1-_r[1]]) for _r in v.values()]
#     js_mle = [distance.jensenshannon([_r[0], 1-_r[0]], [_r[3], 1-_r[3]]) for _r in v.values()]
#     js_marginal = [distance.jensenshannon([_r[0], 1-_r[0]], [_r[4], 1-_r[4]]) for _r in v.values()]
#     print(np.mean(js_mlm))
#     print(np.mean(js_mle))
#     print(np.me