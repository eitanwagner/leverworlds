import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import random
import tqdm
import numpy as np

dev = "cuda" if torch.cuda.is_available() else torch.device("cpu")
class TrfWithMLP(nn.Module):
    def __init__(self, model_name, sizes=(10, 50, 50, 3), sentence_transformer=False, pretrained=False,
                 train_stransformer=True, sparsemax=False, only_partial=False):
        super().__init__()
        n_labels = sizes[0]
        # p_type = "regression"
        # p_type = "classification"
        self.stransformers = sentence_transformer
        self.train_stransformer = train_stransformer
        self.only_partial = only_partial
        if sentence_transformer:
            print("Using sentence transformer")
            import logging
            logging.basicConfig(level=logging.WARNING)
            from sentence_transformers import SentenceTransformer, util
            if pretrained:
                print("Using all-MiniLM-L6-v2")
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            else:
                self.model = SentenceTransformer('/cs/snapless/oabend/eitan.wagner/locations/output/training_nli_v2_' + model_name.replace("/", "-"))
            n_labels = self.model.get_sentence_embedding_dimension()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                            cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                                            num_labels=n_labels)

        self.mlp = nn.Sequential(nn.Linear(n_labels * (3 if only_partial else 4), sizes[1]), nn.ReLU())
        if len(sizes) > 2:
            for i, s in enumerate(sizes[1:-2]):
                self.mlp.append(nn.Linear(s, sizes[i+2]))
                self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Linear(sizes[-2], sizes[-1]))

        # self.mlp = nn.Sequential(
        #     nn.Linear(sizes[0] * 4, sizes[1]),
        #     nn.ReLU(),
        #     nn.Linear(sizes[1], sizes[2]),
        #     nn.ReLU(),
        #     nn.Linear(sizes[2], sizes[3])
        # )
        if sparsemax:
            from sparsemax import Sparsemax
            self.sparsemax = Sparsemax(dim=-1)
            self.loss_fct = nn.NLLLoss()
        else:
            self.sparsemax = None
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None):
        if self.stransformers:
            # in this case the inputs should be the texts and not the tokens
            if self.train_stransformer:
                out1 = self.model.encode(inputs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False).to(dev)
            else:
                with torch.no_grad():
                    out1 = self.model.encode(inputs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False).to(dev)
        else:
            out1 = self.model(**inputs.to(dev)).logits
        out2 = self.mlp(out1.reshape(1, -1))
        if self.sparsemax is not None:
            # if not self.training:
            #     print("\n***")
            #     print(out1)
            #     print(out2)
            out2 = torch.log(self.sparsemax(out2))

        loss = None
        if labels is not None:
            # if self.sparsemax is not None:
            #     out2 = torch.log(self.sparsemax(out2))
            loss = self.loss_fct(out2, labels.to(dev))
        return {"loss": loss, "logits": out2[0]}

def convert_to_separate(text, fixed_order=False):
    labels = ["L", "R", "B"]
    components = text.split(", ")
    b = None
    cs = []
    for c in components:
        if "balance" in c:
            b = labels.index(c[-1])
        else:
            if b is not None:
                cs.append(c.split(":")[0] + ":")
            else:
                cs.append(c)
    if fixed_order:
        cs.sort()
    return cs, b

def train_with_mlp(data, out_path, lr1=5e-5, lr2=1e-3, epochs=3, type="lever", size="125m", sizes=(50, 30, 30, 10, 3),
                   sentence_transformer=False, pretrained=False, train_stransformer=True, fixed_order=False,
                   batch_size=1, accu_grad_steps=4, sparsemax=False, only_partial=False):
    batch_size = batch_size
    accu_grad_steps = accu_grad_steps
    print("Training with MLP")
    print(f"batch_size: {batch_size}, accu_grad_steps: {accu_grad_steps}, lr1: {lr1}, lr2: {lr2}, epochs: {epochs}")
    print("Sizes:")
    print(sizes)
    print(f"Use sentence transformer: {sentence_transformer}")
    print(f"Train sentence transformer: {train_stransformer}")

    model_name = f"facebook/opt-{size}"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_texts, val_texts, test_texts = data[:int(len(data) * 0.8)], \
                                         data[-int(len(data) * 0.2):-int(len(data) * 0.1)], \
                                         data[-int(len(data) * 0.1):]

    # convert to format
    _train = [convert_to_separate(tt, fixed_order=fixed_order) for tt in train_texts]

    full_model = TrfWithMLP(model_name=model_name, sizes=sizes, sentence_transformer=sentence_transformer,
                            pretrained=pretrained, train_stransformer=train_stransformer, sparsemax=sparsemax, only_partial=only_partial)
    to_train = [{"params": full_model.mlp.parameters(), "lr": lr2}]
    if train_stransformer:
        to_train = to_train + [{"params": full_model.model.parameters(), "lr": lr1}]
    optimizer = torch.optim.AdamW(to_train)
    full_model.to(dev)
    # mlp.to(dev)
    full_model.train()
    losses = []

    for e in range(epochs):
        print("\n" + str(e))
        random.shuffle(_train)
        # for i in tqdm.tqdm(range(0, len(_train), batch_size), desc="Train"):
        for i in range(0, len(_train), batch_size):
            # assuming batch_size = 1
            train_batch = _train[i: i + batch_size]
            _batch_size = len(train_batch)
            if _batch_size == 1:
                texts, labels = zip(*train_batch)
            # texts, labels = zip(*train_batch[0])
            if sentence_transformer:
                inputs = texts[0]
            else:
                inputs = tokenizer(texts[0], padding=True, return_tensors="pt")

            loss = full_model(inputs, labels=torch.LongTensor(labels))["loss"]
            losses.append(loss.item())
            loss.backward()
            if i % accu_grad_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        print(f"Epoch {e} loss:")
        print(np.mean(losses))
        losses = []

    torch.save(full_model, out_path + "/full_model.pt")

def mlp_probability(full_model, tokenizer, names, values, fixed_order=False, i=None):
    texts = [n + ": " + (str(_s) if _s != "M" else "") for _s, n in zip(values, names)]
    if full_model.only_partial:
        texts.remove("object2 mass: ")
    if fixed_order:
        texts.sort()
    if full_model.stransformers:
        inputs = texts
    else:
        inputs = tokenizer(texts, padding=True, return_tensors="pt")
    # print(inputs.device)
    with torch.no_grad():
        if full_model.sparsemax is not None:
            # print(full_model(inputs)["logits"])
            probs = full_model(inputs)["logits"].exp().cpu().numpy()
        else:
            probs = full_model(inputs)["logits"].softmax(-1).cpu().numpy()
    if 0 <= i < 30:
        print("\n")
        print(values)
        print(probs)
    # print("Texts: ")
    # print(texts)
    # print(probs)
    return probs

def train_logistic(data, out_path, only_partial=False, type="lever", degree=2):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    import joblib
    print("Degree:")
    print(degree)
    # train, val, test = data[:int(len(data) * 0.8)], \
    #                                      data[-int(len(data) * 0.2):-int(len(data) * 0.1)], \
    #                                      data[-int(len(data) * 0.1):]
    train = data
    if type == "lever":
        if only_partial:
            X_train = train[:, :-1].astype(int)
        else:
            X_train = train[:, [0, 1, 3, 4]].astype(int)
        y_train = [["L", "R", "B"].index(t) for t in train[:, 6]]
    elif type == "cloudy":
        X_train = np.array([[["T", "F"].index(_t) for _t in t] for t in train[:, :3]])
        y_train = [["T", "F"].index(t) for t in train[:, 3]]
    else:
        X_train = train[:, :-1]
        # y_train = [[1, -1].index(t) for t in train[:, -1]]
        y_train = (1 - train[:, -1]) / 2  # convert to 0, 1, with 0 for L
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)
    # multi_class = "multinomial"
    multi_class = "ovr"
    penalty = None
    print("multi_class:")
    print(multi_class)
    print("penalty:")
    print(penalty)
    lr = LogisticRegression(multi_class=multi_class, max_iter=10000, penalty=penalty, solver='saga')
    pipe = Pipeline([('polynomial_features', poly), ('logistic_regression', lr)])
    pipe.fit(X_train, y_train)
    # joblib.dump(pipe, out_path)
    return pipe

def predict_marginal(model, sample, only_partial=False, num_classes=3, type="lever"):
    if not only_partial:
        samples = [sample + [i] for i in range(1, 11)]
    else:
        samples = [sample]
    # probs = model.predict_proba(samples).mean(axis=0)
    probs = model.predict_proba(samples).sum(axis=0)
    if len(probs) == 2 and num_classes == 3:
        probs = np.pad(probs, (0, 1))
    return probs


class DirichletEstimator:
    def __init__(self, label_dict=None, dim=10):
        self.dim = dim
        self.alpha = np.zeros(self.dim)
        self.label_dict = label_dict

    def fit(self, samples, labels):
        def update(l1, l2, m1, b):
            u = np.zeros(self.dim)
            if b == 0:  # L (the side of l1)
                u[:int(l1 * m1 / l2)] = 1
            else:
                u[int(l1 * m1 / l2):] = 1
            if (l1 * m1 / l2) % 1 == 0 and l1 * m1 / l2 <= self.dim:
                u[int(l1 * m1 // l2) - 1] = 1 / 2
            return u / u.sum()

        for input, output in zip(samples.astype(int), labels):
            self.alpha = self.alpha + update(*input, self.label_dict[output])
        return self

    def predict(self, sample):
        """
        return probabilities ["L", "R]
        :param sample:
        :return:
        """
        p = np.zeros(self.dim)
        map = self.alpha / sum(self.alpha)
        for m2 in range(1, self.dim+1):
            p[m2 - 1] = 1. * (int(sample[0]) * int(sample[2]) > int(sample[1]) * m2)
            if int(sample[0]) * int(sample[2]) == int(sample[1]) * m2:
                p[m2 - 1] = 1 / 2
        return [sum(p * map), 1 - sum(p * map)]


def p_b_given_all(l1, l2, m1, m2, b):
    return float(1 - b) if l1 * m1 > l2 * m2 else float(b) if l1 * m1 < l2 * m2 else 0.5

def p_b_given_x(l1, l2, m1, b, m2_probs):
    # p0 = 1 / 1000
    p = sum([p_m2 * p_b_given_all(l1, l2, m1, m2+1, b) for m2, p_m2 in enumerate(m2_probs)])
    return p

def p_m2_given_x(l1, l2, m1, b):
    probs = np.array([p_b_given_all(l1, l2, m1, m2, b) for m2 in range(1, 11)])
    return probs / sum(probs) if sum(probs) > 0 else 0.

def E_m2s_single_sample(m2_probs, squared=False):
    p_x = 1 / 1000
    E = 0.
    for l1 in range(1, 11):
        for l2 in range(1, 11):
            for m1 in range(1, 11):
                for b in [0, 1]:
                    E = E + p_x * p_b_given_x(l1, l2, m1, b, m2_probs) * p_m2_given_x(l1, l2, m1, b) ** (squared+1)
    return E

def var_m2_all(m2_probs, N=10):
    return (E_m2s_single_sample(m2_probs, squared=True) - E_m2s_single_sample(m2_probs, squared=False) ** 2) / N