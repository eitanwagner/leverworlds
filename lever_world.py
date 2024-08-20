import numpy as np
import pandas as pd

class LeverBalanceWorld:
    def __init__(self, max_value=5, normal=True):
        self.max_value = max_value
        self.normal = normal
        self.scale = None
        self.dist = None  # the the mean of the Gaussian or the values of the multimonial
        self.bayesian_loc = None  # predicted mean of the Gaussian
        self.loc_from_preds = None  # the the mean of the Gaussian or the values of the multimonial
        self.name = "LeverBalanceWorld"
        self.data = None
        self.eval_data = None
        self.eval_data2 = None
        self.num_objects = None
        self.states = []
        self.latents = None
        self.used_states = None
        self.visible_states = None

    def generate_world(self, latents=None, exclude=None, num_objects=2, scale=1.5, seed=1):
        np.random.seed(seed)
        self.scale = scale
        self.num_objects = num_objects
        self.states = []
        if self.normal:
            self.dist = np.random.uniform(1., 5.)
        else:
            alpha = np.ones(5)
            self.dist = np.random.dirichlet(alpha)
        for i in range(num_objects):
            self.states = self.states + [f"density{i+1}", f"volume{i+1}", f"mass{i+1}", f"distance{i+1}", f"side{i+1}", f"torque{i+1}"]
        self.latents = latents
        if latents is None:
            # does this make it too hard?
            # num_latents = np.random.randint(5)
            # self.latents = np.random.choice(self.states, size=num_latents, replace=False)
            self.latents = [f"distance{num_objects}"]
            for i in range(num_objects):  # TODO
                self.latents = self.latents + [f"torque{i + 1}"]
        if exclude is None:
            exclude = []
            use_density = np.random.randint(2, size=num_objects)
            # use_side = np.random.randint(2, size=num_objects)
            use_side = np.ones(num_objects)
            for i in range(num_objects):
                if not use_density[i]:
                    exclude = exclude + [f"density{i + 1}", f"volume{i + 1}"]
                if not use_side[i]:
                    exclude = exclude + [f"side{i + 1}"]
        self.states = self.states + ["balance"]
        self.used_states = [s for s in self.states if s not in exclude]
        # self.used_states = set(self.states) - set(exclude)
        self.visible_states = [s for s in self.used_states if s not in self.latents]
        # self.visible_states = list(set(self.used_states) - set(self.latents))
        self.text_visible_states = [s if s[:-1] not in ["side", "balanc"] else "_"+s for s in self.visible_states]
        # self.sample_generator = self._sample_generator
        print("Latent distribution mean:")
        print(self.dist)
        print("Used states:")
        print(self.used_states)
        print("Visible states:")
        print(self.visible_states)

    def _make_pairs(self):
        # generate Heuristic samples. For each sample in the test set, create one that is different in only one dimension
        states = set(self.visible_states) \
                 - set(["balance"] + [f"density{j}" for j in range(1, self.num_objects+1)]
                       + [f"torque{j}" for j in range(1, self.num_objects+1)]
                       + [f"side{j}" for j in range(1, self.num_objects+1)])
        var = np.random.choice(list(states), size=len(self.eval_data))
        self.pairs = []
        new_samples = []
        for i, row in self.eval_data.iterrows():
            n_row = row.copy()
            obj = int(var[i][-1])
            dir = row[f"side{obj}"]
            if row[var[i]] < 5 or (var[i][:-1] == "volume" and row[var[i]] < 10):
                if var[i][:-1] == "density" and row[var[i]] < 1:
                    n_row[var[i]] = n_row[var[i]] + 0.1
                n_row[var[i]] = n_row[var[i]] + 1
                by_order = dir == 1
            else:
                n_row[var[i]] = n_row[var[i]] - 1
                by_order = dir == -1
            if var[i][:-1] == "volume" or var[i][:-1] == "density":
                n_row[f"mass{obj}"] = n_row[f"volume{obj}"] * n_row[f"density{obj}"]
            n_row[f"torque{obj}"] = n_row[f"mass{obj}"] * n_row[f"distance{obj}"] * n_row[f"side{obj}"]
            n_row["balance"] = np.sign(n_row[[f"torque{i + 1}" for i in range(self.num_objects)]].sum())
            n_row["_balance"] = "L" if n_row["balance"] == 1 else "R"
            if by_order:
                self.pairs.append([row[self.visible_states[:-1]].to_numpy(), n_row[self.visible_states[:-1]].to_numpy()])
                # self.pairs.append([row.to_numpy(), n_row.to_numpy()])
            else:
                self.pairs.append([n_row[self.visible_states[:-1]].to_numpy(), row[self.visible_states[:-1]].to_numpy()])
                # self.pairs.append([n_row.to_numpy(), row.to_numpy()])
            new_samples.append(n_row)
        self.eval_data2 = pd.concat(new_samples, axis=1).T

    def generate_samples(self, num_samples, seed=1, eval=False):
        """for extras
        :param num_samples:
        :param random_state:
        :return:
        """
        if eval:
            seed = seed + 42
        self.data = pd.DataFrame()
        np.random.seed(seed)
        for i in range(self.num_objects):
            if i == self.num_objects - 1 and self.normal:
                self.data[f"distance{i+1}"] = np.random.normal(loc=self.dist, scale=self.scale, size=num_samples)
            elif i == self.num_objects - 1 and not self.normal:
                self.data[f"distance{i+1}"] = np.random.choice(np.arange(1, 6), size=num_samples, p=self.dist)
            else:
                self.data[f"distance{i+1}"] = np.random.randint(1, 6, size=num_samples)
            if f"density{i+1}" in self.used_states:
                # density1 = np.random.uniform(0.5, 1.5, size=num_samples)
                density = np.random.randint(1, 11, size=num_samples) / 10
                volume = np.random.randint(1, 11, size=num_samples)
                self.data[f"density{i+1}"] = density
                self.data[f"volume{i+1}"] = volume
                self.data[f"mass{i+1}"] = density * volume
            else:
                self.data[f"mass{i+1}"] = np.random.randint(1, 6, size=num_samples)
            if f"side{i+1}" in self.used_states:
                self.data[f"side{i+1}"] = 2 * np.random.randint(2, size=num_samples) - 1
            else:
                self.data[f"side{i+1}"] = np.ones(num_samples) * (2 * (i % 2) - 1)
            self.data[f"_side{i+1}"] = self.data[f"side{i+1}"].apply(lambda x: "L" if x == 1 else "R")
            self.data[f"torque{i+1}"] = self.data[f"mass{i+1}"] * self.data[f"distance{i+1}"] * self.data[f"side{i+1}"]

        self.data["balance"] = np.sign(self.data[[f"torque{i+1}" for i in range(self.num_objects)]].sum(axis=1))
        zeros = self.data["balance"] == 0
        # self.data["balance"][zeros] = np.random.randint(2, size=sum(zeros)) * 2 - 1
        self.data.loc[zeros, "balance"] = np.random.randint(2, size=sum(zeros)) * 2 - 1
        self.data["_balance"] = self.data["balance"].apply(lambda x: "L" if x == 1 else "R")
        if eval:
            self.eval_data = self.data.round(2).copy()
            self._make_pairs()
        return self.data.round(2)

    def _sample_torques(self, i=0, loc=None):  # ????
        """
        Find values and probabilities for (almost) all possible torques
        :return:
        """
        from scipy.stats import norm
        torques = {}
        # if i == self.num_objects - 1 and self.normal:
        #     pass
        # else:
        if f"density{i + 1}" in self.used_states:
            masses = {}
            # density1 = np.random.uniform(0.5, 1.5, size=num_samples)
            for den in range(1, 11):
                for v in range(1, 11):
                    if den*v*0.1 in masses:
                        masses[den*v*0.1] += 1 / (9*10)
                    else:
                        masses[den*v*0.1] = 1 / (9*10)
        else:
            masses = {m: 0.2 for m in range(1, 6)}
        if i == self.num_objects - 1 and self.normal:
            ds = np.linspace(-5, 15, 2000)
            dx = 20/2000
            ps = norm.pdf(self.dist if loc is None else loc, loc=ds, scale=self.scale) * dx
        else:
            ds = np.arange(1, 6, dtype=float)
            ps = np.full_like(ds, 1/len(ds))
        for m, mp in masses.items():
            for d, p in zip(ds, ps):
                for s in [-1, 1]:
                    if m * d * s in torques:
                        torques[m * d * s] += mp * p * 0.5
                    else:
                        torques[m * d * s] = mp * p * 0.5

        return torques

    def torques_var(self):
        ts = []
        for i in range(self.num_objects):
            ts.append(self._sample_torques(i, loc=None))
        vals = np.array(list(ts[0].keys()), dtype=float)
        probs = np.array(list(ts[0].values()), dtype=float)
        for i in range(1, self.num_objects):
            vals = (vals[:, None] + np.array(list(ts[i].keys()), dtype=float)[None, :]).ravel()
            probs = (probs[:, None] * np.array(list(ts[i].values()), dtype=float)[None, :]).ravel()

        df = pd.DataFrame({"vals": vals,
                          "probs": probs})
        df = df.groupby(df['vals'], as_index=False).sum()

        df2 = df.copy()
        df2["balance"] = 1. * (df2["vals"] > 0)
        mu2 = np.mean(df2['balance'] * df['probs'])
        var2 = np.mean(df2['probs'] * (df2['balance'] - mu2) ** 2)  # variance of the torque sum
        return var2
        # df2 = df2.groupby(df2['balance'], as_index=False).sum()

        # mu = np.mean(df['vals'] * df['probs'])
        # var = np.mean(df['probs'] * (df['vals'] - mu) ** 2)  # variance of the torque sum
        # return var  # is this correct??
        # return np.var(df['vals'] * np.sqrt(df['probs']))  # is this correct??

    def get_samples(self, data, with_output=True):
        return data[self.visible_states if with_output else self.visible_states[:-1]].to_numpy()

    def get_text_samples(self, data, format=":", with_output=True):
        states = self.text_visible_states if with_output else self.text_visible_states[:-1]
        names = [n if n[0] != "_" else n[1:] for n in states]
        if format in [":", "="]:
            text_samples = [", ".join([f"{n}{format} {_s}" for _s, n in zip(d[states], names)]) for _, d in data.iterrows()]
        else:
            dict = {"1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth", "6": "sixth"}
            def convert_name(n):
                if n[:-1] in ["side", "balanc"]:
                    return ""
                if n[0] == "_" or n[-1] == "e":
                    return n[1:]
                return n[:-1]
            text_samples = ["; ".join([f"the {dict[n[-1]]} object's {convert_name(n)} is {v}"
                                       if convert_name(n) != "balance" else f"the balance is {v}"
                                       for n, v in d[1].items() if convert_name(n) != ""])
                            for d in data.iterrows()]
        return text_samples

    def _find_cutoff(self, inputs):
        """
        find cutoff value for the latent variable. the inputs are the visible states
        :param inputs:
        :return:
        """
        # TODO: right now we assume only the last distance is unknown (i.e. cannot be inferred from the rest)
        torques = []
        for i in range(1, self.num_objects):
            if f"torque{i}" in self.visible_states:
                t_i = self.visible_states.index(f"torque{i}")
                torques.append(inputs[t_i])
            else:
                d_i = self.visible_states.index(f"distance{i}")
                s_i = self.visible_states.index(f"side{i}")
                if f"mass{i}" not in self.visible_states:
                    de_i = self.visible_states.index(f"density{i}")
                    v_i = self.visible_states.index(f"volume{i}")
                    m = inputs[de_i] * inputs[v_i]
                else:
                    m_i = self.visible_states.index(f"mass{i}")
                    m = inputs[m_i]
                torques.append(m * inputs[s_i] * inputs[d_i])
        m_last = inputs[self.visible_states.index(f"mass{self.num_objects}")]
        s_last = inputs[self.visible_states.index(f"side{self.num_objects}")]
        # d = self.visible_states.index(f"distance{i}")
        return - sum(torques) / (m_last * s_last), s_last

    def predict(self, inputs, real=True, from_preds=False):
        # make sure the order is correct
        cutoff, direction = self._find_cutoff(inputs)
        if self.normal:
            from scipy.stats import norm
            loc = self.dist if real else self.bayesian_loc if not from_preds else self.loc_from_preds
            p = norm.cdf(cutoff, loc=loc, scale=self.scale)
            return [p, 1 - p] if direction == -1 else [1 - p, p]
        else:
            p = sum(self.dist[:int(cutoff)])
            if cutoff % 1 == 0 and 1 <= cutoff <= 5:
                p -= self.dist[int(cutoff)-1] * 0.5
            return [p, 1 - p] if direction == -1 else [1 - p, p]

    def fit(self, samples, return_probs=False, quants=5000):
        """
        Estimate the latent mean from samples
        :return:
        """
        from scipy.stats import norm
        from scipy.special import logsumexp
        side_index0 = self.visible_states.index(f"side1")
        # xs = np.linspace(1, 5, quants)
        xs = np.linspace(-15, 25, quants)
        ps = np.zeros((len(samples), quants))
        cutoffs = []
        for i, s in enumerate(samples):
            if s[side_index0] == s[-2]:
                continue
            cutoff, direction = self._find_cutoff(s[:-1])
            # if cutoff > 5:
            #     cutoff = 5.
            # if cutoff < 1:
            #     cutoff = 1.
            cutoffs.append(cutoff)
            # ps[i] = norm.cdf(xs, loc=cutoff, scale=self.scale)
            if direction * s[-1] == 1:
                ps[i] = norm.sf(cutoff, loc=xs, scale=self.scale)
                # ps[i] = norm.logsf(cutoff, loc=xs, scale=self.scale)
            else:
                # ps[i] = 1 - norm.cdf(xs, loc=cutoff, scale=self.scale)  # because I switched xs and cutoff
                # ps[i] = norm.sf(xs, loc=cutoff, scale=self.scale)  # because I switched xs and cutoff
                ps[i] = norm.cdf(cutoff, loc=xs, scale=self.scale)
                # ps[i] = norm.logcdf(cutoff, loc=xs, scale=self.scale)
            ps[i] = ps[i] / sum(ps[i])
        # ps = logsumexp(ps, axis=0)
        ps = np.sum(ps, axis=0)
        if return_probs:
            return xs, ps
        self.bayesian_loc = xs[np.argmax(ps)]
        return self

    def fit_from_probs(self, samples, probs=None):
        """

        :param samples:
        :param weights:
        :return:
        """
        from scipy.stats import norm
        xs = np.linspace(1, 5, 1000)
        ps = np.zeros(1000)
        ls = []
        for s, p in zip(samples, probs):
            cutoff, direction = self._find_cutoff(s[:-1])
            side_index0 = self.visible_states.index(f"side1")
            if s[side_index0] == s[-2]:
                continue
            if direction == 1:
                ps = norm.sf(cutoff, loc=xs, scale=self.scale)
            else:
                ps = norm.cdf(cutoff, loc=xs, scale=self.scale)
            ls.append(xs[np.argmin(np.abs(ps - p))])

        l = np.mean(ls)
        self.loc_from_preds = l
        return self

    def reconstruction_error(self, p_model="MLM"):
        samples = self.get_samples(self.data)
        probs = self.eval_data[p_model]
        real = self.eval_data["real"]
        self.fit_from_probs(samples, probs=probs)
        preds = [self.predict(inputs=s, real=False, from_preds=True)[0] for s in samples]
        return np.mean(np.abs(np.array(preds) - np.array(real)))

def find_n(var, eps=0.05):
    return np.ceil(np.sqrt(var * 0.25) / eps)

if __name__ == "__main__":
    model = LeverBalanceWorld(normal=True)
    model.generate_world(scale=1., seed=1)
    # var = model.torques_var()
    # print(find_n(var, eps=0.05))

    data = model.generate_samples(num_samples=2000)
    samples = model.get_samples(data)
    text_samples = model.get_text_samples(data)
    # model.predict(samples[1])
    for q in [50, 100, 500, 1000, 5000]:
        print("prediction  for dist")
        print(q)
        fit = model.fit(samples, quants=q)
        print(fit.bayesian_loc)
        print(model.dist)

    real_preds = np.array([model.predict(s[:-1]) for s in samples])
    # model.dist = fit
    fake_preds = np.array([model.predict(s[:-1], real=False) for s in samples])


    print("pred vs fake")
    print(sum(np.argmax(real_preds, axis=1) == np.argmax(fake_preds, axis=1)) / len(real_preds))
    print(sum(abs(real_preds[:, 0] - fake_preds[:, 0])) / len(real_preds))

    print("real vs fake")
    print(sum(((-np.array([s[-1] for s in samples]) + 1) / 2) == np.argmax(fake_preds, axis=1)) / len(real_preds))

    print("real vs pred")
    print(sum(((-np.array([s[-1] for s in samples]) + 1) / 2) == np.argmax(real_preds, axis=1)) / len(real_preds))


    # print(samples)