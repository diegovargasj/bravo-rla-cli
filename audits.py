import random
from decimal import Decimal

import pandas as pd

import utils


class MaxPollsExceededError(Exception):
    pass


class Audit:
    def __init__(self, risk_limit, n_winners, max_polls, random_seed, preliminary_file):
        self.vote_count = {}
        self.risk_limit = risk_limit
        self.n_winners = n_winners
        self.W = []
        self.L = []
        self.max_polls = max_polls
        self.random_seed = random_seed
        self.preliminary_file = preliminary_file
        self.m = 0
        self.max_p_value = 1
        self.scrambled = []

        self.preliminary = pd.read_csv(preliminary_file)
        random.seed(random_seed)

        self._sanity_check()

    def _sanity_check(self):
        assert 0 < self.risk_limit < 1
        assert self.n_winners > 0
        assert self.max_polls > 0
        assert set(self.preliminary.columns) >= {'table', 'candidate', 'votes'}

    def _preliminary(self):
        print('===========Preliminary results===========')
        utils.print_stats(self.vote_count, self.W, self.L)
        print('=========================================')

    def _stats(self, accum_recount):
        print('=========================================')
        print(f'Max p-value: {self.max_p_value:.5f}')
        print(f'Polled ballots: {self.m}')
        utils.print_stats(accum_recount, self.W, self.L)
        print('=========================================')

    def sample_size(self):
        tot = 0
        for c in self.vote_count:
            tot += self.vote_count[c]

        w_star = self.W[0]
        for w in self.W:
            if self.vote_count[w] < self.vote_count[w_star]:
                w_star = w

        l_star = self.L[0]
        for l in self.L:
            if self.vote_count[l_star] < self.vote_count[l]:
                l_star = l

        return utils.ASN(self.risk_limit, self.vote_count[w_star], self.vote_count[l_star], tot) // 2

    def ballot_polling(self):
        raise NotImplementedError()

    def ballot_polling_recount(self):
        raise NotImplementedError()


class Plurality(Audit):
    def __init__(self, risk_limit, n_winners, max_polls, random_seed, preliminary_file):
        super().__init__(risk_limit, n_winners, max_polls, random_seed, preliminary_file)
        self.vote_count = self.preliminary.groupby('candidate').sum()['votes'].to_dict()
        self.candidates = list(self.vote_count.keys())
        self.table_count = self.preliminary.groupby('table').sum()['votes'].to_dict()
        self.tables = list(self.table_count.keys())
        self.W, self.L = utils.get_W_L_sets(self.vote_count, self.n_winners)
        self.T = {}
        self.S = {}
        for winner in self.W:
            self.T[winner] = {}
            self.S[winner] = {}
            for loser in self.L:
                self.T[winner][loser] = Decimal(1.0)
                self.S[winner][loser] = self.vote_count[winner] / (self.vote_count[winner] + self.vote_count[loser])

    def ballot_polling(self):
        self._preliminary()
        for t in self.tables:
            self.scrambled.extend(zip([t] * self.table_count[t], range(self.table_count[t])))

        random.shuffle(self.scrambled)
        accum_recount = {c: 0 for c in self.candidates}
        while True:
            if self.m > self.max_polls:
                raise MaxPollsExceededError(f'polled {self.m} ballots (max {self.max_polls})')

            recount, m = self.ballot_polling_recount()
            self.m += m
            for c in recount:
                accum_recount[c] += recount[c]

            self.T, self.max_p_value = utils.SPRT(self.vote_count, recount, self.T, self.risk_limit)
            self._stats(accum_recount)
            if utils.validated(self.T, self.risk_limit):
                break

            print('Certainty not sufficient, another round is required.')
            input('Continue... ')

    def ballot_polling_recount(self):
        m = self.sample_size()
        m = min(m, len(self.scrambled))
        sample = self.scrambled[:m]
        self.scrambled = self.scrambled[m:]
        tables = {}
        for t, n in sample:
            if t not in tables:
                tables[t] = []

            tables[t].append(n)

        for table in tables:
            tables[table].sort()

        while True:
            recount = utils.enter_recount(tables, self.candidates)
            recounted_ballots = 0
            for c in recount:
                recounted_ballots += recount[c]

            if recounted_ballots == m:
                break

            print(f'Incorrect recounting: expected {m}, {recounted_ballots} given')

        return recount, m


class SuperMajority(Plurality):
    def __init__(self, risk_limit, max_polls, random_seed, preliminary_file):
        super().__init__(risk_limit, 1, max_polls, random_seed, preliminary_file)
        self.vote_count_s = {'w': self.vote_count[self.W[0]], 'l': sum(self.vote_count[l] for l in self.L)}
        self.Ts = {'w': {'l': Decimal(1.0)}}
        self.Ss = {'w': {'l': self.vote_count[self.W[0]] / sum(self.vote_count.values())}}

    def ballot_polling(self):
        self._preliminary()
        for t in self.tables:
            self.scrambled.extend(zip([t] * self.table_count[t], range(self.table_count[t])))

        random.shuffle(self.scrambled)
        accum_recount = {c: 0 for c in self.candidates}
        while True:
            if self.m > self.max_polls:
                raise MaxPollsExceededError(f'polled {self.m} ballots (max {self.max_polls})')

            recount, m = self.ballot_polling_recount()
            self.m += m
            for c in recount:
                accum_recount[c] += recount[c]

            self.T, self.max_p_value = utils.SPRT(self.vote_count, recount, self.T, self.risk_limit)
            recount_s = {'w': recount[self.W[0]], 'l': sum(recount[l] for l in self.L)}
            self.Ts, max_p_value_s = utils.SPRT(self.vote_count_s, recount_s, self.Ts, self.risk_limit)
            self.max_p_value = max(self.max_p_value, max_p_value_s)
            self._stats(accum_recount)
            if utils.validated(self.T, self.risk_limit) and utils.validated(self.Ts, self.risk_limit):
                break

            print('Certainty not sufficient, another round is required.')
            input('Continue... ')


class DHondt(Audit):
    def __init__(self, risk_limit, n_winners, max_polls, random_seed, preliminary_file):
        super().__init__(risk_limit, n_winners, max_polls, random_seed, preliminary_file)
        self.vote_count = self.preliminary.groupby('party').sum()['votes'].to_dict()
        self.parties = list(self.vote_count.keys())

    def ballot_polling(self):
        pass

    def _sanity_check(self):
        super()._sanity_check()
        assert 'party' in self.preliminary.columns
