import random
from decimal import Decimal

import pandas as pd

import utils


class MaxPollsExceededError(Exception):
    pass


class Audit:
    """
    Audit class that defines the common objects and methods for running a BRAVO audit.
    """
    def __init__(self, risk_limit, n_winners, max_polls, random_seed, preliminary_file):
        self.required_headers = {'table', 'candidate', 'votes'}
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

        self.N = self.preliminary['votes'].sum()

    def sanity_check(self):
        assert 0 < self.risk_limit < 1
        assert self.n_winners > 0
        assert self.max_polls > 0
        assert set(self.preliminary.columns) >= self.required_headers

    def _check_audited_ballots(self):
        if self.m >= self.max_polls:
            raise MaxPollsExceededError()

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

    def _recount_params(self):
        m = self.sample_size()
        sample = self.scrambled[:m]
        self.scrambled = self.scrambled[m:]
        tables = {}
        for t, n in sample:
            if t not in tables:
                tables[t] = []

            tables[t].append(n)

        for table in tables:
            tables[table].sort()

        return tables, m

    def _init_ballot_polling(self, tables, table_count, pseudo_candidates):
        self._preliminary()
        for t in tables:
            self.scrambled.extend(zip([t] * table_count[t], range(table_count[t])))

        random.shuffle(self.scrambled)
        accum_recount = {c: 0 for c in pseudo_candidates}
        return accum_recount

    def ballot_polling(self):
        raise NotImplementedError()

    def recount(self):
        raise NotImplementedError()

    def sample_size(self):
        raise NotImplementedError()


class Plurality(Audit):
    """
    A plurality election chooses the n candidates with most votes.
    If n == 1, this becomes a simple majority election.
    """
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

    def sample_size(self):
        m = utils.plurality_sample_size(self.vote_count, self.W, self.L, self.risk_limit)
        m = min(m // 2, len(self.scrambled))
        return m

    def ballot_polling(self):
        accum_recount = self._init_ballot_polling(self.tables, self.table_count, self.candidates)
        while True:
            self._check_audited_ballots()
            recount, m = self.recount()
            self.m += m
            for c in recount:
                accum_recount[c] += recount[c]

            self.T, self.max_p_value = utils.SPRT(self.vote_count, recount, self.T, self.risk_limit)
            self._stats(accum_recount)
            if utils.validated(self.T, self.risk_limit):
                break

            print('Certainty not sufficient, another round is required.')
            input('Continue... ')

    def recount(self):
        tables, m = self._recount_params()
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
    """
    A super majority election chooses the candidate with most votes, if they
    amount to more than half the total.
    """
    def __init__(self, risk_limit, max_polls, random_seed, preliminary_file):
        super().__init__(risk_limit, 1, max_polls, random_seed, preliminary_file)
        self.vote_count_s = {'w': self.vote_count[self.W[0]], 'l': sum(self.vote_count[loser] for loser in self.L)}
        self.Ts = {'w': {'l': Decimal(1.0)}}
        self.Ss = {'w': {'l': self.vote_count[self.W[0]] / sum(self.vote_count.values())}}

    def ballot_polling(self):
        accum_recount = self._init_ballot_polling(self.tables, self.table_count, self.candidates)
        while True:
            self._check_audited_ballots()
            recount, m = self.recount()
            self.m += m
            for c in recount:
                accum_recount[c] += recount[c]

            self.T, self.max_p_value = utils.SPRT(self.vote_count, recount, self.T, self.risk_limit)
            recount_s = {'w': recount[self.W[0]], 'l': sum(recount[loser] for loser in self.L)}
            self.Ts, max_p_value_s = utils.SPRT(self.vote_count_s, recount_s, self.Ts, self.risk_limit)
            self.max_p_value = max(self.max_p_value, max_p_value_s)
            self._stats(accum_recount)
            if utils.validated(self.T, self.risk_limit) and utils.validated(self.Ts, self.risk_limit):
                break

            print('Certainty not sufficient, another round is required.')
            input('Continue... ')


class DHondt(Audit):
    """
    A proportional method, in which the current party votes is divided by the
    number of seats assigned to them + 1.
    """
    def __init__(self, risk_limit, n_winners, max_polls, random_seed, preliminary_file):
        super().__init__(risk_limit, n_winners, max_polls, random_seed, preliminary_file)
        self.required_headers.add('party')
        self.vote_count = self.preliminary.groupby('party').sum()['votes'].to_dict()
        self.table_count = self.preliminary.groupby('table').sum()['votes'].to_dict()
        self.candidate_count = self.preliminary.groupby('candidate').sum()['votes'].to_dict()
        self.tables = list(self.table_count.keys())
        self.parties = list(self.vote_count.keys())
        self.party_members = {}
        for party in self.parties:
            p = self.preliminary[self.preliminary['party'] == party]
            for c in p['candidate'].unique():
                self.party_members[c] = party

        self.pseudo_vote_count = {(p, i): utils.p(p, self.vote_count, i) for i in range(n_winners) for p in self.parties}
        self.W, self.L = utils.get_W_L_sets(self.pseudo_vote_count, n_winners)
        self.Sw = {}
        self.Sl = {}
        for party in self.parties:
            wp = list(filter(lambda x: x[0] == party, self.W))
            lp = list(filter(lambda x: x[0] == party, self.L))
            if wp:
                self.Sw[party] = max(wp, key=lambda x: x[1])[1]

            if lp:
                self.Sl[party] = min(lp, key=lambda x: x[1])[1]

        self.Wp = []
        for winner in self.W:
            if winner[0] not in self.Wp:
                self.Wp.append(winner[0])

        self.Lp = []
        for loser in self.L:
            if loser[0] not in self.Lp:
                self.Lp.append(loser[0])

        self.T = {}
        for winner in self.Wp:
            self.T[winner] = {loser: Decimal(1) for loser in self.Lp if winner != loser}

        self.Tp = {}
        for p in self.Wp:
            self.Tp[p] = {}
            seats = 0
            for winner in self.W:
                if winner[0] == p:
                    seats = max(seats, winner[1])

            seats += 1
            party = self.preliminary[self.preliminary['party'] == p].groupby('candidate').sum()['votes']
            party = party.sort_values(ascending=False)
            party_members = list(party.keys())
            party_winners = party_members[:seats]
            party_losers = party_members[seats:]
            for winner in party_winners:
                self.Tp[p][winner] = {loser: Decimal(1) for loser in party_losers}

    def sample_size(self):
        m = utils.dhondt_sample_size(self.N, self.risk_limit, self.vote_count, self.Sw, self.Sl)
        m = min(m * 4, len(self.scrambled))
        return m

    def _preliminary(self):
        print('===========Preliminary results===========')
        L = [loser for loser in self.Lp if loser not in self.Wp]
        utils.print_stats(self.vote_count, self.Wp, L)
        print('=========================================')

    def _stats(self, accum_recount):
        print('=========================================')
        print(f'Max p-value: {self.max_p_value:.5f}')
        print(f'Polled ballots: {self.m}')
        W = [winner for winner in self.party_members if self.party_members[winner] in self.Wp]
        L = [loser for loser in self.party_members if self.party_members[loser] not in self.Wp]
        utils.print_stats(accum_recount, W, L)
        print('=========================================')

    def ballot_polling(self):
        accum_recount = self._init_ballot_polling(self.tables, self.table_count, self.party_members)
        while True:
            self._check_audited_ballots()
            recount, m = self.recount()
            self.m += m
            party_recount = {p: 0 for p in self.parties}
            for c in recount:
                t = self.party_members[c]
                accum_recount[c] += recount[c]
                party_recount[t] += recount[c]

            self.T, self.max_p_value = utils.SPRT(self.vote_count, party_recount, self.T, self.risk_limit)
            for tp in self.Tp:
                self.Tp[tp], max_p = utils.SPRT(self.candidate_count, recount, self.Tp[tp], self.risk_limit)
                self.max_p_value = max(self.max_p_value, max_p)

            self._stats(accum_recount)
            if utils.validated(self.T, self.risk_limit) and all([utils.validated(self.Tp[tp], self.risk_limit) for tp in self.Tp]):
                break

            print('Certainty not sufficient, another round is required.')
            input('Continue... ')

    def recount(self):
        tables, m = self._recount_params()
        while True:
            recount = utils.enter_recount(tables, list(self.party_members.keys()))
            recounted_ballots = 0
            for c in recount:
                recounted_ballots += recount[c]

            if recounted_ballots == m:
                break

            print(f'Incorrect recounting: expected {m}, {recounted_ballots} given')

        return recount, m
