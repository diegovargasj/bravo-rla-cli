import itertools
import math
from decimal import Decimal

import numpy as np
import utils


class MaxPollsExceededError(Exception):
    pass


class Audit:
    """
    Audit class that defines the common objects and methods for running a BRAVO audit.
    """
    def __init__(self, risk_limit, audit_type, n_winners, max_polls, random_seed, preliminary):
        self.required_headers = {'table', 'candidate', 'votes'}
        self.vote_count = {}
        self.risk_limit = Decimal(risk_limit)
        self.n_winners = n_winners
        self.audit_type = audit_type
        self.T = {} if self.is_ballot_polling() else 1.0
        self.Sw = None
        self.Sl = None
        self.W = []
        self.L = []
        self.candidates = []
        self.max_polls = max_polls
        self.random_seed = random_seed
        self.m = 0
        self.max_p_value = 1
        self.scrambled = []
        self.primary_column = 'candidate'

        self.preliminary = preliminary

        self.N = self.preliminary['votes'].sum()

        if self.is_ballot_polling():
            self.audit = self._ballot_polling
            self.validate = utils.validate_ballot_polling

        else:
            self.audit = self._batch_comparison
            self.validate = utils.validate_batch_comparison

    def is_ballot_polling(self):
        return self.audit_type == utils.BALLOTPOLLING

    def is_batch_comparison(self):
        return self.audit_type == utils.COMPARISON

    def sanity_check(self):
        assert 0 < self.risk_limit < 1
        assert self.n_winners > 0
        assert self.max_polls > 0
        assert set(self.preliminary.columns) >= self.required_headers

    def _check_audited_ballots(self):
        if self.m >= self.max_polls:
            raise MaxPollsExceededError()

    def _update_accum_recount(self, accum_recount, recount):
        if self.is_ballot_polling():
            for candidate in recount:
                accum_recount[candidate] += recount[candidate]

        else:
            for table in recount:
                for candidate in recount[table]:
                    accum_recount[candidate] += recount[table][candidate]

        return accum_recount

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
        draw_size = m
        if self.is_batch_comparison():
            V = self.preliminary.groupby('table').sum()['votes'].mean()
            draw_size = math.ceil(m / V)

        sample = self.scrambled[:draw_size]
        self.scrambled = self.scrambled[draw_size:]
        tables = {}
        for t, n in sample:
            if t not in tables:
                tables[t] = []

            tables[t].append(str(n))

        for table in tables:
            tables[table].sort()

        if self.is_batch_comparison():
            m = self.preliminary[self.preliminary['table'].isin(tables.keys())]['votes'].sum()

        return tables, m

    def _vote_count_transform(self, vote_count):
        return vote_count

    def _vote_recount_transform(self, recount):
        return recount

    def _init_auditing(self, tables, table_count, pseudo_candidates):
        self._preliminary()
        tables = sorted(tables)
        weights = None
        if self.is_ballot_polling():
            for t in tables:
                self.scrambled.extend(list(zip([t] * table_count[t], range(table_count[t]))))

        else:
            transformed_vote_count = self._vote_count_transform(self.vote_count)
            Wp = [w for w in self.Sw]
            Lp = [l for l in self.Sl]
            margin = {w: {l: transformed_vote_count[w] - transformed_vote_count[l] for l in Lp if l != w} for w in Wp}
            N = len(self.preliminary['table'].unique())
            self.scrambled = np.empty(N, dtype=tuple)
            weights = np.empty(N, dtype=float)
            self.scrambled = list(zip(tables, itertools.repeat('All')))
            i = 0
            for table, group in self.preliminary.groupby('table'):
                vote_count = group.groupby('candidate').sum()['votes'].to_dict()
                self.scrambled[i] = (table, 'All')
                recount = self._vote_recount_transform(vote_count)
                u = utils.batch_error_upper_bound(recount, margin, Wp, Lp)
                weights[i] = u
                i += 1

        # self.scrambled = random_sample(
        #     self.scrambled,
        #     min(self.max_polls, sum(self.vote_count.values())),
        #     method='Fisher-Yates',
        #     prng=int.from_bytes(self.random_seed, 'big')
        # )
        self.scrambled = utils.random_sample(
            population=self.scrambled,
            sample_size=min(self.max_polls, sum(self.vote_count.values())),
            weights=weights,
            seed=self.random_seed
        )
        accum_recount = {c: 0 for c in pseudo_candidates}
        return accum_recount

    def _ballot_polling(self, recount, T):
        transformed_count = self._vote_count_transform(self.vote_count)
        transformed_recount = self._vote_recount_transform(recount)
        return utils.ballot_polling_SPRT(transformed_count, transformed_recount, T, self.risk_limit, self.Sw, self.Sl)

    def _batch_comparison(self, recount, beta):
        transformed_vote_count = self._vote_count_transform(self.vote_count)
        W = [w for w in self.Sw]
        L = [l for l in self.Sl]
        u = utils.MICRO_upper_bound(transformed_vote_count, W, L, self.Sw, self.Sl)
        V = self.preliminary.groupby('table').sum()['votes'].max()
        um = u * V
        U = um * len(self.preliminary['table'].unique())
        for table in recount:
            table_df = self.preliminary[self.preliminary['table'] == table]
            table_report = table_df.groupby(self.primary_column).sum()['votes'].to_dict()
            transformed_table_report = self._vote_count_transform(table_report)
            transformed_recount = self._vote_recount_transform(recount[table])
            beta *= utils.batch_comparison_SPRT(
                transformed_vote_count,
                transformed_table_report,
                transformed_recount,
                self.Sw,
                self.Sl,
                um,
                U
            )

        return beta, 1 / beta

    def recount(self):
        tables, m = self._recount_params()
        while True:
            if self.is_ballot_polling():
                recount = utils.enter_ballot_polling_recount(tables, self.candidates)
                recounted_ballots = sum(recount.values())

            else:
                recount = utils.enter_batch_comparison_recount(tables, self.candidates)
                recounted_ballots = sum([sum(recount[t].values()) for t in recount])

            if recounted_ballots == m:
                break

            print(f'Incorrect recounting: expected {m}, {recounted_ballots} given')

        return recount, m

    def run_audit(self):
        raise NotImplementedError()

    def sample_size(self):
        raise NotImplementedError()


class Plurality(Audit):
    """
    A plurality election chooses the n candidates with most votes.
    If n == 1, this becomes a simple majority election.
    """
    def __init__(self, risk_limit, audit_type, n_winners, max_polls, random_seed, preliminary):
        super().__init__(risk_limit, audit_type, n_winners, max_polls, random_seed, preliminary)
        self.vote_count = self.preliminary.groupby('candidate').sum()['votes'].to_dict()
        self.candidates = list(self.vote_count.keys())
        self.table_count = self.preliminary.groupby('table').sum()['votes'].to_dict()
        self.tables = list(self.table_count.keys())
        self.W, self.L = utils.get_W_L_sets(self.vote_count, self.n_winners)
        self.Sw = {w: 0 for w in self.W}
        self.Sl = {l: 0 for l in self.L}
        self.S = {}
        if self.is_ballot_polling():
            for winner in self.W:
                self.T[winner] = {}
                self.S[winner] = {}
                for loser in self.L:
                    self.T[winner][loser] = Decimal(1.0)
                    self.S[winner][loser] = self.vote_count[winner] / (self.vote_count[winner] + self.vote_count[loser])

    def sample_size(self):
        adjusted_risk_limit = self.risk_limit / self.max_p_value
        m = utils.plurality_sample_size(
            self.vote_count,
            self.W,
            self.L,
            adjusted_risk_limit
        )
        m = min(m, len(self.scrambled))
        return m

    def run_audit(self):
        accum_recount = self._init_auditing(self.tables, self.table_count, self.candidates)
        while True:
            self._check_audited_ballots()
            recount, m = self.recount()
            self.m += m
            accum_recount = self._update_accum_recount(accum_recount, recount)

            self.T, self.max_p_value = self.audit(recount, self.T)
            self._stats(accum_recount)
            if self.validate(self.T, self.risk_limit):
                break

            print('Certainty not sufficient, another round is required.')
            input('Continue... ')


class SuperMajority(Plurality):
    """
    A super majority election chooses the candidate with most votes, if they
    amount to more96 than half the total.
    """
    def __init__(self, risk_limit, audit_type, max_polls, random_seed, preliminary):
        super().__init__(risk_limit, audit_type, 1, max_polls, random_seed, preliminary)
        self.vote_count_s = {'w': self.vote_count[self.W[0]], 'l': sum(self.vote_count[loser] for loser in self.L)}
        self.Sw = {'w': 0}
        self.Sl = {'l': 0}
        self.T = {'w': {'l': Decimal(1.0)}} if self.is_ballot_polling() else 1.0
        self.S = {'w': {'l': self.vote_count[self.W[0]] / sum(self.vote_count.values())}}

    def _vote_count_transform(self, vote_count):
        candidates = list(self.vote_count.keys())
        candidates = sorted(candidates, key=lambda c: self.vote_count[c], reverse=True)
        W = candidates[:self.n_winners]
        L = candidates[self.n_winners:]
        transformed = {
            'w': sum([vote_count[c] for c in W]),
            'l': sum([vote_count[c] for c in L])
        }
        return transformed

    def _vote_recount_transform(self, recount):
        return self._vote_count_transform(recount)


class DHondt(Audit):
    """
    A proportional method, in which the current party votes is divided by the
    number of seats assigned to them + 1.
    """
    def __init__(self, risk_limit, audit_type, n_winners, max_polls, random_seed, preliminary):
        super().__init__(risk_limit, audit_type, n_winners, max_polls, random_seed, preliminary)
        self.required_headers.add('party')
        self.preliminary['party'] = self.preliminary['party'].fillna('')
        self.vote_count = self.preliminary.groupby('party').sum()['votes'].to_dict()
        self.table_count = self.preliminary.groupby('table').sum()['votes'].to_dict()
        self.candidate_count = self.preliminary.groupby('candidate').sum()['votes'].to_dict()
        self.candidates = list(self.candidate_count.keys())
        self.tables = list(self.table_count.keys())
        self.parties = list(self.vote_count.keys())
        self.primary_column = 'party'
        self.party_members = {}
        members_per_party = {}
        for party in self.parties:
            p = self.preliminary[self.preliminary['party'] == party]
            candidates = p.sort_values('votes', ascending=False)['candidate'].unique()
            members_per_party[party] = list(candidates)
            for c in candidates:
                self.party_members[c] = party

        self.pseudo_vote_count = {
            (p, i): utils.p(p, self.vote_count, i) for p in self.parties if p
            for i in range(min(n_winners, len(members_per_party[p])))
        }

        self.W, self.L = utils.get_W_L_sets(self.pseudo_vote_count, n_winners)
        self.winning_candidates = []
        for party in self.parties:
            if any([p == party for p, i in self.W]):
                seats = max([i for p, i in self.W if p == party]) + 1
                self.winning_candidates.extend(members_per_party[party][:seats])

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

        if self.is_ballot_polling():
            for winner in self.Wp:
                self.T[winner] = {loser: Decimal(1) for loser in self.Lp if winner != loser}

        self.Tp = {}
        for p in self.Wp:
            seats = max([w[1] for w in self.W if w[0] == p]) + 1
            self.Tp[p] = Plurality(
                self.risk_limit,
                self.audit_type,
                seats,
                self.max_polls,
                self.random_seed,
                self.preliminary[self.preliminary['party'] == p]
            )

    def sample_size(self):
        adjusted_risk_limit = self.risk_limit / self.max_p_value
        m = utils.dhondt_sample_size(
            self.N,
            adjusted_risk_limit,
            self.vote_count,
            self.Sw,
            self.Sl
        )
        m = min(m, len(self.scrambled))
        return m

    def _party_recount(self, recount):
        if self.is_ballot_polling():
            party_recount = {p: 0 for p in self.parties}
            for c in recount:
                t = self.party_members[c]
                party_recount[t] += recount[c]

        else:
            party_recount = {t: {p: 0 for p in self.parties} for t in recount}
            for table in recount:
                for c in recount[table]:
                    t = self.party_members[c]
                    party_recount[table][t] += recount[table][c]

        return party_recount

    def _preliminary(self):
        print('===========Preliminary results===========')
        L = [loser for loser in self.Lp if loser not in self.Wp]
        utils.print_stats(self.vote_count, self.Wp, L)
        print('=========================================')

    def _stats(self, accum_recount):
        print('=========================================')
        print(f'Max p-value: {self.max_p_value:.5f}')
        print(f'Polled ballots: {self.m}')
        W = [winner for winner in self.party_members if winner in self.winning_candidates]
        L = [loser for loser in self.party_members if loser not in self.winning_candidates]
        utils.print_stats(accum_recount, W, L)
        print('=========================================')

    def _vote_recount_transform(self, recount):
        transformed = {}
        for candidate in recount:
            party = self.party_members[candidate]
            if party not in transformed:
                transformed[party] = 0

            transformed[party] += recount[candidate]

        return transformed

    def run_audit(self):
        accum_recount = self._init_auditing(self.tables, self.table_count, self.party_members)
        while True:
            self._check_audited_ballots()
            recount, m = self.recount()
            self.m += m
            accum_recount = self._update_accum_recount(accum_recount, recount)
            party_recount = self._party_recount(recount)

            self.T, self.max_p_value = self.audit(recount, self.T)
            for p in self.Tp:
                self.Tp[p].T, self.Tp[p].max_p_value = self.Tp[p].audit(recount, self.Tp[p].T)
                self.max_p_value = max(self.max_p_value, self.Tp[p].max_p_value)

            self._stats(accum_recount)
            if self.validate(self.T, self.risk_limit) and all([self.Tp[p].validate(self.Tp[p].T, self.Tp[p].risk_limit) for p in self.Tp]):
                break

            print('Certainty not sufficient, another round is required.')
            input('Continue... ')
