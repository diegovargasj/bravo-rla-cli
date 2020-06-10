import bisect
import itertools
import math
import operator
import random
from decimal import Decimal

from clcert_chachagen.chacha20_generator import ChaChaGen

# Audit types
BALLOTPOLLING = 'ballot-polling'
COMPARISON = 'batch-comparison'

# Social choice functions
PLURALITY = 'plurality'
SUPERMAJORITY = 'super'
DHONDT = 'dhondt'


def plurality_sample_size(vote_count, W, L, risk_limit):
    """
    Finds the Average Sample Number for the audit, given its parameters
    and the preliminary results
    @param vote_count   :   {dict<str->int>}
                            The preliminary results of the election. A dictionary
                            with the reported votes per candidate/party.
    @param W            :   {list<str>}
                            List of winners
    @param L            :   {list<str>}
                            List of losers
    @param risk_limit   :   {float}
                            Audit risk limit
    @return             :   {int}
                            Average Sample Number for the audit
    """
    return ASN(risk_limit, vote_count, W, L)


def uMax(party_votes, Sw, Sl):
    """
    Finds the upper bound on the overstatement per ballot on the MICRO for the contest
    @param party_votes  :   {dict<str->int>}
                            Reported casted ballots per party
    @param Sw           :   {dict<str->int>}
                            Largest divisor for any seat the party won
    @param Sl           :   {dict<str->int>}
                            Smallest divisor for any seat the party lost
    @return             :   {float}
                            Upper bound on overstatement per ballot
    """
    u = 0
    for w in Sw:
        for l in Sl:
            if w != l:
                u = max(u, (d(Sw[w]) + d(Sl[l])) / (d(Sl[l]) * party_votes[w] - d(Sw[w]) * party_votes[l]))

    return u


def dhondt_sample_size(ballots, risk_limit, party_votes, Sw, Sl, gamma=0.95):
    """
    Finds the minimum sample size to audit a D'Hondt election
    @param ballots      :   {int}
                            Number of ballots cast in the contest
    @param party_votes  :   {dict<str->int>}
                            Total ballots cast per party
    @param Sw           :   {dict<str->int>}
                            Largest divisor for any seat the party won
    @param Sl           :   {dict<str->int>}
                            Smallest divisor for any seat the party lost
    @param risk_limit   :   {float}
                            Maximum p-value acceptable for any null hypothesis
                            to consider the election verified
    @param gamma        :   {float}
                            Hedge against finding a ballot that attains
                            the upper bound. Larger values give less protection
    @return             :   {int}
                            Sample size to audit
    """
    u = uMax(party_votes, Sw, Sl)
    return math.ceil(
        math.log(1 / risk_limit) / math.log(gamma / (1 - 1 / (ballots * u)) + 1.0 - gamma)
    )


def print_stats(count, W, L):
    """
    Prints candidate stats
    @param count    :   {dict<str->int>}
                        Ballot count per candidate
    @param W        :   {list<str>}
                        List of winners
    @param L        :   {list<str>}
                        List of losers
    """
    for winner in W:
        print(f' * {winner}: {count[winner]}')

    for loser in L:
        print(f'   {loser}: {count[loser]}')


def get_W_L_sets(vote_count, n_winners):
    """
    Obtains the winner and loser sets, given the amount of votes
    for each candidate
    @param vote_count   :   {dict<str->int>}
                            Dictionary with the reported amount of votes
                            per candidate
    @param n_winners    :   {int}
                            Number of winners for the election
    @return             :   {tuple<list<str>,list<str>>}
                            Tuple with the winners and losers sets
    """
    tuples = list(vote_count.items())
    sorted_tuples = sorted(tuples, key=operator.itemgetter(1), reverse=True)
    W = [c[0] for c in sorted_tuples[:n_winners]]
    L = [c[0] for c in sorted_tuples[n_winners:]]
    return W, L


def d(s):
    """
    Returns the divisor for column s. In this case, the divisor of
    column s is always s + 1
    @param s    :   {int}
                    Column number, starting from 0
    @return     :   {int}
                    Divisor of column s
    """
    return s + 1


def t(p, vote_count):
    """
    Gets the reported number of votes for party p
    @param p            :   {str}
                            The party in question for which we want to get the
                            reported number of votes
    @param vote_count   :   {dict<str->int>}
                            Dictionary with the reported number of votes
                            per party
    @return             :   {int}
                            Reported number of votes for party p
    """
    return vote_count[p]


def p(party, vote_count, s):
    """
    Returns the reported number of votes for party p, divided by the divisor
    of column s
    @param party        :   {str}
                            Party in question
    @param vote_count   :   {dict<str->int>}
                            Dictionary with the reported number of votes
                            per party
    @param s            :   {int}
                            Column number
    @return             :   {float}
                            Reported number of votes for party p, divided by
                            the divisor of column s
    """
    return t(party, vote_count) / d(s)


def e(p, reported, recount):
    """
    Error between reported and recounted cast ballots for a party
    @param p        :   {str}
                        Party name
    @param reported :   {dict<str->int>}
                        Reported cast ballots per party
    @param recount  :   {dict<str->int>}
                        Recounted cast ballots per party
    @return         :   {int}
                        Reported - recounted cast ballots for party p
    """
    return t(p, reported) - t(p, recount)


def gamma(p, q, vote_count, Sw=None, Sl=None):
    """
    Likelihood ratio for the null/alternative hypothesis
    (depending on which candidate is reportedly winning)
    between reported winning and reported losing candidates
    @param p            :   {str}
                            First pseudo candidate
    @param q            :   {str}
                            Second pseudo candidate
    @param vote_count   :   {dict<str->int>}
                            Dictionary with the reported number of votes
                            per pseudo candidate
    @param Sw           :   {dict<str->int>}
                            Max seat number for each winning party
    @param Sl           :   {dict<str->int}
                            Min seat number for each losing party
    @return             :   {float}
                            Likelihood ratio between pseudo candidates p and q,
                            given a vote for p
    """
    if Sw is None:
        Sw = {p: 0}

    if Sl is None:
        Sl = {q: 0}

    y1 = t(p, vote_count) / (t(p, vote_count) + t(q, vote_count))
    y2 = (d(Sw[p]) + d(Sl[q])) / d(Sw[p])
    return Decimal(y1 * y2)


def validate_ballot_polling(T, alpha):
    """
    Checks if the election has been validated using a ballot polling auditing scheme
    @param T    :   {dict<str->dict<str->float>>}
                    Dictionary of dictionaries containing the current inverse of
                    p-values for all the null hypothesis being checked
    @param alpha:   {float}
                    Risk limit for this audit
    @return     :   {bool}
                    True if the election has been validated with a risk limit of
                    <alpha>, else False
    """
    for winner in T:
        for loser in T[winner]:
            if T[winner][loser] < 1 / alpha:
                return False

    return True


def validate_batch_comparison(beta, alpha):
    """
    Checks if the election has been validated using a batch comparison auditing scheme
    @param beta :   {float}
                    Current SPRT coefficient
    @param alpha:   {float}
                    Risk limit
    @return     :   {bool}
                    True if the election has been validated with a risk limit of
                    <alpha>, else False
    """
    return beta >= 1 / alpha


def ballot_polling_SPRT(vote_count, recount, T, risk_limit, Sw=None, Sl=None):
    """
    Applies Wald's Sequential Probability Ration Test to calculate the inverse
    of the p-values for every null hypothesis between every winner and every
    loser
    @param vote_count   :   {dict<str->int>}
                            Dictionary with preliminary vote count per candidate
    @param recount      :   {dict<str->int>}
                            Dictionary with newly recounted ballots per candidate
    @param T            :   {dict<str->dict<str->float>>}
                            Dictionary of dictionaries containing the inverse of
                            the p-values for every null hypothesis per loser,
                            per winner
    @param risk_limit   :   {float}
                            Maximum p-value acceptable for any null hypothesis
                            to consider the election verified
    @param Sw           :   {dict<str->float>}
                            Dictionary of coefficients per winner to adjust
                            SPRT coefficient
    @param Sl           :   {dict<str->float}
                            Dictionary of coefficients per loser to adjust
                            SPRT coefficient
    @return             :   tuple<dict<str->dict<str->float>>, float>
                            Tuple with the inverse of the p-values for every null
                            hypothesis per loser, per winner, and the maximum
                            p-value registered on this round
    """
    max_p_value = 0
    for winner in T:
        for loser in T[winner]:
            if T[winner][loser] < 1 / risk_limit:
                y1 = gamma(winner, loser, vote_count, Sw, Sl)
                y2 = gamma(loser, winner, vote_count, Sl, Sw)
                T[winner][loser] *= y1 ** recount[winner] * y2 ** recount[loser]

            max_p_value = max(max_p_value, 1 / T[winner][loser])

    return T, max_p_value


def batch_comparison_SPRT(reported_count, table_report, table_recount, W, L, um, U, gamma=0.95):
    """
        Calculates Wald's Sequential Probability Ratio Test for the worst possible case
        in the table
        @param reported_count   :   {dict<str->int>}
                                    Reported cast ballots for each candidate
        @param table_report     :   {dict<str->int>}
                                    Reported cast ballots for each candidate in a single table
        @param table_recount    :   {dict<str->int>}
                                    Recounted cast ballots for each candidate in a single table
        @param W                :   {list<tuple<str,int>>}
                                    List of tuples with pairs winning candidate, column
        @param L                :   {list<tuple<str,int>>}
                                    List of tuples with pairs losing candidate, column
        @param um               :   {float}
                                    Upper bound on the MICRO for the table, scaled for multiple
                                    votes per table
        @param U                :   {float}
                                    Upper bound on the MICRO for the whole contest
        @param gamma            :   {float}
                                    Security factor for escalating on errors
        @return                 :   {float}
                                    Update factor for the probability ratio on the contest
    """
    micro = MICRO(reported_count, table_report, table_recount, W, L)
    Dm = micro / um
    return gamma * (1 - Dm) / (1 - 1 / U) + 1 - gamma


def MICRO(reported, table_report, recount, W, L):
    """
    Maximum In Contest Relative Overstatement for a table
    @param reported     :   {dict<str->int>}
                            Reported cast ballots per candidate
    @param table_report :   {dict<str->int>}
                            Reported cast ballots per candidate in a single table
    @param recount      :   {dict<str->int>}
                            Recounted cast ballots per candidate in a single table
    @param W            :   {list<tuple<str,int>>}
                            List of tuples with pairs winning candidate, column
    @param L            :   {list<tuple<str,int>>}
                            List of tuples with pairs losing candidate, column
    @return             :   {float}
                            MICRO for the recounted table
    """
    micro = 0
    for pw in W:
        for pl in L:
            if pw != pl:
                x = d(L[pl]) * e(pw, table_report, recount) - d(W[pw]) * e(pl, table_report, recount)
                y = (d(L[pl]) * reported[pw] - d(W[pw]) * reported[pl])
                micro = max(micro, x / y)

    return micro


def MICRO_upper_bound(reported, Wp, Lp, Sw=None, Sl=None):
    """
    MICRO upper bound for the contest
    @param reported :   {dict<str->int>}
                        Reported cast ballots per candidate
    @param Wp       :   {list<str>}
                        List of candidates that won at least 1 seat
    @param Lp       :   {list<str>}
                        List of candidates that lost at least 1 seat
    @param Sw       :   {dict<str->int>}
                        Largest divisor for any seat the party won
    @param Sl       :   {dict<str->int>}
                        Smallest divisor for any seat the party lost
    @return         :   {float}
                        Upper bound for MICRO in the contest
    """
    if not Sw:
        Sw = {w: 0 for w in Wp}

    if not Sl:
        Sl = {l: 0 for l in Lp}

    u = 0
    for w in Wp:
        for l in Lp:
            if w != l:
                curr_u = (d(Sl[l]) + d(Sw[w])) / (d(Sl[l]) * reported[w] - d(Sw[w]) * reported[l])
                u = max(u, curr_u)

    return u


def ASN(risk_limit, vote_count, W, L):
    """
    Wald's Average Sample Number to estimate the number of ballots needed to
    sample to verify the election
    @param risk_limit   :   {float}
                            Maximum p-value acceptable for any null hypothesis
                            to consider the election verified
    @param vote_count   :   {dict<str->int>}
                            Number of reported ballots per candidate
    @param W            :   {list<str>}
                            List of winners
    @param L            :   {list<str>}
                            List of losers
    @return             :   {int}
                            Estimated number of ballots needed to audit to
                            verify the election
    """
    asn = 0
    for w in W:
        for l in L:
            pw = vote_count[w] / (vote_count[w] + vote_count[l])
            pl = vote_count[l] / (vote_count[w] + vote_count[l])
            sw = pw
            zw = math.log(2 * sw)
            zl = math.log(2 - 2 * sw)
            curr_asn = (math.log(1 / risk_limit) + zw / 2) / (pw * zw + pl * zl)
            asn = max(asn, curr_asn)

    return math.ceil(asn)


def enter_ballot_polling_recount(tables, candidates):
    """
    Terminal interface for the user to enter the recount results
    @param tables       :   {dict<str->list<int>>}
                            Ballot indices to be recounted, per table
    @param candidates   :   {list<str>}
                            List of candidates in the election
    @return             :   {dict<str->int>}
                            Number of recounted votes per candidate
    """
    tot = 0
    for table in sorted(tables):
        print(f'Recount from table {table} ballots number {", ".join(tables[table])}')
        tot += len(tables[table])

    print(f'Total {tot}')
    recount = {c: 0 for c in candidates}
    for candidate in candidates:
        c = input(f'{candidate}: ')
        count = 0
        if c:
            count = int(c)

        recount[candidate] += count

    return recount


def enter_batch_comparison_recount(tables, candidates):
    recount = {table: {} for table in tables}
    tables = sorted(tables)
    print(f'You will need to recount a total of {len(tables)} tables.')
    print(', '.join(tables))
    for table in tables:
        print(f'Recount all ballots from table {table}')
        for candidate in candidates:
            c = input(f'{candidate}: ')
            count = 0
            if c:
                count = int(c)

            recount[table][candidate] = count

    return recount


def random_sample(population, sample_size, weights=None, seed=None):
    """
    Generate a random sample from the given population, following the weight
    distribution and the random seed
    @param population   :   {iterable}
                            Universe from which to obtain the random sample
    @param sample_size  :   {int}
                            Random sample size
    @param weights      :   {list<float>}
                            Weight distribution for the random sample
    @param seed         :   {int|bytes}
                            Random seed
    @return             :   {list<any>}
                            Random sample of size <sample_size>
    """
    chacha = ChaChaGen(seed=seed)
    if weights is None:
        weights = [1] * len(population)

    cum_weights = list(itertools.accumulate(weights))
    total = cum_weights[-1]
    hi = len(cum_weights) - 1
    return [
        population[bisect.bisect(cum_weights, chacha.random() * total, 0, hi)]
        for i in range(sample_size)
    ]


def batch_error_upper_bound(batch_count, margin, Wp, Lp):
    """
    Upper bound on the error for a specific batch
    @param batch_count  :   {dict<str->int>}
                            Vote count for each candidate on the batch
    @param margin       :   {dict<str->dict<str->int>>}
                            Margin between each winner and loser
    @param Wp           :   {list<str>}
                            List of candidates that won at least 1 seat
    @param Lp           :   {list<str>}
                            List of candidates that lost at least 1 seat
    @return             :   {float}
                            Maximum upper bound on the error for the batch
    """
    up = 0
    np = sum(batch_count.values())
    for w in Wp:
        for l in Lp:
            if w != l:
                up = max(
                    up,
                    (batch_count[w] - batch_count[l] + np) / margin[w][l]
                )

    return up
