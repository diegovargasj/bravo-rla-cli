import math
import operator
from decimal import Decimal


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
    column s is always s
    @param s    :   {int}
                    Column number
    @return     :   {int}
                    Divisor of column s
    """
    return s


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
        Sw = {p: 1}

    if Sl is None:
        Sl = {q: 1}

    y1 = t(p, vote_count) / (t(p, vote_count) + t(q, vote_count))
    y2 = (d(Sw[p]) + d(Sl[q])) / d(Sw[p])
    return Decimal(y1 * y2)


def validated(T, alpha):
    """
    Checks if the election has been validated
    @param T    :   {dict<str->dict<str->float>>}
                    Dictionary of dictionaries containing the current inverse of
                    p-values for all the null hypothesis being checked
    @param alpha:   {float}
                    Risk limit for this audit
    @return     :   {bool}
                    True if the election has been validated given the selected
                    risk limit, else False
    """
    for winner in T:
        for loser in T[winner]:
            if T[winner][loser] < 1 / alpha:
                return False

    return True


def SPRT(vote_count, recount, T, risk_limit, Sw=None, Sl=None):
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


def ASN(risk_limit, pw, pl, tot):
    """
    Wald's Average Sample Number to estimate the number of ballots needed to
    sample to verify the election
    @param risk_limit   :   {float}
                            Maximum p-value acceptable for any null hypothesis
                            to consider the election verified
    @param pw           :   {int}
                            Number of reported ballots for the winner with
                            the least votes
    @param pl           :   {int}
                            Number of reported ballots for the loser with
                            most votes
    @param tot          :   {int}
                            Total number of casted ballots
    @return             :   {int}
                            Estimated number of ballots needed to audit to
                            verify the election
    """
    margin = (pw - pl) / tot
    return int(2 * math.log(1 / risk_limit) / margin ** 2)


def enter_recount(tables, candidates):
    tot = 0
    sorted_tables = list(tables.keys())
    sorted_tables.sort()
    for table in sorted_tables:
        print(f'Recount from table {table} ballots number {tables[table]}')
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
