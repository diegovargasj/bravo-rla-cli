import argparse
import os

import pandas as pd

import utils
from audits import Plurality, SuperMajority, DHondt, MaxPollsExceededError

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a Ballot-Polling or Batch-Comparison Risk Limiting Audit through the terminal'
    )
    parser.add_argument(
        '-r',
        '--risk-limit',
        metavar='<alpha>',
        dest='risk_limit',
        type=float,
        required=True,
        help='risk limit for the RLA'
    )
    parser.add_argument(
        '-n',
        '--winners',
        metavar='<n_winners>',
        dest='n_winners',
        type=int,
        default=1,
        help='number of winners for the election (default 1)'
    )
    parser.add_argument(
        '-M',
        '--max-polls',
        metavar='<M>',
        dest='M',
        type=int,
        required=True,
        help='max number of polled ballots'
    )
    parser.add_argument(
        '-s',
        '--social-choice-function',
        metavar='<type>',
        dest='social_choice',
        required=True,
        choices=[utils.PLURALITY, utils.SUPERMAJORITY, utils.DHONDT],
        help=f'social choice function ({utils.PLURALITY}, {utils.SUPERMAJORITY}, {utils.DHONDT})'
    )
    parser.add_argument(
        '-a',
        '--audit-type',
        metavar='<type>',
        dest='audit_type',
        required=True,
        choices=[utils.BALLOTPOLLING, utils.COMPARISON],
        help=f'auditing scheme ({utils.BALLOTPOLLING}, {utils.COMPARISON})'
    )
    parser.add_argument(
        '-x',
        '--random-seed',
        metavar='<seed>',
        dest='seed',
        default=os.urandom(64).hex(),
        help='random seed, default 512 random bits'
    )
    parser.add_argument(
        '-f',
        '--preliminary-count-file',
        metavar='</path/to/file>',
        dest='preliminary',
        required=True,
        help='path to csv file with preliminary results'
    )
    args = parser.parse_args()
    risk_limit = args.risk_limit
    n_winners = args.n_winners
    max_polls = args.M
    social_choice_function = args.social_choice
    audit_type = args.audit_type
    random_seed = args.seed.encode()
    preliminary_file = args.preliminary

    print(f'Reading csv file {preliminary_file}')
    preliminary = pd.read_csv(preliminary_file)

    print(f'Creating audit for {social_choice_function}')
    if social_choice_function == utils.PLURALITY:
        audit = Plurality(risk_limit, audit_type, n_winners, max_polls, random_seed, preliminary)

    elif social_choice_function == utils.SUPERMAJORITY:
        audit = SuperMajority(risk_limit, audit_type, max_polls, random_seed, preliminary)

    elif social_choice_function == utils.DHONDT:
        audit = DHondt(risk_limit, audit_type, n_winners, max_polls, random_seed, preliminary)

    else:
        raise NotImplementedError(f'Social choice function {social_choice_function} not implemented')

    audit.sanity_check()
    try:
        audit.run_audit()
        print(f'Election verified with a p-value of {audit.max_p_value:f} <= {risk_limit}')

    except MaxPollsExceededError as exn:
        print('Could not verify election, consider a full hand count')
        print(exn)
