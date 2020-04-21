import argparse

import pandas as pd

from audits import Plurality, SuperMajority, DHondt, MaxPollsExceededError

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an RLA')
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
        choices=['plurality', 'super', 'dhondt'],
        help='social choice function (plurality, super, dhondt)'
    )
    parser.add_argument(
        '-x',
        '--random-seed',
        metavar='<seed>',
        dest='seed',
        required=True,
        help='verifiable random seed'
    )
    parser.add_argument(
        '-f',
        '--preliminary-count-file',
        metavar='<path/to/file>',
        dest='preliminary',
        required=True,
        help='path to csv file with preliminary results'
    )
    args = parser.parse_args()
    risk_limit = args.risk_limit
    n_winners = args.n_winners
    max_polls = args.M
    social_choice_function = args.social_choice
    random_seed = args.seed
    preliminary_file = args.preliminary

    print(f'Reading csv file {preliminary_file}')
    preliminary = pd.read_csv(preliminary_file)

    print(f'Creating audit for {social_choice_function}')
    if social_choice_function == 'plurality':
        audit = Plurality(risk_limit, n_winners, max_polls, random_seed, preliminary_file)

    elif social_choice_function == 'super':
        audit = SuperMajority(risk_limit, max_polls, random_seed, preliminary_file)

    else:  # social_choice_function == 'dhondt'
        audit = DHondt(risk_limit, n_winners, max_polls, random_seed, preliminary_file)

    try:
        audit.ballot_polling()
        print(f'Election verified with a p-value of {audit.max_p_value:f} <= {risk_limit}')

    except MaxPollsExceededError as exn:
        print('Could not verify election, consider a full hand count')
        print(exn)
