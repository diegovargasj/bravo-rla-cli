# BRAVO Risk Limiting Audit Command Line Interface

Based on `Python3`

This project aims to help running a _Risk Limiting Audit_ following the 
definitions on 
[_BRAVO: Ballot-polling Risk-limiting Audits to Verify Outcomes_](https://www.usenix.org/conference/evtwote12/workshop-program/presentation/lindeman) (2012).
This implementation takes into consideration plurality, super majority and D'Hondt contests.
D'Hondt considers one vote per ballot, and voting for a specific candidate 
instead of a party. You can simulate voting for parties if in the candidates 
column you put the name of the party.

It can be used to verify either classical style elections as well as computer 
assisted ones, if it preserved a _voter-verified, demonstrably secure paper trail_.
This can be achieved with _compliance audits_.

## Dependencies

Install the dependencies with `pip3 install -r requirements.txt`. 

## Usage

To run this tool you need to have the preliminary count results in a `.csv` file, 
with the headers `table`, `candidate` and `votes`, if it's a plurality or 
super majority contest. Otherwise, if you are running a D'Hondt contest, the `.csv` 
file needs to have `table`, `party`, `candidate` and `votes` as headers.

    usage: main.py [-h] -r <alpha> [-n <n_winners>] -M <M> -s <type> -x <seed> -f <path/to/file>
    
    Run an RLA
    
    optional arguments:
      -h, --help            show this help message and exit
      -r <alpha>, --risk-limit <alpha>
                            risk limit for the RLA
      -n <n_winners>, --winners <n_winners>
                            number of winners for the election (default 1)
      -M <M>, --max-polls <M>
                            max number of polled ballots
      -s <type>, --social-choice-function <type>
                            social choice function (plurality, super, dhondt)
      -x <seed>, --random-seed <seed>
                            verifiable random seed
      -f <path/to/file>, --preliminary-count-file <path/to/file>
                            path to csv file with preliminary results

You will be asked to recount specific ballots from certain tables, and 
enter the result. If this produces enough evidence to verify the election
within the _risk limit_, the audit will have finished. Otherwise, 
additional ballots will have to be recounted. This will continue until
either the recount process has verified the election, or the amount of
audited ballots has surpassed the maximum number of polled ballots 
specified. 
