# BRAVO Risk Limiting Audit Command Line Interface

This project aims to help running a _Risk Limiting Audit_ following the 
definitions on 
[_BRAVO: Ballot-polling Risk-limiting Audits to Verify Outcomes_](https://www.usenix.org/conference/evtwote12/workshop-program/presentation/lindeman) (2012).
This implementation takes into consideration plurality, super majority and D'Hondt contests.

It can be used to verify either classical style elections as well as computer 
assisted ones, if it preserved a _voter-verified, demonstrably secure paper trail_.
This can be achieved with _compliance audits_.

## Usage

To run this tool you need to have the preliminary count results in a `.csv` file, 
with the headers `table`, `candidate` and `votes`, if it's a plurality or 
super majority contest. Otherwise, if you are running a D'Hondt contest, the `.csv` 
file needs to have `table`, `party`, `candidate` and `votes` as headers.