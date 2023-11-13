# MOSAiC in-ice light profiles

## Contact
Niels Fuchs
niels.fuchs@uni-hamburg.de

## Scope

The repository is a collection of code to convert measurement data from MOSAiC buoys lightharp and lightchains into NetCDF standard profiles with derived quantities. The NetCDF files follow the CF conventions. Code was developped with the BMBF funded project NiceLABpro. 

## Input data

Input data is available in csv format from Meereisportal and PANGAEA.

## Calibration data

Calibration data used for radiometric corrections and harmonization of individual sensors on instruments is included in the repository.

## Output data

Output data is available on PANGAEA

## Further descriptions

A data descriptor paper is in submission to Nature data journal.

## Usage notes

Data conversion can be run with read_raw_data.py . Derived quantities are calculated in obtain\_published\_variables.py . Attributes (e.g. unit names, variable names) are collected in LigthAttributes.py . 
