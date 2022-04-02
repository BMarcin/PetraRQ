
Eur-lex-documents
=============================

Eur-lex-documents multilabel long documents classification.
Assign one, more than one or none labels to each doc.

Documents were downloaded from [European Commision's website](https://eur-lex.europa.eu/browse/institutions/eu-commission.html).

## Dockerfile
Repository has got already builded docker image. To use it just pull it from ghcr.io:
```bash
docker pull ghcr.io/BMarcin/PetraRQ:<branch tag>
```

For example to pull docker image for branch "models/logistic_regression" (on github) 
or "logistic_regression" (on gonito.net) run:
```bash
docker pull ghcr.io/BMarcin/PetraRQ:logistic_regression
```

Dataset labels
-------------------
- agriculture
- economy
- education
- energy
- environment
- european_union
- foreign_policy
- health
- industry
- internal_security
- law
- media_informations
- politics_political_parties
- research_science_and_technology
- social_life
- sports
- taxes
- transportation
- work_and_employment

Repository source code
-------------------
[https://github.com/BMarcin/PetraRQ](https://github.com/BMarcin/PetraRQ)
