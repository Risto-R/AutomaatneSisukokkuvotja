## Automaatne sisukokkuvõtete tegemine ajaseoste märgenduste põhjal

Kasutatud korpus: https://github.com/soras/EstTimeMLCorpus

Kaustas "Artiklid" on olemas automaatselt loodud kokkuvõtted, kasutades erinevaid meetodeid kujul artikli_nimi_auto_summary_x

|                |                                                          | Kasutatud seosed                                             | Saagis (recall) | Täpsus (precision) | F-mõõt (F-measure) |
|----------------|----------------------------------------------------------|--------------------------------------------------------------|-----------------|--------------------|--------------------|
| auto_summary_1 | Pikim ahel (Üks pikim ahel)                              | BEFORE, AFTER                                                | 0.24873         | 0.44595            | 0.30314            |
| auto_summary_2 | PageRank                                                 | BEFORE, AFTER                                                | 0.46927         | 0.45784            | 0.45890            |
| auto_summary_3 | Pikim ahel (kõik pikimad ahelad)                         | BEFORE, AFTER                                                | 0.28379         | 0.44365            | 0.33071            |
| auto_summary_4 | Pikim ahel (Üks pikim ahel)                              | BEFORE,AFTER, BEFORE-OVERLAP, AFTER-OVERLAP                  | 0.23460         | 0.38149            | 0.27738            |
| auto_summary_5 | PageRank                                                 | BEFORE,AFTER, BEFORE-OVERLAP, AFTER-OVERLAP                  | 0.46356         | 0.44594            | 0.45065            |
| auto_summary_6 | Pikim ahel (kõik pikimad ahelad)                         | BEFORE,AFTER, BEFORE-OVERLAP, AFTER-OVERLAP                  | 0.25516         | 0.39491            | 0.29715            |
| auto_summary_7 | Pikim ahel(kõik pikimad ahelad)                          | BEFORE, AFTER, SIMULTANEOUS, IS_INCLUDED, INCLUDES, IDENTITY | 0.32764         | 0.42348            | 0.35893            |
| auto_summary_8 | PageRank                                                 | BEFORE, AFTER, SIMULTANEOUS, IS_INCLUDED, INCLUDES, IDENTITY | 0.45165         | 0.41234            | 0.42807            |
| auto_summary_9 | Pikim ahel(kõik pikimad ahelad, ajaväljendite olemasolu) | BEFORE, AFTER, SIMULTANEOUS, IS_INCLUDED, INCLUDES, IDENTITY | 0.340001        | 0.44555            | 0.37473            |
