# LoCoMo Breakdown

- source: `../data/locomo10.json`
- adapter_version: `locomo_adapter_v1`
- locomo_version: `locomo10_pinned_release_with_dec_2025_answer_fixes`
- queries: `1986`
- answerable: `1542`
- abstain_like: `444`
- with_evidence: `1982`
- avg_evidence_ids: `1.4215`

## Query Mode

| query_mode  | count | share_% | answerable | abstain_like | with_evidence | no_evidence | multi_evidence | avg_evidence_ids |
| ----------- | ----- | ------- | ---------- | ------------ | ------------- | ----------- | -------------- | ---------------- |
| adversarial | 400   | 20.14   | 2          | 398          | 400           | 0           | 14             | 1.03             |
| current     | 244   | 12.29   | 242        | 2            | 244           | 0           | 233            | 3.06             |
| historical  | 4     | 0.20    | 2          | 2            | 4             | 0           | 0              | 1.00             |
| multi_hop   | 888   | 44.71   | 885        | 3            | 884           | 4           | 126            | 1.27             |
| temporal    | 450   | 22.66   | 411        | 39           | 450           | 0           | 54             | 1.18             |

## Category

| category | count | share_% | answerable | abstain_like | with_evidence | no_evidence | multi_evidence | avg_evidence_ids |
| -------- | ----- | ------- | ---------- | ------------ | ------------- | ----------- | -------------- | ---------------- |
| 1        | 282   | 14.20   | 282        | 0            | 282           | 0           | 277            | 3.13             |
| 2        | 321   | 16.16   | 321        | 0            | 321           | 0           | 40             | 1.17             |
| 3        | 96    | 4.83    | 96         | 0            | 92            | 4           | 50             | 2.17             |
| 4        | 841   | 42.35   | 841        | 0            | 841           | 0           | 46             | 1.07             |
| 5        | 446   | 22.46   | 2          | 444          | 446           | 0           | 14             | 1.03             |

## Query Mode x Category

| query_mode  | 1   | 2   | 3  | 4   | 5   |
| ----------- | --- | --- | -- | --- | --- |
| adversarial | 0   | 0   | 0  | 0   | 400 |
| current     | 236 | 1   | 0  | 5   | 2   |
| historical  | 0   | 0   | 0  | 2   | 2   |
| multi_hop   | 34  | 1   | 92 | 758 | 3   |
| temporal    | 12  | 319 | 4  | 76  | 39  |

## Top Attributes Per Query Mode

- `adversarial`: `other` (254), `location` (35), `possession` (28), `preference` (21), `plan` (18)
- `current`: `other` (110), `possession` (78), `location` (15), `preference` (14), `relationship` (11)
- `historical`: `other` (4)
- `multi_hop`: `other` (560), `location` (73), `preference` (64), `possession` (60), `plan` (37)
- `temporal`: `date` (276), `other` (106), `location` (18), `possession` (15), `plan` (15)

