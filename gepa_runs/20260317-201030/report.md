# GEPA Reasoning V1 Run

- run_id: `20260317-201030`
- track: `mode_abstain`
- budget: `48`
- holdout_budget: `24`
- active_components: `query_mode_rubric, routing_bias_current_vs_temporal, abstain_policy, generic_answer_guardrail, confidence_policy`
- engine_mode: `reflection_lm:openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- reflection_lm: `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- status: `benchmark_candidate`
- leakage_passed: `True`
- num_candidates: `n/a`
- total_metric_calls: `28`

## Candidate Comparison

| candidate | current | temporal | multi_hop | answerable_acc | abstain_precision | abstain_recall | false_abstain | false_confident |
| --------- | ------- | -------- | --------- | -------------- | ----------------- | -------------- | ------------- | --------------- |
| seed      | 0.0000  | 0.0833   | 0.0000    | 0.0000         | 0.3243            | 0.8571         | 0.7353        | 0.2292          |
| gepa_best | 0.0000  | 0.0833   | 0.0000    | 0.0000         | 0.3243            | 0.8571         | 0.7353        | 0.2292          |

Best candidate: `gepa_best`

## Seed Eval

# LoCoMo Reasoning Eval

- query_count: `48`
- joint_answer_or_abstain_acc: `0.2500`
- abstain_precision: `0.3243`
- abstain_recall: `0.8571`
- answer_match_rate: `0.0000`
- answer_evidence_recall: `0.4167`

## Objectives

| objective                      | value  |
| ------------------------------ | ------ |
| answerable_accuracy            | 0.0000 |
| abstain_precision              | 0.3243 |
| abstain_recall                 | 0.8571 |
| false_abstain_penalty          | 0.7353 |
| false_confident_answer_penalty | 0.2292 |

## Offline Slices

| slice        | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| ------------ | ----- | --------- | ------------ | ------------ | ------------ |
| abstain_like | 12    | 0.9167    | 0.9167       | 0.0000       | 0.3333       |
| current      | 12    | 0.0000    | 0.8333       | 0.0000       | 0.4167       |
| multi_hop    | 12    | 0.0000    | 1.0000       | 0.0000       | 0.5000       |
| temporal     | 12    | 0.0833    | 0.3333       | 0.0000       | 0.4167       |

## Predicted Modes

| predicted_mode | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| -------------- | ----- | --------- | ------------ | ------------ | ------------ |
| current        | 35    | 0.3429    | 0.9143       | 0.0000       | 0.3714       |
| multi_hop      | 4     | 0.0000    | 0.7500       | 0.0000       | 0.7500       |
| temporal       | 9     | 0.0000    | 0.2222       | 0.0000       | 0.4444       |

## Failure Buckets

| bucket                   | count |
| ------------------------ | ----- |
| attribute_mismatch       | 10    |
| entity_mismatch          | 0     |
| explanation_quality      | 0     |
| false_abstain            | 25    |
| false_confident_answer   | 16    |
| missing_evidence         | 0     |
| multi_hop_failure        | 1     |
| temporal_selection_error | 5     |

## Failure Examples

1. `locomo_conv-41_q63` | bucket=`false_abstain` | mode=`multi_hop` | gold=`two` | pred=`ABSTAIN`
   question: How many dogs has Maria adopted from the dog shelter she volunteers at?
   support_text: Maria recently adopted a cute pup named Shadow from a shelter.
2. `locomo_conv-42_q177` | bucket=`false_confident_answer` | mode=`temporal` | gold=`movie script` | pred=`25 October 2022`
   question: What did Joanna contribute to that was shown on the big screen on the Sunday before October 25, 2022?
   support_text: Joanna recently had a movie script she contributed to shown on the big screen for the first time.
3. `locomo_conv-47_q80` | bucket=`false_abstain` | mode=`current` | gold=`Ned` | pred=`ABSTAIN`
   question: What is the name of the pup that was adopted by James?
   support_text: James adopted a pup from a shelter in Stamford last week and named it Ned, making his days happier.
4. `locomo_conv-42_q31` | bucket=`false_abstain` | mode=`multi_hop` | gold=`Screenplays,books, online blog posts, journal` | pred=`ABSTAIN`
   question: What kind of writings does Joanna do?
   support_text: Joanna found old notebooks with her early writings and feels appreciative of her progress as a writer.
5. `locomo_conv-41_q23` | bucket=`false_confident_answer` | mode=`temporal` | gold=`March 2023` | pred=`18 April 2023`
   question: When did John go to a convention with colleagues?
   support_text: John attended a convention with colleagues who share his passion for using tech for good in the community.
6. `locomo_conv-41_q86` | bucket=`false_abstain` | mode=`current` | gold=`career fair at a local school` | pred=`ABSTAIN`
   question: What event did John volunteer at last weekend?
   support_text: John recalls having a blast at a live music event with his family last week.
7. `locomo_conv-48_q81` | bucket=`false_confident_answer` | mode=`current` | gold=`no` | pred=`Yes`
   question: Has Jolene tried surfing?
   support_text: Jolene just started learning about surfing but hasn't gone yet.
8. `locomo_conv-47_q64` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`November 4, 2022` | pred=`7 November 2022`
   question: When did James, his family and his dogs start on a road trip together?
   support_text: James and his family visited an animal sanctuary on the road trip and saw many cute rescue dogs.
9. `locomo_conv-41_q84` | bucket=`false_abstain` | mode=`current` | gold=`education and infrastructure` | pred=`ABSTAIN`
   question: What areas is John particularly interested in for policymaking?
   support_text: John is particularly interested in improving education and infrastructure.
10. `locomo_conv-50_q47` | bucket=`false_abstain` | mode=`current` | gold=`Restoring cars` | pred=`ABSTAIN`
   question: What is Dave's favorite activity?
   support_text: Calvin admired Dave's efforts, especially sharing his knowledge and inspiring others.
11. `locomo_conv-49_q59` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`October 2022` | pred=`8 October 2023`
   question: When did Evan start lifting weights?
   support_text: Evan offered advice to Sam on how to start lifting weights and emphasized the importance of good form and technique.
12. `locomo_conv-41_q93` | bucket=`false_abstain` | mode=`current` | gold=`Three times a week` | pred=`ABSTAIN`
   question: How often does John work out with his family?
   support_text: John looks at a picture of his family when times get hard to remember why he is doing his work.
13. `locomo_conv-42_q228` | bucket=`multi_hop_failure` | mode=`multi_hop` | gold=`` | pred=`her writing had a positive impact on someone`
   question: What kind of impact does Joanna hope to have with her painting?
   support_text: Joanna also talked about a touching moment when her writing had a positive impact on someone, making her realize the power of words.
14. `locomo_conv-50_q65` | bucket=`false_abstain` | mode=`current` | gold=`weekly visits to local parks, countryside roadtrip, celebration of the opening of his car maintenance shop, card-playing nights` | pred=`ABSTAIN`
   question: What activities has Dave participated in with his friends?
   support_text: Calvin shared with Dave about throwing a party for his new album last week at his Japanese house, receiving immense love and support from friends and family.
15. `locomo_conv-30_q31` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`20 June, 2023` | pred=`1 February`
   question: When Jon is planning to open his dance studio?
   support_text: At 12:48 am on 1 February 2023, Jon shared with Gina his determination to open a dance studio, while Gina revealed her success in expanding her clothing store.
16. `locomo_conv-50_q46` | bucket=`false_abstain` | mode=`current` | gold=`Working on cars` | pred=`ABSTAIN`
   question: What shared activities do Dave and Calvin have?
   support_text: Calvin shared with Dave about throwing a party for his new album last week at his Japanese house, receiving immense love and support from friends and family.
17. `locomo_conv-49_q50` | bucket=`false_abstain` | mode=`current` | gold=`Evan's son and Evan himself` | pred=`ABSTAIN`
   question: Who was injured in Evan's family?
   support_text: Sam admired Evan's strong family bond and offered support for their upcoming family reunion.
18. `locomo_conv-26_q6` | bucket=`false_confident_answer` | mode=`temporal` | gold=`The sunday before 25 May 2023` | pred=`20 May 2023`
   question: When did Melanie run a charity race?
   support_text: Melanie ran a charity race for mental health last Saturday.
19. `locomo_conv-50_q107` | bucket=`false_abstain` | mode=`current` | gold=`San Francisco` | pred=`ABSTAIN`
   question: Where did Dave come back from with insights on car modification on 1st September 2023?
   support_text: At 9:19 am on 2 September 2023, Dave and Calvin caught up, with Dave sharing car modification insights from San Francisco.
20. `locomo_conv-50_q6` | bucket=`false_abstain` | mode=`current` | gold=`open a car maintenance shop, work on classic cars, build a custom car from scratch` | pred=`ABSTAIN`
   question: What are Dave's dreams?
   support_text: They discussed Ratatouille, following dreams, and Dave's car restoration hobby.

## Explanation Audit

1. `locomo_conv-41_q63` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How many dogs has Maria adopted from the dog shelter she volunteers at?
   explanation: I abstained because the best evidence was not grounded enough to give a stable multi_hop answer. Top support: adopted, maria, shelter.
   support_text: Maria recently adopted a cute pup named Shadow from a shelter.
2. `locomo_conv-47_q80` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What is the name of the pup that was adopted by James?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: adopted, james, pup.
   support_text: James adopted a pup from a shelter in Stamford last week and named it Ned, making his days happier.
3. `locomo_conv-43_q209` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: Where did Tim capture the painting of the sunset over the mountain range?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: over, tim.
   support_text: Tim feels that his passion for fantasy stuff brings him closer to people from all over the world.
4. `locomo_conv-42_q31` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What kind of writings does Joanna do?
   explanation: I abstained because the best evidence was not grounded enough to give a stable multi_hop answer. Top support: joanna, writings.
   support_text: Joanna found old notebooks with her early writings and feels appreciative of her progress as a writer.
5. `locomo_conv-41_q86` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What event did John volunteer at last weekend?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: event, john, last.
   support_text: John recalls having a blast at a live music event with his family last week.
6. `locomo_conv-47_q158` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What did the system John created help the illegal organization with?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: help, john.
   support_text: John is helpful and willing to share resources and tutorials with others to help them learn and grow.
7. `locomo_conv-41_q84` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What areas is John particularly interested in for policymaking?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: interested, john, particularly.
   support_text: John is particularly interested in improving education and infrastructure.
8. `locomo_conv-42_q245` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What does Joanna rely on for cheer and joy?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: joanna, joy.
   support_text: They discussed the joy of having extra cash, watching movies, and supporting each other's endeavors, with Nate even creating something special for Joanna.
9. `locomo_conv-50_q47` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What is Dave's favorite activity?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: dave's.
   support_text: Calvin admired Dave's efforts, especially sharing his knowledge and inspiring others.
10. `locomo_conv-41_q93` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How often does John work out with his family?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: family, his, john, work.
   support_text: John looks at a picture of his family when times get hard to remember why he is doing his work.
11. `locomo_conv-50_q65` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What activities has Dave participated in with his friends?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: dave, friends, his.
   support_text: Calvin shared with Dave about throwing a party for his new album last week at his Japanese house, receiving immense love and support from friends and family.
12. `locomo_conv-50_q46` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What shared activities do Dave and Calvin have?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: calvin, dave, shared.
   support_text: Calvin shared with Dave about throwing a party for his new album last week at his Japanese house, receiving immense love and support from friends and family.


## Best Eval

# LoCoMo Reasoning Eval

- query_count: `48`
- joint_answer_or_abstain_acc: `0.2500`
- abstain_precision: `0.3243`
- abstain_recall: `0.8571`
- answer_match_rate: `0.0000`
- answer_evidence_recall: `0.4167`

## Objectives

| objective                      | value  |
| ------------------------------ | ------ |
| answerable_accuracy            | 0.0000 |
| abstain_precision              | 0.3243 |
| abstain_recall                 | 0.8571 |
| false_abstain_penalty          | 0.7353 |
| false_confident_answer_penalty | 0.2292 |

## Offline Slices

| slice        | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| ------------ | ----- | --------- | ------------ | ------------ | ------------ |
| abstain_like | 12    | 0.9167    | 0.9167       | 0.0000       | 0.3333       |
| current      | 12    | 0.0000    | 0.8333       | 0.0000       | 0.4167       |
| multi_hop    | 12    | 0.0000    | 1.0000       | 0.0000       | 0.5000       |
| temporal     | 12    | 0.0833    | 0.3333       | 0.0000       | 0.4167       |

## Predicted Modes

| predicted_mode | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| -------------- | ----- | --------- | ------------ | ------------ | ------------ |
| current        | 35    | 0.3429    | 0.9143       | 0.0000       | 0.3714       |
| multi_hop      | 4     | 0.0000    | 0.7500       | 0.0000       | 0.7500       |
| temporal       | 9     | 0.0000    | 0.2222       | 0.0000       | 0.4444       |

## Failure Buckets

| bucket                   | count |
| ------------------------ | ----- |
| attribute_mismatch       | 10    |
| entity_mismatch          | 0     |
| explanation_quality      | 0     |
| false_abstain            | 25    |
| false_confident_answer   | 16    |
| missing_evidence         | 0     |
| multi_hop_failure        | 1     |
| temporal_selection_error | 5     |

## Failure Examples

1. `locomo_conv-41_q63` | bucket=`false_abstain` | mode=`multi_hop` | gold=`two` | pred=`ABSTAIN`
   question: How many dogs has Maria adopted from the dog shelter she volunteers at?
   support_text: Maria recently adopted a cute pup named Shadow from a shelter.
2. `locomo_conv-42_q177` | bucket=`false_confident_answer` | mode=`temporal` | gold=`movie script` | pred=`25 October 2022`
   question: What did Joanna contribute to that was shown on the big screen on the Sunday before October 25, 2022?
   support_text: Joanna recently had a movie script she contributed to shown on the big screen for the first time.
3. `locomo_conv-47_q80` | bucket=`false_abstain` | mode=`current` | gold=`Ned` | pred=`ABSTAIN`
   question: What is the name of the pup that was adopted by James?
   support_text: James adopted a pup from a shelter in Stamford last week and named it Ned, making his days happier.
4. `locomo_conv-42_q31` | bucket=`false_abstain` | mode=`multi_hop` | gold=`Screenplays,books, online blog posts, journal` | pred=`ABSTAIN`
   question: What kind of writings does Joanna do?
   support_text: Joanna found old notebooks with her early writings and feels appreciative of her progress as a writer.
5. `locomo_conv-41_q23` | bucket=`false_confident_answer` | mode=`temporal` | gold=`March 2023` | pred=`18 April 2023`
   question: When did John go to a convention with colleagues?
   support_text: John attended a convention with colleagues who share his passion for using tech for good in the community.
6. `locomo_conv-41_q86` | bucket=`false_abstain` | mode=`current` | gold=`career fair at a local school` | pred=`ABSTAIN`
   question: What event did John volunteer at last weekend?
   support_text: John recalls having a blast at a live music event with his family last week.
7. `locomo_conv-48_q81` | bucket=`false_confident_answer` | mode=`current` | gold=`no` | pred=`Yes`
   question: Has Jolene tried surfing?
   support_text: Jolene just started learning about surfing but hasn't gone yet.
8. `locomo_conv-47_q64` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`November 4, 2022` | pred=`7 November 2022`
   question: When did James, his family and his dogs start on a road trip together?
   support_text: James and his family visited an animal sanctuary on the road trip and saw many cute rescue dogs.
9. `locomo_conv-41_q84` | bucket=`false_abstain` | mode=`current` | gold=`education and infrastructure` | pred=`ABSTAIN`
   question: What areas is John particularly interested in for policymaking?
   support_text: John is particularly interested in improving education and infrastructure.
10. `locomo_conv-50_q47` | bucket=`false_abstain` | mode=`current` | gold=`Restoring cars` | pred=`ABSTAIN`
   question: What is Dave's favorite activity?
   support_text: Calvin admired Dave's efforts, especially sharing his knowledge and inspiring others.
11. `locomo_conv-49_q59` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`October 2022` | pred=`8 October 2023`
   question: When did Evan start lifting weights?
   support_text: Evan offered advice to Sam on how to start lifting weights and emphasized the importance of good form and technique.
12. `locomo_conv-41_q93` | bucket=`false_abstain` | mode=`current` | gold=`Three times a week` | pred=`ABSTAIN`
   question: How often does John work out with his family?
   support_text: John looks at a picture of his family when times get hard to remember why he is doing his work.
13. `locomo_conv-42_q228` | bucket=`multi_hop_failure` | mode=`multi_hop` | gold=`` | pred=`her writing had a positive impact on someone`
   question: What kind of impact does Joanna hope to have with her painting?
   support_text: Joanna also talked about a touching moment when her writing had a positive impact on someone, making her realize the power of words.
14. `locomo_conv-50_q65` | bucket=`false_abstain` | mode=`current` | gold=`weekly visits to local parks, countryside roadtrip, celebration of the opening of his car maintenance shop, card-playing nights` | pred=`ABSTAIN`
   question: What activities has Dave participated in with his friends?
   support_text: Calvin shared with Dave about throwing a party for his new album last week at his Japanese house, receiving immense love and support from friends and family.
15. `locomo_conv-30_q31` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`20 June, 2023` | pred=`1 February`
   question: When Jon is planning to open his dance studio?
   support_text: At 12:48 am on 1 February 2023, Jon shared with Gina his determination to open a dance studio, while Gina revealed her success in expanding her clothing store.
16. `locomo_conv-50_q46` | bucket=`false_abstain` | mode=`current` | gold=`Working on cars` | pred=`ABSTAIN`
   question: What shared activities do Dave and Calvin have?
   support_text: Calvin shared with Dave about throwing a party for his new album last week at his Japanese house, receiving immense love and support from friends and family.
17. `locomo_conv-49_q50` | bucket=`false_abstain` | mode=`current` | gold=`Evan's son and Evan himself` | pred=`ABSTAIN`
   question: Who was injured in Evan's family?
   support_text: Sam admired Evan's strong family bond and offered support for their upcoming family reunion.
18. `locomo_conv-26_q6` | bucket=`false_confident_answer` | mode=`temporal` | gold=`The sunday before 25 May 2023` | pred=`20 May 2023`
   question: When did Melanie run a charity race?
   support_text: Melanie ran a charity race for mental health last Saturday.
19. `locomo_conv-50_q107` | bucket=`false_abstain` | mode=`current` | gold=`San Francisco` | pred=`ABSTAIN`
   question: Where did Dave come back from with insights on car modification on 1st September 2023?
   support_text: At 9:19 am on 2 September 2023, Dave and Calvin caught up, with Dave sharing car modification insights from San Francisco.
20. `locomo_conv-50_q6` | bucket=`false_abstain` | mode=`current` | gold=`open a car maintenance shop, work on classic cars, build a custom car from scratch` | pred=`ABSTAIN`
   question: What are Dave's dreams?
   support_text: They discussed Ratatouille, following dreams, and Dave's car restoration hobby.

## Explanation Audit

1. `locomo_conv-41_q63` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How many dogs has Maria adopted from the dog shelter she volunteers at?
   explanation: I abstained because the best evidence was not grounded enough to give a stable multi_hop answer. Top support: adopted, maria, shelter.
   support_text: Maria recently adopted a cute pup named Shadow from a shelter.
2. `locomo_conv-47_q80` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What is the name of the pup that was adopted by James?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: adopted, james, pup.
   support_text: James adopted a pup from a shelter in Stamford last week and named it Ned, making his days happier.
3. `locomo_conv-43_q209` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: Where did Tim capture the painting of the sunset over the mountain range?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: over, tim.
   support_text: Tim feels that his passion for fantasy stuff brings him closer to people from all over the world.
4. `locomo_conv-42_q31` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What kind of writings does Joanna do?
   explanation: I abstained because the best evidence was not grounded enough to give a stable multi_hop answer. Top support: joanna, writings.
   support_text: Joanna found old notebooks with her early writings and feels appreciative of her progress as a writer.
5. `locomo_conv-41_q86` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What event did John volunteer at last weekend?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: event, john, last.
   support_text: John recalls having a blast at a live music event with his family last week.
6. `locomo_conv-47_q158` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What did the system John created help the illegal organization with?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: help, john.
   support_text: John is helpful and willing to share resources and tutorials with others to help them learn and grow.
7. `locomo_conv-41_q84` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What areas is John particularly interested in for policymaking?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: interested, john, particularly.
   support_text: John is particularly interested in improving education and infrastructure.
8. `locomo_conv-42_q245` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What does Joanna rely on for cheer and joy?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: joanna, joy.
   support_text: They discussed the joy of having extra cash, watching movies, and supporting each other's endeavors, with Nate even creating something special for Joanna.
9. `locomo_conv-50_q47` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What is Dave's favorite activity?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: dave's.
   support_text: Calvin admired Dave's efforts, especially sharing his knowledge and inspiring others.
10. `locomo_conv-41_q93` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How often does John work out with his family?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: family, his, john, work.
   support_text: John looks at a picture of his family when times get hard to remember why he is doing his work.
11. `locomo_conv-50_q65` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What activities has Dave participated in with his friends?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: dave, friends, his.
   support_text: Calvin shared with Dave about throwing a party for his new album last week at his Japanese house, receiving immense love and support from friends and family.
12. `locomo_conv-50_q46` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What shared activities do Dave and Calvin have?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: calvin, dave, shared.
   support_text: Calvin shared with Dave about throwing a party for his new album last week at his Japanese house, receiving immense love and support from friends and family.


## Holdout Comparison

| candidate         | current | temporal | multi_hop | answerable_acc | abstain_precision | abstain_recall |
| ----------------- | ------- | -------- | --------- | -------------- | ----------------- | -------------- |
| seed_holdout      | 0.0000  | 0.0000   | 0.0000    | 0.0000         | 0.3000            | 1.0000         |
| gepa_best_holdout | 0.0000  | 0.0000   | 0.0000    | 0.0000         | 0.3000            | 1.0000         |

