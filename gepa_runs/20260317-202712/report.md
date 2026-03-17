# GEPA Reasoning V1 Run

- run_id: `20260317-202712`
- track: `temporal_selection`
- budget: `32`
- holdout_budget: `16`
- active_components: `temporal_policy, temporal_grounding_rule, answer_synthesis_policy`
- engine_mode: `reflection_lm:openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- reflection_lm: `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- status: `benchmark_candidate`
- leakage_passed: `True`
- num_candidates: `n/a`
- total_metric_calls: `22`

## Candidate Comparison

| candidate | current | temporal | multi_hop | answerable_acc | abstain_precision | abstain_recall | false_abstain | false_confident |
| --------- | ------- | -------- | --------- | -------------- | ----------------- | -------------- | ------------- | --------------- |
| seed      | 0.0625  | 0.2500   | 0.0000    | 0.0690         | 0.1579            | 1.0000         | 0.5517        | 0.3438          |
| gepa_best | 0.0625  | 0.2500   | 0.0000    | 0.0690         | 0.1579            | 1.0000         | 0.5517        | 0.3438          |

Best candidate: `gepa_best`

## Seed Eval

# LoCoMo Reasoning Eval

- query_count: `32`
- joint_answer_or_abstain_acc: `0.1562`
- abstain_precision: `0.1579`
- abstain_recall: `1.0000`
- answer_match_rate: `0.0625`
- answer_evidence_recall: `0.4375`

## Objectives

| objective                      | value  |
| ------------------------------ | ------ |
| answerable_accuracy            | 0.0690 |
| abstain_precision              | 0.1579 |
| abstain_recall                 | 1.0000 |
| false_abstain_penalty          | 0.5517 |
| false_confident_answer_penalty | 0.3438 |

## Offline Slices

| slice    | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| -------- | ----- | --------- | ------------ | ------------ | ------------ |
| current  | 16    | 0.0625    | 0.8750       | 0.0000       | 0.4375       |
| temporal | 16    | 0.2500    | 0.3125       | 0.1250       | 0.4375       |

## Predicted Modes

| predicted_mode | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| -------------- | ----- | --------- | ------------ | ------------ | ------------ |
| current        | 21    | 0.1905    | 0.9048       | 0.0476       | 0.3810       |
| multi_hop      | 1     | 0.0000    | 0.0000       | 0.0000       | 1.0000       |
| temporal       | 10    | 0.1000    | 0.0000       | 0.1000       | 0.5000       |

## Failure Buckets

| bucket                   | count |
| ------------------------ | ----- |
| attribute_mismatch       | 10    |
| entity_mismatch          | 0     |
| explanation_quality      | 0     |
| false_abstain            | 16    |
| false_confident_answer   | 15    |
| missing_evidence         | 0     |
| multi_hop_failure        | 1     |
| temporal_selection_error | 6     |

## Failure Examples

1. `locomo_conv-42_q30` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`The weekend of 24June, 2022.` | pred=`24 June 2022`
   question: When is Joanna going to make Nate's ice cream for her family?
   support_text: Joanna appreciates Nate's support for her ice cream making.
2. `locomo_conv-43_q2` | bucket=`false_abstain` | mode=`current` | gold=`get endorsements, build his brand, do charity work` | pred=`ABSTAIN`
   question: What are John's goals for his career that are not related to his basketball skills?
   support_text: They discussed John's passion for basketball, his career goals, and plans for life after sports.
3. `locomo_conv-42_q55` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`The Sunday before 25October, 2022.` | pred=`25 October 2022`
   question: When was Joanna's second movie script shown on the big screens?
   support_text: Joanna recently had a movie script she contributed to shown on the big screen for the first time.
4. `locomo_conv-43_q14` | bucket=`false_abstain` | mode=`current` | gold=`Read fantasy books.` | pred=`ABSTAIN`
   question: What does Tim do to escape reality?
   support_text: Tim loves curling up with a good book to escape reality and get lost in different worlds during his free time.
5. `locomo_conv-48_q143` | bucket=`false_abstain` | mode=`current` | gold=`It really showed how much she loves her.` | pred=`ABSTAIN`
   question: How does Jolene describe the feeling of finding her snake snuggled under the bed after it got out?
   support_text: Jolene had a 'snake adventure' where her snake got out, but she found her snuggling under the bed after hours of searching.
6. `locomo_conv-43_q66` | bucket=`false_abstain` | mode=`current` | gold=`Lord of the Rings, Harry Potter, and Star Wars.` | pred=`ABSTAIN`
   question: What fantasy movies does Tim like?
   support_text: Tim enjoys escaping in books and movies, particularly fantasy genres like "The Lord of the Rings", and looks forward to a new TV series, "The Wheel of Time".
7. `locomo_conv-44_q49` | bucket=`false_abstain` | mode=`current` | gold=`Family` | pred=`ABSTAIN`
   question: What does Andrew view his pets as?
   support_text: Andrew considered getting another dog but decided to focus on his current pets first.
8. `locomo_conv-50_q26` | bucket=`false_confident_answer` | mode=`temporal` | gold=`on the Friday before 22 August, 2023` | pred=`18 August 2023`
   question: When did Dave host a card-playing night with his friends?
   support_text: Dave had a card night with friends last Friday and enjoyed it a lot.
9. `locomo_conv-50_q30` | bucket=`false_abstain` | mode=`current` | gold=`take a walk, go hiking, listen to favorite albums, live concerts, photography` | pred=`ABSTAIN`
   question: What are Dave's hobbies other than fixing cars?
   support_text: Dave and Calvin, at 3:13 pm on 8 October, 2023, discussed Dave's passion for cars.
10. `locomo_conv-50_q71` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`November 16, 2023` | pred=`17 November 2023`
   question: When did Calvin attend a gala in Boston?
   support_text: Calvin attended a fancy gala in Boston recently and met interesting people.
11. `locomo_conv-49_q12` | bucket=`false_abstain` | mode=`current` | gold=`Weight problem` | pred=`ABSTAIN`
   question: What health issue did Sam face that motivated him to change his lifestyle?
   support_text: Sam feels a mix of emotions, somewhat concerned about his health but motivated to make positive changes.
12. `locomo_conv-26_q40` | bucket=`false_abstain` | mode=`current` | gold=`Joining activist group, going to pride parades, participating in an art show, mentoring program` | pred=`ABSTAIN`
   question: In what ways is Caroline participating in the LGBTQ community?
   support_text: Caroline expressed how excited she was about her work volunteering for the LGBTQ+ community and how it inspired her to create art.
13. `locomo_conv-48_q66` | bucket=`false_confident_answer` | mode=`temporal` | gold=`Brazil` | pred=`30 August`
   question: Which country was Jolene located in during the last week of August 2023?
   support_text: Jolene and Deborah spoke at 11:46 am on 30 August 2023.
14. `locomo_conv-26_q53` | bucket=`false_abstain` | mode=`current` | gold=`Oliver, Luna, Bailey` | pred=`ABSTAIN`
   question: What are Melanie's pets' names?
   support_text: Caroline showed interest in Melanie's paintings, and shared her own recent venture into abstract art.
15. `locomo_conv-42_q9` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`2019` | pred=`25 October 2022`
   question: When did Nate get his first two turtles?
   support_text: Nate loves seeing his turtles eat fruit as they get excited and it's cute.
16. `locomo_conv-41_q58` | bucket=`false_abstain` | mode=`current` | gold=`Toy drive, Community food drive, veterans, domestic violence` | pred=`ABSTAIN`
   question: What causes has John done events for?
   support_text: John has organized events for causes other than helping veterans, like raising awareness and funds for victims of domestic abuse.
17. `locomo_conv-48_q82` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`in October 2023` | pred=`17 September 2023`
   question: When did the Deboran and Jolene agree to go surfing?
   support_text: Jolene just started learning about surfing but hasn't gone yet.
18. `locomo_conv-41_q36` | bucket=`false_abstain` | mode=`current` | gold=`Oregon, Florida` | pred=`ABSTAIN`
   question: What states has Maria vacationed at?
   support_text: John excitedly told Maria about joining the fire-fighting brigade at 11:08 am on 16 August, 2023.
19. `locomo_conv-26_q4` | bucket=`temporal_selection_error` | mode=`current` | gold=`Adoption agencies` | pred=`and emotional preparation`
   question: What did Caroline research?
   support_text: Caroline gave Melanie some advice on how to get started with the adoption process, emphasizing the importance of research and emotional preparation.
20. `locomo_conv-42_q4` | bucket=`false_confident_answer` | mode=`temporal` | gold=`the week before 21Janury, 2022` | pred=`21 January 2022`
   question: When did Nate win his first video game tournament?
   support_text: Nate won his first video game tournament playing a team shooter game called Counter-Strike: Global Offensive.

## Explanation Audit

1. `locomo_conv-43_q2` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What are John's goals for his career that are not related to his basketball skills?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: basketball, career, goals, his.
   support_text: They discussed John's passion for basketball, his career goals, and plans for life after sports.
2. `locomo_conv-43_q14` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What does Tim do to escape reality?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: escape, reality, tim.
   support_text: Tim loves curling up with a good book to escape reality and get lost in different worlds during his free time.
3. `locomo_conv-48_q143` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How does Jolene describe the feeling of finding her snake snuggled under the bed after it got out?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: bed, got, her, jolene.
   support_text: Jolene had a 'snake adventure' where her snake got out, but she found her snuggling under the bed after hours of searching.
4. `locomo_conv-43_q66` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What fantasy movies does Tim like?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: fantasy, like, movies, tim.
   support_text: Tim enjoys escaping in books and movies, particularly fantasy genres like "The Lord of the Rings", and looks forward to a new TV series, "The Wheel of Time".
5. `locomo_conv-50_q179` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What emotion does Calvin mention feeling when he sees the relief of someone whose car he fixed?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: calvin, car, fixed.
   support_text: Calvin confirmed the car was fixed and going strong.
6. `locomo_conv-44_q49` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What does Andrew view his pets as?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: andrew, his, pets.
   support_text: Andrew considered getting another dog but decided to focus on his current pets first.
7. `locomo_conv-50_q30` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What are Dave's hobbies other than fixing cars?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: cars, dave's.
   support_text: Dave and Calvin, at 3:13 pm on 8 October, 2023, discussed Dave's passion for cars.
8. `locomo_conv-49_q12` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What health issue did Sam face that motivated him to change his lifestyle?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: health, his, motivated, sam.
   support_text: Sam feels a mix of emotions, somewhat concerned about his health but motivated to make positive changes.
9. `locomo_conv-26_q40` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: In what ways is Caroline participating in the LGBTQ community?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: caroline, community, lgbtq.
   support_text: Caroline expressed how excited she was about her work volunteering for the LGBTQ+ community and how it inspired her to create art.
10. `locomo_conv-26_q53` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What are Melanie's pets' names?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: melanie's.
   support_text: Caroline showed interest in Melanie's paintings, and shared her own recent venture into abstract art.
11. `locomo_conv-41_q58` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What causes has John done events for?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: causes, events, john.
   support_text: John has organized events for causes other than helping veterans, like raising awareness and funds for victims of domestic abuse.
12. `locomo_conv-41_q36` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What states has Maria vacationed at?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: maria.
   support_text: John excitedly told Maria about joining the fire-fighting brigade at 11:08 am on 16 August, 2023.


## Best Eval

# LoCoMo Reasoning Eval

- query_count: `32`
- joint_answer_or_abstain_acc: `0.1562`
- abstain_precision: `0.1579`
- abstain_recall: `1.0000`
- answer_match_rate: `0.0625`
- answer_evidence_recall: `0.4375`

## Objectives

| objective                      | value  |
| ------------------------------ | ------ |
| answerable_accuracy            | 0.0690 |
| abstain_precision              | 0.1579 |
| abstain_recall                 | 1.0000 |
| false_abstain_penalty          | 0.5517 |
| false_confident_answer_penalty | 0.3438 |

## Offline Slices

| slice    | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| -------- | ----- | --------- | ------------ | ------------ | ------------ |
| current  | 16    | 0.0625    | 0.8750       | 0.0000       | 0.4375       |
| temporal | 16    | 0.2500    | 0.3125       | 0.1250       | 0.4375       |

## Predicted Modes

| predicted_mode | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| -------------- | ----- | --------- | ------------ | ------------ | ------------ |
| current        | 21    | 0.1905    | 0.9048       | 0.0476       | 0.3810       |
| multi_hop      | 1     | 0.0000    | 0.0000       | 0.0000       | 1.0000       |
| temporal       | 10    | 0.1000    | 0.0000       | 0.1000       | 0.5000       |

## Failure Buckets

| bucket                   | count |
| ------------------------ | ----- |
| attribute_mismatch       | 10    |
| entity_mismatch          | 0     |
| explanation_quality      | 0     |
| false_abstain            | 16    |
| false_confident_answer   | 15    |
| missing_evidence         | 0     |
| multi_hop_failure        | 1     |
| temporal_selection_error | 6     |

## Failure Examples

1. `locomo_conv-42_q30` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`The weekend of 24June, 2022.` | pred=`24 June 2022`
   question: When is Joanna going to make Nate's ice cream for her family?
   support_text: Joanna appreciates Nate's support for her ice cream making.
2. `locomo_conv-43_q2` | bucket=`false_abstain` | mode=`current` | gold=`get endorsements, build his brand, do charity work` | pred=`ABSTAIN`
   question: What are John's goals for his career that are not related to his basketball skills?
   support_text: They discussed John's passion for basketball, his career goals, and plans for life after sports.
3. `locomo_conv-42_q55` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`The Sunday before 25October, 2022.` | pred=`25 October 2022`
   question: When was Joanna's second movie script shown on the big screens?
   support_text: Joanna recently had a movie script she contributed to shown on the big screen for the first time.
4. `locomo_conv-43_q14` | bucket=`false_abstain` | mode=`current` | gold=`Read fantasy books.` | pred=`ABSTAIN`
   question: What does Tim do to escape reality?
   support_text: Tim loves curling up with a good book to escape reality and get lost in different worlds during his free time.
5. `locomo_conv-48_q143` | bucket=`false_abstain` | mode=`current` | gold=`It really showed how much she loves her.` | pred=`ABSTAIN`
   question: How does Jolene describe the feeling of finding her snake snuggled under the bed after it got out?
   support_text: Jolene had a 'snake adventure' where her snake got out, but she found her snuggling under the bed after hours of searching.
6. `locomo_conv-43_q66` | bucket=`false_abstain` | mode=`current` | gold=`Lord of the Rings, Harry Potter, and Star Wars.` | pred=`ABSTAIN`
   question: What fantasy movies does Tim like?
   support_text: Tim enjoys escaping in books and movies, particularly fantasy genres like "The Lord of the Rings", and looks forward to a new TV series, "The Wheel of Time".
7. `locomo_conv-44_q49` | bucket=`false_abstain` | mode=`current` | gold=`Family` | pred=`ABSTAIN`
   question: What does Andrew view his pets as?
   support_text: Andrew considered getting another dog but decided to focus on his current pets first.
8. `locomo_conv-50_q26` | bucket=`false_confident_answer` | mode=`temporal` | gold=`on the Friday before 22 August, 2023` | pred=`18 August 2023`
   question: When did Dave host a card-playing night with his friends?
   support_text: Dave had a card night with friends last Friday and enjoyed it a lot.
9. `locomo_conv-50_q30` | bucket=`false_abstain` | mode=`current` | gold=`take a walk, go hiking, listen to favorite albums, live concerts, photography` | pred=`ABSTAIN`
   question: What are Dave's hobbies other than fixing cars?
   support_text: Dave and Calvin, at 3:13 pm on 8 October, 2023, discussed Dave's passion for cars.
10. `locomo_conv-50_q71` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`November 16, 2023` | pred=`17 November 2023`
   question: When did Calvin attend a gala in Boston?
   support_text: Calvin attended a fancy gala in Boston recently and met interesting people.
11. `locomo_conv-49_q12` | bucket=`false_abstain` | mode=`current` | gold=`Weight problem` | pred=`ABSTAIN`
   question: What health issue did Sam face that motivated him to change his lifestyle?
   support_text: Sam feels a mix of emotions, somewhat concerned about his health but motivated to make positive changes.
12. `locomo_conv-26_q40` | bucket=`false_abstain` | mode=`current` | gold=`Joining activist group, going to pride parades, participating in an art show, mentoring program` | pred=`ABSTAIN`
   question: In what ways is Caroline participating in the LGBTQ community?
   support_text: Caroline expressed how excited she was about her work volunteering for the LGBTQ+ community and how it inspired her to create art.
13. `locomo_conv-48_q66` | bucket=`false_confident_answer` | mode=`temporal` | gold=`Brazil` | pred=`30 August`
   question: Which country was Jolene located in during the last week of August 2023?
   support_text: Jolene and Deborah spoke at 11:46 am on 30 August 2023.
14. `locomo_conv-26_q53` | bucket=`false_abstain` | mode=`current` | gold=`Oliver, Luna, Bailey` | pred=`ABSTAIN`
   question: What are Melanie's pets' names?
   support_text: Caroline showed interest in Melanie's paintings, and shared her own recent venture into abstract art.
15. `locomo_conv-42_q9` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`2019` | pred=`25 October 2022`
   question: When did Nate get his first two turtles?
   support_text: Nate loves seeing his turtles eat fruit as they get excited and it's cute.
16. `locomo_conv-41_q58` | bucket=`false_abstain` | mode=`current` | gold=`Toy drive, Community food drive, veterans, domestic violence` | pred=`ABSTAIN`
   question: What causes has John done events for?
   support_text: John has organized events for causes other than helping veterans, like raising awareness and funds for victims of domestic abuse.
17. `locomo_conv-48_q82` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`in October 2023` | pred=`17 September 2023`
   question: When did the Deboran and Jolene agree to go surfing?
   support_text: Jolene just started learning about surfing but hasn't gone yet.
18. `locomo_conv-41_q36` | bucket=`false_abstain` | mode=`current` | gold=`Oregon, Florida` | pred=`ABSTAIN`
   question: What states has Maria vacationed at?
   support_text: John excitedly told Maria about joining the fire-fighting brigade at 11:08 am on 16 August, 2023.
19. `locomo_conv-26_q4` | bucket=`temporal_selection_error` | mode=`current` | gold=`Adoption agencies` | pred=`and emotional preparation`
   question: What did Caroline research?
   support_text: Caroline gave Melanie some advice on how to get started with the adoption process, emphasizing the importance of research and emotional preparation.
20. `locomo_conv-42_q4` | bucket=`false_confident_answer` | mode=`temporal` | gold=`the week before 21Janury, 2022` | pred=`21 January 2022`
   question: When did Nate win his first video game tournament?
   support_text: Nate won his first video game tournament playing a team shooter game called Counter-Strike: Global Offensive.

## Explanation Audit

1. `locomo_conv-43_q2` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What are John's goals for his career that are not related to his basketball skills?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: basketball, career, goals, his.
   support_text: They discussed John's passion for basketball, his career goals, and plans for life after sports.
2. `locomo_conv-43_q14` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What does Tim do to escape reality?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: escape, reality, tim.
   support_text: Tim loves curling up with a good book to escape reality and get lost in different worlds during his free time.
3. `locomo_conv-48_q143` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How does Jolene describe the feeling of finding her snake snuggled under the bed after it got out?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: bed, got, her, jolene.
   support_text: Jolene had a 'snake adventure' where her snake got out, but she found her snuggling under the bed after hours of searching.
4. `locomo_conv-43_q66` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What fantasy movies does Tim like?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: fantasy, like, movies, tim.
   support_text: Tim enjoys escaping in books and movies, particularly fantasy genres like "The Lord of the Rings", and looks forward to a new TV series, "The Wheel of Time".
5. `locomo_conv-50_q179` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What emotion does Calvin mention feeling when he sees the relief of someone whose car he fixed?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: calvin, car, fixed.
   support_text: Calvin confirmed the car was fixed and going strong.
6. `locomo_conv-44_q49` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What does Andrew view his pets as?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: andrew, his, pets.
   support_text: Andrew considered getting another dog but decided to focus on his current pets first.
7. `locomo_conv-50_q30` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What are Dave's hobbies other than fixing cars?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: cars, dave's.
   support_text: Dave and Calvin, at 3:13 pm on 8 October, 2023, discussed Dave's passion for cars.
8. `locomo_conv-49_q12` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What health issue did Sam face that motivated him to change his lifestyle?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: health, his, motivated, sam.
   support_text: Sam feels a mix of emotions, somewhat concerned about his health but motivated to make positive changes.
9. `locomo_conv-26_q40` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: In what ways is Caroline participating in the LGBTQ community?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: caroline, community, lgbtq.
   support_text: Caroline expressed how excited she was about her work volunteering for the LGBTQ+ community and how it inspired her to create art.
10. `locomo_conv-26_q53` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What are Melanie's pets' names?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: melanie's.
   support_text: Caroline showed interest in Melanie's paintings, and shared her own recent venture into abstract art.
11. `locomo_conv-41_q58` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What causes has John done events for?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: causes, events, john.
   support_text: John has organized events for causes other than helping veterans, like raising awareness and funds for victims of domestic abuse.
12. `locomo_conv-41_q36` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What states has Maria vacationed at?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: maria.
   support_text: John excitedly told Maria about joining the fire-fighting brigade at 11:08 am on 16 August, 2023.


## Holdout Comparison

| candidate         | current | temporal | multi_hop | answerable_acc | abstain_precision | abstain_recall |
| ----------------- | ------- | -------- | --------- | -------------- | ----------------- | -------------- |
| seed_holdout      | 0.0000  | 0.1250   | 0.0000    | 0.0000         | 0.1111            | 1.0000         |
| gepa_best_holdout | 0.0000  | 0.1250   | 0.0000    | 0.0000         | 0.1111            | 1.0000         |

