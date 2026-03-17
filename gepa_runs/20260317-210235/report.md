# GEPA Reasoning V1 Run

- run_id: `20260317-210235`
- track: `temporal_selection`
- budget: `96`
- holdout_budget: `48`
- active_components: `temporal_strategy, answer_style`
- engine_mode: `reflection_lm:openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- reflection_lm: `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- status: `benchmark_candidate`
- leakage_passed: `True`
- num_candidates: `n/a`
- total_metric_calls: `64`

## Candidate Comparison

| candidate | current | temporal | multi_hop | answerable_acc | abstain_precision | abstain_recall | false_abstain | false_confident |
| --------- | ------- | -------- | --------- | -------------- | ----------------- | -------------- | ------------- | --------------- |
| seed      | 0.0000  | 0.1875   | 0.0000    | 0.1348         | 0.1667            | 0.8571         | 0.3371        | 0.5000          |
| gepa_best | 0.0000  | 0.1875   | 0.0000    | 0.1348         | 0.1667            | 0.8571         | 0.3371        | 0.5000          |

Best candidate: `gepa_best`

## Seed Eval

# LoCoMo Reasoning Eval

- query_count: `96`
- joint_answer_or_abstain_acc: `0.1875`
- abstain_precision: `0.1667`
- abstain_recall: `0.8571`
- answer_match_rate: `0.1250`
- answer_evidence_recall: `0.3854`

## Objectives

| objective                         | value  |
| --------------------------------- | ------ |
| answerable_accuracy               | 0.1348 |
| abstain_precision                 | 0.1667 |
| abstain_recall                    | 0.8571 |
| false_abstain_penalty             | 0.3371 |
| false_confident_answer_penalty    | 0.5000 |
| joint_reward_mean                 | 0.1875 |
| answerable_reward_mean            | 0.1348 |
| temporal_joint_reward_mean        | 0.1875 |
| temporal_evidence_reward_mean     | 0.3854 |
| answer_evidence_reward_mean       | 0.3854 |
| avoid_false_abstain_mean          | 0.6629 |
| avoid_false_confident_answer_mean | 0.5000 |

## Offline Slices

| slice    | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| -------- | ----- | --------- | ------------ | ------------ | ------------ |
| temporal | 96    | 0.1875    | 0.3750       | 0.1250       | 0.3854       |

## Predicted Modes

| predicted_mode | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| -------------- | ----- | --------- | ------------ | ------------ | ------------ |
| current        | 37    | 0.1892    | 0.8649       | 0.0270       | 0.2703       |
| temporal       | 59    | 0.1864    | 0.0678       | 0.1864       | 0.4576       |

## Failure Buckets

| bucket                   | count |
| ------------------------ | ----- |
| attribute_mismatch       | 37    |
| entity_mismatch          | 0     |
| explanation_quality      | 1     |
| false_abstain            | 30    |
| false_confident_answer   | 65    |
| missing_evidence         | 0     |
| multi_hop_failure        | 0     |
| temporal_selection_error | 31    |

## Failure Examples

1. `locomo_conv-42_q30` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`The weekend of 24June, 2022.` | pred=`24 June 2022`
   question: When is Joanna going to make Nate's ice cream for her family?
   support_text: Joanna appreciates Nate's support for her ice cream making.
2. `locomo_conv-42_q55` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`The Sunday before 25October, 2022.` | pred=`25 October 2022`
   question: When was Joanna's second movie script shown on the big screens?
   support_text: Joanna recently had a movie script she contributed to shown on the big screen for the first time.
3. `locomo_conv-48_q143` | bucket=`false_abstain` | mode=`current` | gold=`It really showed how much she loves her.` | pred=`ABSTAIN`
   question: How does Jolene describe the feeling of finding her snake snuggled under the bed after it got out?
   support_text: Jolene had a 'snake adventure' where her snake got out, but she found her snuggling under the bed after hours of searching.
4. `locomo_conv-50_q179` | bucket=`temporal_selection_error` | mode=`current` | gold=`` | pred=`like it was well-deserved after hard work`
   question: What emotion does Calvin mention feeling when he sees the relief of someone whose car he fixed?
   support_text: Calvin shared a picture and mentioned it was his first luxury car, feeling like it was well-deserved after hard work.
5. `locomo_conv-50_q26` | bucket=`false_confident_answer` | mode=`temporal` | gold=`on the Friday before 22 August, 2023` | pred=`18 August 2023`
   question: When did Dave host a card-playing night with his friends?
   support_text: Dave had a card night with friends last Friday and enjoyed it a lot.
6. `locomo_conv-50_q71` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`November 16, 2023` | pred=`17 November 2023`
   question: When did Calvin attend a gala in Boston?
   support_text: Calvin attended a fancy gala in Boston recently and met interesting people.
7. `locomo_conv-48_q66` | bucket=`false_abstain` | mode=`current` | gold=`Brazil` | pred=`ABSTAIN`
   question: Which country was Jolene located in during the last week of August 2023?
   support_text: Jolene and Deborah spoke at 11:46 am on 30 August 2023.
8. `locomo_conv-42_q9` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`2019` | pred=`25 October 2022`
   question: When did Nate get his first two turtles?
   support_text: Nate loves seeing his turtles eat fruit as they get excited and it's cute.
9. `locomo_conv-48_q82` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`in October 2023` | pred=`17 September 2023`
   question: When did the Deboran and Jolene agree to go surfing?
   support_text: Jolene just started learning about surfing but hasn't gone yet.
10. `locomo_conv-42_q4` | bucket=`false_confident_answer` | mode=`temporal` | gold=`the week before 21Janury, 2022` | pred=`21 January 2022`
   question: When did Nate win his first video game tournament?
   support_text: Nate won his first video game tournament playing a team shooter game called Counter-Strike: Global Offensive.
11. `locomo_conv-49_q46` | bucket=`false_abstain` | mode=`current` | gold=`Banff, Rocky Mountains` | pred=`ABSTAIN`
   question: Which places in Canada was Evan visiting in July 2023?
   support_text: At 10:52 am on 27 July, 2023, Sam confided in Evan about being teased for his weight, prompting him to make health changes.
12. `locomo_conv-50_q124` | bucket=`false_abstain` | mode=`current` | gold=`Reminds himself of his passion for goals, gets help from others, and takes a break to recharge` | pred=`ABSTAIN`
   question: How does Calvin stay motivated when faced with setbacks?
   support_text: Calvin encouraged him not to give up and stay positive, sharing his approach to staying motivated during setbacks.
13. `locomo_conv-30_q34` | bucket=`false_confident_answer` | mode=`temporal` | gold=`20 June, 2023` | pred=`21 June 2023`
   question: When did Jon visit networking events for his store?
   support_text: Jon has been attending networking events to make things happen after losing his job.
14. `locomo_conv-26_q80` | bucket=`false_abstain` | mode=`temporal` | gold=`The Friday before 22 October 2023` | pred=`ABSTAIN`
   question: When did Caroline pass the adoption interview?
   support_text: Caroline tells Melanie that she passed the adoption agency interviews last Friday and is excited about the progress she's making towards her goal of having a family.
15. `locomo_conv-47_q54` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`six months` | pred=`1 September`
   question: How long did it take for James to complete his Witcher-inspired game?
   support_text: Summary: At 6:53 pm on 1 September 2022, James excitedly shared with John that he had completed his Unity strategy game, inspired by games like Civilization and Total War.
16. `locomo_conv-41_q134` | bucket=`false_abstain` | mode=`current` | gold=`Had a meaningful experience at a military memorial` | pred=`ABSTAIN`
   question: What did John do the week before August 3, 2023 involving his kids?
   support_text: At 5:19 pm on 5 August, 2023, John informed Maria that he lost his job at a mechanical engineering company that tanked.
17. `locomo_conv-49_q62` | bucket=`false_abstain` | mode=`current` | gold=`work-related stress` | pred=`ABSTAIN`
   question: What kind of stress was Sam dealing with in October 2023?
   support_text: Sam and Evan caught up at 1:50 pm on 17 October, 2023.
18. `locomo_conv-42_q65` | bucket=`false_confident_answer` | mode=`temporal` | gold=`The Saturday before 7November, 2022` | pred=`7 November 2022`
   question: When did Nate win a big Valorant tourney?
   support_text: Nate won a big Valorant tournament making him the champion.
19. `locomo_conv-26_q32` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`The week before 27 June 2023` | pred=`13 September 2023`
   question: When did Melanie go camping in June?
   support_text: Melanie enjoys camping with her kids, exploring the forest, and hiking.
20. `locomo_conv-44_q11` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`June, 2023` | pred=`3 May 2023`
   question: When did Audrey's positive reinforcement training course for dogs take place?
   support_text: Audrey's dogs love going on hikes and exploring nature, their happy place.

## Explanation Audit

1. `locomo_conv-48_q143` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How does Jolene describe the feeling of finding her snake snuggled under the bed after it got out?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: bed, got, her, jolene.
   support_text: Jolene had a 'snake adventure' where her snake got out, but she found her snuggling under the bed after hours of searching.
2. `locomo_conv-48_q66` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: Which country was Jolene located in during the last week of August 2023?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2023, august, jolene.
   support_text: Jolene and Deborah spoke at 11:46 am on 30 August 2023.
3. `locomo_conv-30_q105` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What plans does Gina have after receiving advice at the networking event?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: advice, event, gina, networking.
   support_text: Gina met some investors and got good advice at a recent networking event.
4. `locomo_conv-49_q46` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: Which places in Canada was Evan visiting in July 2023?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2023, evan, july.
   support_text: At 10:52 am on 27 July, 2023, Sam confided in Evan about being teased for his weight, prompting him to make health changes.
5. `locomo_conv-50_q124` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How does Calvin stay motivated when faced with setbacks?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: calvin, motivated, setbacks, stay.
   support_text: Calvin encouraged him not to give up and stay positive, sharing his approach to staying motivated during setbacks.
6. `locomo_conv-26_q80` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: When did Caroline pass the adoption interview?
   explanation: I abstained because the best evidence was not grounded enough to give a stable temporal answer. Top support: adoption, caroline.
   support_text: Caroline tells Melanie that she passed the adoption agency interviews last Friday and is excited about the progress she's making towards her goal of having a family.
7. `locomo_conv-41_q134` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What did John do the week before August 3, 2023 involving his kids?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2023, august, his, john.
   support_text: At 5:19 pm on 5 August, 2023, John informed Maria that he lost his job at a mechanical engineering company that tanked.
8. `locomo_conv-49_q62` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What kind of stress was Sam dealing with in October 2023?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2023, october, sam.
   support_text: Sam and Evan caught up at 1:50 pm on 17 October, 2023.
9. `locomo_conv-50_q167` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What sports activity is Dave planning to try after the tour with Frank Ocean?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: dave, frank, ocean, tour.
   support_text: Dave inquired about Calvin's tour with Frank Ocean, discussing the challenges of fame balancing personal life and work.
10. `locomo_conv-26_q194` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How did Caroline feel about her family after the accident?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: caroline, family, her.
   support_text: Caroline tells Melanie that she passed the adoption agency interviews last Friday and is excited about the progress she's making towards her goal of having a family.
11. `locomo_conv-50_q28` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: Which city was Calvin visiting in August 2023?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2023, august, calvin.
   support_text: Dave and Calvin caught up at 2:55 pm on 31 August, 2023.
12. `locomo_conv-42_q33` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: Where did Joanna travel to in July 2022?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2022, joanna.
   support_text: Joanna excitedly shared with Nate at 12:06 am on 11 November, 2022, that she was filming her own movie from a road-trip script.


## Best Eval

# LoCoMo Reasoning Eval

- query_count: `96`
- joint_answer_or_abstain_acc: `0.1875`
- abstain_precision: `0.1667`
- abstain_recall: `0.8571`
- answer_match_rate: `0.1250`
- answer_evidence_recall: `0.3854`

## Objectives

| objective                         | value  |
| --------------------------------- | ------ |
| answerable_accuracy               | 0.1348 |
| abstain_precision                 | 0.1667 |
| abstain_recall                    | 0.8571 |
| false_abstain_penalty             | 0.3371 |
| false_confident_answer_penalty    | 0.5000 |
| joint_reward_mean                 | 0.1875 |
| answerable_reward_mean            | 0.1348 |
| temporal_joint_reward_mean        | 0.1875 |
| temporal_evidence_reward_mean     | 0.3854 |
| answer_evidence_reward_mean       | 0.3854 |
| avoid_false_abstain_mean          | 0.6629 |
| avoid_false_confident_answer_mean | 0.5000 |

## Offline Slices

| slice    | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| -------- | ----- | --------- | ------------ | ------------ | ------------ |
| temporal | 96    | 0.1875    | 0.3750       | 0.1250       | 0.3854       |

## Predicted Modes

| predicted_mode | count | joint_acc | pred_abstain | answer_match | evidence_hit |
| -------------- | ----- | --------- | ------------ | ------------ | ------------ |
| current        | 37    | 0.1892    | 0.8649       | 0.0270       | 0.2703       |
| temporal       | 59    | 0.1864    | 0.0678       | 0.1864       | 0.4576       |

## Failure Buckets

| bucket                   | count |
| ------------------------ | ----- |
| attribute_mismatch       | 37    |
| entity_mismatch          | 0     |
| explanation_quality      | 1     |
| false_abstain            | 30    |
| false_confident_answer   | 65    |
| missing_evidence         | 0     |
| multi_hop_failure        | 0     |
| temporal_selection_error | 31    |

## Failure Examples

1. `locomo_conv-42_q30` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`The weekend of 24June, 2022.` | pred=`24 June 2022`
   question: When is Joanna going to make Nate's ice cream for her family?
   support_text: Joanna appreciates Nate's support for her ice cream making.
2. `locomo_conv-42_q55` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`The Sunday before 25October, 2022.` | pred=`25 October 2022`
   question: When was Joanna's second movie script shown on the big screens?
   support_text: Joanna recently had a movie script she contributed to shown on the big screen for the first time.
3. `locomo_conv-48_q143` | bucket=`false_abstain` | mode=`current` | gold=`It really showed how much she loves her.` | pred=`ABSTAIN`
   question: How does Jolene describe the feeling of finding her snake snuggled under the bed after it got out?
   support_text: Jolene had a 'snake adventure' where her snake got out, but she found her snuggling under the bed after hours of searching.
4. `locomo_conv-50_q179` | bucket=`temporal_selection_error` | mode=`current` | gold=`` | pred=`like it was well-deserved after hard work`
   question: What emotion does Calvin mention feeling when he sees the relief of someone whose car he fixed?
   support_text: Calvin shared a picture and mentioned it was his first luxury car, feeling like it was well-deserved after hard work.
5. `locomo_conv-50_q26` | bucket=`false_confident_answer` | mode=`temporal` | gold=`on the Friday before 22 August, 2023` | pred=`18 August 2023`
   question: When did Dave host a card-playing night with his friends?
   support_text: Dave had a card night with friends last Friday and enjoyed it a lot.
6. `locomo_conv-50_q71` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`November 16, 2023` | pred=`17 November 2023`
   question: When did Calvin attend a gala in Boston?
   support_text: Calvin attended a fancy gala in Boston recently and met interesting people.
7. `locomo_conv-48_q66` | bucket=`false_abstain` | mode=`current` | gold=`Brazil` | pred=`ABSTAIN`
   question: Which country was Jolene located in during the last week of August 2023?
   support_text: Jolene and Deborah spoke at 11:46 am on 30 August 2023.
8. `locomo_conv-42_q9` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`2019` | pred=`25 October 2022`
   question: When did Nate get his first two turtles?
   support_text: Nate loves seeing his turtles eat fruit as they get excited and it's cute.
9. `locomo_conv-48_q82` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`in October 2023` | pred=`17 September 2023`
   question: When did the Deboran and Jolene agree to go surfing?
   support_text: Jolene just started learning about surfing but hasn't gone yet.
10. `locomo_conv-42_q4` | bucket=`false_confident_answer` | mode=`temporal` | gold=`the week before 21Janury, 2022` | pred=`21 January 2022`
   question: When did Nate win his first video game tournament?
   support_text: Nate won his first video game tournament playing a team shooter game called Counter-Strike: Global Offensive.
11. `locomo_conv-49_q46` | bucket=`false_abstain` | mode=`current` | gold=`Banff, Rocky Mountains` | pred=`ABSTAIN`
   question: Which places in Canada was Evan visiting in July 2023?
   support_text: At 10:52 am on 27 July, 2023, Sam confided in Evan about being teased for his weight, prompting him to make health changes.
12. `locomo_conv-50_q124` | bucket=`false_abstain` | mode=`current` | gold=`Reminds himself of his passion for goals, gets help from others, and takes a break to recharge` | pred=`ABSTAIN`
   question: How does Calvin stay motivated when faced with setbacks?
   support_text: Calvin encouraged him not to give up and stay positive, sharing his approach to staying motivated during setbacks.
13. `locomo_conv-30_q34` | bucket=`false_confident_answer` | mode=`temporal` | gold=`20 June, 2023` | pred=`21 June 2023`
   question: When did Jon visit networking events for his store?
   support_text: Jon has been attending networking events to make things happen after losing his job.
14. `locomo_conv-26_q80` | bucket=`false_abstain` | mode=`temporal` | gold=`The Friday before 22 October 2023` | pred=`ABSTAIN`
   question: When did Caroline pass the adoption interview?
   support_text: Caroline tells Melanie that she passed the adoption agency interviews last Friday and is excited about the progress she's making towards her goal of having a family.
15. `locomo_conv-47_q54` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`six months` | pred=`1 September`
   question: How long did it take for James to complete his Witcher-inspired game?
   support_text: Summary: At 6:53 pm on 1 September 2022, James excitedly shared with John that he had completed his Unity strategy game, inspired by games like Civilization and Total War.
16. `locomo_conv-41_q134` | bucket=`false_abstain` | mode=`current` | gold=`Had a meaningful experience at a military memorial` | pred=`ABSTAIN`
   question: What did John do the week before August 3, 2023 involving his kids?
   support_text: At 5:19 pm on 5 August, 2023, John informed Maria that he lost his job at a mechanical engineering company that tanked.
17. `locomo_conv-49_q62` | bucket=`false_abstain` | mode=`current` | gold=`work-related stress` | pred=`ABSTAIN`
   question: What kind of stress was Sam dealing with in October 2023?
   support_text: Sam and Evan caught up at 1:50 pm on 17 October, 2023.
18. `locomo_conv-42_q65` | bucket=`false_confident_answer` | mode=`temporal` | gold=`The Saturday before 7November, 2022` | pred=`7 November 2022`
   question: When did Nate win a big Valorant tourney?
   support_text: Nate won a big Valorant tournament making him the champion.
19. `locomo_conv-26_q32` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`The week before 27 June 2023` | pred=`13 September 2023`
   question: When did Melanie go camping in June?
   support_text: Melanie enjoys camping with her kids, exploring the forest, and hiking.
20. `locomo_conv-44_q11` | bucket=`temporal_selection_error` | mode=`temporal` | gold=`June, 2023` | pred=`3 May 2023`
   question: When did Audrey's positive reinforcement training course for dogs take place?
   support_text: Audrey's dogs love going on hikes and exploring nature, their happy place.

## Explanation Audit

1. `locomo_conv-48_q143` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How does Jolene describe the feeling of finding her snake snuggled under the bed after it got out?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: bed, got, her, jolene.
   support_text: Jolene had a 'snake adventure' where her snake got out, but she found her snuggling under the bed after hours of searching.
2. `locomo_conv-48_q66` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: Which country was Jolene located in during the last week of August 2023?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2023, august, jolene.
   support_text: Jolene and Deborah spoke at 11:46 am on 30 August 2023.
3. `locomo_conv-30_q105` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What plans does Gina have after receiving advice at the networking event?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: advice, event, gina, networking.
   support_text: Gina met some investors and got good advice at a recent networking event.
4. `locomo_conv-49_q46` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: Which places in Canada was Evan visiting in July 2023?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2023, evan, july.
   support_text: At 10:52 am on 27 July, 2023, Sam confided in Evan about being teased for his weight, prompting him to make health changes.
5. `locomo_conv-50_q124` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How does Calvin stay motivated when faced with setbacks?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: calvin, motivated, setbacks, stay.
   support_text: Calvin encouraged him not to give up and stay positive, sharing his approach to staying motivated during setbacks.
6. `locomo_conv-26_q80` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: When did Caroline pass the adoption interview?
   explanation: I abstained because the best evidence was not grounded enough to give a stable temporal answer. Top support: adoption, caroline.
   support_text: Caroline tells Melanie that she passed the adoption agency interviews last Friday and is excited about the progress she's making towards her goal of having a family.
7. `locomo_conv-41_q134` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What did John do the week before August 3, 2023 involving his kids?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2023, august, his, john.
   support_text: At 5:19 pm on 5 August, 2023, John informed Maria that he lost his job at a mechanical engineering company that tanked.
8. `locomo_conv-49_q62` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What kind of stress was Sam dealing with in October 2023?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2023, october, sam.
   support_text: Sam and Evan caught up at 1:50 pm on 17 October, 2023.
9. `locomo_conv-50_q167` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: What sports activity is Dave planning to try after the tour with Frank Ocean?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: dave, frank, ocean, tour.
   support_text: Dave inquired about Calvin's tour with Frank Ocean, discussing the challenges of fame balancing personal life and work.
10. `locomo_conv-26_q194` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: How did Caroline feel about her family after the accident?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: caroline, family, her.
   support_text: Caroline tells Melanie that she passed the adoption agency interviews last Friday and is excited about the progress she's making towards her goal of having a family.
11. `locomo_conv-50_q28` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: Which city was Calvin visiting in August 2023?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2023, august, calvin.
   support_text: Dave and Calvin caught up at 2:55 pm on 31 August, 2023.
12. `locomo_conv-42_q33` | quality=`good` | abstain=`True` | pred=`ABSTAIN`
   question: Where did Joanna travel to in July 2022?
   explanation: I abstained because the best evidence was not grounded enough to give a stable current answer. Top support: 2022, joanna.
   support_text: Joanna excitedly shared with Nate at 12:06 am on 11 November, 2022, that she was filming her own movie from a road-trip script.


## Holdout Comparison

| candidate         | current | temporal | multi_hop | answerable_acc | abstain_precision | abstain_recall |
| ----------------- | ------- | -------- | --------- | -------------- | ----------------- | -------------- |
| seed_holdout      | 0.0000  | 0.2083   | 0.0000    | 0.1190         | 0.3333            | 0.8333         |
| gepa_best_holdout | 0.0000  | 0.2083   | 0.0000    | 0.1190         | 0.3333            | 0.8333         |

