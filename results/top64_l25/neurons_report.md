# What the top 64 neurons of the emotion writes do (layer 25)

Picked round-robin down each emotion's greedy order (the neurons that best keep the direction of the MLP's
emotion write). Per neuron: the emotions it was picked for (greedy rank, share of |delta_e| it carries along
delta_e); its activation shift dA over the 12 emotions (the top 3, in units of its largest |dA|); the share of
its p97+ hits from TinyStories / FineWeb-edu / UltraChat; Haiku's one-line summary; its modes (share of hits,
emotion label).

## 2862: happy #1 (+0.093)
- dA: happy +1.00, loving +0.27, nervous -0.25 | hits TS/FW/UC: 58% / 18% / 24%
- Fires on timing and connective words in recipes ('until', 'for', 'then') and on words after 'fun' or 'laughing' in children's stories; the fun and laughter modes are happy-related, the recipe and other modes are not.
  - 24% [none] 'of', 'by', 'people', 'irus' in public-health and medical prose: 'impacts of a policy'
  - 14% [happy] pronoun right after a happy or funny sentence: 'Ellie was so happy. [She] smiled'
  - 9% [happy] 'laughed' right after a joke or funny scene: 'the orange cat laughed', 'they all laughed'
  - 6% [none] 'until', 'for', 'then' linking cooking times and steps: 'simmer for', 'sauté until browned'
  - 4% [none] period ending short children's story sentences about play, swallowing water, or poems
  - 4% [emotion-general] 'was' before a feeling adjective: 'Emma was very happy', 'He was so proud', 'Jack was so excited'
  - 2% [happy] 'until', 'they', 'stomachs' right after 'laughing' in 'laughing until they cried'
  - 2% [happy] 'had', 'having', 'much' right after 'fun' in play scenes: 'had lots of fun'
  - 36% unclustered: 'had', 'so', 'begins' in assorted story and recipe sentences; no shared trigger

## 9964: inspired #1 (+0.010), loving #3 (+0.016), calm #1 (+0.012), sad #1 (+0.013), surprised #6 (+0.006)
- dA: loving -1.00, calm +0.77, surprised -0.72 | hits TS/FW/UC: 19% / 52% / 29%
- Fires on hyphens in ISBN numbers, digits in IP addresses, and the final word of some positive story phrases such as 'smiled with pride'; not emotion-specific.
  - 96% [none] last word of a positive phrase: 'smiled with pride', 'softly and kindly', 'explore this magical place'
  - 3% [none] hyphen inside ISBN numbers, e.g. '0-231-10936-9' in book citations
  - 1% unclustered: digit '6' in IP address 220.127.116.11 and 'uta' in forum-post text

## 10272: happy #2 (+0.074), loving #1 (+0.105), proud #3 (+0.018), desperate #2 (+0.014), angry #4 (+0.018), afraid #4 (+0.015), nervous #4 (+0.018)
- dA: loving -1.00, happy -0.68, nervous +0.38 | hits TS/FW/UC: 53% / 19% / 28%
- Fires on clause-level tokens in children's stories and, most strongly, inside love-poem lines ('I love thee', 'my darling'); only the love-poem mode is emotion-related.
  - 95% [none] 'was', 'Ben', 'Lily', 'and', 'Jack' starting clauses in playground or animal scenes
  - 2% [loving] 'i', ')', '.' inside Poe-style love-poem lines: 'I love thee', 'my darling'
  - 2% unclustered: '€' mojibake after a cloudy-sky line; 'into' in Bible text; 's' in 'mom's makeup'

## 2005: proud #1 (+0.020)
- dA: proud +1.00, happy +0.73, afraid -0.50 | hits TS/FW/UC: 55% / 16% / 29%
- Fires on the word after or before a character's emotional state in children's stories, most strongly on 'happy', on 'was/felt/so' lead-ins, and on gratitude and warmth passages, so it is emotion-related alongside non-emotion modes such as punctuation and names.
  - 7% [none] 'far', 'lot', 'relying', 'can', 'as' in policy and business sentences about costs and effects
  - 7% [happy] 'happy' right after 'was so' or 'were so' describing a character's joy
  - 5% [none] character name or pronoun starting a new sentence after an event: 'Lily', 'Mark', 'David', 'Jenna's', 'Baby'
  - 5% [emotion-general] 'was' after a character name in happy or excited scenes: 'Fido [was] so happy', 'Ted [was] very excited'
  - 5% [none] object after 'thanked' or 'hugged': 'thanked the [rug]', 'hugged [her]', 'hugged him [back]'
  - 4% [none] period ending a sentence of encouragement or invitation: 'cheering him on[.]', 'play with them[.]'
  - 3% [emotion-general] 'very', 'so', 'felt' before a feeling word: 'felt [very] relaxed', 'Timmy [felt] excited', 'was [so] thankful'
  - 3% [proud] 's, is, a, an' after 'it' or 'this' calling an honor or award: 'it's a wonderful bridge', 'is an amazing feeling'
  - 3% [emotion-general] 'grateful', 'lucky', 'blessed' right after 'I felt' or 'feeling' in thankful passages
  - 2% [emotion-general] comma ending a gratitude, peace or cherished-memory sentence: 'grateful for this moment[,]', 'precious[,]'
  - 2% [emotion-general] 'and' after a character's happy or relieved reaction: 'smiled [and] said', 'so happy [and]', 'relieved [and]'
  - 2% [none] sentence-ending punctuation after dialogue or a name: 'Tom[.]', 'food[.]', 'Okay[,]', 'we do[!]'
  - 2% [none] 'said' after 'smiled and' or 'laughed and' in dialogue: 'Daisy smiled so big and [said]'
  - 2% [none] 'the', 'little', 'life', 'always' in phrases about appreciating life and the passage of time
  - 48% unclustered: 'like' after 'excited and felt', 'his' in a stuck-foot scene, hyphen in 'game-ch'

## 8776: desperate #1 (+0.044)
- dA: desperate +1.00, proud -0.21, happy -0.20 | hits TS/FW/UC: 63% / 11% / 26%
- Fires on the search-for-something-lost sequence in children's stories (looked, couldn't find, subject pronoun, comma) and, once, on 'so sad' or 'so happy' after the loss; mainly a lost-and-found narrative neuron.
  - 69% [none] Subject pronoun starting the sentence after a failed search: 'Tom searched... [He] felt sad', 'Lily could not find Fluffy. [She] looked'
  - 5% [none] Period ending a sentence after a failed search or confusion: 'couldn't see it anywhere[.]', 'couldn't find it[.]', 'couldn't figure out what it was[.]'
  - 5% [none] 'looked' after 'Lily/She' when a pet or toy goes missing: 'Lily [looked] for him everywhere', 'She [looked] all around'
  - 5% [none] Comma after a failed search phrase: 'looked all around the park[,] but', 'looked and looked[,] but Fluffy was gone'
  - 2% [none] Step number or period in numbered recipe instructions: '7[.] Let the brownies cool', '25-30 [minutes]', '(74°C[)]'
  - 2% [emotion-general] 'was', 'so', or 'very' right before a feeling word (sad, happy) after losing or finding something: 'Lily was [so] sad'
  - 11% unclustered: Stray tokens: 'the' in 'Where is [the] treasure?', 'must' in NFPA safety text, '.' after 'grabbed his hand'

## 3582: calm #8 (+0.007), angry #1 (+0.031)
- dA: angry +1.00, afraid +0.51, calm -0.39 | hits TS/FW/UC: 30% / 36% / 35%
- Mostly non-emotional: word fragments, opinion-essay punctuation and sarcastic 'Great,' or 'Yeah,' lines; only the stage-direction tone-cue mode (angrily, smirking, mockingly) is emotion-related.
  - 32% [none] 'they', 'and', 'she' after a warm moment in children's stories: 'Everyone was so happy for her [and]'
  - 20% [none] 'earlier', 'arrests', punctuation in opinion and essay sentences about politics and policy
  - 12% [none] word fragments mid-word: 'flamewar[-sp]arking', 'ta[unting]', 'spec[ulating]', 'melod[ramatic]'
  - 4% [angry] tone-cue stage directions in a confrontation: 'OUTLAW LEADER (smirking[)]', '(mockingly[)]', '(angrily)'
  - 2% [none] 'Great[,]' and 'Yeah[,]' in sarcastic lines, before an ironic complaint about a bad outcome
  - 30% unclustered: 'is', ',', 'a' in travel and animal fact sentences about museums, aquariums and fangs

## 48: happy #7 (+0.012), guilty #1 (+0.044)
- dA: guilty +1.00, happy -0.61, inspired -0.59 | hits TS/FW/UC: 45% / 26% / 29%
- Fires on the word right after a child's feeling adjective ('felt bad and') and on gaze verbs in tense scenes; emotion-related mainly through the 'and' cluster.
  - 96% [emotion-general] 'and' after a child's feeling word: 'felt bad and', 'ashamed and', 'scared and'
  - 3% [none] DUPLICATE: 'looks' after a character's name in a tense scene (Jonathan looks at him)
  - 1% unclustered: misc.: 'Climate' in 'Exxon's Climate Footprint'; 'at' after 'looked'; commas in German name lists

## 11109: angry #5 (+0.017), sad #2 (+0.020), afraid #1 (+0.046), nervous #3 (+0.027)
- dA: afraid +1.00, nervous +0.74, angry +0.56 | hits TS/FW/UC: 31% / 25% / 44%
- Fires on the word after 'felt' or 'was' in fear, embarrassment or anxiety contexts, plus a weak story-opener mode; partly emotion-related.
  - 98% [emotion-general] token after 'felt' or 'was' naming fear or anxiety: 'felt [scared]', 'Tip was [scared]', 'stress [can]'
  - 2% [none] 'there' in the story opener 'Once upon a time, [there] was a little girl'
  - 0% unclustered: -

## 11976: happy #5 (+0.009), calm #4 (+0.010), afraid #6 (+0.018), nervous #1 (+0.044)
- dA: nervous +1.00, afraid +0.60, inspired -0.45 | hits TS/FW/UC: 68% / 6% / 26%
- Fires on the word before a feeling or on the feeling word itself in children's story scenes ('felt', 'was so', period after a loss), strongest on sad, ashamed and scared contexts; clearly emotion-related.
  - 20% [none] literary-review nouns: 'tracking [shots]', 'final [choice]', 'synth-heavy [soundtrack]', 'the [rush]' of power
  - 15% [emotion-general] period ending a loss or mistake sentence, right before 'He felt sad' or 'Timmy was very sad'
  - 11% [sad] name or pronoun starting the sentence after a dropped, broken or lost object: 'L[ily] was very sad'
  - 6% [happy] 'happy' or 'excited' right after 'was so' or 'was very': 'Timmy was so [happy]', 'Maya was so [excited]'
  - 3% [none] concrete place or object nouns after motion verbs: 'branch', 'bottom' of the hill, 'walked [home]'
  - 3% [emotion-general] 'was' after a character subject, right before 'so sad', 'so happy' or 'very ashamed'
  - 3% [emotion-general] 'felt' after a character subject, right before 'sad', 'scared' or 'ashamed'
  - 2% [none] sentence-opening 'The' after an event: '[The] bunny was so happy', '[The] little girl thanked the policeman'
  - 38% unclustered: mixed: 'still' in 'but still tired', the closing mark in 'are you okay?', 'made me very [worried]'

## 4311: sad #4 (+0.009), surprised #1 (+0.040)
- dA: surprised +1.00, nervous -0.38, desperate -0.35 | hits TS/FW/UC: 84% / 3% / 12%
- Fires on connective or gaze tokens (then, and, looked) that follow a character's reaction, often emotional, but the feeling sits in the context rather than the token, so it is not emotion-specific.
  - 88% [emotion-general] 'then' or 'and' after a character's reaction: 'Lily thought about it and [then] decided', 'Max felt bad [and] didn't know'
  - 4% [none] 'looked' after a character's reaction: 'Spot was very sad. He [looked] up', 'Lily [looked] up at her mom again'
  - 3% [none] sentence-final period after a citation or sentence, right before a new section: '2007[. ]Immunization', 'in fours[. ]C. DECUMBENS'
  - 6% unclustered: scattered tokens: 'two' in 'When the [two] of them', comma in 'asked him[,]', 'turns' in 'She [turns] to Lily'

## 128: inspired #2 (+0.002)
- dA: angry +1.00, calm -0.94, guilty +0.78 | hits TS/FW/UC: 17% / 52% / 32%
- Low-activation function-token neuron (periods, 'the', digits in citations); only the 'so' before feeling adjectives cluster is emotion-related.
  - 44% [emotion-general] 'so' after 'was' before a feeling adjective: 'was so proud', 'so upset', 'so excited'
  - 29% [none] a digit inside citation numbers or years: 'Printed in 20[0]8', 'PMC272[9]227'
  - 3% [none] 'the' after prepositions or verbs: 'into the bottom', 'to the blender', 'of the griddle'
  - 2% [none] sentence-final period closing a story sentence, e.g. 'never looked back.' or 'alive and free.'
  - 22% unclustered: mid-sentence 'and', 'the', and list commas in general prose with no feeling words

## 5663: loving #2 (+0.059)
- dA: loving +1.00, angry -0.34, surprised -0.25 | hits TS/FW/UC: 54% / 18% / 28%
- Fires on a mix of tokens: tender speech verbs (whispered, said), the subject after a parent's hug, 'was' before a feeling, and giggle/smile; part of it is affection or happiness, but many clusters (recipes, encyclopedic text, punctuation) are not emotion-related.
  - 19% [none] 'of', 'French', 'Fij[ian]', 'for' mid-phrase in encyclopedic or history text
  - 14% [happy] 'giggle', 'smiles', 'face' or 'hug' after a happy or playful moment: 'She [giggled]', 'Mia stops crying and [smiles]'
  - 13% [loving] subject 'She'/'He' or 'Mommy' after a parent hugs a child: 'Mom hugged him tightly. [She] said', 'Ben gave Lily a hug. [He] said'
  - 3% [none] sentence-final period and double space in short simple stories: 'it was too big[.  ]Remy asked'
  - 3% [emotion-general] 'was' after a character name, right before a feeling: 'Fuzzy [was] so happy', 'He [was] so happy', 'She [was] very sad'
  - 2% [loving] 'whispered' after a tender or comforting beat: 'She [whispered], I love you so', 'Sweet dreams, my little space...' [whispered]
  - 2% [none] 'said' or 'says' after an apology or forgiveness: 'Of course, I forgive you,' she [said]
  - 2% [none] generic nouns and verbs in advice text: 'employees' [wellbeing]', 'Statistics [help] add credibility', 'as a [person]'
  - 2% [none] cooking-step tokens: 'pan' in 'frying [pan]', period in '1[.]', 'stick' in 'non-[stick]', 'beans' in 'green [beans]'
  - 2% [none] sentence-final punctuation before 'user:' in chat transcripts or articles: 'over the years[.  ]user:'
  - 2% [none] 'told' after 'his mom' or 'The bus driver also': 'his mom [told] him that he needed to study'
  - 36% unclustered: scattered tokens: 'and' after 'It was Wednesday,', 'Tim[my]' split name, 'NS' in a DOI citation

## 3068: proud #2 (+0.017)
- dA: proud +1.00, sad -0.24, happy +0.22 | hits TS/FW/UC: 48% / 27% / 25%
- Fires on 'proud' itself in children's stories and on 'sense' before 'of pride' or 'of community' in festival essays; only the 'proud' mode is clearly emotion-related.
  - 98% [proud] 'proud' itself after 'was very' or 'was so', describing a child's pride in their own success
  - 2% [none] 'sense' before 'of pride' or 'of community' in festival and holiday essays
  - 0% unclustered: 'that' in legal text: 'deemed to be void on the ground [that] it takes away'

## 3634: calm #2 (+0.010), guilty #2 (+0.024)
- dA: guilty +1.00, surprised -0.79, calm -0.59 | hits TS/FW/UC: 83% / 6% / 11%
- Fires on pronouns right after a child's fear, hurt or anger in children's stories, and on 'didn't' when a character hesitates; the pronoun mode is emotion-related, the 'didn't' mode is not.
  - 91% [emotion-general] pronoun or 'was' right after a child's fear, hurt or anger: 'scared and [she] ran', 'Sara got [angry]'
  - 2% [none] 't' in 'didn't' when a character hesitates or fails: 'he didn['t] know how', 'didn['t] want to use the brush'
  - 7% unclustered: 'you', 'got' in dialogue and general text; 'got' comes right before 'angry' in one snippet

## 2066: angry #2 (+0.025)
- dA: angry -1.00, desperate -0.35, nervous +0.31 | hits TS/FW/UC: 57% / 13% / 31%
- Fires on dialogue tags, emotion-state lead-ins ('was', 'felt', 'so') and feeling adjectives ('excited', 'scared', 'sad') in children's stories, so it is emotion-related, alongside non-emotion modes for refusals, citations and code.
  - 9% [none] character name or pronoun starting a sentence after a happy event: 'Lily was amazed', 'Anna smiled', 'Gina smiled'
  - 7% [emotion-general] feeling adjective after 'was so' or 'felt very': 'excited', 'scared', 'sad', 'overjoyed'
  - 6% [none] digits and symbols inside citation or date strings: '261-26[2].', '15-[7].', '190[5]', '195[8]'
  - 4% [none] physical comfort after a child cries or is scared: 'picked him [up]', 'held it [tight]', 'hugged [tightly]'
  - 4% [emotion-general] 'was', 'felt', 'so', 'very' before a feeling word: 'Sara [was] curious', 'Timmy [felt] sad', 'felt [so] miserable'
  - 4% [none] 'to', 'not', 'want' in refusals: 'didn't want [to]', 'Sam did [not] listen', 'do not want [to] share'
  - 3% [none] 'said' as the speech tag after a speaker: 'the librarian [said]', 'he [said]', 'the creature [said]'
  - 3% [none] code and prompt tokens: 'readLine[().]', 'input[String]', 'replace[:]', 'read[Line]', 'commas[):]'
  - 60% unclustered: 'clothes' in a conformity description, 'he' after a quote about genetic research, 'your' in a story prompt

## 7527: afraid #2 (+0.024), nervous #2 (+0.031)
- dA: nervous +1.00, afraid +0.91, inspired -0.85 | hits TS/FW/UC: 66% / 6% / 27%
- Fires on intensifiers, negations and story tokens; several modes sit on fear, sadness or pride words, but it also covers citations, so it is only partly emotion-related.
  - 19% [afraid] 'scared' (also 'sad') right after 'was', describing a character's feeling
  - 8% [emotion-general] 'She' or 'He' starting a sentence right after an excited, amazed or proud sentence
  - 7% [afraid] 't' or 'not' in 'didn't know' or 'did not want' while a character is scared or tense
  - 7% [emotion-general] 'to' after 'started' before cry, frown, walk or feel tired
  - 6% [emotion-general] period right after feeling words: 'feeling guilty and sad.', 'feeling very scared.', 'felt happy.'
  - 4% [none] digits, periods or 'Jul' inside bibliography and citation text
  - 3% [none] opening quote mark starting dialogue after narration, e.g. 'Hi Timmy!' or 'Oh Sweetie'
  - 2% [emotion-general] 'so' or 'very' right before a feeling or pretty adjective: 'was so proud', 'were so happy', 'was very sad'
  - 2% [none] 'wanted' after a character's fear or frustration: 'Jane wanted to escape', 'wanted to cry'
  - 43% unclustered: '.' after 'said Sarah excitedly', 'It' after a soft noise, 'were' in 'They were playing jazz'

## 12216: surprised #2 (+0.023)
- dA: surprised -1.00, calm +0.42, desperate +0.39 | hits TS/FW/UC: 38% / 28% / 33%
- Fires on the function word or pronoun in a clause after a comparison or action, in narrative and how-to text; not emotion-related.
  - 92% [none] 'and' or 'she' right after a character's action, before 'smiled' or 'spun around'
  - 2% [none] pronoun after 'than' or 'as' in comparisons: 'than [he] hoped', 'as [she] thought', 'data [you] need'
  - 6% unclustered: mixed: 'I' starting a reply, colon in 'RxList[:]', period in numbered steps like 'Compare the first two elements... 3[.]'

## 7717: happy #3 (+0.023)
- dA: happy -1.00, loving -0.47, angry +0.42 | hits TS/FW/UC: 49% / 14% / 37%
- Fires on modifier and verb tokens across lists, news and stories; only the 'proud' mode is clearly emotion-related.
  - 31% [proud] 'proud' right after 'felt', 'was so', 'were very' describing a character
  - 25% [none] 'a', 'some', 'for' before a flavour adjective in baked potato recipe toppings
  - 3% [none] adjective after 'a' or 'was' describing an object: thick, heavy, small, silver, soft
  - 2% [none] 'knew' or 'told' right after a character's kind or happy action: 'Max knew', 'Lily knew'
  - 2% [none] 'exposure', 'reach', 'faced', 'social' in child-education and programme text
  - 36% unclustered: 'still' after 'you can', 'secured' in news, 'keep' in 'to keep the story engaging'

## 9904: inspired #3 (+0.001)
- dA: afraid +1.00, inspired -0.72, guilty -0.62 | hits TS/FW/UC: 42% / 43% / 15%
- Fires on sentence endings and feeling adjectives after joyful reactions in short stories, and on 'I am' self-description in one Socrates passage; only the happy-reaction mode is emotion-related.
  - 95% [happy] period or 'excited'/'happy' after a joyful reaction: 'started to laugh.', 'overjoyed.', 'was so happy'
  - 3% [none] DUPLICATE: 'am', 'I', 'not' in Socrates' gadfly speech: 'I am the gadfly', 'I am not going'
  - 2% unclustered: 'such' and 'Law' in legal or religious text; 'fast' in Socrates' gadfly speech

## 8502: calm #3 (+0.011)
- dA: calm -1.00, surprised +0.65, angry +0.61 | hits TS/FW/UC: 50% / 35% / 15%
- Mixed neuron with several function-word and transition modes; only the cluster on 'happy' or 'excited' after 'was very/so' is emotion-related.
  - 25% [none] 'be' or 'will' after 'you'll' in advice about being judged: 'you'll [be] pushed', 'people [will] tell you to calm down'
  - 15% [none] Period or exclamation after a sentence about surprise or excitement: 'couldn't believe it[.]', 'so surprised ... shoulder[!]'
  - 4% [none] Mid-word or mid-number tokens in academic and news text: 'Earth [sh]attering', '5[1]6-652', 'Center [for]', 'the conceived in [‘]racial'
  - 4% [happy] 'happy' or 'excited' after 'was very' or 'was so': 'George was very [happy]', 'Tim was very [happy]', 'was so [excited]'
  - 3% [none] 'At' opening a sentence as a time phrase: 'At [first]' or 'At [last]', 'challenging at [first]', 'at [once]'
  - 3% [none] Subject or conjunction in a character's reply: 'Tim[my] was so excited', 'The little [girl] replied', 'He [was]', 'grinned [and] replied'
  - 3% [none] Repeated comparative or adverb in 'and [higher]', 'and [closer]', 'and [louder]', 'crawling and [crawling]'
  - 43% unclustered: Stray tokens: 'he' in 'lift anything [he]', 'Lily' with 'So[,]' before a jump, '-' in 'self-sustaining[-]'

## 3282: desperate #3 (+0.011)
- dA: desperate +1.00, proud -0.26, happy -0.25 | hits TS/FW/UC: 77% / 10% / 13%
- Mostly narrative tokens in loss and search stories ('looked', 'She', 'mom'); the sad-period and upset modes are emotion-related, the rest are not.
  - 38% [emotion-general] 'upset', 'sad', 'might' after 'was very' or 'worried that she' in distressed scenes
  - 19% [none] 'She' starting a search after a loss: 'the rock was gone! She realized', 'She looked everywhere'
  - 6% [sad] period ending sad or lost sentences: 'couldn't find her shoe.', 'So Jenny was very sad.'
  - 5% [none] 'looked' after something goes missing: 'Lily looked everywhere' for a lost dog, toy or shoe
  - 2% [none] action verbs 'walked', 'ran', 'began', 'started' right after a character subject
  - 2% [none] 'came' or 'comes' after a family member or boy enters the scene: 'Lily's dad came'
  - 2% [none] 'mom' or 'mum' after a child's action, as a parent reacts: 'Billy's mom said'
  - 25% unclustered: 'snowman', 'made', 'January' in assorted story text and a date-range sentence

## 155: angry #3 (+0.024)
- dA: angry -1.00, afraid -0.60, loving +0.47 | hits TS/FW/UC: 27% / 36% / 36%
- Low-activation token neuron on possessives, benefit verbs in health advice, commas and recipe words; not emotion-related.
  - 34% [none] comma closing a clause or quote before a follow-up: 'too loud[,] so that', 'honey[,] Bob'
  - 21% [none] first word of a recipe item name: 'Sweet Potato [Black] Bean', 'sweetened [with] maple syrup'
  - 17% [none] benefit verb in health advice: 'help', 'can', 'support', 'exercise' ('can help reduce inflammation')
  - 2% [none] possessive 'his'/'their' after a verb or preposition: 'with his twinkly eyes', 'used his toy'
  - 25% unclustered: misc. function words: 'the colour', 'adult horse', 'data [in] a database'

## 11631: guilty #3 (+0.021)
- dA: guilty +1.00, loving -0.76, happy -0.74 | hits TS/FW/UC: 60% / 15% / 25%
- Fires on gaze verbs, sentence-start pronouns and sentence-ending periods in narrated exploration scenes; not emotion-related.
  - 43% [none] gaze and face nouns: 'their [eyes] fixed', 'looked at each [other]', 'many sad [faces]'
  - 23% [none] sentence-opening pronoun or name after a stated fact: '[He] decided to jump', '[She] realized'
  - 4% [none] period ending an action sentence: 'looked around[.]', 'examined the chicken up close[.]'
  - 2% [none] 'looked' right after a character subject: 'John [looked] up', 'She [looked] around'
  - 29% unclustered: mixed: period after 'said Jack', 'because it [was] empty', 'After' a few minutes

## 10958: sad #3 (+0.014)
- dA: sad +1.00, loving +0.86, inspired -0.80 | hits TS/FW/UC: 35% / 32% / 33%
- Fires on function words and punctuation inside ordinary narrative, recipe and report sentences, with no consistent content; not emotion-related.
  - 88% [none] function word mid story sentence: 'rubbed them on [each]', 'they [did] not know how', 'friends [wouldn]'t'
  - 2% [none] punctuation or spacing at recipe and list step boundaries: '5[.] Season', 'salsa[,]', 'Bake for [ ]25 minutes'
  - 10% unclustered: mixed: comma after 'fruit', 'glaciers[ have]' in a report, 'including [psychological]' distress

## 6766: afraid #3 (+0.015), nervous #6 (+0.010)
- dA: afraid -1.00, nervous -0.74, inspired +0.59 | hits TS/FW/UC: 35% / 32% / 32%
- Fires on clause-final words in children's stories, exercise instructions and CSS 'background-color'; not emotion-related.
  - 94% [none] clause-final word in children's stories: 'water', 'fast', 'quiet', 'feeling', 'and'
  - 3% [none] period or 'position' ending an exercise instruction step (plank, child's pose, shrug)
  - 2% unclustered: 'background' before '-color' in CSS hover and nth-child rules

## 9184: surprised #3 (+0.013)
- dA: surprised -1.00, inspired +0.53, angry -0.45 | hits TS/FW/UC: 45% / 30% / 25%
- Fires on sentence-level function tokens in children's stories (character names, pronouns, 'to' after speech verbs, sentence-end periods) rather than on feeling words; mostly not emotion-related, though two modes sit in happy or feeling-laden scenes.
  - 41% [none] noun ending a descriptive phrase: 'eyes', 'containers', 'views', 'success', 'summer'
  - 20% [emotion-general] period ending a sentence that names a feeling: 'brighter.', 'put it on.', 'was sad.'
  - 5% [happy] sentence-initial 'The' starting a sentence right after a happy or proud scene
  - 2% [none] sentence-initial character name (Sally, Lila, Lily) starting a new sentence after dialogue or narration
  - 2% [none] 'They' or 'sisters' starting a sentence after a group fun or outing scene
  - 2% [none] 'to' or 'told' after speech verbs: 'said to', 'says to', 'told the other animals'
  - 2% [none] sentence-initial 'Her' before mom, dad, mum or uncle in family scenes
  - 25% unclustered: clause-starting 'whose', 'says', or 'It' in mixed narration, code, and dialogue text

## 3832: happy #4 (+0.014)
- dA: happy -1.00, nervous +0.22, sad +0.22 | hits TS/FW/UC: 35% / 34% / 31%
- Fires on 'sense' in 'a sense of ...' and on 'tried/trying' when characters keep their composure, with some fear, relief or disappointment in the context; the firing token is rarely the feeling itself, so it is only loosely emotion-related.
  - 87% [emotion-general] token after a character's feeling or relief: 'Mum [smiled] as she walked', 'smiled with relief [and]', 'happy [and] friendly'
  - 5% [none] 'sense' in 'a sense of' plus a quality: 'a [sense] of wonder', 'a [sense] of fulfillment', 'a [sense] of self'
  - 2% [emotion-general] 'tried'/'trying' as a character keeps composure despite worry: 'he [tried] to remain brave', 'Maggie asked, [trying] to keep the conversation light'
  - 6% unclustered: scattered tokens: 'comes' in 'cool air [comes] in', comma in 'Once upon a time[,]'

## 6512: inspired #4 (+0.001)
- dA: angry +1.00, inspired -0.75, calm -0.59 | hits TS/FW/UC: 21% / 53% / 26%
- Fires on 'was' in fairy-tale openings and on scattered word pieces and code or poem tokens, with no emotion pattern.
  - 94% [none] word-piece endings in news or history text: 'arrest[ees]', 'Burn[side]', 'running [theme]'
  - 4% [none] DUPLICATE: 'was' in 'Once upon a time, there [was] a boy named Timmy' fairy-tale openings
  - 3% unclustered: scattered tokens: '8' in hex colour '#349[8]db', '[€]' in a story, 'promises' in a poem

## 3563: loving #4 (+0.015)
- dA: loving +1.00, surprised -0.25, angry -0.23 | hits TS/FW/UC: 68% / 12% / 21%
- Fires on 'went' or 'came' when a character notices or responds to a sad or crying person, and on joy and laughter words; the sad, distress and joy modes are emotion-related, the recipe, code and other modes are not.
  - 22% [none] 'She' or 'mom' after a mother's action: 'His mom saw... [She] told the boy to step aside'
  - 16% [happy] 'joy', 'laughed', 'smiled', 'fun' after warm family or friend moments
  - 4% [none] code and markup tokens: 'curl_setopt($[url])', 'writer.writer[ows]', 'Vue [-color]'
  - 4% [none] period ending a sentence about a lonely, cold or sorry moment: 'felt sorry.', 'wanted to play by herself.'
  - 3% [sad] 'went' after seeing a sad boy or woman, before approaching them: 'She went over to him'
  - 2% [none] article or noun in recipe steps: 'in [a] light coating of olive oil', 'let the [dough] rest'
  - 2% [emotion-general] 'came' after a character cries, panics or frowns: 'Her mom saw her and came'
  - 48% unclustered: 'stay' after 'had to', 'm' in an apology, a period before dialogue about a sink

## 4352: proud #4 (+0.013), afraid #5 (+0.016)
- dA: afraid +1.00, inspired -0.78, nervous +0.65 | hits TS/FW/UC: 74% / 9% / 17%
- Fires on narrative connectives and event tokens (then, suddenly, and, heard, a) that drive plot turns, with a few fear or relief contexts; mostly structural rather than emotion-specific.
  - 14% [none] double space after a sentence about crying, fear or happiness: 'started to cry.[  ]His mommy hugged him'
  - 12% [none] technical verbs or nouns mid-sentence: 'may also [cause] squeal', 'leads [to] loaded contact', 'a reddish laser [beam]'
  - 9% [emotion-general] 'He'/'She' opening a sentence after a fear or relief moment: 'Max was scared... [He] started to swing'
  - 7% [none] 'heard'/'hears' after 'But then' or 'Suddenly' introduces a sound: 'But then he [heard] some voices'
  - 5% [none] activity verb after 'enjoyed jumping and' or 'went back to': '[running] around', '[playing] bury', 'no more [jumping]'
  - 4% [emotion-general] 'then' or 'Suddenly' after a sad or scary moment: 'She felt so sad! But [then]', 'Sarah was very scared. But [then] her mom'
  - 3% [none] place noun ending a locative phrase: 'hid behind a [tree]', 'from inside the [room]', 'across the [lake]'
  - 3% [none] indefinite 'a' after 'heard', 'saw' or 'It was': 'It was [a] shark', 'they hear [a] loud noise'
  - 2% [none] pronoun or auxiliary opening dialogue: 'Sara, please open the door. I [love] you. [I] want to help', 'Do [you] understand?'
  - 2% [none] 'and' after a physical action: 'pulled his hand out [and] ran away', 'peeked inside [and] saw a machine'
  - 40% unclustered: scattered tokens: 'when' after 'opened the door', 'animals' in 'no other [animals]', 'but' after 'afraid'

## 5869: desperate #4 (+0.009)
- dA: desperate +1.00, nervous +0.72, calm -0.70 | hits TS/FW/UC: 33% / 30% / 38%
- Fires on 'and' joining a character's feeling to a follow-on action or a second feeling, plus chat-transcript and URL tokens; the 'and' modes are emotion-related but the rest are not.
  - 97% [emotion-general] 'and' linking a character's feeling to the next action: 'Timmy was sad [and] didn't understand', 'Tom was shocked [and] started to cry'
  - 2% [none] 'assistant' tag after a user question in chat transcripts: 'user: Can you give examples... [assistant]:'
  - 1% unclustered: URL or citation tokens: 'buenastareas.com[/]ensayos', 'buenastareas[.com]/ensayos'

## 7954: guilty #4 (+0.013)
- dA: guilty -1.00, nervous +0.83, afraid +0.67 | hits TS/FW/UC: 20% / 51% / 29%
- Weak (about 0.5-0.8) and diffuse neuron with no single mode, covering names in academic text, how-to verbs, and medical word-pieces; not emotion-related.
  - 24% [none] Proper nouns and dates in news/academic text: 'February', 'University', 'Professor', '1' in 'April 11-14'
  - 17% [none] Motion verbs after a scary or exciting moment: 'he [ran] away to hide', 'smiled and [ran]', 'she [went] running'
  - 5% [none] Verbs and adjectives in task instructions: 'offer solutions', 'get by', 'easy to incorporate', 'could be improved'
  - 4% [none] Imperative or list-item verbs in how-to tips: 'Try switching', 'Doing the same workout', 'use your core', 'taking lessons'
  - 4% [none] Word-piece inside medical terms: 'plate|let', 'block|age', 'void|ing', 'double', 'ot|itis'
  - 46% unclustered: Weak, scattered tokens: 'Travel' in list, 'are' before 'most active', comma in 'Hi! However,'

## 7913: surprised #4 (+0.016)
- dA: surprised +1.00, loving -0.28, proud -0.24 | hits TS/FW/UC: 83% / 7% / 10%
- Fires on narrative-structure tokens in children's stories (speech tags, character subjects, 'asked', 'looked'), with one mode on 'was' right before feeling words like excited or sad; mostly not emotion-specific.
  - 19% [none] Character name or pronoun starting a sentence after a line of dialogue: 'Lily', 'Jane', 'Benny', 'Lucy'
  - 17% [none] Sentence-final period or quote before a new scene, e.g. 'very dull.', 'never been to the kitchen before.'
  - 12% [none] Scattered words: 'he' after 'said', 'identity', 'already', 'newfound', 'may' in narrative and advice text
  - 10% [emotion-general] 'was' after a character's name, right before a feeling word: 'Lily was [excited]', 'Lucy was [disappointed]', 'Jane was [surprised]'
  - 6% [none] 'looked' after 'He/She/Jack' hears a sound: 'He heard a loud snap! He [looked] down'
  - 2% [none] Speaker subject right before a speech tag: 'her mother said', 'His mommy said', 'Timmy', 'Their mom said'
  - 2% [none] 'asked' as a speech tag right after a character's question: 'he asked', 'Sarah asked', 'Mia asked', 'mom asked'
  - 32% unclustered: Mixed function words and names in children's stories: 'Tim', 'Their', 'As he', 'a razor', 'was' in 'thought this was'

## 10941: inspired #5 (+0.001)
- dA: afraid +1.00, nervous +0.94, angry +0.58 | hits TS/FW/UC: 32% / 26% / 43%
- Fires on mid-clause function words in descriptive or superlative sentences (sensory description, 'one of the lowest'); not emotion-related.
  - 94% [none] 'and'/'was'/'were' after a sensory adjective describing food, clothing or scenery
  - 2% [none] 'the' in superlative country or species claims: 'one of [the] lowest', 'has [the] highest'
  - 4% unclustered: mixed: 'or' in 'abandoned, [or] derelict', 'the most' before 'hated', 'the [distribution]' in statistics

## 8177: loving #5 (+0.012), afraid #8 (+0.007), nervous #5 (+0.011)
- dA: loving -1.00, nervous +0.80, afraid +0.67 | hits TS/FW/UC: 70% / 9% / 21%
- Fires on discourse-connective tokens in children's stories and user questions ('How can I', 'As', 'Suddenly', 'but', 'knew', opening quotes) rather than on feeling words; not emotion-related.
  - 42% [none] 'knew' after a character's choice or success: 'She [knew] she had made the right choice', 'he [knew] he had to be honest'
  - 14% [none] Period ending a sentence about wonder or a new discovery: 'had never seen a pond before[.]', 'Mia felt sad[.]'
  - 4% [none] Sentence-opening transition 'Suddenly' or 'sudden' or 'then' after a calm scene: 'All of a [sudden]', '[Suddenly] he had an idea'
  - 4% [none] Opening quote mark starting a character's line after a reaction or description: 'she said, ["]Why would he do that?'
  - 3% [none] 'but' after a wish or hesitation clause: 'he wanted to know more, [but]', 'she wanted to join them, [but] she was shy'
  - 2% [none] 'As' or 'But as' opening a clause after a setting sentence: 'But [as] she started walking', '[As] she fluttered'
  - 2% [none] 'can' in 'How can I' user questions asking how to deal with a problem: overwhelm, procrastination, achieving a dream
  - 2% [none] 'I' starting a reply after a comforting line: 'Don't worry, [I]'ll help you', '[I] can help you', '[I]'m sorry'
  - 2% [none] Period ending a scene-setting or alarming sentence before a new paragraph: 'the moon was amazed by[.]', 'a big bad wolf in the garden[.]'
  - 25% unclustered: Pronoun or comma in dialogue and narration: 'give [me] a number', 'If I can make it', 'time he felt tired or scared[,]'

## 9443: proud #5 (+0.009)
- dA: inspired +1.00, happy +0.88, desperate -0.88 | hits TS/FW/UC: 35% / 28% / 38%
- Fires on function words and punctuation in story text, most strongly just before or after happy or grateful phrasing (copula before 'happy', punctuation after 'I'm glad I could help'); partly emotion-related, mostly happy and gratitude.
  - 55% [none] 'and', 'would', 'escape', 'boy', ',' ending a triumphant or hopeful clause: 'a hero and', 'they would always'
  - 6% [emotion-general] '!', '.', or quote mark closing a gratitude or happy line: 'happy to help!', 'nurse.'
  - 5% [none] 'the' after 'of' or 'for' in dark, tense story text: 'weight of the', 'empathy for the'
  - 3% [happy] copula 'was', 'is', 'were', 'so' right before 'happy' or 'glad' describing the character
  - 2% [none] 'He', 'She', 'plane' starting a story's main clause after 'Once upon a time, there was'
  - 29% unclustered: 'What' opening a screenplay question line; 'Fre' mid-name in plain narration

## 1545: calm #5 (+0.009), angry #6 (+0.010)
- dA: angry +1.00, calm -0.80, proud +0.34 | hits TS/FW/UC: 27% / 36% / 38%
- Fires on 'loved' describing a character's attachment and on nouns in heartfelt letters about a loved one (loving); digits and word fragments in captions and citations are not emotional.
  - 57% [none] adjective or verb after 'so' or in a character trait: 'never felt so [happy]', 'you'll do [great]'
  - 3% [loving] 'loved' right after a character pronoun: 'He [loved] music', 'She [loved] blue roses'
  - 3% [none] digit or word fragment inside a number or word: '9[8] feet', '19[3]8', 'numbers [dw]indle'
  - 2% [loving] noun or pronoun in a heartfelt letter about a loved one: 'in my [life]', 'Our [bond]', 'spent with [you]'
  - 35% unclustered: misc.: 'there' after 'as', 'the' in 'All of [the] ridges', opening quote before 'I'

## 6882: desperate #5 (+0.008)
- dA: desperate +1.00, surprised -0.63, nervous +0.58 | hits TS/FW/UC: 21% / 42% / 37%
- Fires on time-span and clause-ending words in news, stories and chat; mainly temporal phrasing, not emotion-related.
  - 74% [none] 'time' in 'had such a great time', 'facing', 'well', 'there' closing a clause
  - 2% [none] 'years' in time spans: 'two years ago', 'a year ago', '15 to 20 years from now'
  - 2% [none] 'next' in 'next week' or 'next time' after a future event (presentation, big test)
  - 2% [none] 'over' in 'over the last year' or 'over the past 30 years' in news and reports
  - 20% unclustered: 'unable' in assistant refusals, 'the' before 'article', 'consistent' in a science paper

## 1967: desperate #6 (+0.008), guilty #5 (+0.012)
- dA: guilty -1.00, desperate -0.93, calm +0.63 | hits TS/FW/UC: 48% / 15% / 36%
- Fires on verbs, names, and function words in narrative and expository text; the only emotion-touching mode is 'laughed' in playful scenes.
  - 40% [none] marketing and business text: 'as', comma, 'promote', 'program', 'automation' mid-sentence
  - 15% [happy] 'laughed' after a child's playful action: 'kicked his feet and laughed', 'Billy laughed and hugged'
  - 11% [none] character name or 'girl' as subject of a new sentence: 'Tim[my]', 'Amy', 'Sarah', 'the little girl'
  - 2% [none] 'saw' after a child's name or pronoun: 'he saw a rabbit', 'Sarah saw two guards'
  - 31% unclustered: 'on' while waiting for mom, 'to' in 'seems to swallow', comma after 'Mrs. Johnson'

## 2707: sad #5 (+0.009)
- dA: surprised +1.00, sad -0.82, loving -0.57 | hits TS/FW/UC: 78% / 7% / 14%
- Fires on action verbs and pronouns in children's stories, such as reaching, rubbing sleepy eyes and clapping, and on the subject just before a surprise, fear or sadness, so its emotion link is weak and mixed.
  - 95% [emotion-general] 'and' or subject right before a surprise, fear or sadness: 'Maddy smiled [and]', 'Jane was very surprised', 'He was scared'
  - 2% [none] 'reached' after a child sees something and wants it: 'She [reached] out to catch', 'reached into her pocket'
  - 2% [none] 'their', 'his', 'rubbed' in 'yawned and rubbed [his] eyes' when a child is sleepy
  - 1% unclustered: 'the' in 'kissed her on [the] forehead', 'are' in 'Sara and Tom [are] friends', 'her' in 'clapped [her] hands'

## 8302: surprised #5 (+0.011)
- dA: surprised +1.00, proud -0.25, loving -0.19 | hits TS/FW/UC: 80% / 6% / 15%
- Discovery-and-description neuron that fires on the words that follow 'saw' or 'found' a new object (article, 'very', head noun, pronoun, period); not emotion-related.
  - 21% [none] Head noun for a newly spotted object after an adjective: 'a big metal [thing]', 'a big [vehicle]', 'a weird [creature]', 'a [symbol]'
  - 19% [none] 'It' or 'He' starting the sentence that describes a just-found creature or object: 'saw something shiny... [It] was so cute'
  - 9% [none] Period ending a sentence where the character notices a hazard or odd object: 'saw a plane nearby[.]', 'saw a worm struggling[.]'
  - 5% [none] Indefinite article 'a' after 'saw' or 'came across' introducing a newly found object: 'she saw [a] beautiful purple flower'
  - 4% [none] Dialogue punctuation after an exclamation or greeting: 'beautiful cricket[!]', 'Hi Mr. Crab[,]', 'said ["]Hello, Timmy'
  - 3% [none] Copula 'was/is' or pronoun after a noun describing a dream object: 'The bow [was] bigger', 'It [is] their dad's car', 'The lip [was] really big'
  - 2% [none] Numbers and nouns in instruction or report text: 'the other is [1] and a half', 'Look for [shoes]', 'Bordering lands [should]'
  - 2% [none] 'very' after 'was/is' before a descriptive adjective of an object or character: 'was [very] special', 'is [very] cute'
  - 36% unclustered: 'careful' after 'but be' in a warning, 'has' in 'it [has] two digits', 'a' in 'had [a] lot of animals'

## 11485: happy #6 (+0.011)
- dA: surprised -1.00, happy -0.98, angry -0.63 | hits TS/FW/UC: 89% / 2% / 9%
- Fires on the token right before a predicate in simple children's stories ('was [so] happy', '[very] excited', '[all] alone', dialogue openers); emotion-related only in its happy and proud modes.
  - 27% [none] sentence-opening subject pronoun or name after an action: '[He] shouted', '[She] closed her eyes', '[Anna] bent down'
  - 21% [emotion-general] period or spacing after a positive feeling line: 'so proud of herself[.]', 'so cool', 'so pleased'
  - 6% [proud] 'was' after a character subject, right before 'so proud' or 'very proud'
  - 4% [none] opening quote mark or comma right after 'said' or 'said,' starting dialogue
  - 3% [none] whitespace or brace tokens inside PHP and bash code, e.g. indent before an else-branch echo
  - 3% [happy] 'so' after 'was', right before 'happy' or 'surprised'
  - 3% [emotion-general] 'very' before a trait or feeling adjective: 'Tim was [very] impatient', 'She was [very] excited'
  - 3% [none] 'all' after 'was' in 'was all alone', 'all smiles', 'all covered in mud'
  - 2% [none] 'all' after 'They' starting a group reaction: 'They [all] ran over', 'They [all] thanked Jack'
  - 27% unclustered: mixed: 'What' opening a question, 'ran off [to]' find mom, 'flew [down]' for a closer look

## 659: inspired #6 (+0.000)
- dA: nervous -1.00, happy +0.57, desperate -0.51 | hits TS/FW/UC: 40% / 33% / 27%
- Weak token neuron: fires on 'and'/'were' after feeling words in one cluster (emotion-related) and on closing quote punctuation in another.
  - 96% [emotion-general] 'and' or 'were' right after a character's feeling or state word: 'so excited and', 'felt tired and'
  - 2% [none] punctuation closing a dialogue line: 'my new toy!"', 'Look!"', 'Sam,"', 'Who is there?"'
  - 1% unclustered: misc. digits and symbols: '7; e10[0]348', '30.90[%]', '[curl]_errno'

## 11593: loving #6 (+0.009)
- dA: loving +1.00, angry -0.43, afraid -0.42 | hits TS/FW/UC: 51% / 4% / 45%
- Fires on verbs, dialogue openers and list tokens in children's narration and assistant-style text, with one smiling-and-laughing happy mode; only weakly emotion-related.
  - 41% [happy] 'and' right after a happy action: 'smiled [and] waved', 'laughed happily [and] laughed'
  - 4% [none] opening quote mark or comma right after 'said,' or 'said to Sam,' before dialogue
  - 3% [none] list number or first letter of a numbered or acrostic item: '5[.] Malala', '4.[ L]ululemon', '[N]ew beginnings'
  - 2% [none] 'play' or 'playing' after 'to', 'would', 'continued' or 'Let's' in children's stories
  - 50% unclustered: mixed: 'extra [special]' comfort, 'leaves the [reader]', 'I love [you] both' after mom's praise

## 8415: proud #6 (+0.011)
- dA: inspired +1.00, nervous -0.84, afraid -0.73 | hits TS/FW/UC: 7% / 46% / 47%
- Mostly weak and unfocused, but one mode fires on the object of 'I love thee' or 'Love is not love' in love poetry, so it is loving-related in part.
  - 91% [none] Final noun or word closing a descriptive phrase: 'the best sausages in the [area]', 'Palm Springs [area]', 'a humor [site]'
  - 2% [loving] 'thee' in 'I love thee' and 'Love is not love [Which]' in love-sonnet lines (Browning, Shakespeare 116)
  - 7% unclustered: Stray tokens: 'user' label, 'the' in 'Tropical Storm Isaac', '1' in a numbered list

## 8249: calm #6 (+0.014)
- dA: calm -1.00, inspired +0.56, sad -0.49 | hits TS/FW/UC: 30% / 49% / 21%
- Weak grab-bag neuron firing on story-opening 'One day,', 'known as', 'in the', and motion-adverbs; no clear coherent mode and not emotion-related.
  - 71% [none] Adverb or adjective after a motion or growth verb: 'went [closer]', 'grew [stronger]', 'the micropipette [close] to', 'Ducky'
  - 7% [none] 'known' in 'known as' or 'known' after a named item or concept: 'a condition [known] as autism', 'the canyon system [known] as'
  - 6% [none] Comma after the story opener 'One day' before a new event: 'One day[,] Rose noticed', 'One day[,] Momo saw'
  - 4% [none] 'the' after a location preposition: 'played outside in [the] sunshine', 'lived in [the] U.S.', 'bath in [the] corner'
  - 3% [none] Period ending a sentence where something is lost or a character gets tired: 'nowhere to be found[.]', 'started to feel tired[.]'
  - 10% unclustered: Mixed tokens: 'called' in 'workshop series [called]', 'to' in 'came [to] be called', 'Identified' in DKIM text

## 4372: guilty #6 (+0.014)
- dA: guilty -1.00, inspired +0.29, angry -0.26 | hits TS/FW/UC: 66% / 20% / 14%
- Fires on many unrelated function and content tokens (thought, quote marks, said, never, modals), and the emotion-related modes (a child's feeling, 'Mom' after fear) are a minority.
  - 19% [emotion-general] character name or 'Mom' as subject after a child's feeling: 'Lily was scared and asked her [mom]', 'Joe was very upset'
  - 10% [none] modal or verb after an opinion frame: 'those who [would]', 'theories [for]', 'no institution [should]', 'too soon to [consider]'
  - 8% [none] random content words: 'maps' in an AR list, 'knight' in pretend play, 'topping' in a recipe, 'bee' after 'Suddenly, a'
  - 3% [none] 'said' after 'smiled and': 'She smiled and [said]', 'The lady smiled and [said] yes'
  - 3% [none] 'never', 'not', 'get' or 'always' in a moral ending: 'and [never] acted grumpy again', 'think things through and [not] jump'
  - 2% [none] opening quote mark starting a character's question: 'said ["]Why are you sad?', 'said ["]What's wrong?'
  - 2% [none] comma after 'asked her mom' or 'Lily said' before a quoted line: 'asked her mom[,] "What happens'
  - 2% [none] 'thought' after a character considers something: 'Jane [thought] about it', 'Benny [thought] for a moment'
  - 2% [none] comma after a family address in speech: 'Mom[,] I am scared', 'Mum[,] is that expensive?'
  - 47% unclustered: scattered tokens: 'had' in 'he [had] offered his help', 'You' after 'assistant:', 'to' in 'impossible [to] predict'

## 2837: sad #6 (+0.007)
- dA: sad -1.00, happy +0.47, angry +0.20 | hits TS/FW/UC: 62% / 19% / 19%
- Fires on 'so' after 'was' in children's stories just before a feeling word (excited, surprised, embarrassed) and on play verbs in happy group scenes, so its emotion link is clear only for the 'so' lead-in mode.
  - 90% [none] end of a verb in happy group scenes: 'gigg[led] with delight', 'happy and [danced]', 'they explored and [played]'
  - 2% [emotion-general] 'so' after 'was' right before an emotion word: 'Sally was [so] excited', 'Joe was [so] embarrassed', 'Ginger was [so] surprised'
  - 9% unclustered: help-page and security text: 'any' and 'related' in Britannica tool text, 'whitelist [of]' for allowed sources

## 2352: inspired #7 (+0.001)
- dA: angry +1.00, inspired -0.99, happy +0.32 | hits TS/FW/UC: 41% / 31% / 28%
- Fires on phrasal particles like 'up' and 'out' and on subject pronouns in simple stories, with emotion appearing only in cluster 3, where it fires on 'excited' or 'scared' after 'was so'.
  - 17% [none] word fragments inside place or name tokens: 'West' Timor, 'Balk'an, 'Shar'psburg, 'dom'ed
  - 12% [none] subject pronoun starting a sentence after a character action: 'He thanked', 'They go to play', 'She felt regret'
  - 12% [none] 'can', 'thoughts', 'technical', 'stop', 'faster' in commentary on people's behavior or strategies
  - 11% [emotion-general] 'excited' or 'scared' right after 'was so' or 'was/were': 'was [excited]', 'Tip was [scared]'
  - 5% [none] 'out' in 'get out of', 'go out', 'get out of hand': 'go [out] into the real world'
  - 2% [none] 'up' in 'up and running' or 'up for the challenge': 'is [up] and running'
  - 41% unclustered: 'up' in 'hadn't given [up]' and 'get' in 'ensuring you [get] the most out of'

## 892: loving #7 (+0.013)
- dA: loving +1.00, guilty -0.41, inspired -0.35 | hits TS/FW/UC: 61% / 15% / 24%
- Broad token neuron (verbs before objects, 'felt' in touch scenes, texture nouns, periods); emotion-related only in the 'felt happy' and excited-'and' clusters.
  - 19% [none] verb or preposition in food-tip lists: 'before cooking [to] get', 'help to [create]', 'serve [with] a side'
  - 11% [happy] word right after 'felt' or 'feeling' in a happy scene: 'felt [happy]', 'feeling [happy]', 'felt [better]'
  - 9% [happy] 'and' after a character's excited or smiling state: 'so excited [and]', 'Jake smiled too [and]'
  - 6% [none] 'felt' after a touched object or creature: 'the mushroom [felt] all squishy', 'it [felt] cool'
  - 4% [none] 'the' after an action verb of holding or taking: 'grabbed the', 'held the', 'hugged the'
  - 3% [none] texture or temperature noun ending a touch phrase: 'warm [rock]', 'cold damp [stone]', 'soft grainy [texture]'
  - 3% [none] period ending a short narrative sentence: 'still worked[.]', 'started to purr[.]'
  - 45% unclustered: misc. mid-word or quote tokens: 'not to [slip]', opening quote before 'Look, Lily', 'said [Mag]u'

## 3361: happy #8 (+0.009), proud #7 (+0.007)
- dA: happy -1.00, loving -0.95, nervous +0.94 | hits TS/FW/UC: 38% / 47% / 15%
- Fires on feeling adjectives ('nervous', 'brave', 'scared') after 'was' or 'feeling', and on 'named' or 'called' in story openers; the feeling-adjective mode is emotion-related, the rest are not.
  - 98% [emotion-general] feeling adjectives after 'was' or 'feeling': 'nervous', 'brave', 'scared', 'embarrassment'
  - 2% [none] 'named' or 'called' after 'a little girl' in story openers: 'a little girl named Lily'
  - 0% unclustered: 'but', 'is' in quotations about Napoleon's view of history: 'history is a set of lies'

## 7555: calm #7 (+0.006)
- dA: calm +1.00, loving -0.99, guilty -0.77 | hits TS/FW/UC: 1% / 88% / 11%
- Fires on sentence boundaries and function words in history, news and essay text; no feeling link.
  - 77% [none] 'he', 'were', 'atmosphere', 'opinion', 'And' in essays on people's temperament and opinions
  - 2% [none] period ending a sentence in essays or news about parties, climate, or markets
  - 21% unclustered: mixed: ')' in a title citation, 'and' in a building description, 'the' before 'main gateway'

## 3497: desperate #7 (+0.005)
- dA: desperate +1.00, guilty -0.81, loving +0.51 | hits TS/FW/UC: 50% / 22% / 28%
- Fires on possessive and object pronouns in narratives, name endings in dialogue, and word-pieces in titles and forum text; none of the modes are emotion-related.
  - 9% [none] pronoun after 'everything' or 'all the': 'exploring everything it could find', 'everything they had'
  - 8% [none] 'her' or 'his' possessive before a person or place: 'saw her friend', 'waved goodbye to his mum'
  - 5% [none] name or 'Mom' endings in dialogue: 'Mom[my]', 'Matt[y]', 'M[olly]'
  - 4% [none] object nouns and 'hugged it' or 'hugged tight' in small stories: 'a cloud', 'a prison'
  - 3% [none] proper-name tokens in titles and bylines: 'Roots' in Grass Roots, 'Grove' in GroveWare, 'Morgan'
  - 2% [none] 'damn near', 'take it way too seriously', 'earlier' in forum opinion text
  - 69% unclustered: 'with' in 'look at it with wonder', 'bite', 'eat' in bedtime and pickle anecdotes

## 1637: angry #7 (+0.008)
- dA: inspired +1.00, angry -0.83, calm +0.52 | hits TS/FW/UC: 8% / 42% / 50%
- Fires on assorted noun, number and punctuation tokens across unrelated text types, with no emotion-related mode.
  - 93% [none] the word closing a noun phrase: pitfalls, designs, eco-friendly, objects, projects
  - 2% [none] list number before a numbered tip: '[5].' Stay hydrated, '[7].' Use visual elements
  - 4% unclustered: misc tokens: 'Last' in a Modified field, mojibake 'a-hat' in quotes, '.' after '3'

## 10553: guilty #7 (+0.008)
- dA: guilty -1.00, inspired -0.54, angry +0.37 | hits TS/FW/UC: 72% / 6% / 23%
- Fires on story-opening 'Once upon a time, there was' tokens and on the subject after a friendship or reunion; a narrative-structure neuron, not emotion-related.
  - 91% [none] 'They', 'they', 'Lily', or comma starting a sentence after a friendship or reunion
  - 3% [none] 'was' in story openers 'Once upon a time, there was a little girl named'
  - 3% [none] comma after the story opener 'Once upon a time' before 'there was a girl named'
  - 3% unclustered: 'assistant' tag right after a user turn in chat transcripts

## 972: sad #7 (+0.005)
- dA: sad -1.00, desperate +0.92, loving -0.88 | hits TS/FW/UC: 8% / 35% / 58%
- Token neuron on digits and punctuation in phone numbers and verse references, with a weak abstract-state cluster; not emotion-related.
  - 73% [none] word continuing a character's state or impact description: 'Suzanne's [addiction] causes', 'grew [stronger]'
  - 17% [none] a digit inside phone or fax numbers: '[5]41-737', '355-[4]523', '20[2] Natural Resources'
  - 4% [none] dash or digit inside Bible chapter-verse ranges: 'xxi. 22-[2]3', 'Num. xxvii. 15[-]23'
  - 5% unclustered: misc.: 'imperceptibly [new] tasks', 'Ulix[ert]inib', '[user]:' tag

## 10870: afraid #7 (+0.010)
- dA: afraid +1.00, nervous +0.76, happy -0.49 | hits TS/FW/UC: 46% / 25% / 29%
- Fires on narrative transition and subject tokens in children's stories, plus 'very' before adjectives and some medical or recipe text; only the scare-or-joy sentence-end mode is loosely emotion-related.
  - 19% [none] 'mashed', 'cream', 'noise', 'and', 'size' in recipe sentences or story lines
  - 6% [emotion-general] period ending a sentence about a scare or a joy: 'looked terrible.', 'laughed.', 'was locked.'
  - 6% [none] 'separate', 'or', 'ular', 'B' in medical or animal-husbandry reference text
  - 4% [none] 'was' after 'It' describing an object: 'It was heading', 'It was dull green'
  - 3% [none] 'very' before an adjective describing an object: 'very special', 'very high', 'very bright'
  - 3% [none] 'noticed' or 'saw' after a character looks around or opens something: 'he noticed', 'she saw'
  - 2% [none] 'Suddenly', 'Then', or comma starting a plot-twist sentence in a children's story
  - 2% [none] 'She', 'He', 'It' starting a clause after a plot beat in a story, e.g. 'She quickly became worried'
  - 2% [none] 'she' or 'that' after yoga-benefit sentences: 'she naturally started', 'that her anxiety'
  - 52% unclustered: ''s', '9', 'Pest' in mixed how-to, dialogue, and reference text

## 6561: proud #8 (+0.007), nervous #7 (+0.010)
- dA: nervous -1.00, happy +0.54, proud +0.54 | hits TS/FW/UC: 40% / 9% / 51%
- Fires on chat-transcript structure ('you' after Could/Can, colons after role labels) and story narration; not emotion-related.
  - 94% [none] 'and' or 'She' after a character smiles or nods in a children's story
  - 2% [none] colon after 'assistant' starting a reply, mostly 'Sure' after a user request
  - 2% [none] 'you' right after 'Could' or 'Can' opening a user's request to the assistant
  - 1% unclustered: colon after 'user' or 'assistant' role label in chat transcripts; also a citation digit

## 6679: surprised #7 (+0.009)
- dA: surprised -1.00, loving +0.78, guilty +0.48 | hits TS/FW/UC: 94% / 2% / 3%
- Fires on function words and subject references in simple children's stories, predicting the next action; not emotion-related.
  - 20% [none] character name or pronoun opening a sentence after another character's problem (Billy, Max, He)
  - 9% [none] 'a', 'her', 'his' after 'had', 'with', 'asked' in a child's story about possessions
  - 9% [none] 'and' right after 'smiled' or 'laughed' in children's stories
  - 4% [none] period ending a child's sentence about an outing or plan (park, lake, candy store)
  - 57% unclustered: function words in simple stories: 'with' before 'their hands', 'I' in 'Can I push'

## 7504: inspired #8 (+0.001)
- dA: inspired -1.00, surprised +0.66, nervous -0.33 | hits TS/FW/UC: 42% / 28% / 30%
- Fires on verbs and 'Thank' openers in children's stories and a few comparison or recipe phrases; not emotion-related.
  - 96% [none] verb or complement after 'happy that she', 'brave enough to', 'had found' in children's stories
  - 3% [none] 'Thank' opening 'Thank you' gratitude in children's dialogue
  - 1% unclustered: 'than' in 'more toys than you', 'To' in 'What To Do', 'a' in 'There are a variety'

## 881: loving #8 (+0.007)
- dA: loving +1.00, inspired -0.43, desperate -0.26 | hits TS/FW/UC: 72% / 14% / 14%
- Mostly a token neuron on 'a' in story openers and 'of' after 'One'; its cluster after characters' emotional gestures is emotion-related.
  - 87% [emotion-general] 'and', ',' or '.' right after a character's emotional gesture: 'smiled [and]', 'in shame[,]', 'got bored[,]'
  - 4% [none] 'of' right after sentence-initial 'One': 'One [of] the ongoing challenges', 'One [of] these days'
  - 2% [none] 'a' after 'Suddenly,' or 'saw' opening a story event: 'Suddenly, [a] storm', 'saw [a] funny stone'
  - 7% unclustered: misc.: period after 'intersections', 'Her' after a date, spaces in PHP code

## 10047: desperate #8 (+0.004)
- dA: desperate +1.00, inspired +0.68, nervous -0.49 | hits TS/FW/UC: 70% / 15% / 14%
- Fires on 'I want' desire statements and narrative subject pronouns in children's stories; desire is not one of the listed emotions, so not emotion-related.
  - 93% [none] 'He', 'he', 'She', 'They' starting a clause after a character is busy, tired, or sad
  - 5% [none] 'want' in 'I want to play', 'I want to touch' when a child sees something fun
  - 2% unclustered: 'not' in 'I do not have', 'to' in 'easy to memorize', 'there' in 'if there are'

## 3754: angry #8 (+0.007)
- dA: loving +1.00, inspired +0.84, angry -0.81 | hits TS/FW/UC: 94% / 1% / 6%
- Fires on narrative action and subject tokens in children's stories (play, stood, was, names), with a smaller set of happy or excited contexts such as 'smiled' and 'was so happy'; only part of it is emotion-related.
  - 12% [none] physical action verbs after a character's cue: 'She [stood] on a chair', 'she [ran] ahead', 'she [sat] down'
  - 11% [emotion-general] character name as sentence subject after an emotional beat: 'Lucy [y] nodded', 'Sarah was excited', 'L[ily] stopped weeping'
  - 10% [happy] 'She'/'He' opening a sentence after a happy or relieved event: 'She felt happy and proud', 'He was so glad'
  - 5% [emotion-general] 'was' after a character name, right before a feeling: 'Lily [was] so happy', 'Jenny [was] so excited'
  - 4% [happy] 'and' right after a feeling or smile: 'Lily was happy [and] warm', 'Peppa smiled [and] relaxed'
  - 3% [none] 'play' after 'loved to' or 'liked to' in children's stories: 'loved to [play] outside'
  - 2% [none] first subword of expressive verbs after a character subject: 'Teddy [p]outed', 'Mittens [p]urred', 'Lucy nodded and [h]opped'
  - 2% [happy] 'smiled' right after a character's name in a warm scene: 'Sue [smiled] and laughed', 'Becky [smiled]'
  - 51% unclustered: scattered tokens: 'out' in 'find [out]', 'Ted' as subject, 'She' after 'Lily felt sad'

## 2215: guilty #8 (+0.007)
- dA: guilty -1.00, sad -0.44, loving -0.36 | hits TS/FW/UC: 49% / 26% / 25%
- Fires on the word after a verb or preposition in simple stories and on technical nouns in essays, with no consistent emotion mode (only a weak 'so ... curious' or 'feels sorry' link).
  - 39% [none] technical nouns in expository text: 'desktop [management]', 'tasks', 'technology', 'fire-safety [professionals]'
  - 15% [none] 'to', 'could' after a verb of movement or asking: 'her room [to] get', 'if she [could] help'
  - 7% [none] subject pronoun or name starting a sentence after a reunion or surprise: 'She held his hand', 'Mom was very proud'
  - 4% [none] adjective after 'so' or 'very' or 'are': 'so [big]', 'very [curious]', 'are [quiet]', 'so happy and [grateful]'
  - 2% [none] object pronoun or name after 'showed' when a visitor arrives: 'showed [him]', 'showed [them]', 'showed [Susan]'
  - 2% [none] 'the' after a place or travel preposition: 'got to [the] street', 'to [the] park', 'pointed to [the]'
  - 31% unclustered: 'and' in 'Mom [and] Dad', '.' after 'He feels sorry', 'go' in a list of New Year plans
