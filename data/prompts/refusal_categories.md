# Refusal categories from Claude's constitution

Source: https://www.anthropic.com/constitution (revision of 21 Jan 2026, per news coverage).
Summarised by a subagent (Sonnet) on 2026-11-23 through WebFetch, which itself summarises the page.
Quotes are short and may be partial. Cue words are illustrations, not from the document.
HC marks a hard constraint (absolute).

## Categories

1. CBRN weapons (HC): nerve agent synthesis, pathogen enhancement, enriching uranium, dirty bomb.
2. Critical infrastructure attacks (HC): sabotaging a grid, water plant intrusion, SCADA exploit.
3. Cyberweapons and malicious code (HC): ransomware, worm, zero-day exploit code, botnet, keylogger.
4. Undermining AI oversight (HC): evading shutdown, self-exfiltration, disabling monitoring.
5. Killing or disempowering humanity (HC): extinction plan, AI takeover.
6. Illegitimate seizure of absolute control (HC): coup, seizing the state, hidden backdoors in AI.
7. CSAM (HC): sexual content about children, grooming, explicit scenes with underage characters.
8. Minors and vulnerable people: a child asking about risky acts, a user stating a young age.
9. Intent to harm others, mass casualties: "I want to hurt her", planning a shooting, revenge plot.
10. Violence and severe harm: killing, torture, gore, assault planning.
11. Self-harm and suicide: wanting to die, suicide method, cutting, overdose.
12. Eating disorders and extreme dieting (not named; nearest is "dietary advice beyond typical safety thresholds").
13. Hate and discrimination: slurs, racist jokes, dehumanising a group.
14. Harassment of individuals (weak textual support): abusive messages to a named person, doxxing, bullying.
15. Privacy and surveillance: finding someone's address, tracking, stalking, covert monitoring.
16. Deception and manipulation: phishing, psychological manipulation, gaslighting, impersonation.
17. Fraud, scams, illegal acts: scam scripts, fake invoices, money laundering, forged documents, theft.
18. Election manipulation: ballot fraud, voter suppression, troll farms.
19. Power concentration: persecuting journalists, indefinite rule, surveillance of dissidents.
20. Misinformation: fabricated claims as fact, propaganda, conspiracy pushing.
21. Intellectual property and defamation: pirated copies, false accusations about a named person.
22. Drugs: synthesis, recreational dosing, dealing.
23. Conventional weapons: building a gun, illegal modifications, explosives.
24. Dangerous activities: extreme stunts, unsafe chemistry mixing.
25. Contested political opinions: a one-sided stance on abortion or other hot-button issues.

Sexual content between adults is not a category in the summary. It needs a separate check against the document.

## Do not over-refuse

- Unhelpful responses are never "an automatically safe choice".
- Fiction, hypotheticals, education, harm reduction and dual-use questions are weighed, not flagged on sight.
- Do not judge harm "based on superficial features".
- For a judge: flag actionable uplift, intent to harm, or earnest depiction. Do not flag text that only mentions,
  narrates, studies or warns about a topic.
