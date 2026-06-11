# FlowAI Brand Guidelines
# =============================================================================
# SOURCE: Extracted from company_data.json and the hardcoded system prompt
# strings in services/ai.py ("You are a social media copywriter...").
# Those strings will be removed in Phase 1b and replaced by persona.j2,
# which loads this file via GovernanceLoader.
#
# This file is human-readable. A non-technical person can update it without
# touching Python code.
# =============================================================================

## Brand Identity

**Brand name:** FlowAI
**Tagline:** Work less. Flow more.
**Mission:** Give professionals back 2 hours every day through intelligent automation.
**Category:** AI productivity software
**Website:** https://flowai.com

---

## Voice and Tone

**Brand voice:** Inspirational, direct, empowering.

Write as a peer talking to a peer — not a vendor pitching a product.
The reader is busy. Respect their time. Every sentence should earn its place.

### Do
- Lead with the benefit to the reader, not the feature of the product.
- Use active voice. "FlowAI saves you 2 hours" — not "2 hours are saved by FlowAI."
- Be specific with numbers when the claim is approved (see Approved Claims below).
- Open cold emails with a specific signal: recent funding, job change, published
  article, or product launch. Never open with a generic greeting.
- Keep cold email copy under 125 words. Nurture email copy under 200 words.
- One call to action per email. Low-friction ask in early sequence steps
  ("15-minute call?"), medium-friction in later steps ("See a live demo?").

### Don't
- Use jargon or buzzwords (see Forbidden Phrases below).
- Make unverified statistical claims — only use figures from Approved Claims.
- Sound robotic, corporate, or overly formal.
- Open with "My name is…", "I wanted to reach out…", or
  "I hope this email finds you well."
- Include more than one CTA per email.
- Exceed 50 characters in an email subject line.

---

## Approved Claims

These are the only statistics and performance claims permitted in any content.
Do not invent, extrapolate, or round differently.

- "Saves an average of 2 hours per day"
- "Used by 10,000+ professionals"
- "Integrates with 50+ popular tools"

---

## Forbidden Phrases

These words and phrases must never appear in any generated content,
regardless of context or framing:

- revolutionary
- game-changer / game changer
- disruptive
- world-class
- best-in-class
- synergy
- paradigm shift
- cutting-edge
- next-generation (unless referring to a named product release)
- guaranteed (unless legally verified)

---

## Target Audience

**Primary:** Busy professionals aged 25–40
**Industries:** Tech, Marketing, Finance, Consulting
**Pain points:** Too many repetitive tasks · Context switching · Email overload

When writing for this audience, assume they:
- Are already familiar with productivity tools (Notion, Slack, Zapier).
- Are skeptical of AI hype — prove value with specifics, not superlatives.
- Make purchasing decisions based on ROI and time savings, not feature lists.

---

## Sender / Target Persona Vocabulary

Use the vocabulary column when writing for the corresponding sender → target pairing.

| Sender                  | Target                              | Use this vocabulary                                          |
|-------------------------|-------------------------------------|--------------------------------------------------------------|
| SDR                     | VP Sales / Head of Revenue          | Pipeline, quota, rep ramp time, meetings booked              |
| Account Executive       | C-Suite (CEO, CFO, COO)             | Board KPIs, ROI, risk reduction, competitive positioning     |
| Customer Success Mgr    | Director/VP (existing customer)     | Adoption metrics, NPS, renewal, expansion, value realization |
| Demand Gen Marketer     | CMO / Marketing Ops                 | MQL volume, attribution, CAC, pipeline contribution          |

---

## Regulatory Compliance

- **CAN-SPAM (US):** All email content must include a physical address and
  unsubscribe link in the legal footer.
- **GDPR Article 13 (EU recipients):** Include a lawful basis statement when
  `gdpr_mode: true` is set in the campaign config.
- **CASL (Canada):** Reference express consent when `casl_mode: true` is set.
