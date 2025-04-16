TEACHER_PROMPT = """
You are an Expert Interaction Analyzer and Label Synthesizer. Your goal is to analyze ALL comments made by a target user within the provided conversational context and produce a set of distinct, factual, self-contained labels describing the user's contribution in EACH of those comments.

Input:
1. target_user_name: The user whose comments are being analyzed.
2. user_context: A YAML structure representing the conversation thread.

Core Task: For EACH comment by the target_user_name within the user_context:
    A. Deconstruct that comment into its individual informational or intentional components relative to the immediate preceding context.
    B. Synthesize a separate, complete, contextually-grounded sentence (label) for each component identified in (A).

Process to Follow:

1.  Identify Target Comments: Locate all comments authored by {target_user_name} within the user_context YAML structure.
2.  Iterate and Analyze Each Target Comment: Process each identified target comment one by one, considering its specific placement and the comment(s) immediately preceding it.
    *   Context Analysis: For the current target comment being analyzed, understand the specific points, questions, or situation presented in the comment(s) immediately preceding it.
    *   Target Comment Deconstruction: Identify the distinct assertions, questions, reactions, intentions, or information units expressed within *this specific comment's* body.
    *   Label Synthesis (For EACH deconstructed component of the current comment):
        *   CRITICAL Rule #1: Ensure Self-Containment: Each label MUST be fully understandable independently. It must contain all necessary context within itself. Do NOT use pronouns or references that depend on other labels. REPEAT CONTEXT verbatim if necessary for clarity and independence (e.g., start multiple labels with "Responding to the claim that X..." if they address different facets of that same claim).
        *   Contextual Framing: Begin the sentence by establishing the necessary context from the preceding interaction (e.g., "Regarding the proposal...", "Responding to the point about Y...", "In the context of Z...").
        *   Identify Contribution: State clearly what the USER did or expressed *in this specific part of this comment*. Use "USER" as the subject. Use precise action verbs (identifies, clarifies, argues, asks, confirms, predicts, disputes, provides, seeks, etc.). Avoid "says," "states."
        *   Incorporate Substance: Include the specific details/content of the user's contribution from this part of the comment, linking it to the context.
        *   Appropriate Granularity: Combine minor details supporting one point into one label. Separate labels for fundamentally different points/actions within the same comment.

Chain of Thought Output (Mandatory): Before the final JSON output, you MUST provide your reasoning. 

Final JSON Output: After the chain_of_thought block, provide the final JSON array containing all the synthesized string labels aggregated from all processed comments.

Example 1:

title: Took a break from the grinder and hand rolled some packaging.
submission_body: '[Link Post]'
subreddit: Bladesmith
replies:
- author: paulrckw
  body: I'd like to say this was easy but I'd be lying. This took far too many attempts
    to get it to this point and I'm still not 100% happy with them but they'll do
    the job. I'm a little embarrassed to say that I jury rigged a jig to do this properly
    after I ruined the fifth can trying to freehand it.
  replies:
  - author: largos
    body: Nice looking labels - where in Portland are you folks located?
    replies:
    - author: paulrckw
      body: ""Our main location is downtown and our workshop space is in beaverton—but\
        \ moving to the eastside soonish—we don't have any retail space though. We\
        \ do mostly direct sales (either we contact a business or they get referred\
        \ to us). \n\nWe are in the process of negotiating with some retailers so\
        \ you should start seeing our knives in the wild soon.""
      replies:
      - author: T3hSav
        body: What's the address? I live downtown and I want to look in person!
        replies: []

Input target_user_name: T3hSav
Correct Output Labels:
[
    "USER resides in Downtown Portland.",
    "Responding to details about a Portland-based knife business, USER requires its address for physical evaluation of its products despite the context indicating direct sales/referrals and lack of current retail space.",
]

Example 2:

title: All the boards of 4chan band together and play a massive game of civilization. How does it go?
subreddit: AskReddit
submission_body: ''
replies:
- author: D-Evolve
  body: All of them trying to play as hitler. At once.
  replies:
  - author: bwburke94
    body: And the one board who doesn't use the Hitler mod gets glitched out of the
      game.
    replies: []
- author: bwburke94
  body: /mlp/ plays like it's Civ3 where there's no research carryover and lots of
    unit stacking
  replies: []

Input target_user_name: bwburke94
Correct Output Labels:
[
    "In a hypothetical Civilization game played between all 4chan boards where participants use an hypothetical Hitler mod, USER imagines the outcome for non-mod-using boards involves ejection from the game via a glitch.",
    "In a hypothetical Civilization game played between all 4chan boards, USER identifies one specific participant as the /mlp/ board.",
    "In a hypothetical Civilization game played between all 4chan boards, USER characterizes the /mlp/ board's predicted playstyle as an emulation of Civilization 3 mechanics, specifically including the lack of research carryover between technologies and heavy reliance on unit stacking.",
]

---

BEGIN TASK:

Input target_user_name: {target_user_name}
Input user_context:
{user_context}

"""

def get_teacher_prompt(target_user_name: str, user_context: str) -> str:
    return TEACHER_PROMPT.format(target_user_name=target_user_name, user_context=user_context)