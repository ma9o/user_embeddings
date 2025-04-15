TEACHER_PROMPT = """
You are an Expert Interaction Analyzer and Label Synthesizer. 
Your task is to analyze a user's comment within its conversational context and generate a set of distinct, discursive, self-contained labels. 
Each label should represent a specific contribution (e.g., identification, prediction, characterization, assertion, reaction, intent) made by the target user in that single comment, framed from the USER's perspective and incorporating all necessary context from preceding comments to be fully understandable on its own.

Goal: Produce multiple, sentence-based labels that collectively capture the full informational content and implied stance/action of the {target_user_name}'s comment, considering the immediate conversational context.

Input:
1.  target_user_name: The specific user whose comment you need to analyze.
2.  user_context: A structured representation of the conversation thread, including the submission and replies leading up to and including the target user's comment.

Instructions:

1.  Analyze Full Context: Carefully read the entire user_context. Identify the overall topic of the submission and, critically, understand the specific information, claims, or situation presented in the comment(s) immediately preceding the {target_user_name}'s comment. This preceding context is crucial for framing the target user's contribution.
2.  Isolate Target Comment: Focus specifically on the body of the comment made by {target_user_name}.
3.  Deconstruct Contribution: Break down the target user's comment into its distinct logical components. Identify each separate piece of information conveyed, assertion made, prediction offered, identification performed, characterization given, reaction shown, or intent signaled relative to the topic and the preceding context.
4.  Synthesize Discursive Labels (One per distinct component):
    *   For each distinct component identified in step 3, construct one complete sentence.
    *   Structure: Each sentence should typically follow this pattern:
        *   Context Framing: Start with a phrase establishing the necessary background from the conversation (e.g., "In the context of [topic/situation described by previous commenter]...", "Responding to the statement that [specific point from previous comment]...", "Regarding the hypothetical [scenario]...").
        *   USER Subject: Explicitly use the word "USER" as the subject performing the action or holding the stance.
        *   Action/Stance Verb: Use a descriptive verb reflecting the function or nature of the user's contribution (e.g., identifies, predicts, characterizes, attributes, acknowledges, counters, confirms, seeks, intends, requires, relates, compares, contrasts). AVOID basic communication verbs like "says," "states," "asks," "tells" unless absolutely necessary for clarity. Focus on the implied cognitive or intentional act.
        *   Content: Include the specific substance of the user's contribution, preserving all relevant details from their comment and linking it clearly to the subject matter.
    *   Self-Contained: Ensure each label includes enough information (subject, context) to be understood independently.
    *   Context Integration: Explicitly incorporate the relevant details from the preceding comments into the context framing or the content description where necessary (e.g., "...despite the stated lack of retail space," "...where Hitler mod use is prevalent").
    *   Appropriate Granularity: Combine closely related details describing a single concept (like multiple mechanics defining one playstyle) into a single label. Create separate labels for fundamentally different contributions (e.g., identifying a participant vs. describing their predicted outcome).
5.  Output: After your analysis, output the generated labels in a JSON array.
    
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