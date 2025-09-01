
# Core Assumptions and Approach Details

## 1. Goal: Universal & Adaptable User Embeddings

*   **Assumption:** It's possible to create a universal user embedding model trained on rich interaction data from potentially multiple platforms that captures nuanced user states, interests, and behaviors.
*   **Adaptability:** This universal embedding should be directly adaptable for downstream tasks, particularly user-item recommendation, potentially using a Q&A framing inspired by models like Gecko.
*   **Motivation:** Move beyond task-specific models or models relying on simplified inputs to leverage the richness of interaction data (initially conversational, potentially broader) for more versatile and accurate user representation.

## 2. Data Sources & Input Representation Strategy

*   **Assumption:** The most informative signal for user understanding lies not just in *what* they interact with, but the *context* of that interaction (e.g., surrounding thread, search history context, etc.).
*   **Initial Data Source:** Utilize the Reddit Pushshift dataset as the primary initial source. The initial semantic unit will be a structured representation of a user's comment along with its surrounding thread context (submission title/body, parent authors/bodies).
*   **Multi-Platform Goal & Universal Input Format:**
    *   The long-term goal is to incorporate data from diverse platforms (e.g., Reddit, search history, etc.).
    *   **Initial Approach:** Develop platform-specific parsers (starting with Reddit) that convert raw data into a defined **universal intermediate representation**. This universal format will serve as the primary input structure for the student model.
    *   **Benefit:** This approach simplifies the student model, promotes modularity (add new platforms by adding parsers), and enforces a consistent input structure for initial training.
*   **Motivation:** Preserving rich context allows for capturing fine-grained interests, opinions, and interaction dynamics. Abstracting diverse sources into a universal format (initially) allows the student model to focus on semantic learning rather than parsing variations.

## 3. Processing & Training Signal Generation: LLM Teacher & Hierarchical Summaries

*   **Assumption:** A powerful "teacher" LLM (akin to Gecko's approach) can effectively distill meaningful training signals (semantic labels) from the complex contextual interaction units, regardless of their original source platform (once parsed).
*   **Approach (Explicit Rejection of Pre-clustering):** We will *not* perform explicit clustering of interaction embeddings before LLM processing. The LLM teacher will process individual contextual interaction units (provided in a consistent format by the parsers) directly.
    *   **Motivation:** Simplifies the pipeline, avoids potential biases introduced by clustering algorithms, and leverages the LLM's capability to understand context without pre-grouping.
*   **Approach (Target Generation - NL Labels):** The LLM teacher will be prompted to generate **hierarchical, item-like Natural Language (NL) summaries/labels** for each relevant user interaction unit. This means producing:
    *   **1. A Medium/Specific Summary:** Capturing the core topic, entity, action, or expressed interest/opinion directly related to the interaction (e.g., "Advising on secured credit card payment strategy"). Should be "item-like" for recommendation relevance.
    *   **2. A Broad Summary:** Capturing the general category or domain the specific interaction falls under (e.g., "Personal Finance Management").
    *   **Rejection of "State-of-Being" Targets:** Explicit summaries like "Is an expert trader" will likely *not* be generated as targets; this state is expected to emerge implicitly in the embedding space based on consistent item/topic interactions and the pre-trained student model's semantics.
*   **Motivation:** Providing consistent NL labels (both broad and specific) derived from potentially diverse interactions explicitly teaches the student model the semantic relationship between detailed actions and general interests across different data sources. This fosters better implicit clustering within the embedding space and supports recommendations at multiple levels of granularity.

## 4. Expected Capabilities

*   **Assumption:** Training on the Q/A task (mapping user context representation `Q` to NL label `A`) will implicitly yield embeddings suitable for **Q/Q (User-User) similarity**.
    *   **Motivation:** Users with similar interests will interact with similar items/topics, leading the teacher LLM to generate similar NL labels (`A`s). Training the student to map user context to these labels will cause their context embeddings (`Q`s) to converge in the shared semantic space. Similarity between user embeddings will thus reflect similarity in their learned interaction patterns and interests.
*   **Assumption:** The model will be capable of supporting **recommendations at varying levels of specificity** due to the hierarchical nature of the NL training targets.

## 5. Student Model Architecture, Training & Input Handling

*   **Assumption:** A single, powerful pre-trained text encoder can effectively learn to map user context representations (initially in the universal format) and the NL item/topic summaries into a shared semantic space.
*   **Approach (Architecture):** Employ a **single student encoder (bi-encoder)** architecture. This model will process both the user context input (`Q`, initially the universal intermediate representation) and the LLM-generated NL summaries (`A`, both positive and negative).
    *   **Motivation:** Aligns with modern practices (e.g., Gecko), offers parameter efficiency, enforces a unified semantic space directly leveraging consistent NL labels, and effectively utilizes pre-trained model capabilities.
*   **Approach (Base Model):** Start fine-tuning from a strong **pre-trained text embedding model**.
    *   **Motivation:** Provides foundational semantic understanding crucial for generalizing to unseen but semantically related concepts/items expressed in the NL labels, and implicitly capturing "states of being" based on interaction patterns represented semantically.
*   **Approach (Context Window):** The student model must operate within a **defined manageable context window** (e.g., targeting 8k-16k tokens). The input representation fed to the student (derived from the user's contextual history, whether universal format or future direct formats) must adhere to this limit.
    *   **Motivation:** Ensures practical deployability and computational efficiency. The teacher LLM handles distillation from potentially much larger raw contexts (or individual units), and sequence modeling within the student manages the compressed history.
*   **Approach (Training Objective):** The primary training objective will be a **Q/A (User Context Representation -> NL Label) contrastive task**. Given a user context representation (`Q`), the model learns to pull the relevant positive hierarchical NL summaries (`A_pos`) closer and push irrelevant negative summaries (`A_neg`) farther away in the embedding space.

### Corollary: Granular Teacher Processing Enables Holistic Student Learning

*   **Problem:** Processing a user's entire history across platforms at once for labeling individual interactions is often infeasible (context limits, cost) and can dilute the specific signal of the interaction being labeled.
*   **Teacher's Role (Granular Processing):** The "teacher" LLM acts as a highly sophisticated **feature extractor/event labeler** operating on individual interaction units (parsed into a consistent format). It leverages its deep understanding of local context to generate high-fidelity, distilled **NL labels (summaries)** for *each interaction event* efficiently and scalably. It focuses on "what happened *semantically* here?".
*   **Student's Role (Sequence Modeling & Compression):** The "student" model (likely Transformer-based) receives a **sequence** of these distilled NL labels or the intermediate representations corresponding to user interactions over time. Its strength lies in modeling sequences – identifying patterns, dependencies, and recurring themes *across* the time-ordered inputs derived from the granular teacher outputs.
*   **Holistic Learning (e.g., Personality/Expertise):** The student learns holistic concepts like personality, expertise, or stable interests by recognizing consistent patterns in the sequence of semantic inputs. For example, if the teacher repeatedly generates NL labels related to "sarcastic tone" or "detailed financial advice" across many interaction units (potentially from different platforms), the student's sequence modeling capabilities allow it to compress this recurring semantic pattern into its final embedding. This embedding implicitly represents the emergent, holistic property.
*   **Conclusion:** This approach uses each model for its strength: the teacher for accurate local context analysis and generation of consistent semantic NL labels at scale, and the student for sequence modeling, temporal integration, and compression of semantic patterns into a holistic user representation. Granular semantic processing by the teacher enables the student to learn global properties across diverse inputs.

## 6. Multi-Platform Strategy & Input Generalization

*   **Initial Approach (Universal Representation Focus):**
    *   Creating platform-specific parsers (starting with Reddit) that convert raw data into the defined **universal intermediate representation**.
    *   The student model will initially be trained *solely* on mapping this universal format (`Q`) to the semantic NL labels (`A`).
    *   **Benefit:** Simplifies the student model, promotes modularity (add new platforms by adding parsers), enforces a consistent input structure.
*   **Future Enhancement (Direct Multi-Format Training):**
    *   Potentially extending the training data to include inputs from different sources in *different formats* (e.g., raw-ish Google search history alongside the universal Reddit format). The NL labels generated by the teacher would remain the consistent semantic target (`A`).
    *   **Purpose:** To potentially capture richer interaction patterns found in diverse formats that are difficult to standardize perfectly, and to make the student model robust to parsing specific *known* input format variations.
    *   **Generalization Caveat:** It is crucial to understand that training on multiple formats improves robustness and parsing capability *for those specific, trained formats*. It does **not** automatically grant the ability to parse or understand the structure of **completely novel, unseen input formats**. The primary source of *semantic* generalization (understanding new topics, comparing across sources, handling unseen NL labels) stems from the use of consistent, semantic **NL labels** as training targets and leveraging a strong pre-trained **text** embedding student model.