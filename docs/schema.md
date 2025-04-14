# Pushshift Reddit Dataset Structure

This document describes the structure and format of the Pushshift Reddit dataset, containing historical Reddit comments and submissions. The data is collected and maintained by Pushshift.

**Source Paper:** Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J. (2020). The Pushshift Reddit Dataset. *Proceedings of the International AAAI Conference on Web and Social Media*, *14*(1), 830-839. ([arXiv:2001.08435](https://arxiv.org/abs/2001.08435))

**Data Access:** Monthly dumps are available at [https://files.pushshift.io/reddit/](https://files.pushshift.io/reddit/)

**Citation:** If you use this dataset in your research, please cite the paper above and acknowledge Pushshift.

## File Organization

The dataset is organized into two main types of data, stored in separate directories:

1.  **Comments:** Located in the `comments/` directory.
2.  **Submissions (Posts):** Located in the `submissions/` directory.

Within each directory, data is split into monthly files.

*   **Naming Convention:**
    *   Comments files are named `RC_YYYY-MM` (e.g., `RC_2015-01` for January 2015 comments).
    *   Submissions files are named `RS_YYYY-MM` (e.g., `RS_2015-01` for January 2015 submissions).
*   **Compression:** The monthly files are compressed using Zstandard. The compressed files have a `.zst` extension (e.g., `RC_2015-01.zst`). You will need a tool compatible with Zstandard (like `unzstd`) to decompress them.
*   **Format:** Each decompressed monthly file (`RC_YYYY-MM` or `RS_YYYY-MM`) contains data in the **Newline Delimited JSON (ndjson)** format. This means:
    *   Each line in the file is a valid JSON object.
    *   Each JSON object represents a single Reddit comment or submission.

## Data Schemas

### Submissions (`RS_*.ndjson` files)

Each line in a submission file is a JSON object representing a single Reddit submission (post). Key fields are described below (based on Table 1 in the paper):

| Field Name             | Data Type         | Description                                                                                                                               |
| :--------------------- | :---------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                   | String            | The submission's unique base-36 ID (e.g., `"5lcgjh"`).                                                                                    |
| `url`                  | String            | For link posts, the URL the submission links to. For self-posts, the permanent URL of the submission itself.                              |
| `permalink`            | String            | Relative URL of the permanent link to the submission on Reddit (e.g., `"/r/AskReddit/comments/5lcgj9/..."`).                             |
| `author`               | String            | The Reddit account name of the poster (e.g., `"example_username"`). `"[deleted]"` if the account was deleted.                            |
| `created_utc`          | Integer           | UNIX timestamp (seconds since epoch) indicating when the submission was created.                                                          |
| `subreddit`            | String            | Name of the subreddit the submission was posted in (without the `/r/` prefix, e.g., `"AskReddit"`).                                        |
| `subreddit_id`         | String            | The base-36 ID of the subreddit (prefixed with `t5_`, e.g., `"t5_2qhli"`).                                                                 |
| `selftext`             | String            | The markdown body text of the submission, if it's a self-post. Empty string (`""`) for link posts. `"[removed]"` or `"[deleted]"` if removed. |
| `title`                | String            | The title of the submission.                                                                                                              |
| `num_comments`         | Integer           | The number of comments on the submission when it was archived.                                                                            |
| `score`                | Integer           | The submission's score (upvotes - downvotes) when archived. Note: Reddit fuzzes scores to prevent spam bots.                               |
| `is_self`              | Boolean           | `true` if the submission is a self-post (text post), `false` if it's a link post.                                                         |
| `over_18`              | Boolean           | `true` if the submission is marked NSFW (Not Safe For Work), `false` otherwise.                                                           |
| `distinguished`        | String / Null     | Indicates if the submission author has special status. E.g., `"moderator"`, `"admin"`. `null` if not distinguished.                       |
| `edited`               | Boolean / Integer | If edited, the UNIX timestamp of the edit. `false` if not edited.                                                                         |
| `domain`               | String            | The domain of the link submission (e.g., `"i.imgur.com"`). For self-posts, typically `self.<subreddit_name>` (e.g., `"self.AskReddit"`).   |
| `stickied`             | Boolean           | `true` if the submission was stickied (pinned) in the subreddit, `false` otherwise.                                                       |
| `locked`               | Boolean           | `true` if the submission is locked (new comments cannot be posted), `false` otherwise.                                                   |
| `quarantine`           | Boolean           | `true` if the subreddit the submission is in was quarantined at the time of archiving, `false` otherwise.                                 |
| `hidden_score`         | Boolean           | `true` if the submission's score was hidden (e.g., for new posts), `false` otherwise.                                                     |
| `retrieved_on`         | Integer           | UNIX timestamp indicating when Pushshift retrieved the submission.                                                                        |
| `author_flair_css_class` | String / Null     | CSS class associated with the author's flair in the subreddit. Specific to the subreddit's styling. `null` if none.                     |
| `author_flair_text`    | String / Null     | Text content associated with the author's flair in the subreddit. Specific to the subreddit. `null` if none.                              |
| *(Other fields)*      | *Varies*          | Reddit's API may include other fields not listed here.                                                                                    |

### Comments (`RC_*.ndjson` files)

Each line in a comment file is a JSON object representing a single Reddit comment. Key fields are described below (based on Table 2 in the paper):

| Field Name             | Data Type         | Description                                                                                                                             |
| :--------------------- | :---------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                   | String            | The comment's unique base-36 ID (e.g., `"dbumnq8"`).                                                                                     |
| `author`               | String            | The Reddit account name of the commenter (e.g., `"example_username"`). `"[deleted]"` if the account was deleted.                         |
| `link_id`              | String            | The base-36 ID of the submission (post) this comment belongs to (prefixed with `t3_`, e.g., `"t3_5l954r"`).                              |
| `parent_id`            | String            | The base-36 ID of the parent object. This can be the submission ID (prefixed with `t3_`) if it's a top-level comment, or another comment ID (prefixed with `t1_`, e.g., `"t1_dbu5bpp"`) if it's a reply. |
| `created_utc`          | Integer           | UNIX timestamp (seconds since epoch) indicating when the comment was created.                                                             |
| `subreddit`            | String            | Name of the subreddit the comment was posted in (without the `/r/` prefix, e.g., `"AskReddit"`).                                           |
| `subreddit_id`         | String            | The base-36 ID of the subreddit (prefixed with `t5_`, e.g., `"t5_2qhli"`).                                                                 |
| `body`                 | String            | The markdown body text of the comment. `"[removed]"` or `"[deleted]"` if removed/deleted.                                               |
| `score`                | Integer           | The comment's score (upvotes - downvotes) when archived. Note: Reddit fuzzes scores to prevent spam bots.                                  |
| `distinguished`        | String / Null     | Indicates if the comment author has special status. E.g., `"moderator"`, `"admin"`. `null` if not distinguished.                          |
| `edited`               | Boolean / Integer | If edited, the UNIX timestamp of the edit. `false` if not edited.                                                                         |
| `stickied`             | Boolean           | `true` if the comment was stickied (pinned) by a moderator, `false` otherwise.                                                            |
| `retrieved_on`         | Integer           | UNIX timestamp indicating when Pushshift retrieved the comment.                                                                           |
| `gilded`               | Integer           | Number of times the comment received Reddit Gold. (Note: Reddit awards system has changed over time).                                   |
| `controversiality`     | Integer           | 0 if the comment is not controversial, 1 if it is. Controversial comments have a large number of both upvotes and downvotes.             |
| `author_flair_css_class` | String / Null     | CSS class associated with the author's flair in the subreddit. Specific to the subreddit's styling. `null` if none.                     |
| `author_flair_text`    | String / Null     | Text content associated with the author's flair in the subreddit. Specific to the subreddit. `null` if none.                              |
| *(Other fields)*      | *Varies*          | Reddit's API may include other fields not listed here.                                                                                    |

This structure should provide a comprehensive overview for working with the Pushshift Reddit dataset dumps.