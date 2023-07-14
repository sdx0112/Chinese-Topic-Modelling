# Chinese-Topic-Modelling
Topic modelling for Chinese meeting notes followed by extensive topic analysis.

# Data Exploration
Please refer to [EDA.ipynb](https://github.com/sdx0112/Chinese-Topic-Modelling/blob/main/EDA.ipynb) for details.

The raw data `data/meeting notes.csv` contains 531 notes for 496 meetings, with two duplicated notes having the same content. The duplicated row is 
the note for `政治局会议` with title `研究部署在全党深入开展党的群众路线教育实践活动`. The two rows only differ one character on the `TITLE`, so it should be a typo.
The row with typo title was removed and the cleaned data was saved to `data/meeting notes clean.csv`.

From the histogram of Year below, we can see the number of notes in Year 2023 is much smaller than that of Year 2022 (9 to 50).![Image](./asset/year_histogram.png)

Going deeply into Year 2023 notes, we can see we only have notes from Jan to Apr: ![Image](./asset/month_histogram.png)


# Classify topics of the notes
There are 24 pre-defined topics in the given task. Since each note contains multiple paragraphs, and they could talk about different topics, it is more reasonable to classify the topic at paragraph level instead of document level.

As the topics are pre-defined and there are no training data, I manually labelled some paragraphs for training `data/labelled_sample.csv` and some others for testing `data/labelled_test.csv`.
Traditionally, this is a classification task. But building a classification model with 24 classes with a small training dataset will not produce acceptable performance.
So large language models (LLMs) are adopted to do this task.

There are two approaches to involve LLMs. One is to use OpenAI API to call GPT models. The other is to use open-sourced LLMs which can be deployed locally, such as LLaMA, Vicuna and ChatGLM.
pros and cons of two approaches.

Zero shot, few shot, P-tuning, Prompt-tuning, Fine-tuning with LoRA

Past work on comparing LLMs and why choose ChatGLM2-6B.

Due to the quota limitation of OpenAI API, I produced a small sample set using GPT-4 model. To classify the topic for entire dataset, ChatGLM2 is used here.
After the topic at paragraph-level and the topic for each title are generated, these topics are aggregated to document level.

Performance of the two approaches

It is possible that LLM gives a new topic for some paragraph. There are two cases. One is that the paragraph is talking a topic in the pre-defined list but the LLM did not strictly follow the prompt.
In this case we can find the topic in the pre-defined list which has the most similar embedding to the embedding of the new topic.
The other case is that the paragraph is indeed talking about something new. This case can be flagged out by setting a similarity threshold when looking for the most similar topic as in case one.
When all such new topics are identified, we can use LLM again to cluster these new topics into a small number of more concentrate new topics, and suggest to include these new topics in the pre-defined list.

# Identify top 3 emerging topics in Year 2023
To identify emerging topics in Year 2023, the first thought is to compare the topic frequency in Year 2022 and in Year 2023.
Since there are only 9 notes in Year 2023, it is more meaningful to compare the frequency in percentage.

The top 3 topics which increases most in percentage from Year 2022 to Year 2023 are `出口贸易`, `企业发展`, `教育`.

# Identify the subtopics or key messages for topic `企业发展` in Year 2023
There are several ways to extract topics and key messages from a list of texts.
LDA
Summarizer
LLM

# Conclusion


id_topics_all.csv: Topics at document level by using ChatGLM2-6B p-tuning. Each document has at most 3 topics. Topics are aggregated from content topics and title topics.

data/para_2023.txt: paragraphs in Year 2023 with topic "企业发展"