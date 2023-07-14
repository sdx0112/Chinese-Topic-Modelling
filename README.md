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

## 1. LLMs and improvement
There are two approaches to involve LLMs. One is to use OpenAI API to call GPT models. The other is to use open-sourced LLMs
which can be deployed locally, such as [`LLaMA`](https://github.com/facebookresearch/llama), [`Vicuna`](https://github.com/lm-sys/FastChat) 
and [`ChatGLM`](https://github.com/THUDM/ChatGLM-6B).
The following table compares some key features of the two approaches.

|  Feature   |            OpenAI API             |                 Open-sourced models                 |
|:----------:|:---------------------------------:|:---------------------------------------------------:|
|    Size    |         175B for ChatGPT          |        Most models have 6B, 7B, 13B versions        |
|   Usage    |             Paid API              |              Mostly non-commercial use              |
|    Code    |           Not available           | Mostly available for train, inference and fine-tune |
| Limitation | Monthly quota on number of tokens |        Need large GPU for better performance        |

LLMs are trained on general data, and may not deliver good performance on tasks in specific data. To overcome this challenge, there are several
ways to incorporate with a set of task-specific samples, such as few-shot learning, P-tuning, Prompt-tuning, Fine-tuning with LoRA.
While few-shot learning is to explicitly put some examples in the prompt, the rest 3 methods employ a training process with a small training samples,
and have been shown to be efficient and comparable to fully fine-tuning. See paper: [The Power of Scale for Parameter-Efficient Prompt Tuning
](https://arxiv.org/pdf/2104.08691.pdf).
![Image](./asset/tuning.png)

Prior to this task, my team has conducted benchmarking analysis of `LLaMA-13B`, `Vicuna-13B`, `ChatGLM-6B` and `ChatGPT`. We designed questions from several
aspects to compare their performance both in English and Chinese. The result shows `ChatGPT` is always the best, and `ChatGLM-6B` is ranked second on Chinese questions.
`ChatGLM` is developed by Tsinghua University and gained immense popularity. Recently the enhanced version [`ChatGLM2-6B`](https://github.com/THUDM/ChatGLM2-6B/tree/main)
was released with new features and better performance. So `ChatGLM2-6B` is selected for this task.

## 2. Topic classification at paragraph-level

In this work:
- For `GPT-4`,  I tested [zero-shot learning and few-shot learning](GPT-FewShot-Test.ipynb). The accuracy of zero-shot learning is `87.5%` and that of few-shot learning is `79.2%`.
- For `ChatGLM2-6B`, I tested [zero-shot learning, few-shot learning](ChatGLM2_6B_zero_shot_vs_few_shot.ipynb), and [P-tuning](ChatGLM2_6B_P_Tuning_v2.ipynb). 
The accuracy of zero-shot learning is `12.5%` and that of few-shot learning is `8.3%`. The accuracy of P-tuning is `58.3%`, which is a significant improvement.
The reason that the performance decreased from zero-shot learning to few-shot learning could be that the sample data is very small.

Due to the quota limitation of OpenAI API, I can only produce a small sample set using [`GPT-4` model zero-shot learning](GPT-4%20Zero%20Shot%20Paragraph.ipynb).
To classify the topic for entire dataset, [ChatGLM2 with P-tuning](ChatGLM2_6B_P_Tuning_v2.ipynb) is applied to the paragraph-level content `data/all_para.csv` and titles `data/title_topic.csv`.

Sometimes LLMs do not follow the prompt to pick a topic from pre-defined list. In this task, we map the new topic to one of the pre-defined topics in the following steps:
- Get the embeddings of the new topic and the pre-defined topics.
- Find the pre-defined topic which has the most similar embeddings to the new topic.
- If the highest similarity score is lower than the threshold, then drop this paragraph. This assumes all paragraphs do not talk about any new topic.
For the case where some paragraph is talking about new topics, refer to [here](#4-what-about-new-topics).

Titles also play an important role in topic identification. So the same approach for paragraphs also applies to titles.

After [the topic at paragraph-level and the topic for each title are generated](Aggregate_Topic.ipynb), these topics are aggregated to document level.

## 3. Topic aggregation to document-level

For each document / note, the topic is aggregated by the following steps:
- Get the top 2 frequent topics from its paragraphs.
- Include the topic obtained from the title.
- Remove duplicates from the 3 topics produced above.

The final output is saved to [`data/id_topics_all.csv`](Topic%20Aggregation%20Clean.ipynb).

## 4. What about new topics?

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

# Dependencies


# Code and Data

Remove duplicated row in final output