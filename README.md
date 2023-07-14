# Chinese-Topic-Modelling
Topic modelling for Chinese meeting notes followed by extensive topic analysis.

# Data Exploration
Please refer to [EDA.ipynb](https://github.com/sdx0112/Chinese-Topic-Modelling/blob/main/EDA.ipynb) for details.

For data profiling, see [report.html](https://github.com/sdx0112/Chinese-Topic-Modelling/blob/main/report.hrml)

The raw data `meeting notes.csv` contains 531 notes for 496 meetings, with two duplicated notes having the same content. The duplicated row is 
the note for `政治局会议` with title `研究部署在全党深入开展党的群众路线教育实践活动`. The two rows only differ one character on the `TITLE`, so it should be a typo.
The row with typo title was removed and the cleaned data was saved to `meeting notes clean.csv`.

From the histogram of Year, we can see the number of notes in Year 2023 is much smaller than that of Year 2022 (9 to 50).

# Classify topics of the notes


Manually label some paragraph

Few-shot agent

Prompt-tuning

Fine-tuning with LoRA


id_topics_all.csv: Topics at document level by using ChatGLM2-6B p-tuning. Each document has at most 3 topics. Topics are aggregated from content topics and title topics.

data/para_2023.txt: paragraphs in Year 2023 with topic "企业发展"