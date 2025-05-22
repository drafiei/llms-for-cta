# llms-for-cta
Column Type Annotation using LLMs

## Table of contents
* [Source Repo](#source)
* [Dataset](#dataset)
* [Setup](#setup)
* [Citation](#citation)

* ## source
This repo is based on [Amir's code and data](https://github.com/amirbabamahmoudi/LLM-for-CTA) with some changes and restructuring.

* ## dataset
The datasets (test and train) are in the data folder.


## setup
To run this project, use the following commands. Change the llm on Line 71 to the one being tested and comment out the models that are not being tested in main. It is currently running all models.

```
$ pip3 install -r requirements.txt
$ export OPENAI_API_KEY=your-api-key-here
$ echo "Start running cta.py"
$ python3 cta.py > exp-log.txt
$ echo "Finished running cta.py"
$ echo "The file exp-log.txt has the log."
```
## citation

```
@article{babamahmoudi2025cta,
  title={Improving Column Type Annotation Using Large Language Models},
  author={Babamahmoudi, Amir and Rafiei, Davood and Nascimento, Mario A.},
  journal={tba},
  year={2025}
}
```
