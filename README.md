# Unimorph_Inflect: A Python NLP Library for Generating Morphological Inflection in Many Human Languages



### References

If you use our neural pipeline including the tokenizer, the multi-word token expansion model, the lemmatizer, the POS/morphological features tagger, or the dependency parser in your research, please kindly cite our CoNLL 2018 Shared Task [system description paper](https://nlp.stanford.edu/pubs/qi2018universal.pdf):

```bibtex
@inproceedings{anastasopoulos19emnlp,
    title = {Pushing the Limits of Low-Resource Morphological Inflection},
    author = {Anastasopoulos, Antonios and Neubig, Graham},
    booktitle = {Proc. EMNLP},
    address = {Hong Kong},
    month = {November},
    year = {2019},
    note = {to appear}
}
```
This release is not the same as CMU's SIGMORPHON 2019 Shared Task system. The system is a cleaned up version of the shared task code and the models are trained on almost all Unimorph data for each language, whereas in the competition we used the designated datasets.


## Issues and Usage Q&A

Please use the following channels for questions and issue reports.

| Purpose | Channel |
|---|---|
| Bug Reports and Feature Requests | [GitHub Issue Tracker](https://github.com/antonisa/unimorph_inflection/issues) |

## Setup

Unimorph_inflect supports Python 3.6 or later. We strongly recommend that you install Unimorph_inflect from PyPI. If you already have [pip installed](https://pip.pypa.io/en/stable/installing/), simply run
```bash
pip install unimorph_inflect
```
this should also help resolve all of the dependencies of unimorph_inflect, for instance [DyNet](https://dynet.readthedocs.io/) 2.0.0 or above.

Alternatively, you can also install from source of this git repository, which will give you more flexibility in developing on top of unimorph_inflect and training your own models. For this option, run
```bash
git clone git@github.com:antonisa/unimorph_inflect.git
cd unimorph_inflect
pip install -e .
```

## Running unimorph_inflect

### Getting Started with the neural pipeline

To run your first StanfordNLP pipeline, simply following these steps in your Python interactive interpreter:

```python
>>> import unimorph_inflect
>>> unimorph_inflect.download('eng')   # This downloads the English models, if you don't have them already
>>>
>>> from unimorph_inflect import inflect
>>> result = inflect("laugh", "V;PST", language='eng')
>>> print(result[0])
```

The last command should print the following (the result of return is a list of outputs, hence the "[0]") there

```
laughed
```

### Trained Models for unimorph_inflect

We currently provide models trained on all Unimorph data (except 1000 examples used as a development set) for some high-resource languages, trained in a monolingual setting. The accuracy on the development sets are as follows:

| Language | Supported PoS | Dev Accuracy |
|----------|---------------|--------------|
|  eng     | V             | 0.97         |
|  ell     | V, N, ADJ     | 0.84         |
|  rus     | V, N, ADJ     | 0.94         |


A simple call of the `inflect` function with your desired language should download the necessary models, but you can also download them from [here](http://www.cs.cmu.edu/~aanastas/software/inflection_models/latest/).


## LICENSE

Unimorph_inflect is released under the Apache License, Version 2.0. See the [LICENSE](https://github.com/antonisa/unimorph_inflect/LICENSE) file for more details.