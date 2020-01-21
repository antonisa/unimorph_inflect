# Unimorph_Inflect: A Python NLP Library for Generating Morphological Inflection in Many Human Languages

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

### Getting Started

You can get started by simply following these steps in your Python interactive interpreter:

```python
>>> import unimorph_inflect
>>> unimorph_inflect.download('eng')   # This downloads the English models, if you don't have them already
>>>
>>> from unimorph_inflect import inflect
>>> result = inflect("laugh", "V;PST", language='eng')
>>> print(result[0])
laughed
```
Note: `inflect()` returns a list of outputs, hence the "[0]") there

You don't really need to explicitly download each dataset (as shown in the second line above); the `inflect()` function will ask you about downloading the model for a language if it is not downloaded already.

### Trained Models for unimorph_inflect

We currently provide models trained on all Unimorph data (except 1000 examples used as a development set) for some high-resource languages, trained in a monolingual setting. 

You can list the available languages/models with:
```
>>> unimorph_inflect.supported_languages
['ady', 'ang', 'ast', 'bel', 'bul', 'cat', 'dan', ...]
```

The accuracy on the development sets are as follows:

| Language      | ISO | Supported PoS | Dev Accuracy |
|---------------|-----|---------------|--------------|
| Adyghe        | ady | V, ADJ        | 90.0         |
| Armenian      | hye | V, N, ADJ     | 98.9         |
| Albanian      | sqi | V, N,         | 69.0         |
| Bashkir       | bak | N, ADJ        | 81.0         |
| Basque        | eus | V             | 48.0         |
| Belarusian    | bel | V, N, ADJ     | 91.0         |
| Bulgarian     | bul | V, N, ADJ     | 99.0         |
| Catalan       | cat | V             | 100          |
| Danish        | dan | N, ADJ        | 82.0         |
| Dutch         | nld | N, ADJ        | 98.0         |
| English       | eng | V             | 97.0         |
| Estonian      | est | V, N          | 84.0         |
| Farsi         | fas | V             | 93.0         |
| French        | fra | V             | 97.0         |
| German        | deu | V, N          | 100          |
| Georgian      | kat | V, N, ADJ     | 100          |
| Greek         | ell | V, N, ADJ     | 84.0         |
| Hindi         | hin | V             | 78.0         |
| Irish         | gle | V             | 85.6         |
| Icelandic     | isl | V, N          | 93.0         |
| Italian       | ita | V             | 99.2         |
| Latvian       | lav | V, N, ADJ     | 99.0         |
| Old English   | ang | V, N, ADJ     | 84.0         |
| Old Saxon     | osx | V, N, ADJ     | 93.0         |
| Polish        | pol | V, N, ADJ     | 95.0         |
| Quechua       | que | V, N, ADJ     | 90.0         |
| Russian       | rus | V, N, ADJ     | 94.0         |
| Serbocroatian | hbs | V             | 92.7         |
| Spanish       | spa | V             | 100          |
| Swedish       | swe | V, N, ADJ     | 96.0         |
| Ukranian      | ukr | V, N, ADJ     | 97.0         |
| Urdu          | urd | V, N          | 71.0         |
| Zulu          | zul | V, N, ADJ     | 87.0         |


A simple call of the `inflect` function with your desired language should download the necessary models, but you can also download them from [here](https://github.com/antonisa/unimorph_inflect/blob/master/models).

## References

If you use our models in your research, please cite our EMNLP 2019 [paper](https://www.aclweb.org/anthology/D19-1091.pdf) along with the necessary [Unimorph datasets](http://www.lrec-conf.org/proceedings/lrec2018/pdf/789.pdf):

```bibtex
@inproceedings{anastasopoulos19emnlp,
    title = {Pushing the Limits of Low-Resource Morphological Inflection},
    author = {Anastasopoulos, Antonios and Neubig, Graham},
    booktitle = {Proc. EMNLP},
    address = {Hong Kong},
    month = {November},
    year = {2019},
}
```

This release is not the same as CMU's SIGMORPHON 2019 Shared Task system. The system is a cleaned up version of the shared task code and the models are trained on almost all Unimorph data for each language, whereas in the competition we used the designated datasets.

## Issues and Usage Q&A

Please use the [GitHub Issue Tracker](https://github.com/antonisa/unimorph_inflect/issues) for bug reports, language/feature requests, and other questions.

## LICENSE

Unimorph_inflect is released under the Apache License, Version 2.0. See the [LICENSE](https://github.com/antonisa/unimorph_inflect/LICENSE) file for more details.