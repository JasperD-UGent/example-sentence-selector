# example-sentence-selector
This module allows you to select pedagogically suitable example sentences from a set of sentences and rank the selected sentences according to their degree of pedagogical suitability. The selection method is based on the SelEjemCor sentence selection framework (Degraeuwe, 2024), which has been specifically designed to search corpora for example sentences that are fit to be used in foreign language learning materials. The method targets Spanish as a foreign language and can be applied to [Universal Dependencies (UD) treebank data](#ud-treebank-data), [custom preprocessed data](#custom-preprocessed-data) or [custom plain text data](#custom-plain-text-data) as data sources. This repository includes demo data for all three possible data sources, based on the [UD Spanish GSD treebank](https://universaldependencies.org/treebanks/es_gsd/index.html). Below you can find more information on the main steps performed by the method, as well as on the data sources and where to put the dataset files. 

**NOTE**: the sentence selection method is initialised through the ``exampleSentenceSelector.py`` script. Normally, this script should be the only one you need to modify in order to be able to apply the sentence selection method to your own data and to set the parameters to the values of your choice.

## Required Python modules
See ``requirements.txt``.

## Method
### Step_1
The goal of a sentence selection method is to select sentences for a set of target words. These target items can be defined by using the ``l_target_items`` parameter in the ``exampleSentenceSelector.py`` script. Target items have to be nouns, verbs, or adjectives. For nouns, the lemma, part-of-speech tag and gender are required, separated by a pipe (e.g., ``"acción|NOUN|f"``); for verbs and adjectives, the lemma and part-of-speech tag are required (e.g., ``"andar|VERB"`` and ``"apto|ADJ"``).

### Step_2
Secondly, the dataset (see below for more details on the data sources) is processed and converted into a Python dictionary. To learn more about the dataset processing function and its parameters, have a look at the [source code](https://github.com/JasperD-UGent/example-sentence-selector/blob/003a3ce42bacc70f5f421fc964a4367e40c81cd9/exampleSentenceSelector_defs.py#L34).

### Step_3
Finally, the sentence selection method is applied to the dataset, with the list of selected and ranked sentences being saved in the ``output/exampleSelection`` directory. The proficiency level of the target audience and the number of years of experience they have can be defined through the ``level_target_audience`` and ``n_years_experience`` parameters, respectively. The remaining criteria to be taken into account by the selection algorithm are defined in the ``d_criteria`` parameter. For detailed descriptions of the criteria, please refer to Degraeuwe (2024). To learn more about the ``apply_example_selection_method`` function and its parameters, have a look at the [source code](https://github.com/JasperD-UGent/example-sentence-selector/blob/003a3ce42bacc70f5f421fc964a4367e40c81cd9/exampleSentenceSelector_defs.py#L180).

## Data sources
### UD treebank data
The first data source is UD treebank data in CoNNL-U format. Readily available treebanks can be downloaded from the [UD site](https://universaldependencies.org/#download). The treebank data need to be saved into the ``input/datasets_raw/UD`` directory. In this directory, first-level subdirectories should indicate the version of the UD data (e.g., "v2_11") and second-level subdirectories should indicate the name of the treebank (e.g., "UD_Spanish-AnCora"). The demo data are located in the ``input/dataset_raw/UD/demo/UD_Spanish-GSD`` directory.

**NOTE**: by default, the script will look for the treebank file ending on "train.conllu" and take it as input for the dataset. If you want to use a file with a different name, you can use the ``ud_dataset`` keyword parameter of the ``process_dataset`` function.

### Custom preprocessed data
The second data source is custom data which have already been preprocessed (i.e. tokenised and with the index of the target item being known). The data need to be stored as TXT files (for each target item separately) in the ``input/dataset_raw/custom_preprocessed`` directory. The names of the files should be the target item code, which corresponds to the entry of the target item in the ``l_target_items`` parameters (with underscores instead of pipes). Contentwise, the TXTs should adhere to the following format:
- One sentence per line
- Four items on one line (each of them separated by a tab)
  1. Sentence ID 
  2. Sentence tokens as a space-separated string 
  3. Index of the target item in the list of tokens 
  4. Sentence text in one string ("NA" if not available)

The demo data (for the noun *acción*, the verb *andar*, and the adjective *apto*; taken from the UD Spanish GSD treebank) are located in the ``input/dataset_raw/custom_preprocessed/demo`` directory.

### Custom plain text data
The third and final data source is custom data which have not been preprocessed yet. To tokenise and tag the sentences, [spaCy](https://spacy.io/) is used. As was the case for the preprocessed data, the plain text data also need to be stored in separate TXT files for each target item (in the ``input/dataset_raw/custom_plain_text`` directory). The names of the files should again correspond to the target item code. Contentwise, the TXTs should adhere to the following format:
- One sentence per line
- Two items on one line (each of them separated by a tab)
  1. Sentence ID
  2. Sentence text in one string

The demo data (for the noun *acción*, the verb *andar*, and the adjective *apto*; taken from the UD Spanish GSD treebank) are located in the ``input/dataset_raw/custom_plain_text/demo`` directory.

## Resources
The module makes uses of several linguistic resources, stored in a [dedicated repository](https://github.com/JasperD-UGent/resources). The following resources need to be copied to ``input/resources``:
- ``frequencyDictionary-percentiles_SCAP_[version].json`` (to the ``additionalLexicalCriteria`` directory)
- ``sensitiveVocabularyList_custom_[version].txt`` (to the ``additionalLexicalCriteria`` directory)
- ``tokenList_SCAP_[version].json`` (to the ``additionalLexicalCriteria`` directory)
- ``speakingVerbList_custom_[version].txt`` (to the ``additionalStructuralCriteria`` directory)
- ``adverbialAnaphorList_custom_[version].txt`` (to the ``contextIndependence`` directory)
- ``deltaP-NOUN-ADJ_SCAP_[version].json`` (to the ``typicality`` directory)
- ``deltaP-VERB-NOUN_SCAP_[version].json`` (to the ``typicality`` directory)
- ``lemma-n-grams_SCAP_[version].json`` (to the ``typicality`` directory)
- ``LMI-NOUN-ADJ_SCAP_[version].json`` (to the ``typicality`` directory)
- ``LMI-VERB-NOUN_SCAP_[version].json`` (to the ``typicality`` directory)
- ``lemmaList_SCAP_[version].json`` (to the ``well-formedness`` directory)

## References
- Degraeuwe, J. (2024). *IVESS: Intelligent Vocabulary and Example Selection for Spanish vocabulary learning* [PhD thesis]. Universiteit Gent.

