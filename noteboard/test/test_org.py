from noteboard import org
import pytest


@pytest.fixture
def org_contents():
    return ":PROPERTIES:\n:ID:       78523f9b-b835-4f3e-be95-96671a599479\n:END:\n#+title: pragmatics\n\n[[id:1d087d1c-7d62-4876-a4b0-9469ac4aa199][computational_pragmatics]]\n\n* Sources\n\n[[https://www.ucl.ac.uk/~ucjtudo/][Yasutada Sudo - Experimental approaches to scalar implicatures (under teaching)]]\n\n* Intro\n\n** Pragmatics\n\nPragmatics explores *meaning in context*.\nMeaning not relative to context, so called *truth-value semantics* is only a tip of the iceberg; the hard part of natural language is that it uses non-literal meaning.\n\nThe non-literal differs from literal meaning in that it also depends on *goals*.\n\nFailure to discern the goals is an example of failure of AI models, for example GPT\n#+BEGIN_EXAMPLE\nHUMAN: Can you help me find my keys?\nGPT: Yes, I can help you\n#+END_EXAMPLE\n\nThis is taken from [[cite:&ruis22_large_languag_model_are_not]]\n\n** Maxims\n\nRules for inference about speaker's intentions\n\n** Implicature\n\nInference about the meaning relative to context\n\n[[id:6a7500f7-931d-4478-b046-9cadf5f78755][pragmatics_context]]\n\n* Rational Speech Act framework\n\n** TL;DR\n\n#+BEGIN_QUOTE\nThe RSA model implements a social cognition approach to utterance understanding. At its core,\nit captures the idea (due to Grice, David Lewis, and others) that speakers are assumed to\nproduce utterances to be helpful yet parsimonious, relative to some particular topic or goal.\n#+END_QUOTE\nFrom [[cite:&pragmatic_language_interpretation_probabilistic]]\n\n** Sources\n\n[[id:6a027580-d5c7-4f9d-95b1-0b2a93d2a27f][pragmatic_language_interpretation_goodman_frank]] - intro by the authors\n\n[[cite:&scontras21_pract_introd_to_ration_speec]]\n\n*** Probabilistic language understanding\n\nNice online book with examples in PPL\n\nhttp://www.problang.org/\n\n* Rate-distortion and RSA\n\nIsn't this kinda redundant?\n\n[[cite:&zaslavsky20_rate_distor_view_human_pragm_reason]]\n"


def test_parse_links():
    str_link = "[[id:6a7500f7-931d-4478-b046-9cadf5f78755][pragmatics_context]]"
    assert len(org.OrgElement.parse_links(str_link)) == 1


def test_loads(org_contents):
    assert len(org.Org.loads(org_contents)) == 11
