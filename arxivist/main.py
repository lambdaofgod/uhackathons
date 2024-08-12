import fire
from arxivist.reference_abstracts import ReferenceAbstractExtractor
import json
import logging

logging.basicConfig(level=logging.INFO)


class Main:

    def extract_referenced_abstracts(self, arxiv_id):
        logging.info(f"Extracting referenced abstracts for {arxiv_id}")
        referenced_abstract_records = ReferenceAbstractExtractor.get_referenced_paper_abstracts(
            arxiv_id)
        return json.dumps(referenced_abstract_records, indent=2)


if __name__ == "__main__":
    fire.Fire(Main())
