from typing import List, Sequence
from pydantic import BaseModel
import re
import pandas as pd
import logging
from pathlib import Path
import orgparse
import glob


class OrgElement(BaseModel):
    org_id: str
    file_name: str
    heading: str
    level: int
    body: str
    links: List[str]
    text: str

    @classmethod
    def get_org_node_elements(cls, file_name, file_node, only_root_contents=False):
        root_element = cls.get_root_node_element(file_node, file_name, only_root_contents)
        if only_root_contents:
            return [root_element]
        else:
            children_elements = [
                cls._get_child_node_element(root_element, child_node)
                for child_node in file_node[1:]
            ]
            return [root_element] + children_elements

    @classmethod
    def get_root_node_element(cls, file_node, file_name=None, only_root_contents=False):
        org_id = file_node.properties["ID"]
        if file_name is None:
            file_name = file_node.body.replace("#+title: ", "").strip()
        all_links = [link for child_node in file_node[1:]
                     for link in cls.parse_links(str(child_node))]
        return OrgElement(
            org_id=org_id,
            file_name=file_name,
            heading=file_node.heading,
            level=file_node.level,
            body=file_node.body,
            links=all_links,
            text=cls._get_root_contents(file_node, only_root_contents)
        )

    @classmethod
    def _get_root_contents(cls, file_node, only_root_contents):
        if only_root_contents:
            return "\n\n".join(str(elem) for elem in file_node)
        else:
            return file_node.heading + "\n" + file_node.body

    @classmethod
    def _get_child_node_element(cls, root_element, child_node):
        return OrgElement(
            org_id=root_element.org_id,
            file_name=root_element.file_name,
            heading=child_node.heading,
            level=child_node.level,
            body=child_node.body,
            links=cls.parse_links(str(child_node)),
            text=child_node.heading + "\n" + child_node.body
        )

    @classmethod
    def parse_links(cls, content):
        return re.findall(r'(?<=\[\[id:)([a-zA-Z0-9\-]+)(?=\]\[)', content)


class Org:
    element_cls = OrgElement

    @classmethod
    def load(cls, org_path, only_root_contents) -> List[OrgElement]:
        file_name = Path(org_path).stem
        return cls.element_cls.get_org_node_elements(file_name, orgparse.load(org_path), only_root_contents)

    @classmethod
    def loads(cls, org_contents, only_root_contents, file_name=None) -> List[OrgElement]:
        return cls.element_cls.get_org_node_elements(file_name, orgparse.loads(org_contents), only_root_contents)

    @classmethod
    def load_dir_generator(cls, dir, only_root_contents) -> Sequence[OrgElement]:
        dir_p = Path(dir) / "**" / "*.org"
        for org_file_path in glob.glob(str(dir_p)):
            try:
                elems = cls.load(org_file_path, only_root_contents)
            except Exception as e:
                logging.warn(f"cannot parse {org_file_path} elements")
                logging.warn(str(e))
                continue
            else:
                for elem in elems:
                    yield elem

    @classmethod
    def to_df(cls, org_elements: Sequence[OrgElement]) -> pd.DataFrame:
        return pd.DataFrame.from_records([elem.dict() for elem in org_elements])
