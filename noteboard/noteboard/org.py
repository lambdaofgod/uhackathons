from typing import List, Sequence, Optional
from pydantic import BaseModel
import re
import pandas as pd
import logging
from pathlib import Path
import orgparse
import glob
import os
import datetime


class OrgElement(BaseModel):
    org_id: str
    file_name: str
    heading: str
    level: int
    body: str
    links: List[str]
    text: str
    creation_date: Optional[datetime.datetime] = None
    last_edit_date: Optional[datetime.datetime] = None

    @classmethod
    def get_org_node_elements(cls, file_name, file_node, file_path=None, only_root_contents=False):
        # Extract creation date from filename
        creation_date = cls._extract_creation_date(file_name)
        
        # Extract last edit date from file mtime if file_path is provided
        last_edit_date = None
        if file_path:
            last_edit_date = cls._get_last_edit_date(file_path)
        
        root_element = cls.get_root_node_element(
            file_node, 
            file_name, 
            only_root_contents,
            creation_date=creation_date,
            last_edit_date=last_edit_date
        )
        if only_root_contents:
            return [root_element]
        else:
            children_elements = [
                cls._get_child_node_element(root_element, child_node)
                for child_node in file_node[1:]
            ]
            return [root_element] + children_elements

    @classmethod
    def get_root_node_element(cls, file_node, file_name=None, only_root_contents=False, 
                             creation_date=None, last_edit_date=None):
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
            text=cls._get_root_contents(file_node, only_root_contents),
            creation_date=creation_date,
            last_edit_date=last_edit_date
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
            text=child_node.heading + "\n" + child_node.body,
            creation_date=root_element.creation_date,
            last_edit_date=root_element.last_edit_date
        )

    @classmethod
    def parse_links(cls, content):
        return re.findall(r'(?<=\[\[id:)([a-zA-Z0-9\-]+)(?=\]\[)', content)
        
    @staticmethod
    def _extract_creation_date(file_name):
        """Extract creation date from filenames like '20240207133304-paradox_games'"""
        match = re.match(r'^(\d{14})-', file_name)
        if match:
            date_str = match.group(1)
            try:
                return datetime.datetime.strptime(date_str, '%Y%m%d%H%M%S')
            except ValueError:
                return None
        return None

    @staticmethod
    def _get_last_edit_date(file_path):
        """Get the last modification time of the file and convert to datetime"""
        try:
            mtime = os.path.getmtime(file_path)
            return datetime.datetime.fromtimestamp(mtime)
        except (OSError, ValueError):
            return None


class Org:
    element_cls = OrgElement

    @classmethod
    def load(cls, org_path, only_root_contents) -> List[OrgElement]:
        file_name = Path(org_path).stem
        return cls.element_cls.get_org_node_elements(
            file_name, 
            orgparse.load(org_path), 
            file_path=org_path,
            only_root_contents=only_root_contents
        )

    @classmethod
    def loads(cls, org_contents, only_root_contents, file_name=None, file_path=None) -> List[OrgElement]:
        return cls.element_cls.get_org_node_elements(
            file_name, 
            orgparse.loads(org_contents), 
            file_path=file_path,
            only_root_contents=only_root_contents
        )

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
